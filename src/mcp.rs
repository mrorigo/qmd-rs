// Rust guideline compliant 2026-03-08

use crate::{config::Config, db::Database, search};
use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_stream::{wrappers::IntervalStream, StreamExt};

const JSONRPC_VERSION: &str = "2.0";
const PROTOCOL_VERSION: &str = "2025-11-25";

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

#[derive(Debug, Clone)]
struct AppState {
    cfg: Config,
    initialized: std::sync::Arc<AtomicBool>,
}

#[derive(Debug, Serialize)]
struct ToolDef {
    name: &'static str,
    description: &'static str,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

/// Run MCP server in stdio mode.
pub async fn run_stdio(cfg: Config) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();
    let mut initialized = false;

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let parsed: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(err) => {
                let resp = jsonrpc_error(Value::Null, PARSE_ERROR, &format!("parse error: {err}"));
                stdout.write_all(serde_json::to_string(&resp)?.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        if let Some(id) = parsed.get("id").cloned() {
            let response = handle_request(&cfg, initialized, &parsed, id).await;
            stdout.write_all(serde_json::to_string(&response)?.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        } else {
            handle_notification_stdio(&mut initialized, &parsed);
        }
    }

    Ok(())
}

/// Run MCP server over Streamable HTTP with a single MCP endpoint.
pub async fn run_http(cfg: Config, port: u16) -> Result<()> {
    let state = AppState {
        cfg,
        initialized: std::sync::Arc::new(AtomicBool::new(false)),
    };

    let app = Router::new()
        .route("/mcp", post(http_post).get(http_get))
        .with_state(state);

    let addr: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .context("invalid bind address")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn http_post(State(state): State<AppState>, body: String) -> Response {
    let payload: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(err) => {
            let resp = jsonrpc_error(Value::Null, PARSE_ERROR, &format!("parse error: {err}"));
            return (StatusCode::BAD_REQUEST, Json(resp)).into_response();
        }
    };

    let initialized = state.initialized.load(Ordering::SeqCst);
    if let Some(id) = payload.get("id").cloned() {
        let resp = handle_request(&state.cfg, initialized, &payload, id).await;
        (StatusCode::OK, Json(resp)).into_response()
    } else {
        handle_notification(&state.initialized, &payload);
        StatusCode::ACCEPTED.into_response()
    }
}

async fn http_get() -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let interval = tokio::time::interval(Duration::from_secs(5));
    let stream = IntervalStream::new(interval)
        .map(|_| Ok(Event::default().event("heartbeat").data("qmd-mcp-alive")));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn handle_request(cfg: &Config, initialized: bool, payload: &Value, id: Value) -> Value {
    let method = match payload.get("method").and_then(Value::as_str) {
        Some(m) => m,
        None => return jsonrpc_error(id, INVALID_REQUEST, "missing method"),
    };

    let params = payload.get("params").cloned().unwrap_or_else(|| json!({}));

    match method {
        "initialize" => {
            let result = json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "qmd-rs",
                    "version": env!("CARGO_PKG_VERSION")
                }
            });
            jsonrpc_result(id, result)
        }
        "ping" => jsonrpc_result(id, json!({})),
        "tools/list" => {
            if !initialized {
                return jsonrpc_error(id, INVALID_REQUEST, "server not initialized");
            }
            jsonrpc_result(id, json!({ "tools": mcp_tools() }))
        }
        "tools/call" => {
            if !initialized {
                return jsonrpc_error(id, INVALID_REQUEST, "server not initialized");
            }
            let name = match params.get("name").and_then(Value::as_str) {
                Some(v) if !v.trim().is_empty() => v,
                _ => return jsonrpc_error(id, INVALID_PARAMS, "missing tools/call.params.name"),
            };
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

            match execute_tool_call(cfg, name, args).await {
                Ok(result) => jsonrpc_result(id, result),
                Err(err) => jsonrpc_error(id, INTERNAL_ERROR, &err.to_string()),
            }
        }
        _ => jsonrpc_error(id, METHOD_NOT_FOUND, &format!("unknown method: {method}")),
    }
}

fn handle_notification(initialized: &AtomicBool, payload: &Value) {
    if payload.get("method") == Some(&Value::String("notifications/initialized".to_string())) {
        initialized.store(true, Ordering::SeqCst);
    }
}

fn handle_notification_stdio(initialized: &mut bool, payload: &Value) {
    if payload.get("method") == Some(&Value::String("notifications/initialized".to_string())) {
        *initialized = true;
    }
}

async fn execute_tool_call(cfg: &Config, name: &str, args: Value) -> Result<Value> {
    let structured = match name {
        "qmd_search" => {
            let query = required_string(&args, "query")?;
            let db = Database::open(cfg)?;
            serde_json::to_value(search::run_bm25_search(&db, query, 20)?)?
        }
        "qmd_vector_search" => {
            let query = required_string(&args, "query")?;
            serde_json::to_value(search::run_vector_search(cfg, query, 20).await?)?
        }
        "qmd_deep_search" => {
            let query = required_string(&args, "query")?;
            serde_json::to_value(search::run_hybrid_query(cfg, query).await?)?
        }
        "qmd_get" => {
            let selector = required_string(&args, "selector")?;
            let db = Database::open(cfg)?;
            serde_json::to_value(db.get_document(selector)?)?
        }
        "qmd_multi_get" => {
            let pattern = required_string(&args, "pattern")?;
            let db = Database::open(cfg)?;
            serde_json::to_value(db.multi_get_documents(pattern)?)?
        }
        "qmd_status" => {
            let db = Database::open(cfg)?;
            serde_json::to_value(db.health_report()?)?
        }
        _ => anyhow::bail!("unknown tool: {name}"),
    };

    Ok(json!({
        "content": [
            {"type": "text", "text": serde_json::to_string_pretty(&structured)?}
        ],
        "structuredContent": structured
    }))
}

fn mcp_tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "qmd_search",
            description: "Execute BM25 keyword search.",
            input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
        },
        ToolDef {
            name: "qmd_vector_search",
            description: "Execute semantic vector search.",
            input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
        },
        ToolDef {
            name: "qmd_deep_search",
            description: "Execute hybrid deep search.",
            input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
        },
        ToolDef {
            name: "qmd_get",
            description: "Retrieve one document by selector.",
            input_schema: json!({"type":"object","properties":{"selector":{"type":"string"}},"required":["selector"]}),
        },
        ToolDef {
            name: "qmd_multi_get",
            description: "Retrieve multiple documents by pattern.",
            input_schema: json!({"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}),
        },
        ToolDef {
            name: "qmd_status",
            description: "Return index health and metadata.",
            input_schema: json!({"type":"object","properties":{}}),
        },
    ]
}

fn jsonrpc_result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": JSONRPC_VERSION, "id": id, "result": result})
}

fn jsonrpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": JSONRPC_VERSION, "id": id, "error": {"code": code, "message": message}})
}

fn required_string<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing required string argument: {key}"))
}
