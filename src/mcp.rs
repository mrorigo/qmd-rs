// Rust guideline compliant 2026-03-08

use crate::{config::Config, db::Database, search};
use anyhow::{Context, Result};
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{convert::Infallible, net::SocketAddr, time::Duration};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_stream::{wrappers::IntervalStream, StreamExt};

/// Tool request payload for stdio/HTTP endpoints.
#[derive(Debug, Deserialize)]
pub struct ToolRequest {
    /// Tool name.
    pub tool: String,
    /// Tool arguments.
    #[serde(default)]
    pub args: Value,
}

#[derive(Debug, Serialize)]
struct ToolResponse {
    ok: bool,
    result: Value,
}

#[derive(Debug, Clone)]
struct AppState {
    cfg: Config,
}

/// Run MCP tool server in stdio mode.
///
/// # Arguments
/// `cfg` - Effective runtime configuration.
///
/// # Errors
/// Returns an error for stdin/stdout I/O or tool execution failures.
pub async fn run_stdio(cfg: Config) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let payload: Value = serde_json::from_str(&line).unwrap_or_else(|_| json!({}));
        let normalized = normalize_request(payload);
        let response = match normalized {
            Ok(req) => match dispatch_tool(&cfg, &req.tool, req.args).await {
                Ok(result) => ToolResponse { ok: true, result },
                Err(err) => ToolResponse {
                    ok: false,
                    result: json!({"error": err.to_string()}),
                },
            },
            Err(err) => ToolResponse {
                ok: false,
                result: json!({"error": err.to_string()}),
            },
        };

        let out = serde_json::to_string(&response)?;
        stdout.write_all(out.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    Ok(())
}

/// Run MCP server over HTTP with a tool endpoint and SSE stream.
///
/// # Arguments
/// `cfg` - Effective runtime configuration.
/// `port` - Listen port.
///
/// # Errors
/// Returns an error if binding or serving fails.
pub async fn run_http(cfg: Config, port: u16) -> Result<()> {
    let app = Router::new()
        .route(
            "/tool",
            post(
                |State(state): State<AppState>, Json(raw): Json<Value>| async move {
                    Json(handle_http_tool(&state.cfg, raw))
                },
            ),
        )
        .route("/events", get(sse_events))
        .with_state(AppState { cfg });

    let addr: SocketAddr = format!("0.0.0.0:{port}")
        .parse()
        .context("invalid bind address")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn handle_http_tool(cfg: &Config, raw: Value) -> ToolResponse {
    let req: ToolRequest = match serde_json::from_value(raw) {
        Ok(v) => v,
        Err(err) => {
            return ToolResponse {
                ok: false,
                result: json!({"error": err.to_string()}),
            };
        }
    };

    let res = match tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(dispatch_tool(cfg, &req.tool, req.args))
    }) {
        Ok(result) => ToolResponse { ok: true, result },
        Err(err) => ToolResponse {
            ok: false,
            result: json!({"error": err.to_string()}),
        },
    };
    res
}

async fn sse_events() -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let interval = tokio::time::interval(Duration::from_secs(5));
    let stream = IntervalStream::new(interval)
        .map(|_| Ok(Event::default().event("heartbeat").data("qmd-mcp-alive")));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn normalize_request(payload: Value) -> Result<ToolRequest> {
    if payload.get("tool").is_some() {
        return Ok(serde_json::from_value(payload)?);
    }

    if payload.get("method") == Some(&Value::String("tools/call".to_string())) {
        let params = payload.get("params").cloned().unwrap_or_else(|| json!({}));
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        return Ok(ToolRequest { tool: name, args });
    }

    anyhow::bail!("invalid request: expected {{tool,args}} or tools/call payload")
}

async fn dispatch_tool(cfg: &Config, tool: &str, args: Value) -> Result<Value> {
    let db = Database::open(cfg)?;

    match tool {
        "qmd_search" => {
            let query = required_string(&args, "query")?;
            let results = search::run_bm25_search(&db, query, 20)?;
            Ok(serde_json::to_value(results)?)
        }
        "qmd_vector_search" => {
            let query = required_string(&args, "query")?;
            let results = search::run_vector_search(cfg, &db, query, 20).await?;
            Ok(serde_json::to_value(results)?)
        }
        "qmd_deep_search" => {
            let query = required_string(&args, "query")?;
            let results = search::run_hybrid_query(cfg, &db, query).await?;
            Ok(serde_json::to_value(results)?)
        }
        "qmd_get" => {
            let selector = required_string(&args, "selector")?;
            let doc = db.get_document(selector)?;
            Ok(serde_json::to_value(doc)?)
        }
        "qmd_multi_get" => {
            let pattern = required_string(&args, "pattern")?;
            let docs = db.multi_get_documents(pattern)?;
            Ok(serde_json::to_value(docs)?)
        }
        "qmd_status" => {
            let status = db.health_report()?;
            Ok(serde_json::to_value(status)?)
        }
        _ => anyhow::bail!("unknown tool: {tool}"),
    }
}

fn required_string<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing required string argument: {key}"))
}
