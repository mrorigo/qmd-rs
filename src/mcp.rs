// Rust guideline compliant 2026-03-08

use crate::{
    config::Config,
    db::Database,
    ingest::{self, is_markdown, matches_collection_filters},
    search,
};
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::post,
    Json, Router,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    convert::Infallible,
    net::SocketAddr,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, SystemTime},
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_stream::{wrappers::IntervalStream, StreamExt};
use walkdir::WalkDir;

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
    #[serde(rename = "outputSchema")]
    output_schema: Value,
}

/// Run MCP server in stdio mode.
pub async fn run_stdio(cfg: Config) -> Result<()> {
    spawn_collection_watcher(cfg.clone());
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
                stdout
                    .write_all(serde_json::to_string(&resp)?.as_bytes())
                    .await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        if let Some(id) = parsed.get("id").cloned() {
            let response = handle_request(&cfg, initialized, &parsed, id).await;
            stdout
                .write_all(serde_json::to_string(&response)?.as_bytes())
                .await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        } else {
            handle_notification_stdio(&mut initialized, &parsed);
        }
    }

    Ok(())
}

/// Run MCP server over Streamable HTTP with a single MCP endpoint.
pub async fn run_http(
    cfg: Config,
    bind_address: Option<std::net::IpAddr>,
    port: u16,
) -> Result<()> {
    spawn_collection_watcher(cfg.clone());
    let state = AppState {
        cfg,
        initialized: std::sync::Arc::new(AtomicBool::new(false)),
    };

    let app = Router::new()
        .route("/mcp", post(http_post).get(http_get))
        .with_state(state);

    let bind_address = bind_address.unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let addr = SocketAddr::new(bind_address, port);
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
            jsonrpc_result(id, json!({ "tools": mcp_tools(cfg) }))
        }
        "tools/call" => {
            if !initialized {
                return jsonrpc_error(id, INVALID_REQUEST, "server not initialized");
            }
            let name = match params.get("name").and_then(Value::as_str) {
                Some(v) if !v.trim().is_empty() => v,
                _ => return jsonrpc_error(id, INVALID_PARAMS, "missing tools/call.params.name"),
            };
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));

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
        "search" => {
            let query = required_string(&args, "query")?;
            let db = Database::open(cfg)?;
            serde_json::to_value(search::run_bm25_search(&db, query, 20)?)?
        }
        "vector_search" => {
            if cfg.mode == crate::config::ModeConfig::Offline {
                anyhow::bail!("vector_search is disabled in offline mode");
            }
            let query = required_string(&args, "query")?;
            serde_json::to_value(search::run_vector_search(cfg, query, 20).await?)?
        }
        "deep_search" => {
            if cfg.mode == crate::config::ModeConfig::Offline {
                let query = required_string(&args, "query")?;
                let db = Database::open(cfg)?;
                let results = search::run_bm25_search(&db, query, 20)?;
                let structured_content =
                    normalize_structured_content(serde_json::to_value(results)?);
                return Ok(json!({
                    "content": [{"type": "text", "text": serde_json::to_string_pretty(&structured_content)?}],
                    "structuredContent": structured_content
                }));
            }
            let query = required_string(&args, "query")?;
            serde_json::to_value(search::run_hybrid_query(cfg, query).await?)?
        }
        "get" => {
            let selector = required_string(&args, "selector")?;
            let db = Database::open(cfg)?;
            let doc = db
                .get_document(selector)?
                .ok_or_else(|| anyhow::anyhow!("document not found for selector: {selector}"))?;
            serde_json::to_value(doc)?
        }
        "multi_get" => {
            let pattern = required_string(&args, "pattern")?;
            let db = Database::open(cfg)?;
            serde_json::to_value(db.multi_get_documents(pattern)?)?
        }
        "status" => {
            let db = Database::open(cfg)?;
            serde_json::to_value(db.health_report(cfg)?)?
        }
        _ => anyhow::bail!("unknown tool: {name}"),
    };
    let structured_content = normalize_structured_content(structured);

    Ok(json!({
        "content": [
            {"type": "text", "text": serde_json::to_string_pretty(&structured_content)?}
        ],
        "structuredContent": structured_content
    }))
}

fn spawn_collection_watcher(cfg: Config) {
    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();

        let Ok(runtime) = runtime else {
            tracing::warn!("failed to start collection watcher runtime");
            return;
        };

        let mut state = WatchState::default();
        loop {
            if let Err(err) = runtime.block_on(sync_collections_once(&cfg, &mut state)) {
                tracing::warn!(error = %err, "collection watcher sync failed");
            }
            thread::sleep(Duration::from_secs(5));
        }
    });
}

async fn sync_collections_once(cfg: &Config, state: &mut WatchState) -> Result<()> {
    let db = Database::open(cfg)?;
    let collections = db.list_collections()?;
    let mut next_seen = HashMap::new();

    for collection in collections {
        let collection_path = std::path::Path::new(&collection.path);
        if !collection_path.exists() {
            continue;
        }

        for entry in WalkDir::new(collection_path)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            if !is_markdown(path) {
                continue;
            }
            if !matches_collection_filters(
                path,
                collection_path,
                collection.include_glob.as_deref(),
                collection.exclude_glob.as_deref(),
            )? {
                continue;
            }

            let fingerprint = file_fingerprint(path)?;
            next_seen.insert(path.to_path_buf(), fingerprint.clone());
            if state.seen.get(path) != Some(&fingerprint) {
                let _ = ingest::sync_markdown_file(cfg, &db, path).await?;
            }
        }
    }

    for path in state.seen.keys() {
        if !next_seen.contains_key(path) {
            let _ = ingest::remove_markdown_file(&db, path)?;
        }
    }

    state.seen = next_seen;
    Ok(())
}

#[derive(Default)]
struct WatchState {
    seen: HashMap<std::path::PathBuf, FileFingerprint>,
}

#[derive(Clone, PartialEq, Eq)]
struct FileFingerprint {
    modified_secs: Option<u64>,
    len: Option<u64>,
}

fn file_fingerprint(path: &std::path::Path) -> Result<FileFingerprint> {
    let meta = std::fs::metadata(path)?;
    let modified_secs = meta
        .modified()
        .ok()
        .and_then(|ts| ts.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    Ok(FileFingerprint {
        modified_secs,
        len: Some(meta.len()),
    })
}

fn normalize_structured_content(value: Value) -> Value {
    match value {
        Value::Object(_) => value,
        Value::Array(items) => json!({ "results": items }),
        scalar => json!({ "value": scalar }),
    }
}

fn mcp_tools(cfg: &Config) -> Vec<ToolDef> {
    let search_result_schema = json!({
        "type": "object",
        "properties": {
            "docid": { "type": "string" },
            "path": { "type": "string" },
            "title": { "type": ["string", "null"] },
            "snippet": { "type": "string" },
            "score": { "type": "number" },
            "contexts": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["docid", "path", "title", "snippet", "score", "contexts"]
    });
    let search_results_schema = json!({
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": search_result_schema
            }
        },
        "required": ["results"]
    });
    let document_payload_schema = json!({
        "type": "object",
        "properties": {
            "docid": { "type": "string" },
            "path": { "type": "string" },
            "title": { "type": ["string", "null"] },
            "content": { "type": "string" }
        },
        "required": ["docid", "path", "title", "content"]
    });
    let multi_get_schema = json!({
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": document_payload_schema
            }
        },
        "required": ["results"]
    });
    let status_schema = json!({
        "type": "object",
        "properties": {
            "mode": { "type": "string" },
            "db_path": { "type": "string" },
            "applied_migrations": { "type": "integer" },
            "has_documents_fts": { "type": "boolean" },
            "has_vectors_vec": { "type": "boolean" },
            "vectors_note": { "type": ["string", "null"] },
            "vector_mode": { "type": "string" },
            "embedding_dimensions": { "type": "integer" },
            "total_collections": { "type": "integer" },
            "total_contexts": { "type": "integer" },
            "total_documents": { "type": "integer" },
            "total_chunks": { "type": "integer" }
        },
        "required": [
            "mode",
            "db_path",
            "applied_migrations",
            "has_documents_fts",
            "has_vectors_vec",
            "vectors_note",
            "vector_mode",
            "embedding_dimensions",
            "total_collections",
            "total_contexts",
            "total_documents",
            "total_chunks"
        ]
    });

    let mut tools = vec![
        ToolDef {
            name: "search",
            description: "Execute BM25 keyword search. Safe in offline mode.",
            input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
            output_schema: search_results_schema.clone(),
        },
        ToolDef {
            name: "get",
            description: "Retrieve one document by selector. Safe in offline mode.",
            input_schema: json!({"type":"object","properties":{"selector":{"type":"string"}},"required":["selector"]}),
            output_schema: document_payload_schema,
        },
        ToolDef {
            name: "multi_get",
            description: "Retrieve multiple documents by pattern. Safe in offline mode.",
            input_schema: json!({"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}),
            output_schema: multi_get_schema,
        },
        ToolDef {
            name: "status",
            description: "Return index health and metadata. Safe in offline mode.",
            input_schema: json!({"type":"object","properties":{}}),
            output_schema: status_schema,
        },
    ];

    if cfg.mode != crate::config::ModeConfig::Offline {
        tools.insert(
            1,
            ToolDef {
                name: "vector_search",
                description: "Execute semantic vector search. Enhanced mode only.",
                input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
                output_schema: search_results_schema.clone(),
            },
        );
        tools.insert(
            2,
            ToolDef {
                name: "deep_search",
                description: "Execute hybrid deep search. Enhanced mode only.",
                input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
                output_schema: search_results_schema,
            },
        );
    }

    tools
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

#[cfg(test)]
mod tests {
    use super::{execute_tool_call, mcp_tools};
    use crate::{
        chunker::Chunk,
        cli::{Cli, Commands, StatusArgs},
        config,
        db::{CollectionUpsert, Database},
    };
    use serde_json::json;
    use tempfile::tempdir;

    fn cfg_with_db(path: &std::path::Path) -> config::Config {
        let cli = Cli {
            config: None,
            db_path: Some(path.to_path_buf()),
            api_base_url: None,
            api_key: None,
            offline: true,
            model_embedding: None,
            model_embedding_dim: None,
            model_llm: None,
            model_reranker: None,
            command: Commands::Status(StatusArgs {
                verbose: false,
                smoke_api: false,
            }),
        };
        config::load(&cli).expect("load config")
    }

    #[tokio::test]
    async fn wraps_search_results_in_structured_content_object() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("index.sqlite");
        let cfg = cfg_with_db(&db_path);
        let db = Database::open(&cfg).expect("open db");

        db.upsert_collection(dir.path(), &CollectionUpsert::default())
            .expect("add collection");
        let collection = db
            .list_collections()
            .expect("list collections")
            .into_iter()
            .next()
            .expect("collection row");
        let doc_path = dir.path().join("metrics.md");
        db.upsert_document(
            "doc-1",
            collection.id,
            &doc_path,
            Some("Company Metrics"),
            "hash-1",
            None,
        )
        .expect("upsert document");
        db.replace_document_chunks(
            "doc-1",
            &doc_path,
            &[Chunk {
                content: "company metrics status green".to_string(),
                token_count: 4,
                start_line: 1,
                end_line: 1,
            }],
            &[vec![0.0_f32; cfg.models.embedding_dimensions]],
        )
        .expect("replace chunks");

        let result = execute_tool_call(&cfg, "search", json!({ "query": "company metrics" }))
            .await
            .expect("execute search");

        let structured = result
            .get("structuredContent")
            .and_then(serde_json::Value::as_object)
            .expect("structured content object");
        let results = structured
            .get("results")
            .and_then(serde_json::Value::as_array)
            .expect("results array");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["docid"], "doc-1");
    }

    #[tokio::test]
    async fn get_returns_error_when_selector_is_missing() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("index.sqlite");
        let cfg = cfg_with_db(&db_path);

        let err = execute_tool_call(&cfg, "get", json!({ "selector": "missing.md" }))
            .await
            .expect_err("missing selector should error");

        assert!(
            err.to_string()
                .contains("document not found for selector: missing.md"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn exposes_output_schema_for_search_tool() {
        let runtime_cfg = cfg_with_db(tempdir().expect("tempdir").path());
        let tool = mcp_tools(&runtime_cfg)
            .into_iter()
            .find(|tool| tool.name == "search")
            .expect("search tool");

        assert_eq!(tool.output_schema["type"], "object");
        assert_eq!(tool.output_schema["required"], json!(["results"]));
        assert_eq!(
            tool.output_schema["properties"]["results"]["type"],
            json!("array")
        );
    }
}
