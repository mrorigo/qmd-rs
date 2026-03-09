# `qmd-rs` Phased Implementation Plan

## 0. Testing Baseline (Ollama Local Endpoint)

Before core implementation begins, standardize a local test target that satisfies the PRD's external API architecture.

### Target Endpoint and Models

- Base URL: `http://localhost:11434/v1`
- Embedding model: `embeddinggemma:latest`
- LLM model (query expansion): `llama3.2:3b`
- Re-ranker model: `sam860/qwen3-reranker:0.6b-Q8_0` (https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

### Required Config Surface

Implement config fields (file + env override):

```toml
# ~/.config/qmd/config.toml
[api]
base_url = "http://localhost:11434/v1"
api_key = "ollama" # placeholder for OpenAI-compatible clients

[models]
embedding = "embeddinggemma:latest"
llm = "llama3.2:3b"
reranker = "sam860/qwen3-reranker:0.6b-Q8_0"

[query]
expansion_variants = 2
rerank_top_k = 30
```

Environment variable overrides:

- `QMD_API_BASE_URL`
- `QMD_API_KEY`
- `QMD_MODEL_EMBEDDING`
- `QMD_MODEL_LLM`
- `QMD_MODEL_RERANKER`

### Baseline Validation

- `qmd embed` succeeds against local Ollama embeddings endpoint.
- `qmd query "test"` performs expansion via `llama3.2:3b` and reranking via `sam860/qwen3-reranker:0.6b-Q8_0`.
- Startup diagnostics print effective endpoint/model config in `qmd status --verbose`.

---

## Phase 1: Foundation and Project Skeleton

### Scope

- Initialize Rust workspace and crate layout.
- Add core dependencies: `tokio`, `rusqlite`, `serde`, `serde_json`, `reqwest`, `clap`, `anyhow`/`thiserror`, logging.
- Establish config loading precedence: defaults < config file < env vars < CLI flags.
- Create typed API client traits for embeddings, chat completion, and reranking.

### Deliverables

- Buildable `qmd` binary with placeholder command tree.
- Config module with validated parsing and helpful errors.
- API client smoke checks for `/v1/embeddings` and `/v1/chat/completions`.
- Reranker smoke checks with the configured dedicated reranker model.

### Exit Criteria

- `qmd --help` shows all PRD command groups.
- Local Ollama smoke tests pass with the baseline config.

---

## Phase 2: Storage Layer and Index Schema

### Scope

- Implement SQLite initialization/migrations.
- Create core tables: `collections`, `path_contexts`, `documents`, `content_vectors`.
- Create search indexes: `documents_fts` (FTS5), `vectors_vec` (`sqlite-vec`).
- Implement repository interfaces and transaction boundaries.

### Deliverables

- Deterministic schema migration system.
- Read/write operations for collections, contexts, and docs.
- Index metadata + health checks.

### Exit Criteria

- Fresh database bootstraps on first run.
- `qmd status` reports table/index presence and doc counts.

---

## Phase 3: Ingestion and Smart Chunking

### Scope

- Implement collection traversal with include/exclude globs.
- Build markdown-aware chunker with scoring and distance decay from PRD.
- Preserve code fences where possible.
- Generate stable doc/chunk identifiers.

### Deliverables

- `qmd embed` ingestion pipeline from filesystem -> chunking -> embeddings -> storage.
- Incremental embedding mode and `--force` full rebuild mode.

### Exit Criteria

- Chunk boundaries follow heading/code/paragraph heuristics.
- Re-runs avoid duplicate embeddings unless source changed or `--force` is set.

---

## Phase 4: Retrieval Pipeline (Search, Vector Search, Hybrid Query)

### Scope

- Implement BM25 keyword search over FTS5.
- Implement vector search with cosine distance via `sqlite-vec`.
- Implement expansion calls (1-2 variants), parallel retrieval, and RRF (`k=60` + top-rank bonus).
- Implement dedicated-model reranking and position-aware blending.

### Deliverables

- `qmd search`, `qmd vsearch`, and `qmd query` with consistent output schema.
- Tunable retrieval parameters in config.

### Exit Criteria

- `qmd query` returns ranked results using full PRD pipeline.
- Regression tests verify RRF + blending math.

---

## Phase 5: Document Access and Context Management

### Scope

- Implement `get`, `multi-get`, and context CRUD commands.
- Resolve doc retrieval by path or 6-char hash.
- Inject path-context metadata into query responses.

### Deliverables

- Fully functional `qmd context <add|rm|list>`.
- `qmd get` and `qmd multi-get` with robust glob/list behavior.

### Exit Criteria

- Retrieval commands produce stable, script-friendly output.
- Context strings are included in deep search response payloads.

---

## Phase 6: MCP Server via `rmcp`

### Scope

- Implement MCP tool handlers:
  - `search`
  - `vector_search`
  - `deep_search`
  - `get`
  - `multi_get`
  - `status`
- Support transports: stdio and HTTP/SSE.

### Deliverables

- `qmd mcp` command (stdio default, `--http` optional).
- Tool schemas and structured error mapping.

### Exit Criteria

- MCP client can execute all required tools end-to-end.
- Concurrent requests are handled safely with bounded resources.

---

## Phase 7: Hardening, Testing, and Release Readiness

### Scope

- Unit tests for chunking, ranking, config precedence, and schema migrations.
- Integration tests against local Ollama endpoint.
- Performance and memory checks for lean idle/runtime behavior.
- Packaging, docs, and release checklist.

### Deliverables

- CI workflow (format, lint, test).
- Benchmarks for ingestion/search latency and memory profile.
- Operator documentation for local and production API targets.

### Exit Criteria

- All tests green in CI.
- Documented operational runbook for `qmd-rs` local Ollama testing and deployment.

---

## Suggested Build Order (Execution Sequence)

1. Phase 1 + Testing Baseline
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7

## Definition of Done

- CLI command set matches PRD.
- Hybrid retrieval behavior matches PRD formulas and blending policy.
- MCP toolset and transports are complete.
- Local testing works with:
  - `http://localhost:11434/v1`
  - `embeddinggemma:latest`
  - `llama3.2:3b`
  - `sam860/qwen3-reranker:0.6b-Q8_0`
