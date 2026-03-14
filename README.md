# qmd-rs

Lean Query Markup Documents in Rust: a fast, local-first markdown retrieval engine for humans and agents.

`qmd-rs` is a joyful little powerhouse: small binary, quick startup, practical CLI, and MCP tools that make your notes and docs easy to search from both terminal and AI workflows.

## Why qmd-rs

- Rust-native rewrite of the original qmd idea
- No in-process ML runtime baggage
- Uses OpenAI-compatible APIs for embeddings, expansion, and reranking
- SQLite + FTS5 core with vector-ready storage
- Agent-facing tool surface via MCP modes

## Architecture Snapshot

- Language: Rust 2021
- Runtime: Tokio
- Storage: SQLite (`rusqlite`, bundled SQLite)
- Full-text: FTS5 (`documents_fts`)
- Vector data: native `sqlite-vec` (`vec0`) index for vector retrieval
- API client: `reqwest` against OpenAI-compatible endpoints

## Install and Build

```bash
git clone <repo-url>
cd qmd-rs
cargo build
```

Run checks:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo check
```

## Configuration

Precedence: `defaults < config file < env/CLI`.

Default config file path:

- macOS/Linux: `~/.config/qmd/config.toml`

Example:

```toml
[api]
base_url = "http://localhost:11434/v1"
api_key = "ollama"

[models]
embedding = "embeddinggemma:latest"
embedding_dimensions = 1536
llm = "llama3.2:3b"
reranker = "sam860/qwen3-reranker:0.6b-Q8_0"

[query]
expansion_variants = 2
rerank_top_k = 30

[storage]
db_path = "~/.cache/qmd/index.sqlite"
```

Supported env vars:

- `QMD_DB_PATH`
- `QMD_API_BASE_URL`
- `QMD_API_KEY`
- `QMD_MODEL_EMBEDDING`
- `QMD_MODEL_EMBEDDING_DIM`
- `QMD_MODEL_LLM`
- `QMD_MODEL_RERANKER`

## CLI Commands

```bash
qmd collection <add|remove|list|rename>
qmd context <add|rm|list>
qmd embed [--force]
qmd search <query>
qmd vsearch <query>
qmd query <query>
qmd get <docid|path>
qmd multi-get <pattern>
qmd mcp [--http] [--port]
qmd status [--verbose] [--smoke-api]
```

## Quickstart

1. Register a collection:

```bash
qmd collection add /path/to/notes
qmd collection add /path/to/notes --name notes --include-glob "**/*.md" --exclude-glob "**/.git/**"
```

2. Embed markdown content:

```bash
qmd embed
```

3. Search:

```bash
qmd search "rust error handling"
qmd vsearch "safe async abstraction"
qmd query "best chunking strategy for markdown"
```

4. Fetch docs:

```bash
qmd get abc123
qmd multi-get "/path/to/notes/*.md"
```

## Path Contexts (`qmd context`)

Path contexts let you attach human-readable metadata to path prefixes. During hybrid query (`qmd query`), matched context descriptions are included in result output as `contexts=...`.

When a document path starts with a context `scope`, that context is considered a match.

Commands:

```bash
qmd context add <scope> <description>
qmd context rm <scope>
qmd context list
```

Examples:

```bash
# Add or update contexts
qmd context add /Users/alice/notes/work "Work docs: projects, architecture, ADRs"
qmd context add /Users/alice/notes/personal "Personal notes: journaling and planning"

# View configured scopes
qmd context list

# Remove a scope
qmd context rm /Users/alice/notes/personal
```

Typical workflow:

1. Add one or more collections with `qmd collection add ...`.
2. Add contexts for important subtrees (for example `/notes/work` and `/notes/runbooks`).
3. Run `qmd embed` to index markdown.
4. Run `qmd query "..."` and inspect the `contexts=...` line per result.

Notes:

- `qmd context add` is an upsert: adding an existing scope updates its description.
- Use absolute path prefixes for predictable matching.
- Contexts are attached to search result output for `qmd search`, `qmd vsearch`, and `qmd query`.

## Search Pipeline (Hybrid)

`qmd query` currently performs:

- LLM-based query expansion
- Parallel BM25 + vector retrieval
- Reciprocal Rank Fusion (`k=60`) with top-rank bonus
- Reranker call on top candidates
- Position-aware blend between RRF and reranker scores

## Chunking Strategy

Markdown chunking targets semantic boundaries with weighted breakpoints:

- `#` heading: strong break
- `##` heading: strong break
- code fence: protected break
- paragraph/blank lines: moderate break
- other lines: weak break

Also includes distance decay and overlap behavior to improve downstream retrieval quality.

## MCP Usage

### Stdio Mode (Full Toolset)

```bash
qmd mcp
```

Send JSON-RPC lines like:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"demo","version":"0.1.0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search","arguments":{"query":"test"}}}
```

Available tools:

- `search`
- `vector_search`
- `deep_search`
- `get`
- `multi_get`
- `status`

### HTTP/SSE Mode

```bash
qmd mcp --http --port 8080
```

MCP endpoint:

- `POST /mcp`: JSON-RPC requests/notifications
- `GET /mcp`: SSE heartbeat stream

Vector activation visibility:

- `qmd status` prints `vector.mode`
- `native-sqlite-vec` means vec0 is active

## CI

GitHub Actions workflow runs on push and PR:

- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test --all-targets`
- `cargo check --all-targets`

See [ci.yml](.github/workflows/ci.yml).

## Project Structure

- [src/main.rs](src/main.rs): CLI entrypoint and command dispatch
- [src/config.rs](src/config.rs): config loading and validation
- [src/db.rs](src/db.rs): migrations and repository operations
- [src/chunker.rs](src/chunker.rs): markdown-aware chunking
- [src/ingest.rs](src/ingest.rs): embed pipeline
- [src/search.rs](src/search.rs): BM25/vector/hybrid retrieval
- [src/api.rs](src/api.rs): OpenAI-compatible API calls
- [src/mcp.rs](src/mcp.rs): MCP stdio/HTTP server surfaces

## What’s Next

- HTTP transport hardening (timeouts/auth/rate limiting) and MCP ergonomics
- More deterministic reranker parsing and scoring contracts
- richer tests for ranking math and retrieval regression fixtures

## Contributing

If you open a PR, please run:

```bash
cargo fmt
cargo clippy -- -D warnings
cargo test
cargo check
```

Happy hacking. May your markdown always be discoverable and your retrieval results delightfully relevant.
