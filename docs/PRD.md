# Product Requirements Document: `qmd-rs` (Lean Query Markup Documents)

## 1. Product Overview

`qmd-rs` is a lightweight, high-performance, on-device search engine for markdown documents, optimized for agentic workflows and AI context retrieval. It is a complete Rust rewrite of the original Node.js/Node-llama-cpp `qmd` project.

The primary differentiator of `qmd-rs` is its **lean architecture**: it strips out embedded machine learning runtimes (no bundled GGUF weights, `llama.cpp` bindings, or local VRAM management). Instead, it acts purely as an orchestrator and database engine, delegating all ML operations (embedding, query expansion, and re-ranking) to external, OpenAI-compatible APIs (such as a local Ollama daemon, vLLM, or OpenAI directly). It fully exposes its capabilities to AI agents via the Model Context Protocol (MCP) using the `rmcp` crate.

## 2. Goals & Non-Goals

**Goals:**

* **Minimal Footprint:** Deliver a single, fast-starting Rust binary with negligible memory overhead at idle.
* **API-Driven ML:** Connect to OpenAI-compatible REST endpoints for all language model tasks.
* **Native MCP Support:** Implement a robust MCP server using the `rmcp` crate to expose search and retrieval tools to agents.
* **Parity with Original `qmd`:** Replicate the hybrid search pipeline (RRF, position-aware blending), markdown-aware smart chunking, and CLI interface of the original project.

**Non-Goals:**

* Downloading, managing, or loading model weights directly into the application process.
* Compiling in heavy ML frameworks (e.g., Candle, Burn, ONNX).

## 3. Technology Stack

* **Language:** Rust (Edition 2021)
* **Async Runtime:** Tokio
* **Database:** `rusqlite` (SQLite)
* **Full-Text Search:** SQLite FTS5 extension
* **Vector Search:** `sqlite-vec` extension (statically linked or dynamically loaded)
* **MCP Protocol:** `rmcp` crate
* **HTTP Client:** `reqwest` (for communication with LLM APIs)
* **Serialization:** `serde` and `serde_json`

## 4. System Architecture & Workflows

### 4.1. External API Integration

The system relies on configuration (e.g., `~/.config/qmd/config.toml` or environment variables) to target external model servers.

* **Embeddings:** Calls `POST /v1/embeddings` to vectorize chunks and queries.
* **Query Expansion:** Calls `POST /v1/chat/completions` to generate search variations from the original user query.
* **Re-ranking:** Calls `POST /v1/chat/completions` (using a deterministic Yes/No logprobs prompt) or a dedicated `POST /v1/rerank` endpoint (if using an API like Cohere or a supported Ollama extension) to score the top K retrieval candidates.

### 4.2. Ingestion & Smart Chunking

The Rust implementation must parse markdown files from registered collections and chunk them while preserving semantic boundaries.

* **Target Size:** ~900 tokens per chunk with 15% overlap.
* **Heuristic Scoring:** Replicate the original decay-based boundary detection. Breakpoints are scored based on markdown syntax (H1 = 100, H2 = 90, Code Fence = 80, Paragraph = 20, Line break = 1).
* **Distance Decay:** When approaching the token limit, the algorithm evaluates a 200-token lookback window. Breakpoint scores are penalized by distance using the formula: `final_score = base_score * (1 - (distance/window)^2 * 0.7)`.
* **Code Fence Protection:** Code blocks must be preserved as single cohesive chunks whenever strictly possible.

### 4.3. Hybrid Search Pipeline

The core `query` command executes a multi-stage retrieval pipeline:

1. **Expansion:** Fetch 1-2 semantic variants of the query via the configured `/v1/chat/completions` API.
2. **Parallel Retrieval:** Dispatch the original query and its variants concurrently against both the FTS5 index (BM25) and the `sqlite-vec` index (Cosine Distance).
3. **Reciprocal Rank Fusion (RRF):** Combine the parallel result lists.
* Formula: `score = Σ(1/(k+rank+1))` where `k=60`.
* Apply a top-rank bonus (+0.05 for #1 exact matches, +0.02 for #2-3) to prevent dilution of highly accurate BM25 hits.


4. **Top-K Selection:** Isolate the top 30 candidates.
5. **Re-ranking & Blending:** Pass the 30 candidates to the LLM for re-ranking. Apply the final position-aware blend:
* RRF Rank 1-3: 75% RRF / 25% Reranker
* RRF Rank 4-10: 60% RRF / 40% Reranker
* RRF Rank 11+: 40% RRF / 60% Reranker



### 4.4. Collection & Context Management

* Support managing directories via `collection add/remove`.
* Support virtual path contexts (`qmd context add qmd://notes "Personal notes"`). Context strings are injected into the agent's context window alongside retrieved snippets to vastly improve LLM comprehension.

## 5. Model Context Protocol (MCP) Integration

The system must expose itself as an MCP server using `rmcp`. This allows agents (like Claude Desktop or custom tools) to directly call QMD's search capabilities.

**Required MCP Tools:**

* `qmd_search`: Execute fast BM25 keyword searches (supports collection filtering).
* `qmd_vector_search`: Execute semantic vector searches.
* `qmd_deep_search`: Execute the full hybrid pipeline (expansion + FTS + Vector + reranking).
* `qmd_get`: Retrieve a full document by path or 6-character document hash (docid).
* `qmd_multi_get`: Retrieve multiple documents by glob pattern or list of docids.
* `qmd_status`: Return index health, total documents, and collection metadata.

**Transports:**

* Standard I/O (`stdio`) mode for default subprocess execution by MCP clients.
* HTTP/SSE mode for long-running daemon setups, allowing multiple clients to query the same index without launching separate processes.

## 6. CLI Interface Requirements

The binary must expose the following commands matching the original tool:

* `qmd collection <add|remove|list|rename>`
* `qmd context <add|rm|list>`
* `qmd embed [--force]`
* `qmd search <query> [options]`
* `qmd vsearch <query> [options]`
* `qmd query <query> [options]`
* `qmd get <docid|path>`
* `qmd multi-get <pattern>`
* `qmd mcp [--http] [--port]`

## 7. Data Storage & Schema

Data is stored locally in `~/.cache/qmd/index.sqlite` using the following schema structure:

* `collections`: Registered directory paths and glob masks.
* `path_contexts`: Virtual path context descriptions.
* `documents`: Raw markdown metadata and docids (6-character hashes).
* `documents_fts`: FTS5 virtual table for BM25 search.
* `content_vectors`: Chunks and metadata.
* `vectors_vec`: `sqlite-vec` virtual table containing embeddings (`hash_seq` keys and float arrays).
