// Rust guideline compliant 2026-03-08

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Command-line interface for qmd.
#[derive(Debug, Parser)]
#[command(name = "qmd", version, about = "Lean Query Markup Documents")]
pub struct Cli {
    /// Optional config file path override.
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// SQLite database path override.
    #[arg(long, env = "QMD_DB_PATH", global = true)]
    pub db_path: Option<PathBuf>,

    /// API base URL override.
    #[arg(long, env = "QMD_API_BASE_URL", global = true)]
    pub api_base_url: Option<String>,

    /// API key override.
    #[arg(long, env = "QMD_API_KEY", global = true)]
    pub api_key: Option<String>,

    /// Embedding model override.
    #[arg(long, env = "QMD_MODEL_EMBEDDING", global = true)]
    pub model_embedding: Option<String>,

    /// LLM model override.
    #[arg(long, env = "QMD_MODEL_LLM", global = true)]
    pub model_llm: Option<String>,

    /// Reranker model override.
    #[arg(long, env = "QMD_MODEL_RERANKER", global = true)]
    pub model_reranker: Option<String>,

    /// Top-level command.
    #[command(subcommand)]
    pub command: Commands,
}

/// Top-level qmd commands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Manage collections.
    Collection(CollectionCommand),
    /// Manage path contexts.
    Context(ContextCommand),
    /// Embed indexed markdown content.
    Embed(EmbedArgs),
    /// Run BM25 search.
    Search(QueryArgs),
    /// Run vector search.
    Vsearch(QueryArgs),
    /// Run hybrid query.
    Query(QueryArgs),
    /// Retrieve one document.
    Get(GetArgs),
    /// Retrieve multiple documents.
    MultiGet(MultiGetArgs),
    /// Start MCP server.
    Mcp(McpArgs),
    /// Show status and optional diagnostics.
    Status(StatusArgs),
}

/// Collection command group.
#[derive(Debug, Args)]
pub struct CollectionCommand {
    /// Collection action.
    #[command(subcommand)]
    pub action: CollectionAction,
}

/// Collection actions.
#[derive(Debug, Subcommand)]
pub enum CollectionAction {
    /// Add a collection path.
    Add { path: PathBuf },
    /// Remove a collection path.
    Remove { path: PathBuf },
    /// List collections.
    List,
    /// Rename collection alias.
    Rename { old_name: String, new_name: String },
}

/// Context command group.
#[derive(Debug, Args)]
pub struct ContextCommand {
    /// Context action.
    #[command(subcommand)]
    pub action: ContextAction,
}

/// Context actions.
#[derive(Debug, Subcommand)]
pub enum ContextAction {
    /// Add a virtual context.
    Add { scope: String, description: String },
    /// Remove a virtual context.
    Rm { scope: String },
    /// List virtual contexts.
    List,
}

/// Options for embed.
#[derive(Debug, Args)]
pub struct EmbedArgs {
    /// Force full re-embedding.
    #[arg(long)]
    pub force: bool,
}

/// Shared query argument shape.
#[derive(Debug, Args)]
pub struct QueryArgs {
    /// Query text.
    pub query: String,
}

/// Arguments for get.
#[derive(Debug, Args)]
pub struct GetArgs {
    /// Document id or path.
    pub docid_or_path: String,
}

/// Arguments for multi-get.
#[derive(Debug, Args)]
pub struct MultiGetArgs {
    /// Glob pattern or list input.
    pub pattern: String,
}

/// Arguments for MCP mode.
#[derive(Debug, Args)]
pub struct McpArgs {
    /// Use HTTP/SSE transport.
    #[arg(long)]
    pub http: bool,

    /// HTTP listen port.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
}

/// Arguments for status.
#[derive(Debug, Args)]
pub struct StatusArgs {
    /// Print effective config values.
    #[arg(long)]
    pub verbose: bool,

    /// Execute API smoke checks.
    #[arg(long)]
    pub smoke_api: bool,
}
