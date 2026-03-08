// Rust guideline compliant 2026-03-08

use crate::cli::Cli;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Effective runtime configuration for qmd.
#[derive(Debug, Clone, Serialize)]
pub struct Config {
    /// API endpoint configuration.
    pub api: ApiConfig,
    /// Model names used for each pipeline stage.
    pub models: ModelsConfig,
    /// Query tuning parameters.
    pub query: QueryConfig,
    /// Local storage settings.
    pub storage: StorageConfig,
}

/// OpenAI-compatible API settings.
#[derive(Debug, Clone, Serialize)]
pub struct ApiConfig {
    /// Base URL for all model calls.
    pub base_url: String,
    /// API key or placeholder token.
    pub api_key: String,
}

/// Model selection settings.
#[derive(Debug, Clone, Serialize)]
pub struct ModelsConfig {
    /// Embedding model id.
    pub embedding: String,
    /// Chat/query-expansion model id.
    pub llm: String,
    /// Reranker model id.
    pub reranker: String,
}

/// Query behavior settings.
#[derive(Debug, Clone, Serialize)]
pub struct QueryConfig {
    /// Number of expansion variants.
    pub expansion_variants: u8,
    /// Candidate count passed to reranker.
    pub rerank_top_k: u16,
}

/// Local storage settings.
#[derive(Debug, Clone, Serialize)]
pub struct StorageConfig {
    /// Absolute or relative path to index.sqlite.
    pub db_path: PathBuf,
}

#[derive(Debug, Default, Deserialize)]
struct PartialConfig {
    api: Option<PartialApiConfig>,
    models: Option<PartialModelsConfig>,
    query: Option<PartialQueryConfig>,
    storage: Option<PartialStorageConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct PartialApiConfig {
    base_url: Option<String>,
    api_key: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct PartialModelsConfig {
    embedding: Option<String>,
    llm: Option<String>,
    reranker: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct PartialQueryConfig {
    expansion_variants: Option<u8>,
    rerank_top_k: Option<u16>,
}

#[derive(Debug, Default, Deserialize)]
struct PartialStorageConfig {
    db_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        let db_path = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("qmd")
            .join("index.sqlite");

        Self {
            api: ApiConfig {
                base_url: "http://localhost:11434/v1".to_string(),
                api_key: "ollama".to_string(),
            },
            models: ModelsConfig {
                embedding: "embeddinggemma:latest".to_string(),
                llm: "llama3.2:3b".to_string(),
                reranker: "sam860/qwen3-reranker:0.6b-Q8_0".to_string(),
            },
            query: QueryConfig {
                expansion_variants: 2,
                rerank_top_k: 30,
            },
            storage: StorageConfig { db_path },
        }
    }
}

/// Load configuration using defaults, file, env/CLI overrides, and validation.
///
/// # Arguments
/// `cli` - Parsed CLI arguments including optional overrides.
///
/// # Returns
/// A validated effective [`Config`].
///
/// # Errors
/// Returns an error if file parsing fails or validation constraints are violated.
///
/// # Panics
/// This function does not panic.
pub fn load(cli: &Cli) -> Result<Config> {
    let mut cfg = Config::default();

    if let Some(path) = resolve_config_path(cli.config.clone()) {
        if path.exists() {
            let parsed = load_file(&path)?;
            merge_file(&mut cfg, parsed);
        }
    }

    merge_cli(&mut cfg, cli);
    validate(&cfg)?;
    Ok(cfg)
}

fn resolve_config_path(cli_override: Option<PathBuf>) -> Option<PathBuf> {
    if cli_override.is_some() {
        return cli_override;
    }

    dirs::config_dir().map(|d| d.join("qmd").join("config.toml"))
}

fn load_file(path: &Path) -> Result<PartialConfig> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;

    toml::from_str(&data)
        .with_context(|| format!("failed to parse TOML config: {}", path.display()))
}

fn merge_file(cfg: &mut Config, file: PartialConfig) {
    if let Some(api) = file.api {
        if let Some(base_url) = api.base_url {
            cfg.api.base_url = base_url;
        }
        if let Some(api_key) = api.api_key {
            cfg.api.api_key = api_key;
        }
    }

    if let Some(models) = file.models {
        if let Some(embedding) = models.embedding {
            cfg.models.embedding = embedding;
        }
        if let Some(llm) = models.llm {
            cfg.models.llm = llm;
        }
        if let Some(reranker) = models.reranker {
            cfg.models.reranker = reranker;
        }
    }

    if let Some(query) = file.query {
        if let Some(expansion_variants) = query.expansion_variants {
            cfg.query.expansion_variants = expansion_variants;
        }
        if let Some(rerank_top_k) = query.rerank_top_k {
            cfg.query.rerank_top_k = rerank_top_k;
        }
    }

    if let Some(storage) = file.storage {
        if let Some(db_path) = storage.db_path {
            cfg.storage.db_path = db_path;
        }
    }
}

fn merge_cli(cfg: &mut Config, cli: &Cli) {
    if let Some(v) = &cli.db_path {
        cfg.storage.db_path = v.clone();
    }
    if let Some(v) = &cli.api_base_url {
        cfg.api.base_url = v.clone();
    }
    if let Some(v) = &cli.api_key {
        cfg.api.api_key = v.clone();
    }
    if let Some(v) = &cli.model_embedding {
        cfg.models.embedding = v.clone();
    }
    if let Some(v) = &cli.model_llm {
        cfg.models.llm = v.clone();
    }
    if let Some(v) = &cli.model_reranker {
        cfg.models.reranker = v.clone();
    }
}

fn validate(cfg: &Config) -> Result<()> {
    anyhow::ensure!(
        !cfg.api.base_url.trim().is_empty(),
        "api.base_url cannot be empty"
    );
    anyhow::ensure!(
        !cfg.models.embedding.trim().is_empty(),
        "models.embedding cannot be empty"
    );
    anyhow::ensure!(
        !cfg.models.llm.trim().is_empty(),
        "models.llm cannot be empty"
    );
    anyhow::ensure!(
        !cfg.models.reranker.trim().is_empty(),
        "models.reranker cannot be empty"
    );
    anyhow::ensure!(
        cfg.query.expansion_variants <= 4,
        "query.expansion_variants must be <= 4"
    );
    anyhow::ensure!(cfg.query.rerank_top_k > 0, "query.rerank_top_k must be > 0");
    anyhow::ensure!(
        !cfg.storage.db_path.as_os_str().is_empty(),
        "storage.db_path cannot be empty"
    );
    Ok(())
}
