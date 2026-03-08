// Rust guideline compliant 2026-03-08

use crate::{
    api::ApiClient,
    chunker::{chunk_markdown, ChunkerConfig},
    config::Config,
    db::Database,
};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use walkdir::WalkDir;

/// Summary of an embed run.
#[derive(Debug, Clone, Copy)]
pub struct EmbedSummary {
    /// Processed markdown files.
    pub scanned_files: usize,
    /// Files skipped due to unchanged content hash.
    pub skipped_files: usize,
    /// Documents written or updated.
    pub indexed_documents: usize,
    /// Total chunk rows written.
    pub indexed_chunks: usize,
}

/// Execute filesystem ingestion, markdown chunking, embedding calls, and DB upserts.
///
/// # Arguments
/// `cfg` - Effective runtime configuration.
/// `db` - Database handle.
/// `force` - If true, clear indexed docs/chunks before ingesting.
///
/// # Returns
/// Embed run summary counts.
///
/// # Errors
/// Returns an error for I/O, API, or database failures.
pub async fn run_embed(cfg: &Config, db: &Database, force: bool) -> Result<EmbedSummary> {
    if force {
        db.clear_documents_and_chunks()?;
    }

    let collections = db.list_collections()?;
    let client = ApiClient::from_config(cfg);

    let mut summary = EmbedSummary {
        scanned_files: 0,
        skipped_files: 0,
        indexed_documents: 0,
        indexed_chunks: 0,
    };

    for collection in collections {
        let collection_path = Path::new(&collection.path);
        if !collection_path.exists() {
            continue;
        }

        for entry in WalkDir::new(collection_path)
            .follow_links(false)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }

            if !is_markdown(entry.path()) {
                continue;
            }

            summary.scanned_files = summary.scanned_files.saturating_add(1);
            let path = entry.path();
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed to read file: {}", path.display()))?;
            let content_hash = content_hash(&text);

            if !force && db.is_document_unchanged(path, &content_hash)? {
                summary.skipped_files = summary.skipped_files.saturating_add(1);
                continue;
            }

            let docid = docid_for_path(path);
            let title = extract_title(&text);
            let chunks = chunk_markdown(&text, ChunkerConfig::default());
            if chunks.is_empty() {
                continue;
            }

            let inputs = chunks
                .iter()
                .map(|c| c.content.as_str())
                .collect::<Vec<_>>();
            let embeddings = client.embed_texts(&cfg.models.embedding, &inputs).await?;

            db.upsert_document(
                &docid,
                collection.id,
                path,
                title.as_deref(),
                &content_hash,
                path_modified(path),
            )?;

            db.replace_document_chunks(&docid, path, &chunks, &embeddings)?;

            summary.indexed_documents = summary.indexed_documents.saturating_add(1);
            summary.indexed_chunks = summary.indexed_chunks.saturating_add(chunks.len());
        }
    }

    Ok(summary)
}

fn is_markdown(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|v| v.to_str()),
        Some("md") | Some("markdown") | Some("mdx")
    )
}

fn content_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

fn docid_for_path(path: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    let hexed = hex::encode(hasher.finalize());
    hexed.chars().take(6).collect()
}

fn extract_title(text: &str) -> Option<String> {
    text.lines()
        .find(|line| line.trim_start().starts_with("# "))
        .map(|line| {
            line.trim_start()
                .trim_start_matches("# ")
                .trim()
                .to_string()
        })
}

fn path_modified(path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?;
    let secs = modified
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(secs.to_string())
}
