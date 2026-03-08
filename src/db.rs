// Rust guideline compliant 2026-03-08

use crate::{chunker::Chunk, config::Config};
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Once,
};

const MIGRATIONS: &[(&str, &str)] = &[
    (
        "0001_core",
        r#"
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    path TEXT NOT NULL UNIQUE,
    include_glob TEXT,
    exclude_glob TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS path_contexts (
    scope TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    docid TEXT PRIMARY KEY,
    collection_id INTEGER,
    path TEXT NOT NULL UNIQUE,
    title TEXT,
    content_hash TEXT,
    modified_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (collection_id) REFERENCES collections(id)
);

CREATE TABLE IF NOT EXISTS content_vectors (
    hash_seq TEXT PRIMARY KEY,
    docid TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    start_line INTEGER,
    end_line INTEGER,
    embedding_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (docid) REFERENCES documents(docid)
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    docid UNINDEXED,
    path,
    title,
    content
);
"#,
    ),
    (
        "0002_content_vectors_embedding_json",
        r#"
ALTER TABLE content_vectors ADD COLUMN embedding_json TEXT;
"#,
    ),
];

/// Collection record.
#[derive(Debug, Clone)]
pub struct Collection {
    /// Surrogate row id.
    pub id: i64,
    /// Optional human alias.
    pub name: Option<String>,
    /// Filesystem path.
    pub path: String,
}

/// Path context record.
#[derive(Debug, Clone)]
pub struct PathContext {
    /// Virtual scope identifier.
    pub scope: String,
    /// Human-readable description.
    pub description: String,
}

/// Index health summary for status output.
#[derive(Debug, Clone, Serialize)]
pub struct HealthReport {
    /// Database file location.
    pub db_path: PathBuf,
    /// Number of applied schema migrations.
    pub applied_migrations: usize,
    /// Whether `documents_fts` is present.
    pub has_documents_fts: bool,
    /// Whether `vectors_vec` exists and is queryable.
    pub has_vectors_vec: bool,
    /// Optional note when vector virtual table is unavailable.
    pub vectors_note: Option<String>,
    /// Active vector execution mode.
    pub vector_mode: String,
    /// Collection rows.
    pub total_collections: i64,
    /// Context rows.
    pub total_contexts: i64,
    /// Document rows.
    pub total_documents: i64,
    /// Chunk rows.
    pub total_chunks: i64,
}

/// BM25 row result.
#[derive(Debug, Clone)]
pub struct Bm25Hit {
    /// Document id.
    pub docid: String,
    /// Document path.
    pub path: String,
    /// Optional document title.
    pub title: Option<String>,
    /// Matched snippet.
    pub snippet: String,
}

/// Full document payload resolved from indexed chunks.
#[derive(Debug, Clone, Serialize)]
pub struct DocumentPayload {
    /// Document id.
    pub docid: String,
    /// Document path.
    pub path: String,
    /// Optional title.
    pub title: Option<String>,
    /// Reconstructed markdown body from stored chunks.
    pub content: String,
}

/// SQLite-backed repository and migration manager.
pub struct Database {
    conn: Connection,
    db_path: PathBuf,
}

static REGISTER_VEC0: Once = Once::new();

impl Database {
    /// Open the database, run migrations, and initialize virtual indexes.
    ///
    /// # Arguments
    /// `cfg` - Effective application configuration.
    ///
    /// # Returns
    /// Initialized [`Database`] ready for repository operations.
    ///
    /// # Errors
    /// Returns an error when opening, migrating, or creating directories fails.
    pub fn open(cfg: &Config) -> Result<Self> {
        register_vec0_extension();
        ensure_parent_dir(&cfg.storage.db_path)?;
        let conn = Connection::open(&cfg.storage.db_path).with_context(|| {
            format!(
                "failed to open sqlite db: {}",
                cfg.storage.db_path.display()
            )
        })?;

        conn.pragma_update(None, "foreign_keys", "ON")?;

        let db = Self {
            conn,
            db_path: cfg.storage.db_path.clone(),
        };

        db.run_migrations()?;
        db.ensure_vectors_virtual_table()?;
        db.vec_version()?;
        Ok(db)
    }

    /// Insert or update a collection entry keyed by path.
    pub fn upsert_collection(&self, path: &Path) -> Result<()> {
        let path_text = path.to_string_lossy();
        self.conn.execute(
            r#"
INSERT INTO collections(path, updated_at)
VALUES (?1, datetime('now'))
ON CONFLICT(path) DO UPDATE SET updated_at=datetime('now')
"#,
            params![path_text.as_ref()],
        )?;
        Ok(())
    }

    /// Remove a collection by exact path.
    pub fn remove_collection(&self, path: &Path) -> Result<usize> {
        let path_text = path.to_string_lossy();
        let changed = self.conn.execute(
            "DELETE FROM collections WHERE path = ?1",
            params![path_text.as_ref()],
        )?;
        Ok(changed)
    }

    /// Rename collection alias.
    pub fn rename_collection(&self, old_name: &str, new_name: &str) -> Result<usize> {
        let changed = self.conn.execute(
            "UPDATE collections SET name = ?2, updated_at=datetime('now') WHERE name = ?1",
            params![old_name, new_name],
        )?;
        Ok(changed)
    }

    /// List all collections sorted by insertion order.
    pub fn list_collections(&self) -> Result<Vec<Collection>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, path FROM collections ORDER BY id ASC")?;

        let rows = stmt.query_map([], |row| {
            Ok(Collection {
                id: row.get(0)?,
                name: row.get(1)?,
                path: row.get(2)?,
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(anyhow::Error::from)
    }

    /// Insert or update a context by scope.
    pub fn upsert_context(&self, scope: &str, description: &str) -> Result<()> {
        self.conn.execute(
            r#"
INSERT INTO path_contexts(scope, description, updated_at)
VALUES (?1, ?2, datetime('now'))
ON CONFLICT(scope) DO UPDATE SET description=excluded.description, updated_at=datetime('now')
"#,
            params![scope, description],
        )?;
        Ok(())
    }

    /// Remove a context by scope.
    pub fn remove_context(&self, scope: &str) -> Result<usize> {
        let changed = self
            .conn
            .execute("DELETE FROM path_contexts WHERE scope = ?1", params![scope])?;
        Ok(changed)
    }

    /// List contexts sorted by scope.
    pub fn list_contexts(&self) -> Result<Vec<PathContext>> {
        let mut stmt = self
            .conn
            .prepare("SELECT scope, description FROM path_contexts ORDER BY scope ASC")?;

        let rows = stmt.query_map([], |row| {
            Ok(PathContext {
                scope: row.get(0)?,
                description: row.get(1)?,
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(anyhow::Error::from)
    }

    /// Return whether a document exists with the same content hash.
    pub fn is_document_unchanged(&self, path: &Path, content_hash: &str) -> Result<bool> {
        let path_text = path.to_string_lossy();
        let existing = self
            .conn
            .query_row(
                "SELECT content_hash FROM documents WHERE path = ?1",
                params![path_text.as_ref()],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(existing.as_deref() == Some(content_hash))
    }

    /// Upsert document metadata.
    pub fn upsert_document(
        &self,
        docid: &str,
        collection_id: i64,
        path: &Path,
        title: Option<&str>,
        content_hash: &str,
        modified_at: Option<String>,
    ) -> Result<()> {
        let path_text = path.to_string_lossy();
        self.conn.execute(
            r#"
INSERT INTO documents(docid, collection_id, path, title, content_hash, modified_at, updated_at)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'))
ON CONFLICT(path) DO UPDATE SET
    docid=excluded.docid,
    collection_id=excluded.collection_id,
    title=excluded.title,
    content_hash=excluded.content_hash,
    modified_at=excluded.modified_at,
    updated_at=datetime('now')
"#,
            params![
                docid,
                collection_id,
                path_text.as_ref(),
                title,
                content_hash,
                modified_at
            ],
        )?;
        Ok(())
    }

    /// Replace all chunk rows and FTS rows for a document.
    pub fn replace_document_chunks(
        &self,
        docid: &str,
        path: &Path,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        anyhow::ensure!(
            chunks.len() == embeddings.len(),
            "chunk/embedding length mismatch"
        );

        self.conn.execute(
            "DELETE FROM content_vectors WHERE docid = ?1",
            params![docid],
        )?;
        self.conn.execute(
            "DELETE FROM vectors_vec WHERE hash_seq LIKE ?1",
            params![format!("{docid}:%")],
        )?;
        self.conn
            .execute("DELETE FROM documents_fts WHERE docid = ?1", params![docid])?;

        let title: Option<String> = self
            .conn
            .query_row(
                "SELECT title FROM documents WHERE docid = ?1",
                params![docid],
                |row| row.get(0),
            )
            .optional()?
            .flatten();

        let path_text = path.to_string_lossy();
        for (index, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
            let hash_seq = format!("{}:{:04}", docid, index);
            let embedding_json = serde_json::to_string(embedding)?;

            self.conn.execute(
                r#"
INSERT INTO content_vectors(
    hash_seq, docid, chunk_index, content, token_count, start_line, end_line, embedding_json
)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
"#,
                params![
                    hash_seq,
                    docid,
                    index as i64,
                    chunk.content,
                    chunk.token_count as i64,
                    chunk.start_line as i64,
                    chunk.end_line as i64,
                    embedding_json,
                ],
            )?;
            let vector_json = serde_json::to_string(embedding)?;
            self.conn.execute(
                "INSERT INTO vectors_vec(hash_seq, embedding) VALUES (?1, ?2)",
                params![hash_seq, vector_json],
            )?;

            self.conn.execute(
                "INSERT INTO documents_fts(docid, path, title, content) VALUES (?1, ?2, ?3, ?4)",
                params![docid, path_text.as_ref(), title.as_deref(), chunk.content],
            )?;
        }

        Ok(())
    }

    /// Clear indexed documents and chunk data.
    pub fn clear_documents_and_chunks(&self) -> Result<()> {
        self.conn.execute("DELETE FROM documents_fts", [])?;
        self.conn.execute("DELETE FROM vectors_vec", [])?;
        self.conn.execute("DELETE FROM content_vectors", [])?;
        self.conn.execute("DELETE FROM documents", [])?;
        Ok(())
    }

    /// Resolve one document by docid or exact path.
    pub fn get_document(&self, docid_or_path: &str) -> Result<Option<DocumentPayload>> {
        let row = self
            .conn
            .query_row(
                "SELECT docid, path, title FROM documents WHERE docid = ?1 OR path = ?1 LIMIT 1",
                params![docid_or_path],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .optional()?;

        match row {
            Some((docid, path, title)) => {
                let content = self.reconstruct_document_content(&docid)?;
                Ok(Some(DocumentPayload {
                    docid,
                    path,
                    title,
                    content,
                }))
            }
            None => Ok(None),
        }
    }

    /// Resolve multiple documents by glob path pattern or comma-separated ids/paths.
    pub fn multi_get_documents(&self, pattern: &str) -> Result<Vec<DocumentPayload>> {
        let selectors = pattern
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        if selectors.len() > 1 {
            let mut out = Vec::new();
            for selector in selectors {
                if let Some(doc) = self.get_document(selector)? {
                    out.push(doc);
                }
            }
            return Ok(out);
        }

        let mut stmt = self
            .conn
            .prepare("SELECT docid, path, title FROM documents ORDER BY path ASC")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;

        let matcher = glob::Pattern::new(pattern).ok();
        let mut out = Vec::new();
        for row in rows {
            let (docid, path, title) = row?;
            let is_match = matcher
                .as_ref()
                .map(|m| m.matches(&path))
                .unwrap_or_else(|| path.contains(pattern) || docid.contains(pattern));

            if is_match {
                out.push(DocumentPayload {
                    content: self.reconstruct_document_content(&docid)?,
                    docid,
                    path,
                    title,
                });
            }
        }
        Ok(out)
    }

    /// Return context descriptions associated with a path.
    pub fn context_descriptions_for_path(&self, path: &str) -> Result<Vec<String>> {
        let contexts = self.list_contexts()?;
        let matched = contexts
            .into_iter()
            .filter(|ctx| path.starts_with(&ctx.scope))
            .map(|ctx| ctx.description)
            .collect::<Vec<_>>();
        Ok(matched)
    }

    /// Run BM25 search against FTS table.
    pub fn bm25_search(&self, query: &str, limit: usize) -> Result<Vec<Bm25Hit>> {
        let mut stmt = self.conn.prepare(
            "SELECT docid, path, title, content FROM documents_fts WHERE documents_fts MATCH ?1 ORDER BY bm25(documents_fts) LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![query, limit as i64], |row| {
            Ok(Bm25Hit {
                docid: row.get(0)?,
                path: row.get(1)?,
                title: row.get(2)?,
                snippet: row.get(3)?,
            })
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(anyhow::Error::from)
    }

    /// Run native sqlite-vec search and return matched chunks with distance.
    pub fn vector_search(
        &self,
        query_embedding_json: &str,
        limit: usize,
    ) -> Result<Vec<(Bm25Hit, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT cv.docid, d.path, d.title, cv.content, vv.distance
             FROM vectors_vec vv
             JOIN content_vectors cv ON cv.hash_seq = vv.hash_seq
             JOIN documents d ON d.docid = cv.docid
             WHERE vv.embedding MATCH ?1
             ORDER BY vv.distance
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![query_embedding_json, limit as i64], |row| {
            Ok((
                Bm25Hit {
                    docid: row.get(0)?,
                    path: row.get(1)?,
                    title: row.get(2)?,
                    snippet: row.get(3)?,
                },
                row.get::<_, f64>(4)?,
            ))
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(anyhow::Error::from)
    }

    /// Compute status health and index presence.
    pub fn health_report(&self) -> Result<HealthReport> {
        let applied_migrations = self.migration_count()?;
        let has_documents_fts = self.has_table("documents_fts")?;
        let has_vectors_vec = self.has_table("vectors_vec")?;
        let vec_version = self.vec_version()?;
        let vectors_note = Some(format!("sqlite-vec active ({vec_version})"));

        Ok(HealthReport {
            db_path: self.db_path.clone(),
            applied_migrations,
            has_documents_fts,
            has_vectors_vec,
            vectors_note,
            vector_mode: "native-sqlite-vec".to_string(),
            total_collections: self.count("collections")?,
            total_contexts: self.count("path_contexts")?,
            total_documents: self.count("documents")?,
            total_chunks: self.count("content_vectors")?,
        })
    }

    fn run_migrations(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )?;

        for (version, sql) in MIGRATIONS {
            let already = self
                .conn
                .query_row(
                    "SELECT 1 FROM schema_migrations WHERE version = ?1",
                    params![version],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .is_some();

            if already {
                continue;
            }

            if *version == "0002_content_vectors_embedding_json"
                && self.column_exists("content_vectors", "embedding_json")?
            {
                self.conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?1, datetime('now'))",
                    params![version],
                )?;
                continue;
            }

            self.conn.execute_batch(sql)?;
            self.conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?1, datetime('now'))",
                params![version],
            )?;
        }
        Ok(())
    }

    fn ensure_vectors_virtual_table(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding FLOAT[1536]);",
        )?;
        Ok(())
    }

    fn vec_version(&self) -> Result<String> {
        let version: String = self
            .conn
            .query_row("SELECT vec_version()", [], |row| row.get(0))?;
        Ok(version)
    }

    fn count(&self, table: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        let value = self.conn.query_row(&sql, [], |row| row.get(0))?;
        Ok(value)
    }

    fn has_table(&self, name: &str) -> Result<bool> {
        let found = self
            .conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE name = ?1 LIMIT 1",
                params![name],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .is_some();
        Ok(found)
    }

    fn column_exists(&self, table: &str, column: &str) -> Result<bool> {
        let pragma = format!("PRAGMA table_info({table})");
        let mut stmt = self.conn.prepare(&pragma)?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        for row in rows {
            if row? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn migration_count(&self) -> Result<usize> {
        let total: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM schema_migrations", [], |row| {
                    row.get(0)
                })?;
        Ok(total as usize)
    }

    fn reconstruct_document_content(&self, docid: &str) -> Result<String> {
        let mut stmt = self.conn.prepare(
            "SELECT content FROM content_vectors WHERE docid = ?1 ORDER BY chunk_index ASC",
        )?;
        let rows = stmt.query_map(params![docid], |r| r.get::<_, String>(0))?;
        let mut content = String::new();
        for row in rows {
            content.push_str(&row?);
            if !content.ends_with('\n') {
                content.push('\n');
            }
        }
        Ok(content)
    }
}

fn register_vec0_extension() {
    REGISTER_VEC0.call_once(|| {
        type SqliteExtInit = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> i32;

        // SAFETY: sqlite3_auto_extension expects a C entrypoint pointer with static lifetime.
        // sqlite3_vec_init is provided by bundled sqlite-vec and remains valid for process life.
        unsafe {
            let entrypoint: SqliteExtInit = std::mem::transmute::<*const (), SqliteExtInit>(
                sqlite_vec::sqlite3_vec_init as *const (),
            );
            rusqlite::ffi::sqlite3_auto_extension(Some(entrypoint));
        }
    });
}

fn ensure_parent_dir(db_path: &Path) -> Result<()> {
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create db parent dir: {}", parent.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Database;
    use crate::{
        cli::{Cli, Commands, StatusArgs},
        config,
    };
    use tempfile::tempdir;

    fn cfg_with_db(path: &std::path::Path) -> config::Config {
        let cli = Cli {
            config: None,
            db_path: Some(path.to_path_buf()),
            api_base_url: None,
            api_key: None,
            model_embedding: None,
            model_llm: None,
            model_reranker: None,
            command: Commands::Status(StatusArgs {
                verbose: false,
                smoke_api: false,
            }),
        };
        config::load(&cli).expect("load config")
    }

    #[test]
    fn initializes_schema_and_health() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("index.sqlite");
        let cfg = cfg_with_db(&db_path);

        let db = Database::open(&cfg).expect("open db");
        let health = db.health_report().expect("health");
        assert!(health.applied_migrations >= 1);
        assert!(health.has_documents_fts);
    }

    #[test]
    fn collection_and_context_crud_roundtrip() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("index.sqlite");
        let cfg = cfg_with_db(&db_path);
        let db = Database::open(&cfg).expect("open db");

        db.upsert_collection(dir.path()).expect("add collection");
        let collections = db.list_collections().expect("list collections");
        assert_eq!(collections.len(), 1);

        db.upsert_context("/tmp", "Temporary files")
            .expect("add context");
        let contexts = db.list_contexts().expect("list contexts");
        assert_eq!(contexts.len(), 1);
    }
}
