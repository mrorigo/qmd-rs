// Rust guideline compliant 2026-03-08

use crate::{api::ApiClient, config::Config, db::Database};
use anyhow::Result;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

/// Unified search result row.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    /// Document id.
    pub docid: String,
    /// Document path.
    pub path: String,
    /// Optional title.
    pub title: Option<String>,
    /// Matched snippet content.
    pub snippet: String,
    /// Final ranking score.
    pub score: f64,
    /// Matched context descriptions.
    pub contexts: Vec<String>,
}

#[derive(Debug, Clone)]
struct Candidate {
    docid: String,
    path: String,
    title: Option<String>,
    snippet: String,
    source_score: f64,
}

/// Execute keyword BM25 search.
///
/// # Arguments
/// `db` - Database repository.
/// `query` - User query text.
/// `limit` - Maximum result count.
///
/// # Returns
/// Ranked keyword results.
///
/// # Errors
/// Returns an error if the SQL query fails.
pub fn run_bm25_search(db: &Database, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let hits = db.bm25_search(query, limit)?;
    Ok(hits
        .into_iter()
        .enumerate()
        .map(|(idx, h)| SearchResult {
            docid: h.docid,
            path: h.path,
            title: h.title,
            snippet: h.snippet,
            score: 1.0 / (idx as f64 + 1.0),
            contexts: Vec::new(),
        })
        .collect())
}

/// Execute vector similarity search over stored chunk embeddings.
///
/// # Arguments
/// `cfg` - Effective runtime config.
/// `db` - Database repository.
/// `query` - User query text.
/// `limit` - Maximum result count.
///
/// # Returns
/// Ranked vector search results.
///
/// # Errors
/// Returns an error if API embedding or data loading fails.
pub async fn run_vector_search(
    cfg: &Config,
    db: &Database,
    query: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let client = ApiClient::from_config(cfg);
    let vectors = client.embed_texts(&cfg.models.embedding, &[query]).await?;
    let query_embedding_json = serde_json::to_string(&vectors[0])?;

    let mut scored = db
        .vector_search(&query_embedding_json, limit)?
        .into_iter()
        .map(|(hit, distance)| SearchResult {
            docid: hit.docid,
            path: hit.path,
            title: hit.title,
            snippet: hit.snippet,
            score: 1.0 / (1.0 + distance),
            contexts: Vec::new(),
        })
        .collect::<Vec<_>>();

    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(scored)
}

/// Execute the hybrid query pipeline with expansion, RRF, and rerank blending.
///
/// # Arguments
/// `cfg` - Effective runtime config.
/// `db` - Database repository.
/// `query` - User query text.
///
/// # Returns
/// Hybrid-ranked results.
///
/// # Errors
/// Returns an error when retrieval stages fail.
pub async fn run_hybrid_query(
    cfg: &Config,
    db: &Database,
    query: &str,
) -> Result<Vec<SearchResult>> {
    let client = ApiClient::from_config(cfg);

    let mut queries = vec![query.to_string()];
    let expansions = client
        .expand_queries(
            &cfg.models.llm,
            query,
            cfg.query.expansion_variants as usize,
        )
        .await
        .unwrap_or_default();
    for variant in expansions {
        if !queries.contains(&variant) {
            queries.push(variant);
        }
    }

    let mut all_lists: Vec<Vec<Candidate>> = Vec::new();
    for q in &queries {
        let bm = db
            .bm25_search(q, 25)?
            .into_iter()
            .map(|h| Candidate {
                docid: h.docid,
                path: h.path,
                title: h.title,
                snippet: h.snippet,
                source_score: 0.0,
            })
            .collect::<Vec<_>>();
        all_lists.push(bm);

        let vv = run_vector_search(cfg, db, q, 25)
            .await?
            .into_iter()
            .map(|h| Candidate {
                docid: h.docid,
                path: h.path,
                title: h.title,
                snippet: h.snippet,
                source_score: h.score,
            })
            .collect::<Vec<_>>();
        all_lists.push(vv);
    }

    let mut fused = rrf_fuse(&all_lists, 60.0);
    fused.sort_by(|a, b| b.source_score.total_cmp(&a.source_score));
    fused.truncate(cfg.query.rerank_top_k as usize);

    let rerank_scores = client
        .rerank_candidates(
            &cfg.models.reranker,
            query,
            &fused.iter().map(|c| c.snippet.clone()).collect::<Vec<_>>(),
        )
        .await
        .unwrap_or_else(|_| vec![0.0; fused.len()]);

    let mut blended = fused
        .into_iter()
        .enumerate()
        .map(|(idx, c)| {
            let rr = c.source_score;
            let rs = rerank_scores.get(idx).copied().unwrap_or(0.0);
            let contexts = db
                .context_descriptions_for_path(&c.path)
                .unwrap_or_default();
            let (w_rrf, w_rerank) = if idx <= 2 {
                (0.75, 0.25)
            } else if idx <= 9 {
                (0.60, 0.40)
            } else {
                (0.40, 0.60)
            };

            SearchResult {
                docid: c.docid,
                path: c.path,
                title: c.title,
                snippet: c.snippet,
                score: (rr * w_rrf) + (rs * w_rerank),
                contexts,
            }
        })
        .collect::<Vec<_>>();

    blended.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(blended)
}

fn rrf_fuse(lists: &[Vec<Candidate>], k: f64) -> Vec<Candidate> {
    let mut map: HashMap<String, Candidate> = HashMap::new();
    let mut bonus_seen: HashSet<String> = HashSet::new();

    for list in lists {
        for (rank, item) in list.iter().enumerate() {
            let key = format!("{}:{}", item.docid, item.snippet);
            let base = 1.0 / (k + rank as f64 + 1.0);
            let bonus = if rank == 0 {
                0.05
            } else if rank <= 2 {
                0.02
            } else {
                0.0
            };

            let entry = map.entry(key.clone()).or_insert_with(|| item.clone());
            entry.source_score += base;
            if !bonus_seen.contains(&key) {
                entry.source_score += bonus;
                bonus_seen.insert(key);
            }
        }
    }

    map.into_values().collect()
}
