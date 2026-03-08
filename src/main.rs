// Rust guideline compliant 2026-03-08

mod api;
mod chunker;
mod cli;
mod config;
mod db;
mod ingest;
mod mcp;
mod search;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, CollectionAction, Commands, ContextAction};
use db::Database;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let cli = Cli::parse();
    let cfg = config::load(&cli)?;
    let db = Database::open(&cfg)?;

    match cli.command {
        Commands::Collection(cmd) => match cmd.action {
            CollectionAction::Add { path } => {
                db.upsert_collection(&path)?;
                println!("collection.added path={}", path.display());
            }
            CollectionAction::Remove { path } => {
                let changed = db.remove_collection(&path)?;
                println!(
                    "collection.removed rows={} path={}",
                    changed,
                    path.display()
                );
            }
            CollectionAction::List => {
                for item in db.list_collections()? {
                    println!(
                        "collection id={} name={} path={}",
                        item.id,
                        item.name.unwrap_or_else(|| "-".to_string()),
                        item.path
                    );
                }
            }
            CollectionAction::Rename { old_name, new_name } => {
                let changed = db.rename_collection(&old_name, &new_name)?;
                println!(
                    "collection.renamed rows={} old_name={} new_name={}",
                    changed, old_name, new_name
                );
            }
        },
        Commands::Context(cmd) => match cmd.action {
            ContextAction::Add { scope, description } => {
                db.upsert_context(&scope, &description)?;
                println!("context.upserted scope={}", scope);
            }
            ContextAction::Rm { scope } => {
                let changed = db.remove_context(&scope)?;
                println!("context.removed rows={} scope={}", changed, scope);
            }
            ContextAction::List => {
                for item in db.list_contexts()? {
                    println!(
                        "context scope={} description={}",
                        item.scope, item.description
                    );
                }
            }
        },
        Commands::Embed(args) => {
            let summary = ingest::run_embed(&cfg, &db, args.force).await?;
            println!("embed.scanned_files={}", summary.scanned_files);
            println!("embed.skipped_files={}", summary.skipped_files);
            println!("embed.indexed_documents={}", summary.indexed_documents);
            println!("embed.indexed_chunks={}", summary.indexed_chunks);
        }
        Commands::Search(args) => {
            let results = search::run_bm25_search(&db, &args.query, 20)?;
            print_results(&results);
        }
        Commands::Vsearch(args) => {
            let results = search::run_vector_search(&cfg, &db, &args.query, 20).await?;
            print_results(&results);
        }
        Commands::Query(args) => {
            let results = search::run_hybrid_query(&cfg, &db, &args.query).await?;
            print_results(&results);
        }
        Commands::Get(args) => match db.get_document(&args.docid_or_path)? {
            Some(doc) => print_document(&doc),
            None => println!("get.not_found selector={}", args.docid_or_path),
        },
        Commands::MultiGet(args) => {
            let docs = db.multi_get_documents(&args.pattern)?;
            for doc in docs {
                print_document(&doc);
            }
        }
        Commands::Mcp(args) => {
            if args.http {
                mcp::run_http(cfg.clone(), args.port).await?;
            } else {
                mcp::run_stdio(cfg.clone()).await?;
            }
        }
        Commands::Status(args) => {
            let health = db.health_report()?;
            print_status(&cfg, &health, args.verbose);
            if args.smoke_api {
                let client = api::ApiClient::from_config(&cfg);
                client.smoke_embeddings(&cfg.models.embedding).await?;
                let llm = client
                    .smoke_chat(&cfg.models.llm, "Respond with: qmd-ok")
                    .await?;
                let rerank = client.smoke_reranker(&cfg.models.reranker).await?;
                println!("api_smoke.embeddings=ok");
                println!("api_smoke.chat={}", compact(&llm));
                println!("api_smoke.reranker={}", compact(&rerank));
            }
        }
    }

    Ok(())
}

fn print_status(cfg: &config::Config, health: &db::HealthReport, verbose: bool) {
    println!("qmd-rs status=ok");
    println!("db.path={}", health.db_path.display());
    println!("db.migrations_applied={}", health.applied_migrations);
    println!("index.documents_fts={}", health.has_documents_fts);
    println!("index.vectors_vec={}", health.has_vectors_vec);
    println!("vector.mode={}", health.vector_mode);
    println!("count.collections={}", health.total_collections);
    println!("count.contexts={}", health.total_contexts);
    println!("count.documents={}", health.total_documents);
    println!("count.chunks={}", health.total_chunks);

    if let Some(note) = &health.vectors_note {
        println!("index.vectors_vec_note={note}");
    }

    if verbose {
        println!("api.base_url={}", cfg.api.base_url);
        println!("api.api_key_set={}", !cfg.api.api_key.is_empty());
        println!("models.embedding={}", cfg.models.embedding);
        println!("models.llm={}", cfg.models.llm);
        println!("models.reranker={}", cfg.models.reranker);
        println!("query.expansion_variants={}", cfg.query.expansion_variants);
        println!("query.rerank_top_k={}", cfg.query.rerank_top_k);
        println!("storage.db_path={}", cfg.storage.db_path.display());
    }
}

fn compact(s: &str) -> String {
    s.trim().replace('\n', " ")
}

fn print_results(results: &[search::SearchResult]) {
    for (idx, result) in results.iter().enumerate() {
        println!(
            "result rank={} score={:.6} docid={} path={} title={}",
            idx + 1,
            result.score,
            result.docid,
            result.path,
            result.title.clone().unwrap_or_else(|| "-".to_string())
        );
        println!("snippet={}", compact(&result.snippet));
        if !result.contexts.is_empty() {
            println!("contexts={}", result.contexts.join(" | "));
        }
    }
}

fn print_document(doc: &db::DocumentPayload) {
    println!(
        "document docid={} path={} title={}",
        doc.docid,
        doc.path,
        doc.title.clone().unwrap_or_else(|| "-".to_string())
    );
    println!("{}", doc.content.trim_end());
}
