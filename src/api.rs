// Rust guideline compliant 2026-03-08

use crate::config::Config;
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// OpenAI-compatible API client for embeddings and chat-completion calls.
#[derive(Clone)]
pub struct ApiClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl ApiClient {
    /// Construct an API client from runtime config.
    ///
    /// # Arguments
    /// `cfg` - Effective application configuration.
    ///
    /// # Returns
    /// An initialized API client.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            client: Client::new(),
            base_url: cfg.api.base_url.clone(),
            api_key: cfg.api.api_key.clone(),
        }
    }

    /// Run a minimal embeddings endpoint smoke test.
    ///
    /// # Arguments
    /// `model` - Embedding model id.
    ///
    /// # Returns
    /// `Ok(())` when at least one vector is returned.
    ///
    /// # Errors
    /// Returns an error when request or response parsing fails.
    pub async fn smoke_embeddings(&self, model: &str) -> Result<()> {
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let req = EmbeddingRequest {
            model: model.to_string(),
            input: vec!["qmd smoke test".to_string()],
        };

        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .context("embeddings request failed")?
            .error_for_status()
            .context("embeddings endpoint returned non-success")?
            .json::<EmbeddingResponse>()
            .await
            .context("failed to parse embeddings response")?;

        anyhow::ensure!(
            !response.data.is_empty(),
            "embeddings response had no vectors"
        );
        anyhow::ensure!(
            !response.data[0].embedding.is_empty(),
            "embeddings response vector was empty"
        );
        Ok(())
    }

    /// Run a minimal chat completion smoke test.
    ///
    /// # Arguments
    /// `model` - Chat model id.
    /// `prompt` - Prompt sent to the model.
    ///
    /// # Returns
    /// Returned content from the first choice, or empty content.
    ///
    /// # Errors
    /// Returns an error when request or response parsing fails.
    pub async fn smoke_chat(&self, model: &str, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let req = ChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            temperature: Some(0.0),
            max_tokens: Some(24),
        };

        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .context("chat completion request failed")?
            .error_for_status()
            .context("chat completion endpoint returned non-success")?
            .json::<ChatResponse>()
            .await
            .context("failed to parse chat completion response")?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(content)
    }

    /// Run a minimal reranker-model smoke test through chat completions.
    ///
    /// # Arguments
    /// `model` - Reranker model id.
    ///
    /// # Returns
    /// Returned content from the reranker smoke prompt.
    ///
    /// # Errors
    /// Returns an error when request or response parsing fails.
    pub async fn smoke_reranker(&self, model: &str) -> Result<String> {
        let prompt = "Return only YES if this passage answers the query. Query: qmd smoke. Passage: qmd is a markdown retrieval tool.";
        self.smoke_chat(model, prompt).await
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageOut,
}

#[derive(Debug, Deserialize)]
struct ChatMessageOut {
    content: Option<String>,
}
