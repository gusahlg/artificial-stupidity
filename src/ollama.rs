//! Thin Ollama client. Used by `convert_discord` to ask a local LLM where
//! conversation section boundaries should fall.
//!
//! Reads `OLLAMA_HOST` (default `127.0.0.1`), `OLLAMA_PORT` (default `11434`)
//! and `OLLAMA_MODEL` (default `qwen2.5:7b`). Hits `/api/generate` with
//! `stream: false`, so we get a single JSON response.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub struct OllamaClient {
    base_url: String,
    pub model: String,
    agent: ureq::Agent,
}

impl OllamaClient {
    pub fn from_env() -> Self {
        let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port: u16 = std::env::var("OLLAMA_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(11434);
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(300))
            .build();
        Self {
            base_url: format!("http://{}:{}", host, port),
            model,
            agent,
        }
    }

    /// Send a prompt, return the model's response text.
    pub fn generate(&self, prompt: &str) -> Result<String> {
        let req = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
            options: GenerateOptions {
                // Low temperature for a structured-output task.
                temperature: 0.1,
                // Limit context to keep latency predictable.
                num_ctx: 8192,
            },
        };
        let body = serde_json::to_string(&req).context("serialize Ollama request")?;
        let url = format!("{}/api/generate", self.base_url);
        let resp = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body)
            .map_err(|e| anyhow!("POST {} failed: {}", url, e))?;
        let status = resp.status();
        let text = resp.into_string().context("read Ollama response body")?;
        if !(200..300).contains(&status) {
            return Err(anyhow!("Ollama returned {}: {}", status, text));
        }
        let parsed: GenerateResponse =
            serde_json::from_str(&text).context("parse Ollama response JSON")?;
        Ok(parsed.response)
    }
}

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: GenerateOptions,
}

#[derive(Serialize)]
struct GenerateOptions {
    temperature: f32,
    num_ctx: u32,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}
