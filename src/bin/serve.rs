//! HTTP server that exposes the SuperSighurt LLM over the network.
//!
//! Single-process, single-worker: the Network struct holds mutable per-layer
//! caches that `neural_network::generate` walks on every request, so the
//! cheapest safe thing is to serialize requests. A `Mutex<Network>` makes
//! that contract explicit.
//!
//! Endpoints:
//!   GET  /healthz  -> 200 "ok"                          (unauth; readiness probe)
//!   POST /chat     -> 200 {"reply": "..."}              (X-API-Key required)
//!
//! Env vars:
//!   SIGHURT_BIND     default 127.0.0.1:8088
//!   SIGHURT_API_KEY  required (server refuses to start without one)
//!   SIGHURT_MODEL    default ./model.bin

use anyhow::{Context, Result, anyhow};
use rust_fun::dialogs::Data;
use rust_fun::gpu::Gpu;
use rust_fun::memory;
use rust_fun::neural_network::{
    CONTEXT_WINDOW, EMBED_DIM, HIDDEN_SIZE, NUMBER_OF_HIDDEN_LAYERS, generate, network_init,
};
use rust_fun::persist::{self, LoadedShape};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::Instant;
use tiny_http::{Header, Method, Response, Server};

const CHANNEL_HISTORY_CAP: usize = 20;

struct State {
    api_key: String,
    gpu: Gpu,
    net: Mutex<rust_fun::neural_network::Network>,
    vocab: Vec<String>,
    history: Mutex<HashMap<String, VecDeque<String>>>,
    rag: Mutex<rust_fun::rag::RagStore>,
}

const RAG_TOP_K: usize = 5;

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Extract a string field from a flat JSON object. Tolerant of whitespace and
/// simple escapes (\", \\, \n, \r, \t). Returns None if missing or malformed.
fn extract_string_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut i = body.find(&needle)?;
    i += needle.len();
    // skip whitespace + ':'
    let bytes = body.as_bytes();
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
        return None;
    }
    i += 1;
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1;
    let mut out = String::new();
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            i += 1;
            if i >= bytes.len() {
                return None;
            }
            match bytes[i] {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    if i + 4 >= bytes.len() {
                        return None;
                    }
                    let hex = std::str::from_utf8(&bytes[i + 1..i + 5]).ok()?;
                    let cp = u32::from_str_radix(hex, 16).ok()?;
                    out.push(char::from_u32(cp)?);
                    i += 4;
                }
                _ => return None,
            }
            i += 1;
        } else if c == b'"' {
            return Some(out);
        } else {
            // Decode UTF-8 char starting at byte i.
            let s = std::str::from_utf8(&bytes[i..]).ok()?;
            let ch = s.chars().next()?;
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    None
}

fn text_response(status: u16, body: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut r = Response::from_string(body.to_string()).with_status_code(status);
    r.add_header(
        Header::from_bytes(&b"Content-Type"[..], &b"text/plain; charset=utf-8"[..]).unwrap(),
    );
    r
}

fn json_response(status: u16, body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    let mut r = Response::from_string(body).with_status_code(status);
    r.add_header(
        Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap(),
    );
    r
}

fn header_value<'a>(req: &'a tiny_http::Request, name: &str) -> Option<&'a str> {
    let lname = name.to_ascii_lowercase();
    req.headers()
        .iter()
        .find(|h| h.field.as_str().as_str().to_ascii_lowercase() == lname)
        .map(|h| h.value.as_str())
}

fn handle_chat(state: &State, mut req: tiny_http::Request) -> Result<()> {
    // Auth
    let key = header_value(&req, "X-API-Key").unwrap_or("");
    if key != state.api_key {
        return req
            .respond(text_response(401, "unauthorized"))
            .map_err(|e| e.into());
    }

    // Body
    let mut body = String::new();
    req.as_reader().read_to_string(&mut body).ok();
    let input = match extract_string_field(&body, "input") {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => return req.respond(text_response(400, "missing or empty 'input'")).map_err(|e| e.into()),
    };
    let channel_id = extract_string_field(&body, "channel_id").unwrap_or_else(|| "default".to_string());
    let user = extract_string_field(&body, "user").unwrap_or_default();

    // Pull this channel's memory (lock briefly, clone into Vec).
    let channel_memory: Vec<String> = {
        let h = state.history.lock().unwrap();
        h.get(&channel_id)
            .map(|d| d.iter().cloned().collect())
            .unwrap_or_default()
    };

    // Run inference with RAG-augmented context. We compute the user-input
    // centroid via the embedding table, pull the top-K most similar past
    // turns from the corpus-indexed RagStore, and prepend them to the
    // channel-local memory. RAG hits act as "long-term memory" while the
    // channel memory captures the immediate conversation.
    let t0 = Instant::now();
    let reply = {
        let mut net = state.net.lock().unwrap();
        let rag = state.rag.lock().unwrap();
        let mut memory_vec: Vec<String> = Vec::new();
        if !rag.is_empty() {
            let vocab = rust_fun::neural_network::VocabIndex::new(&state.vocab);
            let q_tokens = rust_fun::tokenizer::tokenize(&input);
            let q_ids = vocab.ids_or_unk(&q_tokens);
            let q_emb = net.embedding.centroid(&q_ids);
            for hit in rag.top_k(&q_emb, RAG_TOP_K) {
                memory_vec.push(rust_fun::rag::RagStore::render(hit));
            }
        }
        memory_vec.extend(channel_memory.iter().cloned());
        match generate(&state.gpu, &mut net, &input, &memory_vec, &state.vocab) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("generate failed: {e}");
                return req
                    .respond(text_response(503, "generation failed"))
                    .map_err(|e| e.into());
            }
        }
    };
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "[chat] channel={} user={:?} in={:?} ({} chars) reply={:?} ({} chars) dt={:.3}s",
        channel_id,
        user,
        input,
        input.len(),
        reply,
        reply.len(),
        dt,
    );

    // Append to history.
    {
        let mut h = state.history.lock().unwrap();
        let entry = h.entry(channel_id.clone()).or_default();
        entry.push_back(input.clone());
        entry.push_back(reply.clone());
        while entry.len() > CHANNEL_HISTORY_CAP {
            entry.pop_front();
        }
    }

    let body = format!("{{\"reply\":\"{}\"}}", json_escape(&reply));
    req.respond(json_response(200, body)).map_err(|e| e.into())
}

fn main() -> Result<()> {
    let bind = std::env::var("SIGHURT_BIND").unwrap_or_else(|_| "127.0.0.1:8088".to_string());
    let api_key = std::env::var("SIGHURT_API_KEY")
        .map_err(|_| anyhow!("SIGHURT_API_KEY env var is required"))?;
    if api_key.len() < 16 {
        return Err(anyhow!(
            "SIGHURT_API_KEY looks too short ({} chars); generate one with `openssl rand -hex 32`",
            api_key.len()
        ));
    }
    let model_path = std::env::var("SIGHURT_MODEL").unwrap_or_else(|_| "model.bin".to_string());

    let gpu = Gpu::new().context("init GPU/CPU backend")?;
    eprintln!("serve: backend = {}", gpu.device_name());

    let dialog = Data::load()?;
    let vocab = dialog.build_vocab();
    memory::save_vocab(&vocab);
    eprintln!("serve: vocab size = {}", vocab.len());

    let vocab_hash = persist::compute_vocab_hash(&vocab);
    let shape = LoadedShape {
        embed_dim: EMBED_DIM,
        context_window: CONTEXT_WINDOW,
        vocab_size: vocab.len(),
        hidden_size: HIDDEN_SIZE,
        hidden_layers: NUMBER_OF_HIDDEN_LAYERS,
        vocab_hash,
    };
    let net = match persist::load_with_vocab(&model_path, &gpu, shape, Some(&vocab)) {
        Ok(Some(n)) => {
            eprintln!("serve: loaded model from {model_path}");
            n
        }
        Ok(None) => {
            eprintln!("serve: no model at {model_path}; using fresh random weights (low quality)");
            network_init(
                &gpu,
                EMBED_DIM,
                CONTEXT_WINDOW,
                HIDDEN_SIZE,
                NUMBER_OF_HIDDEN_LAYERS,
                vocab.len(),
            )?
        }
        Err(e) => return Err(anyhow!("failed to load model at {model_path}: {e}")),
    };

    let rag = rust_fun::rag::RagStore::populate_from_corpus(&dialog, &net.embedding, &vocab);
    eprintln!("serve: RAG indexed {} turns", rag.len());

    let state = State {
        api_key,
        gpu,
        net: Mutex::new(net),
        vocab,
        history: Mutex::new(HashMap::new()),
        rag: Mutex::new(rag),
    };

    let server = Server::http(&bind).map_err(|e| anyhow!("bind {bind} failed: {e}"))?;
    eprintln!("serve: listening on {bind}");

    for req in server.incoming_requests() {
        match (req.method(), req.url()) {
            (Method::Get, "/healthz") => {
                let _ = req.respond(text_response(200, "ok"));
            }
            (Method::Post, "/chat") => {
                if let Err(e) = handle_chat(&state, req) {
                    eprintln!("chat handler error: {e}");
                }
            }
            (m, u) => {
                eprintln!("404 {} {}", m, u);
                let _ = req.respond(text_response(404, "not found"));
            }
        }
    }
    Ok(())
}
