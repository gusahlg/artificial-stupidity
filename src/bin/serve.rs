//! HTTP server that exposes the SuperSighurt LLM over the network.
//!
//! The Network struct holds mutable per-layer caches that
//! `neural_network::generate` walks on every request, so generation is
//! serialized behind a `Mutex<Network>`. A small worker pool accepts
//! requests concurrently so `/healthz` stays responsive while a chat
//! request is generating.
//!
//! Endpoints:
//!   GET  /healthz  -> 200 "ok"                          (unauth; readiness probe)
//!   POST /chat     -> 200 {"reply": "..."}              (X-API-Key required)
//!
//! /chat request fields (JSON object, all values strings; only `input`
//! is required — unknown fields are ignored so older bots keep working):
//!   input              the message text (mentions pre-resolved to @name)
//!   channel_id         conversation key for per-channel memory
//!   user               author display name (log-only)
//!   user_id            author Discord snowflake; drives PERSON_N tagging
//!   user_is_bot        "true" when the author is another bot
//!   reply_to_user      display name of the replied-to message's author
//!   reply_to_user_id   snowflake of the replied-to author
//!   reply_to_text      content of the replied-to message
//!   reply_to_is_self   "true" when the replied-to message is OUR bot's
//!
//! Speakers are mapped to section-local-style PERSON_N tags per channel
//! (the bot itself is always PERSON_0), and input/history/reply context
//! are fed to the model wrapped in those tags — the same shape the
//! training corpus uses, unlike the old untagged plain-text prompts.
//!
//! Env vars:
//!   SIGHURT_BIND         default 127.0.0.1:8088
//!   SIGHURT_API_KEY      required (server refuses to start without one)
//!   SIGHURT_MODEL        default ./model.bin
//!   SIGHURT_CORPUS       default data/dialogs.txt — pin this to a snapshot
//!                        taken when the model was trained so weekly corpus
//!                        refreshes can't drift the vocab out from under a
//!                        live model (the v4 vocab-hash check would refuse
//!                        to load and the server would not come up).
//!   SIGHURT_ALLOW_FRESH  "1" to serve fresh random weights when the model
//!                        file is missing (default: refuse to start — random
//!                        weights in production are a silent-garbage trap).

use anyhow::{Context, Result, anyhow};
use rust_fun::dialogs::Data;
use rust_fun::gpu::Gpu;
use rust_fun::neural_network::{
    CONTEXT_WINDOW, EMBED_DIM, HIDDEN_SIZE, NUMBER_OF_HIDDEN_LAYERS, VocabIndex, generate,
    network_init,
};
use rust_fun::persist::{self, LoadedShape};
use rust_fun::persons;
use rust_fun::tokenizer::tokenize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tiny_http::{Header, Method, Response, Server};

/// Tagged turns kept per channel (2 per exchange -> ~10 exchanges).
const CHANNEL_HISTORY_CAP: usize = 20;
/// Most-similar corpus turns prepended as long-term memory.
const RAG_TOP_K: usize = 5;
/// Request body size cap; chat messages are tiny, anything bigger is abuse.
const MAX_BODY_BYTES: u64 = 64 * 1024;
/// Accept-loop workers. Generation itself still serializes on the model
/// mutex; the extra workers keep /healthz and auth errors fast.
const WORKERS: usize = 4;

/// Per-channel mapping of Discord user ids to PERSON_N tags, mirroring the
/// corpus convention: the bot is PERSON_0, everyone else gets 1, 2, 3... in
/// first-seen order. Ids wrap around max_person so we never emit a tag the
/// vocab doesn't have a slot for.
#[derive(Default)]
struct ChannelSpeakers {
    by_user: HashMap<String, u32>,
    assigned: u32,
}

struct State {
    api_key: String,
    gpu: Gpu,
    net: Mutex<rust_fun::neural_network::Network>,
    /// Leaked to 'static so the borrowing VocabIndex can live in State —
    /// the vocab is immutable for the process lifetime anyway.
    vocab: &'static [String],
    vocab_index: VocabIndex<'static>,
    /// Highest PERSON id with a vocab slot; speaker assignment wraps at this.
    max_person: u32,
    history: Mutex<HashMap<String, VecDeque<String>>>,
    speakers: Mutex<HashMap<String, ChannelSpeakers>>,
    rag: rust_fun::rag::RagStore,
}

impl State {
    /// Resolve (channel, user) to a PERSON id. `is_self` pins PERSON_0.
    /// Unknown/absent user ids fall back to PERSON_1 (the "some human"
    /// slot, consistent with the seed-pair corpus shape).
    fn person_for(&self, channel_id: &str, user_id: &str, is_self: bool) -> u32 {
        if is_self {
            return persons::BOT_PERSON_ID;
        }
        if user_id.is_empty() || self.max_person == 0 {
            return 1;
        }
        let mut speakers = lock_recover(&self.speakers);
        let ch = speakers.entry(channel_id.to_string()).or_default();
        if let Some(&id) = ch.by_user.get(user_id) {
            return id;
        }
        let id = (ch.assigned % self.max_person) + 1;
        ch.assigned += 1;
        ch.by_user.insert(user_id.to_string(), id);
        id
    }
}

/// Mirror of the ingest-side sanitizer (convert_discord::sanitize_for_corpus)
/// so request text is tokenized the same way the training corpus was — this
/// is a modeling correctness requirement, not just safety. The model never
/// saw literal angle brackets/newlines (mapped to `(`/`)`/space, which also
/// blocks `<PERSON_N>`/`<SEC>` tag forgery), and every URL / emoji-shortcode
/// in the corpus was collapsed to `__URL__` / `__EMOJI__` via the SAME
/// shared `text_utils` helpers. Without this, a pasted link shatters into
/// ~20 punctuation/UNK fragments that flood the 32-token context window and
/// evict the actual message — token sequences the model was never trained
/// on. (Discord `<@id>`/`<#id>`/`<@&id>` refs are already resolved to
/// `@name` or stripped on the bot side before they reach us; any that slip
/// through are defanged by the bracket mapping.)
fn sanitize_inference_text(s: &str) -> String {
    let mapped: String = s
        .chars()
        .map(|c| match c {
            '<' => '(',
            '>' => ')',
            '\n' | '\r' | '\t' => ' ',
            c => c,
        })
        .collect();
    let mut out = String::with_capacity(mapped.len());
    let mut first = true;
    for word in mapped.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        if rust_fun::text_utils::looks_like_url(word) {
            out.push_str("__URL__");
        } else if rust_fun::text_utils::is_emoji_shortcode(word) {
            out.push_str("__EMOJI__");
        } else {
            out.push_str(word);
        }
    }
    out
}

/// Wrap sanitized text in its speaker's PERSON tags — the exact shape of a
/// corpus turn, which is the distribution the model was trained on.
fn tagged_turn(person_id: u32, text: &str) -> String {
    format!(
        "{} {} {}",
        persons::open_tag(person_id),
        text,
        persons::close_tag(person_id)
    )
}

/// Lock, recovering from poison. A panic mid-generation (indexing/GPU bug)
/// poisons the model mutex; without recovery every subsequent request would
/// panic at `.lock().unwrap()`, killing workers one per request until the
/// service is a green-healthcheck outage. The cached state a panic leaves
/// behind is per-request scratch that the next `forward` overwrites, so
/// continuing on the inner value is safe.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn field_str(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(|f| f.as_str()).map(|s| s.to_string())
}

fn field_flag(v: &Value, key: &str) -> bool {
    match v.get(key) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s.eq_ignore_ascii_case("true"),
        _ => false,
    }
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

    // Reject an oversized declared body BEFORE reading it. Otherwise a
    // bogus `Content-Length: 18446744073709551615` would drive the body
    // read (and any drain) toward a ~2^64 allocation. The real chat
    // payload is a few hundred bytes.
    if let Some(len) = req.body_length() {
        if len as u64 > MAX_BODY_BYTES {
            return req
                .respond(text_response(413, "request body too large"))
                .map_err(|e| e.into());
        }
    }

    // Body
    let mut body = String::new();
    req.as_reader()
        .take(MAX_BODY_BYTES)
        .read_to_string(&mut body)
        .ok();
    let parsed: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => {
            return req
                .respond(text_response(400, "invalid JSON body"))
                .map_err(|e| e.into());
        }
    };
    let input = match field_str(&parsed, "input") {
        Some(s) if !s.trim().is_empty() => sanitize_inference_text(&s),
        _ => {
            return req
                .respond(text_response(400, "missing or empty 'input'"))
                .map_err(|e| e.into());
        }
    };
    let channel_id = field_str(&parsed, "channel_id").unwrap_or_else(|| "default".to_string());
    let user = field_str(&parsed, "user").unwrap_or_default();
    let user_id = field_str(&parsed, "user_id").unwrap_or_default();
    let reply_to_text = field_str(&parsed, "reply_to_text")
        .map(|s| sanitize_inference_text(&s))
        .filter(|s| !s.is_empty());
    let reply_to_user_id = field_str(&parsed, "reply_to_user_id").unwrap_or_default();
    let reply_to_is_self = field_flag(&parsed, "reply_to_is_self");

    // Speaker tags. The author is never our own bot (the Discord side drops
    // self-messages), so is_self is always false for the input author.
    let author_person = state.person_for(&channel_id, &user_id, false);
    let tagged_input = tagged_turn(author_person, &input);

    // The turn being replied to, as a tagged context turn. If it is
    // literally the previous thing in this channel's history (the common
    // "user replies to the bot's last message" case) we skip the extra
    // copy — it is already adjacent in context.
    let reply_context: Option<String> = reply_to_text.map(|text| {
        let pid = state.person_for(&channel_id, &reply_to_user_id, reply_to_is_self);
        tagged_turn(pid, &text)
    });

    // Context assembly + generation + history append all happen under the
    // net lock, so the whole per-channel exchange is atomic: two rapid
    // messages in one channel serialize, and the second sees the first's
    // appended turns instead of a stale snapshot taken before it ran.
    //
    // Context order = least important first (the model only sees the
    // trailing CONTEXT_WINDOW tokens): RAG "long-term memory" hits, then
    // channel history (oldest to newest), then the replied-to turn, then
    // (as `start`) the tagged input itself — so the reply target and the
    // input always survive the window truncation.
    let t0 = Instant::now();
    let reply = {
        let mut net = lock_recover(&state.net);

        let channel_memory: Vec<String> = {
            let h = lock_recover(&state.history);
            h.get(&channel_id)
                .map(|d| d.iter().cloned().collect())
                .unwrap_or_default()
        };

        let mut memory_vec: Vec<String> = Vec::new();
        if !state.rag.is_empty() {
            let q_tokens = tokenize(&input);
            let q_ids = state.vocab_index.ids_or_unk(&q_tokens);
            let q_emb = net.embedding.centroid(&q_ids);
            for hit in state.rag.top_k(&q_emb, RAG_TOP_K) {
                memory_vec.push(rust_fun::rag::RagStore::render(hit));
            }
        }
        memory_vec.extend(channel_memory.iter().cloned());
        if let Some(ctx) = &reply_context {
            if channel_memory.last() != Some(ctx) {
                memory_vec.push(ctx.clone());
            }
        }
        let reply = match generate(&state.gpu, &mut net, &tagged_input, &memory_vec, state.vocab) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("generate failed: {e}");
                return req
                    .respond(text_response(503, "generation failed"))
                    .map_err(|e| e.into());
            }
        };

        // Append this exchange as tagged turns so the next request in this
        // channel sees properly attributed speakers.
        {
            let mut h = lock_recover(&state.history);
            let entry = h.entry(channel_id.clone()).or_default();
            entry.push_back(tagged_input.clone());
            entry.push_back(tagged_turn(
                persons::BOT_PERSON_ID,
                &sanitize_inference_text(&reply),
            ));
            while entry.len() > CHANNEL_HISTORY_CAP {
                entry.pop_front();
            }
        }
        reply
    };
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "[chat] channel={} user={:?} person={} reply_ctx={} in={:?} ({} chars) reply={:?} ({} chars) dt={:.3}s",
        channel_id,
        user,
        author_person,
        reply_context.is_some(),
        input,
        input.len(),
        reply,
        reply.len(),
        dt,
    );

    let body = serde_json::json!({ "reply": reply }).to_string();
    req.respond(json_response(200, body)).map_err(|e| e.into())
}

fn main() -> Result<()> {
    let bind = std::env::var("SIGHURT_BIND").unwrap_or_else(|_| "127.0.0.1:8088".to_string());
    // Preflight check mode (promote_model.sh) only loads the model; it never
    // serves, so it doesn't need a real API key.
    let check_mode = std::env::var("SIGHURT_CHECK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let api_key = std::env::var("SIGHURT_API_KEY").unwrap_or_default();
    if !check_mode {
        if api_key.is_empty() {
            return Err(anyhow!("SIGHURT_API_KEY env var is required"));
        }
        if api_key.len() < 16 {
            return Err(anyhow!(
                "SIGHURT_API_KEY looks too short ({} chars); generate one with `openssl rand -hex 32`",
                api_key.len()
            ));
        }
    }
    let model_path = std::env::var("SIGHURT_MODEL").unwrap_or_else(|_| "model.bin".to_string());
    let corpus_path = PathBuf::from(
        std::env::var("SIGHURT_CORPUS").unwrap_or_else(|_| "data/dialogs.txt".to_string()),
    );

    let gpu = Gpu::new().context("init GPU/CPU backend")?;
    eprintln!("serve: backend = {}", gpu.device_name());

    // The corpus drives vocab + RAG and must be the snapshot the model was
    // trained against — see SIGHURT_CORPUS in the module docs. The cache
    // sidecar lives next to whatever corpus we were pointed at. Note serve
    // deliberately does NOT rewrite vocab.txt (that is a training-side
    // artifact; a read-only server should not race the trainer for it).
    let dialog = Data::load_from_with_cache(&corpus_path, corpus_path.with_extension("bin"))?;
    let vocab = dialog.build_vocab();
    eprintln!(
        "serve: corpus = {} ({} sections), vocab size = {}",
        corpus_path.display(),
        dialog.sections.len(),
        vocab.len()
    );

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
            let allow_fresh = std::env::var("SIGHURT_ALLOW_FRESH")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !allow_fresh {
                return Err(anyhow!(
                    "no model at {model_path}; refusing to serve random weights \
                     (set SIGHURT_ALLOW_FRESH=1 to override for testing)"
                ));
            }
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

    // Preflight check mode: load-and-exit. scripts/promote_model.sh runs
    // `SIGHURT_CHECK=1 serve` against a candidate (model, corpus) pair so a
    // vocab-hash mismatch is caught BEFORE the pair is promoted into place —
    // otherwise the path-watcher restart would loop on a model serve refuses
    // to load. Reaching here means the model loaded against this corpus, so
    // the pair is consistent.
    if check_mode {
        eprintln!("serve: check ok — model loads against this corpus (vocab hash matches)");
        return Ok(());
    }

    let rag = rust_fun::rag::RagStore::populate_from_corpus(&dialog, &net.embedding, &vocab);
    eprintln!("serve: RAG indexed {} turns", rag.len());

    // Leak the vocab: it is immutable for the process lifetime and the
    // cached VocabIndex borrows from it.
    let vocab: &'static [String] = Box::leak(vocab.into_boxed_slice());
    let vocab_index = VocabIndex::new(vocab);
    let max_person = vocab
        .iter()
        .filter_map(|t| persons::parse_open_tag(t))
        .max()
        .unwrap_or(0);
    eprintln!("serve: PERSON tag slots 0..={max_person}");

    let state = Arc::new(State {
        api_key,
        gpu,
        net: Mutex::new(net),
        vocab,
        vocab_index,
        max_person,
        history: Mutex::new(HashMap::new()),
        speakers: Mutex::new(HashMap::new()),
        rag,
    });

    let server = Arc::new(Server::http(&bind).map_err(|e| anyhow!("bind {bind} failed: {e}"))?);
    eprintln!("serve: listening on {bind} with {WORKERS} workers");

    let mut handles = Vec::new();
    for _ in 0..WORKERS {
        let server = Arc::clone(&server);
        let state = Arc::clone(&state);
        handles.push(std::thread::spawn(move || {
            loop {
                let req = match server.recv() {
                    Ok(r) => r,
                    Err(e) => {
                        // tiny_http's accept thread `break`s permanently on
                        // the first listener error (e.g. fd exhaustion) and
                        // drops the socket — recv() then never yields another
                        // request. Treat it as FATAL so systemd's
                        // Restart=on-failure brings the service back, instead
                        // of a live-but-permanently-deaf process.
                        eprintln!("serve: fatal accept error: {e}; exiting for restart");
                        std::process::exit(1);
                    }
                };
                // catch_unwind so a panic inside generation (indexing/GPU
                // bug) kills only this request, not the worker. The model
                // mutex is taken with poison recovery below, so a panic that
                // poisoned it doesn't cascade into every later request.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
                }));
                if let Err(_panic) = result {
                    // The Request was moved into the closure; on unwind it is
                    // dropped, and tiny_http sends the client an automatic
                    // 500. Log and keep serving.
                    eprintln!("serve: request handler panicked; recovered, continuing");
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    // Every worker loops forever or exit(1)s on a fatal accept error, so a
    // normal return here means all workers vanished unexpectedly — exit
    // non-zero so the service restarts rather than "succeeding".
    eprintln!("serve: all workers exited unexpectedly");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_maps_angle_brackets_and_whitespace() {
        assert_eq!(
            sanitize_inference_text("<PERSON_0> hi </PERSON_0>\nnew\tline"),
            "(PERSON_0) hi (/PERSON_0) new line"
        );
    }

    #[test]
    fn tagged_turn_shape() {
        assert_eq!(tagged_turn(0, "hello"), "<PERSON_0> hello </PERSON_0>");
        assert_eq!(tagged_turn(3, "yo"), "<PERSON_3> yo </PERSON_3>");
    }

    #[test]
    fn field_flag_accepts_string_and_bool() {
        let v: Value = serde_json::json!({"a": "true", "b": true, "c": "false", "d": 1});
        assert!(field_flag(&v, "a"));
        assert!(field_flag(&v, "b"));
        assert!(!field_flag(&v, "c"));
        assert!(!field_flag(&v, "d"));
        assert!(!field_flag(&v, "missing"));
    }
}
