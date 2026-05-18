//! Corpus loader for the PERSON_N format.
//!
//! Sections are sequences of `Turn` values, each carrying a per-section
//! PERSON id (a local discriminator that resets at every `<SEC>` boundary)
//! and tokenized content. The parser recognizes `<SEC>` between sections
//! and `<PERSON_N>...</PERSON_N>` for each turn.
//!
//! PERSON_N is intentionally **section-local**: the same Discord user can
//! be PERSON_2 in one section and PERSON_1 in another. The model uses the
//! tag only to distinguish speakers within a single exchange, not to
//! identify users globally. By convention PERSON_0 is the bot (when it
//! speaks in that section).

use crate::persons::{open_tag, parse_close_tag, parse_open_tag};
use crate::tokenizer::{PAD, SEC, UNK, tokenize};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub const DIALOGS_FILE: &str = "data/dialogs.txt";
pub const DIALOGS_CACHE_FILE: &str = "data/dialogs.bin";

/// File-format version for `dialogs.bin`. Bump if `Turn` shape changes.
const CACHE_VERSION: u32 = 1;

/// Default cap on the number of content tokens (excluding PAD/UNK/SEC and
/// PERSON tags) kept in the vocab. Tokens not in the top-K by corpus
/// frequency get mapped to `<UNK>` at training and inference time. The cap
/// is the main lever for training speed: the output softmax + Adam update
/// are O(vocab) per training step. 3000 keeps the bulk of useful tokens
/// while making CPU training tractable.
pub const DEFAULT_VOCAB_CONTENT_CAP: usize = 3000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Turn {
    pub person_id: u32,
    pub tokens: Vec<String>,
}

/// On-disk layout for the bincode cache. The `source_hash` is a hash of the
/// `dialogs.txt` content at parse time; we refuse to load the cache unless
/// it matches, so a stale cache after a corpus edit fails the check and we
/// fall back to re-parsing. `version` guards against shape changes to
/// `Turn` between releases.
#[derive(Serialize, Deserialize)]
struct CachePayload {
    version: u32,
    source_hash: u64,
    sections: Vec<Vec<Turn>>,
}

impl Turn {
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
}

#[derive(Default)]
pub struct Data {
    pub sections: Vec<Vec<Turn>>,
}

impl Data {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from the canonical path `data/dialogs.txt`. Missing file
    /// produces an empty corpus.
    pub fn load() -> Result<Self> {
        Self::load_from(DIALOGS_FILE)
    }

    pub fn load_from<P: AsRef<Path>>(dialogs_path: P) -> Result<Self> {
        Self::load_from_with_cache(dialogs_path, DIALOGS_CACHE_FILE)
    }

    /// Cache-aware load. Reads `dialogs.txt`, hashes it, and consults a
    /// sidecar bincode cache (`dialogs.bin`). On hash match we deserialize
    /// the cache and skip the full parse — ~100× faster on warm boot for
    /// our current corpus size. On miss (or cache absent / version skew)
    /// we parse from text and write a fresh cache.
    ///
    /// Cache writes are best-effort: if the disk is read-only or the
    /// rename fails, we log a warning and continue. The text file remains
    /// authoritative.
    pub fn load_from_with_cache<P: AsRef<Path>, Q: AsRef<Path>>(
        dialogs_path: P,
        cache_path: Q,
    ) -> Result<Self> {
        let dialogs_path = dialogs_path.as_ref();
        let cache_path = cache_path.as_ref();
        if !dialogs_path.exists() {
            return Ok(Self {
                sections: Vec::new(),
            });
        }
        let raw = fs::read_to_string(dialogs_path)
            .with_context(|| format!("read {:?}", dialogs_path))?;
        let source_hash = hash_corpus(&raw);

        if let Some(sections) = try_load_cache(cache_path, source_hash) {
            return Ok(Self { sections });
        }

        let sections = parse_corpus(&raw)?;
        write_cache_best_effort(cache_path, source_hash, &sections);
        Ok(Self { sections })
    }

    /// Highest PERSON id seen anywhere in the corpus. The vocab registers
    /// `<PERSON_0>` / `</PERSON_0>` through `<PERSON_max>` / `</PERSON_max>`
    /// so every tag that can appear has a stable slot. An empty corpus
    /// returns `None`.
    pub fn max_person_id(&self) -> Option<u32> {
        self.sections
            .iter()
            .flat_map(|s| s.iter())
            .map(|t| t.person_id)
            .max()
    }

    /// Build the canonical vocabulary: `<PAD>`, `<UNK>`, `<SEC>` first,
    /// then `<PERSON_0>` / `</PERSON_0>` through the highest PERSON id
    /// observed, then the **top-K content tokens by corpus frequency**.
    /// `K` defaults to `DEFAULT_VOCAB_CONTENT_CAP` and can be overridden
    /// via the `VOCAB_CONTENT_CAP` env var (0 = no cap, keep all).
    ///
    /// Ties in frequency are broken alphabetically so the vocab order is
    /// deterministic — critical because `model.bin`'s output-layer rows
    /// are indexed by vocab position and a different order at serve time
    /// would silently scramble outputs.
    pub fn build_vocab(&self) -> Vec<String> {
        let cap = std::env::var("VOCAB_CONTENT_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_VOCAB_CONTENT_CAP);
        self.build_vocab_with_cap(cap)
    }

    pub fn build_vocab_with_cap(&self, content_cap: usize) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();
        for t in [PAD, UNK, SEC] {
            seen.insert(t.to_string());
            out.push(t.to_string());
        }
        let max_pid = self.max_person_id().unwrap_or(0);
        for pid in 0..=max_pid {
            let o = open_tag(pid);
            let c = crate::persons::close_tag(pid);
            if seen.insert(o.clone()) {
                out.push(o);
            }
            if seen.insert(c.clone()) {
                out.push(c);
            }
        }

        let mut counts: HashMap<&str, u32> = HashMap::new();
        for section in &self.sections {
            for turn in section {
                for tok in &turn.tokens {
                    if !seen.contains(tok.as_str()) {
                        *counts.entry(tok.as_str()).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut ranked: Vec<(&&str, &u32)> = counts.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let take = if content_cap == 0 {
            ranked.len()
        } else {
            content_cap.min(ranked.len())
        };
        for (tok, _) in ranked.into_iter().take(take) {
            out.push(tok.to_string());
        }
        out
    }
}

fn hash_corpus(raw: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    raw.hash(&mut hasher);
    hasher.finish()
}

fn try_load_cache(cache_path: &Path, expected_hash: u64) -> Option<Vec<Vec<Turn>>> {
    if !cache_path.exists() {
        return None;
    }
    let bytes = fs::read(cache_path).ok()?;
    let payload: CachePayload = bincode::deserialize(&bytes).ok()?;
    if payload.version != CACHE_VERSION || payload.source_hash != expected_hash {
        return None;
    }
    Some(payload.sections)
}

fn write_cache_best_effort(cache_path: &Path, source_hash: u64, sections: &[Vec<Turn>]) {
    let payload = CachePayload {
        version: CACHE_VERSION,
        source_hash,
        sections: sections.to_vec(),
    };
    let bytes = match bincode::serialize(&payload) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("dialogs cache: serialize failed: {} (continuing)", e);
            return;
        }
    };
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = cache_path.with_extension("bin.tmp");
    if let Err(e) = fs::write(&tmp, &bytes) {
        eprintln!("dialogs cache: write {:?} failed: {} (continuing)", tmp, e);
        return;
    }
    if let Err(e) = fs::rename(&tmp, cache_path) {
        eprintln!(
            "dialogs cache: rename {:?} -> {:?} failed: {} (continuing)",
            tmp, cache_path, e
        );
    }
}

fn parse_corpus(raw: &str) -> Result<Vec<Vec<Turn>>> {
    let mut sections: Vec<Vec<Turn>> = Vec::new();
    let mut current: Vec<Turn> = Vec::new();
    let mut in_turn: Option<u32> = None;
    let mut buf = String::new();

    for (idx, chunk) in raw.split_whitespace().enumerate() {
        if chunk == SEC {
            if let Some(id) = in_turn.take() {
                anyhow::bail!(
                    "token {}: <SEC> while still inside <PERSON_{}>",
                    idx,
                    id
                );
            }
            if !current.is_empty() {
                sections.push(std::mem::take(&mut current));
            }
            continue;
        }
        if let Some(open_id) = parse_open_tag(chunk) {
            if let Some(prev) = in_turn {
                anyhow::bail!(
                    "token {}: nested <PERSON_{}> inside still-open <PERSON_{}>",
                    idx,
                    open_id,
                    prev
                );
            }
            in_turn = Some(open_id);
            buf.clear();
            continue;
        }
        if let Some(close_id) = parse_close_tag(chunk) {
            let open_id = in_turn.take().ok_or_else(|| {
                anyhow::anyhow!(
                    "token {}: </PERSON_{}> without matching open",
                    idx,
                    close_id
                )
            })?;
            if open_id != close_id {
                anyhow::bail!(
                    "token {}: opened <PERSON_{}> but closed </PERSON_{}>",
                    idx,
                    open_id,
                    close_id
                );
            }
            current.push(Turn {
                person_id: open_id,
                tokens: tokenize(&buf),
            });
            buf.clear();
            continue;
        }
        if in_turn.is_some() {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(chunk);
        }
        // tokens outside any turn and outside the SEC marker are ignored.
    }
    if let Some(id) = in_turn {
        anyhow::bail!("EOF inside <PERSON_{}>", id);
    }
    if !current.is_empty() {
        sections.push(current);
    }
    Ok(sections)
}

/// Convenience for tests and migration tools.
#[doc(hidden)]
pub fn parse_corpus_for_tests(raw: &str) -> Result<Vec<Vec<Turn>>> {
    parse_corpus(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_section_two_turns() {
        let raw = r#"
            <SEC>
            <PERSON_1> hello world </PERSON_1>
            <PERSON_0> hi back </PERSON_0>
        "#;
        let sections = parse_corpus(raw).unwrap();
        assert_eq!(sections.len(), 1);
        let s = &sections[0];
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].person_id, 1);
        assert_eq!(s[0].tokens, vec!["hello", "world"]);
        assert_eq!(s[1].person_id, 0);
        assert_eq!(s[1].tokens, vec!["hi", "back"]);
    }

    #[test]
    fn parses_multiple_sections() {
        let raw = r#"
            <SEC>
            <PERSON_1> a </PERSON_1>
            <SEC>
            <PERSON_2> b </PERSON_2>
        "#;
        let sections = parse_corpus(raw).unwrap();
        assert_eq!(sections.len(), 2);
    }

    #[test]
    fn rejects_mismatched_close() {
        let raw = "<PERSON_1> a </PERSON_2>";
        assert!(parse_corpus(raw).is_err());
    }

    #[test]
    fn rejects_unclosed_turn() {
        let raw = "<PERSON_1> a";
        assert!(parse_corpus(raw).is_err());
    }

    #[test]
    fn build_vocab_emits_observed_person_tags() {
        let raw = r#"
            <SEC>
            <PERSON_0> hi </PERSON_0>
            <PERSON_2> hey </PERSON_2>
        "#;
        let sections = parse_corpus(raw).unwrap();
        let data = Data { sections };
        assert_eq!(data.max_person_id(), Some(2));
        let vocab = data.build_vocab();
        // PAD, UNK, SEC, then PERSON_0/close, PERSON_1/close, PERSON_2/close,
        // then content tokens "hi" and "hey".
        assert!(vocab.contains(&"<PERSON_0>".to_string()));
        assert!(vocab.contains(&"</PERSON_2>".to_string()));
        // PERSON_1 is registered even though no turn used it: vocab slots are
        // assigned by max observed id, so all ids 0..=max have stable slots.
        assert!(vocab.contains(&"<PERSON_1>".to_string()));
    }

    #[test]
    fn empty_corpus_still_has_specials() {
        let data = Data::default();
        let vocab = data.build_vocab();
        assert_eq!(vocab[0], "<PAD>");
        assert_eq!(vocab[1], "<UNK>");
        assert_eq!(vocab[2], "<SEC>");
        // PERSON_0 open/close always present (max defaults to 0).
        assert!(vocab.contains(&"<PERSON_0>".to_string()));
        assert!(vocab.contains(&"</PERSON_0>".to_string()));
    }
}
