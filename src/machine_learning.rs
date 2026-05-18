//! Teacher response lookup, person-agnostic.
//!
//! Two-stage match:
//!   1. Section selection — char-similarity heuristic over every turn's
//!      tokens (not filtered by speaker).
//!   2. Nearest turn within that section by cosine similarity over
//!      embedding centroids supplied by the caller, again ignoring speaker.
//!
//! Returns the tokens of the turn IMMEDIATELY FOLLOWING the best match,
//! regardless of who wrote it. The model treats that as the supervised
//! template for what to say next.

use crate::dialogs::{Data, Turn};
use crate::embeddings::cosine;

pub fn string_similarity(word1: &str, word2: &str) -> u8 {
    let mut result: u8 = 0;
    if word1 == word2 {
        return 100;
    }
    let mut len_diff: u8 = 0;
    if word1.len() > word2.len() {
        len_diff = (word1.len() - word2.len()) as u8;
    }
    result = result.saturating_add(len_diff);
    for (c1, c2) in word1.chars().zip(word2.chars()) {
        if c1 == c2 {
            result = result.saturating_add(1);
        }
    }
    result
}

fn index_of_most_similar_section(sections: &[Vec<Turn>], memory_tokens: &[String]) -> usize {
    let mut best = 0usize;
    let mut best_score: u64 = 0;
    for (sec_idx, section) in sections.iter().enumerate() {
        let mut section_words: Vec<&String> = Vec::new();
        for turn in section {
            section_words.extend(turn.tokens.iter());
        }
        let mut score: u64 = 0;
        for (i, word) in section_words.iter().enumerate() {
            if let Some(mem_word) = memory_tokens.get(i) {
                score += string_similarity(word.as_str(), mem_word.as_str()) as u64;
            } else {
                break;
            }
        }
        if score > best_score {
            best_score = score;
            best = sec_idx;
        }
    }
    best
}

/// `embed_centroid` maps a token list to a fixed-length vector (e.g. via
/// `Embedding::centroid` after id lookup). Returns the tokens of the next
/// turn after the best-matching one in the best-matching section. Empty Vec
/// on failure (no sections, best match has no successor, etc).
pub fn teacher_response<F>(
    dialog: &Data,
    bot_memory_tokens: &[String],
    user_input_tokens: &[String],
    embed_centroid: F,
) -> Vec<String>
where
    F: Fn(&[String]) -> Vec<f32>,
{
    let candidates: Vec<&Vec<Turn>> = dialog
        .sections
        .iter()
        .filter(|s| !s.is_empty())
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    let sections_owned: Vec<Vec<Turn>> = candidates.iter().map(|s| (*s).clone()).collect();
    let sec_idx = index_of_most_similar_section(&sections_owned, bot_memory_tokens);
    let section = &sections_owned[sec_idx];

    let query = embed_centroid(user_input_tokens);

    let mut best: Option<(f32, usize)> = None;
    for (i, turn) in section.iter().enumerate() {
        if turn.tokens.is_empty() {
            continue;
        }
        let cand = embed_centroid(&turn.tokens);
        let sim = cosine(&query, &cand);
        best = Some(match best {
            Some((b, _)) if b >= sim => best.unwrap(),
            _ => (sim, i),
        });
    }

    match best {
        Some((_, idx)) if idx + 1 < section.len() => section[idx + 1].tokens.clone(),
        _ => Vec::new(),
    }
}
