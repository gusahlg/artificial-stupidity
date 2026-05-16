//! Teacher response lookup.
//!
//! Section selection (which conversation block resembles the current
//! conversation history) intentionally stays on the char-based heuristic.
//! User-line selection within a section now uses cosine similarity over
//! embedding centroids supplied by the caller, which gives semantic
//! matching instead of literal character overlap.

use crate::dialogs::{Data, Text};
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

fn index_of_most_similar_section(sections: &[Vec<Text>], memory_tokens: &[String]) -> usize {
    let mut best = 0usize;
    let mut best_score: u64 = 0;
    for (sec_idx, section) in sections.iter().enumerate() {
        let mut section_words: Vec<&String> = Vec::new();
        for text in section {
            if let Text::User(toks) = text {
                section_words.extend(toks.iter());
            }
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

/// `embed_centroid` should map a token list to a fixed-length embedding-
/// centroid vector (e.g. via `Embedding::centroid` after id lookup). Returns
/// the bot reply (as tokens) immediately following the best-matching user
/// line in the best-matching section. Empty Vec on failure.
pub fn teacher_response<F>(
    dialog: &Data,
    bot_memory_tokens: &[String],
    user_input_tokens: &[String],
    embed_centroid: F,
) -> Vec<String>
where
    F: Fn(&[String]) -> Vec<f32>,
{
    let candidates: Vec<&Vec<Text>> = dialog
        .sections
        .iter()
        .filter(|s| !s.is_empty())
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    let sections_owned: Vec<Vec<Text>> = candidates.iter().map(|s| (*s).clone()).collect();
    let sec_idx = index_of_most_similar_section(&sections_owned, bot_memory_tokens);
    let section = &sections_owned[sec_idx];

    let query = embed_centroid(user_input_tokens);

    let mut best: Option<(f32, usize)> = None;
    for (i, text) in section.iter().enumerate() {
        if let Text::User(toks) = text {
            let cand = embed_centroid(toks);
            let sim = cosine(&query, &cand);
            best = Some(match best {
                Some((b, _)) if b >= sim => best.unwrap(),
                _ => (sim, i),
            });
        }
    }

    match best {
        Some((_, idx)) if idx + 1 < section.len() => match &section[idx + 1] {
            Text::Bot(toks) => toks.clone(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}
