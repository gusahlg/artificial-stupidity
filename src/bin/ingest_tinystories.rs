//! Ingest a sample of the TinyStories dataset and append it to
//! `data/dialogs.txt` in our PERSON_N format.
//!
//! TinyStories is a collection of short, simple narrative stories with a
//! constrained vocabulary, designed for very small language models to learn
//! grammar and basic prose patterns. Each story becomes one `<SEC>` where
//! every sentence is a `<PERSON_0>` turn — i.e. the model trains on
//! "given the previous sentence (as the bot), what comes next (as the
//! bot)." That's pure narrative-continuation supervision for PERSON_0,
//! exactly the voice we want the Discord bot to speak in.
//!
//! Usage:
//!   cargo run --release --bin ingest_tinystories -- \
//!       ~/TinyStories/TinyStories-valid.txt [--limit N]
//!
//! `--limit` (default 1000) caps the number of stories to ingest. The full
//! TinyStories train file is ~2 GB; a thousand stories is enough to teach
//! our toy model grammar without dwarfing the Discord corpus or blowing
//! up training time.

use anyhow::{Context, Result};
use rust_fun::persons::{close_tag, open_tag};
use std::fs::{self, OpenOptions};
use std::io::Write;

const DIALOGS_OUT: &str = "data/dialogs.txt";
const STORY_DELIM: &str = "<|endoftext|>";

/// Hard cap on tokens per sentence-turn so each turn fits inside the
/// training target window. Longer sentences are dropped at this length.
const MAX_TOKENS_PER_TURN: usize = 18;

/// Drop sentences shorter than this — they're usually fragments or noise.
const MIN_TOKENS_PER_TURN: usize = 3;

/// Drop stories that have fewer than this many turns after splitting —
/// nothing to train a pair from with only one turn.
const MIN_TURNS_PER_STORY: usize = 2;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .ok_or_else(|| anyhow::anyhow!(
            "usage: ingest_tinystories <path/to/TinyStories-*.txt> [--limit N]"
        ))?;
    let limit: usize = parse_limit(&args).unwrap_or(1000);

    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path))?;

    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(DIALOGS_OUT)
        .with_context(|| format!("open {} for append", DIALOGS_OUT))?;

    let bot_open = open_tag(0);
    let bot_close = close_tag(0);

    let mut written_sections = 0u32;
    let mut written_turns = 0u32;
    let mut stories_considered = 0u32;

    for raw_story in raw.split(STORY_DELIM) {
        if written_sections as usize >= limit {
            break;
        }
        stories_considered += 1;
        let story = raw_story.trim();
        if story.is_empty() {
            continue;
        }
        let sentences = split_into_sentences(story);
        let turns: Vec<String> = sentences
            .into_iter()
            .filter_map(|s| {
                let s = sanitize(&s);
                let tok_count = s.split_whitespace().count();
                if tok_count < MIN_TOKENS_PER_TURN || tok_count > MAX_TOKENS_PER_TURN {
                    None
                } else {
                    Some(s)
                }
            })
            .collect();
        if turns.len() < MIN_TURNS_PER_STORY {
            continue;
        }
        writeln!(out, "<SEC>")?;
        for t in &turns {
            writeln!(out, "{} {} {}", bot_open, t, bot_close)?;
            written_turns += 1;
        }
        written_sections += 1;
    }

    eprintln!(
        "ingest_tinystories: appended {} sections / {} turns to {} \
         (considered {} stories, limit {})",
        written_sections, written_turns, DIALOGS_OUT, stories_considered, limit
    );
    Ok(())
}

fn parse_limit(args: &[String]) -> Option<usize> {
    let i = args.iter().position(|a| a == "--limit")?;
    args.get(i + 1)?.parse().ok()
}

/// Split a story into sentences. Treats `.`, `!`, `?` as terminators when
/// followed by whitespace. Newlines collapse to spaces because the corpus
/// parser is whitespace-tokenized and a literal newline would split words
/// awkwardly.
fn split_into_sentences(text: &str) -> Vec<String> {
    let collapsed: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' || c == '\t' { ' ' } else { c })
        .collect();
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let bytes = collapsed.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        buf.push(c);
        if matches!(c, '.' | '!' | '?') {
            // Look ahead: only end the sentence if followed by space/EOF.
            let next = bytes.get(i + 1).copied().unwrap_or(b' ');
            if (next as char).is_whitespace() || i + 1 == bytes.len() {
                let trimmed = buf.trim().to_string();
                if !trimmed.is_empty() {
                    out.push(trimmed);
                }
                buf.clear();
            }
        }
        i += 1;
    }
    let leftover = buf.trim().to_string();
    if !leftover.is_empty() {
        out.push(leftover);
    }
    out
}

fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push('('),
            '>' => out.push(')'),
            '\n' | '\r' | '\t' => out.push(' '),
            c => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sentences_by_punctuation() {
        let text = "Hello world. How are you? I am fine!";
        let s = split_into_sentences(text);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], "Hello world.");
        assert_eq!(s[1], "How are you?");
        assert_eq!(s[2], "I am fine!");
    }

    #[test]
    fn collapses_newlines_within_paragraph() {
        let text = "First sentence.\nSecond sentence here.";
        let s = split_into_sentences(text);
        assert_eq!(s.len(), 2);
        assert!(s[0].starts_with("First"));
        assert!(s[1].starts_with("Second"));
    }

    #[test]
    fn handles_no_terminator() {
        let text = "no terminator here";
        let s = split_into_sentences(text);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], "no terminator here");
    }

    #[test]
    fn sanitize_strips_angles() {
        assert_eq!(sanitize("a <b> c"), "a (b) c");
    }
}
