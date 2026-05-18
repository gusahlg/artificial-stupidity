//! Convert a DailyDialog raw text file into our PERSON_N corpus format and
//! append to `data/dialogs.txt`.
//!
//! Usage:
//!   cargo run --release --bin ingest_dailydialog -- /path/to/dialogues_text.txt
//!
//! The DailyDialog file has one dialog per line, with utterances separated
//! by the literal token `__eou__`. Speakers alternate. We map the first
//! utterance to PERSON_1 (a generic user), the second to PERSON_0 (the
//! bot's voice), and so on. This way the model learns response patterns
//! aimed at PERSON_0 specifically.
//!
//! How to get the file: DailyDialog is a freely-available academic dataset
//! (yanran.li/dailydialog). The plain-text form `dialogues_text.txt` is
//! mirrored in several public GitHub repos; download it manually and pass
//! the path to this binary. The classifier didn't allow me to fetch it
//! automatically.

use anyhow::{Context, Result};
use rust_fun::persons::{close_tag, open_tag};
use std::fs::{self, OpenOptions};
use std::io::Write;

const DIALOGS_OUT: &str = "data/dialogs.txt";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| {
        anyhow::anyhow!(
            "usage: ingest_dailydialog <path/to/dialogues_text.txt>"
        )
    })?;

    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path))?;

    let mut out = OpenOptions::new()
        .create(true)
        .append(true)
        .open(DIALOGS_OUT)
        .with_context(|| format!("open {} for append", DIALOGS_OUT))?;

    let user_open = open_tag(1);
    let user_close = close_tag(1);
    let bot_open = open_tag(0);
    let bot_close = close_tag(0);

    let mut sections = 0u32;
    let mut turns = 0u32;
    for line in raw.lines() {
        let utterances: Vec<&str> = line
            .split("__eou__")
            .map(|u| u.trim())
            .filter(|u| !u.is_empty())
            .collect();
        if utterances.len() < 2 {
            continue;
        }
        writeln!(out, "<SEC>")?;
        for (i, utt) in utterances.iter().enumerate() {
            // Speaker alternates. First (i=0) is the user (PERSON_1) so the
            // second (i=1, the response) trains the bot's PERSON_0 voice.
            let (open, close) = if i % 2 == 0 {
                (&user_open, &user_close)
            } else {
                (&bot_open, &bot_close)
            };
            writeln!(out, "{} {} {}", open, sanitize(utt), close)?;
            turns += 1;
        }
        sections += 1;
    }

    eprintln!(
        "ingest_dailydialog: appended {} sections / {} turns from {}",
        sections, turns, path
    );
    Ok(())
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
    // Collapse runs of whitespace to a single space; the corpus parser uses
    // split_whitespace so this is just for cleanliness.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
