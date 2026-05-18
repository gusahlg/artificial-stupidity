//! Vocab persistence. The canonical vocab is rebuilt from the corpus on each
//! startup; this module just writes it to `vocab.txt` so a human can inspect
//! what the network sees.

use anyhow::{Context, Result};
use std::io::Write;

pub const VOCAB_FILE: &str = "vocab.txt";

/// Write the vocab list to `vocab.txt`. Returns an error on IO failure
/// (callers decide whether to abort or warn-and-continue).
pub fn save_vocab_result(vocab: &[String]) -> Result<()> {
    let mut f = std::fs::File::create(VOCAB_FILE)
        .with_context(|| format!("create {}", VOCAB_FILE))?;
    for w in vocab {
        writeln!(f, "{w}").with_context(|| format!("write {}", VOCAB_FILE))?;
    }
    Ok(())
}

/// Back-compat wrapper that logs and continues on failure. Existing call
/// sites in `train.rs` / `serve.rs` / `main.rs` use this; new code should
/// prefer `save_vocab_result` and decide its own failure policy.
pub fn save_vocab(vocab: &[String]) {
    if let Err(e) = save_vocab_result(vocab) {
        eprintln!("warning: failed to write {}: {}", VOCAB_FILE, e);
    }
}
