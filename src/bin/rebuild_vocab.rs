//! Rebuild `vocab.txt` from `data/dialogs.txt` in isolation.
//!
//! `cargo run --bin train` already regenerates the vocab every run as a side
//! effect. This binary exists for the case where the corpus changed (e.g.
//! `convert_discord` just appended new turns) and you want the vocab file to
//! reflect that without paying for a training run.
//!
//! `memory::save_vocab` opens `vocab.txt` with `File::create`, so any stale
//! tokens from previous corpora are dropped.

use anyhow::Result;
use rust_fun::dialogs::Data;
use rust_fun::memory;

fn main() -> Result<()> {
    let data = Data::load()?;
    let vocab = data.build_vocab();
    memory::save_vocab(&vocab);
    println!(
        "vocab.txt rewritten: {} tokens (max PERSON id: {})",
        vocab.len(),
        data.max_person_id().map(|n| n as i64).unwrap_or(-1)
    );
    Ok(())
}
