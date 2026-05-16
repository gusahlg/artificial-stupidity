//! Vocab persistence. The canonical vocab is rebuilt from the corpus on each
//! startup; this module just writes it to `vocab.txt` so a human can inspect
//! what the network sees.

use std::io::Write;

pub fn save_vocab(vocab: &[String]) {
    let mut f = std::fs::File::create("vocab.txt").expect("failed to open vocab.txt");
    for w in vocab {
        writeln!(f, "{w}").expect("failed to write vocab.txt");
    }
}
