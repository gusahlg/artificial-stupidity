//! Shared tokenizer. Lower-cases input, keeps each punctuation character as its
//! own token, glues apostrophes/digits into the surrounding word (so "don't"
//! and "it's" stay whole). Special structural tokens (`<SEC>` etc.) are passed
//! through verbatim because the corpus parser uses them as control markers.

pub const PAD: &str = "<PAD>";
pub const UNK: &str = "<UNK>";
pub const END_OF_BOT: &str = "</BOT>";
pub const SEC: &str = "<SEC>";
pub const USER_OPEN: &str = "<USER>";
pub const USER_CLOSE: &str = "</USER>";
pub const BOT_OPEN: &str = "<BOT>";

pub const SPECIAL_TOKENS: &[&str] = &[
    PAD, UNK, END_OF_BOT, SEC, USER_OPEN, USER_CLOSE, BOT_OPEN,
];

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for chunk in text.split_whitespace() {
        if SPECIAL_TOKENS.contains(&chunk) {
            tokens.push(chunk.to_string());
            continue;
        }
        let mut buf = String::new();
        for ch in chunk.chars().flat_map(|c| c.to_lowercase()) {
            if ch.is_alphanumeric() || ch == '\'' {
                buf.push(ch);
            } else {
                if !buf.is_empty() {
                    tokens.push(std::mem::take(&mut buf));
                }
                tokens.push(ch.to_string());
            }
        }
        if !buf.is_empty() {
            tokens.push(buf);
        }
    }
    tokens
}
