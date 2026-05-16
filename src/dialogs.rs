//! Corpus loader. Parses `data/dialogs.txt` and tokenizes each user/bot turn
//! with the shared tokenizer, so the rest of the system never sees raw text.

use crate::tokenizer::{
    BOT_OPEN, END_OF_BOT, PAD, SEC, UNK, USER_CLOSE, USER_OPEN, tokenize,
};
use std::collections::HashSet;
use std::fs;

#[derive(Clone, Debug)]
pub enum Text {
    User(Vec<String>),
    Bot(Vec<String>),
}

impl Text {
    pub fn tokens(&self) -> &[String] {
        match self {
            Text::User(t) | Text::Bot(t) => t,
        }
    }
}

#[derive(Clone, Default)]
pub struct Data {
    pub sections: Vec<Vec<Text>>,
}

impl Data {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read `data/dialogs.txt`, walk the `<SEC>`/`<USER>...</USER>`/
    /// `<BOT>...</BOT>` stream, and store each turn as a tokenized list.
    pub fn load(&mut self) {
        let raw = fs::read_to_string("data/dialogs.txt").expect("failed to read corpus");

        let mut sections: Vec<Vec<Text>> = Vec::new();
        let mut current: Vec<Text> = Vec::new();
        let mut mode_user = false;
        let mut mode_bot = false;
        let mut buf = String::new();

        for chunk in raw.split_whitespace() {
            match chunk {
                _ if chunk == SEC => {
                    if !current.is_empty() {
                        sections.push(std::mem::take(&mut current));
                    }
                }
                _ if chunk == USER_OPEN => {
                    mode_user = true;
                    buf.clear();
                }
                _ if chunk == USER_CLOSE => {
                    if mode_user {
                        current.push(Text::User(tokenize(&buf)));
                        mode_user = false;
                        buf.clear();
                    }
                }
                _ if chunk == BOT_OPEN => {
                    mode_bot = true;
                    buf.clear();
                }
                _ if chunk == END_OF_BOT => {
                    if mode_bot {
                        current.push(Text::Bot(tokenize(&buf)));
                        mode_bot = false;
                        buf.clear();
                    }
                }
                _ => {
                    if mode_user || mode_bot {
                        if !buf.is_empty() {
                            buf.push(' ');
                        }
                        buf.push_str(chunk);
                    }
                }
            }
        }
        if !current.is_empty() {
            sections.push(current);
        }
        self.sections = sections;
    }

    /// Build the canonical vocabulary: reserved specials first (so PAD=0,
    /// UNK=1, </BOT>=2 — stable across runs), then every distinct token
    /// from the corpus in first-seen order.
    pub fn build_vocab(&self) -> Vec<String> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();
        for &t in &[PAD, UNK, END_OF_BOT] {
            seen.insert(t.to_string());
            out.push(t.to_string());
        }
        for section in &self.sections {
            for text in section {
                for tok in text.tokens() {
                    if seen.insert(tok.clone()) {
                        out.push(tok.clone());
                    }
                }
            }
        }
        out
    }
}
