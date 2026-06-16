//! Shared tokenizer. Lower-cases input, keeps each punctuation character as
//! its own token, glues apostrophes/digits into the surrounding word (so
//! "don't" and "it's" stay whole). Structural specials — `<PAD>`, `<UNK>`,
//! `<SEC>`, and any `<PERSON_N>` / `</PERSON_N>` tag — pass through verbatim
//! because the corpus parser and the model use them as control markers.
//!
//! Legacy `<USER>`, `</USER>`, `<BOT>`, `</BOT>` constants remain only so the
//! pre-PERSON_N corpus migration code compiles. They are NOT used by the new
//! corpus and will be removed once any lingering callers are gone.

use crate::persons::{parse_close_tag, parse_open_tag};

pub const PAD: &str = "<PAD>";
pub const UNK: &str = "<UNK>";
pub const SEC: &str = "<SEC>";
/// Placeholder substituted for URL whitespace-tokens by the ingest
/// sanitizer (`convert_discord::sanitize_for_corpus`) and by the
/// corpus cleaner (`clean_corpus` rule 9). Without a passthrough
/// rule the tokenizer would split it on the underscores into
/// `_ _ url _ _`, defeating the purpose; we want one stable token.
pub const URL_PLACEHOLDER: &str = "__URL__";
/// Placeholder for parenthesized Discord-style references — user
/// mentions `(@<id>)`, role mentions `(@&<id>)`, channel refs
/// `(#<id>)`, relative timestamps `(t:<unix>:R)`, and the `(id:browse)`
/// link marker. The `<>` form was already collapsed to `()` by
/// `convert_discord::sanitize_for_corpus`; cleaner rule 10 then
/// replaces the surviving `(...)` blobs with this placeholder so
/// the tokenizer doesn't fragment them and the model doesn't learn
/// to emit them in chat.
pub const MENTION_PLACEHOLDER: &str = "__MENTION__";
/// Placeholder for Discord custom-emoji shortcodes (`:name:`,
/// `:PanFrown:`, `:yaycat:`, `:kek:`, etc.). Without a passthrough
/// rule the tokenizer splits these into `:`, `name`, `:` — pure
/// vocab noise (~50 distinct shortcodes appear in our corpus).
/// Cleaner rule 11 collapses them to this token.
pub const EMOJI_PLACEHOLDER: &str = "__EMOJI__";

// Legacy markers — kept only to let old code compile during the redesign.
pub const END_OF_BOT: &str = "</BOT>";
pub const USER_OPEN: &str = "<USER>";
pub const USER_CLOSE: &str = "</USER>";
pub const BOT_OPEN: &str = "<BOT>";

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for chunk in text.split_whitespace() {
        if is_special_passthrough(chunk) {
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

/// Tokens that must pass through `tokenize` unmodified.
pub fn is_special_passthrough(chunk: &str) -> bool {
    matches!(
        chunk,
        PAD | UNK
            | SEC
            | URL_PLACEHOLDER
            | MENTION_PLACEHOLDER
            | EMOJI_PLACEHOLDER
            | END_OF_BOT
            | USER_OPEN
            | USER_CLOSE
            | BOT_OPEN
    ) || parse_open_tag(chunk).is_some()
        || parse_close_tag(chunk).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn person_tags_pass_through() {
        let toks = tokenize("<PERSON_3> hello there </PERSON_3>");
        assert_eq!(
            toks,
            vec!["<PERSON_3>", "hello", "there", "</PERSON_3>"]
        );
    }

    #[test]
    fn legacy_tags_still_passthrough() {
        let toks = tokenize("<USER> hi </USER> <BOT> hi back </BOT>");
        assert!(toks.contains(&"<USER>".to_string()));
        assert!(toks.contains(&"</BOT>".to_string()));
    }

    #[test]
    fn apostrophe_glued() {
        let toks = tokenize("don't worry it's fine");
        assert_eq!(toks, vec!["don't", "worry", "it's", "fine"]);
    }

    #[test]
    fn punctuation_split() {
        let toks = tokenize("hello, world!");
        assert_eq!(toks, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn url_placeholder_passes_through() {
        // Without the passthrough rule this would tokenize to
        // ["_", "_", "url", "_", "_"] because '_' is non-alphanumeric.
        let toks = tokenize("hey __URL__ lol");
        assert_eq!(toks, vec!["hey", "__URL__", "lol"]);
    }

    #[test]
    fn mention_placeholder_passes_through() {
        let toks = tokenize("hi __MENTION__ how are you");
        assert_eq!(toks, vec!["hi", "__MENTION__", "how", "are", "you"]);
    }

    #[test]
    fn emoji_placeholder_passes_through() {
        let toks = tokenize("lmao __EMOJI__ that's hilarious");
        assert_eq!(
            toks,
            vec!["lmao", "__EMOJI__", "that's", "hilarious"]
        );
    }
}
