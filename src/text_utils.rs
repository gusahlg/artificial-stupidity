//! Small text utilities shared between the corpus cleaner and the
//! Discord ingest binary. Kept here so the URL-detection rules stay
//! in sync between the writer side (`convert_discord` sanitization)
//! and the filter side (`clean_corpus` rule 4 / dedup). When the two
//! drift, you get URLs that one stage replaces with `__URL__` and the
//! other still treats as raw — defensive-in-depth, but it confuses
//! the per-rule drop counts.

/// True if `word` (one whitespace-token) is a Discord custom-emoji
/// shortcode of the form `:name:` where `name` is 2–32 chars of
/// `[A-Za-z0-9_]`. Catches both unicode-emoji shortcodes
/// (`:thumbsup:`) and server-custom-emoji (`:PanFrown:`, `:yaycat:`,
/// etc.). Used by `clean_corpus` rule 11 to collapse all of them to
/// `__EMOJI__` so the tokenizer doesn't shatter them on the colons.
pub fn is_emoji_shortcode(word: &str) -> bool {
    let inner = match word.strip_prefix(':').and_then(|s| s.strip_suffix(':')) {
        Some(s) => s,
        None => return false,
    };
    let len = inner.chars().count();
    if !(2..=32).contains(&len) {
        return false;
    }
    inner
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// True if `word` (one whitespace-token) is a parenthesized Discord
/// reference that survived `convert_discord::sanitize_for_corpus`
/// (which collapses `<>` → `()` so user content can't impersonate a
/// PERSON tag). Covers:
///   - `(@<digits>)`      — user mention
///   - `(@&<digits>)`     — role mention
///   - `(#<digits>)`      — channel mention
///   - `(t:<digits>:R)`   — relative-time tag
///   - `(id:browse)`      — Discord inline-browse anchor
/// These have no natural-language value and bloat the vocab with
/// per-id rare tokens. Cleaner rule 10 replaces them with one stable
/// `__MENTION__` placeholder.
pub fn is_paren_mention(word: &str) -> bool {
    let inner = match word.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        Some(s) if !s.is_empty() => s,
        _ => return false,
    };
    if inner == "id:browse" {
        return true;
    }
    // (@<digits>) or (@&<digits>) or (#<digits>)
    let digit_after = if let Some(rest) = inner.strip_prefix("@&") {
        rest
    } else if let Some(rest) = inner.strip_prefix('@') {
        rest
    } else if let Some(rest) = inner.strip_prefix('#') {
        rest
    } else if let Some(rest) = inner.strip_prefix("t:") {
        // (t:<digits>:R) — relative timestamp
        return rest
            .strip_suffix(":R")
            .map(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false);
    } else {
        return false;
    };
    !digit_after.is_empty() && digit_after.chars().all(|c| c.is_ascii_digit())
}

/// True if `word` (one whitespace-token) looks like the start of a URL
/// we want to treat as a single placeholder during ingest, or strip
/// outright during cleanup.
///
/// Covers the ~99% of URLs we actually see in the Discord corpus:
/// explicit `http(s)://`, plus the bare hostnames Discord renders as
/// links (`tenor.com/…`, `discord.gg/…`, `discord.com/…`,
/// `cdn.discordapp.com/…`, `youtu.be/…`, `youtube.com/…`, `www.…`).
/// Leading punctuation (`(`, `[`, `<`) from copy-paste is stripped
/// before the prefix check.
pub fn looks_like_url(word: &str) -> bool {
    let w = word.trim_start_matches(|c: char| c == '(' || c == '[' || c == '<');
    w.starts_with("http://")
        || w.starts_with("https://")
        || w.starts_with("www.")
        || w.starts_with("tenor.com/")
        || w.starts_with("discord.gg/")
        || w.starts_with("discord.com/")
        || w.starts_with("cdn.discordapp.com/")
        || w.starts_with("youtube.com/")
        || w.starts_with("youtu.be/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_common_schemes() {
        assert!(looks_like_url("https://example.com/foo"));
        assert!(looks_like_url("http://example.com"));
        assert!(looks_like_url("www.example.com"));
    }

    #[test]
    fn matches_bare_discord_hosts() {
        assert!(looks_like_url("tenor.com/view/foo-gif-12345"));
        assert!(looks_like_url("discord.gg/abc123"));
        assert!(looks_like_url("cdn.discordapp.com/attachments/1/2/x.png"));
    }

    #[test]
    fn ignores_non_urls() {
        assert!(!looks_like_url("hello"));
        assert!(!looks_like_url("tenor.com"));        // no trailing slash → not a link
        assert!(!looks_like_url("@everyone"));
        assert!(!looks_like_url(""));
    }

    #[test]
    fn strips_leading_punctuation() {
        assert!(looks_like_url("(https://example.com)"));
        assert!(looks_like_url("<https://example.com>"));
        assert!(looks_like_url("[tenor.com/view/foo]"));
    }

    #[test]
    fn paren_mention_matches_known_forms() {
        assert!(is_paren_mention("(@1234567890)"));
        assert!(is_paren_mention("(@&1234567890)"));
        assert!(is_paren_mention("(#1234567890)"));
        assert!(is_paren_mention("(t:1778085052:R)"));
        assert!(is_paren_mention("(id:browse)"));
    }

    #[test]
    fn paren_mention_rejects_normal_parens() {
        assert!(!is_paren_mention("(hello)"));
        assert!(!is_paren_mention("(@notdigits)"));
        assert!(!is_paren_mention("(123)"));
        assert!(!is_paren_mention("()"));
        assert!(!is_paren_mention("@1234"));
        assert!(!is_paren_mention(""));
    }

    #[test]
    fn emoji_shortcode_matches_typical_forms() {
        assert!(is_emoji_shortcode(":thumbsup:"));
        assert!(is_emoji_shortcode(":PanFrown:"));
        assert!(is_emoji_shortcode(":yaycat:"));
        assert!(is_emoji_shortcode(":CanWePinThisGuy:"));
        assert!(is_emoji_shortcode(":kek:"));
        assert!(is_emoji_shortcode(":a_b_c:"));
    }

    #[test]
    fn emoji_shortcode_rejects_non_shortcodes() {
        assert!(!is_emoji_shortcode(":"));        // empty inner
        assert!(!is_emoji_shortcode("::"));       // empty inner
        assert!(!is_emoji_shortcode(":a:"));      // too short
        assert!(!is_emoji_shortcode(":1234567890123456789012345678901234:")); // too long
        assert!(!is_emoji_shortcode(":has space:")); // disallowed char
        assert!(!is_emoji_shortcode(":hello-world:")); // hyphen not allowed
        assert!(!is_emoji_shortcode("noprefix:"));
        assert!(!is_emoji_shortcode(":nosuffix"));
        assert!(!is_emoji_shortcode("hello"));
    }
}
