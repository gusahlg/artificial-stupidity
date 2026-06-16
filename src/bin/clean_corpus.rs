//! Filter and re-shuffle `data/dialogs.txt` so the training pipeline isn't
//! poisoned by the three concrete problems we measured:
//!
//!  1. TinyStories-style monologue-only sections (every turn is PERSON_0)
//!     dominating the corpus tail, which is where `train --val-frac 0.1`
//!     samples from. Train and val ended up on different distributions,
//!     and the apparent "overfitting at epoch 5" was actually distribution
//!     drift, not capacity exhaustion.
//!  2. URL-only / emoji-only / role-ping-only turns that contribute pure
//!     noise after tokenization.
//!  3. A handful of specific strings (Lily/Timmy story openings, Patreon
//!     plug, "once upon a time…") repeated ≥10× each, totalling ~6% of
//!     the entire corpus — a heavy gradient bias toward almost-no signal.
//!
//! The cleaner is idempotent: running it twice on the same input leaves
//! the corpus byte-identical. The fixed-seed section shuffle (Rule 8)
//! makes the val tail a random sample of the corpus rather than the
//! topical tail.
//!
//! Usage:
//!   cargo run --release --bin clean_corpus [INPUT] [OUTPUT]
//!
//! Defaults: INPUT = OUTPUT = data/dialogs.txt. Writes via a `.tmp`
//! sibling + atomic rename, so a crash mid-write doesn't truncate the
//! corpus. Back up the corpus separately before running anyway.

use anyhow::{Context, Result, anyhow, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rust_fun::persons::{BOT_PERSON_ID, close_tag, open_tag, parse_close_tag, parse_open_tag};
use rust_fun::text_utils::{is_emoji_shortcode, is_paren_mention, looks_like_url};
use rust_fun::tokenizer::{EMOJI_PLACEHOLDER, MENTION_PLACEHOLDER, URL_PLACEHOLDER};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const DEFAULT_PATH: &str = "data/dialogs.txt";
const SEC_TAG: &str = "<SEC>";
/// Cap any single turn-text from appearing more than this many times.
const MAX_REPEATS: u32 = 3;
/// A turn whose non-whitespace content drops below this many chars after
/// URL stripping is considered URL-dominant.
const MIN_NON_URL_CHARS: usize = 8;
/// Rule 14: cap BOT turns sharing the same first-N-token prefix at this
/// many occurrences. Targets OASST helper-template domination on the
/// bot side ("here are some …", "there are several …", "sure, here are
/// …", "sure, i'd be …") which the per-whole-turn rule 7 doesn't catch
/// because the suffixes vary. Bot-only: USER prompts have legitimate
/// template diversity ("what is the …", "how do i …") we want to keep.
/// Retuned 2026-06-09 after analyzing bot-side prefix distribution on
/// L's corpus: PREFIX=4 MAX=15 only fired 37 times (0.7%); PREFIX=3
/// captures the helper templates because the 4th token already varies
/// ("here are some [tips/ways/things/ideas/...]"). At PREFIX=3 MAX=10
/// the cap drops ~450 bot turns (~8.6%), all OASST helper-template
/// openers — exactly the register we're trying to shake off.
const MAX_PREFIX_REPEATS: u32 = 10;
/// Number of leading tokens that define a prefix-key for rule 14.
const PREFIX_LEN: usize = 3;
/// Section shuffle seed. Fixed so cleaner output is deterministic — the
/// trainer still shuffles within an epoch, so this only affects which
/// sections land in the val tail.
const SHUFFLE_SEED: u64 = 0;

#[derive(Clone)]
struct RawTurn {
    person_id: u32,
    /// Whitespace-normalized raw text between `<PERSON_N>` and `</PERSON_N>`.
    /// Token-level analysis (URL detection, role-ping detection) works on
    /// this; the in-process tokenizer is only used for downstream training
    /// once the cleaner has run.
    text: String,
}

#[derive(Default)]
struct DropCounts {
    sec_monologue: u32,    // Rule 1
    sec_single_speaker: u32, // Rule 2
    sec_too_short: u32,    // Rule 3 (post-turn-filter)
    turn_url_dominant: u32, // Rule 4
    turn_no_alphanum: u32, // Rule 5
    turn_role_ping: u32,   // Rule 6
    turn_dedup_cap: u32,   // Rule 7
    /// Rule 9: per-turn URL tokens rewritten to __URL__ (not a drop,
    /// just a count of turns that were touched).
    turn_urls_rewritten: u32,
    /// Rule 9: total individual URL whitespace-tokens replaced.
    url_tokens_rewritten: u32,
    /// Rule 10: per-turn mention rewrites (touched turns count).
    turn_mentions_rewritten: u32,
    /// Rule 10: total individual `(@123)`-style tokens replaced.
    mention_tokens_rewritten: u32,
    /// Rule 11: per-turn emoji-shortcode rewrites (touched turns count).
    turn_emojis_rewritten: u32,
    /// Rule 11: total `:name:`-style tokens replaced.
    emoji_tokens_rewritten: u32,
    /// Rule 12: per-turn markdown strips (touched turns count).
    turn_markdown_stripped: u32,
    /// Rule 12: total individual markdown decorations removed
    /// (bold/italic `*`, bullet `-` runs, leading `:` decorations).
    markdown_tokens_stripped: u32,
    /// Rule 13: per-turn escape-character strips.
    turn_escapes_stripped: u32,
    /// Rule 13: total backslashes and escape sequences removed.
    escape_chars_stripped: u32,
    /// Rule 14: turns dropped for exceeding the shared-prefix cap.
    turn_prefix_cap: u32,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PATH));
    let output: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| input.clone());

    let raw = fs::read_to_string(&input)
        .with_context(|| format!("read {:?}", input))?;
    let mut sections = parse_raw(&raw)?;
    let input_sections = sections.len();
    let input_turns: usize = sections.iter().map(|s| s.len()).sum();

    let mut counts = DropCounts::default();

    // Section-level rules 1 and 2: identify monologue / single-speaker
    // sections up-front. Doing this before turn-level filtering means
    // we don't waste cycles examining individual turns of sections we
    // already know we're going to drop.
    sections.retain(|sec| {
        if sec.iter().all(|t| t.person_id == BOT_PERSON_ID) {
            counts.sec_monologue += 1;
            return false;
        }
        let mut ids = sec.iter().map(|t| t.person_id);
        let first = ids.next();
        if let Some(first) = first {
            if ids.all(|id| id == first) {
                counts.sec_single_speaker += 1;
                return false;
            }
        }
        true
    });

    // Turn-level rules 4, 5, 6.
    for sec in sections.iter_mut() {
        sec.retain(|turn| {
            if is_url_dominant(&turn.text) {
                counts.turn_url_dominant += 1;
                return false;
            }
            if !turn.text.chars().any(|c| c.is_alphanumeric()) {
                counts.turn_no_alphanum += 1;
                return false;
            }
            if is_role_ping_only(&turn.text) {
                counts.turn_role_ping += 1;
                return false;
            }
            true
        });
    }

    // Cross-corpus rule 7: dedup repeated turns to MAX_REPEATS occurrences.
    // Canonical key is the lowercased trimmed text — case differences in
    // copy-pasted spam shouldn't bypass the cap.
    let mut seen_counts: HashMap<String, u32> = HashMap::new();
    for sec in sections.iter_mut() {
        sec.retain(|turn| {
            let key = turn.text.trim().to_lowercase();
            let entry = seen_counts.entry(key).or_insert(0);
            if *entry >= MAX_REPEATS {
                counts.turn_dedup_cap += 1;
                return false;
            }
            *entry += 1;
            true
        });
    }

    // Section-level rule 3, re-applied: a section that lost too many turns
    // to the rules above is no longer useful (the trainer's
    // `extract_train_examples` already skips <2-turn sections, but
    // dropping them on disk also tightens the val partition).
    sections.retain(|sec| {
        if sec.len() < 2 {
            counts.sec_too_short += 1;
            return false;
        }
        true
    });

    // Rule 9: replace URL whitespace-tokens embedded in surviving turns
    // with the placeholder `__URL__`. Rule 4 only dropped URL-DOMINANT
    // turns; turns like "check this https://tenor.com/foo lol" passed
    // and would otherwise still leak ~20 punctuation-split URL fragments
    // ("com/", "view/", "tenor-", ...) into the vocab and training
    // gradient. Doing the rewrite here, after dedup, means dedup keys
    // are still based on raw text (so identical-URL spam still gets
    // dedup'd, while different-URL-same-prefix turns don't collapse).
    // `convert_discord::sanitize_for_corpus` already does this rewrite
    // at ingest time; this rule is for the historical corpus that was
    // ingested before that patch existed.
    for sec in sections.iter_mut() {
        for turn in sec.iter_mut() {
            let (rewritten, n) = rewrite_urls(&turn.text);
            if n > 0 {
                counts.turn_urls_rewritten += 1;
                counts.url_tokens_rewritten += n;
                turn.text = rewritten;
            }
        }
    }

    // Rule 10: replace parenthesized Discord references — `(@<id>)`,
    // `(@&<id>)`, `(#<id>)`, `(t:<unix>:R)`, `(id:browse)` — with
    // `__MENTION__`. These are leftovers from `convert_discord`'s
    // `<>` → `()` sanitization step; if left raw, the tokenizer
    // shatters them into useless single-char tokens and the model
    // learns to emit them in chat.
    for sec in sections.iter_mut() {
        for turn in sec.iter_mut() {
            let (rewritten, n) = rewrite_mentions(&turn.text);
            if n > 0 {
                counts.turn_mentions_rewritten += 1;
                counts.mention_tokens_rewritten += n;
                turn.text = rewritten;
            }
        }
    }

    // Rule 11: replace Discord custom-emoji shortcodes (`:name:`) with
    // `__EMOJI__`. Without this, the tokenizer splits each shortcode
    // into three tokens (`:`, `name`, `:`); ~50 distinct shortcodes
    // appear in the corpus, all contributing nothing semantically but
    // bloating the vocab head.
    for sec in sections.iter_mut() {
        for turn in sec.iter_mut() {
            let (rewritten, n) = rewrite_emoji_shortcodes(&turn.text);
            if n > 0 {
                counts.turn_emojis_rewritten += 1;
                counts.emoji_tokens_rewritten += n;
                turn.text = rewritten;
            }
        }
    }

    // Rule 12: strip OASST-style Markdown decorations that leak into
    // the corpus. The manual-test pass on 2026-05-22 found the bot
    // emitting `*****` runs, leading `:- ` bullet markers, and
    // numbered-list `1. ` openers — all artifacts of OASST replies
    // that survive the punctuation-splitting tokenizer as bare
    // single-char tokens. We strip them at the cleaner level (rather
    // than at ingest) so the same rule applies retroactively to
    // anything already in dialogs.txt.
    //
    // What this rule removes:
    //   - Bold (`**text**`)  → keeps `text`
    //   - Italic (`*text*`) → keeps `text`
    //   - Runs of `*` or `_` 2+ chars → collapsed to nothing
    //   - Leading bullet markers (`- `, `* `, `: ` at start of a
    //     space-delimited word at turn-start) → dropped
    //   - Em-dashes (`---`, `--`, `--`) → single `-` (one hyphen is
    //     still legitimate punctuation in dialog)
    //   - Numbered-list openers ("1.", "2.") at turn-start → dropped
    for sec in sections.iter_mut() {
        for turn in sec.iter_mut() {
            let (rewritten, n) = strip_markdown(&turn.text);
            if n > 0 {
                counts.turn_markdown_stripped += 1;
                counts.markdown_tokens_stripped += n;
                turn.text = rewritten;
            }
        }
    }

    // Rule 13: strip backslash-escape artifacts that leak in from
    // OASST source text. The 2026-05-22 manual test caught the bot
    // emitting a literal `\` as a one-token reply. Inspection of the
    // corpus showed ~132 standalone backslashes (file paths, regex
    // examples, escape sequences). They have no natural-language
    // value and contaminate the vocab head. We strip them entirely;
    // a turn that becomes empty after stripping gets dropped by the
    // surviving-too-short check below.
    for sec in sections.iter_mut() {
        for turn in sec.iter_mut() {
            let (rewritten, n) = strip_escapes(&turn.text);
            if n > 0 {
                counts.turn_escapes_stripped += 1;
                counts.escape_chars_stripped += n;
                turn.text = rewritten;
            }
        }
    }

    // Re-apply the empty-turn / single-speaker-section / short-section
    // filters since rules 9-13 may have left empty turns or wiped
    // out enough of a section to make it sub-2-turns.
    for sec in sections.iter_mut() {
        sec.retain(|turn| !turn.text.trim().is_empty());
    }
    sections.retain(|sec| {
        if sec.len() < 2 {
            counts.sec_too_short += 1;
            return false;
        }
        true
    });

    // Rule 14: cap BOT turns sharing the same first-PREFIX_LEN-token
    // prefix at MAX_PREFIX_REPEATS occurrences. OASST contributes
    // hundreds of distinct bot turns whose first 4 tokens collapse to a
    // small set ("sure, here are some", "here is a list", "there are
    // several ways", "sure, i'd be happy") — rule 7 doesn't dedup them
    // because the suffixes vary, but they dominate the helper-template
    // distribution. Bot-only because USER prompts have legitimate
    // template diversity ("what is the …", "how do i …") we want to
    // preserve. We compute the key on FINAL turn text (after rules
    // 9-13). Short turns (< PREFIX_LEN tokens) are exempt — already
    // capped by rule 7.
    let mut prefix_counts: HashMap<String, u32> = HashMap::new();
    for sec in sections.iter_mut() {
        sec.retain(|turn| {
            if turn.person_id != BOT_PERSON_ID {
                return true;
            }
            let toks: Vec<&str> = turn.text.split_whitespace().collect();
            if toks.len() < PREFIX_LEN {
                return true;
            }
            let key = toks[..PREFIX_LEN]
                .iter()
                .map(|t| t.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            let entry = prefix_counts.entry(key).or_insert(0);
            if *entry >= MAX_PREFIX_REPEATS {
                counts.turn_prefix_cap += 1;
                return false;
            }
            *entry += 1;
            true
        });
    }
    // Re-apply short-section filter after rule 14.
    sections.retain(|sec| {
        if sec.len() < 2 {
            counts.sec_too_short += 1;
            return false;
        }
        true
    });

    // Rule 8: deterministic section shuffle. The trainer reshuffles per
    // epoch, so this only affects the val-tail split (`train.rs:166`,
    // `examples.split_off(total - val_n)`), which is what we wanted to
    // de-bias.
    let mut rng = StdRng::seed_from_u64(SHUFFLE_SEED);
    sections.shuffle(&mut rng);

    write_corpus(&output, &sections)?;

    let output_turns: usize = sections.iter().map(|s| s.len()).sum();
    eprintln!("clean_corpus: read {:?}", input);
    eprintln!(
        "  sections: {} → {} (Δ -{})",
        input_sections,
        sections.len(),
        input_sections.saturating_sub(sections.len()),
    );
    eprintln!(
        "  turns:    {} → {} (Δ -{})",
        input_turns,
        output_turns,
        input_turns.saturating_sub(output_turns),
    );
    eprintln!("  rule 1 (monologue section):       -{}", counts.sec_monologue);
    eprintln!("  rule 2 (single-speaker section):  -{}", counts.sec_single_speaker);
    eprintln!("  rule 3 (section <2 turns after):  -{}", counts.sec_too_short);
    eprintln!("  rule 4 (URL-dominant turn):       -{}", counts.turn_url_dominant);
    eprintln!("  rule 5 (no alphanumeric content): -{}", counts.turn_no_alphanum);
    eprintln!("  rule 6 (role-ping only turn):     -{}", counts.turn_role_ping);
    eprintln!("  rule 7 (dedup cap = {}):           -{}", MAX_REPEATS, counts.turn_dedup_cap);
    eprintln!(
        "  rule 9 (inline URL → __URL__):    {} turns touched, {} URL tokens replaced",
        counts.turn_urls_rewritten, counts.url_tokens_rewritten,
    );
    eprintln!(
        "  rule 10 (mention → __MENTION__):  {} turns touched, {} mention tokens replaced",
        counts.turn_mentions_rewritten, counts.mention_tokens_rewritten,
    );
    eprintln!(
        "  rule 11 (emoji → __EMOJI__):      {} turns touched, {} emoji tokens replaced",
        counts.turn_emojis_rewritten, counts.emoji_tokens_rewritten,
    );
    eprintln!(
        "  rule 12 (markdown strip):         {} turns touched, {} markdown tokens removed",
        counts.turn_markdown_stripped, counts.markdown_tokens_stripped,
    );
    eprintln!(
        "  rule 13 (escape strip):           {} turns touched, {} backslashes removed",
        counts.turn_escapes_stripped, counts.escape_chars_stripped,
    );
    eprintln!(
        "  rule 14 (prefix cap = {} per {}-token prefix): -{}",
        MAX_PREFIX_REPEATS, PREFIX_LEN, counts.turn_prefix_cap,
    );
    eprintln!("clean_corpus: wrote {:?} (seed={})", output, SHUFFLE_SEED);

    Ok(())
}

/// Parse the corpus into raw turn-text retained verbatim (post
/// whitespace-normalization). We can't reuse `dialogs.rs::parse_corpus`
/// here because that one tokenizes during parse; we need the raw text
/// for URL detection, role-ping detection, and dedup-canonicalization
/// before any token splitting happens.
fn parse_raw(raw: &str) -> Result<Vec<Vec<RawTurn>>> {
    let mut sections: Vec<Vec<RawTurn>> = Vec::new();
    let mut current: Vec<RawTurn> = Vec::new();
    let mut in_turn: Option<u32> = None;
    let mut buf = String::new();

    for (idx, chunk) in raw.split_whitespace().enumerate() {
        if chunk == SEC_TAG {
            if let Some(id) = in_turn.take() {
                bail!("token {}: <SEC> while still inside <PERSON_{}>", idx, id);
            }
            if !current.is_empty() {
                sections.push(std::mem::take(&mut current));
            }
            continue;
        }
        if let Some(open_id) = parse_open_tag(chunk) {
            if let Some(prev) = in_turn {
                bail!(
                    "token {}: nested <PERSON_{}> inside still-open <PERSON_{}>",
                    idx,
                    open_id,
                    prev
                );
            }
            in_turn = Some(open_id);
            buf.clear();
            continue;
        }
        if let Some(close_id) = parse_close_tag(chunk) {
            let open_id = in_turn
                .take()
                .ok_or_else(|| anyhow!("token {}: </PERSON_{}> with no open", idx, close_id))?;
            if open_id != close_id {
                bail!(
                    "token {}: opened <PERSON_{}> but closed </PERSON_{}>",
                    idx,
                    open_id,
                    close_id
                );
            }
            current.push(RawTurn {
                person_id: open_id,
                text: buf.trim().to_string(),
            });
            buf.clear();
            continue;
        }
        if in_turn.is_some() {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(chunk);
        }
    }
    if let Some(id) = in_turn {
        bail!("EOF inside <PERSON_{}>", id);
    }
    if !current.is_empty() {
        sections.push(current);
    }
    Ok(sections)
}

/// Replace URL-looking whitespace-tokens with the literal `__URL__`
/// placeholder, returning the rewritten text and the number of tokens
/// replaced. Used by rule 9 to scrub URL fragments out of otherwise-
/// substantive turns so the punctuation-splitting tokenizer doesn't
/// blow them into "com/", "view/", "tenor-" noise.
fn rewrite_urls(text: &str) -> (String, u32) {
    let mut out = String::with_capacity(text.len());
    let mut count = 0u32;
    let mut first = true;
    for word in text.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        if looks_like_url(word) {
            out.push_str(URL_PLACEHOLDER);
            count += 1;
        } else {
            out.push_str(word);
        }
    }
    (out, count)
}

/// Replace parenthesized-Discord-mention whitespace-tokens with the
/// `__MENTION__` placeholder. Used by rule 10.
fn rewrite_mentions(text: &str) -> (String, u32) {
    let mut out = String::with_capacity(text.len());
    let mut count = 0u32;
    let mut first = true;
    for word in text.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        if is_paren_mention(word) {
            out.push_str(MENTION_PLACEHOLDER);
            count += 1;
        } else {
            out.push_str(word);
        }
    }
    (out, count)
}

/// Strip OASST-style Markdown decorations from a single turn. Returns
/// the cleaned text plus a count of individual decorations removed
/// (so the cleaner can summarize how aggressively this rule fired).
///
/// Performs (in order):
///   1. Collapse runs of `*` or `_` (length ≥ 2) to a single space.
///      This catches both `**bold**` → ` bold ` and stray `*****`
///      runs that leaked into the corpus from OASST formatting.
///   2. Replace standalone single `*` (surrounded by whitespace) with
///      a single space. This catches `* item` bullet markers.
///   3. Collapse runs of `-` (length ≥ 2) to a single `-`. Em-dashes
///      `--` and triple `---` were tokenizing to multiple `-` chars.
///   4. Drop turn-leading `:- `, `: `, `*- ` decorations.
///   5. Drop turn-leading numbered-list openers (`1. `, `2. `, ...).
///   6. Collapse the resulting multi-space runs back to single space
///      and trim.
fn strip_markdown(text: &str) -> (String, u32) {
    let mut count: u32 = 0;
    let mut s = text.to_string();

    // 1. Runs of '*' length ≥ 2 → single space, counted as one
    //    decoration each.
    while let Some(start) = s.find("**") {
        let end = s[start..]
            .find(|c: char| c != '*')
            .map(|i| start + i)
            .unwrap_or(s.len());
        s.replace_range(start..end, " ");
        count += 1;
    }
    // 2. Runs of '_' length ≥ 2 → single space.
    while let Some(start) = s.find("__") {
        // BUT: respect our own placeholder tokens (`__URL__`,
        // `__MENTION__`, `__EMOJI__`). If the underscore run is
        // part of one of those, skip it.
        let after_run_start = s[start..]
            .find(|c: char| c != '_')
            .map(|i| start + i)
            .unwrap_or(s.len());
        // Check if this looks like a placeholder by checking the
        // surrounding tokens. Cheap heuristic: if the rest of the
        // word after the underscores is all-caps then `__`, leave it.
        let after_word_end = s[after_run_start..]
            .find(|c: char| c.is_whitespace())
            .map(|i| after_run_start + i)
            .unwrap_or(s.len());
        let token = &s[start..after_word_end];
        if token == "__URL__"
            || token == "__MENTION__"
            || token == "__EMOJI__"
        {
            // Skip ahead past this token to avoid an infinite loop
            // on the next while-iteration finding the same `__`.
            // We do this by replacing it with itself + sentinel,
            // but simpler: just truncate-search forward from
            // after_word_end. We'll use a manual scan loop instead.
            break;
        }
        s.replace_range(start..after_run_start, " ");
        count += 1;
    }
    // 2b. Re-do the '_' scan with placeholder-skip, since the break
    //     above bails on the first placeholder encountered. We need
    //     to keep scanning past it.
    {
        let mut i = 0;
        while let Some(rel) = s[i..].find("__") {
            let start = i + rel;
            let after_run_start = s[start..]
                .find(|c: char| c != '_')
                .map(|j| start + j)
                .unwrap_or(s.len());
            let after_word_end = s[after_run_start..]
                .find(|c: char| c.is_whitespace())
                .map(|j| after_run_start + j)
                .unwrap_or(s.len());
            let token = &s[start..after_word_end];
            if token == "__URL__" || token == "__MENTION__" || token == "__EMOJI__"
            {
                i = after_word_end;
                continue;
            }
            // Otherwise strip the underscore run.
            s.replace_range(start..after_run_start, " ");
            count += 1;
            // i stays at start; next find continues from there. The
            // replacement is a single space, so we won't re-match
            // `__` at the same position.
            i = start + 1;
        }
    }
    // 3. Runs of '-' length ≥ 2 → single '-'. Em-dashes become a
    //    plain hyphen.
    while let Some(start) = s.find("--") {
        let end = s[start..]
            .find(|c: char| c != '-')
            .map(|i| start + i)
            .unwrap_or(s.len());
        s.replace_range(start..end, "-");
        count += 1;
    }
    // 4. Standalone single `*` surrounded by whitespace → space.
    //    Walk word-by-word so we don't false-match inside a token.
    {
        let mut buf = String::with_capacity(s.len());
        let mut first = true;
        for word in s.split_whitespace() {
            if word == "*" {
                count += 1;
                continue;
            }
            if !first {
                buf.push(' ');
            }
            first = false;
            buf.push_str(word);
        }
        s = buf;
    }
    // 5. Drop turn-leading bullet/list markers.
    let trimmed = s.trim_start();
    if trimmed.starts_with(":- ") {
        s = trimmed[3..].to_string();
        count += 1;
    } else if trimmed.starts_with("- ") {
        s = trimmed[2..].to_string();
        count += 1;
    } else if trimmed.starts_with(": ") {
        s = trimmed[2..].to_string();
        count += 1;
    }
    // 6. Drop turn-leading numbered-list openers like "1. ", "12. ".
    let trimmed = s.trim_start();
    let leading_digits_end = trimmed
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit())
        .last()
        .map(|(i, _)| i + 1)
        .unwrap_or(0);
    if leading_digits_end > 0 && leading_digits_end < trimmed.len() {
        let rest = &trimmed[leading_digits_end..];
        if rest.starts_with(". ") {
            s = rest[2..].to_string();
            count += 1;
        }
    }
    // 7. Collapse the multi-spaces produced by the strips above.
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    (s, count)
}

/// Strip backslash-escape artifacts from a turn. Returns the cleaned
/// text plus a count of individual backslashes removed.
///
/// Behavior:
///   - `\n`, `\r`, `\t` → single space (they're whitespace escapes
///     from JSON-encoded OASST source text)
///   - `\"` → `"` (drop the escape)
///   - `\\` → drop both characters (the escaped backslash itself is
///     also meaningless in dialogue)
///   - any other `\<x>` → drop the `\`, keep `<x>` (defensive — the
///     character after a backslash was probably the intended content)
///   - standalone trailing `\` → drop
fn strip_escapes(text: &str) -> (String, u32) {
    let mut out = String::with_capacity(text.len());
    let mut count: u32 = 0;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        count += 1;
        match chars.peek().copied() {
            Some('n') | Some('r') | Some('t') => {
                // Escape-as-whitespace.
                out.push(' ');
                chars.next();
            }
            Some('"') => {
                out.push('"');
                chars.next();
            }
            Some('\\') => {
                // `\\` → drop both (an extra count for the second `\`).
                count += 1;
                chars.next();
            }
            Some(other) => {
                // Drop the `\`, keep the next char as-is. Don't count
                // the next char as a stripped escape.
                out.push(other);
                chars.next();
            }
            None => {
                // Trailing standalone backslash; drop it.
            }
        }
    }
    // Collapse runs of whitespace introduced by escape→space.
    let normalized = out
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (normalized, count)
}

/// Replace Discord custom-emoji shortcodes (`:name:`) with the
/// `__EMOJI__` placeholder. Used by rule 11.
fn rewrite_emoji_shortcodes(text: &str) -> (String, u32) {
    let mut out = String::with_capacity(text.len());
    let mut count = 0u32;
    let mut first = true;
    for word in text.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        if is_emoji_shortcode(word) {
            out.push_str(EMOJI_PLACEHOLDER);
            count += 1;
        } else {
            out.push_str(word);
        }
    }
    (out, count)
}

/// Drop URL-looking whitespace-tokens and report what's left, joined by
/// single spaces.
fn strip_urls(text: &str) -> String {
    text.split_whitespace()
        .filter(|w| !looks_like_url(w))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A turn is URL-dominant if removing URL-looking tokens leaves fewer
/// than `MIN_NON_URL_CHARS` non-whitespace characters of content. Catches
/// turns that are nothing but a tenor/discord link, possibly with a
/// trailing "lol" or emoji.
///
/// `looks_like_url` is shared with `convert_discord` via
/// `rust_fun::text_utils` so a turn that the ingest sanitizer replaced
/// with `__URL__` and a turn that arrived as a raw link are recognized
/// by exactly the same rule.
fn is_url_dominant(text: &str) -> bool {
    // Cheap early-out: no URL prefix anywhere → can't be URL-dominant.
    if !text.split_whitespace().any(looks_like_url) {
        return false;
    }
    let stripped = strip_urls(text);
    let remaining = stripped.chars().filter(|c| !c.is_whitespace()).count();
    remaining < MIN_NON_URL_CHARS
}

/// A whitespace-token that's `@everyone`, `@here`, or `@<all-digits>`
/// (a Discord user/role mention with the surrounding markup already
/// removed by `convert_discord::sanitize`).
fn is_role_ping_word(word: &str) -> bool {
    if word == "@everyone" || word == "@here" {
        return true;
    }
    if let Some(rest) = word.strip_prefix('@') {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    false
}

/// True if every whitespace-token in the turn is a role-ping. Catches
/// `"@everyone"`, `"@here please vote"` would fail (has "please vote"),
/// `"@123 @456"` succeeds.
fn is_role_ping_only(text: &str) -> bool {
    let mut any = false;
    for word in text.split_whitespace() {
        if !is_role_ping_word(word) {
            return false;
        }
        any = true;
    }
    any
}

fn write_corpus(out_path: &Path, sections: &[Vec<RawTurn>]) -> Result<()> {
    // Atomic write: stage to a `.tmp` sibling then rename. If we crash
    // mid-write the original is untouched (no truncation).
    let tmp = out_path.with_extension("txt.tmp");
    {
        let mut w = std::io::BufWriter::new(
            fs::File::create(&tmp)
                .with_context(|| format!("create {:?}", tmp))?,
        );
        for sec in sections {
            writeln!(w, "{}", SEC_TAG)?;
            for turn in sec {
                let open = open_tag(turn.person_id);
                let close = close_tag(turn.person_id);
                writeln!(w, "{} {} {}", open, turn.text, close)?;
            }
        }
        w.flush()?;
    }
    fs::rename(&tmp, out_path)
        .with_context(|| format!("rename {:?} -> {:?}", tmp, out_path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_dominant_drops_pure_link() {
        assert!(is_url_dominant("https://tenor.com/view/foo-gif-12345"));
        assert!(is_url_dominant("https://tenor.com/view/foo lol"));
        assert!(is_url_dominant("https://discord.gg/abc123"));
    }

    #[test]
    fn url_dominant_preserves_text_with_link() {
        // "check this cool gif" is well over MIN_NON_URL_CHARS after stripping.
        assert!(!is_url_dominant(
            "check this cool gif https://tenor.com/view/foo"
        ));
    }

    #[test]
    fn url_dominant_ignores_non_url_text() {
        assert!(!is_url_dominant("hello world this is a normal turn"));
    }

    #[test]
    fn role_ping_only_detects_everyone_and_here() {
        assert!(is_role_ping_only("@everyone"));
        assert!(is_role_ping_only("@here"));
        assert!(is_role_ping_only("@everyone @here"));
    }

    #[test]
    fn role_ping_only_detects_numeric_mentions() {
        assert!(is_role_ping_only("@123456789012345678"));
        assert!(is_role_ping_only("@123 @456"));
    }

    #[test]
    fn role_ping_only_rejects_mixed_content() {
        assert!(!is_role_ping_only("@everyone please vote"));
        assert!(!is_role_ping_only("hello @here"));
        assert!(!is_role_ping_only(""));
    }

    #[test]
    fn rewrite_urls_replaces_embedded_links() {
        let (out, n) = rewrite_urls("check this https://tenor.com/view/foo lol");
        assert_eq!(out, "check this __URL__ lol");
        assert_eq!(n, 1);
    }

    #[test]
    fn rewrite_urls_handles_multiple_links() {
        let (out, n) = rewrite_urls("https://a.com/x and https://b.com/y");
        assert_eq!(out, "__URL__ and __URL__");
        assert_eq!(n, 2);
    }

    #[test]
    fn rewrite_urls_no_op_on_clean_text() {
        let (out, n) = rewrite_urls("just plain conversation here");
        assert_eq!(out, "just plain conversation here");
        assert_eq!(n, 0);
    }

    #[test]
    fn rewrite_urls_is_idempotent() {
        let (first, _) = rewrite_urls("hey https://tenor.com/foo lol");
        let (second, n) = rewrite_urls(&first);
        assert_eq!(first, second);
        assert_eq!(n, 0);
    }

    #[test]
    fn rewrite_mentions_replaces_paren_forms() {
        let (out, n) = rewrite_mentions(
            "hi (@1234567890) and (@&999) check (#42) on (t:1778085052:R) (id:browse)",
        );
        assert_eq!(
            out,
            "hi __MENTION__ and __MENTION__ check __MENTION__ on __MENTION__ __MENTION__"
        );
        assert_eq!(n, 5);
    }

    #[test]
    fn rewrite_mentions_leaves_natural_parens_alone() {
        let (out, n) = rewrite_mentions("hello (world) and (some text)");
        assert_eq!(out, "hello (world) and (some text)");
        assert_eq!(n, 0);
    }

    #[test]
    fn rewrite_mentions_is_idempotent() {
        let (first, _) = rewrite_mentions("ping (@1234)");
        let (second, n) = rewrite_mentions(&first);
        assert_eq!(first, second);
        assert_eq!(n, 0);
    }

    #[test]
    fn rewrite_emoji_shortcodes_replaces_typical_forms() {
        let (out, n) = rewrite_emoji_shortcodes("lol :kek: that's :PanFrown:");
        assert_eq!(out, "lol __EMOJI__ that's __EMOJI__");
        assert_eq!(n, 2);
    }

    #[test]
    fn rewrite_emoji_shortcodes_leaves_normal_punctuation_alone() {
        let (out, n) = rewrite_emoji_shortcodes("hi: how are you");
        assert_eq!(out, "hi: how are you");
        assert_eq!(n, 0);
    }

    #[test]
    fn rewrite_emoji_shortcodes_is_idempotent() {
        let (first, _) = rewrite_emoji_shortcodes("yo :kek:");
        let (second, n) = rewrite_emoji_shortcodes(&first);
        assert_eq!(first, second);
        assert_eq!(n, 0);
    }

    #[test]
    fn strip_markdown_removes_bold_italic_runs() {
        let (out, _n) = strip_markdown("hello **bold** and **more** text");
        assert_eq!(out, "hello bold and more text");
        let (out, _n) = strip_markdown("look at ********* runs");
        assert_eq!(out, "look at runs");
    }

    #[test]
    fn strip_markdown_collapses_em_dashes() {
        let (out, _n) = strip_markdown("hello -- world --- foo");
        assert_eq!(out, "hello - world - foo");
    }

    #[test]
    fn strip_markdown_preserves_placeholders() {
        let (out, n) = strip_markdown("hi __URL__ there __MENTION__ ok");
        assert_eq!(out, "hi __URL__ there __MENTION__ ok");
        assert_eq!(n, 0);
    }

    #[test]
    fn strip_markdown_strips_bullet_prefix() {
        let (out, _n) = strip_markdown("- here is an item");
        assert_eq!(out, "here is an item");
        let (out, _n) = strip_markdown(":- another item");
        assert_eq!(out, "another item");
    }

    #[test]
    fn strip_markdown_strips_numbered_list_prefix() {
        let (out, _n) = strip_markdown("1. first item");
        assert_eq!(out, "first item");
        let (out, _n) = strip_markdown("42. answer");
        assert_eq!(out, "answer");
    }

    #[test]
    fn strip_markdown_handles_standalone_star() {
        let (out, n) = strip_markdown("yes * no");
        assert_eq!(out, "yes no");
        assert_eq!(n, 1);
    }

    #[test]
    fn strip_markdown_no_op_on_clean_text() {
        let (out, n) = strip_markdown("the cat sat on the mat.");
        assert_eq!(out, "the cat sat on the mat.");
        assert_eq!(n, 0);
    }

    #[test]
    fn strip_markdown_is_idempotent() {
        let (first, _) = strip_markdown("**bold** with -- dash and __URL__");
        let (second, n) = strip_markdown(&first);
        assert_eq!(first, second);
        assert_eq!(n, 0);
    }

    #[test]
    fn strip_escapes_handles_common_sequences() {
        let (out, n) = strip_escapes("hello\\nworld");
        assert_eq!(out, "hello world");
        assert_eq!(n, 1);
        let (out, n) = strip_escapes("path is C:\\Users\\foo");
        // \U and \f both lose the backslash, keep the next char.
        assert_eq!(out, "path is C:Usersfoo");
        assert_eq!(n, 2);
    }

    #[test]
    fn strip_escapes_handles_double_backslash() {
        let (out, n) = strip_escapes("a\\\\b");
        assert_eq!(out, "ab");
        assert_eq!(n, 2);
    }

    #[test]
    fn strip_escapes_handles_trailing_backslash() {
        let (out, n) = strip_escapes("the\\");
        assert_eq!(out, "the");
        assert_eq!(n, 1);
    }

    #[test]
    fn strip_escapes_no_op_on_clean_text() {
        let (out, n) = strip_escapes("hello world, how are you?");
        assert_eq!(out, "hello world, how are you?");
        assert_eq!(n, 0);
    }

    #[test]
    fn strip_escapes_is_idempotent() {
        let (first, _) = strip_escapes("test\\nfoo\\\\bar");
        let (second, n) = strip_escapes(&first);
        assert_eq!(first, second);
        assert_eq!(n, 0);
    }

    #[test]
    fn parse_raw_roundtrip_basic() {
        let raw = "<SEC>\n<PERSON_1> hi there </PERSON_1>\n<PERSON_0> hello back </PERSON_0>\n";
        let secs = parse_raw(raw).unwrap();
        assert_eq!(secs.len(), 1);
        assert_eq!(secs[0].len(), 2);
        assert_eq!(secs[0][0].person_id, 1);
        assert_eq!(secs[0][0].text, "hi there");
        assert_eq!(secs[0][1].person_id, 0);
        assert_eq!(secs[0][1].text, "hello back");
    }

    fn make_section(turns: &[(u32, &str)]) -> Vec<RawTurn> {
        turns
            .iter()
            .map(|(pid, txt)| RawTurn {
                person_id: *pid,
                text: (*txt).to_string(),
            })
            .collect()
    }

    fn run_prefix_cap(sections: &mut Vec<Vec<RawTurn>>) -> u32 {
        let mut counts = 0u32;
        let mut prefix_counts: HashMap<String, u32> = HashMap::new();
        for sec in sections.iter_mut() {
            sec.retain(|turn| {
                if turn.person_id != BOT_PERSON_ID {
                    return true;
                }
                let toks: Vec<&str> = turn.text.split_whitespace().collect();
                if toks.len() < PREFIX_LEN {
                    return true;
                }
                let key = toks[..PREFIX_LEN]
                    .iter()
                    .map(|t| t.to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ");
                let entry = prefix_counts.entry(key).or_insert(0);
                if *entry >= MAX_PREFIX_REPEATS {
                    counts += 1;
                    return false;
                }
                *entry += 1;
                true
            });
        }
        counts
    }

    #[test]
    fn prefix_cap_caps_shared_prefix_at_max_repeats() {
        let bot = BOT_PERSON_ID;
        let user = 1;
        let mut sections: Vec<Vec<RawTurn>> = (0..MAX_PREFIX_REPEATS + 5)
            .map(|i| {
                make_section(&[
                    (user, "question"),
                    (
                        bot,
                        // Same first 5 tokens, varying suffix.
                        Box::leak(
                            format!("here are some ways to do thing number {}", i).into_boxed_str(),
                        ),
                    ),
                ])
            })
            .collect();
        let dropped = run_prefix_cap(&mut sections);
        assert_eq!(dropped, 5, "should drop 5 turns above the cap");
        let bot_turns: u32 = sections
            .iter()
            .map(|sec| sec.iter().filter(|t| t.person_id == bot).count() as u32)
            .sum();
        assert_eq!(bot_turns, MAX_PREFIX_REPEATS);
    }

    #[test]
    fn prefix_cap_exempts_short_turns() {
        let bot = BOT_PERSON_ID;
        let user = 1;
        // Each bot turn has fewer than PREFIX_LEN tokens, so rule 14
        // shouldn't touch them no matter how many times they recur.
        let mut sections: Vec<Vec<RawTurn>> = (0..MAX_PREFIX_REPEATS + 10)
            .map(|_| make_section(&[(user, "question"), (bot, "hi friend")]))
            .collect();
        let dropped = run_prefix_cap(&mut sections);
        assert_eq!(dropped, 0);
        for sec in &sections {
            assert_eq!(sec.len(), 2);
        }
    }

    #[test]
    fn prefix_cap_exempts_user_turns() {
        let bot = BOT_PERSON_ID;
        let user = 1;
        // User asks the same prefix many times; rule 14 should NOT
        // cap user-side, only bot-side helper templates.
        let mut sections: Vec<Vec<RawTurn>> = (0..MAX_PREFIX_REPEATS + 20)
            .map(|i| {
                make_section(&[
                    (
                        user,
                        Box::leak(
                            format!("what is the difference between thing {}", i)
                                .into_boxed_str(),
                        ),
                    ),
                    (bot, "ok"),
                ])
            })
            .collect();
        let dropped = run_prefix_cap(&mut sections);
        assert_eq!(dropped, 0, "user-side prefix repetition should be exempt");
    }

    #[test]
    fn prefix_cap_is_case_insensitive() {
        let bot = BOT_PERSON_ID;
        let user = 1;
        let mut sections: Vec<Vec<RawTurn>> = (0..MAX_PREFIX_REPEATS + 2)
            .map(|i| {
                let txt = if i % 2 == 0 {
                    format!("Here Are Some Ways To do thing {}", i)
                } else {
                    format!("here are some ways to do thing {}", i)
                };
                make_section(&[(user, "q"), (bot, Box::leak(txt.into_boxed_str()))])
            })
            .collect();
        let dropped = run_prefix_cap(&mut sections);
        assert_eq!(dropped, 2);
    }

    #[test]
    fn prefix_cap_distinct_prefixes_unaffected() {
        let bot = BOT_PERSON_ID;
        let user = 1;
        let mut sections: Vec<Vec<RawTurn>> = (0..MAX_PREFIX_REPEATS + 20)
            .map(|i| {
                make_section(&[
                    (user, "q"),
                    (
                        bot,
                        // Different first token each time.
                        Box::leak(
                            format!("reply{} this is a varying sentence", i).into_boxed_str(),
                        ),
                    ),
                ])
            })
            .collect();
        let dropped = run_prefix_cap(&mut sections);
        assert_eq!(dropped, 0);
    }
}
