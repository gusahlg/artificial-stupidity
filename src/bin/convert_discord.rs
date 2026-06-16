//! Convert scraped Discord TSVs into the PERSON_N training corpus.
//!
//! Reads every `data/discord/<bucket>/<channel>.tsv`, splits messages into
//! sections, and writes `data/dialogs.txt` in the format the trainer +
//! matcher expect.
//!
//! PERSON_N is **section-local**: within each section, the bot is always
//! `PERSON_0` (when it speaks), and every other Discord user gets a fresh
//! id starting from 1 in the order they first speak in that section. The
//! same Discord user therefore can be `PERSON_2` in one section and
//! `PERSON_1` in another — the tag is a within-section discriminator, not
//! a global identity.
//!
//! Within a section, consecutive messages from the same author are merged
//! into a single `<PERSON_N> ... </PERSON_N>` turn. This matches how
//! Discord conversations actually flow (people often send three short
//! messages in a row that really form one turn).
//!
//! Section boundary strategy:
//!   - Default: 30-minute idle in the same channel opens a new section.
//!   - With `USE_OLLAMA=1`: ask a local LLM (`qwen2.5:7b` by default) where
//!     topic shifts occur. Results cached per channel in
//!     `data/discord/<bucket>/<channel>.sections.json` so re-runs are cheap.
//!
//! Filters:
//!   - Drops runs of ≥3 consecutive bot messages (monologues we don't want
//!     to train on).
//!   - Drops messages whose content is empty after sanitization.
//!   - Sanitizes content by mapping `<` / `>` to `(` / `)` so user text
//!     can't impersonate a PERSON tag or `<SEC>`.

use anyhow::{Context, Result};
use rust_fun::ollama::OllamaClient;
use rust_fun::persons::{close_tag, open_tag};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const DIALOGS_OUT: &str = "data/dialogs.txt";
const DISCORD_ROOT: &str = "data/discord";

/// Idle gap (seconds) that opens a new section without any LLM input.
const IDLE_GAP_SECS: i64 = 30 * 60;

/// Bot's discord user id. The bot is always PERSON_0 in any section where
/// it appears; if it doesn't appear, PERSON_0 simply isn't used in that
/// section.
const BOT_DISCORD_ID: u64 = 1440038621230010418;

/// Length of the message window sent to Ollama in one prompt.
const WINDOW_SIZE: usize = 20;
/// Sliding-window stride. (overlap = WINDOW_SIZE - WINDOW_STRIDE = 5).
const WINDOW_STRIDE: usize = 15;

#[derive(Clone, Debug)]
struct Msg {
    id: u64,
    ts_secs: i64,
    author_id: u64,
    display_name: String,
    content: String,
}

fn main() -> Result<()> {
    let use_ollama = std::env::var("USE_OLLAMA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let ollama = if use_ollama {
        let c = OllamaClient::from_env();
        eprintln!("convert_discord: USE_OLLAMA=1, model = {}", c.model);
        Some(c)
    } else {
        eprintln!("convert_discord: time-only section split (30 min idle). Set USE_OLLAMA=1 for AI-driven splits.");
        None
    };

    let tsvs = collect_tsv_paths(DISCORD_ROOT)?;
    if tsvs.is_empty() {
        eprintln!("convert_discord: no TSV files found under {}", DISCORD_ROOT);
        return Ok(());
    }
    eprintln!("convert_discord: {} channel TSVs", tsvs.len());

    let mut out = fs::File::create(DIALOGS_OUT)
        .with_context(|| format!("create {}", DIALOGS_OUT))?;
    let mut total_messages: u64 = 0;
    let mut total_sections: u64 = 0;
    let mut total_turns: u64 = 0;
    let mut max_persons_in_section: u32 = 0;

    for tsv_path in tsvs {
        let messages = parse_tsv(&tsv_path)
            .with_context(|| format!("parse {:?}", tsv_path))?;
        if messages.is_empty() {
            continue;
        }
        let filtered = filter_bot_monologues(messages);
        if filtered.is_empty() {
            continue;
        }
        let boundaries = compute_boundaries(&tsv_path, &filtered, ollama.as_ref())?;
        let sections = split_into_sections(&filtered, &boundaries);

        let mut emitted_sections_for_channel: u64 = 0;
        for section in sections.iter() {
            let turns = build_section_turns(section);
            // Sections with <2 turns have no dialog signal:
            // `extract_train_examples` in src/neural_network.rs skips them
            // anyway. Filtering at ingest time keeps the file smaller
            // and prevents them from biasing the val-tail split.
            if turns.len() < 2 {
                continue;
            }
            writeln!(out, "<SEC>")?;
            let mut peak_pid: u32 = 0;
            for (pid, content) in turns.iter() {
                writeln!(out, "{} {} {}", open_tag(*pid), content, close_tag(*pid))?;
                if *pid > peak_pid {
                    peak_pid = *pid;
                }
            }
            total_sections += 1;
            total_turns += turns.len() as u64;
            emitted_sections_for_channel += 1;
            // peak_pid + 1 is the count of distinct persons in this section.
            if peak_pid + 1 > max_persons_in_section {
                max_persons_in_section = peak_pid + 1;
            }
        }
        total_messages += filtered.len() as u64;
        eprintln!(
            "  {:?}: {} msgs -> {} sections",
            tsv_path.file_name().unwrap_or_default(),
            filtered.len(),
            emitted_sections_for_channel
        );
    }

    eprintln!(
        "convert_discord: {} sections, {} merged turns from {} messages -> {}; \
         max distinct persons in a section: {}",
        total_sections,
        total_turns,
        total_messages,
        DIALOGS_OUT,
        max_persons_in_section,
    );
    Ok(())
}

/// Build the emitted turn list for one section: per-section PERSON_N
/// assignment (bot → 0, others → 1, 2, ... in first-seen order) plus
/// merging of consecutive same-author messages into one turn.
fn build_section_turns(section: &[Msg]) -> Vec<(u32, String)> {
    let mut local: HashMap<u64, u32> = HashMap::new();
    let mut next_local: u32 = 1;
    let mut turns: Vec<(u32, String)> = Vec::new();
    for m in section.iter() {
        let pid = if m.author_id == BOT_DISCORD_ID {
            // Bot always claims PERSON_0, even if humans spoke first.
            *local.entry(m.author_id).or_insert(0)
        } else if let Some(&id) = local.get(&m.author_id) {
            id
        } else {
            let id = next_local;
            local.insert(m.author_id, id);
            next_local += 1;
            id
        };
        let content = sanitize_for_corpus(&m.content);
        let trimmed = content.trim();
        if trimmed.is_empty() {
            continue;
        }
        match turns.last_mut() {
            Some(last) if last.0 == pid => {
                last.1.push(' ');
                last.1.push_str(trimmed);
            }
            _ => turns.push((pid, trimmed.to_string())),
        }
    }
    turns
}

fn collect_tsv_paths(root: &str) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let root = Path::new(root);
    if !root.exists() {
        return Ok(out);
    }
    for guild_entry in fs::read_dir(root).with_context(|| format!("read_dir {:?}", root))? {
        let guild = guild_entry?.path();
        if !guild.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&guild)? {
            let p = entry?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("tsv") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn parse_tsv(path: &Path) -> Result<Vec<Msg>> {
    let raw = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (n, line) in raw.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 6 {
            eprintln!("  WARN: {:?}:{} too few columns ({})", path, n + 1, parts.len());
            continue;
        }
        let id: u64 = match parts[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ts_secs = parse_rfc3339_to_secs(parts[1]).unwrap_or_else(|| {
            // Fallback: derive from snowflake epoch.
            snowflake_to_secs(id)
        });
        let author_id: u64 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let display_name = parts[3].to_string();
        let content = unescape_tsv(parts[5..].join("\t").as_str());
        out.push(Msg {
            id,
            ts_secs,
            author_id,
            display_name,
            content,
        });
    }
    // The scraper writes newest-first per batch; sort by id (≈ time).
    out.sort_by_key(|m| m.id);
    Ok(out)
}

fn unescape_tsv(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_rfc3339_to_secs(s: &str) -> Option<i64> {
    // Very loose parser: "YYYY-MM-DDTHH:MM:SSZ" → unix seconds.
    if s.len() < 20 {
        return None;
    }
    let year: i64 = s[0..4].parse().ok()?;
    let mon: i64 = s[5..7].parse().ok()?;
    let day: i64 = s[8..10].parse().ok()?;
    let hr: i64 = s[11..13].parse().ok()?;
    let min: i64 = s[14..16].parse().ok()?;
    let sec: i64 = s[17..19].parse().ok()?;
    // Days from civil date (Howard Hinnant's algorithm).
    let y = year - (if mon <= 2 { 1 } else { 0 });
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let doy = (153 * (if mon > 2 { mon - 3 } else { mon + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hr * 3_600 + min * 60 + sec)
}

const DISCORD_EPOCH_MS: u64 = 1_420_070_400_000;

fn snowflake_to_secs(id: u64) -> i64 {
    ((id >> 22) as i64 + DISCORD_EPOCH_MS as i64) / 1000
}

fn filter_bot_monologues(messages: Vec<Msg>) -> Vec<Msg> {
    if messages.len() < 3 {
        return messages;
    }
    let bot_id = BOT_DISCORD_ID;
    let n = messages.len();
    let mut keep = vec![true; n];
    let mut i = 0;
    while i < n {
        if messages[i].author_id == bot_id {
            let mut j = i;
            while j < n && messages[j].author_id == bot_id {
                j += 1;
            }
            let run = j - i;
            if run >= 3 {
                for k in i..j {
                    keep[k] = false;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    messages
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| keep[*idx])
        .map(|(_, m)| m)
        .collect()
}

/// Compute the set of indices that *start* a new section. Index 0 is always
/// the start of the first section (not included in the returned set).
fn compute_boundaries(
    tsv_path: &Path,
    messages: &[Msg],
    ollama: Option<&OllamaClient>,
) -> Result<BTreeSet<usize>> {
    let mut boundaries: BTreeSet<usize> = BTreeSet::new();
    // Always include time-based gaps; AI adds *additional* boundaries.
    for i in 1..messages.len() {
        if messages[i].ts_secs - messages[i - 1].ts_secs >= IDLE_GAP_SECS {
            boundaries.insert(i);
        }
    }
    if let Some(client) = ollama {
        match ollama_boundaries(client, tsv_path, messages) {
            Ok(extra) => {
                for b in extra {
                    boundaries.insert(b);
                }
            }
            Err(e) => {
                eprintln!(
                    "  WARN: Ollama section detection failed for {:?}: {}",
                    tsv_path.file_name(),
                    e
                );
            }
        }
    }
    Ok(boundaries)
}

fn split_into_sections<'a>(
    messages: &'a [Msg],
    boundaries: &BTreeSet<usize>,
) -> Vec<&'a [Msg]> {
    let mut starts: Vec<usize> = std::iter::once(0).chain(boundaries.iter().copied()).collect();
    starts.sort_unstable();
    starts.dedup();
    let mut out = Vec::with_capacity(starts.len());
    for (i, &s) in starts.iter().enumerate() {
        let e = starts.get(i + 1).copied().unwrap_or(messages.len());
        if s < e {
            out.push(&messages[s..e]);
        }
    }
    out
}

fn ollama_boundaries(
    client: &OllamaClient,
    tsv_path: &Path,
    messages: &[Msg],
) -> Result<Vec<usize>> {
    let cache_path = tsv_path.with_extension("sections.json");
    let cache_key = format!("{}:{}", messages.first().map(|m| m.id).unwrap_or(0), messages.last().map(|m| m.id).unwrap_or(0));
    if let Ok(s) = fs::read_to_string(&cache_path) {
        if let Ok(cache) = serde_json::from_str::<CacheFile>(&s) {
            if cache.key == cache_key {
                return Ok(cache.boundaries);
            }
        }
    }

    let mut found: BTreeSet<usize> = BTreeSet::new();
    let mut start = 0usize;
    while start < messages.len() {
        let end = (start + WINDOW_SIZE).min(messages.len());
        let slice = &messages[start..end];
        let prompt = build_window_prompt(slice);
        let resp = client.generate(&prompt)?;
        let local_indices = parse_index_array(&resp);
        for idx in local_indices {
            // Map window-local index to global; idx 0 isn't a "new" boundary,
            // it's the start. We only respect idx >= 1.
            if idx >= 1 && idx < slice.len() {
                found.insert(start + idx);
            }
        }
        if end == messages.len() {
            break;
        }
        start += WINDOW_STRIDE;
    }

    let boundaries: Vec<usize> = found.iter().copied().collect();
    let _ = fs::write(
        &cache_path,
        serde_json::to_string(&CacheFile {
            key: cache_key,
            boundaries: boundaries.clone(),
        })?,
    );
    Ok(boundaries)
}

fn build_window_prompt(messages: &[Msg]) -> String {
    let mut lines = String::new();
    for (i, m) in messages.iter().enumerate() {
        let snippet = sanitize_for_prompt(&m.content);
        lines.push_str(&format!("{}: {}: {}\n", i, m.display_name, snippet));
    }
    format!(
        "You are a tool that segments Discord chat into conversation sections.\n\
         A new section begins when there is a clear topic shift, a new exchange after \
         a long lull, or an unrelated greeting/question. Otherwise messages stay in the \
         same section.\n\n\
         Here are {} consecutive messages, numbered 0..{}:\n\
         {}\n\
         Reply with ONLY a JSON array of the line numbers where a NEW section should begin. \
         Do not include 0 — line 0 always starts the first section. If everything in this \
         window is one section, reply []. Examples: [3, 14] or [].\n\
         JSON only, no commentary.",
        messages.len(),
        messages.len() - 1,
        lines
    )
}

fn parse_index_array(s: &str) -> Vec<usize> {
    // Find the first '[' .. ']' span and parse it as JSON.
    let start = match s.find('[') {
        Some(i) => i,
        None => return Vec::new(),
    };
    let end = match s[start..].find(']') {
        Some(i) => start + i + 1,
        None => return Vec::new(),
    };
    let slice = &s[start..end];
    serde_json::from_str::<Vec<usize>>(slice).unwrap_or_default()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheFile {
    key: String,
    boundaries: Vec<usize>,
}

fn sanitize_for_corpus(s: &str) -> String {
    // Step 1: replace control chars + angle brackets so user content
    // can't impersonate a PERSON tag or break newline framing.
    let prepped: String = s
        .chars()
        .map(|c| match c {
            '\n' | '\r' | '\t' => ' ',
            '<' => '(',
            '>' => ')',
            c => c,
        })
        .collect();
    // Step 2: collapse URL-looking and emoji-shortcode whitespace-
    // tokens to single placeholders (`__URL__`, `__EMOJI__`). Our
    // punctuation-splitting tokenizer otherwise fragments a single
    // Tenor/Discord link into ~20 single-char tokens that consume
    // most of the 32-token context window with unpredictable noise,
    // and shatters a `:name:` shortcode into `:`, `name`, `:`. The
    // placeholders are alphanumeric+underscore tokens recognized
    // symmetrically by `clean_corpus` rules 4/9/11 via the shared
    // `looks_like_url` / `is_emoji_shortcode` helpers.
    let mut out = String::with_capacity(prepped.len());
    let mut first = true;
    for word in prepped.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        if rust_fun::text_utils::looks_like_url(word) {
            out.push_str("__URL__");
        } else if rust_fun::text_utils::is_emoji_shortcode(word) {
            out.push_str("__EMOJI__");
        } else {
            out.push_str(word);
        }
    }
    out
}

fn sanitize_for_prompt(s: &str) -> String {
    // Keep prompt human-readable; just collapse whitespace and truncate.
    let collapsed: String = s
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect();
    let trimmed: String = collapsed.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.chars().count() > 240 {
        let cut: String = trimmed.chars().take(237).collect();
        format!("{}...", cut)
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_brackets() {
        assert_eq!(sanitize_for_corpus("<PERSON_3>hi"), "(PERSON_3)hi");
        assert_eq!(sanitize_for_corpus("a\nb\tc"), "a b c");
    }

    #[test]
    fn parse_index_array_basic() {
        assert_eq!(parse_index_array("[3, 14]"), vec![3, 14]);
        assert_eq!(parse_index_array("answer: [1,2,3] hope this helps"), vec![1, 2, 3]);
        assert_eq!(parse_index_array("[]"), Vec::<usize>::new());
        assert_eq!(parse_index_array("nothing here"), Vec::<usize>::new());
    }

    #[test]
    fn unescape_tsv_handles_escapes() {
        assert_eq!(unescape_tsv(r"hello\nworld"), "hello\nworld");
        assert_eq!(unescape_tsv(r"a\\b"), "a\\b");
        assert_eq!(unescape_tsv(r"a\tb"), "a\tb");
    }

    #[test]
    fn rfc3339_parser_aligns_with_unix() {
        // 2026-05-17T12:00:00Z → known unix seconds.
        let s = parse_rfc3339_to_secs("2026-05-17T12:00:00Z").unwrap();
        // Just check it's in a sane ballpark (>2020, <2030).
        assert!(s > 1_577_836_800 && s < 1_893_456_000);
    }

    #[test]
    fn snowflake_decode_recent_id() {
        // ID ~ May 14 2026
        let s = snowflake_to_secs(1_504_363_660_191_989_851);
        assert!(s > 1_700_000_000);
    }

    fn mk(id: u64, author: u64, content: &str) -> Msg {
        Msg {
            id,
            ts_secs: id as i64,
            author_id: author,
            display_name: format!("user{}", author),
            content: content.to_string(),
        }
    }

    #[test]
    fn build_section_turns_assigns_bot_to_zero() {
        let section = vec![
            mk(1, 100, "hi"),
            mk(2, 200, "hello"),
            mk(3, BOT_DISCORD_ID, "greetings"),
        ];
        let turns = build_section_turns(&section);
        // PERSON_0 reserved for bot. Humans get 1 and 2 in arrival order.
        assert_eq!(turns, vec![
            (1u32, "hi".to_string()),
            (2u32, "hello".to_string()),
            (0u32, "greetings".to_string()),
        ]);
    }

    #[test]
    fn build_section_turns_merges_consecutive_same_author() {
        let section = vec![
            mk(1, 100, "hi"),
            mk(2, 100, "there"),
            mk(3, 100, "everyone"),
            mk(4, 200, "hello"),
            mk(5, 100, "back to me"),
        ];
        let turns = build_section_turns(&section);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0], (1u32, "hi there everyone".to_string()));
        assert_eq!(turns[1], (2u32, "hello".to_string()));
        assert_eq!(turns[2], (1u32, "back to me".to_string()));
    }

    #[test]
    fn build_section_turns_skips_empty_after_sanitization() {
        let section = vec![
            mk(1, 100, "hi"),
            mk(2, 100, "   "),   // whitespace-only, dropped
            mk(3, 100, "still me"),
            mk(4, 200, "<<<>>>"), // sanitizes to "(((>)))" which is non-empty after trim
        ];
        let turns = build_section_turns(&section);
        // First-two merge through the dropped whitespace into one PERSON_1 turn.
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0], (1u32, "hi still me".to_string()));
        assert_eq!(turns[1].0, 2u32);
    }

    #[test]
    fn build_section_turns_local_ids_reset_per_section() {
        // Same author 100 is PERSON_1 in section A and PERSON_2 in section B
        // (because someone else spoke first in B). Verifies local mapping
        // does not leak across calls.
        let section_a = vec![mk(1, 100, "first"), mk(2, 200, "second")];
        let section_b = vec![mk(10, 200, "different"), mk(11, 100, "after")];
        let a = build_section_turns(&section_a);
        let b = build_section_turns(&section_b);
        assert_eq!(a[0].0, 1);
        assert_eq!(a[1].0, 2);
        assert_eq!(b[0].0, 1);
        assert_eq!(b[1].0, 2);
        // The author_id mapping is local per call — no global state.
    }

    #[test]
    fn build_section_turns_bot_first_keeps_pid_zero() {
        let section = vec![
            mk(1, BOT_DISCORD_ID, "I speak first"),
            mk(2, 100, "human reply"),
        ];
        let turns = build_section_turns(&section);
        assert_eq!(turns[0].0, 0);
        assert_eq!(turns[1].0, 1);
    }

    #[test]
    fn filter_drops_long_bot_runs() {
        let msgs = vec![
            mk(1, 100, "x"),              // human
            mk(2, BOT_DISCORD_ID, "x"),   // bot
            mk(3, BOT_DISCORD_ID, "x"),   // bot
            mk(4, BOT_DISCORD_ID, "x"),   // bot — run of 3, all three dropped
            mk(5, 200, "x"),              // human
            mk(6, BOT_DISCORD_ID, "x"),   // bot — short reply, kept
            mk(7, 100, "x"),              // human
        ];
        let kept = filter_bot_monologues(msgs);
        assert_eq!(kept.len(), 4);
        assert!(kept.iter().all(|m| m.id != 2 && m.id != 3 && m.id != 4));
    }
}
