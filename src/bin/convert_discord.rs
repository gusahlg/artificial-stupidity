//! Convert scraped Discord TSVs into the PERSON_N training corpus.
//!
//! Reads every `<DISCORD_ROOT>/<bucket>/<channel>.tsv` (default root
//! `data/discord`, overridable via the `DISCORD_ROOT` env var), splits
//! messages into sections, and writes `data/dialogs.txt` in the format
//! the trainer + matcher expect.
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
//! Reply awareness (TSV column 5 = replied-to message id):
//!   Reply edges are stitched into **chains** — the transitive closure of
//!   reply links forms one conversation, which is emitted as its own
//!   section in timestamp order. A threaded exchange scattered across an
//!   hour of interleaved channel chatter therefore becomes a clean
//!   adjacent dialog, which is exactly the signal the model trains on.
//!   Chains need >=2 messages from >=2 authors; anything else stays in
//!   the ambient flow. Messages claimed by a chain are removed from the
//!   ambient flow so they aren't double-counted.
//!
//! Mentions:
//!   `<@id>` / `<@!id>` are rewritten to `@displayname` (lowercased,
//!   alphanumeric-only) using an id->name map built from every TSV author
//!   column. This makes pings *learnable*: the corpus contains real
//!   `@name` tokens the live bot can resolve back into Discord pings.
//!   Unknown ids, role pings, channel refs, and timestamps become
//!   `__MENTION__`; custom emoji (`<:name:id>`) become `__EMOJI__`.
//!
//! Section boundary strategy (ambient flow):
//!   - Default: 30-minute idle in the same channel opens a new section.
//!   - With `USE_OLLAMA=1`: ask a local LLM (`qwen2.5:7b` by default) where
//!     topic shifts occur. Results cached per channel in
//!     `<root>/<bucket>/<channel>.sections.json` so re-runs are cheap.
//!
//! Run modes (the scraper's TSVs are cumulative, so re-converting must
//! not duplicate old sections):
//!   --full          truncate data/dialogs.txt and convert everything,
//!                   then record per-channel cursors. The classic build:
//!                   follow with inject_seed (x3) + clean_corpus.
//!   (default)       incremental: convert only messages newer than each
//!                   channel's cursor in `<root>/.convert-state.json` and
//!                   APPEND the resulting sections. Refuses to run if the
//!                   state file is missing (that would re-append all of
//!                   history on top of an existing corpus).
//!   --init-cursors  emit nothing; set cursors to the current TSV maxima.
//!                   Use once when adopting incremental mode on a corpus
//!                   that already contains the historical Discord ingest.
//!
//! Filters:
//!   - Drops runs of >=3 consecutive bot messages (monologues we don't
//!     want to train on).
//!   - Drops messages whose content is empty after sanitization.
//!   - Sanitizes content by mapping `<` / `>` to `(` / `)` so user text
//!     can't impersonate a PERSON tag or `<SEC>`.

use anyhow::{Context, Result, bail};
use rust_fun::ollama::OllamaClient;
use rust_fun::persons::{close_tag, open_tag};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const DIALOGS_OUT: &str = "data/dialogs.txt";
const DEFAULT_DISCORD_ROOT: &str = "data/discord";
const STATE_FILE_NAME: &str = ".convert-state.json";

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
    reply_to: Option<u64>,
    content: String,
}

enum Mode {
    Full,
    Incremental,
    InitCursors,
}

/// Per-channel "everything up to this message id is already in the
/// corpus" cursors, serialized to `<root>/.convert-state.json`.
#[derive(serde::Serialize, serde::Deserialize, Default)]
struct ConvertState {
    channels: BTreeMap<String, u64>,
}

fn main() -> Result<()> {
    let mode = match std::env::args().nth(1).as_deref() {
        Some("--full") => Mode::Full,
        Some("--init-cursors") => Mode::InitCursors,
        None => Mode::Incremental,
        Some(other) => bail!("unknown flag {other}; use --full, --init-cursors, or no flag"),
    };

    let root = std::env::var("DISCORD_ROOT").unwrap_or_else(|_| DEFAULT_DISCORD_ROOT.to_string());
    let state_path = Path::new(&root).join(STATE_FILE_NAME);
    let mut state = load_state(&state_path)?;
    if matches!(mode, Mode::Incremental) && state.channels.is_empty() {
        bail!(
            "no cursor state at {:?}; an incremental run on top of an existing corpus \
             would duplicate all of history. Run with --full (rebuild from scratch) or \
             --init-cursors (adopt the current TSVs as already-converted) first.",
            state_path
        );
    }

    let use_ollama = std::env::var("USE_OLLAMA")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let ollama = if use_ollama {
        let c = OllamaClient::from_env();
        eprintln!("convert_discord: USE_OLLAMA=1, model = {}", c.model);
        Some(c)
    } else {
        eprintln!(
            "convert_discord: time-only section split (30 min idle). Set USE_OLLAMA=1 for AI-driven splits."
        );
        None
    };

    let tsvs = collect_tsv_paths(&root)?;
    if tsvs.is_empty() {
        eprintln!("convert_discord: no TSV files found under {}", root);
        return Ok(());
    }
    eprintln!("convert_discord: {} channel TSVs under {}", tsvs.len(), root);

    // Pass 1: parse everything and build the global id -> display-name map
    // used to turn `<@id>` mentions into learnable `@name` tokens. Later
    // messages win so renames converge on the newest name.
    let mut per_channel: Vec<(PathBuf, String, Vec<Msg>)> = Vec::new();
    let mut names: HashMap<u64, String> = HashMap::new();
    for tsv_path in tsvs {
        let key = channel_key(&root, &tsv_path);
        let messages = parse_tsv(&tsv_path).with_context(|| format!("parse {:?}", tsv_path))?;
        for m in &messages {
            names.insert(m.author_id, m.display_name.clone());
        }
        per_channel.push((tsv_path, key, messages));
    }

    if matches!(mode, Mode::InitCursors) {
        for (_, key, messages) in &per_channel {
            if let Some(max_id) = messages.iter().map(|m| m.id).max() {
                state.channels.insert(key.clone(), max_id);
            }
        }
        save_state(&state_path, &state)?;
        eprintln!(
            "convert_discord: cursors initialized for {} channels (nothing emitted)",
            state.channels.len()
        );
        return Ok(());
    }

    let mut out: fs::File = match mode {
        Mode::Full => {
            fs::File::create(DIALOGS_OUT).with_context(|| format!("create {}", DIALOGS_OUT))?
        }
        _ => fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(DIALOGS_OUT)
            .with_context(|| format!("append {}", DIALOGS_OUT))?,
    };

    let mut total_messages: u64 = 0;
    let mut total_sections: u64 = 0;
    let mut total_chain_sections: u64 = 0;
    let mut total_turns: u64 = 0;
    let mut max_persons_in_section: u32 = 0;

    for (tsv_path, key, messages) in per_channel {
        if messages.is_empty() {
            continue;
        }
        let cursor = match mode {
            Mode::Full => 0,
            _ => state.channels.get(&key).copied().unwrap_or(0),
        };
        let max_id = messages.iter().map(|m| m.id).max().unwrap_or(0);
        if max_id <= cursor {
            continue; // nothing new in this channel
        }

        let filtered = filter_bot_monologues(messages);
        if filtered.is_empty() {
            state.channels.insert(key, max_id);
            continue;
        }

        // Reply chains first: they may reach back past the cursor (a new
        // reply resurrects its older context — that duplication is bounded
        // and clean_corpus rule 7 caps exact repeats). A chain is emitted
        // only if it contains at least one NEW message AND survives to >=2
        // turns. Turns are built up front so a chain that COLLAPSES (e.g.
        // an attachment-only reply whose content sanitizes to empty) does
        // NOT claim its messages — they fall through to the ambient flow
        // instead of being silently deleted along with the dead chain.
        let chains = extract_reply_chains(&filtered);
        let mut chained_ids: HashSet<u64> = HashSet::new();
        // (sort key, already-built turns).
        let mut emit_units: Vec<(u64, Vec<(u32, String)>)> = Vec::new();
        let mut n_chains = 0u64;
        for chain in chains {
            if !chain.iter().any(|m| m.id > cursor) {
                // Entirely old: already in the corpus (and all id <= cursor,
                // so ambient's `id > cursor` filter already excludes them).
                continue;
            }
            let turns = build_section_turns(&chain, &names);
            if turns.len() < 2 {
                // Dead chain — leave its messages for the ambient flow.
                continue;
            }
            for m in &chain {
                chained_ids.insert(m.id);
            }
            let first_id = chain.first().map(|m| m.id).unwrap_or(0);
            emit_units.push((first_id, turns));
            n_chains += 1;
        }

        let ambient: Vec<Msg> = filtered
            .iter()
            .filter(|m| m.id > cursor && !chained_ids.contains(&m.id))
            .cloned()
            .collect();
        if !ambient.is_empty() {
            let boundaries = compute_boundaries(&tsv_path, &ambient, ollama.as_ref())?;
            for section in split_into_sections(&ambient, &boundaries) {
                let turns = build_section_turns(section, &names);
                // Sections with <2 turns have no dialog signal:
                // `extract_train_examples` in src/neural_network.rs skips
                // them anyway. Filtering at ingest time keeps the file
                // smaller and prevents biasing the val-tail split.
                if turns.len() < 2 {
                    continue;
                }
                let first_id = section.first().map(|m| m.id).unwrap_or(0);
                emit_units.push((first_id, turns));
            }
        }
        emit_units.sort_by_key(|(first_id, _)| *first_id);

        // Render the whole channel into one buffer, then write it in a
        // single `write_all` + `flush`. This shrinks the torn-line window
        // (a hard crash mid-`writeln!` could otherwise leave `<PERSON_1>
        // foo` with no close tag, which makes clean_corpus bail) to a
        // single syscall, and lets us advance + persist the cursor only
        // AFTER the sections are durably on disk.
        let mut buf = String::new();
        let mut emitted_sections_for_channel: u64 = 0;
        for (_, turns) in &emit_units {
            buf.push_str("<SEC>\n");
            let mut peak_pid: u32 = 0;
            for (pid, content) in turns.iter() {
                buf.push_str(&format!("{} {} {}\n", open_tag(*pid), content, close_tag(*pid)));
                if *pid > peak_pid {
                    peak_pid = *pid;
                }
            }
            total_sections += 1;
            total_turns += turns.len() as u64;
            emitted_sections_for_channel += 1;
            if peak_pid + 1 > max_persons_in_section {
                max_persons_in_section = peak_pid + 1;
            }
        }
        out.write_all(buf.as_bytes())?;
        out.flush()?;
        // Cursor advanced + persisted per channel AFTER its sections are
        // durable, so a mid-run crash re-appends at most this one channel's
        // (already rule-7-capped) sections next run instead of all of them.
        state.channels.insert(key.clone(), max_id);
        save_state(&state_path, &state)?;

        total_chain_sections += n_chains;
        total_messages += filtered.iter().filter(|m| m.id > cursor).count() as u64;
        eprintln!(
            "  {:?}: {} new msgs -> {} sections ({} reply chains)",
            tsv_path.file_name().unwrap_or_default(),
            filtered.iter().filter(|m| m.id > cursor).count(),
            emitted_sections_for_channel,
            n_chains,
        );
    }

    out.flush()?;
    save_state(&state_path, &state)?;
    eprintln!(
        "convert_discord: {} sections ({} from reply chains), {} merged turns from {} messages -> {}; \
         max distinct persons in a section: {}",
        total_sections,
        total_chain_sections,
        total_turns,
        total_messages,
        DIALOGS_OUT,
        max_persons_in_section,
    );
    Ok(())
}

fn channel_key(root: &str, tsv_path: &Path) -> String {
    tsv_path
        .strip_prefix(root)
        .unwrap_or(tsv_path)
        .with_extension("")
        .to_string_lossy()
        .trim_start_matches('/')
        .to_string()
}

fn load_state(path: &Path) -> Result<ConvertState> {
    if !path.exists() {
        return Ok(ConvertState::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("read {:?}", path))?;
    serde_json::from_str(&raw).with_context(|| format!("parse {:?}", path))
}

fn save_state(path: &Path, state: &ConvertState) -> Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    fs::rename(&tmp, path).with_context(|| format!("rename {:?} -> {:?}", tmp, path))?;
    Ok(())
}

/// Extract one root-to-leaf path for each reply branch. Each returned path has
/// >=2 messages and >=2 distinct authors. Reply targets missing from the TSV
/// (deleted messages, pre-scrape history) simply terminate their path.
fn extract_reply_chains(messages: &[Msg]) -> Vec<Vec<Msg>> {
    let by_id: HashMap<u64, &Msg> = messages.iter().map(|m| (m.id, m)).collect();

    // A reply component is a tree. Flattening the whole component by time
    // fabricates adjacency between sibling replies. Shared ancestors are
    // intentionally repeated so every emitted turn follows its actual edge.
    let parent_ids: HashSet<u64> = messages
        .iter()
        .filter_map(|m| m.reply_to.filter(|id| by_id.contains_key(id)))
        .collect();
    let mut leaves: Vec<&Msg> = messages
        .iter()
        .filter(|m| {
            let has_known_parent = m.reply_to.is_some_and(|id| by_id.contains_key(&id));
            has_known_parent && !parent_ids.contains(&m.id)
        })
        .collect();
    leaves.sort_by_key(|m| m.id);

    let mut chains = Vec::new();
    for leaf in leaves {
        let mut reversed = Vec::new();
        let mut seen = HashSet::new();
        let mut current = Some(leaf);
        while let Some(message) = current {
            // Valid replies point backward, but corrupt input must not loop.
            if !seen.insert(message.id) {
                break;
            }
            reversed.push(message.clone());
            current = message.reply_to.and_then(|id| by_id.get(&id).copied());
        }
        reversed.reverse();
        if reversed.len() >= 2
            && reversed
                .iter()
                .map(|m| m.author_id)
                .collect::<HashSet<_>>()
                .len()
                >= 2
        {
            chains.push(reversed);
        }
    }
    chains
}

/// Build the emitted turn list for one section: per-section PERSON_N
/// assignment (bot → 0, others → 1, 2, ... in first-seen order) plus
/// merging of consecutive same-author messages into one turn.
fn build_section_turns(section: &[Msg], names: &HashMap<u64, String>) -> Vec<(u32, String)> {
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
        let content = sanitize_for_corpus(&m.content, names);
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
    let mut seen: HashSet<u64> = HashSet::new();
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
        if !seen.insert(id) {
            // The scraper's append is not idempotent across a crash between
            // batch write and cursor write; dedup by snowflake here.
            continue;
        }
        let ts_secs = parse_rfc3339_to_secs(parts[1]).unwrap_or_else(|| {
            // Fallback: derive from snowflake epoch.
            snowflake_to_secs(id)
        });
        let author_id: u64 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let display_name = parts[3].to_string();
        let reply_to: Option<u64> = parts[4].parse().ok();
        let content = unescape_tsv(parts[5..].join("\t").as_str());
        out.push(Msg {
            id,
            ts_secs,
            author_id,
            display_name,
            reply_to,
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

/// Lowercased alphanumeric form of a display name, used for `@name`
/// mention tokens. Returns None when nothing survives (emoji-only names).
fn normalize_handle(name: &str) -> Option<String> {
    let handle: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    if handle.is_empty() { None } else { Some(handle) }
}

/// Rewrite Discord's `<...>` reference syntax in RAW content (before the
/// angle-bracket sanitize): user mentions become learnable `@name` tokens,
/// custom emoji become `__EMOJI__`, everything else structured becomes
/// `__MENTION__`. Spans that don't look like Discord refs are left for the
/// bracket sanitize to defang.
fn rewrite_discord_refs(s: &str, names: &HashMap<u64, String>) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = match after.find('>') {
            Some(c) => c,
            None => {
                out.push_str(&rest[open..]);
                return out;
            }
        };
        let inner = &after[..close];
        let replacement: Option<String> = if let Some(id_str) =
            inner.strip_prefix("@!").or_else(|| inner.strip_prefix('@'))
        {
            if id_str.starts_with('&') {
                // Role ping — no role-name map; neutralize.
                id_str[1..]
                    .parse::<u64>()
                    .ok()
                    .map(|_| "__MENTION__".to_string())
            } else {
                id_str.parse::<u64>().ok().map(|id| {
                    names
                        .get(&id)
                        .and_then(|n| normalize_handle(n))
                        .map(|h| format!("@{h}"))
                        .unwrap_or_else(|| "__MENTION__".to_string())
                })
            }
        } else if let Some(id_str) = inner.strip_prefix('#') {
            id_str.parse::<u64>().ok().map(|_| "__MENTION__".to_string())
        } else if inner.starts_with("t:") {
            Some("__MENTION__".to_string())
        } else if inner.starts_with(':') || inner.starts_with("a:") {
            // Custom emoji `<:name:id>` / animated `<a:name:id>`.
            if inner.matches(':').count() >= 2 {
                Some("__EMOJI__".to_string())
            } else {
                None
            }
        } else {
            None
        };
        match replacement {
            Some(r) => {
                out.push_str(&r);
                rest = &after[close + 1..];
            }
            None => {
                // Not a recognized ref; keep the '<' literally (it will be
                // defanged to '(' by the bracket sanitize) and continue
                // scanning right after it.
                out.push('<');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn sanitize_for_corpus(s: &str, names: &HashMap<u64, String>) -> String {
    // Step 0: turn Discord `<...>` refs into `@name` / placeholders while
    // the raw syntax is still intact.
    let derefed = rewrite_discord_refs(s, names);
    // Step 1: replace control chars + angle brackets so user content
    // can't impersonate a PERSON tag or break newline framing.
    let prepped: String = derefed
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

    fn no_names() -> HashMap<u64, String> {
        HashMap::new()
    }

    #[test]
    fn sanitize_strips_brackets() {
        assert_eq!(
            sanitize_for_corpus("<PERSON_3>hi", &no_names()),
            "(PERSON_3)hi"
        );
        assert_eq!(sanitize_for_corpus("a\nb\tc", &no_names()), "a b c");
    }

    #[test]
    fn mentions_become_names_when_known() {
        let mut names = HashMap::new();
        names.insert(42u64, "Wal Nutty2".to_string());
        assert_eq!(
            sanitize_for_corpus("hey <@42> look", &names),
            "hey @walnutty2 look"
        );
        assert_eq!(
            sanitize_for_corpus("hey <@!42> look", &names),
            "hey @walnutty2 look"
        );
        // Unknown id degrades to the neutral placeholder.
        assert_eq!(
            sanitize_for_corpus("hey <@999> look", &names),
            "hey __MENTION__ look"
        );
    }

    #[test]
    fn roles_channels_timestamps_neutralized() {
        assert_eq!(
            sanitize_for_corpus("ping <@&123> in <#456> at <t:1700000000:R>", &no_names()),
            "ping __MENTION__ in __MENTION__ at __MENTION__"
        );
    }

    #[test]
    fn custom_emoji_become_placeholder() {
        assert_eq!(
            sanitize_for_corpus("lol <:kekw:12345> nice <a:party:67>", &no_names()),
            "lol __EMOJI__ nice __EMOJI__"
        );
    }

    #[test]
    fn normalize_handle_examples() {
        assert_eq!(normalize_handle("Wal Nutty2"), Some("walnutty2".into()));
        assert_eq!(normalize_handle("✨✨"), None);
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
            reply_to: None,
            content: content.to_string(),
        }
    }

    fn mk_reply(id: u64, author: u64, content: &str, reply_to: u64) -> Msg {
        Msg {
            reply_to: Some(reply_to),
            ..mk(id, author, content)
        }
    }

    #[test]
    fn build_section_turns_assigns_bot_to_zero() {
        let section = vec![
            mk(1, 100, "hi"),
            mk(2, 200, "hello"),
            mk(3, BOT_DISCORD_ID, "greetings"),
        ];
        let turns = build_section_turns(&section, &no_names());
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
        let turns = build_section_turns(&section, &no_names());
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
            mk(4, 200, "<<<>>>"), // sanitizes to non-empty punctuation
        ];
        let turns = build_section_turns(&section, &no_names());
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
        let a = build_section_turns(&section_a, &no_names());
        let b = build_section_turns(&section_b, &no_names());
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
        let turns = build_section_turns(&section, &no_names());
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

    #[test]
    fn reply_chain_extraction_groups_transitively() {
        // 1 <- 3 <- 5 form a chain across interleaved chatter (2, 4).
        let msgs = vec![
            mk(1, 100, "does anyone know rust"),
            mk(2, 200, "unrelated chatter"),
            mk_reply(3, 300, "yes I do", 1),
            mk(4, 200, "more chatter"),
            mk_reply(5, 100, "can you help me", 3),
        ];
        let chains = extract_reply_chains(&msgs);
        assert_eq!(chains.len(), 1);
        let ids: Vec<u64> = chains[0].iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![1, 3, 5]);
    }

    #[test]
    fn reply_chain_requires_two_authors() {
        // Self-reply thread: single author, stays ambient.
        let msgs = vec![
            mk(1, 100, "note to self"),
            mk_reply(2, 100, "more notes", 1),
        ];
        assert!(extract_reply_chains(&msgs).is_empty());
    }

    #[test]
    fn reply_to_missing_target_stays_ambient() {
        // Reply to a deleted/unscraped message: no chain forms.
        let msgs = vec![mk_reply(2, 100, "orphan reply", 999), mk(3, 200, "hi")];
        assert!(extract_reply_chains(&msgs).is_empty());
    }

    #[test]
    fn forked_replies_become_separate_paths() {
        // Sibling replies are separate conversations, not sequential turns.
        let msgs = vec![
            mk(1, 100, "hot take"),
            mk_reply(2, 200, "disagree", 1),
            mk_reply(3, 300, "agree", 1),
        ];
        let chains = extract_reply_chains(&msgs);
        assert_eq!(chains.len(), 2);
        assert_eq!(chains[0].iter().map(|m| m.id).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(chains[1].iter().map(|m| m.id).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn nested_fork_preserves_each_reply_ancestry() {
        // A chronological flatten would fabricate 2 -> 3 -> 4, although 4
        // explicitly replies to 2.
        let msgs = vec![
            mk(1, 100, "root"),
            mk_reply(2, 200, "branch a", 1),
            mk_reply(3, 300, "branch b", 1),
            mk_reply(4, 100, "follow a", 2),
        ];
        let chains = extract_reply_chains(&msgs);
        assert_eq!(chains.len(), 2);
        assert_eq!(chains[0].iter().map(|m| m.id).collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(chains[1].iter().map(|m| m.id).collect::<Vec<_>>(), vec![1, 2, 4]);
    }
}
