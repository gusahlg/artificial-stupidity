#!/usr/bin/env bash
# Full corpus rebuild from all raw sources. Use when the converter's
# rules changed (e.g. the reply-chain/@name-mention overhaul) and the
# improvement should apply to ALL history, not just future messages —
# or to bootstrap the incremental cursor state for refresh_corpus.sh.
#
# Build order matters (convert_discord --full TRUNCATES dialogs.txt):
#   1. convert_discord --full   — every scraped Discord message
#   2. snapshot + clean Discord — preserves a Discord-only, auditable corpus
#   3. ingest_oasst             — OpenAssistant trees (needs the dump; see below)
#   4. inject_seed x3           — hand-curated persona pairs; x3 because
#                                 clean_corpus rule 7 dedup-caps at 3
#   5. clean_corpus             — the 14 cleaning rules + shuffle
#   6. rebuild_vocab
#
# OASST dump: pass the path as $1 or set OASST_JSONL. Download with:
#   curl -LO https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz
# If absent, the OASST step is SKIPPED with a loud warning (the corpus
# then loses the assistant-dialog bulk the docs call the biggest data
# lever — only do that deliberately).
#
# Serving is unaffected (pinned snapshot; see scripts/promote_model.sh).
# After this, train fresh (vocab order changed, old model.bin will not
# load against the new vocab — that is expected and by design).
set -euo pipefail
cd "$(dirname "$0")/.."

BOT_DATA="${BOT_DATA:-$HOME/discord-bot/data/channels}"
OASST_JSONL="${1:-${OASST_JSONL:-}}"
BIN=target/release

for tool in convert_discord inject_seed clean_corpus rebuild_vocab; do
    if [ ! -x "$BIN/$tool" ]; then
        echo "rebuild_full_corpus: $BIN/$tool missing — run 'nix develop -c cargo build --release' first" >&2
        exit 1
    fi
done

# Fail before syncing or truncating anything. On NixOS, python3 may only be
# present inside `nix develop`; discovering that after `convert_discord --full`
# leaves a valid but incomplete Discord-only corpus.
if [ -n "$OASST_JSONL" ] && [ -f "$OASST_JSONL" ] && ! command -v python3 >/dev/null; then
    echo "rebuild_full_corpus: python3 is required for OASST ingestion" >&2
    echo "Run: nix develop -c scripts/rebuild_full_corpus.sh '$OASST_JSONL'" >&2
    exit 1
fi

if [ -d "$BOT_DATA" ]; then
    echo "rebuild_full_corpus: syncing TSVs from $BOT_DATA"
    mkdir -p data/discord
    rsync -a --exclude '*.cursor' "$BOT_DATA/" data/discord/
else
    echo "rebuild_full_corpus: WARN bot data dir $BOT_DATA not found; using existing data/discord" >&2
fi

if [ -f data/dialogs.txt ]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    backup="data/dialogs.txt.pre-rebuild-$stamp"
    if [ -e "$backup" ]; then
        backup="$backup.$$"
    fi
    cp data/dialogs.txt "$backup"
    echo "rebuild_full_corpus: previous corpus kept at $backup"
fi

"$BIN/convert_discord" --full

# Freeze the Discord-only stage before assistant corpora are appended.  The
# modern transformer trainer consumes this file directly so it can enforce a
# measured Discord share; using the merged dialogs.txt made that impossible to
# prove because source labels are intentionally absent from the canonical
# PERSON_N format.  Keep the raw snapshot too so cleaner changes are auditable.
cp data/dialogs.txt data/dialogs.discord.raw.txt
"$BIN/clean_corpus" data/dialogs.discord.raw.txt data/dialogs.discord.txt
echo "rebuild_full_corpus: wrote cleaned Discord-only corpus data/dialogs.discord.txt"

if [ -n "$OASST_JSONL" ] && [ -f "$OASST_JSONL" ]; then
    echo "rebuild_full_corpus: ingesting OASST from $OASST_JSONL"
    python3 scripts/ingest_oasst.py "$OASST_JSONL"
else
    echo "rebuild_full_corpus: WARN no OASST dump (pass path as \$1) — skipping the corpus's largest data source!" >&2
fi

for i in 1 2 3; do
    "$BIN/inject_seed"
done

"$BIN/clean_corpus"
"$BIN/rebuild_vocab"

echo "rebuild_full_corpus: done. Corpus stats:"
grep -c '^<SEC>' data/dialogs.discord.txt | sed 's/^/  Discord-only sections: /'
grep -c '^<SEC>' data/dialogs.txt | sed 's/^/  sections: /'
wc -l < vocab.txt | sed 's/^/  vocab tokens: /'
echo "rebuild_full_corpus: NOTE vocab order changed — train a fresh model before promoting."
