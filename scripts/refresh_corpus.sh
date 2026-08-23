#!/usr/bin/env bash
# Weekly training-corpus refresh (run by sighurt-refresh.timer).
#
# Pulls the Discord bot's scraped TSVs into this repo's data/discord
# mirror, converts only the NEW messages into corpus sections (the
# converter keeps per-channel cursors in data/discord/.convert-state.json
# and understands reply chains), then re-cleans the corpus and rebuilds
# the vocab.
#
# Serving is deliberately UNAFFECTED: the live server reads the pinned
# snapshot pair (model.serving.bin + data/dialogs.serving.txt, see
# scripts/promote_model.sh). This job only advances the TRAINING corpus;
# the next training run picks it up, and promotion is a separate,
# deliberate step.
#
# Requirements: release binaries already built (cargo build --release);
# the bot's live data under $BOT_DATA. The bot process may be appending
# to the TSVs while we copy — a torn final line is skipped by the
# converter's >=6-column check and picked up next week.
set -euo pipefail
cd "$(dirname "$0")/.."

BOT_DATA="${BOT_DATA:-$HOME/discord-bot/data/channels}"
BIN=target/release

for tool in convert_discord clean_corpus rebuild_vocab; do
    if [ ! -x "$BIN/$tool" ]; then
        echo "refresh_corpus: $BIN/$tool missing — run 'nix develop -c cargo build --release' first" >&2
        exit 1
    fi
done

if [ ! -d "$BOT_DATA" ]; then
    echo "refresh_corpus: bot data dir $BOT_DATA not found" >&2
    exit 1
fi

echo "refresh_corpus: syncing TSVs from $BOT_DATA"
mkdir -p data/discord
rsync -a --exclude '*.cursor' "$BOT_DATA/" data/discord/

if [ ! -f data/discord/.convert-state.json ]; then
    echo "refresh_corpus: no converter cursor state — refusing to run incrementally." >&2
    echo "Bootstrap once with scripts/rebuild_full_corpus.sh (preferred) or" >&2
    echo "'$BIN/convert_discord --init-cursors' (adopt current TSVs as already-converted)." >&2
    exit 1
fi

# One compressed rollback copy. The corpus is derived from the TSVs (the real
# archive), so a single gzip'd snapshot is enough insurance and ~3x smaller
# than the plain text; stale uncompressed backups are cleaned up here too.
gzip -c data/dialogs.txt > "data/dialogs.txt.pre-refresh.gz"
rm -f data/dialogs.txt.pre-refresh data/dialogs.txt.pre-rebuild-* data/dialogs.txt.session-*

"$BIN/convert_discord"
"$BIN/clean_corpus"
"$BIN/rebuild_vocab"

echo "refresh_corpus: done. Corpus stats:"
grep -c '^<SEC>' data/dialogs.txt | sed 's/^/  sections: /'
wc -l < vocab.txt | sed 's/^/  vocab tokens: /'
echo "refresh_corpus: previous corpus kept at data/dialogs.txt.pre-refresh.gz"
