#!/usr/bin/env bash
# Rebuild data/dialogs.txt with OASST capped at --limit N trees.
#
# Mechanics:
#   1. Snapshot current data/dialogs.txt to .pre-trim
#   2. Reset corpus to the pre-OASST baseline (Discord + early seed)
#   3. Ingest OASST with a tree cap
#   4. Re-inject all current seed pairs
#   5. Run clean_corpus (rules 1-14) + rebuild_vocab
#
# Usage: scripts/build_trim_corpus.sh [LIMIT]
# Default LIMIT = 1500 trees (vs full ingest of 3670).
set -euo pipefail

LIMIT="${1:-1500}"
REPO="/home/gusahlg/repos/hello_rust"
OASST_JSONL="/home/gusahlg/ai-data/openassistant/2023-04-12_oasst_ready.trees.jsonl.gz"

cd "$REPO"

if [[ ! -f data/dialogs.txt.pre-oasst ]]; then
  echo "ERROR: missing data/dialogs.txt.pre-oasst baseline" >&2
  exit 1
fi

echo "trim-build: snapshotting current corpus → data/dialogs.txt.pre-trim-${LIMIT}"
cp data/dialogs.txt "data/dialogs.txt.pre-trim-${LIMIT}"

echo "trim-build: restoring pre-OASST baseline ($(wc -l < data/dialogs.txt.pre-oasst) lines)"
cp data/dialogs.txt.pre-oasst data/dialogs.txt

echo "trim-build: ingesting OASST with --limit ${LIMIT}"
python3 scripts/ingest_oasst.py "$OASST_JSONL" --limit "$LIMIT"

echo "trim-build: injecting seed pairs"
./target/release/inject_seed

echo "trim-build: running clean_corpus (rules 1-14)"
./target/release/clean_corpus

echo "trim-build: rebuilding vocab"
./target/release/rebuild_vocab

echo "trim-build: done. Corpus: $(wc -l < data/dialogs.txt) lines."
