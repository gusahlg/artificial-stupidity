#!/usr/bin/env bash
# Promote the current training artifacts to serving.
#
# The live server reads a PINNED pair — model.serving.bin +
# data/dialogs.serving.txt — because the model's vocab hash must match
# the corpus it was trained on. The trainer meanwhile churns model.bin
# against the (weekly-refreshed) data/dialogs.txt. This script snapshots
# the two together, atomically, and the sighurt-llm.path watcher
# restarts the server on the model rename.
#
# Order matters: corpus first, model last — the watcher fires on the
# model file, so when serve restarts, the corpus snapshot is already in
# place and consistent.
#
# Usage: promote_model.sh [model-file]     (default: model.bin — pass
#        model.bin.best to promote the best-val checkpoint instead)
set -euo pipefail
cd "$(dirname "$0")/.."

SRC_MODEL="${1:-model.bin}"

if [ ! -f "$SRC_MODEL" ]; then
    echo "promote_model: $SRC_MODEL not found" >&2
    exit 1
fi
if [ ! -f data/dialogs.txt ]; then
    echo "promote_model: data/dialogs.txt not found" >&2
    exit 1
fi

# Preflight: refuse to promote a model that won't load against the corpus
# we're about to pin with it. serve rebuilds the vocab from the corpus and
# the v4/v5 vocab hash refuses a mismatch — so if the corpus was refreshed
# (vocab reordered) after this model was trained, promoting would leave the
# path-watcher restarting serve in a crash loop. Catch it here instead.
SERVE_BIN="${SERVE_BIN:-target/release/serve}"
if [ -x "$SERVE_BIN" ]; then
    echo "promote_model: preflight — checking $SRC_MODEL loads against data/dialogs.txt"
    if ! SIGHURT_CHECK=1 SIGHURT_MODEL="$SRC_MODEL" SIGHURT_CORPUS=data/dialogs.txt "$SERVE_BIN"; then
        echo "promote_model: PREFLIGHT FAILED — $SRC_MODEL does not match the current corpus." >&2
        echo "The corpus vocab was likely reordered by a refresh after this model was trained." >&2
        echo "Train a fresh model on the current data/dialogs.txt before promoting." >&2
        exit 1
    fi
else
    echo "promote_model: WARN $SERVE_BIN not built — skipping vocab-match preflight" >&2
fi

echo "promote_model: snapshotting corpus -> data/dialogs.serving.txt"
cp data/dialogs.txt data/dialogs.serving.txt.tmp
mv data/dialogs.serving.txt.tmp data/dialogs.serving.txt
# Drop a stale serving cache so serve re-parses the fresh snapshot.
rm -f data/dialogs.serving.bin

echo "promote_model: promoting $SRC_MODEL -> model.serving.bin"
cp "$SRC_MODEL" model.serving.bin.tmp
mv model.serving.bin.tmp model.serving.bin

echo "promote_model: done — sighurt-llm.path will restart the server."
