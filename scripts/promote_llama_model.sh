#!/usr/bin/env bash
# Validate and atomically promote a transformer GGUF + tokenizer pair.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <artifact-dir> <expected-parent-sha256>" >&2
    exit 2
fi

ARTIFACT_DIR=$(realpath "$1")
EXPECTED_PARENT_SHA256=$2
if ! printf '%s\n' "$EXPECTED_PARENT_SHA256" | grep -Eq '^[0-9a-f]{64}$'; then
    echo "expected parent SHA-256 must be exactly 64 lowercase hex characters" >&2
    exit 2
fi
SOURCE_MODEL="$ARTIFACT_DIR/model.f16.gguf"
SOURCE_TOKENIZER="$ARTIFACT_DIR/tokenizer.json"
DEST_MODEL=${SIGHURT_SERVING_MODEL:-model.serving.f16.gguf}
DEST_TOKENIZER=${SIGHURT_SERVING_TOKENIZER:-tokenizer.serving.json}
SERVE_BIN=${SERVE_BIN:-scripts/run_serve_llama.sh}

test -s "$SOURCE_MODEL" || { echo "missing $SOURCE_MODEL" >&2; exit 1; }
test -s "$SOURCE_TOKENIZER" || { echo "missing $SOURCE_TOKENIZER" >&2; exit 1; }
test -x "$SERVE_BIN" || { echo "missing executable $SERVE_BIN" >&2; exit 1; }

if [ -f "$ARTIFACT_DIR/SHA256SUMS" ]; then
    echo "promote_llama_model: verifying artifact hashes"
    (cd "$ARTIFACT_DIR" && sha256sum --check SHA256SUMS)
fi

echo "promote_llama_model: verifying training provenance, losses, and semantic probes"
if command -v python3 >/dev/null 2>&1; then
    python3 training/verify_artifact.py \
        "$ARTIFACT_DIR" \
        --expected-parent-sha256 "$EXPECTED_PARENT_SHA256"
else
    # The headless NixOS host intentionally has no global Python. The flake's
    # pinned development shell supplies it without mutating the system profile.
    nix develop -c python3 training/verify_artifact.py \
        "$ARTIFACT_DIR" \
        --expected-parent-sha256 "$EXPECTED_PARENT_SHA256"
fi

echo "promote_llama_model: loading model and running a real generation preflight"
SIGHURT_CHECK=1 \
SIGHURT_MODEL="$SOURCE_MODEL" \
SIGHURT_TOKENIZER="$SOURCE_TOKENIZER" \
SIGHURT_CONTEXT_TOKENS="${SIGHURT_CONTEXT_TOKENS:-2048}" \
SIGHURT_MAX_NEW_TOKENS="${SIGHURT_MAX_NEW_TOKENS:-32}" \
SIGHURT_TEMPERATURE=0 \
SIGHURT_TOP_P=1 \
SIGHURT_TOP_K=1 \
"$SERVE_BIN"

# Preserve one recoverable transformer generation. Reflinks are cheap on
# supporting filesystems and transparently fall back to a normal copy.
if [ -f "$DEST_MODEL" ]; then
    cp --reflink=auto "$DEST_MODEL" "$DEST_MODEL.previous.tmp"
    mv "$DEST_MODEL.previous.tmp" "$DEST_MODEL.previous"
fi
if [ -f "$DEST_TOKENIZER" ]; then
    cp "$DEST_TOKENIZER" "$DEST_TOKENIZER.previous.tmp"
    mv "$DEST_TOKENIZER.previous.tmp" "$DEST_TOKENIZER.previous"
fi

# Tokenizer first, model last: sighurt-llm.path watches the model rename, so
# the service can never observe a new model with the old tokenizer.
cp "$SOURCE_TOKENIZER" "$DEST_TOKENIZER.tmp"
mv "$DEST_TOKENIZER.tmp" "$DEST_TOKENIZER"
cp --reflink=auto "$SOURCE_MODEL" "$DEST_MODEL.tmp"
mv "$DEST_MODEL.tmp" "$DEST_MODEL"

for metadata in \
    training-manifest.json \
    metrics.json \
    probes.json \
    sampling-evaluation.json \
    discord-corpus-manifest.json \
    aux-corpus-manifest.json
do
    if [ -f "$ARTIFACT_DIR/../$metadata" ]; then
        cp "$ARTIFACT_DIR/../$metadata" "${DEST_MODEL}.${metadata}.tmp"
        mv "${DEST_MODEL}.${metadata}.tmp" "${DEST_MODEL}.${metadata}"
    fi
done

echo "promote_llama_model: promoted $DEST_MODEL + $DEST_TOKENIZER"
