#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/artificial-stupidity}
DATA_DIR=${DATA_DIR:-/workspace/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/outputs/super-sighurt-sig-v2-20260822}
PARENT_MODEL=${PARENT_MODEL:-/workspace/outputs/super-sighurt-1.1b/final-hf}
PARENT_SHA256=${PARENT_SHA256:-}
HF_HOME=${HF_HOME:-/workspace/hf-cache}
export HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

status_file="$OUTPUT_ROOT/run-status.txt"
finish() {
    code=$?
    printf 'exit_code=%s finished_utc=%s\n' "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
}
trap finish EXIT

mkdir -p "$OUTPUT_ROOT" "$HF_HOME"
cd "$REPO_ROOT"

if [ -z "$PARENT_SHA256" ]; then
    echo "run_on_runpod: PARENT_SHA256 is required; refusing an unproven continuation parent" >&2
    exit 2
fi
if [ ! -d "$PARENT_MODEL" ]; then
    echo "run_on_runpod: parent HF directory not found: $PARENT_MODEL" >&2
    exit 2
fi

python3 -m pip install --break-system-packages -r training/requirements-runpod.txt
python3 training/preflight.py \
    --discord-train "$DATA_DIR/discord-train.jsonl" \
    --discord-validation "$DATA_DIR/discord-validation.jsonl" \
    --discord-manifest "$DATA_DIR/discord-manifest.json" \
    --aux-train "$DATA_DIR/aux-train.jsonl" \
    --aux-validation "$DATA_DIR/aux-validation.jsonl" \
    --aux-manifest "$DATA_DIR/aux-manifest.json" \
    --base-model "$PARENT_MODEL" \
    --expected-parent-sha256 "$PARENT_SHA256" \
    --output-dir "$OUTPUT_ROOT" \
    --require-cuda \
    --cuda-smoke
python3 training/train_llama.py \
    --discord-train "$DATA_DIR/discord-train.jsonl" \
    --discord-validation "$DATA_DIR/discord-validation.jsonl" \
    --aux-train "$DATA_DIR/aux-train.jsonl" \
    --aux-validation "$DATA_DIR/aux-validation.jsonl" \
    --base-model "$PARENT_MODEL" \
    --expected-parent-sha256 "$PARENT_SHA256" \
    --require-local-parent \
    --output-dir "$OUTPUT_ROOT" \
    "$@"

training/convert_to_gguf.sh "$OUTPUT_ROOT/final-hf" "$OUTPUT_ROOT/artifact"
cp "$DATA_DIR/discord-manifest.json" "$OUTPUT_ROOT/discord-corpus-manifest.json"
cp "$DATA_DIR/aux-manifest.json" "$OUTPUT_ROOT/aux-corpus-manifest.json"
python3 training/verify_artifact.py \
    "$OUTPUT_ROOT/artifact" \
    --expected-parent-sha256 "$PARENT_SHA256"
echo "RUNPOD_TRAINING_DONE $OUTPUT_ROOT/artifact/model.f16.gguf"
