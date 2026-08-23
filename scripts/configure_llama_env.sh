#!/usr/bin/env bash
# Preserve the existing API key and unrelated settings while switching the
# production service from the legacy model/corpus pair to transformer files.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd -P)
ENV_FILE=${1:-"$HOME/.config/sighurt-llm.env"}

test -f "$ENV_FILE" || { echo "missing environment file: $ENV_FILE" >&2; exit 1; }
awk -F= '
    $1 == "SIGHURT_API_KEY" && length(substr($0, index($0, "=") + 1)) >= 16 { found = 1 }
    END { exit !found }
' "$ENV_FILE" || {
    echo "refusing to rewrite $ENV_FILE: no API key of at least 16 characters" >&2
    exit 1
}

TEMP_FILE=$(mktemp "$(dirname "$ENV_FILE")/.sighurt-llm.env.tmp.XXXXXX")
cleanup() {
    if test -e "$TEMP_FILE"; then
        rm -f -- "$TEMP_FILE"
    fi
}
trap cleanup EXIT

awk \
    -v model="$REPO_ROOT/model.serving.f16.gguf" \
    -v tokenizer="$REPO_ROOT/tokenizer.serving.json" '
    function setting(key, value) {
        if (!seen[key]++) print key "=" value
    }
    /^[[:space:]]*SIGHURT_MODEL=/              { setting("SIGHURT_MODEL", model); next }
    /^[[:space:]]*SIGHURT_TOKENIZER=/          { setting("SIGHURT_TOKENIZER", tokenizer); next }
    /^[[:space:]]*SIGHURT_CONTEXT_TOKENS=/     { setting("SIGHURT_CONTEXT_TOKENS", "2048"); next }
    /^[[:space:]]*SIGHURT_MAX_NEW_TOKENS=/     { setting("SIGHURT_MAX_NEW_TOKENS", "64"); next }
    /^[[:space:]]*SIGHURT_REQUEST_TIMEOUT=/    { setting("SIGHURT_REQUEST_TIMEOUT", "75"); next }
    /^[[:space:]]*SIGHURT_TEMPERATURE=/        { setting("SIGHURT_TEMPERATURE", "0"); next }
    /^[[:space:]]*SIGHURT_TOP_P=/              { setting("SIGHURT_TOP_P", "1"); next }
    /^[[:space:]]*SIGHURT_TOP_K=/              { setting("SIGHURT_TOP_K", "1"); next }
    /^[[:space:]]*SIGHURT_REPETITION_PENALTY=/ { setting("SIGHURT_REPETITION_PENALTY", "1.08"); next }
    /^[[:space:]]*ML_DEVICE=/                   { setting("ML_DEVICE", "discrete"); next }
    /^[[:space:]]*SIGHURT_CORPUS=/              { next }
                                                    { print }
    END {
        setting("SIGHURT_MODEL", model)
        setting("SIGHURT_TOKENIZER", tokenizer)
        setting("SIGHURT_CONTEXT_TOKENS", "2048")
        setting("SIGHURT_MAX_NEW_TOKENS", "64")
        setting("SIGHURT_REQUEST_TIMEOUT", "75")
        setting("SIGHURT_TEMPERATURE", "0")
        setting("SIGHURT_TOP_P", "1")
        setting("SIGHURT_TOP_K", "1")
        setting("SIGHURT_REPETITION_PENALTY", "1.08")
        setting("ML_DEVICE", "discrete")
    }
' "$ENV_FILE" > "$TEMP_FILE"

chmod 600 "$TEMP_FILE"
mv -- "$TEMP_FILE" "$ENV_FILE"
trap - EXIT
echo "configure_llama_env: updated $ENV_FILE without changing SIGHURT_API_KEY"
