#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <final-hf-dir> <artifact-dir>" >&2
    exit 2
fi

HF_DIR=$(realpath "$1")
ARTIFACT_DIR=$(realpath -m "$2")
LLAMA_CPP_REV=${LLAMA_CPP_REV:-54ee5ee643f29abba6852903ddfdb688c2361b5b}
TOOLS_ROOT=${TOOLS_ROOT:-/workspace/tools}
LLAMA_CPP="$TOOLS_ROOT/llama.cpp"

mkdir -p "$TOOLS_ROOT" "$ARTIFACT_DIR"
if [ ! -d "$LLAMA_CPP/.git" ]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
fi
git -C "$LLAMA_CPP" fetch origin "$LLAMA_CPP_REV"
git -C "$LLAMA_CPP" checkout --detach "$LLAMA_CPP_REV"

python3 -m pip install --break-system-packages -r "$LLAMA_CPP/requirements.txt"
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$HF_DIR" \
    --outfile "$ARTIFACT_DIR/model.f16.gguf" \
    --outtype f16

cp "$HF_DIR/tokenizer.json" "$ARTIFACT_DIR/tokenizer.json"
cp "$HF_DIR/tokenizer_config.json" "$ARTIFACT_DIR/tokenizer_config.json"
printf '%s\n' "$LLAMA_CPP_REV" > "$ARTIFACT_DIR/llama-cpp-revision.txt"
(cd "$ARTIFACT_DIR" && sha256sum model.f16.gguf tokenizer.json tokenizer_config.json > SHA256SUMS)
test -s "$ARTIFACT_DIR/model.f16.gguf"
echo "GGUF_DONE $ARTIFACT_DIR/model.f16.gguf"
