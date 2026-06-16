#!/usr/bin/env bash
# Manual probe — 9 standard prompts × 3 samples each.
# Usage: scripts/probe_bot.sh > docs/manual-test-<DATE>.md
# Or: SAMPLES=1 scripts/probe_bot.sh for a quick single-shot
set -euo pipefail

KEY="${SIGHURT_API_KEY:-$(grep ^SIGHURT_API_KEY= ~/.config/sighurt-llm.env | cut -d= -f2)}"
HOST="${HOST:-http://100.118.41.103:8088}"
N="${SAMPLES:-3}"

probe() {
  local prompt="$1"
  printf '\n>>> %s\n' "$prompt"
  for i in $(seq 1 "$N"); do
    local body
    body=$(python3 -c 'import json,sys; print(json.dumps({"input": sys.argv[1]}))' "$prompt")
    local raw
    raw=$(curl -sS -X POST "$HOST/chat" \
       -H 'Content-Type: application/json' \
       -H "X-API-Key: $KEY" \
       -d "$body" 2>&1)
    local out
    out=$(printf '%s' "$raw" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("reply") or d.get("response") or d.get("text") or json.dumps(d))' 2>/dev/null || echo "RAW: $raw")
    printf '  [%d] %s\n' "$i" "$out"
  done
}

probe "hi"
probe "hello"
probe "how are you"
probe "what are you working on"
probe "i'm building a chatbot in rust"
probe "tell me a joke"
probe "good morning"
probe "do you like games"
probe "are you a language model"
