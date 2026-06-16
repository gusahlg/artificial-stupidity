#!/usr/bin/env bash
# Adversarial probe — prompts specifically designed to elicit OASST
# helper-template openers. Helps measure whether the bot has shaken
# off the "here are some" / "sure, i'd be happy" / "i am a language
# model" register. Use alongside the standard 9-prompt `probe_bot.sh`.
#
# Usage: scripts/probe_register.sh > docs/manual-test-<DATE>-register.md
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

# Direct anti-LM persona probes
probe "are you a language model"
probe "are you chatgpt"
probe "are you an llm"
probe "are you a bot"

# Helper-template trigger phrases — bot should NOT open with "here are some"
probe "give me 5 examples"
probe "list the steps to make pasta"
probe "what are some ways to learn rust"
probe "explain how a compiler works"
probe "write me a poem about cats"
probe "what's the best way to debug"
probe "how do i learn programming"
probe "tell me about machine learning"

# Story-prompt — bot should NOT continue with "once upon a time"
probe "tell me a story"
probe "write a story about a dragon"

# Greeting / identity sanity
probe "hi"
probe "good morning"
probe "tell me about yourself"
probe "what can you do"
