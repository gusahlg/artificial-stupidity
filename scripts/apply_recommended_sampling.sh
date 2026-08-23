#!/usr/bin/env bash
# Apply the sampling configuration recommended by evaluate_sampling.py to the
# live serving environment file. Greedy decoding is never an option here: it
# provably collapses long conversations into a self-reinforcing repetition
# spiral once the bot's own reply re-enters the rolling Discord context.
set -euo pipefail

REPORT=${1:?usage: apply_recommended_sampling.sh <sampling-evaluation.json> [env-file]}
ENV_FILE=${2:-"$HOME/.config/sighurt-llm.env"}

recommended=$(jq -r '.recommended // empty' "$REPORT")
if [ -z "$recommended" ]; then
    echo "apply_recommended_sampling: report has no recommended config" >&2
    exit 1
fi
read -r temperature top_p top_k repetition_penalty < <(
    jq -r --arg name "$recommended" '
        .results[] | select(.config.name == $name) | .config |
        "\(.temperature) \(.top_p) \(.top_k) \(.repetition_penalty)"
    ' "$REPORT"
)
if [ -z "${repetition_penalty:-}" ]; then
    echo "apply_recommended_sampling: config $recommended not found in report" >&2
    exit 1
fi

cp "$ENV_FILE" "$ENV_FILE.pre-sampling-update"
set_var() {
    local key=$1 value=$2
    if grep -q "^$key=" "$ENV_FILE"; then
        sed -i "s|^$key=.*|$key=$value|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    fi
}
set_var SIGHURT_TEMPERATURE "$temperature"
set_var SIGHURT_TOP_P "$top_p"
set_var SIGHURT_TOP_K "$top_k"
set_var SIGHURT_REPETITION_PENALTY "$repetition_penalty"

echo "apply_recommended_sampling: applied '$recommended'" \
     "(T=$temperature top_p=$top_p top_k=$top_k rp=$repetition_penalty) to $ENV_FILE"
echo "apply_recommended_sampling: restart sighurt-llm.service to take effect"
