#!/usr/bin/env bash
# Restart sighurt-llm if a real generation fails.
#
# The NVK/GTX 1650 Vulkan stack can lose the logical device under sustained
# load. serve_llama survives the loss as a process — /healthz still answers —
# but every generation returns "generation failed" until the GPU context is
# recreated, so systemd's Restart=always never fires. This probe exercises the
# only path that actually breaks (token generation) and restarts the service
# when it fails. It runs from a systemd timer, uses only curl, and never
# restarts a service that is merely busy (a queue-full 503 or timeout counts
# as alive).
set -u

BIND=${SIGHURT_BIND:-127.0.0.1:8088}
KEY=${SIGHURT_API_KEY:-}
[ -n "$KEY" ] || { echo "watchdog: SIGHURT_API_KEY missing" >&2; exit 1; }

# Not running at all? Leave recovery to Restart=always.
systemctl --user is-active --quiet sighurt-llm.service || exit 0

body=$(curl -s -m 90 -X POST "http://$BIND/chat" \
    -H "X-API-Key: $KEY" \
    -H "Content-Type: application/json" \
    -d '{"user":"watchdog","input":"ping","context":[],"web_results":[]}')

case "$body" in
    *'"reply"'*)
        exit 0
        ;;
    "generation failed")
        echo "watchdog: generation failed — restarting sighurt-llm.service"
        systemctl --user restart sighurt-llm.service
        ;;
    *)
        # Busy (queue full / timeout) or transient transport error: alive
        # enough. The next timer tick will re-check.
        echo "watchdog: inconclusive probe result: ${body:0:120}"
        ;;
esac
