# Discord AI setup (single machine)

Everything runs on one desktop: the Discord bot (`~/discord-bot`, twilight),
the transformer HTTP server (`serve_llama` from this repo), and the weekly
archive refresh. Full-parameter training runs separately on a temporary
Runpod H100 or H200; only a validated f16 GGUF is returned to the desktop. The old split
(bot on the Raspberry Pi, server reached over Tailscale) is retired;
the Pi's `discord-bot.service` should stay disabled.

## Architecture

```
Discord gateway
      │
      ▼
discord-bot (systemd --user discord-bot.service, ~/discord-bot)
  │     │ logs EVERY message (incl. other bots) to data/channels/*.tsv
  │     │ + daily in-process catch-up scrape heals offline gaps
  │     ▼
  │  POST /chat  (127.0.0.1:8088, X-API-Key)
  │     with structured recent messages + explicit reply target
  ▼
serve_llama (systemd --user sighurt-llm.service, ~/artificial-stupidity)
  tensor-ash Vulkan inference from model.serving.f16.gguf
      ▲
      │ hash + real-generation gate, then atomic promotion
      │
Runpod H100/H200 full SFT ◄── leakage-safe JSONL ◄── data/dialogs.txt
                                                ▲
                                                │ all Discord TSV history
                                                │ + OASST + seed dialogs
```

The legacy `serve` + `model.serving.bin` path remains available for rollback,
but production uses the transformer artifact and its matching tokenizer. The
watched model rename happens only after both files have passed checks.

## The pieces

| Unit (systemd --user) | What it does |
|---|---|
| `discord-bot.service` | The bot. WorkingDirectory `~/discord-bot`, env from `~/discord-bot/.env`. |
| `sighurt-llm.service` | The LLM server. WorkingDirectory `~/artificial-stupidity`, env from `~/.config/sighurt-llm.env`. |
| `sighurt-llm.path` + `sighurt-llm-reload.service` | Watch `model.serving.f16.gguf`; restart the server after atomic promotion. |
| `sighurt-refresh.timer` + `.service` | Weekly (Sun 04:37) corpus refresh: rsync bot TSVs → incremental convert → clean → rebuild vocab. |

`~/.config/sighurt-llm.env` (mode 600):

```ini
SIGHURT_BIND=127.0.0.1:8088
SIGHURT_API_KEY=<same value as LLM_API_KEY in ~/discord-bot/.env>
SIGHURT_MODEL=/home/gusahlg/artificial-stupidity/model.serving.f16.gguf
SIGHURT_TOKENIZER=/home/gusahlg/artificial-stupidity/tokenizer.serving.json
SIGHURT_CONTEXT_TOKENS=2048
SIGHURT_MAX_NEW_TOKENS=96
```

Everything is loopback-only; nothing binds Tailscale anymore. The
server refuses to start unless the model and tokenizer match and the GPU has
the required Vulkan f16 features. Sampling knobs include
`SIGHURT_TEMPERATURE` (default 0.65), `SIGHURT_TOP_P` (0.9),
`SIGHURT_TOP_K` (40), and `SIGHURT_REPETITION_PENALTY` (1.08).

## /chat wire protocol

POST `/chat` with `X-API-Key`. Scalar values are strings; only `input` is
required and unknown fields are ignored:

```json
{
  "channel_id": "…", "user": "display name", "user_id": "snowflake",
  "user_is_bot": "true|false",
  "input": "text with <@id> mentions already resolved to @name",
  "context": [
    {"message_id":"…", "user":"…", "user_id":"…", "text":"…",
     "is_bot":"false", "is_self":"false", "reply_to_message_id":"…"}
  ],
  "reply_to_message_id": "…",
  "reply_to_user": "…", "reply_to_user_id": "…", "reply_to_text": "…",
  "reply_to_is_bot": "true|false", "reply_to_is_self": "true|false"
}
```

Response: `{"reply":"…"}`. The bot fetches recent channel messages through
Discord's API, excludes the triggering message and webhooks, resolves mentions,
orders context oldest first, and retains message/reply IDs. The server renders
that structure with stable numbered speakers. If the prompt exceeds its token
budget, it removes oldest ambient messages first; the current message and
explicit reply target are never discarded. The bot resolves safe `@name`
references back into real `<@id>` pings under a strict allowed-mentions list.

## Data flow (training)

1. The bot logs every message (humans *and* bots) to
   `~/discord-bot/data/channels/<guild|dm>/<channel>.tsv` — 6 columns:
   id, timestamp, author id, display name, **reply-to id**, content.
   Backfill + daily forward catch-up cover offline gaps, threads and
   forum posts included.
2. `scripts/refresh_corpus.sh` (weekly timer) rsyncs the TSVs to
   `data/discord/` and runs `convert_discord` **incrementally** (per-
   channel cursors in `data/discord/.convert-state.json`): only new
   messages become new `<SEC>` sections, appended to `data/dialogs.txt`,
   then `clean_corpus` + `rebuild_vocab`.
3. `convert_discord` is reply-aware: every reply-tree branch becomes its own
   root-to-leaf section. Sibling replies are never placed next to each other as
   though one answered the other. `<@id>` mentions become learnable display
   names.
4. `scripts/rebuild_full_corpus.sh [oasst.jsonl.gz]` rebuilds both the cleaned
   Discord-only snapshot and the legacy merged corpus. The transformer exporter
   deliberately reads the Discord-only snapshot; general replay stays separately
   labelled and cannot dilute the provenance count.
5. `training/export_sft.py` makes a deterministic section-level 95/5 split.
   Bot-involved dialogs train only the bot's `PERSON_0` responses; human-only
   dialogs supply next-speaker style examples. No channel or user IDs leave the
   machine in the SFT files.
6. `training/preflight.py` verifies hashes, revisions, templates, dependency
   versions, split isolation, CUDA/bf16, and a real optimizer step before the
   paid run continues.
7. `training/build_auxiliary_data.py` adds 30,000 test-scored OpenCodeInstruct
   examples, 1,400 revision-attributed Wikipedia knowledge rows, 1,400 matching
   live-search grounding rows (including prompt-injection negatives), and 140
   reviewed original Sig demonstrations. Teacher-generated drafts are excluded.
8. `training/train_llama.py` mixes two copies of the Discord set with 25,000
   pinned general-conversation examples and the labelled auxiliary sets. The
   compact curated persona set is repeated five times while Discord still holds
   the majority exposure. Three full bf16 epochs retain checkpoints and enforce
   source-specific loss, semantic, context, web-grounding, and diversity gates.
   Loss is macro-averaged per completion so long code answers cannot silently
   overpower the 70% Discord example share.
9. `scripts/promote_llama_model.sh artifact/ <expected-parent-sha256>` verifies
   hashes and provenance against the external parent proof, then performs real
   tensor-ash generation on the deployment GPU before replacing anything. It
   preserves `.previous` files for rollback; the watcher then restarts the
   server.

## Bot behavior notes

- The bot answers DMs and @-mentions. It also answers *other bots*
  (`[chat] respond_to_bots`), guarded by `max_bot_chain` (default 3
  consecutive bot-triggered replies per channel, reset by any human
  message) so two bots can't ping-pong forever.
- Replies to the bot (Discord reply feature) carry the replied-to
  message as context; the bot's answers are sent as real Discord
  replies.
- `!ai on|off|status` still gates the chat runtime
  (`[chat] admin_user_ids`).

## Health checks

```sh
curl -fsS http://127.0.0.1:8088/healthz
journalctl --user -u discord-bot.service -n 20 --no-pager
journalctl --user -u sighurt-llm.service -n 20 --no-pager
systemctl --user list-timers sighurt-refresh.timer
```

`/healthz` alone cannot detect a lost Vulkan GPU context (the process answers
while every generation fails). `sighurt-llm-watchdog.timer` +
`scripts/watchdog_probe.sh` cover that gap with a real `/chat` probe every two
minutes.

## Status 2026-08-23 and the sig-v2.1 rerun runbook

Deployed model: the 2026-08-22 continuation artifact (parent SHA
`ada45f66…728efe`) with sampling `T=0.75 top_p=0.92 top_k=50 rp=1.15` and the
full-sequence repetition penalty in `serve_llama`. Greedy decoding caused the
"bot repeats one phrase forever" incident; see the sampling section of the
README before changing these.

A better continuation (sig-v2.1) was trained on 2026-08-22 but the RunPod
account ran out of credits mid-pipeline ("Exited by Runpod", pod
`n9tl05td0zoj7g`; its 80 GB pod volume still holds the v2 checkpoints, and the
separate 100 GB network volume `super-sighurt-training-20260822` was never
attached to anything). To rerun after topping up credits:

1. Create an H100 SXM pod (secure cloud, ≥80 GB disk), rsync the repo's
   `training/` plus `training/data/` and the parent
   `~/super-sighurt-1.1b-20260822/final-hf` from the desktop.
2. Build the aux set with the repair rows:
   `python3 training/make_repair_data.py /workspace/data/repair-rows.jsonl`
   then `cat aux-train.jsonl repair-rows.jsonl > aux-train-v21.jsonl`.
3. Run `training/run_on_runpod.sh` (or the equivalent chain: preflight →
   train_llama with `--aux-train aux-train-v21.jsonl` → convert_to_gguf →
   verify_artifact → evaluate_sampling). Budget ≈ 5 h ≈ $17 at 2026 prices.
4. Transfer the artifact, `scripts/promote_llama_model.sh artifact/ <sha>`,
   `scripts/apply_recommended_sampling.sh sampling-evaluation.json`, restart,
   then rerun `training/conversation_acceptance.py` against the live server.
5. Stop the pod the moment the artifact is transferred.
