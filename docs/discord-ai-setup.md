# Discord AI setup (single machine)

As of 2026-07-24 everything runs on one desktop: the Discord bot
(`~/discord-bot`, twilight), the LLM HTTP server (`serve` from this
repo), the trainer, and the weekly corpus refresh. The old split
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
  │     with user/user_id/reply_to_* context
  ▼
serve (systemd --user sighurt-llm.service, ~/artificial-stupidity)
  reads the PINNED pair: model.serving.bin + data/dialogs.serving.txt
      ▲
      │ scripts/promote_model.sh  (deliberate promotion)
      │
trainer (model.bin) ◄── data/dialogs.txt ◄── scripts/refresh_corpus.sh
                                             (sighurt-refresh.timer, weekly)
```

Two corpus/model pairs exist on purpose:

- **Serving pair** — `model.serving.bin` + `data/dialogs.serving.txt`,
  snapshotted together by `scripts/promote_model.sh`. The v4+ model
  format hashes its vocab; serving from a pinned snapshot means the
  weekly corpus refresh can never drift the vocab out from under the
  live model.
- **Training pair** — `model.bin` + `data/dialogs.txt`, which the
  weekly refresh and training sessions churn freely.

## The pieces

| Unit (systemd --user) | What it does |
|---|---|
| `discord-bot.service` | The bot. WorkingDirectory `~/discord-bot`, env from `~/discord-bot/.env`. |
| `sighurt-llm.service` | The LLM server. WorkingDirectory `~/artificial-stupidity`, env from `~/.config/sighurt-llm.env`. |
| `sighurt-llm.path` + `sighurt-llm-reload.service` | Watch `model.serving.bin`; restart the server when a promotion lands. |
| `sighurt-refresh.timer` + `.service` | Weekly (Sun 04:37) corpus refresh: rsync bot TSVs → incremental convert → clean → rebuild vocab. |

`~/.config/sighurt-llm.env` (mode 600):

```ini
SIGHURT_BIND=127.0.0.1:8088
SIGHURT_API_KEY=<same value as LLM_API_KEY in ~/discord-bot/.env>
SIGHURT_MODEL=/home/gusahlg/artificial-stupidity/model.serving.bin
SIGHURT_CORPUS=data/dialogs.serving.txt
```

Everything is loopback-only; nothing binds Tailscale anymore. The
server refuses to start without a model (`SIGHURT_ALLOW_FRESH=1`
overrides, for testing only). Sampling knobs: `SIGHURT_TEMPERATURE`
(default 1.0) and `SIGHURT_TOP_P` (unset = top-k 5).

## /chat wire protocol

POST `/chat` with `X-API-Key`. JSON object, all values strings; only
`input` required — unknown fields ignored:

```json
{
  "channel_id": "…", "user": "display name", "user_id": "snowflake",
  "user_is_bot": "true|false",
  "input": "text with <@id> mentions already resolved to @name",
  "reply_to_user": "…", "reply_to_user_id": "…", "reply_to_text": "…",
  "reply_to_is_bot": "true|false", "reply_to_is_self": "true|false"
}
```

Response: `{"reply":"…"}`. The server maps speakers to per-channel
PERSON_N tags (the bot itself = PERSON_0, `reply_to_is_self` marks a
reply to the bot's own message) and feeds tagged turns — the same
shape as the training corpus. The bot resolves `@name` tokens in the
reply back into real `<@id>` pings with a strict allowed-mentions
list.

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
3. `convert_discord` is reply-aware: reply chains are stitched into
   their own sections (a threaded exchange scattered through an hour of
   channel noise becomes one clean dialog), and `<@id>` mentions become
   learnable `@displayname` tokens.
4. `scripts/rebuild_full_corpus.sh [oasst.jsonl.gz]` rebuilds the whole
   corpus from raw sources (all Discord history + OASST + seed pairs +
   cleaner) — use after converter-rule changes, then train fresh.
5. Training: `nix develop -c cargo run --release --bin train -- …` on
   `data/dialogs.txt`/`model.bin`; when a run is worth shipping, run
   `scripts/promote_model.sh [model.bin.best]` — the watcher restarts
   the server on the new pair with ~3s downtime.

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
