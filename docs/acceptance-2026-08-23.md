# Long-conversation acceptance results — 2026-08-23

Measured with `training/conversation_acceptance.py` against the live
`serve_llama` `/chat` endpoint (production prompt contract, rolling 12-message
context, exactly like the Discord bot). A local ollama `llama3.2:1b` played a
different human persona per scenario so every conversation is reactive, not
scripted. The serving configuration under test: the 2026-08-22 continuation
model with `SIGHURT_TEMPERATURE=0.75 SIGHURT_TOP_P=0.92 SIGHURT_TOP_K=50
SIGHURT_REPETITION_PENALTY=1.15` and the full-sequence repetition penalty in
`serve_llama` (prompt tail + generated tokens).

## Why this exists

The bot shipped with greedy decoding and a penalty window that ignored the
prompt. In 26-turn conversations that collapsed into a self-reinforcing
repetition spiral: **72 of 78 replies (92%) were byte-identical** across three
scenarios on 2026-08-22, and the live server spent an evening answering every
message with "I am a bot". Diagnosis and fix are documented in the README
sampling section.

## Results (after the fix)

26-turn reactive conversations, distinct scenario and persona per row:

| scenario | exact-duplicate replies | no-reply | identity leaks | prompt echoes | service restarts |
|---|---|---|---|---|---|
| Marcus — Rust debugging at 2am | 1/26 | 0 | 0 | 0 | 0 |
| Priya — indie roguelike hype | 0/26 | 0 | 0 | 0 | 0 |
| Tomas — cursed programming memes | 5/26 | 0 | 0 | 0 | 0 |
| Lena — how the internet works | 0/26 | 0 | 0 | 0 | 0 |
| Kwame — pineapple pizza debate | 0/26 | 0 | 0 | 0 | 0 |

Aggregate: **6 duplicate replies in 130 turns (4.6%)**, zero identity leaks,
zero prompt echoes, zero failed replies, zero service restarts. An earlier
same-day run on the serving host adds two more clean 26-turn scenarios
(Marcus 0/26, Priya 1/26) plus a poisoned-context probe in which the bot
escaped a channel history pre-filled with "I am a bot" spam in 3 of 4 replies.

Qualitatively the transcripts hold topic across all 26 turns, answer basic
questions correctly, advance the conversation (follow-up questions, natural
closings), and keep the casual Discord persona (lowercase asides, emoji,
first-person opinions). Full transcripts: `/tmp/conv-acceptance-local/` on the
desktop at the time of the run; regenerate any time with the command below.

All six duplicates came from one hedge line inside the single hardest
register (absurdist meme humor) — the model falls back to "I am aware of this
problem but I can't..." when it has nothing to say. Known limit of the 1.1B
model; the sig-v2.1 rerun (see the runbook in `discord-ai-setup.md`) targets
exactly this deflection habit.

## Reproduce

```sh
export SIGHURT_API_KEY=...            # from ~/.config/sighurt-llm.env
export SIGHURT_RESTART_CMD="ssh nixos-server systemctl --user restart sighurt-llm.service"
python3 training/conversation_acceptance.py \
  --turns 26 --conversations 17 --output /tmp/conv-acceptance --label mylabel
```

Run it from a machine with local ollama and an SSH tunnel
(`ssh -N -L 8088:127.0.0.1:8088 nixos-server`) to keep the test load off the
serving host.
