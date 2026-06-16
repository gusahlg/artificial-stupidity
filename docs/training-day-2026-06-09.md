# Training-day log — 2026-06-09

Starting state: bot serving L weights (val 4.7453 on seed-augmented corpus,
session 2026-06-02). 7 days since last training session.

## Hypothesis from prior session

`docs/training-day-2026-06-02.md` ended with the conclusion that L's
val improvement masked a register regression — every probe answer
opened with an OASST helper template ("here are some", "there are
several", "sure, here are"). The 67 anti-OASST pairs added that day
were at a ~1:45 ratio against OASST volume — too sparse to shift the
register. Next-session.md prescribed: grow seed pairs from 200 → 500-800
and lean on bot-side rule 14 to cap helper templates.

## Pre-train analysis: bot-side prefix distribution

Ran `Counter` over bot turns in L's corpus.

| PREFIX_LEN | distinct prefixes | >30 reps | >15 reps | >10 reps |
|---|---|---|---|---|
| 3 | 3706 | 7 | 20 | 26 |
| 4 | 4319 | 1 | 5 | 13 |
| 5 | 4646 | 1 | 1 | 4 |

At PREFIX_LEN=3 the helper templates concentrate cleanly:
`here are some` (77), `there are several` (66), `sure, here are` (52),
`here is a` (50), `there are many` (43), …

L's rule 14 (PREFIX=4, MAX=15) was effectively a no-op (drops 37 turns)
because suffixes vary at the 4th token: "here are some [tips/ways/things/...]".
PREFIX=3 catches the helper templates regardless of what fills slot 4.

User-side at PREFIX_LEN=3 is dominated by legit question patterns
("what is the" 244, "how do i" 93, "can you give" 68) — keep bot-only.

Estimated drop counts at various tunings:

| PREFIX | CAP | drops | % of bot turns |
|---|---|---|---|
| 3 | 10 | 452 | 8.6% |
| 3 | 15 | 337 | 6.4% |
| 4 | 10 |  81 | 1.5% |
| 4 | 15 |  37 | 0.7% (L) |

Picked PREFIX=3, MAX=10 — most aggressive helper-template cap that
still leaves >10 instances of each template (plenty of signal but no
register domination).

## Changes shipped today

### 1. Rule 14 retune (`src/bin/clean_corpus.rs`)
- `MAX_PREFIX_REPEATS`: 15 → 10
- `PREFIX_LEN`: 4 → 3
- Tests updated: `prefix_cap_exempts_short_turns` now uses a 2-token
  bot reply so it stays below the new PREFIX_LEN.
- 35 tests still green.

### 2. Seed-pair expansion (`src/bin/inject_seed.rs`)
Added 262 new pairs across 11 categories, all explicitly NOT-list-opener:
- **More anti-LM paraphrases (34 pairs)**: every phrasing of "are you a
  robot/bot/person/human/LLM/transformer/neural net/conscious/sentient"
  + identity probes ("how many parameters", "what's your architecture",
  "are you trained on the internet").
- **Greeting reflex expansion (23 pairs)**: "hi how are you", "hey bot",
  named greetings ("hi supersighurt"), multi-lang ("hej", "salut").
- **Short reactions (24 pairs)**: oh, ah, ok then, got it, nope, nah,
  oh damn, oh shit, oh no, ugh, meh, eh, aight, k, yup, …
- **Discord slang (43 pairs)**: kek, lul, lulw, monkas, pepega, sadge,
  copium, fr, no cap, on god, bet, vibes, mood, real, facts, true, ez,
  rip, oof, yikes, bruh, smh, ngl, tbh, imo, idc, fml, wtf, damn, goated,
  clutch, sus, sheesh, fire, slay, vibe check.
- **Helpful-AI deflections (31 pairs)**: "write me a poem", "write a
  story", "write code for", "explain quantum physics", "teach me",
  "describe yourself", "solve this problem", "fix my code" — all reply
  with a counter-question that REFUSES the helper-template opener.
- **Story-prompt deflections (5 pairs)**: "tell me a story", "once upon
  a time", "a long time ago" → bot does NOT continue the story.
- **Code/debug talk (30 pairs)**: compile errors, runtime errors, null
  pointer, segfault, panic, deadlock, borrow checker, lifetimes, force
  push, revert, ci is failing, tests failing, ship it.
- **Emotional reactions (15 pairs)**: i'm stressed/anxious/frustrated/
  angry/excited/in love/scared/worried/sick/overwhelmed/done/give up.
- **Continuation (9 pairs)**: and / then / so / but → "and?" / "then what?".
- **Time-of-day + gaming + food + status + meta + server (60 pairs)**:
  filler conversational topics.

Total seed pairs: 205 → 467 (+262).

### 3. Corpus rebuild
- Snapshot L's corpus → `data/dialogs.txt.pre-session-P-2026-06-09`.
- Ran `inject_seed` THREE times so each pair lands 3-4 times before
  rule 7 dedup-cap kicks in. Original 205 pairs were already 1×, new
  262 were 0× — after 3× inject and rule 7 cap=3, all 467 pairs end
  at exactly 3 copies.
- Ran `clean_corpus` with retuned rule 14.
- Final corpus: 5178 sections, 19681 turns (was L's 5400 sections /
  18538 turns).
- Rule 14 dropped 452 turns (exactly as predicted).
- Rule 7 dropped 924 (the duplicated seed re-injections).
- Vocab unchanged at 3029.

Seed/OASST ratio: 467 pairs × 3 copies × 2 turns each = 2802 seed
turns vs ~8550 OASST turns post-rule-14 = **1:3** (was 1:45 in L).

### 4. Sibling crate API fix (`src/gpu.rs`)
The `tensor-ash` crate in `~/repos/ml_project/` renamed
`Tensor::zeros_device` → `Tensor::uninit_device` since 2026-06-02.
Straight rename; LayerGpu fields are uploaded before reading anyway.
Build clean after the change.

## Sessions

| ID | Setup | Best val | Notes |
|---|---|---|---|
| P | Fresh init, 5 epochs cosine, LS 0.10, 14503 examples | **4.99** (epoch 4) | Regressed vs L on both val (+0.24) and probe register. Rolled back. |

P trajectory: 5.24 → 5.12 → 5.02 → **4.99** → 5.00 (overfit at epoch 5).

Live model held at L (in-memory) the entire training run — path
watcher stopped before training to prevent serve from reloading
partial weights. Restored to L on disk after probe.

## Probe results (P, val 4.99)

`scripts/probe_bot.sh` standard 9-prompt probe (3 samples each):

| Prompt | Worst sample |
|---|---|
| hi | "here is the best way to use an example of the- 1- a time to be: in" |
| hello | "sure! i can help you with a few code:- you can help you with your time:" |
| how are you | "the most popular and help you with the most popular. to use the code and that. it is" |
| tell me a joke | "here's a if you can make an a to do you know how to a here's the be of your" |
| do you like games | "i'm not a! i am open assistant, i can help you with a. i can" |
| are you a language model | "you have an ai language model, i have been an ai language with all. can you have been" |

P at val 4.99 is **strictly worse than L at val 4.75** — same OASST drift but
*also* less coherent than L's already-broken output. Verdict: not deploying.

Full probe at `docs/manual-test-2026-06-09-P-standard.md`.
Live L baseline probe at `docs/manual-test-2026-06-09-L-baseline-register.md`
also confirms L still leaks OASST register on every prompt — both
models are in the same broken-coherence zone, P just more so.

## Diagnosis

Rule 14 retune from PREFIX=4 MAX=15 to PREFIX=3 MAX=10 dropped 452
bot turns (8.6% of bot side), almost all OASST helper templates
("here are some …", "there are several …", "sure, here are …"). The
intention: starve the model of helper-template patterns so it learns
the casual register from the expanded seeds.

What actually happened: the model learned **fragmentary** helper
templates instead. With 8.6% fewer complete helper-template examples,
it picked up the *shape* ("here is a …", "in a …", "you can …") but
not enough end-to-end coherent template completions to generate them
cleanly. Result: broken OASST openings instead of clean OASST openings.

The seed-expansion lift on short prompts is real — epoch 4 training
sample for "Hey there!" was "good and what about?" (clean conversational
register from the new pairs). But the seed lift is only visible on
prompts that pattern-match seed-pair user turns. Anything that
doesn't pattern-match falls back to fragmented OASST.

## What I learned

1. **Two changes at once → can't attribute.** Rule 14 retune AND seed
   expansion shipped together. Need to test them independently next
   time. The seed expansion alone may be net-positive; the rule 14
   retune is probably net-negative on its own.
2. **Rule 14 PREFIX=3 MAX=10 is too aggressive.** The 8.6% drop
   damages the model's ability to complete OASST patterns coherently.
   PREFIX=4 MAX=15 (L's setting) drops only 0.7% — too soft to shift
   register but doesn't damage coherence either. Need to find a
   midpoint, OR pair the cut with a seed-pair tsunami strong enough
   to fill the gap (much more than 1:3 ratio).
3. **At ~10M params, val is not a coherence proxy.** L at 4.75 and P
   at 4.99 produce qualitatively similar broken output; the val
   difference is mostly OASST-perplexity, not real understanding.
   Sub-5.0 val on a 10M tanh MLP just means the model fits OASST
   templates tightly, not that it generates coherent text.
4. **Capacity, not corpus, is now the blocker.** This is the second
   session in a row where corpus changes failed to materially improve
   probe quality (06-02's N/O trim experiments had the same pattern:
   val cost without proportional quality gain). Time to attempt the
   v5 model format → bigger model or layer norm. See next-session.md.

## Final decision (chose to keep L serving)

- `model.bin` ← L (val 4.7453)
- `model.bin.best` ← L
- `model.bin.best.val` ← 4.7453
- path watcher restarted, healthz ok

P's weights preserved at `model.bin.best.session-P-val4.99` for future
reference / ablation.

## Today's new snapshots

- `data/dialogs.txt.pre-session-P-2026-06-09` — L's corpus (pre-expansion)
- `data/dialogs.txt` — P's corpus (467 seeds, rule 14 PREFIX=3 MAX=10) **CURRENTLY ACTIVE**
- `model.bin.best.session-P-val4.99` — P's best
- `model.bin.session-L-live` — L's in-memory weights, saved to disk
- `model.bin.best.session-L-recorded` — L's `.best` snapshot
- `model.bin.best.val.session-L` — L's `.val` sidecar
- `model.bin.pre-session-P` / `model.bin.best.val.pre-session-P` — initial backups
- `training-session-P-2026-06-09.log` — full training log
- `docs/manual-test-2026-06-09-P-standard.md` — P probe
- `docs/manual-test-2026-06-09-L-baseline-register.md` — L register probe
- `scripts/probe_register.sh` — new adversarial register probe

Note: `data/dialogs.txt` is currently P's corpus (not L's). The live
model L was trained on a different (now-archived) corpus. If we want
to re-train against L's exact corpus, restore from
`data/dialogs.txt.pre-session-P-2026-06-09` and `rebuild_vocab` first.
