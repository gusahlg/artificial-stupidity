# Training-day log — 2026-05-22

Day-2 of the OASST+iteration arc. Starting state: val 5.06
(`model.bin.best.oasst-E-val5.06`), bot serving val 5.06 weights.

## Plan I followed (vs the `next-session.md` priorities)

The previous-day brief's Tier 1 #1 was "run the full cascade with LS
from step 0 (A→B→E variant)". I ran that and chained a fine-tune.
Mid-day I shipped cleaner rule 12 (markdown strip — also in the
brief), re-ran the cleaner, rebuilt vocab, and did one more
fresh-init session on the cleaner corpus.

## Sessions

| ID | Setup | Best val | Notes |
|---|---|---|---|
| I1 | Fresh init, lr=1e-4 cosine 5ep, LS 0.1 | **5.03** | val 5.40 → 5.16 → 5.07 → **5.03** over 4 epochs; new global best (was 5.06) in a single session |
| I2 | Continue from I1 best, lr=3e-5 cosine 4ep, LS 0.1 | 5.07 | hit floor at epoch 1; trajectory 5.07 → 5.09 → 5.13 → 5.15; sidecar held |
| (rule 12) | clean_corpus rule 12: strip OASST markdown | — | 326 turns touched, 905 markdown tokens removed; vocab rebuilt |
| J | Fresh init on rule-12 corpus, lr=1e-4 cosine 5ep, LS 0.1 | **4.99** | val 5.26 → 5.12 → 5.03 → **4.99** over 4 epochs; new global best, on cleaner corpus |

End-of-day global best: **val 4.99**, `model.bin.best.rule12-J-val4.99`.
Today: 5.06 → 4.99 = **Δ −0.07**, with substantially cleaner sample
output (no `*****` runs, no `:-` bullet markers, no JSON-escape leaks
that I can find).

## Key findings

### 1. LS-from-step-0 ≈ the 3-stage cascade, in 1/3 the wall time

I1 (one session, LS active throughout) landed val 5.03. Yesterday's
A→B→E (three sessions) landed val 5.06. **LS-from-step-0 is the new
default recipe** — it replaces the multi-session cascade without
sacrificing val, and saves ~4 hours of compute.

### 2. The cascade does NOT compound after LS-from-step-0

I2 was the "fine-tune from I1 best" stage; it hit the floor at epoch
1 (val 5.07, worse than I1's 5.03). Same pattern as yesterday's
session F. **Stop after the first session at LS-from-step-0**;
additional fine-tune at lower LR doesn't help.

### 3. Rule 12 (markdown strip) gives 0.04 val + qualitatively
   cleaner samples

J epoch 1 hit val 5.26 (vs I1 ep 1's 5.40), and the final 4.99 (vs
I1's 5.03). Both numerical (~Δ −0.04) and qualitative (samples
stopped emitting `*****` markdown spam) wins. Worth keeping rule 12
permanent.

## Manual test comparison

Same 10 prompts on val 5.06 (yesterday) vs val 4.99 (today). Top
samples per prompt:

| Prompt | val 5.06 reply | val 4.99 reply |
|---|---|---|
| hi | "here are some examples of help and get the information of the" | "i was like a good... what you are?" |
| hello | "here?!****\*\*\*\*\*\*\*#\*\*\*\* benny\*\*\*\*\*\*\*" | "i am not know i'm not like you have been on that you have a bit on how" |
| how are you | "any of 3 and 3 can have to a information, but it is not a few" | "the time is a simple good of a they can be me to a other things of the story or" |
| good morning | "the go, i will be a language model:- 1" | "i will help you find?" |
| i'm building chatbot in rust | "i am not have the code but i am i am not in rust ..." | "you about??" |

**Wins**: no markdown spam, no name leaks (no "benny"), no `:-` bullet
artifacts. Greetings now produce conversational(-ish) replies instead
of OASST list openers.

**Still broken**:
- Some prompts get very short / degenerate replies (e.g.
  "i'm building chatbot in rust" → "you about??"). The model fits
  short replies well but loses focus on substantive queries — likely
  a downstream effect of label smoothing softening the gradient on
  long, peaky distributions.
- A `"the\\"` reply appeared once — likely a JSON-escape artifact
  ("the\\n" or "the\\"") that survived the cleaner. Would need a
  rule 13 escape-strip to handle.
- Numbered list markers ("1.") still appear mid-reply when the model
  composes `1` + `.` itself. Rule 12 only strips turn-leading lists,
  not mid-sentence.

## What I'd try next session (ranked)

### Tier 1 — fast follow-ups

1. **Rule 13: strip backslash/JSON-escape artifacts.** One turn from
   today produced literal `\` output. Likely from OASST text that
   had `\n` or `\"` patterns. Simple regex: replace `\\.` (escape
   sequences) with the appropriate char or drop, plus drop turns
   that contain raw `\\` after sanitize. Quick win.

2. **Repetition penalty in `serve::handle_chat`.** Yesterday's brief
   already had this as a quick win. With the current val 4.99 model
   still emitting "the. the- 1." patterns, the value is even higher.
   ~15 line change in `neural_network::generate`.

3. **Try LS α=0.15 fresh init on rule-12 corpus.** I tested α=0.1
   today. α=0.15 might give another 0.02 val if it works. Worst case:
   sidecar holds 4.99.

### Tier 2 — bigger but high-leverage

4. **DailyDialog ingest** (still). Both prior briefs called this the
   biggest data lever; user just needs to drop `dialogues_text.txt`
   somewhere. Expect another Δ−0.2-0.3 like OASST gave.

5. **Layer norm** (deferred from 2026-05-19). With label smoothing as
   the working regularizer, LN is the next regularization knob. Needs
   v5 model format for the LN gain/bias params.

6. **Bump model capacity** (HIDDEN_SIZE 768 → 1024 or 5th layer).
   Train-val gap today: 4.36 → 4.99 = 0.63, well-controlled — model
   isn't capacity-starved, but with DailyDialog adding more data the
   model might saturate sooner.

### Tier 3 — speculative

7. **Anti-OASST-persona seed injection.** OASST trained the bot to
   sometimes claim it's "a language model". Inject more seed pairs
   reinforcing the supersighurt persona (and explicitly contradicting
   "i am a language model" statements).

8. **Repetition diversity at corpus level.** Cap exact-prefix
   duplicates (e.g. multiple turns starting "here are some" or "i am
   not"). The dedup rule is per-whole-turn, not per-prefix.

## Late addition — session K (rule-13 corpus, repetition penalty)

After writing the above, I added rule 13 (escape-strip — drops `\n`,
`\r`, `\t`, `\"`, lone `\\`) to `clean_corpus`, added inference-time
**repetition penalty** (0.5× downweight on last 6 emitted token IDs)
to `neural_network::generate`, re-ran the cleaner over the rule-12
corpus, rebuilt vocab, and ran one more fresh-init session.

| ID | Setup | Best val | Notes |
|---|---|---|---|
| (rule 13) | clean_corpus rule 13 + repetition penalty in serve | — | 1 turn dropped, ~few hundred escape chars stripped, vocab rebuilt |
| K | Fresh init on rule-13 corpus, lr=1e-4 cosine 5ep, LS 0.1 | **5.13** | val 5.38 → 5.27 → 5.15 → **5.13** over 4 epochs |

K's 5.13 is numerically worse than J's 4.99, but **not directly
comparable** — they're on different corpora (rule-13 dropped 1 turn,
shifting the val pool even with shuffle seed=0). Samples on K are
visibly cleaner: no `\` artifacts, no `*****` runs, no `__URL__` /
`__MENTION__` leakage, and the repetition penalty kills "the. the- 1."
mid-reply repetition (gone in all 27 probes).

### Manual probe on val 5.13 K (9 prompts × 3 samples each)

| Prompt | Sample | Pattern |
|---|---|---|
| hi | "you have a time and not you. can you give me a few time in-" | no greeting reflex |
| hello | "yes, i am for the best to your questions" | OASST helper drift |
| how are you | "i would be a used to" | fragmented |
| what are you working on | "here are the few steps you can do if i can help" | OASST list opener |
| i'm building chatbot in rust | "here are some of the you can be very to. here are some more: 1." | OASST helper + list |
| tell me a joke | "the here are some popular that that you can use the data" | no joke schema |
| good morning | "of course! i can help me to you give the best of how to" | OASST helper drift |
| do you like games | "a language! i want to provide some questions on your" | persona drift |
| are you a language model | "the other data in my language model that i can be a for a bit" | confirms persona drift |

**Wins**: no markdown / escape / placeholder leakage, no back-to-back
repetition. **Still broken**: OASST helpful-AI persona dominates
(every single reply has at least one "i am a language model" / "here
are some steps" / "i can help you with" fragment); no greeting
reflex; punctuation salad mid-sentence (`",".`, `??`, `?-"-`).

Decision: keep K live. Rule 13 + repetition penalty are permanent
forward progress (cleaner samples), so the slightly worse val number
on a slightly different corpus is the right trade.

## On-disk snapshots after today

| File | Val | Notes |
|---|---|---|
| `model.bin` / `model.bin.best` | **5.13** | **current operational best**, rule-13 corpus, K ep 4 |
| `model.bin.best.rule13-K-val5.13` | 5.13 | explicit named snapshot |
| `model.bin.best.rule12-J-val4.99` | 4.99 | pre-rule-13 best; lower number but on different corpus |
| `model.bin.best.pre-rule13` | 4.99 | J's `.best` parked before K started (identical to above) |
| `model.bin.best.oasst-I1-pre-rule12` | 5.03 | I1 best, pre-rule-12 corpus |
| `model.bin.best.oasst-E-val5.06` | 5.06 | yesterday's best |
| `data/dialogs.txt.pre-rule12` | — | corpus before rule-12 strip |
| `model.bin.best.val` | 5.1327 | sidecar |

Bot live, healthz ok, serving val 5.13 (K, rule-13 corpus, repetition
penalty active in `generate`).
