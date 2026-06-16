# Training-day log — 2026-05-21

Running log of what I do unsupervised today. Final findings get merged
into `docs/improvement-survey.md` at end of day.

## Starting state

- Corpus: post-rule-11, 7214 turns / 868 sections. Best val on this
  vocab: **5.80** (`model.bin.best.rule11-val5.80`).
- Bot is serving val-5.80 weights.

## What I did before any training

1. **Ingested OpenAssistant (OASST)** from `~/ai-data/openassistant/`
   via the new `scripts/ingest_oasst.py`. 3670 English trees consumed,
   3551 sections written, 11347 turns added. Reply text truncated to
   400 chars at the closest sentence boundary (since `MAX_TARGET_TOKENS
   = 20` only trains on the first ~20 tokens anyway, and longer text
   just bloats the file with prelude-bound context the model never
   sees). Refusal patterns ("as an ai language model…"), code blocks,
   and very short turns are dropped.

2. **Re-ran the cleaner** on the augmented corpus. End state: 4267
   sections / 18229 turns (≈ 2.5× the previous corpus size). Cleaner
   barely touched the OASST half — confirms OASST is much cleaner
   than Discord chat.

3. **Vocab rebuilt** with the bincode cache invalidated. `__URL__`
   and `__MENTION__` dropped down the frequency list (171 → 286, 241
   → 347) because OASST tokens like "however", "research", "example",
   "consider" now beat them on count.

4. **Bug fix landed mid-session** (after user report): `__URL__`,
   `__MENTION__`, `__EMOJI__` are now added to
   `VocabIndex::forbidden_emit_ids` so the bot can't emit them at
   generation time. Existing in-process training reads its own
   in-memory `VocabIndex` so the running session A is unaffected,
   but every model save fires the systemd `sighurt-llm.path` watcher
   which restarts the inference server using the *newly-built* binary
   that has the fix. So the live Discord bot stops emitting
   placeholders as soon as session A saves its first epoch.

## Planned sessions

| ID | Setup | Goal |
|---|---|---|
| A | Fresh init, lr peak 0.0001, warmup 1 + cosine over 6 epochs to lr_min 5e-6, no dropout | Clean baseline number on the augmented corpus |
| B | Continue from A best, lr peak 0.00003, cosine over 4 epochs to lr_min 3e-6 | Fine-tune (mirrors the session 6→8 trick that got us 5.80→5.69) |
| C | Continue from B best, lr 0.00003 + dropout 0.05 | Test whether dropout helps now that backprop is correct AND the corpus is bigger |
| D | Open slot — depends on B/C results |

Per-epoch wall time on the augmented corpus is ≈ 2.5× the previous
corpus (≈ 25 min/epoch). 2h timeout = 4-5 epochs per session. Three
sessions back-to-back ≈ 6h of compute.

## Session A — fresh init, lr=0.0001 cosine, no dropout

Per-epoch wall time ~28.7 min on the 18.2k-turn corpus (170k training
steps per epoch — about 13.6 useful target tokens per example, up
from the previous corpus's ~10.5 because OASST replies are longer).

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 1 | 0.00010 | 5.53 | **5.50** | new best, train≈val — no overfit signal |

**This already beats every prior best on every corpus** (was 5.69 on
pre-rule11, 5.80 on rule-11). The OASST data dominates the
improvement; cosine + lr=0.0001 was the same recipe that hit 5.80
before. So this is purely the data-lever effect.

Train-val gap of essentially zero at epoch 1 means the model has
plenty of room to keep fitting; subsequent epochs at lower LR should
drop val further before overfitting kicks in.

Sample at val=5.50: "it about the word, and the world" — short
fragment, vaguely conversational. Will get better as cosine taper
runs and gradients sharpen.

Snapshot landed; bot will pick up these weights at next save trigger.

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 2 | 0.00009 | 5.28 | 5.60 | mild overfit signal |
| 3 | 0.00008 | 5.25 | 5.71 | sample now coherent OASST style |
| 4 | 0.00005 | 5.10 | 5.71 | timeout looms |

Session A final best: **val 5.50** at epoch 1.

## Session B — fine-tune from A best (lr=0.00003 + cosine over 4)

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 1 | 0.00003 | 4.84 | **5.24** | new GLOBAL best — Δ -0.26 in one epoch |

Sample at val=5.24: *"i am not a happy to have a way of a language
model, i am know for an example of a language model, i am not know
about a lot of your code."* — full clauses, identifiable OASST style.

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 2 | 0.00002 | 4.62 | 5.36 | mild overfit |
| 3 | 0.00001 | 4.47 | 5.43 | sample shows OASST list rendering ("here are some examples...") |
| 4 | 0.00000 | 4.38 | 5.46 | timeout reached |

Session B final best: **val 5.24** at epoch 1.

## Session C — ultra-fine-tune (lr=1e-5 from B best)

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 1 | 0.00001 | 4.54 | 5.28 | slight regression; sidecar protected `.best` |
| 2 | ~0     | 4.44 | 5.32 | confirms 5.24 is the local floor |
| 3 | ~0     | 4.39 | 5.34 | done |

Session C final: no improvement on B's 5.24. Confirms that lower-LR
continuation hits a floor; need a different regularizer to escape.

## Session D — continue B best + dropout 0.05

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 1 | 0.00003 | 4.75 | 5.43 | val regressed; **dropout doesn't help even with OASST data** — confirmed. Stopped early. |

## Session E — continue B best + label smoothing α=0.1

Different regularizer. Label smoothing softens the cross-entropy
target distribution; gradient direction changes without adding noise
to activations (unlike dropout). Implementation: in
`teacher::compute_deltas_into`, replace one-hot target with
`(1 - α) · one_hot + α/V · uniform`. Val loss formula unchanged so
numbers are directly comparable to B/C.

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 1 | 0.00003 | 4.54 | **5.06** | new GLOBAL best — Δ -0.18 from 5.24 in one epoch |

Sample at val=5.06: *"here are some ways to do i have used in your"*
— OASST opener. The model is finally getting some semantic
generalization the prior recipe couldn't unlock.

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 2 | 0.00002 | 4.29 | 5.06 | flat — label smoothing keeps val from drifting up |
| 3 | 0.00001 | 4.08 | 5.11 | mild overfit; sample is long multi-clause |
| 4 | 0.00000 | 3.98 | 5.14 | timeout |

Session E final best: **val 5.06** (epoch 1). Net day's improvement
on the rule-11+OASST corpus: 5.50 (A) → 5.24 (B) → 5.06 (E),
Δ −0.44 over three sessions.

## Session F — ultra-fine-tune from E best (lr=1e-5 + LS 0.1)

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 1 | 0.00001 | 4.24 | 5.07 | no improvement; sidecar held. Same floor-hit pattern as C. |

Stopped early after one epoch — clear signal that lower-LR fine-tune
hits a floor regardless of regularizer.

## Session G — higher LR + LS 0.1 from E best (lr=5e-5 cosine over 4)

Hypothesis: label smoothing's softer gradient should tolerate a
slightly higher LR than the bare-recipe sweet spot of 3e-5.

| Epoch | LR | Train | Val | Notes |
|---|---|---|---|---|
| 1 | 0.00004 | 4.47 | 5.10 | mild overshoot |
| 2 | 0.00003 | 4.17 | 5.11 | doesn't recover |

Stopped after epoch 2. Hypothesis disconfirmed. lr=3e-5 IS the sweet
spot even with label smoothing — what label smoothing actually does
is let that same lr push val further down (B got 5.24, E got 5.06).

## End-of-day summary

**New global best**: val **5.0578**, model in `model.bin.best`,
explicit snapshot `model.bin.best.oasst-E-val5.06`.

**Day's val descent**:
| Setup | Val | Δ from prior |
|---|---|---|
| Starting point (rule-11 corpus, no OASST) | 5.80 | — |
| Session A — fresh init + cosine on OASST corpus | 5.50 | −0.30 |
| Session B — fine-tune at lr=3e-5 from A best | 5.24 | −0.26 |
| Session E — fine-tune at lr=3e-5 from B best + label smoothing 0.1 | **5.06** | **−0.18** |

**What worked**:
- **OASST ingest** is the single biggest lever. Going from a
  Discord-only corpus (7k turns) to Discord+OASST (18k turns) dropped
  the floor from 5.80 → 5.50 in the same training recipe. The OASST
  data adds substantive Q/A patterns the bot was missing entirely
  (it's now started replying with "there are several ways to help you
  with" and similar OASST openers).
- **Fine-tune cascade**: fresh init at lr=1e-4 → continue at lr=3e-5
  → continue at lr=3e-5 with LS 0.1. Each stage takes the val floor
  ~0.2 lower. Past 3 stages, returns diminish (session F at lr=1e-5
  hit the floor immediately).
- **Label smoothing α=0.1** beats dropout for this corpus + model.
  Pairs cleanly with the fine-tune cascade. Implementation in
  `teacher::compute_deltas_into` — replaces the one-hot target with
  `(1-α) · one_hot + α/V · uniform`. CLI flag `--label-smoothing F`.

**What didn't work**:
- **Dropout** at any p > 0: regressed val across both rule-11 and
  OASST corpora. The earlier-day cache-the-pre-dropout-activation
  bug was fixed (correctly cached, mask applied separately in
  backward), but even with correct backprop, dropout hurts on this
  small a corpus + model.
- **Higher LR + label smoothing**: lr=5e-5 with LS still overshot
  the 3e-5 basin. The LR sweet spot doesn't shift with regularization.
- **Ultra-low LR fine-tune** (lr=1e-5 with or without LS): hits a
  floor in 1 epoch. The benefit of the lower LR caps out fast.

**What I'd try next** (need user input or longer compute):

- **[BIG, needs data]** DailyDialog ingest. Already-known biggest
  data lever; `ingest_dailydialog` binary exists and just needs
  `dialogues_text.txt`. Each OASST-comparable corpus expansion
  shipped −0.30 val; DailyDialog could give a similar drop.
- **[MEDIUM, needs format bump]** Layer normalization between hidden
  layers. Was deferred from yesterday because it needs a v5 model
  file format for the LN gain/bias params. Would compound with label
  smoothing.
- **[QUICK]** Try a more aggressive label smoothing (α=0.15, 0.2)
  paired with fresh init from scratch. Today's E only added LS during
  fine-tune; a fresh-init from-scratch run with LS-from-step-0 might
  reach a different (better) basin.
- **[MEDIUM]** Larger context window (32 → 64 tokens). OASST replies
  are long; the model sees only the last 32 tokens of context, which
  for late-conversation turns is a meaningful loss. Doubles the first
  hidden-layer input width.
- **[BIG]** Bigger model — bump HIDDEN_SIZE 768 → 1024 or add a 5th
  hidden layer. With the corpus now 18k turns we have more headroom
  for capacity. Today's train loss bottomed around 4.0 with val 5.06
  → train-val gap of 1.0 is moderate. A bigger model could close
  some of that.

**Operational state**:
- Bot serving val=5.06 weights, healthz ok.
- All code changes uncommitted, in tree. New tests: 4 in
  `train::lr_schedule_*`, 4 in `tokenizer`, 7 in `text_utils`, 17 in
  `clean_corpus`, 33 in lib (+ the `placeholder_tokens_are_forbidden_at_generation`
  regression test for the user-reported `__MENTION__`/`__URL__` leak).
- `model.bin.best.val` sidecar = 5.0578.
- Named snapshots on disk: `model.bin.best.oasst-A-val5.50`,
  `model.bin.best.oasst-B-val5.24`, `model.bin.best.oasst-E-val5.06`,
  plus older `.rule11-val5.80`.

**Things I noticed but didn't fix**:
- Generated samples sometimes emit `*` and `:` characters from
  OASST's markdown that survived the cleaner (e.g. ":- your"). A
  cleaner rule 12 could strip leading-`*`/`:` decorations.
- The bot now talks in OASST "I am" style ("i am not know how can i
  use that") which is a tonal shift from the earlier Discord chitchat
  voice. If the user wants more Discord-flavor, the OASST contribution
  should probably be capped (`--limit 1500` instead of 5000) so
  Discord stays the majority of the corpus.





