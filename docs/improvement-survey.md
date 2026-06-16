# artificial-stupidity — improvement survey

A snapshot, taken 2026-05-19, of what we know about where the model is
limited and the concrete changes most likely to unlock further gains.
Three parallel passes informed this: a corpus content audit, a
format/tokenizer/training-mechanics audit, and an ML-alternatives
ranking. The signal across all three is consistent — see the "single
biggest finding" box below.

**2026-05-20 update**: implemented dropout, cosine LR schedule with
warmup, attempted global gradient-norm clipping. Findings below in
§"What we tried, what worked" — short version: dropout was the
biggest engineering bug-hunt of the day (cached the wrong activation
for the tanh derivative; fixed). Global grad-norm clip turned out to
need weight-gradient sums, not delta sums; disabled the implementation
that clips deltas. Cosine LR schedule works. Net result on val: ~5.80
on the rule-11 corpus, marginally below the 5.82 we set at the start
of the day's experiments but not directly comparable to pre-rule11
sessions (vocab changed).

**2026-05-21 update**: Ingested OpenAssistant (OASST) trees as a new
data source (corpus 7214 → 18229 turns, ~2.5×); landed label
smoothing as a new regularizer; ran sessions A–G. Net result: val
**5.06** on the OASST-augmented corpus.

**2026-05-22 update**: Ran the LS-from-step-0 recipe (Tier 1 #1 from
yesterday's brief) — landed val **5.03** in a single session,
matching the prior 3-session A→B→E cascade in 1/3 the wall time.
Then shipped cleaner rule 12 (markdown strip — 326 turns / 905
tokens) and re-ran the same recipe on the cleaner corpus: val
**4.99**, new global best. Full details in
`docs/training-day-2026-05-22.md`. Headline findings:
- LS-from-step-0 is the new default recipe; the multi-session cascade
  is no longer needed.
- Rule 12 gives Δ −0.04 val AND removes the `*****` / `:-` markdown
  noise that was visibly degrading samples.
- Mid-reply numbered lists ("1.") and JSON-escape artifacts ("the\")
  still occasionally appear — left as items for a future rule 13.
Full details in `docs/training-day-2026-05-21.md`. Headline
findings:
- OASST ingest is the single largest data lever ever measured here
  (val 5.80 → 5.50 at same recipe — pure data effect, no other
  changes).
- Label smoothing α=0.1 beats dropout for this corpus + model. The
  smoothed-target gradient lets the fine-tune basin go ~0.2 deeper.
- The three-stage fine-tune cascade (fresh init at lr=1e-4 → continue
  at lr=3e-5 → continue at lr=3e-5 with LS 0.1) is the new playbook.
- Dropout still doesn't help, even on the larger corpus (session D
  regressed val from 5.24 to 5.43 at p=0.05).

## Where we are today

- **Best val cross-entropy: 5.6867** (perplexity ≈ 295), reached by
  session 8 epoch 1 at lr=0.00003 fine-tune from session 6 epoch 1
  (val 5.80). Snapshot in `model.bin.best`, hash-validated in v4.
- **Train-val gap: ~1.5+ at every LR we tried** (0.0003, 0.0001,
  0.00003, 0.00001). Train descends to ~4.0 while val sits at ~5.7.
- Corpus state: ~7200 turns / ~870 sections after cleaner rules 1–10
  (drops monologue/single-speaker/short sections; URL→`__URL__`;
  parenthesized Discord refs → `__MENTION__`; dedup cap 3; fixed-seed
  shuffle so the val tail is random).
- Architecture: 4 tanh hidden layers × 768, 256-dim embeddings,
  context 32 tokens, output softmax over a 3029-token vocab. ~10M
  parameters total.
- Optimizer: AdamW (β1 0.9, β2 0.999, ε 1e-8), weight decay 1e-4,
  element-wise gradient clip ±5, no dropout, no layer norm, no LR
  schedule.

> **Single biggest finding** (all three audits converge here): the
> model has enough capacity. The bottleneck is **generalization** —
> we lack standard regularization (dropout, layer norm), an LR
> schedule, and we're running out of in-distribution dialog data.

## 1. Corpus content findings

### 1a. What the cleaner already fixed

| Rule | Effect | Drops on the original corpus |
|---|---|---|
| 1 | Drop monologue-only sections (every turn is PERSON_0). Catches `ingest_tinystories` artifacts. | 300 sections |
| 2 | Drop single-speaker sections (no dialog signal). | 894 sections |
| 3 | Drop sections that fall below 2 turns after turn-level filters. | 87+16 sections |
| 4 | Drop URL-dominant turns (≥80% URL after strip). | 192 turns |
| 5 | Drop turns with zero alphanumeric content (emoji spam, punctuation). | 61 turns |
| 6 | Drop role-ping-only turns (`@everyone`, `@<digits>`). | 0 in practice (Discord export already sanitized to `(@...)`, see rule 10) |
| 7 | Dedup repeated turns to ≤3 occurrences each. | 821+16 turn occurrences |
| 9 | Replace inline URL whitespace-tokens with `__URL__`. | 172 tokens |
| 10 | Replace parenthesized Discord refs (`(@…)`, `(@&…)`, `(#…)`, `(t:…:R)`, `(id:browse)`) with `__MENTION__`. | 349 tokens across 284 turns |
| 8 | Fixed-seed section shuffle so val tail is a random sample. | — |

Net effect: original 13883 turns / 2077 sections → cleaned 7214 / 868.

### 1b. What's still poisoning training

These showed up in the post-cleaner audit and would each pay off with
modest cleanup work.

- **Discord emoji shortcodes** (~50 distinct `:name:` tokens —
  `:yaycat:`, `:PanFrown:`, `:kek:`, `:CanWePinThisGuy:`, etc.).
  Tokenizer splits these into `:`, `name`, `:` — pure vocab noise.
  - **[ACTION-Q1]** Add cleaner rule 11: replace `:identifier:`
    (alphanumeric+underscore, length 2–32, surrounded by colons) with
    `__EMOJI__`. Whitelist that token in `tokenizer::is_special_passthrough`
    like we already do for `__URL__` and `__MENTION__`.
- **Markdown formatting markers**: 101 `**` (bold) and ~493 `*`
  (italics), 521 stray `_`. These tokenize as bare punctuation and
  bloat the vocab head without semantic value.
  - **[ACTION-Q2]** Strip `**`, single `*` (when used as wrapper, not
    bullet), and `_` (when not part of `__URL__`/`__MENTION__`/`__EMOJI__`)
    in `convert_discord::sanitize_for_corpus` and re-add as a cleaner
    rule for the historical corpus.
- **Code blocks** (~14 triple-backtick turns). Code is highly non-
  natural-language; gives the model bad samples. Either drop the
  whole turn, or replace the fenced block with `__CODE__`.
- **Corrupted artifacts**: a handful of bizarre lines from the
  upstream ingest (`I am a robot. Pain. Pain. Pain.`, `AI mode: ON`,
  comma/period-prefixed turns). 11 lines flagged in the audit.
  - **[ACTION-M1]** Review lines around 1480–1502 of
    `data/dialogs.txt.pre-clean`; either delete the section or trace
    the source data and fix at ingest.
- **`.edit` / `.avatar` command-spam sections**: long sections that
  look like dialogue but are 70%+ a single user mashing a Discord
  slash-command-style prefix. These survive because the *section* has
  multiple speakers — but most of one speaker's turns are
  `.edit X .edit Y .edit Z`.
  - **[ACTION-M2]** Add cleaner rule: if ≥70% of a person's turns
    in a section start with a `.` or `/` slash-command prefix, drop
    the section. Estimated 15–30 sections affected.

### 1c. Topical balance (sample of 200 random sections)

| Topic | % of sections |
|---|---|
| Mixed / non-topical chitchat | 54% |
| Personal / emotional | 12% |
| Jokes & nonsense | 12% |
| **Coding / Rust** | **8.5%** |
| Discord meta (server admin, channel pings) | 7.5% |
| Gaming | 6% |

The server's stated focus (Rust + game-dev) is **under-represented**.
The bot is being trained mostly on off-topic chitchat.

### 1d. Gaps (things the bot should arguably know but doesn't see)

- **Multi-turn problem-solving threads**. ~45% of sections are
  2 turns. Long sections exist but are mostly `.edit` spam, not
  debugging back-and-forth.
  - **[ACTION-M3]** Extract 10–20 representative
    "problem → hypothesis → fix" threads from the Discord archive
    by hand and add them. Smaller than the DailyDialog ingest but
    much higher quality per turn.
- **Bot's own voice is rare**. PERSON_0 is only **3.0% of turns**
  (~219/7214). Most signal is human-human dialog with no PERSON_0
  participant. The 81 + 30 = 111 seed pairs add a baseline
  vocabulary but saturate after a few epochs.
  - **[ACTION-M4]** Generate another batch of seed pairs targeting
    the actual server topics surfaced above (Rust questions, build
    diagnostics, "what are you working on" follow-ups).

### 1e. Realistic ceiling

Given content as-is: val ~5.2–5.4 (perplexity 180–220) is plausible
with just regularization + scheduling. Pushing lower needs new data
(DailyDialog, hand-curated threads) or a real architecture shift.

## 2. Format / tokenizer / training mechanics

### 2a. Tokenizer (`src/tokenizer.rs`)

- **OK**: apostrophe-gluing handles English contractions
  (`don't`, `i'll`, `won't`, `y'all`) as single tokens. PERSON tags,
  `__URL__`, `__MENTION__` pass through verbatim.
- **Lossy**: lowercase strips proper-noun identity ("GitHub" →
  "github" → never re-cased at generation).
- **Lossy**: hyphenated words split (`well-known` → `well`, `-`,
  `known`). Numbers split on commas/periods (`10,000` → `10`, `,`,
  `000`; `10.5` → `10`, `.`, `5`).
- **Out-of-vocab cost**: vocab is capped at 3000 content tokens; the
  long tail of token forms gets `<UNK>`. Subword tokenization (BPE
  / Unigram / WordPiece) would cover ~98–99% of token forms instead
  of ~94%, at a similar effective vocab size.
- **[ACTION-M5]** Measure UNK rate per epoch (count of `<UNK>` events
  hit during `VocabIndex::ids_or_unk` in `train_one_epoch`). If
  >2%, raise `DEFAULT_VOCAB_CONTENT_CAP` to 5000 (manageable; the
  Adam-dense cost is roughly linear in vocab size).
- **[ACTION-B1]** Replace tokenizer with a Unigram-LM subword
  tokenizer (e.g. the `tokenizers` crate). Big refactor (output
  layer, generation logic, every corpus binary) but unblocks proper
  handling of slang/names/typos that Discord chat is full of.

### 2b. Context window (`src/neural_network.rs:54`)

- **CONTEXT_WINDOW = 32 tokens.** First hidden layer takes
  `32 × 256 + 1 = 8193` inputs.
- For sections with >32-token preludes (common for late turns in a
  long section), the model only sees the last 32 — loses prefix
  context.
- **[ACTION-M6]** Log a histogram of context lengths per epoch. If
  >10% of training examples exceed 32 tokens of usable context,
  raise to 64 (2× input width on the first layer, ~2× training time
  on that layer, manageable).
- **Position encoding is a single scalar** `min(position/100, 1.0)`
  (`make_input` in `neural_network.rs:323`). Saturates above
  position 100, gives the model essentially no positional signal.
  - **[ACTION-Q3]** Replace with sinusoidal positional embedding
    (16–32 features added to each token's embedding before the
    concat). ~20 lines. Cost negligible, qualitative win on long-
    context predictions.

### 2c. Architecture (`src/neural_network.rs`)

- **Activation**: tanh. Defensible at this size; ReLU/GELU would
  train faster but need layer norm to stay stable. Not urgent.
- **Depth 4 hidden layers + linear output**. Train-val gap of 1.5
  says capacity is sufficient. A 3-layer variant might generalize
  better; worth an A/B.
- **No skip connections / residuals**. With 4 layers, not yet a
  gradient-flow problem, but residuals usually help small-data
  generalization.
- **No layer normalization**. Most directly impactful architectural
  add — see Q4 below.
- **No weight tying** between embedding (256-dim) and output (768-dim).
  Would require a projection layer or matching dimensions; not free.

### 2d. Training mechanics

- **Adam clipping is element-wise** (`g.clamp(-5, 5)` in
  `train_step`). This is much weaker than global gradient-norm
  clipping. A single huge-grad weight gets pinned to 5 while every
  other weight goes unscaled, so the *direction* of the update
  diverges from the true gradient direction.
  - **[ACTION-Q4]** Implement global-norm clip: compute
    `‖g‖₂` across all layers' deltas before the per-row Adam loop,
    scale all deltas by `min(1.0, clip_norm / ‖g‖₂)`. ~30 lines.
- **No LR schedule**. We empirically found lr=0.0001 fresh → 5.80,
  then lr=0.00003 fine-tune → 5.69. A proper schedule formalizes
  this and removes the need to chain sessions manually.
  - **[ACTION-Q5]** Add cosine annealing across the run: per-epoch
    `lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π * epoch / max_epochs))`.
    Combine with a 100–200-step linear warmup at session start.
- **No dropout, anywhere**. With 10M params on ~6000 training
  examples, this is the missing piece most directly explaining the
  train-val gap.
  - **[ACTION-Q6]** Add dropout p=0.2 between hidden layers
    (after each tanh, before the next layer's linear). Disable
    during validation forward. ~10 lines in `Layer::forward` +
    plumb `training: bool` flag.
- **No layer normalization**. Compounds with dropout — both fight
  internal covariate shift and over-confident activations.
  - **[ACTION-Q7]** Add `LayerNorm` after each hidden layer. ~30
    lines.
- **No mini-batching**. Each train step is one token of one example
  going through forward+backward+Adam. Adam cost (the dominant CPU
  share at ~60% per step) amortizes across batched examples — a
  batch of N reduces Adam-per-token cost by ~N. This is also
  `docs/optimization-roadmap.md` item 6.
  - **[ACTION-B2]** Mini-batch (B=16-64) refactor: stack examples
    into a matrix forward pass, average gradients before Adam.
    Multi-day; unlocks GPU efficiency too.
- **MAX_TARGET_TOKENS = 20**. Bot literally cannot learn to emit
  >20 tokens. Cap chosen for speed (paragraph-length Discord turns
  blow up training time), and the late tokens see only 32-token
  context anyway. If we increase context window, this can grow too.

### 2e. Validation set

- Currently the last 10% of (shuffled-with-seed-0) sections is val.
  Fixed seed makes the val composition deterministic across runs,
  which is good for comparing experiments — but it means we're
  always training against the same val sample, which is a soft
  overfit-to-val risk.
- **[ACTION-M7]** Add a separate ~5% **test** holdout that gets
  computed only once per session (e.g. for the final epoch's
  sample), so we don't unconsciously tune to val.

### 2f. Persistence (`src/persist.rs`)

Already solid:
- v4 file format stamps a vocab hash; `load_with_vocab` checks the
  prefix hash against the current vocab and bails on mismatch —
  prevents silent garbage when corpus changes reorder the vocab.
- Adam moments persist (v3+), so resumed runs skip the warmup tax.
- Atomic save via `<path>.tmp` + rename — the `sighurt-llm.path`
  watcher never reads a half-written file.
- `model.bin.best.val` sidecar tracks the global best val across
  sessions; `train.rs` reads it at startup and won't overwrite
  `.best` unless it actually improves.

## 3. Prioritized improvements

`[QUICK]` ≤ 1 day, `[MEDIUM]` 1–3 days, `[BIG]` 1+ week.

### Top tier (compounds on the val plateau; do these first)

1. **[QUICK] Dropout (p=0.2) on every hidden layer.** Most direct
   fix for the train-val gap. Expected drop: 0.1–0.3 on val.
2. **[QUICK] Layer normalization after each hidden layer.**
   Compounds with dropout, stabilizes optimization, lets us
   experiment with larger LRs safely. Expected drop: 0.05–0.15.
3. **[QUICK] LR schedule: linear warmup (100–200 steps) +
   cosine anneal across the run.** Removes the manual session-
   chaining we did to find lr=0.00003. Expected drop: 0.05–0.1.
4. **[QUICK] Global gradient-norm clipping (replace element-wise
   `clamp(-5, 5)`).** Cleaner gradient direction; usually small
   win on its own but enables larger LRs.

If items 1–4 land, expected val: low 5s (5.0–5.3).

### Next tier (open new headroom)

5. **[BIG] Full DailyDialog ingest (~26k clean turns).** The
   ingester already exists (`src/bin/ingest_dailydialog.rs`); user
   just needs to provide `dialogues_text.txt`. 4× corpus size of
   clean, balanced human dialog. Single biggest data lever.
6. **[QUICK] Cleaner rule 11: emoji shortcodes → `__EMOJI__`.**
   Plus strip `**`/`*`/free `_` markdown. Re-run rebuild_vocab,
   fresh-init session.
7. **[MEDIUM] Hand-extract 10–20 high-quality multi-turn Rust /
   game-dev debugging threads from the Discord archive** and add
   them. Adds the "problem→fix" pattern the corpus currently lacks.
8. **[MEDIUM] Sinusoidal positional embedding** (replace
   scalar position feature). Cheap, qualitatively unlocks better
   long-context predictions.
9. **[QUICK] Increase WEIGHT_DECAY 1e-4 → 1e-3.** Pair with #1.
10. **[QUICK] Bigger seed-pair set targeting actual server
    topics** (extend `inject_seed::PAIRS` again). Cheap.

### Architecture tier (only if 1–10 plateau)

11. **[MEDIUM] Mini-batch training (B=16–64).** Speed win + better
    gradient estimates. Multi-day refactor. Also makes GPU
    worthwhile again.
12. **[MEDIUM] BPE / Unigram subword tokenizer.** Bigger refactor
    (output layer indexing changes, all binaries touched) but
    unblocks slang/names/typos.
13. **[BIG] Self-attention block (transformer-style)** between
    hidden layers. Much larger change, much more capacity. Only
    worth doing once we've exhausted MLP regularization gains.
14. **[MEDIUM] Wider context window (32 → 128).** Pair with
    sinusoidal positional embedding. ~4× input to first layer.

## 4. Diagnostic experiments

To find out which limit we're hitting before committing to a big
change, run these 5-epoch pilot sessions:

| Experiment | Setup | Reads as |
|---|---|---|
| Halve hidden size | `HIDDEN_SIZE = 384` | If val stays ~5.7 → capacity isn't the limit. If val rises → it is. |
| Double the corpus | Ingest DailyDialog into a copy and train fresh | If val drops to ~5.4 → data is the lever. |
| Freeze embedding | Don't train `embedding.weights` | If loss barely changes → embeddings aren't doing much; consider pretrained init or smaller `EMBED_DIM`. |
| Dropout-only | Add p=0.2 dropout, otherwise unchanged | Isolated effect of the single biggest regularization lever. |

Each is one fresh-init session (~1h). Stack them: each gives a clear
yes/no on a hypothesis.

## 5. Inference-side wins (no retraining)

Cheap, immediate quality bumps even without touching weights.

- **Repetition penalty.** Inside `generate`, track the last 16
  emitted tokens; downweight their logits by ~0.5× before
  `sample_top_k`. Directly fixes the "i love you i love you" failure
  mode we kept seeing.
- **Top-p (nucleus) sampling.** Replace `TOP_K_SAMPLE = 5` with
  cumulative-mass-0.9 nucleus. More naturalistic; avoids the
  always-the-same-5-tokens trap.
- **Minimum reply length.** Forbid emitting `</PERSON_0>` until at
  least 4–6 content tokens have been produced. Stops the "Bot: ." /
  "Bot: no." degenerate outputs at low val.
- **Temperature parameter.** Add a temperature scalar to the
  softmax in `generate`. Default 1.0; expose via env var. Lets the
  user dial creativity vs. coherence per session.
- **System prompt prepend.** Prepend a fixed token sequence like
  `"<PERSON_1> hello "` to bias the start of generation toward
  greetings rather than punctuation.

## 5b. What we tried 2026-05-20, what worked

### Cleaner rule 11 — emoji shortcodes → `__EMOJI__`
Shipped. Replaces `:name:` Discord shortcodes (and the `__EMOJI__`
placeholder is now whitelisted in `tokenizer::is_special_passthrough`,
sibling to `__URL__` and `__MENTION__`). Effect on the cleaned
corpus: only 3 tokens / 2 turns rewritten — most actual emoji noise
had already been stripped by earlier rules. `dialogs.bin` cache
version bumped to 3 so the tokenizer change re-parses on next run.

### Cosine LR schedule with linear warmup
Shipped. New CLI flags on `train`:
- `--lr-warmup-epochs N` (linear ramp 0 → peak over N epochs)
- `--lr-anneal-epochs N` (cosine peak → lr_min over N epochs from end of warmup)
- `--lr-min F` (cosine asymptote; lr stays here forever after anneal)

When `--lr-anneal-epochs 0` the trainer falls back to the legacy
multiplicative `--lr-decay`. 5 unit tests cover the math at corners
(warmup linear, cosine endpoints, tail at lr_min, warmup+cosine
composition). Empirically: replicates session 8's two-session
manual lr=0.0001 → 0.00003 step in a single run.

### Hidden-layer dropout
**Shipped after one full-day bug hunt.** First implementation cached
post-dropout activations (`a = scale * tanh(z)`) in `last_activations`,
which `compute_deltas_into` then used to compute the tanh
derivative as `1 - a²`. That's `1 - scale²·tanh(z)²`, not
`1 - tanh(z)²`. For p=0.2 (scale=1.25) and tanh(z)=0.8, the cached
activation hits 1.0 and the derivative collapses to zero — gradient
dies on the kept neurons. Training diverged (gradient norms grew
10× per epoch, val rose from 6.86 → 7.10 → 7.28 over 3 epochs).

Fix: cache *pre*-dropout activations in `last_activations`, apply
the mask separately in backward (`compute_deltas_into` multiplies
each layer's delta by the per-neuron mask after the tanh derivative).
After the fix, dropout p=0.1 trains stably (grad norms 17 → 30 → 36
over a 4-epoch session, no explosion).

Verdict: stable but not a win at p=0.1 with this corpus + model
size. Train still falls fast (4.06 by ep 1 of fine-tune, 2.99 by ep
4); val sits at 5.80–6.10. Train-val gap doesn't close more than
without dropout. The model has enough capacity to memorize the small
corpus regardless of dropout strength tested (0.05, 0.1, 0.2 all
overfit similarly past 2 epochs).

### Global gradient-norm clipping
**Implementation attempted, currently disabled** (`GLOBAL_GRAD_NORM_CLIP
= f32::INFINITY`). The clip ran on the sum-of-squares of `layer_deltas`
+ `input_grad` — i.e. the *post-activation gradient vectors*, not the
*weight gradients*. The conventional "clip-by-global-norm" operates on
weight gradients (`dL/dW` per layer, summed and L2-normed). Because the
weight gradient is `delta_out ⊗ input_in`, scaling delta by `s` also
scales the weight gradient by `s` — but the implied scale factor on
the *weight* gradient depends on `‖input_in‖` which differs per layer.
So scaling deltas uniformly does not produce a uniform weight-gradient
scaling, and the textbook clip semantics aren't preserved.

Empirically: clip=1.0 fired on 100% of steps (mean delta norm ~10 at
fresh init, growing to thousands during training), scaling gradients
down 11× and erasing the learning signal. clip=50.0 still fired on
84% of steps after one full epoch (delta norms grow into the hundreds
of thousands during training). Diagnostics are kept in the
`StepProfile` (mean ‖g‖₂, clip count, observations) for future tuning.

If we want this back: implement properly by materializing weight
gradients (`delta_out * input_in.T` per layer) and computing the
combined L2 norm across all layers' weight matrices + biases. That's
a non-trivial refactor since the current Adam loop computes
`delta * input` inline without storing the result.

### Best-checkpoint sidecar (already shipped 2026-05-19)
`model.bin.best.val` text sidecar persists the best-val across
sessions. New trainer reads it at startup ("Resuming best-val
tracking from model.bin.best.val = 5.6867"), and won't overwrite
`.best` unless the new run actually improves on it. Saved the
overnight val 5.69 several times today.

### Where val stands

| Vocab | Best val | Session | Notes |
|---|---|---|---|
| Pre-rule11 | **5.69** | session 8 (lr=0.00003 fine-tune from session 6 best) | 2026-05-19 |
| Rule-11 | **5.80** | fine-tune (lr=0.00003 + dropout 0.05 + cosine) | 2026-05-20, current `model.bin` |

`model.bin.best.rule11-val5.80` keeps an explicit named snapshot.
The two numbers AREN'T directly comparable — different vocabs mean
different denominators in the perplexity. Both are around perplexity
≈ 300.

### Today's net diagnostic conclusion

Reaffirms the earlier survey's call: regularization knobs (dropout,
schedule) don't close the train-val gap because the gap is dominated
by **corpus structure** (small, repetitive, off-topic). The
"single highest-ROI change" claim in §3 should be downgraded —
dropout alone isn't enough. The next experiments to try, in this
order, are:

1. **[QUICK]** A single fresh-init session at lr=0.0001 *without*
   dropout, *with* cosine, on the rule-11 corpus, as a clean
   baseline number on the current vocab. (Today's experiments mostly
   compared against pre-rule11 numbers, which is unfair.)
2. **[BIG]** DailyDialog ingest. Still the single biggest data
   lever. Path is already in code; just needs `dialogues_text.txt`.
3. **[MEDIUM]** Layer normalization between hidden layers. Was
   item Q7 in the survey; deferred today because it requires a
   v5 model format bump for the LN gain/bias params. With dropout
   bugs cleared, LN is the next regularization to try.
4. **[QUICK]** Add label smoothing (0.1) on cross-entropy. Trivial
   to add (one tweak in `compute_deltas_into`'s output-layer
   one-hot computation: replace `1.0` → `1.0 - smooth`,
   `0.0` → `smooth/vocab_size`). Soft target distribution often
   helps train-val gaps.

## 6. Operational footprint (kept for the morning briefer)

- Live bot is served from `model.bin` by the `sighurt-llm.service`
  systemd unit on the desktop (Tailscale `100.118.41.103:8088`).
- `sighurt-llm.path` watches `model.bin` for `IN_CLOSE_WRITE` and
  fires the oneshot `sighurt-llm-reload.service`, which restarts
  the inference server. Effectively: every training save rolls
  the bot to the new weights with ~3 seconds of downtime.
- `model.bin.best` + `model.bin.best.val` sidecar track the
  global best across sessions; trainer reads them on start and
  won't overwrite `.best` without an improvement.
- `cargo run --release --bin clean_corpus` is idempotent and
  always safe to re-run; it writes via tmp+rename. Always
  `cp data/dialogs.txt data/dialogs.txt.pre-clean` first when
  about to invoke any rule change.

## 7. Key files

| Path | Why it matters |
|---|---|
| `src/tokenizer.rs` | The whole vocab story starts here. `is_special_passthrough` whitelist controls which tokens survive splitting. |
| `src/dialogs.rs` | Corpus parsing, vocab construction, bincode cache. `DEFAULT_VOCAB_CONTENT_CAP = 3000` is the knob for vocab size. `CACHE_VERSION` must bump when tokenizer behavior changes. |
| `src/neural_network.rs` | All architecture + training constants live at the top (`EMBED_DIM`, `HIDDEN_SIZE`, `NUMBER_OF_HIDDEN_LAYERS`, `CONTEXT_WINDOW`, `MAX_TARGET_TOKENS`, `GRAD_CLIP`, `WEIGHT_DECAY`). `train_one_epoch` is the training loop. `train_step` is one Adam update. |
| `src/teacher.rs` | Backprop kernel. Cache-friendly chunked layout. |
| `src/persist.rs` | v4 model format; vocab-hash safety; atomic save. Don't break the version dispatch. |
| `src/bin/clean_corpus.rs` | Adds rule X here when corpus needs another scrub. Don't forget to also patch `convert_discord::sanitize_for_corpus` so new ingests are clean from the start. |
| `src/bin/train.rs` | CLI flags, sidecar-aware best-val tracking. Where the LR schedule would land. |
| `docs/optimization-roadmap.md` | Earlier roadmap, speed-focused. Items 1, 4–6 are still relevant (cache-friendly backward already done; mini-batch and GPU shaders still open). |

## 8. What I'd do if I had one week

1. Day 1: ship dropout + layer norm + global grad-norm clip + cosine
   LR schedule with 200-step warmup. Single PR, all four items
   compound. Run a session to measure.
2. Day 2: cleaner rule 11 (emoji + markdown) + bigger seed pair set.
   Rebuild vocab, fresh-init, train.
3. Day 3: implement sinusoidal positional embedding. Train.
4. Day 4: ask the user for DailyDialog, ingest, fresh-init, train
   overnight.
5. Day 5: review samples, write up val/perplexity numbers, decide
   whether to push for mini-batch refactor next week.

Expected val after a focused week: **mid-to-low 5s** (5.0–5.3),
maybe **high 4s** if DailyDialog lands. Perplexity 150–200 — small
model territory but coherent for short conversational replies.
