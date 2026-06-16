# Next session — short brief

End of 2026-06-09. Bot is live serving **L weights** (val 4.7453,
`model.bin.best.session-L-recorded`). Today's session P attempted
seed expansion + rule 14 retune; **P regressed** vs L on both val
(+0.24) and probe quality. Rolled back to L. Details in
`docs/training-day-2026-06-09.md`.

## What I tried today (and why it failed)

| Knob | Before | After | Impact |
|---|---|---|---|
| Seed pairs | 205 (×1 inject) | 467 (×3 inject, rule 7 caps at ×3) | seed/OASST ratio shifted 1:45 → 1:3 |
| Rule 14 | PREFIX=4 MAX=15 (37 drops) | PREFIX=3 MAX=10 (452 drops, 8.6% bot turns) | targets helper templates correctly |
| Final val | 4.75 (L) | 4.99 (P, epoch 4) | +0.24 worse |
| Final register | OASST drift on every prompt | OASST drift **+ fragmented**, less coherent | strictly worse |

**Root cause analysis**: Rule 14 at PREFIX=3 MAX=10 dropped 452 OASST
helper-template bot turns. The model learned the *shape* of helper
templates ("here is a …", "in a …", "you can …") but no longer had
enough complete end-to-end completions to generate them coherently.
The seed expansion helped *some* — epoch-4 sample for "Hey there!"
came back "good and what about?" (clean conversational register) —
but only on prompts that pattern-match seed-pair user turns.

**Lesson**: Two changes at once destroyed attribution. The seed
expansion alone may be net-positive; rule 14 retune is probably
net-negative on its own.

## Things I'd try first when you're back (ranked)

### Tier 1 — fast follow-ups (single-variable experiments)

1. **Seed expansion alone** (no rule 14 retune). Revert rule 14 to
   PREFIX=4 MAX=15, keep the 467 expanded pairs, train fresh. Tests
   the "more seed signal" hypothesis cleanly. 2.5h training session.
   Corpus build:
   ```sh
   # revert clean_corpus.rs rule 14 constants first
   git checkout -- src/bin/clean_corpus.rs  # OR edit MAX→15, PREFIX→4 manually
   cp data/dialogs.txt.pre-session-P-2026-06-09 data/dialogs.txt
   ./target/release/inject_seed
   ./target/release/inject_seed
   ./target/release/inject_seed
   ./target/release/clean_corpus
   ./target/release/rebuild_vocab
   ```

2. **Rule 14 alone** (no seed expansion). Keep PREFIX=3 MAX=10, but
   skip the seed expansion. Tests whether the rule-14 cut alone is
   net-positive. Useful for ablation but probably not deployment.

### Tier 2 — bigger bets

3. **Bump model capacity.** Val plateaued ~5.0 for two consecutive
   sessions on a 10M-param 4×768 tanh MLP. Probe quality is similar
   between L (val 4.75) and P (val 4.99) — both produce broken OASST
   templates. The val improvement is fitting helper templates more
   tightly, not understanding language. We're at the architectural
   ceiling. Plan: bump HIDDEN_SIZE 768 → 1024 OR add a 5th hidden
   layer. Either needs v5 model format (current is v4) — add a `len`
   field to the layer-stack section so the loader can handle variable
   layer counts.

4. **Layer normalization.** Deferred from 2026-05-19. With LS as the
   working regularizer, LN is the next knob. Needs v5 model format
   for LN gain/bias params. Pair with #3.

5. **Subword tokenizer (BPE).** Larger refactor. Would help slang /
   names / typos. Tier 2 because every data lever still works with
   the punctuation-splitting tokenizer (and breaks at the same
   capacity ceiling).

### Tier 3 — investigated today, parked

- **Rule 14 PREFIX=3 MAX=10**: net-negative. Damages coherence by
  cutting 8.6% of complete helper-template examples without filling
  the gap with seed-pair variety strong enough to take over.
- **Seed expansion to 467 pairs (×3 inject)**: probably net-positive
  on its own but masked by rule 14 today. Re-test as #1.
- **Two changes at once**: never again. Single variable per session.

## What's blocking me / needs you

- **DailyDialog dataset file**. Same blocker for the fifth day in a
  row. Drop `dialogues_text.txt` in `data/`.
- **Decision: pursue capacity bump (v5 format) or keep iterating on
  10M MLP?** Probe quality suggests the latter is dead. v5 format is
  ~1 day of refactor work.
- **Sign-off on uncommitted changes** — accumulating across days:
  - `src/bin/clean_corpus.rs` — rules 12, 13, 14 + retune (this session)
  - `src/bin/inject_seed.rs` — 467 pairs (was 205 going in)
  - `src/gpu.rs` — `zeros_device` → `uninit_device` (sibling crate API change)
  - `data/dialogs.txt` — restored to L's seed-augmented corpus (post-rollback)
  - all model.bin snapshots (now blocking git push since 2026-05-21)

## Operational snapshot

- Bot live, healthz ok, val 4.75 (L) weights loaded by serve.
- Serve restarted at 11:30 against L's corpus (RAG indexes 18538 turns).
- 35 tests green in clean_corpus.
- All P-session snapshots preserved:
  - `model.bin` / `model.bin.best` ← L (val 4.7453, current)
  - `model.bin.best.session-L-recorded` (manual restore target)
  - `model.bin.session-L-live` (in-memory state captured)
  - `model.bin.best.session-P-val4.99` (P final, for reference)
  - `data/dialogs.txt.pre-session-P-2026-06-09` (L's pre-P corpus, current corpus too)
  - `data/dialogs.txt.session-P-corpus` (P's corpus, archived)
- Sidecar `model.bin.best.val` = 4.7453.

Five sessions across two and a half weeks. Two regressions in a row
on the corpus axis. Time to consider whether the model — not the data
— is the bottleneck.
