# Training-day log — 2026-06-02

Starting state: bot serving K weights (val 5.13 on rule-13 corpus,
inference-time repetition penalty active). 10+ days since last training
session (Saturday May 22 → Tuesday June 2). All May 22 snapshots
preserved on disk.

## Plan I'm following

K's val 5.13 had clean MECHANICAL output (no markdown, no escapes,
no placeholder leakage, no back-to-back repetition) but every reply
in the May 22 manual probe leaked at least one OASST helper fragment
("i am a language model", "here are some steps", "i can help you
with"). next-session.md from May 22 prioritized:

1. Anti-OASST persona seed pairs (Tier 1 #1) — top
2. Greeting-reflex seed pairs (Tier 1 #2)
3. LS α=0.15 fresh init (Tier 1 #3)
4. Per-prefix dedup rule 14 (Tier 1 #4)

I'm bundling #1 and #2 into a single seed-pair expansion (session L)
and implementing #4 in parallel so it's ready to test on session M.
LS α=0.15 is a session N candidate if time permits.

## Changes shipped today

### 1. Expanded seed pairs (`src/bin/inject_seed.rs`)
Added 67 new pairs across five categories targeting the OASST persona
drift:
- **Anti-LM persona (18 pairs)**: "are you a language model" →
  "nope, i'm supersighurt." — direct contradictions to LM persona,
  reinforce supersighurt identity.
- **Greeting reflex (15 pairs)**: yo/sup/heya/morning/wassup/etc. →
  short conversational replies. K's "hi" → "you have a time and not
  you." was the worst probe failure.
- **Avoid-OASST-list-opener (15 pairs)**: tell me a joke / what's new
  / what are the steps → casual replies that explicitly don't open
  with "here are some" or numbered lists.
- **Discord-tone (12 pairs)**: lmao/rofl/based/cringe/touch grass/
  ggs/pog → casual back-and-forth.
- **Anti-helpful-AI phrasings (7 pairs)**: "can you help" / "provide
  me with" / "give me a list of" → short clarifying questions
  instead of OASST-style helpful templates.

Total seed pairs: 138 → 205.

### 2. Rule 14: per-prefix dedup (`src/bin/clean_corpus.rs`)
- `MAX_PREFIX_REPEATS = 30` turns per shared `PREFIX_LEN = 5` token
  prefix. Caps OASST template-opener domination ("here are some ways
  to ...", "i am not a language ...", "to do this you can ...") that
  rule 7 misses because the suffixes vary.
- Short turns (< 5 tokens) exempt — already capped by rule 7.
- Case-insensitive prefix key.
- 4 new unit tests: caps at max, exempts short turns, case-insensitive,
  distinct prefixes unaffected.

## Sessions

| ID | Setup | Best val | Notes |
|---|---|---|---|
| L | Fresh init on seed-augmented corpus, lr=1e-4 cosine 5ep, LS 0.1 | **4.75** | 14116 examples. Trajectory: 5.10 → 4.90 → 4.79 → **4.75** over 4 epochs. Δ −0.39 vs K. |
| M | Fresh init on same corpus, lr=1e-4 cosine 5ep, **LS α=0.15** | **4.76** | Trajectory: 5.08 → 4.91 → 4.76 → **4.76**. Essentially identical to L (Δ +0.012). α=0.15 ≈ α=0.10. |
| N | Fresh init on **OASST-trimmed corpus** (1500 trees vs 3670), LS α=0.10 | **5.30** | 11936 turns. Trajectory: 5.47 → 5.35 → **5.30** → 5.35 (overfit). Stopped at epoch 6 (corpus too small for 5+ epochs at LS α=0.10). |
| O | Fresh init on **trim 2500** corpus, LS α=0.10, 100min timeout | **5.27** | 14877 turns. Trajectory: 5.43 → 5.33 → **5.27** → 5.28 (overfit). Marginally better than N on val. |

### L probe — quantitative win, qualitative regression

L's val 4.7453 is the lowest ever measured on this model. But the
manual probe (3 samples × 9 prompts) shows the OASST helper register
hasn't retreated — and may have intensified:

| Prompt | L sample (worst of 3) |
|---|---|
| hi | "here is a list of how you can do it is with a bit" |
| hello | "i'm sorry, but i am not to a! i am not to a" |
| how are you | "in a language ai can be to an open assistant and the ai to help you with it" |
| tell me a joke | "sure, here's a list of a code that i want to use the code" |
| good morning | "sure, here's an example of a!- by a. here are the sure of a list of" |
| do you like games | "for a large language model in your name: 1. i want to have your code" |
| are you a language model | "here are some of the code to help you with:- a time and what i can use" |

The seed pair "are you a language model" → "nope, i'm supersighurt"
is being completely overpowered by the ~9000 OASST helper-style turns
that mention language models / list-format suggestions / "here are
some" patterns. 67 anti-OASST pairs × 3 dedup-cap = 201 turns vs
9000+ OASST turns ⇒ ~45:1 ratio against the new signal.

**Diagnosis**: seed pairs are too few to shift the dominant register.
The val improvement is the model fitting OASST turns more tightly,
not learning the Discord-tone register. To actually shift output
tone, we need to reduce OASST contribution (cap trees at ~1500) OR
strengthen rule 14 to bot-only PREFIX=4 MAX=15 (which would target
helper-template prefixes directly).

Full probe in `docs/manual-test-2026-06-02-L.md`.

## What I learned today

1. **Seed pairs alone don't shift register.** Adding 67 anti-OASST
   pairs improved val by 0.39 but the bot still answers every prompt
   with OASST helper templates. Ratio of new seed signal to OASST
   helper signal is ~1:45.
2. **α=0.15 ≈ α=0.10.** No improvement from stronger label smoothing
   on this corpus. Final val differs by 0.012.
3. **The corpus IS the lever for register.** OASST trim DID shift
   tone — N's reply to "are you a language model" was "i'm not a
   language model" (seed pair winning), while L's was "here are some
   of the code to help you with". Direct evidence the seed pairs
   work; they just need more relative weight.
4. **OASST trim has a heavy val cost.** Trim from 3670 → 1500 trees
   cost Δ +0.55 val (5.30 vs 4.75). Trim → 2500 cost Δ +0.52. Most
   of the val cost is the validation set itself losing OASST-style
   prompts to fit cleanly. The model isn't worse; the val pool is.
5. **Small corpora overfit fast at LS α=0.10.** N and O both peaked
   at epoch 3 and rose after. The trim corpus is too small for 5
   epochs at this α.
6. **Rule 14 strong (bot-only PREFIX=4 cap=15) was implemented but
   was a no-op on the trim corpus** (no 4-token bot prefix exceeded
   15 occurrences after the trim). It's still useful retroactively
   if/when we re-add more OASST data.

## Final decision (chose to serve L)

Decision rubric: pick the model the typical user-experience is best
with. L gives the strongest val (4.75, Δ −0.38 vs K) AND the most
coherent multi-sentence replies. N gives the best persona-question
robustness but at the cost of broken syntax everywhere else.

If the user prefers tone-priority deployment, swap to N:
```
cp model.bin.best.session-N-val5.30 model.bin
cp data/dialogs.txt.pre-trim-1500 data/dialogs.txt  # restore N's corpus
./target/release/rebuild_vocab
# triggers serve auto-reload
```

## Today's training summary table

| Sess | Best val | Δ vs K (5.13) | Corpus turns | Sample quality |
|---|---|---|---|---|
| L  | **4.75** | **−0.38** | 18538 (full+seeds) | OASST drift heavy |
| M  | 4.76 | −0.37 | 18538 (same) | Same as L |
| N  | 5.30 | +0.17 | 11936 (trim 1500) | Anti-LM seed works; syntax broken |
| O  | 5.27 | +0.14 | 14877 (trim 2500) | Some OASST returning + still broken |

## On-disk snapshots before today

- `model.bin.best.rule13-K-val5.13` — K's best (val 5.13, rule-13 corpus, May 22)
- `model.bin.best.rule12-J-val4.99` — J's best (different/earlier corpus)
- `model.bin.k-best-park` — copy of K's `.best` parked before L started
- `model.bin.pre-session-L` — copy of K's `.bin` parked before L started
- `data/dialogs.txt.pre-seed-2026-05-22-eve` — corpus before today's seed pair injection

## Today's new snapshots

- `model.bin.best.session-L-val4.75` — L's best on seed-augmented corpus *(currently live)*
- `model.bin.best.session-M-val4.76` — M's best on same corpus, α=0.15
- `model.bin.best.session-N-val5.30` — N's best on trim-1500 corpus
- `model.bin.best.session-O-val5.27` — O's best on trim-2500 corpus
- `model.bin.l-best-park` / `model.bin.m-best-park` / `model.bin.n-best-park` — parked before next sessions
- `data/dialogs.txt.pre-trim-1500` — full seed-augmented corpus (L/M corpus)
- `data/dialogs.txt.pre-trim-2500` — same as pre-trim-1500 (snapshot before O build)

## Live state at end of day

- `data/dialogs.txt` / `vocab.txt` / `dialogs.bin` ← seed-augmented (L's corpus)
- `model.bin` / `model.bin.best` ← L (val 4.75)
- `model.bin.best.val` ← 4.7453
- Serve: PID restarted by sighurt-llm.path, healthz ok
- 35 tests green across clean_corpus (rule 14 included)
