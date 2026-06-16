# Manual test patterns — 2026-05-22 morning

Live bot serving val 5.06 weights (OASST-augmented corpus, three-stage
fine-tune cascade A→B→E from 2026-05-21). 10 short probes via the
HTTP `/chat` endpoint. Each prompt got a fresh channel_id to avoid
RAG context bleed.

| Prompt | Reply | Issue category |
|---|---|---|
| hi | "here are some examples of help and get the information of the" | wrong register (greeting → OASST opener) |
| hello | "here?!****\*\*\*\*\*\*\*\*\*\*\*\*\*\*#\*\*\*\* benny\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*" | asterisk-spam, name leak ("benny") |
| how are you | "any of 3 and 3 can have to a information, but it is not a few" | semantic incoherence + number artifacts |
| what is rust | "in a play? that is not have a great, and the popular who are not in the but" | doesn't engage the topic |
| tell me about yourself | "here is an example of the following between the world in: the" | persona drift — should say "i'm supersighurt" |
| what are you | "there can be a lot that are used in the" | same — should self-identify |
| i'm working on a chatbot in rust | "i am not have the code but i am i am not in rust, i can be a way of a bit who is there a language model but it is not so to the most of a short" | best of the set — multi-clause, has "rust" and "language model" content; repeats "i am" 3× |
| can you help me debug this code | "sure you here's a few specific code, you can help you," | reasonable opener, then degenerates |
| what's the weather like | "in the world was a way in the play and this is that it was like to the feel like" | no semantic engagement, "way", "play", "feel like" rambling |
| good morning | "the go, i will be a language model:- 1" | persona drift + bullet-list artifact |

## Failure patterns, ranked by frequency

### 1. OASST "here are some / here is an example" overuse (3/10 replies)

The OASST data has a strong opener distribution. Once the model
learned "user asks something → I begin with 'here are some / is an'",
it generalized that to every short input, including bare greetings.

**Fix:** rebalance the corpus toward conversational chat. Options:
- Cap OASST contribution at ~1500 trees (vs the full 3670 today)
- Inject many more greeting-flavored seed pairs (today's
  `inject_seed.rs` has ~10 greeting variants — add 30-50 more)
- Both

### 2. Markdown noise leaking from OASST text (asterisks, hyphens, "1:-") (3/10 replies)

Cleaner rule 11 only handled `:emoji:` shortcodes. OASST replies
contain bold `**text**`, italics `*text*`, bullet `- item`,
em-dashes `--`, and numbered lists `1.` / `1:` that all survive
into training and then get emitted at generation. The "benny"
appearance was bracketed by `*****` runs that the model treats as a
valid token sequence.

**Fix:** Cleaner **rule 12** — strip leading/standalone `*`, `**`,
collapse `-{2,}` and `*{2,}` runs, drop turns that are pure markdown
decoration. Requires corpus re-clean + vocab rebuild + fresh-init
training (vocab will reorder).

### 3. Persona drift toward "I am a language model" (2/10 replies, both identity-probes)

OASST replies frequently start "as an AI" / "I am a language model".
The ingester (`scripts/ingest_oasst.py`) drops "as an ai" via
REFUSAL_PATTERNS but doesn't catch "i am a language model". So a
slice of OASST persona slid through.

**Fix:** Tighten REFUSAL_PATTERNS to also strip "i am a language
model", "i'm an ai assistant", "as a language model". Re-ingest +
reclean + retrain. Or: weight the seed-pair persona by injecting it
multiple times into the corpus.

### 4. Within-reply token repetition ("i am i am", "the the") (4/10 replies)

Model has no repetition penalty at sample time. Survey doc Tier 5
already flags this as a quick inference-side fix.

**Fix:** Add `repetition_penalty` to `serve::handle_chat` /
`neural_network::generate`: track the last N emitted token IDs;
downweight their logits by 0.5–0.7× before `sample_top_k`. ~15 line
change in `generate`. Zero retraining cost.

### 5. Number / list-marker leakage ("1", "3", ":- 1") (2/10 replies)

Similar root cause as #2 but specifically numeric. OASST replies use
"1.", "2.", "First, ...". After punctuation-splitting tokenization
these become bare "1", "2" tokens.

**Fix:** Same as #2 (markdown cleanup), plus maybe forbid bare
digit-only tokens in `forbidden_emit_ids` if they're appearing
gratuitously.

### 6. No conversational greeting reflex

The bot doesn't return "hi" with "hi". The cleaned Discord corpus
had short greeting exchanges, but only ~10 are explicit seed pairs.
With ~14k OASST turns now in the mix (~3.5× the Discord half),
greetings are a vanishing minority.

**Fix:** Add 40-60 greeting / closing / acknowledgment seed pairs
to `inject_seed.rs::PAIRS`. Run `inject_seed`, re-clean, retrain.

## What's working

- Multi-clause replies are common at longer prompts (#7 was the best
  result — 4 clauses, contractions, topical content).
- `__MENTION__` / `__URL__` never appear (yesterday's regression fix
  holds).
- No "i love you i love you" Costco-loop pattern (the prior failure
  mode is gone — credit OASST diversity + the dedup cap).
- The model now USES words from the technical domain ("rust",
  "code", "language model", "debug") — pre-OASST it didn't.

## Recommendations for today's training cascade

Per yesterday's `next-session.md` Tier 1: run the LS-from-step-0
cascade. Won't fix the qualitative issues above (those are corpus
problems, not training problems), but will give us a measured
comparison point — does LS help epoch 1 enough that the full cascade
ends below the current val 5.06?

After cascade: if time, ship cleaner rule 12 + a chunk of greeting
seed pairs and run one more fresh-init session to see if sample
quality jumps.
