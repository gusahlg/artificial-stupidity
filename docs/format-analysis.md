# Corpus format analysis

## Current format

```
<SEC>
<PERSON_1> hi how are you </PERSON_1>
<PERSON_0> doing well thanks </PERSON_0>
<PERSON_2> nice to hear </PERSON_2>
<SEC>
<PERSON_0> a brand new exchange </PERSON_0>
<PERSON_1> what's up </PERSON_1>
```

- **Sections** separated by `<SEC>`. Each section is one coherent exchange.
- **Turns** wrapped with `<PERSON_N> … </PERSON_N>` tags. N is **section-local**: a new mapping every section, PERSON_0 reserved for the bot when it speaks. Same Discord user can be PERSON_2 in one section and PERSON_1 in another — the tag distinguishes speakers, it doesn't identify them.
- **Tokens** inside a turn are whitespace-separated. The shared `tokenize()` lowercases and splits punctuation as its own tokens.
- **Single text file** `data/dialogs.txt`. Re-parsed at startup.

## Where the format works well

1. **Inspectable.** A human can scroll the file and immediately see what the model is training on. That made the "wait why is it saying brick" investigation cheap last night.
2. **Append-only friendly.** Every ingestor (`convert_discord`, `inject_seed`, `ingest_dailydialog`, `ingest_tinystories`) is a one-shot binary that opens the file in append mode and writes self-contained `<SEC>` blocks. No mutex, no schema migration.
3. **Tooling-agnostic.** Standard Unix tools work: `grep -c "<SEC>"`, `wc -l`, etc. We've used this repeatedly for debugging.
4. **Per-section locality maps to the algorithm.** The "best section → most similar turn → next turn" matcher in `machine_learning.rs` operates on sections; the format mirrors that unit of work exactly.
5. **`<PERSON_N>` tags double as model tokens.** The same string `<PERSON_0>` that delimits a turn in the corpus also occupies a vocab slot in the model, so the model can emit it. Generation forces `<PERSON_0>` to prime the bot's voice and stops at `</PERSON_0>`.

## Where it pays an ongoing cost

1. **Parse overhead at every startup.** ~10k sections, ~25k turns, ~50k content tokens — re-tokenized on every `serve` boot. Takes ~50-100 ms today; will grow with corpus.
2. **Format duplication.** `<PERSON_N>` and `</PERSON_N>` get written into every turn — that's 4-30% of the file's bytes depending on average turn length.
3. **Vocab footprint of PERSON tags.** Each distinct PERSON id used anywhere in the corpus burns two vocab slots (open + close). Today we use PERSON_0…PERSON_12 → 26 slots out of 3000 content slots = 0.9%. Fine at this scale; would matter at PERSON_100+.
4. **No structured metadata.** Timestamps, original Discord IDs, source-of-data tags (Discord vs seed vs TinyStories vs DailyDialog) all get lost at conversion time. We can't, e.g., "train only on the last 30 days" without re-running the ingestors.
5. **Whitespace-token coupling.** A user message that happens to contain the literal substring `<PERSON_3>` would be mis-parsed. We currently sanitize `<` → `(` on ingest, which is robust but lossy.
6. **No fast random access.** Section retrieval is linear; you can't jump to "section 5,237" without scanning from the top.

## Alternative formats considered

### A. JSONL (one section per line)

```json
{"sec_id": 0, "turns": [{"pid": 1, "text": "hi"}, {"pid": 0, "text": "hi back"}]}
{"sec_id": 1, "turns": [{"pid": 0, "text": "new exchange"}]}
```

Pros: structured metadata is natural (add fields freely), random access by line, ecosystem (`jq`, etc.).
Cons: parsing JSON is heavier than `split_whitespace`, less readable in a pager, loses the "PERSON tag is also a vocab token" identity (would need to re-emit it during training-context assembly).

### B. Custom binary, pre-tokenized

Tokens stored as `u16` ids. Sections delimited by 0xFFFF. Per-turn header: `[u8 pid][u16 token_count]`.

Pros: tiny on disk (~6 bytes/turn header + 2 bytes/token), instant load (just `mmap`), no re-tokenization, can `mmap` once at startup.
Cons: opaque to humans, needs a `dump_dialogs` tool to inspect, vocab and corpus become coupled (binary file is invalid if vocab changes), schema migration burden.

### C. Hybrid: human-readable canonical + binary cache

Keep `data/dialogs.txt` as the source of truth. Add `data/dialogs.bin` as a derived cache (timestamp + content hash). Trainer/serve check that the cache is newer than the txt; if not, regenerate.

Pros: best of both. Human can edit `dialogs.txt`, the binary cache makes startup nearly free.
Cons: two files to keep in sync, cache invalidation logic is a small bug surface, doubles disk usage.

### D. SQLite

One table per turn with columns `(section_id, position_in_section, person_id, text)`. Indexes on `section_id` and `person_id`.

Pros: rich queries ("all bot turns in the last 7 days"), structured metadata, ACID for live inserts (the `RagStore::insert_live` followup needs something like this), perfectly indexed access.
Cons: SQLite dep adds ~1MB to the binary, schema migration overhead, parsing rows is still slower than `split_whitespace` on a pre-tokenized binary, less inspection-friendly than text.

## Recommendation

**Keep the current text format. Don't redesign.** Reasons:

- The format is *not* the bottleneck. Training time is dominated by Adam updates on the output layer, not by corpus parsing. Speeding parsing 100× doesn't move the needle.
- The append-only one-shot ingestor model is genuinely valuable. Every new dataset (DailyDialog, TinyStories, future Reddit dumps) gets its own small binary that writes self-contained `<SEC>` blocks. Schemas don't need to evolve.
- We just successfully ingested four data sources (Discord, seed, DailyDialog binary, TinyStories) without touching the format — that's the design working.
- The "<PERSON_N> tags are also vocab tokens" identity is structurally clean. The model learns the tag, emits it, the parser recognizes it. One representation, three uses.

**What's worth doing later** (small effort, real wins):

1. **Add a `data/dialogs.bin` cache** for serve startup. Don't change the source format. Just cache the parsed `Vec<Vec<Turn>>` as bincode'd bytes; regenerate when `dialogs.txt` is newer. Wins: faster `sighurt-llm` restart (matters during overnight checkpointing loops). ~50 lines of code.

2. **Tag the source on ingest** as a comment line above each `<SEC>` block:

   ```
   # source=tinystories
   <SEC>
   <PERSON_0> ... </PERSON_0>
   ```

   The parser already ignores tokens outside `<SEC>` / `<PERSON_N>` brackets, so this is backwards-compatible. Lets us audit "how much of training came from where" and selectively rebuild.

3. **Live capture should write to a structured sidecar** rather than only the corpus. Right now Discord live messages append to `data/channels/<guild>/<chan>.tsv` on the Pi (great — has metadata) and then `convert_discord` collapses that into the lossy text form. The TSVs are the real archive; the txt is derived. Keep that invariant.

## Optimization audit notes (relevant to the format question)

- **Forward pass**: dispatches to `Gpu::matvec`. CPU fallback uses `cpu_matvec` which is rayon `par_iter_mut` over output rows. ✓
- **Backward pass** (`compute_deltas` in `src/teacher.rs`): I parallelized both the hidden-layer delta loop and the input-grad loop with rayon last night. ✓
- **Adam update** (`Network::train_step`): outer loop over rows is `par_chunks_mut` with three zipped weight/m/v slices. ✓ Inner k-loop is sequential; could be SIMD-ized with `wide` (f32x8) for a ~1.5-2× win, but it's memory-bound so the gain is bounded by DRAM bandwidth, not vector width.
- **`apply_grad_adam` on embeddings**: sequential. Called once per training step, touches ~16-32 distinct token IDs. Fast enough not to matter (~10 µs per call).
- **The actual training bottleneck**: my hypothesis (after profiling-by-arithmetic) is that the `compute_deltas` hidden→output backward step accesses `next_w[k*cols + j]` with stride `cols`, which is cache-unfriendly. Loop-reordering with k outer (cache-friendly) is plausibly a 3-5× win on backward, which is ~30% of the training step. Net: maybe 1.5-2× total training speedup. Worth doing in a focused session with timing instrumentation; not worth doing under time pressure with no profiler in the loop.

## Concrete next steps if you want to act on this

Order of payoff:

1. **TinyStories ingest** (done this session) → corpus is ~10× larger → more grammar signal.
2. **Cache-friendly backward** in `compute_deltas`: rewrite the inner loop with k outer, parallelize over j chunks. ~30 lines, possible 1.5-2× total training speedup.
3. **Bincode cache for parsed Data**: cuts `sighurt-llm` restart from "seconds" to "instant". Matters during the overnight restart loop.
4. **SIMD inner Adam loop** via `wide`: ~1.5× speedup on Adam's inner update. Adds one small dep.
5. **GPU compute shader for Adam update**: real engineering project. Could be 5-10× but bug surface is large.

None of these justify a corpus-format redesign.
