# artificial-stupidity / SuperSighurt

SuperSighurt is a Discord-focused 1.1B-parameter Llama transformer. It is
continued from an immutable TinyLlama 1.1B Chat v1.0 lineage on the complete
cleaned Discord archive, test-scored code examples, attributed encyclopedia and
web-grounding examples, curated Sig personality demonstrations, and a pinned
general-conversation replay set, then
served as an f16 GGUF by the Rust/Vulkan `tensor-ash` runtime. The runtime is
pinned to tensor-ash commit `57bbf3884daf8e483650fdace37144021b17b93d`
(v2.0.0) so builds do not silently change underneath a trained artifact.

The original small, from-scratch tanh-MLP remains in the repository as a
legacy research implementation and rollback target. Production uses
`serve_llama`, not `serve`.

The transformer is roughly 100 times larger than the roughly 11M-parameter legacy
network. Its approximately 2.2 GB f16 GGUF also leaves safe headroom on the
production server's 4 GB GTX 1650; a larger unquantized family would not fit the
current tensor-ash deployment path.

## Layout

- `src/main.rs` — interactive chat REPL (`rust_fun` binary)
- `src/bin/train.rs` — standalone auto-trainer (`train` binary)
- `src/bin/serve.rs` — legacy MLP HTTP server (`serve` binary)
- `src/bin/serve_llama.rs` — production TinyLlama GGUF server through tensor-ash
- `src/bin/convert_discord.rs` — ingest a Discord export into the dialog format
- `src/bin/ingest_dailydialog.rs` / `ingest_tinystories.rs` — corpus loaders
- `src/bin/inject_seed.rs` — splice a fixed set of "seed" Q/A pairs into the corpus
- `src/bin/clean_corpus.rs` — filter junk turns (URL-only, emoji-only, role-ping, repeats), drop monologue / single-speaker / sub-2-turn sections, then deterministically shuffle so the val-tail split is a random sample. Idempotent
- `src/bin/rebuild_vocab.rs` — regenerate `vocab.txt` without retraining
- `src/tokenizer.rs` — lowercase + punctuation-splitting tokenizer
- `src/text_utils.rs` — small text helpers shared by `clean_corpus` and `convert_discord` (URL detection, kept in sync between writer and filter)
- `src/embeddings/mod.rs` — trainable word embedding table (with Adam state)
- `src/neural_network.rs` — model, forward, generation, training loop
- `src/teacher.rs` — softmax + cross-entropy backprop (incl. input gradient for embeddings)
- `src/persist.rs` — save/load `model.bin` (v5: weights, biases, Adam moments, `adam_step`, vocab hash, hyperparams, optional LayerNorm)
- `src/gpu.rs` — Vulkan/CPU backend dispatch
- `src/machine_learning.rs` — section-similarity + embedding-cosine teacher lookup
- `src/rag.rs` — embedding-cosine retrieval store, used by `serve`
- `src/dialogs.rs`, `src/memory.rs` — corpus + vocab loaders (with a bincode cache)
- `data/dialogs.txt` — training corpus
- `data/dialogs.bin` — ignored bincode cache of the parsed corpus (auto-invalidated on edit)
- `vocab.txt` — derived vocabulary (regenerated from corpus on every run)
- `model.bin` — saved embeddings + weights + biases + Adam state (created on first run)
- `model.serving.bin` + `data/dialogs.serving.txt` — pinned pair the live server reads (see `scripts/promote_model.sh`)
- `scripts/refresh_corpus.sh` — weekly incremental Discord→corpus refresh (systemd timer)
- `scripts/rebuild_full_corpus.sh` — full corpus rebuild from raw sources
- `scripts/promote_model.sh` — promote trained model + corpus snapshot to serving
- `training/export_sft.py` — leakage-safe canonical-corpus → chat SFT export
- `training/generate_persona.py` — write the reviewed 140-example Sig personality/context set
- `training/build_auxiliary_data.py` — build pinned code/wiki/web/personality splits
- `training/preflight.py` — fail-fast paid-GPU/data/revision/CUDA checks
- `training/train_llama.py` — full-parameter bf16 SFT with replay and quality gates
- `training/evaluate_sampling.py` — compare safe variation settings on the final HF model
- `training/conversation_acceptance.py` — 26-turn live-endpoint conversation suite (ollama plays the human)
- `training/make_repair_data.py` — author the curated sig-v2.1 behavior-repair demonstrations
- `training/run_on_runpod.sh` — reproducible Runpod preflight, training, and conversion
- `training/convert_to_gguf.sh` — pinned llama.cpp f16 GGUF conversion
- `scripts/promote_llama_model.sh` — hash-check, real-generation check, atomic promotion
- `scripts/apply_recommended_sampling.sh` — apply the evaluate_sampling winner to the live env
- `scripts/watchdog_probe.sh` + `systemd/sighurt-llm-watchdog.*` — restart serving when GPU-context loss bricks generation

## Build

```sh
cargo build --release
```

Produces in `target/release/`:

- `rust_fun` — interactive chat REPL
- `train`    — auto-trainer
- `serve`    — legacy MLP HTTP inference server
- `serve_llama` — production tensor-ash transformer HTTP server
- assorted corpus ingestors / utility binaries

## Transformer training and promotion

Rebuild and export the complete corpus before renting a GPU:

```sh
nix develop -c scripts/rebuild_full_corpus.sh
nix develop -c python3 training/export_sft.py
python3 training/generate_persona.py
# Run in a Python environment containing Hugging Face datasets:
python3 training/build_auxiliary_data.py
PYTHONPATH=training python3 -m unittest \
  training/test_export_sft.py training/test_build_data.py \
  training/test_verify_artifact.py training/test_train_loss.py
```

The export hashes whole conversation sections into one deterministic split, so
different targets from the same Discord thread cannot leak into validation.
When a section contains bot turns, only `PERSON_0` is used as the assistant
target; human-only sections contribute next-speaker examples. IDs are not
exported. Serving and training intentionally use the same structured,
oldest-first context representation.

`training/run_on_runpod.sh` installs exact dependency versions, runs the full
preflight (including a real bf16 forward/backward/fused-Adam step), performs
full-parameter continuation training, compares held-out loss separately for
Discord, general chat, code, personality, encyclopedia, and web grounding, and
only then emits an f16 GGUF. Semantic, context, prompt-injection, and diversity
gates run as well. Completion loss is averaged within each example before the
example losses are averaged, so long code answers cannot outweigh the verified
70% Discord example share. A failed quality
gate exits before conversion. Checkpoints are retained on the attached network
volume and may be resumed with `--resume-from-checkpoint`.

Before promotion, compare conservative, balanced, and playful sampling on the
finished GPU model:

```sh
python3 training/evaluate_sampling.py \
  --model /workspace/outputs/super-sighurt-sig-v2-20260822/final-hf \
  --output /workspace/outputs/super-sighurt-sig-v2-20260822/sampling-evaluation.json
```

Promote only the resulting artifact directory:

```sh
scripts/promote_llama_model.sh \
  /path/to/output/artifact \
  ada45f66f1955a727aa854a8c2b79db5064500908a06c860f7079897eb728efe
```

Promotion verifies `SHA256SUMS`, pinned data/model provenance, loss gates, and
semantic probes against the externally supplied parent-model SHA-256; it then
loads the complete model on the deployment GPU and runs real generation. It
preserves one previous generation and renames the tokenizer before the watched
model file, so the live service never sees a new model with an old tokenizer.

Production transformer settings live in `~/.config/sighurt-llm.env`:

```ini
SIGHURT_MODEL=/home/gusahlg/artificial-stupidity/model.serving.f16.gguf
SIGHURT_TOKENIZER=/home/gusahlg/artificial-stupidity/tokenizer.serving.json
SIGHURT_CONTEXT_TOKENS=2048
SIGHURT_MAX_NEW_TOKENS=160
SIGHURT_REQUEST_TIMEOUT=90
SIGHURT_TEMPERATURE=0.75
SIGHURT_TOP_P=0.92
SIGHURT_TOP_K=50
SIGHURT_REPETITION_PENALTY=1.15
ML_DEVICE=discrete
```

Never deploy greedy decoding (`SIGHURT_TEMPERATURE=0`, `SIGHURT_TOP_K=1`).
Discord feeds the bot's own replies back as context, so a deterministic
sampler that emits one generic line will see that line in-context on the next
turn and reproduce it forever — a self-reinforcing repetition spiral (observed
live: 92% byte-identical replies across 26-turn conversations). The sampler
applies the repetition penalty over the prompt tail plus the generated tokens
(the Hugging Face semantics used by training-time evaluations), which is what
lets the bot escape a channel history already poisoned with repeats.
`scripts/apply_recommended_sampling.sh` applies the winner from
`training/evaluate_sampling.py` to the env file.

Long-conversation quality is measured end-to-end with
`training/conversation_acceptance.py`, which drives the live `/chat` endpoint
through 26-turn conversations on 17 scenarios (a local ollama model plays the
human) and reports duplicate-reply, persona-leak, and reliability metrics per
transcript.

On an existing installation, `scripts/configure_llama_env.sh` performs this
switch atomically while preserving the current API key and unrelated settings.
`scripts/run_serve_llama.sh` supplies the Vulkan loader and NVK/RADV ICDs from
the GC-rooted `~/.local/state/nix/profiles/sighurt-vulkan` profile, then selects
the discrete GPU. This keeps headless deployments reproducible even before the
same graphics packages are activated system-wide by a NixOS rebuild.

## Chat with the bot

```sh
./target/release/rust_fun
```

First run with no `model.bin` initializes a fresh network, runs a short
pretraining pass over `data/dialogs.txt`, saves `model.bin`, then drops
you into the prompt. Subsequent runs load the saved weights instantly.

> `data/dialogs.txt` and `vocab.txt` are **not tracked in git** — they are
> derived artifacts, mutated weekly by the automated refresh. On a fresh
> clone, bootstrap the corpus with
> `scripts/rebuild_full_corpus.sh [oasst.jsonl.gz]` (the scraped Discord
> TSVs live under `data/discord/`, also untracked) before training.

Commands inside the chat:

| Command | Effect |
|---|---|
| `:q` | quit (saves first) |
| `:save` | checkpoint `model.bin` now |
| `:train on` / `:train off` | toggle online learning during chat |

Online learning is **off by default**. When enabled (`:train on`), every
turn pulls a "teacher response" from the dialog corpus and runs one
SGD step against it at a deliberately small per-turn LR (`ONLINE_LR =
0.0001`, well below the offline trainer's typical 0.0003). Keep it off
unless you intend to deliberately nudge the model — a chat session is
a noisy signal and the updates go straight into the live `model.bin`.

## Auto-train

```sh
./target/release/train --epochs 50
```

The trainer shuffles every epoch, randomly drops the prior-turn prelude
to simulate cold-start prompts, prints train + validation cross-entropy
per epoch, saves after each epoch, and prints a sample generation every
few epochs so you can watch the model improve.

Flags:

| Flag | Default | What it does |
|---|---|---|
| `--epochs N` | 50 | how many supervised passes over the corpus |
| `--lr F` | 0.05 | starting learning rate |
| `--lr-decay F` | 0.985 | per-epoch multiplicative LR decay |
| `--save-every N` | 1 | checkpoint every N epochs |
| `--sample-every N` | 5 | print a sample generation every N epochs (0 = never) |
| `--prelude-drop F` | 0.3 | probability of dropping the in-section prelude during training |
| `--val-frac F` | 0.1 | fraction of examples held out for validation |
| `--max-train-examples N` | (none) | cap the training pool size (post-split); useful for short timing benchmarks |
| `--max-val-examples N` | (none) | cap the validation pool size (post-split) |

Note: the default `--lr 0.05` is appropriate for a freshly-initialized
network on a tiny vocab. For continued training on the full Discord
corpus it overshoots; empirically `0.0003-0.0005` is a safer band when
resuming a trained `model.bin`.

Loss starts near `ln(vocab_size)` (≈ 8 for vocab 3000) and should drop
into the low single digits with enough epochs. The chat REPL and the
trainer share `model.bin`, so you can leave the trainer running in one
terminal and chat in another — closing either one saves first.

### Picking a backend (CPU vs GPU)

The trainer defaults to **CPU**. Empirically, on the current model
shape (256 embed, 768 hidden, 4 layers, 3029 vocab) the CPU rayon
matmul is ~5–6× faster per step than Vulkan, because Vulkan
dispatch+sync overhead dwarfs the actual math on these 768×768 matvecs
(measured: ~5 ms/step CPU vs ~28 ms/step GPU). GPU only starts to win
once matmuls are batched large enough to amortize dispatch — i.e. when
mini-batch training is implemented.

Override:

```sh
SIGHURT_TRAIN_GPU=1 ./target/release/train ...
```

This opts in to Vulkan; falls back to CPU automatically if Vulkan
init fails. (The legacy `SIGHURT_TRAIN_CPU=1` env var is harmless but
moot, since CPU is now the default.)

### Per-phase timing

Run with `SIGHURT_TIME_STEPS=1` to print a one-line per-epoch breakdown
of where wall time goes (forward / backward / dense Adam / embedding
Adam) so you can target the right bottleneck:

```
  timing> steps=1052 fwd=21.8%/1064µs back=45.7%/2229µs adam_dense=31.7%/1545µs adam_embed=0.7%/36µs
```

## Serve over HTTP

```sh
SIGHURT_API_KEY=$(openssl rand -hex 32) \
SIGHURT_BIND=127.0.0.1:8088 \
./target/release/serve
```

Endpoints:

- `GET /healthz` → `200 ok` (unauthenticated; readiness probe)
- `POST /chat` → `{"reply":"..."}` with `X-API-Key` header

POST body schema (all values strings; only `input` is required, unknown
fields are ignored so older clients keep working):

```json
{"channel_id":"...","user":"...","user_id":"...","user_is_bot":"true|false",
 "input":"...",
 "reply_to_user":"...","reply_to_user_id":"...","reply_to_text":"...",
 "reply_to_is_bot":"true|false","reply_to_is_self":"true|false"}
```

The server maps each channel's speakers onto PERSON_N tags (the bot is
always PERSON_0; `reply_to_is_self` marks the replied-to message as the
bot's own) and feeds the model *tagged* turns — the same shape as the
training corpus. A `reply_to_*` context turn is injected directly
before the input so it survives the 32-token window truncation.

The server holds the model in memory and serializes generation behind a
`Mutex<Network>` (the per-layer caches are mutable per request); a small
accept pool keeps `/healthz` responsive while a request generates. It
also indexes the corpus into a RAG store at startup and prepends the
top-K most embedding-similar past turns to the per-channel chat memory
before generating each reply.

The server only reads the model at startup. In production it reads a
**pinned pair** — `model.serving.bin` + `data/dialogs.serving.txt` —
promoted together by `scripts/promote_model.sh`, so the weekly corpus
refresh can never drift the vocab out from under the live model (the
v4+ vocab hash would refuse to load). A `sighurt-llm.path` watcher
restarts the service when a promotion lands. For the full single-machine
deployment (bot + server + timers), see
[Discord AI setup](docs/discord-ai-setup.md).

The NVK/GTX 1650 Vulkan stack can lose the logical device under sustained
load. The server process survives that loss — `/healthz` still answers — but
every generation fails until the GPU context is recreated, which
`Restart=always` never notices. `systemd/sighurt-llm-watchdog.timer` runs
`scripts/watchdog_probe.sh` every two minutes: it exercises a real `/chat`
generation and restarts the service only on the consistent-failure signature
(a queue-full 503 or timeout counts as alive).

Env vars:

| Var | Default | Notes |
|---|---|---|
| `SIGHURT_BIND` | `127.0.0.1:8088` | listen address |
| `SIGHURT_API_KEY` | (required) | refuses to start without one, requires ≥ 16 chars |
| `SIGHURT_MODEL` | `model.bin` | model file to load; **missing model = startup error** (`SIGHURT_ALLOW_FRESH=1` overrides, for testing) |
| `SIGHURT_CORPUS` | `data/dialogs.txt` | corpus for vocab + RAG; must match the model's training snapshot |
| `SIGHURT_TEMPERATURE` | `1.0` | sampling temperature |
| `SIGHURT_TOP_P` | (unset) | nucleus sampling mass; unset keeps top-k 5 |

## Tweaking the model

Most knobs live as `pub const` at the top of `src/neural_network.rs`:

| Constant | Meaning | Notes |
|---|---|---|
| `EMBED_DIM` | width of each word embedding | bigger = more semantic capacity per word |
| `CONTEXT_WINDOW` | how many recent tokens feed the network | embeddings are concatenated, so input grows linearly |
| `HIDDEN_SIZE` | width of each hidden layer | bigger = more capacity, slower |
| `NUMBER_OF_HIDDEN_LAYERS` | depth (output layer added on top) | 2–4 works well; deeper needs more data |
| `MAX_TARGET_TOKENS` | cap on target sequence length per example | guards against paragraph-length Discord turns dominating training |
| `GRAD_CLIP` | symmetric per-element gradient clip | raise to allow bigger updates |
| `ADAM_BETA1` / `ADAM_BETA2` / `ADAM_EPS` | AdamW hyperparameters | standard defaults |
| `WEIGHT_DECAY` | AdamW (decoupled) weight decay | `1e-4` |
| `MAX_GENERATION_LEN` | hard cap on tokens per reply | model also learns to emit `</PERSON_0>` to stop earlier |
| `TOP_K_SAMPLE` | sample from the top-k softmax outputs | 1 = greedy/deterministic, larger = more random |

> Changing `EMBED_DIM`, `CONTEXT_WINDOW`, `HIDDEN_SIZE`, or
> `NUMBER_OF_HIDDEN_LAYERS` invalidates an existing `model.bin`. The
> loader detects the shape mismatch, throws away the stale weights, and
> the next run pretrains a fresh network. A larger vocab (new words in
> the corpus) is handled automatically — the loader extends the
> embedding/output layers with new random rows (and zero Adam moments
> for the new rows).

Online-chat hyperparameters live in `src/main.rs`:

| Constant | Meaning |
|---|---|
| `PRETRAIN_EPOCHS` | epochs of corpus pretraining on first run |
| `PRETRAIN_LR` | learning rate during that pretrain |
| `ONLINE_LR` | learning rate during interactive chat |
| `SAVE_EVERY_N_TURNS` | how often the chat auto-saves |

## On-disk model format (`model.bin`)

Binary, little-endian. Header: magic `0x4D4F_444C` ("MODL"), `u32`
version. Versions:

- **v2** — weights + biases only. Adam moments are recreated as zeros
  on load, so a resumed run pays a bias-correction "warmup tax".
- **v3** — adds the AdamW moment buffers (`w_m`, `w_v`, `b_m`, `b_v`
  per layer, plus the embedding's `m`, `v`) and the global `adam_step`
  counter. A resumed run picks up Adam exactly where it left off.
  Files are ~3× larger than v2 because of the moment arrays.
- **v4** — adds a hash of the vocab the model was trained against.
  Loading refuses on mismatch instead of silently scrambling the
  output-row-to-word mapping after a corpus/vocab change. Append-only
  vocab growth stays allowed (the hash covers the saved-size prefix).
- **v5** (current) — persists the training hyperparameters
  (`dropout_p`, `label_smoothing`) and optional per-layer LayerNorm
  parameters (gain/bias + Adam state). Old files load with those
  features off, exactly as before.

`save()` always writes the current version; `load()` accepts all of
the above.

## Adding training data

Edit `data/dialogs.txt`. The format is:

```
<SEC>
<PERSON_1> ...one turn from speaker 1... </PERSON_1>
<PERSON_2> ...the reply from speaker 2... </PERSON_2>

<PERSON_1> ...follow-up... </PERSON_1>
<PERSON_2> ...follow-up reply... </PERSON_2>

<SEC>
<PERSON_1> ...start of a new conversation... </PERSON_1>
...
```

`<SEC>` starts a new conversation. Within a section, each `<PERSON_N>
...</PERSON_N>` is one turn. By convention `<PERSON_0>` is the bot —
generation primes with `<PERSON_0>` and stops at `</PERSON_0>`. Other
PERSON ids are arbitrary section-local discriminators (e.g. `<PERSON_2>`
in section A and section B may be different real speakers).

A local, ignored bincode cache (`data/dialogs.bin`) is regenerated
whenever the text file's content hash changes, so corpus edits invalidate it
automatically. Any new tokens appended to the corpus get added to
`vocab.txt` the next time the data is loaded; that grows the embedding
table and the output layer (the loader extends them in place rather
than discarding the model).

### The Discord pipeline (the normal way data arrives)

The Discord bot (`~/discord-bot`, separate repo) logs every message in
every server it's in — humans *and* other bots — to per-channel TSVs
with reply-to ids, and heals offline gaps with a daily catch-up scrape.
From there:

- **Weekly (automated)**: `scripts/refresh_corpus.sh` (run by the
  `sighurt-refresh` systemd user timer, Sun 04:37) rsyncs the TSVs into
  `data/discord/` and runs `convert_discord` **incrementally** — per-
  channel cursors in `data/discord/.convert-state.json` mean only new
  messages become new sections — then `clean_corpus` + `rebuild_vocab`.
- **Full rebuild**: `scripts/rebuild_full_corpus.sh data/oasst1.trees.jsonl.gz`
  reconverts ALL Discord history (use after converter-rule changes),
  re-ingests OASST, injects seed pairs ×3, cleans, rebuilds vocab.
  Train fresh afterwards — the vocab order will have changed.

`convert_discord` is reply-aware: reply chains are stitched into their
own `<SEC>` sections in time order (a threaded exchange scattered
through an hour of channel noise becomes one clean dialog), and
`<@id>` mentions are rewritten to learnable `@displayname` tokens
(the live bot resolves `@name` in model output back into real pings).

### Recommended workflow after editing the corpus by hand

1. Ingest / edit `data/dialogs.txt` (e.g. `convert_discord`, `inject_seed`,
   or a hand edit).
2. Run `cargo run --release --bin clean_corpus` to drop URL-only /
   emoji-only / role-ping turns, drop monologue and single-speaker
   sections, dedup over-repeated turns, and deterministically shuffle
   sections so the val-tail split is not topical. The cleaner is
   idempotent and writes via a `.tmp` sibling + rename (crash-safe).
   Before the first run, back up the original corpus
   (`cp data/dialogs.txt data/dialogs.txt.pre-clean`).
3. Run `cargo run --release --bin rebuild_vocab` so `vocab.txt`
   reflects the cleaned corpus.
4. **Decide what to do with `model.bin`**: if the vocab order changed
   (likely whenever long-tail frequencies shift), the v4+ vocab hash
   makes the loader refuse the old model — move it aside and train
   fresh. The live server is immune: it reads the pinned
   `data/dialogs.serving.txt` snapshot, not the corpus you just edited
   (see `scripts/promote_model.sh`).

## Resetting the model

Just delete (or rename away) `model.bin`:

```sh
mv model.bin model.bin.bak
./target/release/train --epochs 50
```

The next run starts from random weights and retrains. Renaming rather
than deleting is the safer habit — if the model that just got
discarded was the only copy of a good checkpoint, the `.bak` lets you
roll back.
