# artificial-stupidity

A tiny from-scratch neural language model written in Rust. Embedding table
+ a few tanh-MLP layers + cross-entropy softmax output, trained with
AdamW. Has an interactive REPL, a standalone trainer, and an HTTP server
that exposes inference over the network (driven by a separate Discord
bot, not in this repo). The math runs on the CPU by default; a Vulkan
GEMM backend (via the sibling `ml_project` crate) is available as an
opt-in for experiments.

## Layout

- `src/main.rs` — interactive chat REPL (`rust_fun` binary)
- `src/bin/train.rs` — standalone auto-trainer (`train` binary)
- `src/bin/serve.rs` — HTTP server for inference (`serve` binary)
- `src/bin/convert_discord.rs` — ingest a Discord export into the dialog format
- `src/bin/ingest_dailydialog.rs` / `ingest_tinystories.rs` — corpus loaders
- `src/bin/inject_seed.rs` — splice a fixed set of "seed" Q/A pairs into the corpus
- `src/bin/rebuild_vocab.rs` — regenerate `vocab.txt` without retraining
- `src/tokenizer.rs` — lowercase + punctuation-splitting tokenizer
- `src/embeddings/mod.rs` — trainable word embedding table (with Adam state)
- `src/neural_network.rs` — model, forward, generation, training loop
- `src/teacher.rs` — softmax + cross-entropy backprop (incl. input gradient for embeddings)
- `src/persist.rs` — save/load `model.bin` (v3: weights, biases, Adam moments, `adam_step`)
- `src/gpu.rs` — Vulkan/CPU backend dispatch
- `src/machine_learning.rs` — section-similarity + embedding-cosine teacher lookup
- `src/rag.rs` — embedding-cosine retrieval store, used by `serve`
- `src/dialogs.rs`, `src/memory.rs` — corpus + vocab loaders (with a bincode cache)
- `data/dialogs.txt` — training corpus
- `data/dialogs.bin` — bincode cache of the parsed corpus (auto-invalidated on edit)
- `vocab.txt` — derived vocabulary (regenerated from corpus on every run)
- `model.bin` — saved embeddings + weights + biases + Adam state (created on first run)

## Build

```sh
cargo build --release
```

Produces in `target/release/`:

- `rust_fun` — interactive chat REPL
- `train`    — auto-trainer
- `serve`    — HTTP inference server
- assorted corpus ingestors / utility binaries

## Chat with the bot

```sh
./target/release/rust_fun
```

First run with no `model.bin` initializes a fresh network, runs a short
pretraining pass over `data/dialogs.txt`, saves `model.bin`, then drops
you into the prompt. Subsequent runs load the saved weights instantly.

Commands inside the chat:

| Command | Effect |
|---|---|
| `:q` | quit (saves first) |
| `:save` | checkpoint `model.bin` now |
| `:train on` / `:train off` | toggle online learning during chat |

Online learning is on by default: every turn pulls a "teacher response"
from the dialog corpus and runs one SGD step against it, so the model
keeps drifting while you chat.

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
SIGHURT_BIND=0.0.0.0:8088 \
./target/release/serve
```

Endpoints:

- `GET /healthz` → `200 ok` (unauthenticated; readiness probe)
- `POST /chat` → `{"reply":"..."}` with `X-API-Key` header

POST body schema: `{"channel_id":"...","user":"...","input":"..."}`.

The server holds the model in memory and serializes requests behind a
`Mutex<Network>` (the per-layer caches are mutable per request). It
also indexes the corpus into a RAG store at startup and prepends the
top-K most embedding-similar past turns to the per-channel chat memory
before generating each reply.

The server only reads `model.bin` at startup, so a running trainer can
write `model.bin` without disturbing it — restart the server when you
want it to pick up new weights.

Env vars:

| Var | Default | Notes |
|---|---|---|
| `SIGHURT_BIND` | `127.0.0.1:8088` | listen address |
| `SIGHURT_API_KEY` | (required) | refuses to start without one, requires ≥ 16 chars |
| `SIGHURT_MODEL` | `model.bin` | model file to load |

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
version. Two versions exist:

- **v2** — weights + biases only. Adam moments are recreated as zeros
  on load, so a resumed run pays a one-step bias-correction "warmup
  tax" on every restart.
- **v3** (current) — adds the AdamW moment buffers (`w_m`, `w_v`,
  `b_m`, `b_v` per layer, plus the embedding's `m`, `v`) and the global
  `adam_step` counter. A resumed run picks up Adam exactly where it
  left off, so restarting mid-training is no longer destructive.
  Files are ~3× larger than v2 because of the moment arrays.

`save()` always writes v3. `load()` accepts either; v2 files come back
with zeroed moments (preserving the old behavior) and become v3 the next
time the trainer saves.

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

A bincode cache (`data/dialogs.bin`) is regenerated whenever the text
file's content hash changes, so corpus edits invalidate the cache
automatically. Any new tokens appended to the corpus get added to
`vocab.txt` the next time the data is loaded; that grows the embedding
table and the output layer (the loader extends them in place rather
than discarding the model).

## Resetting the model

Just delete it:

```sh
rm model.bin
./target/release/train --epochs 50
```

The next run starts from random weights and retrains.
