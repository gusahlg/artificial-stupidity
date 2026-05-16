# SuperSighurt LLM

A tiny from-scratch neural language model written in Rust. It has a Vulkan
GPU backend (via the sibling `ml_project` crate) and a CPU fallback, learns
from a small dialog corpus, persists its weights between runs, and ships
with an auto-trainer binary.

## Layout

- `src/main.rs` — interactive chat (`rust_fun` binary)
- `src/bin/train.rs` — standalone auto-trainer (`train` binary)
- `src/tokenizer.rs` — lowercase + punctuation-splitting tokenizer
- `src/embeddings/mod.rs` — trainable word embedding table
- `src/neural_network.rs` — model, training loop, generation
- `src/teacher.rs` — softmax + cross-entropy backprop (incl. input gradient for embeddings)
- `src/persist.rs` — save/load `model.bin` (v2: includes embeddings)
- `src/gpu.rs` — Vulkan/CPU backend dispatch
- `src/machine_learning.rs` — section-similarity + embedding-cosine teacher lookup
- `src/dialogs.rs`, `src/memory.rs` — corpus + vocab loaders
- `data/dialogs.txt` — training corpus
- `vocab.txt` — derived vocabulary (regenerated from corpus on every run)
- `model.bin` — saved embeddings + weights + biases (created on first run)

## Build

```sh
cargo build --release
```

Two binaries land in `target/release/`:

- `rust_fun` — the chat
- `train`    — the auto-trainer

The GPU backend uses Vulkan. If `libvulkan.so.1` isn't on the system the
project transparently falls back to a CPU backend, so neither binary
requires a GPU to run.

## Chat with the bot

```sh
./target/release/rust_fun
```

First run: with no `model.bin` present it initializes a fresh network,
runs a short pretraining pass over `data/dialogs.txt`, saves
`model.bin`, then drops you into the prompt. Subsequent runs load the
saved weights instantly.

Commands inside the chat:

| Command | Effect |
|---|---|
| `:q` | quit (saves first) |
| `:save` | checkpoint `model.bin` now |
| `:train on` / `:train off` | toggle online learning during chat |

Online learning is on by default: every turn pulls a "teacher response"
from the dialog corpus and runs one SGD step against it, so the model
keeps improving while you chat.

## Auto-train (recommended way to get useful behavior)

```sh
./target/release/train --epochs 50
```

The auto-trainer shuffles every epoch, randomly drops the prior-turn
prelude to simulate cold-start prompts, prints cross-entropy per epoch,
saves after each epoch, and prints a sample generation every few epochs
so you can watch the model improve.

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `--epochs N` | 50 | how many supervised passes over the corpus |
| `--lr F` | 0.05 | starting learning rate |
| `--lr-decay F` | 0.985 | per-epoch multiplicative LR decay |
| `--save-every N` | 1 | checkpoint every N epochs |
| `--sample-every N` | 5 | print a sample generation every N epochs (0 = never) |
| `--prelude-drop F` | 0.3 | probability of dropping the in-section prelude during training |

Loss starts near `ln(vocab_size) ≈ 8.5` and should drop to ~1–2 within
a few minutes of CPU training on the included corpus. The chat and the
trainer share `model.bin`, so you can leave the trainer running in one
terminal and chat in another — closing either one saves first.

## Tweaking the model

Most knobs live as `pub const` at the top of `src/neural_network.rs`:

| Constant | Meaning | Notes |
|---|---|---|
| `EMBED_DIM` | width of each word embedding | bigger = more semantic capacity per word |
| `CONTEXT_WINDOW` | how many recent tokens feed the network | embeddings are concatenated, so input grows linearly |
| `HIDDEN_SIZE` | width of each hidden layer | bigger = more capacity, slower |
| `NUMBER_OF_HIDDEN_LAYERS` | depth (output layer added on top) | 2 works well; deeper needs more data |
| `GRAD_CLIP` | symmetric per-element gradient clip | raise to allow bigger updates |
| `MOMENTUM` | SGD momentum (0 = pure SGD) | 0.0 is the default; >0 needs careful tuning here |
| `WEIGHT_DECAY` | L2 weight decay coefficient | 1e-4 is a sensible starting point |
| `MAX_GENERATION_LEN` | hard cap on tokens per reply | model also learns to emit `</BOT>` to stop earlier |
| `TOP_K_SAMPLE` | sample from the top-k softmax outputs | 1 = greedy/deterministic, larger = more random |

> Changing `EMBED_DIM`, `CONTEXT_WINDOW`, `HIDDEN_SIZE`, or
> `NUMBER_OF_HIDDEN_LAYERS` invalidates an existing `model.bin`. The
> loader detects the shape mismatch, throws away the stale weights, and
> the next run pretrains a fresh network. A larger vocab (new words in
> the corpus) is handled automatically — the loader extends the
> embedding/output layers with new random rows.

Online-chat hyperparameters live in `src/main.rs`:

| Constant | Meaning |
|---|---|
| `PRETRAIN_EPOCHS` | epochs of corpus pretraining on first run |
| `PRETRAIN_LR` | learning rate during that pretrain |
| `ONLINE_LR` | learning rate during interactive chat |
| `SAVE_EVERY_N_TURNS` | how often the chat auto-saves |

## Adding training data

Edit `data/dialogs.txt`. The format is:

```
<SEC>
<USER> ...one turn from the user... </USER>
<BOT> ...the desired bot reply... </BOT>

<USER> ...follow-up... </USER>
<BOT> ...follow-up reply... </BOT>

<SEC>
<USER> ...start of a new conversation... </USER>
...
```

`<SEC>` starts a new conversation. Within a section, `<USER>` /
`<BOT>` pairs are turns in order. Any new tokens get appended to
`vocab.txt` automatically the next time the corpus is loaded. That
also changes the network's output size, so the loader will discard the
old `model.bin` and re-pretrain from scratch on the new vocab.

## Resetting the model

Just delete it:

```sh
rm model.bin
./target/release/train --epochs 50
```

The next run starts from random weights and retrains.
