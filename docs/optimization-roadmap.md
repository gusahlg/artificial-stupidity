# Training-speed optimization roadmap

A survey of viable optimizations for the SuperSighurt training pipeline.
None are implemented yet (this is the "things worth doing later" list); the
table at the bottom ranks them by expected payoff vs effort.

Baseline numbers used throughout (current observed):
- Corpus: ~14k turns → ~12k train examples / 590 val.
- Model: EMBED_DIM 256, HIDDEN_SIZE 768, 4 layers, CONTEXT_WINDOW 32, ~10M params.
- Vocab: 3029 tokens (output layer ≈ 2.3M weights).
- Hardware: NVIDIA RTX 3070 (GPU available), recent multi-core CPU with rayon.
- Observed: ~36-55 min/epoch on the prior 9k-turn corpus, scales linearly with corpus size.

The dominant single cost per training step is the **Adam update on the
output layer** (writes 3 × 2.3M floats) followed by the **backward pass
through the hidden layers** (`compute_deltas` in `src/teacher.rs`). Forward
is already on GPU via `Gpu::matvec` and is not the bottleneck.

---

## 1. Cache-friendly backward (loop reorder in `compute_deltas`)

**What.** In the hidden-layer delta computation we currently iterate:

```rust
for j in 0..curr.len() {
    let mut sum = 0.0f32;
    for (k, &d) in next_d.iter().enumerate() {
        sum += next_w[k * next_cols + j] * d;   // stride-cols access
    }
    curr[j] = sum * slope_j;
}
```

`next_w[k * next_cols + j]` is a column-access into a row-major matrix:
the stride between successive reads is `next_cols * 4` bytes (3 KB for the
output layer's preceding hidden layer). Each cache line load gets used for
exactly one j inside the inner loop, then evicted before the next j
iteration in the same thread.

**Fix.** Swap loop nesting (k outer, j inner) and accumulate per j:

```rust
for slot in curr.iter_mut() { *slot = 0.0; }
for (k, &d) in next_d.iter().enumerate() {
    let row = &next_w[k * next_cols..(k + 1) * next_cols];
    for j in 0..curr.len() {
        curr[j] += row[j] * d;            // contiguous reads ✓
    }
}
// apply slope per j afterwards
```

**Parallelism.** Lose the natural per-j independence; recover it by
chunking j. Each thread takes a contiguous slice of `curr`, runs the full
k loop into its slice. Each thread reads disjoint columns of `next_w`,
so no shared-write contention.

```rust
curr.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, slot_chunk)| {
    let j_start = ci * CHUNK;
    for slot in slot_chunk.iter_mut() { *slot = 0.0; }
    for (k, &d) in next_d.iter().enumerate() {
        let row_off = k * next_cols + j_start;
        for (off, slot) in slot_chunk.iter_mut().enumerate() {
            *slot += next_w[row_off + off] * d;
        }
    }
    // slope ...
});
```

Pick `CHUNK` so each thread's working set fits L2 (~256 KB typical). For
the output→last-hidden case with `next_cols=768` and k up to vocab_size
3029, a chunk of 16 j-slots reads `3029 × 16 × 4 = 194 KB` of `next_w` per
thread — fits.

**Expected payoff.** Backward is currently ~30-40% of per-step CPU time.
The cache-unfriendly access pattern is plausibly costing 3-5× on that
portion. Net training speedup: **1.5-2×** (35 min/epoch → ~20 min).

**Risk.** Low-medium. The math is identical (we're just reordering an
associative sum), but rewriting the loop nesting is the kind of change
that gets bugs at boundary conditions. Cover with a numerical test that
runs old + new on the same inputs and asserts max-abs diff < 1e-4.

**Effort.** ~30 lines of Rust + a small numerical test. Half a day.

**Same fix applies to the input-grad loop** at the bottom of
`compute_deltas` — currently `l0_w[j * cols + k]` for varying j is the
same anti-pattern.

---

## 2. SIMD the Adam inner k-loop

**What.** `Network::train_step` updates each output row sequentially in
its inner k-loop:

```rust
for k in 0..cols {
    let x = input_ref[k];
    let gk = g * x;
    let m_new = ADAM_BETA1 * m_row[k] + (1.0 - ADAM_BETA1) * gk;
    let v_new = ADAM_BETA2 * v_row[k] + (1.0 - ADAM_BETA2) * gk * gk;
    m_row[k] = m_new;
    v_row[k] = v_new;
    let m_hat = m_new / bc1;
    let v_hat = v_new / bc2;
    let mut w = w_row[k];
    w -= lr * (m_hat / (v_hat.sqrt() + ADAM_EPS) + WEIGHT_DECAY * w);
    w_row[k] = w;
}
```

Each iteration is independent (no carried dependency on `k-1`). All ops
are f32, mostly multiply-add with one `sqrt`. Vectorizing 8 lanes at a
time (f32x8 / AVX2) is the canonical SIMD opportunity.

**Fix.** Add `wide = "0.7"` and rewrite as 8-wide chunks with a scalar
tail:

```rust
use wide::f32x8;
let beta1 = f32x8::splat(ADAM_BETA1);
let one_minus_b1 = f32x8::splat(1.0 - ADAM_BETA1);
// ...
let chunks = cols / 8;
for ci in 0..chunks {
    let off = ci * 8;
    let x  = f32x8::from(&input_ref[off..off+8]);
    let gk = f32x8::splat(g) * x;
    // ... full Adam in vector ops ...
}
// scalar tail for cols % 8
```

**Expected payoff.** Adam is partly memory-bound (3 reads + 3 writes per
weight), so SIMD won't give a clean 8× speedup on it. Empirical Rust SIMD
on this kind of mixed memory+compute kernel is typically **1.5-2×**. Net
training speedup: **~1.3-1.5×** (since Adam is roughly half the per-step
cost).

**Risk.** Medium. Adam math is fiddly; off-by-one on the scalar tail is
the usual bug. The `wide` crate has clean semantics but the rewrite is
~80 lines. Cover with a numerical equivalence test.

**Effort.** One day to write + test, plus careful review.

**Dependency.** `wide` (no nightly required, builds on stable, no system
deps).

---

## 3. Bincode cache for parsed corpus

**What.** `Data::load_from` reads `data/dialogs.txt`, splits by whitespace,
walks a small state machine, tokenizes each turn. For 14k turns this
takes ~100-200 ms today; grows linearly. `sighurt-llm` does this every
startup; the overnight restart loop hits it 12+ times.

**Fix.** Sidecar `data/dialogs.bin` (bincode-encoded `Vec<Vec<Turn>>`):
- After `Data::load_from` succeeds, write it.
- On next `Data::load_from`, check the bin's mtime vs the txt's. If bin
  is newer AND its checksum matches the txt's content hash, load the bin
  and skip parsing.
- Any ingestor that appends to dialogs.txt automatically invalidates the
  cache (bin becomes older).

**Expected payoff.** `sighurt-llm` restart drops by 100-500 ms. Doesn't
help training speed, but matters for the operational loop during overnight
training (each restart wastes ~10s of bot downtime today; this cuts it).

**Risk.** Low. Cache invalidation logic is the only sharp edge — if we
miss an invalidation we serve stale data. Mitigate with content-hash
verification (cheap: blake3 over the txt).

**Effort.** ~50 lines + a content-hash check. Few hours.

**Dependencies.** `bincode = "1.3"`, optionally `blake3` for the hash
(or use `std::hash::Hasher` with `DefaultHasher`).

---

## 4. GPU compute shader for Adam update

**What.** Move the per-row Adam math from CPU (rayon) to a Vulkan compute
shader. Each thread group handles one row; within the group, threads
cover the inner k iterations.

**Sketch.**
- Existing `vulkano` infrastructure (`src/gpu.rs`) already has buffer
  upload/download and a matmul kernel. Add a second shader: `adam.comp`,
  bind weights / m / v / grad / input / hyperparams as storage buffers.
- Per training step: upload `g_row` (the per-row gradient = `delta[j]`),
  dispatch one workgroup per row.
- Weights and Adam state live persistently on the GPU; sync back to host
  only when CPU needs them (rarely — for sampling and serialization).

**Expected payoff.** Adam is the single biggest training step cost. On a
3070, even a naive compute shader should beat 4-8 CPU cores by 5-10× on
this kind of dense f32 work. **Net training speedup: 3-5×** combined with
already-on-GPU forward.

**Risk.** High for first implementation. Vulkan kernels are fiddly:
binding layout, descriptor sets, synchronization, host/device staging.
Numerical correctness has to match CPU Adam bit-for-bit to allow swapping
back. Easy to make a kernel that runs but produces subtly wrong updates.

**Effort.** Big. A focused week. Realistically a "second project" not a
"sneak it in" change.

**Prerequisite.** A test harness that compares CPU-Adam vs GPU-Adam on
the same starting weights + same training step. Without this, debugging
is hopeless.

---

## 5. GPU compute shader for backward (`compute_deltas`)

**What.** Same idea as #4 but for the hidden-layer delta computation.

**Sketch.** One workgroup per layer (sequential between layers because
of the dependency chain). Within a workgroup, threads cover the j
dimension. Reads `next_w` (persistent on GPU) and `next_d` (small,
upload once), writes `curr` (persistent on GPU).

**Expected payoff.** Compounded with #4, backward goes from CPU rayon to
GPU. Combined with #4 you'd see **5-10× training speedup**. But on its
own without #4, the gain is limited to backward's share (~30% of step) →
~1.5×.

**Risk.** High, similar to #4.

**Effort.** Another week-scale project; share infrastructure with #4.

---

## 6. Mini-batch training

**What.** Currently `train_one_epoch` processes one example at a time
through forward → backward → update. Modern training batches N examples,
runs forward on all N in parallel (one big matmul), averages the
gradients, applies one update.

**Sketch.**
- `forward_and_cache` becomes `forward_batch(x_batch: &[Vec<f32>])` →
  returns `Vec<Vec<f32>>`. Vulkan executes a single batched matmul per
  layer (`y = W @ X` where X is a matrix of stacked inputs).
- Similar for backward — gradients average across batch before Adam.
- Adam runs ONCE per N examples instead of N times.

**Expected payoff.** Two effects:
- Better GPU utilization: bigger matmuls amortize dispatch cost.
- Fewer Adam updates: N=16 cuts Adam cost ~16×.

**Net training speedup**: optimistic 4-8× on its own; compounds heavily
with #4/#5. The trade is more wall-time per "step" but each step does
more work.

**Risk.** Medium-high. Touches forward, backward, and Adam. Changes the
optimizer dynamics — effective LR vs batch size relationship needs
re-tuning (linear scaling rule is a reasonable default).

**Effort.** Multi-day. Best paired with #4/#5 since it's the same area
of code.

---

## 7. Fused activation + softmax + grad in output layer

**What.** Currently:
- Layer forward computes `z = Wx + b`, applies activation, returns `a`.
- Then softmax is applied in-place.
- Then backward computes `delta = softmax - one_hot`.

These three steps could be fused into a single kernel that consumes `z`
and emits both `softmax(z)` and `delta`. Saves one full output-layer
sweep.

**Expected payoff.** Small. Maybe **1.05-1.1×** on training. The output
layer is `vocab_size = 3029` floats per step, not huge.

**Risk.** Low.

**Effort.** ~20 lines of Rust on the CPU path; if doing GPU shaders (#4)
the fusion happens naturally there.

**Verdict.** Not worth doing on its own. Free if #4 happens.

---

## 8. Pre-allocate scratch buffers in `compute_deltas`

**What.** Each call to `compute_deltas` allocates:
- `layer_deltas: Vec<Vec<f32>>` — N small Vecs.
- `input_grad: Vec<f32>` — one larger Vec.

These get dropped at function return. Allocator churn = ~10 µs per step
on glibc malloc. With 70k steps that's ~700 ms / epoch. Not huge but
free to fix.

**Fix.** Add a `BackpropScratch` field on `Network`. `compute_deltas`
takes `&mut Network` (it already does) and uses the pre-allocated
buffers. Resizes only if dimensions change.

**Expected payoff.** **~1.02×.** Worth bundling with bigger refactors.

**Risk.** Trivial.

**Effort.** ~30 lines. An hour.

---

## 9. Lookup table for tanh

**What.** Hidden-layer activation is `tanh`. The hardware `f32::tanh` is
implemented as a series expansion or a libm call; either way it's not
free.

**Fix.** 256-entry LUT covering the saturation range (≈ ±3 σ), linear
interpolation. Loses ~5e-4 of precision at the worst input.

**Expected payoff.** Small. Forward layer activation is `hidden_size`
floats per layer (768) × layers × steps. Maybe **1.05×** overall.

**Risk.** Low. Numerical accuracy slightly degraded; in practice
indistinguishable from `f32::tanh` for training.

**Effort.** ~20 lines.

**Verdict.** Cosmetic. Skip unless we're in the last 10% of perf.

---

## 10. Smaller / sparser Adam state

**What.** Adam keeps 3 floats per weight (`w`, `m`, `v`). For 10M params
that's ~120 MB. Memory bandwidth on the inner Adam loop is exactly this
× steps per second. If we used FP16 for `m`/`v` (with loss-scaling tricks),
the bandwidth requirement halves.

**Expected payoff.** Adam is memory-bound, so **~1.3-1.5×** on the Adam
portion. Net **1.2×**.

**Risk.** Medium. FP16 Adam state is well-known but has numerical
gotchas (very small `v` underflows). Mature implementations use mixed
precision carefully.

**Effort.** Medium. Touches Layer struct, forward, train_step.

**Verdict.** Defer; compounds nicely with GPU Adam (#4) but is fiddly
solo.

---

## Cross-cutting: profiling first

Before any of this work I'd want one profiling pass with `perf` or
`cargo flamegraph` to confirm where time is actually going. The analyses
above are arithmetic guesses; profiling typically reveals one or two
surprise hotspots (allocator, sync primitives, branch mispredicts) that
trump the theoretical analysis.

Specifically:
```
cargo install flamegraph        # (one-time)
cargo flamegraph --release --bin train -- --epochs 1 --val-frac 0.0
```

Look at the flamegraph for `train_one_epoch` and `compute_deltas`.
Confirm or refute the cache-pattern hypothesis. Maybe 30 minutes to find
out which optimizations would actually matter.

---

## Ranked payoff / effort table

| # | Optimization | Net speedup | Effort | Risk |
|---|--------------|-------------|--------|------|
| 1 | Cache-friendly backward loop reorder | 1.5-2× | 0.5d | Low-Med |
| 6 | Mini-batch training | 4-8× | 3-5d | Med-High |
| 4 | GPU compute shader for Adam | 3-5× | 1w | High |
| 5 | GPU compute shader for backward | 1.5× (with #4: compounds to 5-10× total) | 1w | High |
| 2 | SIMD Adam inner loop | 1.3-1.5× | 1d | Med |
| 10 | FP16 Adam state | 1.2× | 2d | Med |
| 3 | Bincode cache for corpus | (operational) | 0.5d | Low |
| 8 | Pre-allocate scratch buffers | 1.02× | 1h | Trivial |
| 7 | Fused softmax+grad | 1.05-1.1× | 0.5d | Low |
| 9 | tanh LUT | 1.05× | 0.5d | Low |

**Suggested order** if working through this:
1. Profile first.
2. #1 (cache-friendly backward) — biggest win for the effort, gives ~2× immediately.
3. #3 (bincode cache) — fast quality-of-life win for the operational loop.
4. #8 (pre-allocate) — cheap, do while doing #1.
5. #2 (SIMD Adam) — solid 1.3-1.5× from a single dep.
6. #6 (mini-batch) — when ready for a serious refactor; sets up nicely for #4.
7. #4 + #5 + #10 — the GPU-everything endgame. Probably a 5-10× total speedup over the current baseline.

Conservative target after #1-#3: **3-4× faster training** (15 min/epoch → 4-5 min on the 14k-turn corpus). At that rate a 60-epoch overnight run becomes routine.

Ambitious target after #4-#6: **15-25× faster training**, putting it
under 1 min/epoch and making interactive retraining viable.
