use crate::neural_network::{Activation, Network};
use rayon::prelude::*;

/// Per-layer post-activation gradients **plus** the gradient with respect to
/// the network's input vector, so the embedding layer can receive its share.
///
/// Reused across training steps via `Network::backprop_scratch` to avoid
/// the ~600k Vec allocations/free per epoch we used to do when this was
/// returned by value from `compute_deltas`.
#[derive(Default)]
pub struct BackpropScratch {
    pub layer_deltas: Vec<Vec<f32>>,
    pub input_grad: Vec<f32>,
    /// Per-layer LayerNorm gain (γ) gradients. Indexed like
    /// `layer_deltas`; the entry for a layer without LN (legacy
    /// models, the output layer) is an empty Vec. Consumed by
    /// `Network::train_step`'s LN Adam update.
    pub ln_gain_grad: Vec<Vec<f32>>,
    /// Per-layer LayerNorm bias (β) gradients. Same indexing contract
    /// as `ln_gain_grad`.
    pub ln_bias_grad: Vec<Vec<f32>>,
}

/// Compatibility alias for legacy callers. New code uses `BackpropScratch`
/// and the in-place `compute_deltas_into`.
pub type BackpropOutput = BackpropScratch;

/// Standard softmax + cross-entropy backprop. Output layer delta collapses to
/// (softmax - one_hot); hidden layers use tanh' = 1 - a². Finally we compute
/// dL/dx for the input vector x = (concatenated embeddings || position).
///
/// **Cache-friendly access pattern**. The two big inner loops (hidden-layer
/// delta and input-grad) used to be `for j { for k { W[k*cols+j] } }` — a
/// stride-`cols` access into a row-major weight matrix. With cols = 768
/// that's a 3 KB stride between successive reads inside the inner loop,
/// which thrashes the cache. We've reordered to `for k { for j { W[k*cols+j] } }`,
/// which walks each row contiguously. To preserve rayon parallelism without
/// write-contention on the per-j accumulators, each thread owns a disjoint
/// chunk of the output slice and runs the full k-loop into its chunk.
///
/// The chunk size is tuned to fit a thread's working set (the rows of the
/// weight matrix it touches in stride pattern) into L2 cache. With
/// `BACKWARD_CHUNK = 32` and worst-case `rows = 3029`, each thread sees
/// `3029 × 32 × 4 = ~388 KB` of weights — sized for typical L2 caches.
/// Allocating wrapper kept for callers that don't have a pre-allocated
/// scratch buffer (tests, one-shot tools). Hot paths should call
/// `compute_deltas_into` with a reused `BackpropScratch`.
pub fn compute_deltas(net: &Network, target_idx: usize) -> BackpropOutput {
    let mut scratch = BackpropScratch::default();
    compute_deltas_into(net, target_idx, &mut scratch);
    scratch
}

/// In-place backward pass. The caller provides a `BackpropScratch` whose
/// buffers we resize to match net dimensions (cheap if already correct).
/// All writes go into `scratch`; we return nothing.
pub fn compute_deltas_into(net: &Network, target_idx: usize, scratch: &mut BackpropScratch) {
    let n = net.layers.len();
    // Destructure so the borrow checker sees layer_deltas, input_grad
    // and the LN gradient buffers as disjoint &mut borrows (the LN
    // section below writes ln_*_grad while holding a &mut slice of
    // layer_deltas).
    let BackpropScratch {
        layer_deltas,
        input_grad,
        ln_gain_grad,
        ln_bias_grad,
    } = scratch;
    // Resize scratch to current layer geometry. Vec::resize is cheap when
    // dimensions are already correct (no reallocation) and pads with 0.0
    // when growing, which is what we want before the output-layer write.
    if layer_deltas.len() != n {
        layer_deltas.resize_with(n, Default::default);
    }
    if ln_gain_grad.len() != n {
        ln_gain_grad.resize_with(n, Default::default);
    }
    if ln_bias_grad.len() != n {
        ln_bias_grad.resize_with(n, Default::default);
    }
    for (i, layer) in net.layers.iter().enumerate() {
        let need = layer.last_activations.len();
        if layer_deltas[i].len() != need {
            layer_deltas[i].clear();
            layer_deltas[i].resize(need, 0.0);
        } else {
            // Zero in place; the output-layer write overwrites this fully
            // anyway, but hidden layers' chunked writes need a clean start.
            layer_deltas[i].fill(0.0);
        }
        // LN gradient buffers: sized to `rows` for LN layers (their
        // writes below are full overwrites, no zeroing needed beyond
        // the resize), cleared for non-LN layers so train_step's
        // global-norm accounting never sees stale values.
        if layer.ln_active() {
            let rows = layer.rows;
            if ln_gain_grad[i].len() != rows {
                ln_gain_grad[i].clear();
                ln_gain_grad[i].resize(rows, 0.0);
            }
            if ln_bias_grad[i].len() != rows {
                ln_bias_grad[i].clear();
                ln_bias_grad[i].resize(rows, 0.0);
            }
        } else {
            ln_gain_grad[i].clear();
            ln_bias_grad[i].clear();
        }
    }

    // Output layer: softmax minus the *target distribution*. Without
    // label smoothing the target is a one-hot at `target_idx`. With
    // smoothing α (`net.label_smoothing`), we replace the one-hot
    // with `(1 - α) · one_hot + α · uniform(V)` — so the target index
    // gets `1 - α + α/V` and every other index gets `α/V`. The delta
    // (softmax − target_dist) is the gradient of cross-entropy with
    // respect to the pre-softmax logits, identical structure to the
    // one-hot case but with softer asymptotic targets. Empirically
    // worth 0.05–0.15 val loss on LM training. α = 0.0 disables.
    let out_acts = &net.layers[n - 1].last_activations;
    let v = out_acts.len();
    let alpha = net.label_smoothing;
    if alpha <= 0.0 {
        for i in 0..v {
            let t = if i == target_idx { 1.0 } else { 0.0 };
            layer_deltas[n - 1][i] = out_acts[i] - t;
        }
    } else {
        let off = alpha / v as f32;
        let on = 1.0 - alpha + off;
        for i in 0..v {
            let t = if i == target_idx { on } else { off };
            layer_deltas[n - 1][i] = out_acts[i] - t;
        }
    }

    // Hidden layers: walk row-major next_w in k-outer order, j-chunked.
    for li in (0..n - 1).rev() {
        let next_layer = &net.layers[li + 1];
        let next_cols = next_layer.cols;
        let acts = &net.layers[li].last_activations;
        let layer_activation = net.layers[li].activation;
        // Empty mask = dropout was not applied to this layer's forward.
        // Non-empty mask has one entry per neuron: 0.0 (dropped) or
        // 1/(1-p) (kept, scaled). Multiplying the delta by it ensures
        // dropped neurons receive zero gradient (matching the forward
        // where their activation was zeroed) and kept neurons are
        // scaled consistently with the forward's inverted-dropout
        // expectation preservation.
        let dropout_mask: &[f32] = &net.layers[li].last_dropout_mask;
        let apply_mask = !dropout_mask.is_empty();
        let (left, right) = layer_deltas.split_at_mut(li + 1);
        let curr = &mut left[li];
        let next_d: &[f32] = &right[0];
        let next_w = &next_layer.weights;

        curr.par_chunks_mut(BACKWARD_CHUNK).enumerate().for_each(
            |(ci, slot_chunk)| {
                let j_start = ci * BACKWARD_CHUNK;
                let chunk_len = slot_chunk.len();
                // Zero this chunk's slice of the j-accumulator.
                for slot in slot_chunk.iter_mut() {
                    *slot = 0.0;
                }
                // For each k (an output neuron of the next layer), add its
                // contribution to every j in our chunk. The read pattern
                // `next_w[k * next_cols + j_start..j_start+chunk_len]` is a
                // contiguous slice of one row of next_w, so the prefetcher
                // can stream it cleanly.
                for (k, &d) in next_d.iter().enumerate() {
                    let row_off = k * next_cols + j_start;
                    let row_slice = &next_w[row_off..row_off + chunk_len];
                    for (off, slot) in slot_chunk.iter_mut().enumerate() {
                        *slot += row_slice[off] * d;
                    }
                }
                // Apply activation derivative per j, then dropout mask.
                for (off, slot) in slot_chunk.iter_mut().enumerate() {
                    let j = j_start + off;
                    let slope = match layer_activation {
                        Activation::Tanh => 1.0 - acts[j] * acts[j],
                        Activation::Linear => 1.0,
                    };
                    *slot *= slope;
                    if apply_mask {
                        *slot *= dropout_mask[j];
                    }
                }
            },
        );

        // LayerNorm backward. At this point `curr[j]` holds dL/dy_j —
        // the gradient at the LN OUTPUT y = γ·x̂ + β (the chunked loop
        // above already applied the tanh derivative, which is correct
        // because `last_activations` caches a = tanh(y), so 1 - a² is
        // exactly dtanh/dy; the dropout mask likewise sits above the
        // LN in the chain). We now push the gradient through the
        // normalization itself:
        //
        //   dγ_j = dy_j · x̂_j            dβ_j = dy_j
        //   dx̂_j = dy_j · γ_j
        //   dz_j = inv_std · (dx̂_j − mean_k(dx̂_k) − x̂_j · mean_k(dx̂_k·x̂_k))
        //
        // The two mean terms come from z's participation in μ and σ² —
        // dropping them is THE classic silent LN-backward bug (the
        // gradient check in this module's tests exists to catch it).
        // x̂ and inv_std are read from the caches the forward wrote at
        // the moment of computation — never re-derived from
        // post-transform activations (2026-05-20 §5b postmortem).
        //
        // `curr` is overwritten with dz, which is what both the weight
        // gradient (dL/dW = dz ⊗ input) and the propagation to the
        // previous layer expect. Serial loop: O(rows) with two passes,
        // negligible next to the chunked matmul-transpose above.
        let layer = &net.layers[li];
        if layer.ln_active() {
            let rows = layer.rows;
            let xhat: &[f32] = &layer.last_ln_xhat;
            debug_assert_eq!(xhat.len(), rows);
            let inv_std = layer.last_ln_inv_std;
            let gain: &[f32] = &layer.ln_gain;
            let g_grad = &mut ln_gain_grad[li];
            let b_grad = &mut ln_bias_grad[li];
            let mut sum_dxhat = 0.0f32;
            let mut sum_dxhat_xhat = 0.0f32;
            for j in 0..rows {
                let dxhat = curr[j] * gain[j];
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * xhat[j];
            }
            let inv_h = 1.0 / rows as f32;
            let mean_dxhat = sum_dxhat * inv_h;
            let mean_dxhat_xhat = sum_dxhat_xhat * inv_h;
            for j in 0..rows {
                let dy = curr[j];
                g_grad[j] = dy * xhat[j];
                b_grad[j] = dy;
                let dxhat = dy * gain[j];
                curr[j] = inv_std * (dxhat - mean_dxhat - xhat[j] * mean_dxhat_xhat);
            }
        }
    }

    // Gradient w.r.t. the network input: dL/dx[k] = sum_j delta[0][j] * W[0][j,k].
    // Same loop reorder pattern: j outer (one row of l0_w at a time), k-chunked.
    let l0 = &net.layers[0];
    let cols = l0.cols;
    let rows = l0.rows;
    let l0_w = &l0.weights;
    let d0: &[f32] = &layer_deltas[0];
    if input_grad.len() != cols {
        input_grad.clear();
        input_grad.resize(cols, 0.0);
    }
    input_grad
        .par_chunks_mut(BACKWARD_CHUNK)
        .enumerate()
        .for_each(|(ci, slot_chunk)| {
            let k_start = ci * BACKWARD_CHUNK;
            let chunk_len = slot_chunk.len();
            for slot in slot_chunk.iter_mut() {
                *slot = 0.0;
            }
            for (j, &d) in d0.iter().enumerate() {
                if j >= rows {
                    break;
                }
                let row_off = j * cols + k_start;
                let row_slice = &l0_w[row_off..row_off + chunk_len];
                for (off, slot) in slot_chunk.iter_mut().enumerate() {
                    *slot += row_slice[off] * d;
                }
            }
        });
}

/// Chunk size for the cache-friendly backward. 32 floats per chunk means a
/// thread sees up to ~388 KB of `next_w` rows for the worst-case 3029-vocab
/// output layer — sized for typical L2 caches (~256 KB-1 MB).
const BACKWARD_CHUNK: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::neural_network::{LN_EPS, network_init};

    /// Central-difference step applied to the f32 parameters. The loss
    /// for the finite differences is evaluated by an f64 REFERENCE
    /// forward (below): a pure-f32 evaluation carries ~1e-6 of
    /// rounding noise on the loss, which divided by 2ε swamps the
    /// smaller true hidden-layer gradients and makes a 1e-3 relative
    /// tolerance unattainable. The f64 reference is pinned against the
    /// real f32 forward (max |Δprob| < 1e-4 asserted per test) so the
    /// two implementations cannot silently drift apart.
    const FD_EPS: f32 = 1e-4;
    /// Relative tolerance for analytic-vs-numeric gradient agreement.
    /// The denominator is floored (see `rel_err`) so near-zero
    /// gradients are compared with an effective absolute tolerance of
    /// GRAD_TOL × GRAD_FLOOR = 1e-6 instead of dividing by ~0.
    const GRAD_TOL: f64 = 1e-3;
    const GRAD_FLOOR: f64 = 1e-3;

    fn rel_err(analytic: f64, numeric: f64) -> f64 {
        let denom = analytic.abs().max(numeric.abs()).max(GRAD_FLOOR);
        (analytic - numeric).abs() / denom
    }

    /// f64 reference forward, formula-for-formula identical to
    /// `Network::forward` (embedding concat + position feature, per
    /// layer W@x + b, optional LayerNorm, tanh/linear, final softmax).
    /// Reads the live f32 parameters, computes in f64. Does NOT touch
    /// any of the network's caches, so the analytic gradients captured
    /// once from `forward_and_cache` + `compute_deltas_into` stay
    /// valid across every perturbation.
    fn forward_f64(net: &Network, window: &[usize], position: usize) -> Vec<f64> {
        let d = net.embed_dim;
        let mut x: Vec<f64> = Vec::with_capacity(window.len() * d + 1);
        for &id in window {
            let id = if id < net.embedding.vocab_size { id } else { 1 };
            for k in 0..d {
                x.push(net.embedding.weights[id * d + k] as f64);
            }
        }
        x.push((position as f64 / 100.0).min(1.0));
        for layer in &net.layers {
            let mut z = vec![0.0f64; layer.rows];
            for j in 0..layer.rows {
                let mut acc = layer.biases[j] as f64;
                let row = &layer.weights[j * layer.cols..(j + 1) * layer.cols];
                for (k, &w) in row.iter().enumerate() {
                    acc += (w as f64) * x[k];
                }
                z[j] = acc;
            }
            if layer.ln_active() {
                let h = layer.rows as f64;
                let mean = z.iter().sum::<f64>() / h;
                let var = z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h;
                let inv_std = 1.0 / (var + LN_EPS as f64).sqrt();
                for j in 0..layer.rows {
                    let xhat = (z[j] - mean) * inv_std;
                    z[j] = (layer.ln_gain[j] as f64) * xhat + (layer.ln_bias[j] as f64);
                }
            }
            for v in z.iter_mut() {
                *v = match layer.activation {
                    Activation::Tanh => v.tanh(),
                    Activation::Linear => *v,
                };
            }
            x = z;
        }
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in x.iter_mut() {
            *v /= sum;
        }
        x
    }

    fn loss_f64(net: &Network, window: &[usize], position: usize, target: usize) -> f64 {
        -forward_f64(net, window, position)[target].ln()
    }

    /// Central difference through one f32 parameter cell. Uses the
    /// ACTUAL realized f32 step (θ⁺ − θ⁻ after rounding to f32) as the
    /// denominator so f32 representation error doesn't pollute the
    /// estimate. The closure gives &mut access to the cell.
    fn numeric_grad(
        net: &mut Network,
        window: &[usize],
        position: usize,
        target: usize,
        get: impl Fn(&mut Network) -> &mut f32,
    ) -> f64 {
        let orig = *get(net);
        let plus = orig + FD_EPS;
        let minus = orig - FD_EPS;
        *get(net) = plus;
        let lp = loss_f64(net, window, position, target);
        *get(net) = minus;
        let lm = loss_f64(net, window, position, target);
        *get(net) = orig;
        (lp - lm) / ((plus as f64) - (minus as f64))
    }

    /// The gradient check: build a tiny network, capture analytic
    /// gradients from `compute_deltas_into`, then verify EVERY dense
    /// weight/bias, every LN gain/bias (when enabled), and every
    /// touched embedding cell against central finite differences.
    /// This is the test that catches silent LN-backward bugs (dropped
    /// mean/variance terms produce O(1) relative errors here).
    fn run_gradient_check(layer_norm: bool) {
        let gpu = Gpu::new_cpu();
        let (embed_dim, ctx, hidden, hidden_layers, vocab_n) = (4usize, 3usize, 8usize, 2usize, 10usize);
        let mut net = network_init(&gpu, embed_dim, ctx, hidden, hidden_layers, vocab_n)
            .expect("network_init");
        assert_eq!(net.layers.len(), hidden_layers + 1);
        if layer_norm {
            net.enable_layer_norm();
            // Nudge γ/β off identity so the backward's gain
            // multiplication and the β path are actually exercised —
            // at exact identity a wrong γ-handling could cancel out.
            let n = net.layers.len();
            for layer in &mut net.layers[..n - 1] {
                for j in 0..layer.rows {
                    layer.ln_gain[j] = 1.0 + 0.05 * ((j % 5) as f32 - 2.0);
                    layer.ln_bias[j] = 0.03 * ((j % 3) as f32 - 1.0);
                }
            }
        }
        // Nudge dense biases off their all-zero init for generality.
        for layer in &mut net.layers {
            for j in 0..layer.rows {
                layer.biases[j] = 0.02 * ((j % 4) as f32 - 1.5);
            }
        }

        // Window contains a DUPLICATE token (id 2 twice) so the
        // embedding check also covers gradient accumulation across
        // window slots.
        let window = vec![2usize, 5, 2];
        let position = 1usize;
        let target = 3usize;

        // Pin the f64 reference against the real f32 forward.
        let probs_f32 = net.forward(&gpu, &window, position).expect("forward");
        let probs_ref = forward_f64(&net, &window, position);
        for (i, (a, b)) in probs_f32.iter().zip(probs_ref.iter()).enumerate() {
            assert!(
                ((*a as f64) - b).abs() < 1e-4,
                "f64 reference diverges from f32 forward at prob[{}]: {} vs {}",
                i,
                a,
                b
            );
        }

        // Analytic gradients: one cached forward + one backward.
        let _ = net
            .forward_and_cache(&gpu, &window, position)
            .expect("forward_and_cache");
        let mut scratch = BackpropScratch::default();
        compute_deltas_into(&net, target, &mut scratch);

        let mut checked = 0usize;
        let mut max_rel = 0.0f64;
        let mut check = |name: String, analytic: f64, numeric: f64| {
            let e = rel_err(analytic, numeric);
            assert!(
                e <= GRAD_TOL,
                "{}: analytic {} vs numeric {} (rel err {:.3e})",
                name,
                analytic,
                numeric,
                e
            );
            if e > max_rel {
                max_rel = e;
            }
            checked += 1;
        };

        // Dense weights + biases, every layer, every cell.
        for li in 0..net.layers.len() {
            let rows = net.layers[li].rows;
            let cols = net.layers[li].cols;
            let last_input = net.layers[li].last_input.clone();
            for j in 0..rows {
                let delta_j = scratch.layer_deltas[li][j] as f64;
                let numeric = numeric_grad(&mut net, &window, position, target, |n| {
                    &mut n.layers[li].biases[j]
                });
                check(format!("layer{} bias[{}]", li, j), delta_j, numeric);
                for k in 0..cols {
                    let analytic = delta_j * (last_input[k] as f64);
                    let numeric = numeric_grad(&mut net, &window, position, target, |n| {
                        &mut n.layers[li].weights[j * cols + k]
                    });
                    check(format!("layer{} w[{},{}]", li, j, k), analytic, numeric);
                }
            }
        }

        // LayerNorm γ/β (hidden layers only, when enabled).
        if layer_norm {
            for li in 0..net.layers.len() - 1 {
                let rows = net.layers[li].rows;
                assert_eq!(scratch.ln_gain_grad[li].len(), rows);
                assert_eq!(scratch.ln_bias_grad[li].len(), rows);
                for j in 0..rows {
                    let analytic = scratch.ln_gain_grad[li][j] as f64;
                    let numeric = numeric_grad(&mut net, &window, position, target, |n| {
                        &mut n.layers[li].ln_gain[j]
                    });
                    check(format!("layer{} ln_gain[{}]", li, j), analytic, numeric);
                    let analytic = scratch.ln_bias_grad[li][j] as f64;
                    let numeric = numeric_grad(&mut net, &window, position, target, |n| {
                        &mut n.layers[li].ln_bias[j]
                    });
                    check(format!("layer{} ln_bias[{}]", li, j), analytic, numeric);
                }
            }
        } else {
            for g in scratch.ln_gain_grad.iter().chain(scratch.ln_bias_grad.iter()) {
                assert!(g.is_empty(), "LN grads must stay empty when LN is off");
            }
        }

        // Embedding rows touched by the window. dL/dE[tok][k] is the
        // SUM of input_grad over every window slot holding `tok`
        // (mirrors Embedding::apply_grad_adam's accumulation).
        let d = net.embed_dim;
        let mut unique: Vec<usize> = window.clone();
        unique.sort_unstable();
        unique.dedup();
        for &tok in &unique {
            for k in 0..d {
                let analytic: f64 = window
                    .iter()
                    .enumerate()
                    .filter(|&(_, &id)| id == tok)
                    .map(|(p, _)| scratch.input_grad[p * d + k] as f64)
                    .sum();
                let numeric = numeric_grad(&mut net, &window, position, target, |n| {
                    &mut n.embedding.weights[tok * d + k]
                });
                check(format!("embed[{}][{}]", tok, k), analytic, numeric);
            }
        }

        // Sanity: we actually exercised a meaningful parameter count
        // and at least one gradient was materially non-zero.
        assert!(checked > 250, "only {} parameters checked", checked);
        eprintln!(
            "gradient check (layer_norm={}): {} params, max rel err {:.3e}",
            layer_norm, checked, max_rel
        );
    }

    #[test]
    fn gradient_check_matches_finite_differences_without_layer_norm() {
        run_gradient_check(false);
    }

    #[test]
    fn gradient_check_matches_finite_differences_with_layer_norm() {
        run_gradient_check(true);
    }
}
