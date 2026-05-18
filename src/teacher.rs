use crate::neural_network::{Activation, Network};
use rayon::prelude::*;

/// Per-layer post-activation gradients **plus** the gradient with respect to
/// the network's input vector, so the embedding layer can receive its share.
pub struct BackpropOutput {
    pub layer_deltas: Vec<Vec<f32>>,
    pub input_grad: Vec<f32>,
}

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
pub fn compute_deltas(net: &Network, target_idx: usize) -> BackpropOutput {
    let n = net.layers.len();
    let mut layer_deltas: Vec<Vec<f32>> = net
        .layers
        .iter()
        .map(|l| vec![0.0f32; l.last_activations.len()])
        .collect();

    // Output layer: softmax - one_hot.
    let out_acts = &net.layers[n - 1].last_activations;
    for i in 0..out_acts.len() {
        let t = if i == target_idx { 1.0 } else { 0.0 };
        layer_deltas[n - 1][i] = out_acts[i] - t;
    }

    // Hidden layers: walk row-major next_w in k-outer order, j-chunked.
    for li in (0..n - 1).rev() {
        let next_layer = &net.layers[li + 1];
        let next_cols = next_layer.cols;
        let acts = &net.layers[li].last_activations;
        let layer_activation = net.layers[li].activation;
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
                // Apply activation derivative per j.
                for (off, slot) in slot_chunk.iter_mut().enumerate() {
                    let j = j_start + off;
                    let slope = match layer_activation {
                        Activation::Tanh => 1.0 - acts[j] * acts[j],
                        Activation::Linear => 1.0,
                    };
                    *slot *= slope;
                }
            },
        );
    }

    // Gradient w.r.t. the network input: dL/dx[k] = sum_j delta[0][j] * W[0][j,k].
    // Same loop reorder pattern: j outer (one row of l0_w at a time), k-chunked.
    let l0 = &net.layers[0];
    let cols = l0.cols;
    let rows = l0.rows;
    let l0_w = &l0.weights;
    let d0: &[f32] = &layer_deltas[0];
    let mut input_grad = vec![0.0f32; cols];
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

    BackpropOutput {
        layer_deltas,
        input_grad,
    }
}

/// Chunk size for the cache-friendly backward. 32 floats per chunk means a
/// thread sees up to ~388 KB of `next_w` rows for the worst-case 3029-vocab
/// output layer — sized for typical L2 caches (~256 KB-1 MB).
const BACKWARD_CHUNK: usize = 32;
