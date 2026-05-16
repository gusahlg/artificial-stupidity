use crate::neural_network::{Activation, Network};

/// Standard softmax + cross-entropy backprop. The output layer's cached
/// `last_activations` already hold post-softmax probabilities, so the output
/// delta collapses to (p - one_hot[target]). Hidden layers use tanh', i.e.
/// 1 - a^2 where a is the cached tanh activation.
pub fn compute_deltas(net: &Network, target_idx: usize) -> Vec<Vec<f32>> {
    let n = net.layers.len();
    let mut deltas: Vec<Vec<f32>> = net
        .layers
        .iter()
        .map(|l| vec![0.0f32; l.last_activations.len()])
        .collect();

    // Output layer: delta = softmax - one_hot
    let out_acts = &net.layers[n - 1].last_activations;
    for i in 0..out_acts.len() {
        let t = if i == target_idx { 1.0 } else { 0.0 };
        deltas[n - 1][i] = out_acts[i] - t;
    }

    // Backprop through hidden layers
    for li in (0..n - 1).rev() {
        let next_layer = &net.layers[li + 1];
        let next_cols = next_layer.cols;
        let acts = &net.layers[li].last_activations;
        let (left, right) = deltas.split_at_mut(li + 1);
        let curr = &mut left[li];
        let next_d = &right[0];

        for j in 0..curr.len() {
            let mut sum = 0.0f32;
            for k in 0..next_d.len() {
                sum += next_layer.weights[k * next_cols + j] * next_d[k];
            }
            let slope = match net.layers[li].activation {
                Activation::Tanh => 1.0 - acts[j] * acts[j],
                Activation::Linear => 1.0,
            };
            curr[j] = sum * slope;
        }
    }

    deltas
}
