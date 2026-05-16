use crate::neural_network::{Activation, Network};

/// Per-layer post-activation gradients **plus** the gradient with respect to
/// the network's input vector, so the embedding layer can receive its share.
pub struct BackpropOutput {
    pub layer_deltas: Vec<Vec<f32>>,
    pub input_grad: Vec<f32>,
}

/// Standard softmax + cross-entropy backprop. Output layer delta collapses to
/// (softmax - one_hot); hidden layers use tanh' = 1 - a². Finally we compute
/// dL/dx for the input vector x = (concatenated embeddings || position).
pub fn compute_deltas(net: &Network, target_idx: usize) -> BackpropOutput {
    let n = net.layers.len();
    let mut layer_deltas: Vec<Vec<f32>> = net
        .layers
        .iter()
        .map(|l| vec![0.0f32; l.last_activations.len()])
        .collect();

    // Output layer
    let out_acts = &net.layers[n - 1].last_activations;
    for i in 0..out_acts.len() {
        let t = if i == target_idx { 1.0 } else { 0.0 };
        layer_deltas[n - 1][i] = out_acts[i] - t;
    }

    // Hidden layers
    for li in (0..n - 1).rev() {
        let next_layer = &net.layers[li + 1];
        let next_cols = next_layer.cols;
        let acts = &net.layers[li].last_activations;
        let (left, right) = layer_deltas.split_at_mut(li + 1);
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

    // Gradient w.r.t. the network input: dL/dx[k] = sum_j delta[0][j] * W[0][j,k].
    let l0 = &net.layers[0];
    let mut input_grad = vec![0.0f32; l0.cols];
    for k in 0..l0.cols {
        let mut sum = 0.0f32;
        for j in 0..l0.rows {
            sum += layer_deltas[0][j] * l0.weights[j * l0.cols + k];
        }
        input_grad[k] = sum;
    }

    BackpropOutput {
        layer_deltas,
        input_grad,
    }
}
