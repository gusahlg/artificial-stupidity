use crate::neural_network::Network;
use rand::Rng;
use std::collections::HashSet;

/// Per-step negative samples for output-layer training. The multi-output sigmoid
/// design pulls *every* off-class activation toward 0 each step, which on a vocab
/// of thousands drowns out the single "should fire" signal. Sampling a handful of
/// negatives keeps the signal balanced without changing the architecture.
const NEG_SAMPLES: usize = 16;

fn output_delta_sigmoid(error: f32, activation: f32) -> f32 {
    let slope = activation * (1.0 - activation);
    error * slope
}

pub fn word_change_delta(
    teacher_word: &str,
    vocab: &[String],
    net: &Network,
) -> Option<Vec<Vec<f32>>> {
    let teacher_word_idx = vocab.iter().position(|w| w == teacher_word)?;
    let lcount = net.layers.len();

    let mut deltas: Vec<Vec<f32>> = net
        .layers
        .iter()
        .map(|layer| vec![0.0f32; layer.cache.len()])
        .collect();

    // Output layer: positive target on the teacher word; pull a small random set
    // of negatives toward 0. All other output neurons get zero gradient this step.
    let out = lcount - 1;
    let mut negatives: HashSet<usize> = HashSet::with_capacity(NEG_SAMPLES);
    if vocab.len() > 1 {
        let mut rng = rand::thread_rng();
        let target_neg = NEG_SAMPLES.min(vocab.len() - 1);
        while negatives.len() < target_neg {
            let n = rng.gen_range(0..vocab.len());
            if n != teacher_word_idx {
                negatives.insert(n);
            }
        }
    }
    for (i, n_cache) in net.layers[out].cache.iter().enumerate() {
        if i == teacher_word_idx {
            let error = 1.0 - n_cache.activation;
            deltas[out][i] = output_delta_sigmoid(error, n_cache.activation);
        } else if negatives.contains(&i) {
            let error = -n_cache.activation;
            deltas[out][i] = output_delta_sigmoid(error, n_cache.activation);
        }
    }

    // Hidden layers backward, using split_at_mut to avoid borrow conflict on `deltas`.
    for layer_idx in (0..out).rev() {
        let (deltas_left, deltas_right) = deltas.split_at_mut(layer_idx + 1);
        let curr_deltas = &mut deltas_left[layer_idx];
        let next_deltas = &deltas_right[0];

        let next_layer = &net.layers[layer_idx + 1];
        let next_cols = next_layer.cols;

        for (j, n_cache) in net.layers[layer_idx].cache.iter().enumerate() {
            let mut weighted_sum = 0.0f32;
            for k in 0..next_deltas.len() {
                weighted_sum += next_layer.weights[k * next_cols + j] * next_deltas[k];
            }
            let slope = n_cache.activation * (1.0 - n_cache.activation);
            curr_deltas[j] = weighted_sum * slope;
        }
    }

    Some(deltas)
}
