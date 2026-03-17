use crate::neural_network::{Network, WordIndexPair};
fn output_delta_sigmoid(error: f32, activation: f32) -> f32 {
    // sigmoid'(z) can be computed from activation: a * (1 - a)
    let slope = activation * (1.0 - activation);
    error * slope
}

fn hidden_delta_sigmoid(hidden_activation: f32, neuron_j: usize, next_layer_weights: &Vec<Vec<f32>>, next_layer_deltas: &Vec<f32>) -> f32 {
    let mut weighted_sum = 0.0f32;
    for k in 0..next_layer_deltas.len() {
        weighted_sum += next_layer_weights[k][neuron_j] * next_layer_deltas[k];
    }
    let slope = hidden_activation * (1.0 - hidden_activation);
    weighted_sum * slope
}

pub fn word_change_delta(word_and_index: &WordIndexPair, teacher_word: &str, vocab: &Vec<String>, net: &Network) -> Vec<Vec<f32>> {
    let lcount = net.Layers.len();

    // Pre-allocate deltas for each layer
    let mut deltas: Vec<Vec<f32>> = net
        .Layers
        .iter()
        .map(|layer| vec![0.0f32; layer.Cache.len()])
        .collect();

    // Find the index of the teacher word in vocabulary
    let teacher_word_idx = vocab
        .iter()
        .position(|w| w == teacher_word)
        .unwrap();

    // Output layer: set target for each neuron
    let out = lcount - 1;
    for (i, n_cache) in net.Layers[out].Cache.iter().enumerate() {
        let target = if i == teacher_word_idx {
            1.0  // Correct word - should fire
        } else {
            0.0  // Wrong word - should not fire
        };

        let error = target - n_cache.Activation;
        deltas[out][i] = output_delta_sigmoid(error, n_cache.Activation);
    }

    // 2) Hidden layers backwards, using split_at_mut to avoid borrow conflict
    for layer_idx in (0..out).rev() {
        // deltas_left contains [0..=layer_idx], deltas_right contains [layer_idx+1..]
        let (deltas_left, deltas_right) = deltas.split_at_mut(layer_idx + 1);

        let curr_deltas = &mut deltas_left[layer_idx];
        let next_deltas = &deltas_right[0]; // this is deltas[layer_idx + 1]

        let next_weights = &net.Layers[layer_idx + 1].Weights;

        for (j, n_cache) in net.Layers[layer_idx].Cache.iter().enumerate() {
            curr_deltas[j] = hidden_delta_sigmoid(
                n_cache.Activation,
                j,
                next_weights,
                next_deltas,
            );
        }
    }

    deltas
}
