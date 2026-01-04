use crate::neural_network::Network;
use crate::machine_learning::{string_similarity};
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

/*
pub fn word_change_delta(word: &str, teacher_word: &str, net: &Network) -> Vec<Vec<f32>> {
    // Turn loss into responsibility
    let mut net_change: Vec<Vec<f32>> = Vec::with_capacity(net.Layers.len());
    for (layer_idx, layer) in net.Layers.iter().enumerate() {
        net_change.push(Vec::new());
        if layer_idx == net.Layers.len()-1 {
            for n_cache in &layer.Cache {
                let loss = -(0.000000001f32 + (string_similarity(word, teacher_word) as f32)).ln();
                net_change[layer_idx].push(output_delta_sigmoid(loss, n_cache.Activation));
            }
        }
        else {
            for n_cache in &layer.Cache {
                let loss = -(0.000000001f32 + (string_similarity(word, teacher_word) as f32)).ln();

                let j = net_change[layer_idx].len();
                let delta = hidden_delta_sigmoid(
                    n_cache.Activation,
                    j,
                    &net.Layers[layer_idx + 1].Weights,
                    &net_change[layer_idx + 1],
                );

                net_change[layer_idx].push(delta);
            }
        }
    } 
    net_change
}
*/
pub fn word_change_delta(word: &str, teacher_word: &str, net: &Network) -> Vec<Vec<f32>> {
    let lcount = net.Layers.len();

    // Pre-allocate deltas for each layer
    let mut deltas: Vec<Vec<f32>> = net
        .Layers
        .iter()
        .map(|layer| vec![0.0f32; layer.Cache.len()])
        .collect();

    // 1) Output layer first
    let out = lcount - 1;
    for (i, n_cache) in net.Layers[out].Cache.iter().enumerate() {
        let loss = -(0.000000001f32 + (string_similarity(word, teacher_word) as f32)).ln();
        deltas[out][i] = output_delta_sigmoid(loss, n_cache.Activation);
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
