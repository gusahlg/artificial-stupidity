use crate::neural_network::Network;

// sigmoid'(z) computed from activation a: a*(1-a)
fn sigmoid_prime_from_activation(a: f32) -> f32 {
    a * (1.0 - a)
}

// Returns deltas[layer_index][neuron_index]
pub fn weight_change(net: &Network, target_index: usize, strength: f32) -> Vec<Vec<f32>> {
    let layer_count = net.Layers.len();
    let mut deltas: Vec<Vec<f32>> = vec![Vec::new(); layer_count];

    // ----- 1) Output layer deltas -----
    let out_l = layer_count - 1;
    let out_cache = &net.Layers[out_l].Cache;

    let mut delta_out = vec![0.0f32; out_cache.len()];
    for j in 0..out_cache.len() {
        let y = if j == target_index { 1.0 } else { 0.0 };
        let a = out_cache[j].Activation;
        delta_out[j] = (a - y) * sigmoid_prime_from_activation(a) * strength;
    }
    deltas[out_l] = delta_out;

    // ----- 2) Hidden layer deltas (backprop) -----
    // delta_l[k] = (sum_j W_next[j][k] * delta_next[j]) * a*(1-a)
    for l in (0..out_l).rev() {
        let next = l + 1;

        let a_l = &net.Layers[l].Cache;           // activations of layer l
        let w_next = &net.Layers[next].Weights;   // [j][k]
        let delta_next = &deltas[next];

        let mut delta_l = vec![0.0f32; a_l.len()];
        for k in 0..a_l.len() {
            let mut sum = 0.0f32;
            for j in 0..delta_next.len() {
                sum += w_next[j][k] * delta_next[j];
            }
            let a = a_l[k].Activation;
            delta_l[k] = sum * sigmoid_prime_from_activation(a);
        }

        deltas[l] = delta_l;
    }

    deltas
}


