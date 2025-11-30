mod memory;
use crate::memory::{load, append};
use rand::Rng;

fn sigmoid(x: f32) -> f32 {
    let exponent = (-x).exp(); // e^(-x)
    let denominator = 1.0 + exponent;
    return 1.0 / denominator;
}
struct Layer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}
impl Layer {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut outputs = Vec::with_capacity(self.biases.len());
        for neuron_index in 0..self.biases.len(){
            let mut sum = self.biases[neuron_index];
            for i in 0..input.len(){
                sum += input[i] * self.weights[neuron_index][i];
            }
            let activated = sigmoid(sum);
            outputs.push(activated);
        }
        return outputs;
    }
}
pub struct Network {
    layers: Vec<Layer>,
}
impl Network {
    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            input = layer.forward(&input);
        }
        input
    }
}
pub fn interpret_input(input_text: &str) -> Vec<f32> {
    // Feature 1: message length
    let f1 = (input_text.len() as f32 / 100.0).min(1.0);

    // Feature 2: number of messages in memory
    let f2 = (memory.len() as f32 / 20.0).min(1.0);

    // Feature 3: total characters in memory
    let total_chars: usize = memory.iter().map(|s| s.len()).sum();
    let f3 = (total_chars as f32 / 1000.0).min(1.0);

    // Feature 4: contains greeting?
    let lower = input_text.to_lowercase();
    let f4 = if lower.contains("hello")
        || lower.contains("hi")
        || lower.contains("hej")
        || lower.contains("tja")
    {
        1.0
    } else {
        0.0
    };

    vec![f1, f2, f3, f4]
}
pub fn network_init(input_size: usize, hidden_size: usize, output_size: usize) -> Network {
    let mut hidden_weights: Vec<Vec<f32>> = Vec::new();
    let mut hidden_biases: Vec<f32> = Vec::new();
    for h in 0..hidden_size{
        let mut row: Vec<f32> = Vec::new();
        for i in 0..input_size{
            let w = ((h + i) as f32 / (input_size + hidden_size) as f32) -0.5;
            row.push(w);
        }
        hidden_weights.push(row);
    }
    for h in 0..hidden_size{
        let b = (h as f32 / hidden_size as f32) -0.5;
        hidden_biases.push(b);
    }
    let hidden_layer = Layer{
        weights: hidden_weights,
        biases: hidden_biases,
    };

    let mut output_weights: Vec<Vec<f32>> = Vec::new();
    let mut output_biases: Vec<f32> = Vec::new();

    for o in 0..output_size{
        let mut row: Vec<f32> = Vec::new();
        for h in 0..hidden_size{
            let w = ((o + h) as f32 / (hidden_size + output_size) as f32) - 0.5;
            row.push(w);
        }
        output_weights.push(row);
    }
    for o in 0..output_size{
        let b = (o as f32 / output_size.max(1) as f32) - 0.5;
        output_biases.push(b);
    }

    
    let output_layer = Layer {
        weights: output_weights,
        biases: output_biases,
    };
    Network {
        layers: vec![hidden_layer, output_layer],
    }

}
struct VocabPair {
    activation: f32,
    word: String,
}
//NOTE translate number into a value from the vocab through save and load.
fn interpret_output(activations: Vec<f32>) -> String{
    // Pair all output neuron activations with words
    let pairs = Vec<VocabPair>::with_capacity(activations.size());
    let words = load();
    for i in 0..activations.size(){
        pairs.push({activations[i], words[i]});
    }

    
    // Randomly pick number.
    let idx = rand::thread_rng().gen_range(0..10);

    pairs[idx].word 
}

pub fn generate(net: &Network, input: mut String) -> String{
    for i in 0..100{
        let activations = net.forward(interpret_input(&input as &str))
        input = interpret_output(activations);
    }
    input
}

