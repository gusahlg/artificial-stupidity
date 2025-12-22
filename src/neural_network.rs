use crate::memory::{load};
use rand::Rng;
use std::collections::HashSet;
const NUMBER_OF_LAYERS: usize = 100;

fn sigmoid(x: f32) -> f32 {
    let exponent = (-x).exp(); // e^(-x)
    let denominator = 1.0 + exponent;
    return 1.0 / denominator;
}
struct Layer {
    Weights: Vec<Vec<f32>>,
    Biases: Vec<f32>,
}
impl Layer {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut outputs = Vec::with_capacity(self.Biases.len());
        for neuron_index in 0..self.Biases.len() {
            let mut sum = self.Biases[neuron_index];
            for i in 0..input.len() {
                sum += input[i] * self.Weights[neuron_index][i];
            }
        outputs.push(sigmoid(sum));
        }
        outputs
    }
}
pub struct Network {
    Layers: Vec<Layer>,
}
impl Network {
    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &self.Layers {
            input = layer.forward(&input);
        }
        input
    }
}
fn interpret_input(input: &str, memory: &Vec<String>) -> Vec<f32> {
    let input_text: String = memory.join(" ") + " " + input;

    let word_count = input_text.split_whitespace().count();
    let words: Vec<&str> = input_text.split_whitespace().collect();

    // Feature 1: average word length
    let avg = words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
    let f1 = (avg / 10.0).min(1.0);

    // Feature 2: number of words (roughly spaces + 1), scaled
    let f2 = (word_count as f32 / 50.0).min(1.0);

    // Feature 3: punctuation amount (e.g. question marks)
    let question_marks = input_text.chars().filter(|&c| c == '?').count();
    let f3 = (question_marks as f32 / 5.0).min(1.0);

    // Feature 4: vowel to consonant ratio
    let vowels = "aeiou";
    let mut v = 0;
    let mut c = 0;
    for ch in input_text.to_lowercase().chars(){
        if vowels.contains(ch){v +=1;}
        else {c += 1;} 
    }
    let ratio = v as f32 / c as f32;
    let f4 = (ratio / 5.0).min(1.0);

    // Feature 5: number of common letters
    let mut score = 0;
    for ch in input_text.to_lowercase().chars(){
        if ch == 'e' {score+=10}
        else if ch == 'a'{score+=9}
        else if ch == 'r'{score+=8}
        else if ch == 'i'{score+=7}
        else if ch == 'o'{score+=6}
        else if ch == 't'{score+=5}
        else if ch == 'n'{score+=4}
        else if ch == 's'{score+=3}
    }
    let f5 = score as f32 / 5.0;

    // Feature 6: letter diversity
    let len = input_text.len() as f32;
    let uniq = input_text.chars().collect::<HashSet<char>>().len() as f32;
    let f6 = uniq / len;

    // Feature 7: 
    // Feature 8: 
    // Feature 9: 
    // Feature 10: 
    vec![f1, f2, f3, f4, f5, f6/*, f7, f8, f9, f10*/]
}

pub fn network_init(hidden_size: usize, output_size: usize) -> Network {
    let layers: Vec<Layer> = Vec::new();
    let mut neural_network: Network = Network{Layers: layers};

    for _ in 0..NUMBER_OF_LAYERS {
        let hidden_weights: Vec<Vec<f32>> = (0..hidden_size).map(|_| Vec::new()).collect();
        let hidden_biases: Vec<f32> = vec!(0.0f32; hidden_size);

        neural_network.Layers.push(
            Layer{ Weights: hidden_weights, Biases: hidden_biases, }
        );
    }
    let output_weights: Vec<Vec<f32>> =  (0..output_size).map(|_| Vec::new()).collect();
    let output_biases: Vec<f32> = vec!(0.0f32; output_size);
    neural_network.Layers.push(Layer{Weights: output_weights, Biases: output_biases,});
    neural_network
}
struct VocabPair {
    activation: f32,
    word: String,
}
fn interpret_output(activations: Vec<f32>, words: &Vec<String>) -> String{
    // Pair all output neuron activations with words
    let mut pairs: Vec<VocabPair> = Vec::with_capacity(activations.len());
    pairs.retain(|p| !p.activation.is_nan());
    for i in 0..activations.len() {
        pairs.push(VocabPair {
            activation: activations[i],
            word: words[i].clone(),
        });
    }
    pairs.sort_by(|a, b| b.activation.partial_cmp(&a.activation)
        .unwrap_or(std::cmp::Ordering::Equal));

    let choices = 3;
    
    let idx = rand::thread_rng().gen_range(0..choices);
    pairs[idx].word.clone()
}

pub fn generate(net: &Network, start: &str, memory: &Vec<String>) -> String {
    let mut text = start.to_string();
    
    let words = load();
    for _ in 0..100 {
        let features = interpret_input(&text, &memory);
        let activations = net.forward(features);
        let next_word = interpret_output(activations, &words);
        
        text.push(' ');
        text.push_str(&next_word);
    }

    let len = start.to_string().len();
    text[len..].to_string()
}


