use crate::dialogs::{Data};
use crate::teacher::{word_change_delta};
use crate::machine_learning::{teacher_response};
use rand::Rng;
use std::collections::HashSet;
const NUMBER_OF_LAYERS: usize = 500;

fn sigmoid(x: f32) -> f32 {
    let exponent = (-x).exp(); // e^(-x)
    let denominator = 1.0 + exponent;
    return 1.0 / denominator;
}
pub struct NeuronCache {
    pub(crate) Activation: f32,
    pub(crate) PreActivation: f32,
}
pub(crate) struct Layer {
     pub(crate)Weights: Vec<Vec<f32>>,
    pub(crate)Biases: Vec<f32>,

    pub(crate)Cache: Vec<NeuronCache>,
    pub(crate)LastInput: Vec<f32>,
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
    fn forward_and_cache(&mut self, input: &[f32]) -> Vec<f32> {
        self.Cache.clear();
        self.LastInput.clear();
        self.LastInput.extend_from_slice(input);

        let mut outputs = Vec::with_capacity(self.Biases.len());
        for neuron_index in 0..self.Biases.len() {
            let mut sum = self.Biases[neuron_index];
            for i in 0..input.len() {
                sum += input[i] * self.Weights[neuron_index][i];
            }

            let a = sigmoid(sum);
            outputs.push(a);
            self.Cache.push(NeuronCache { Activation: a, PreActivation: sum });
        }
        outputs
    }
}
pub struct Network {
    pub(crate) Layers: Vec<Layer>,
}
impl Network {
    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &self.Layers {
            input = layer.forward(&input);
        }
        input
    }
    pub fn forward_and_cache(&mut self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &mut self.Layers {
            input = layer.forward_and_cache(&input);
        }
        input
    }

    fn update_neuron_sgd(weights_j: &mut [f32], bias_j: &mut f32, prev_acts: &[f32], delta_j: f32, lr: f32) {
        *bias_j -= lr * delta_j;
        for (w, &a_prev) in weights_j.iter_mut().zip(prev_acts.iter()) {
            *w -= lr * delta_j * a_prev;
        }
    }

    pub fn adjust_weights(&mut self, lr: f32, teacher_response: String, word: String) {
        let layer_deltas = word_change_delta(&word, &teacher_response, self);
        for (layer_idx, layer) in self.Layers.iter_mut().enumerate() {
            for neuron in 0..layer.Biases.len() {
                let n_delta = layer_deltas[layer_idx][neuron];
                Network::update_neuron_sgd(&mut layer.Weights[neuron], &mut layer.Biases[neuron], &layer.LastInput, n_delta, lr); 
            }
        }
    }
}

fn interpret_input(input: &str, memory: &Vec<String>) -> Vec<f32> {
    let input_text: String = memory.join(" ") + " " + input;

    let word_count = input_text.split_whitespace().count();
    let words: Vec<&str> = input_text.split_whitespace().collect();

    let avg = words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32;
    let f1 = (avg / 10.0).min(1.0);

    let f2 = (word_count as f32 / 50.0).min(1.0);

    let question_marks = input_text.chars().filter(|&c| c == '?').count();
    let f3 = (question_marks as f32 / 5.0).min(1.0);

    let vowels = "aeiou";
    let mut v = 0;
    let mut c = 0;
    for ch in input_text.to_lowercase().chars(){
        if vowels.contains(ch){v +=1;}
        else {c += 1;} 
    }
    let ratio = v as f32 / c as f32;
    let f4 = (ratio / 5.0).min(1.0);

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

    let len = input_text.len() as f32;
    let uniq = input_text.chars().collect::<HashSet<char>>().len() as f32;
    let f6 = uniq / len;

    vec![f1, f2, f3, f4, f5, f6/*, f7, f8, f9, f10*/]
}

pub fn network_init(hidden_size: usize, output_size: usize) -> Network {
    let mut rng = rand::thread_rng();

    let input_size: usize = 6;

    let mut neural_network = Network { Layers: Vec::new() };

    let w0: Vec<Vec<f32>> = (0..hidden_size)
        .map(|_| {
            (0..input_size)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect();
    let b0: Vec<f32> = vec![0.0; hidden_size];
    neural_network.Layers.push(Layer { Cache: Vec::with_capacity(hidden_size), LastInput: Vec::new(), Weights: w0, Biases: b0 });

    for _ in 1..NUMBER_OF_LAYERS {
        let w: Vec<Vec<f32>> = (0..hidden_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let b: Vec<f32> = vec![0.0; hidden_size];
        neural_network.Layers.push(Layer { Cache: Vec::with_capacity(hidden_size), LastInput: Vec::new(), Weights: w, Biases: b });
    }

    let w_out: Vec<Vec<f32>> = (0..output_size)
        .map(|_| {
            (0..hidden_size)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect();
    let b_out: Vec<f32> = vec![0.0; output_size];
    neural_network.Layers.push(Layer { Cache: Vec::new(), LastInput: Vec::new(), Weights: w_out, Biases: b_out });

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

pub fn generate(net: &Network, start: &str, memory: &Vec<String>, words: Vec<String>) -> String {
    let mut text = start.to_string();
    for _ in 0..100 {
        // 1. Make text into input for network
        // 2. Let the network do the thinking
        // 3. Convert output into a word

        let features = interpret_input(&text, &memory);
        let activations = net.forward(features);
        let next_word = interpret_output(activations, &words);
        
        // 4. Look at how good the word matches with the corresponding word in the teacher response
        //    and then adjust neurons in network accordingly.
        text.push(' ');
        text.push_str(&next_word);
    }

    let len = start.to_string().len();
    text[len..].to_string()
}
pub fn generate_and_train(net: &mut Network, start: &str, memory: &Vec<String>, words: Vec<String>, data: &Data, lr: f32) -> String {
    let mut text = start.to_string();
    let teacher: Vec<String> =
    teacher_response(data, memory, start)
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();

    for i in 0..100 {
        let features = interpret_input(&text, &memory);
        let activations = net.forward_and_cache(features);
        let next_word = interpret_output(activations, &words);
        
        let teacher_word = match teacher.get(i) {
            Some(s) => s.clone(),
            None => {
                text.push(' ');
                text.push_str(&next_word);
                let len = start.to_string().len();
                return text[len..].to_string();
            },
        };
        net.adjust_weights(lr, teacher_word, next_word.clone());
        text.push(' ');
        text.push_str(&next_word);
    }

    let len = start.to_string().len();
    text[len..].to_string()
}
