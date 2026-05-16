use crate::dialogs::Data;
use crate::gpu::Gpu;
use crate::machine_learning::teacher_response;
use crate::teacher::word_change_delta;
use anyhow::Result;
use ml_project::Tensor;
use rand::Rng;
use std::collections::HashSet;

const NUMBER_OF_HIDDEN_LAYERS: usize = 15;
pub const INPUT_SIZE: usize = 7;
const TOP_K_SAMPLE: usize = 3;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct NeuronCache {
    pub activation: f32,
}

pub struct Layer {
    pub weights: Vec<f32>, // [rows * cols], row-major
    pub biases: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub cache: Vec<NeuronCache>,
    pub last_input: Vec<f32>,

    // GPU mirror of weights + scratch tensors for the per-layer matvec.
    gpu_weights: Tensor,
    gpu_input: Tensor,
    gpu_output: Tensor,
    matmul_out: Vec<f32>,
    gpu_dirty: bool,
}

impl Layer {
    fn new(rows: usize, cols: usize, gpu: &Gpu, rng: &mut rand::rngs::ThreadRng) -> Result<Self> {
        // Xavier/Glorot for sigmoid: uniform[-bound, bound], bound = sqrt(6 / (in + out)).
        let bound = (6.0f32 / (rows + cols) as f32).sqrt();
        let mut weights = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            weights.push(rng.gen_range(-bound..bound));
        }
        let biases = vec![0.0f32; rows];

        let gpu_weights = Tensor::zeros_device(&gpu.ctx, &[rows as u32, cols as u32])?;
        gpu.exec.upload(&weights, &gpu_weights)?;
        let gpu_input = Tensor::zeros_device(&gpu.ctx, &[cols as u32, 1])?;
        let gpu_output = Tensor::zeros_device(&gpu.ctx, &[rows as u32, 1])?;

        Ok(Self {
            weights,
            biases,
            rows,
            cols,
            cache: Vec::with_capacity(rows),
            last_input: Vec::new(),
            gpu_weights,
            gpu_input,
            gpu_output,
            matmul_out: vec![0.0; rows],
            gpu_dirty: false,
        })
    }

    fn sync_weights_if_dirty(&mut self, gpu: &Gpu) -> Result<()> {
        if self.gpu_dirty {
            gpu.exec.upload(&self.weights, &self.gpu_weights)?;
            self.gpu_dirty = false;
        }
        Ok(())
    }

    fn forward(&mut self, gpu: &Gpu, input: &[f32], cache: bool) -> Result<Vec<f32>> {
        debug_assert_eq!(input.len(), self.cols);
        self.sync_weights_if_dirty(gpu)?;
        gpu.matvec(
            &self.gpu_weights,
            &self.gpu_input,
            &self.gpu_output,
            input,
            &mut self.matmul_out,
        )?;

        if cache {
            self.cache.clear();
            self.last_input.clear();
            self.last_input.extend_from_slice(input);
        }

        let mut outputs = Vec::with_capacity(self.rows);
        for j in 0..self.rows {
            let z = self.matmul_out[j] + self.biases[j];
            let a = sigmoid(z);
            outputs.push(a);
            if cache {
                self.cache.push(NeuronCache { activation: a });
            }
        }
        Ok(outputs)
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn forward(&mut self, gpu: &Gpu, mut input: Vec<f32>) -> Result<Vec<f32>> {
        for layer in &mut self.layers {
            input = layer.forward(gpu, &input, false)?;
        }
        Ok(input)
    }

    pub fn forward_and_cache(&mut self, gpu: &Gpu, mut input: Vec<f32>) -> Result<Vec<f32>> {
        for layer in &mut self.layers {
            input = layer.forward(gpu, &input, true)?;
        }
        Ok(input)
    }

    fn update_neuron_sgd(
        weights_j: &mut [f32],
        bias_j: &mut f32,
        prev_acts: &[f32],
        delta_j: f32,
        lr: f32,
    ) {
        *bias_j -= lr * delta_j;
        for (w, &a_prev) in weights_j.iter_mut().zip(prev_acts.iter()) {
            *w -= lr * delta_j * a_prev;
        }
    }

    /// Returns `true` if the update was applied. Skips silently when the teacher word
    /// is not in the vocabulary captured at network init time.
    pub fn adjust_weights(&mut self, lr: f32, teacher_word: &str, vocab: &[String]) -> bool {
        let Some(layer_deltas) = word_change_delta(teacher_word, vocab, self) else {
            return false;
        };
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let cols = layer.cols;
            for neuron in 0..layer.biases.len() {
                let n_delta = layer_deltas[layer_idx][neuron];
                let weights_slice = &mut layer.weights[neuron * cols..(neuron + 1) * cols];
                Network::update_neuron_sgd(
                    weights_slice,
                    &mut layer.biases[neuron],
                    &layer.last_input,
                    n_delta,
                    lr,
                );
            }
            layer.gpu_dirty = true;
        }
        true
    }
}

fn interpret_input(input: &str, memory: &[String], position: usize) -> Vec<f32> {
    let input_text: String = memory.join(" ") + " " + input;
    let chars_total = input_text.chars().count().max(1) as f32;

    let words: Vec<&str> = input_text.split_whitespace().collect();
    let word_count = words.len().max(1);

    let avg = words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count as f32;
    let f1 = (avg / 10.0).min(1.0);

    let f2 = (word_count as f32 / 50.0).min(1.0);

    let question_marks = input_text.chars().filter(|&c| c == '?').count();
    let f3 = (question_marks as f32 / 5.0).min(1.0);

    let vowels = "aeiou";
    let mut v = 0u32;
    let mut c = 0u32;
    for ch in input_text.to_lowercase().chars() {
        if vowels.contains(ch) {
            v += 1;
        } else if ch.is_alphabetic() {
            c += 1;
        }
    }
    let ratio = if c > 0 { v as f32 / c as f32 } else { 0.0 };
    let f4 = (ratio / 2.0).min(1.0);

    let mut score = 0u32;
    for ch in input_text.to_lowercase().chars() {
        match ch {
            'e' => score += 10,
            'a' => score += 9,
            'r' => score += 8,
            'i' => score += 7,
            'o' => score += 6,
            't' => score += 5,
            'n' => score += 4,
            's' => score += 3,
            _ => {}
        }
    }
    let f5 = (score as f32 / (chars_total * 10.0)).min(1.0);

    let uniq = input_text.chars().collect::<HashSet<char>>().len() as f32;
    let f6 = (uniq / chars_total).min(1.0);

    // Position is critical: without it, the network sees ~the same features at every
    // step of the generation loop and can't learn different teacher words per position.
    let f7 = (position as f32 / 100.0).min(1.0);

    vec![f1, f2, f3, f4, f5, f6, f7]
}

pub fn network_init(gpu: &Gpu, hidden_size: usize, output_size: usize) -> Result<Network> {
    let mut rng = rand::thread_rng();
    let mut layers = Vec::with_capacity(NUMBER_OF_HIDDEN_LAYERS + 1);

    layers.push(Layer::new(hidden_size, INPUT_SIZE, gpu, &mut rng)?);
    for _ in 1..NUMBER_OF_HIDDEN_LAYERS {
        layers.push(Layer::new(hidden_size, hidden_size, gpu, &mut rng)?);
    }
    layers.push(Layer::new(output_size, hidden_size, gpu, &mut rng)?);

    Ok(Network { layers })
}

fn interpret_output(activations: Vec<f32>, words: &[String]) -> String {
    let mut pairs: Vec<(f32, usize)> = activations
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.is_nan())
        .map(|(i, a)| (*a, i))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let choices = TOP_K_SAMPLE.min(pairs.len()).max(1);
    let pick = rand::thread_rng().gen_range(0..choices);
    words[pairs[pick].1].clone()
}

pub fn generate(
    gpu: &Gpu,
    net: &mut Network,
    start: &str,
    memory: &[String],
    words: &[String],
) -> Result<String> {
    let mut text = start.to_string();
    for i in 0..100 {
        let features = interpret_input(&text, memory, i);
        let activations = net.forward(gpu, features)?;
        let next_word = interpret_output(activations, words);
        text.push(' ');
        text.push_str(&next_word);
    }
    let len = start.len();
    Ok(text[len..].to_string())
}

pub fn generate_and_train(
    gpu: &Gpu,
    net: &mut Network,
    start: &str,
    memory: &[String],
    words: &[String],
    data: &Data,
    lr: f32,
) -> Result<String> {
    let mut text = start.to_string();
    let teacher: Vec<String> = teacher_response(data, memory, start)
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();

    for i in 0..100 {
        let features = interpret_input(&text, memory, i);
        let activations = net.forward_and_cache(gpu, features)?;
        let next_word = interpret_output(activations, words);

        match teacher.get(i) {
            Some(t) => {
                net.adjust_weights(lr, t, words);
            }
            None => {
                text.push(' ');
                text.push_str(&next_word);
                let len = start.len();
                return Ok(text[len..].to_string());
            }
        }
        text.push(' ');
        text.push_str(&next_word);
    }

    let len = start.len();
    Ok(text[len..].to_string())
}
