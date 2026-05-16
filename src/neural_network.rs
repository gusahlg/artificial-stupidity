use crate::dialogs::{Data, Text};
use crate::gpu::{Backend, Gpu, LayerGpu};
use crate::machine_learning::teacher_response;
use crate::teacher::compute_deltas;
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

pub const HIDDEN_SIZE: usize = 256;
pub const NUMBER_OF_HIDDEN_LAYERS: usize = 2;
pub const CONTEXT_WINDOW: usize = 8;
pub const EXTRA_FEATURES: usize = 7;
pub const GRAD_CLIP: f32 = 5.0;
pub const MAX_GENERATION_LEN: usize = 40;
pub const TOP_K_SAMPLE: usize = 5;
pub const END_OF_BOT: &str = "</BOT>";

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Activation {
    Tanh,
    Linear,
}

pub struct Layer {
    pub weights: Vec<f32>, // row-major [rows * cols]
    pub biases: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub activation: Activation,
    pub last_input: Vec<f32>,
    pub last_activations: Vec<f32>,

    layer_gpu: Option<LayerGpu>,
    matmul_out: Vec<f32>,
    gpu_dirty: bool,
}

impl Layer {
    pub fn new(
        rows: usize,
        cols: usize,
        activation: Activation,
        gpu: &Gpu,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Result<Self> {
        let bound = (6.0f32 / (rows + cols) as f32).sqrt();
        let mut weights = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            weights.push(rng.gen_range(-bound..bound));
        }
        let biases = vec![0.0f32; rows];
        let layer_gpu = Self::alloc_gpu(rows, cols, &weights, gpu)?;
        Ok(Self {
            weights,
            biases,
            rows,
            cols,
            activation,
            last_input: Vec::new(),
            last_activations: Vec::new(),
            layer_gpu,
            matmul_out: vec![0.0; rows],
            gpu_dirty: false,
        })
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        activation: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
        gpu: &Gpu,
    ) -> Result<Self> {
        let layer_gpu = Self::alloc_gpu(rows, cols, &weights, gpu)?;
        Ok(Self {
            weights,
            biases,
            rows,
            cols,
            activation,
            last_input: Vec::new(),
            last_activations: Vec::new(),
            layer_gpu,
            matmul_out: vec![0.0; rows],
            gpu_dirty: false,
        })
    }

    fn alloc_gpu(
        rows: usize,
        cols: usize,
        weights: &[f32],
        gpu: &Gpu,
    ) -> Result<Option<LayerGpu>> {
        match &gpu.backend {
            Backend::Vulkan(v) => {
                let lg = LayerGpu::new(v, rows, cols)?;
                v.exec.upload(weights, &lg.gpu_weights)?;
                Ok(Some(lg))
            }
            Backend::Cpu => Ok(None),
        }
    }

    fn forward(&mut self, gpu: &Gpu, input: &[f32], cache: bool) -> Result<Vec<f32>> {
        debug_assert_eq!(input.len(), self.cols);
        gpu.matvec(
            &self.weights,
            self.rows,
            self.cols,
            input,
            &mut self.matmul_out,
            self.layer_gpu.as_ref(),
            self.gpu_dirty,
        )?;
        self.gpu_dirty = false;

        let mut out = Vec::with_capacity(self.rows);
        for j in 0..self.rows {
            let z = self.matmul_out[j] + self.biases[j];
            let a = match self.activation {
                Activation::Tanh => z.tanh(),
                Activation::Linear => z,
            };
            out.push(a);
        }

        if cache {
            self.last_input.clear();
            self.last_input.extend_from_slice(input);
            self.last_activations.clone_from(&out);
        }
        Ok(out)
    }
}

pub struct Network {
    pub layers: Vec<Layer>,
    pub input_size: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_layers: usize,
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let n = x.len() as f32;
        for v in x.iter_mut() {
            *v = 1.0 / n;
        }
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv;
        }
    } else {
        let n = x.len() as f32;
        for v in x.iter_mut() {
            *v = 1.0 / n;
        }
    }
}

impl Network {
    pub fn forward(&mut self, gpu: &Gpu, mut input: Vec<f32>) -> Result<Vec<f32>> {
        for layer in &mut self.layers {
            input = layer.forward(gpu, &input, false)?;
        }
        softmax_inplace(&mut input);
        Ok(input)
    }

    pub fn forward_and_cache(&mut self, gpu: &Gpu, mut input: Vec<f32>) -> Result<Vec<f32>> {
        let n = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            input = layer.forward(gpu, &input, true)?;
            if i == n - 1 {
                softmax_inplace(&mut input);
                // Replace cached pre-softmax logits with softmax probs so the
                // cross-entropy gradient collapses to (p - one_hot).
                layer.last_activations.clone_from(&input);
            }
        }
        Ok(input)
    }

    /// Single SGD update from one cached forward pass. Sparse-aware: skips
    /// weight updates for input entries that are exactly zero, which makes the
    /// input-layer update ~400x cheaper for our BoW input.
    pub fn train_step(&mut self, lr: f32, target_idx: usize) {
        let deltas = compute_deltas(self, target_idx);
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let d = &deltas[idx];
            let cols = layer.cols;
            // Pre-collect non-zero input columns so we don't probe `x == 0` in
            // the inner loop for every row.
            let nz_inputs: Vec<(usize, f32)> = layer
                .last_input
                .iter()
                .enumerate()
                .filter_map(|(k, &v)| if v != 0.0 { Some((k, v)) } else { None })
                .collect();
            for j in 0..layer.rows {
                let g = d[j].clamp(-GRAD_CLIP, GRAD_CLIP);
                if g == 0.0 {
                    continue;
                }
                layer.biases[j] -= lr * g;
                let lr_g = lr * g;
                let row_base = j * cols;
                for &(k, v) in &nz_inputs {
                    layer.weights[row_base + k] -= lr_g * v;
                }
            }
            layer.gpu_dirty = true;
        }
    }
}

pub struct VocabIndex<'a> {
    pub words: &'a [String],
    pub idx: HashMap<&'a str, usize>,
}

impl<'a> VocabIndex<'a> {
    pub fn new(words: &'a [String]) -> Self {
        let idx = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();
        Self { words, idx }
    }
    pub fn lookup(&self, w: &str) -> Option<usize> {
        self.idx.get(w).copied()
    }
}

fn handcrafted_features(text: &str, position: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), EXTRA_FEATURES);
    let chars_total = text.chars().count().max(1) as f32;
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len().max(1);

    let avg = words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count as f32;
    out[0] = (avg / 10.0).min(1.0);
    out[1] = (word_count as f32 / 50.0).min(1.0);
    let qm = text.chars().filter(|&c| c == '?').count();
    out[2] = (qm as f32 / 5.0).min(1.0);

    let vowels = "aeiou";
    let (mut v, mut c) = (0u32, 0u32);
    for ch in text.to_lowercase().chars() {
        if vowels.contains(ch) {
            v += 1;
        } else if ch.is_alphabetic() {
            c += 1;
        }
    }
    let ratio = if c > 0 { v as f32 / c as f32 } else { 0.0 };
    out[3] = (ratio / 2.0).min(1.0);

    let mut score = 0u32;
    for ch in text.to_lowercase().chars() {
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
    out[4] = (score as f32 / (chars_total * 10.0)).min(1.0);
    let uniq = text
        .chars()
        .collect::<std::collections::HashSet<char>>()
        .len() as f32;
    out[5] = (uniq / chars_total).min(1.0);
    out[6] = (position as f32 / 100.0).min(1.0);
}

/// Build the network input as: [bag-of-words over the last CONTEXT_WINDOW tokens
/// || EXTRA_FEATURES hand-crafted scalars]. The BoW part is what actually carries
/// word-level context; the scalars add a bit of structural prior.
pub fn build_input(text: &str, position: usize, vocab: &VocabIndex<'_>) -> Vec<f32> {
    let v = vocab.words.len();
    let mut features = vec![0.0f32; v + EXTRA_FEATURES];
    let words: Vec<&str> = text.split_whitespace().collect();
    let start = words.len().saturating_sub(CONTEXT_WINDOW);
    let mut counted = 0u32;
    for w in &words[start..] {
        if let Some(idx) = vocab.lookup(w) {
            features[idx] += 1.0;
            counted += 1;
        }
    }
    if counted > 0 {
        let inv = 1.0 / counted as f32;
        for x in features[..v].iter_mut() {
            *x *= inv;
        }
    }
    handcrafted_features(text, position, &mut features[v..]);
    features
}

pub fn input_size_for(vocab_len: usize) -> usize {
    vocab_len + EXTRA_FEATURES
}

pub fn network_init(
    gpu: &Gpu,
    input_size: usize,
    hidden_size: usize,
    hidden_layers: usize,
    output_size: usize,
) -> Result<Network> {
    let mut rng = rand::thread_rng();
    let mut layers = Vec::with_capacity(hidden_layers + 1);
    layers.push(Layer::new(hidden_size, input_size, Activation::Tanh, gpu, &mut rng)?);
    for _ in 1..hidden_layers {
        layers.push(Layer::new(
            hidden_size,
            hidden_size,
            Activation::Tanh,
            gpu,
            &mut rng,
        )?);
    }
    layers.push(Layer::new(
        output_size,
        hidden_size,
        Activation::Linear,
        gpu,
        &mut rng,
    )?);
    Ok(Network {
        layers,
        input_size,
        vocab_size: output_size,
        hidden_size,
        hidden_layers,
    })
}

fn sample_top_k(probs: &[f32], k: usize) -> usize {
    let mut pairs: Vec<(f32, usize)> = probs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_finite())
        .map(|(i, p)| (*p, i))
        .collect();
    if pairs.is_empty() {
        return 0;
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.min(pairs.len()).max(1);
    let top = &pairs[..k];
    let sum: f32 = top.iter().map(|(p, _)| *p).sum();
    if sum <= 0.0 {
        return top[0].1;
    }
    let mut r = rand::thread_rng().gen_range(0.0..sum);
    for (p, i) in top {
        r -= *p;
        if r <= 0.0 {
            return *i;
        }
    }
    top.last().unwrap().1
}

pub fn generate(
    gpu: &Gpu,
    net: &mut Network,
    start: &str,
    memory: &[String],
    words: &[String],
) -> Result<String> {
    let vocab = VocabIndex::new(words);
    let prelude = memory.join(" ");
    let mut text = if prelude.is_empty() {
        start.to_string()
    } else {
        format!("{} {}", prelude, start)
    };
    let prefix_len = text.len();

    for i in 0..MAX_GENERATION_LEN {
        let features = build_input(&text, i, &vocab);
        let probs = net.forward(gpu, features)?;
        let idx = sample_top_k(&probs, TOP_K_SAMPLE);
        let next_word = &words[idx];
        if next_word == END_OF_BOT || next_word == "<USER>" || next_word == "<SEC>" {
            break;
        }
        text.push(' ');
        text.push_str(next_word);
    }
    Ok(text[prefix_len..].trim().to_string())
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
    let vocab = VocabIndex::new(words);
    let mut scoring_memory: Vec<String> = memory.to_vec();
    scoring_memory.push(start.to_string());
    let teacher_sentence = teacher_response(data, &scoring_memory, start);
    let mut teacher: Vec<String> = teacher_sentence
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();
    if !teacher.is_empty() {
        teacher.push(END_OF_BOT.to_string());
    }

    let prelude = memory.join(" ");
    let mut text = if prelude.is_empty() {
        start.to_string()
    } else {
        format!("{} {}", prelude, start)
    };
    let prefix_len = text.len();
    let mut produced = String::new();

    for i in 0..MAX_GENERATION_LEN {
        match teacher.get(i) {
            Some(t) => {
                let features = build_input(&text, i, &vocab);
                let _probs = net.forward_and_cache(gpu, features)?;
                if let Some(target) = vocab.lookup(t) {
                    net.train_step(lr, target);
                }
                if t == END_OF_BOT {
                    break;
                }
                text.push(' ');
                text.push_str(t);
                produced.push(' ');
                produced.push_str(t);
            }
            None => {
                // No teacher word for this position — pure inference.
                let features = build_input(&text, i, &vocab);
                let probs = net.forward(gpu, features)?;
                let idx = sample_top_k(&probs, TOP_K_SAMPLE);
                let next_word = &words[idx];
                if next_word == END_OF_BOT || next_word == "<USER>" || next_word == "<SEC>" {
                    break;
                }
                text.push(' ');
                text.push_str(next_word);
                produced.push(' ');
                produced.push_str(next_word);
            }
        }
    }

    if produced.is_empty() {
        return Ok(text[prefix_len..].trim().to_string());
    }
    Ok(produced.trim().to_string())
}

/// A single supervised training example: user prompt + (already-built) prelude
/// context + bot target sequence.
#[derive(Clone)]
pub struct TrainPair {
    pub prelude: String,
    pub user: String,
    pub bot: String,
}

pub fn extract_train_pairs(data: &Data) -> Vec<TrainPair> {
    let mut pairs = Vec::new();
    for section in &data.Sections {
        let mut prelude_words: Vec<String> = Vec::new();
        let mut idx = 0;
        while idx + 1 < section.len() {
            let user = match &section[idx] {
                Text::User(s) => s.clone(),
                _ => {
                    idx += 1;
                    continue;
                }
            };
            let bot = match &section[idx + 1] {
                Text::Bot(s) => s.clone(),
                _ => {
                    idx += 1;
                    continue;
                }
            };
            pairs.push(TrainPair {
                prelude: prelude_words.join(" "),
                user: user.clone(),
                bot: bot.clone(),
            });
            prelude_words.push(user);
            prelude_words.push(bot);
            idx += 2;
        }
    }
    pairs
}

pub struct EpochStats {
    pub avg_loss: f64,
    pub targets: u64,
}

/// Run one supervised epoch over `pairs`. Shuffles every epoch; randomly drops
/// the in-section prelude with probability `prelude_drop_prob` so the model
/// also learns to answer cold-start prompts.
pub fn train_one_epoch(
    gpu: &Gpu,
    net: &mut Network,
    pairs: &mut [TrainPair],
    words: &[String],
    lr: f32,
    prelude_drop_prob: f32,
) -> Result<EpochStats> {
    let vocab = VocabIndex::new(words);
    let mut rng = rand::thread_rng();
    pairs.shuffle(&mut rng);

    let mut total_loss = 0.0f64;
    let mut count = 0u64;

    for pair in pairs.iter() {
        let drop_prelude: f32 = rng.gen_range(0.0..1.0);
        let drop_prelude = drop_prelude < prelude_drop_prob;
        let prelude = if drop_prelude { "" } else { pair.prelude.as_str() };
        let mut running = if prelude.is_empty() {
            pair.user.clone()
        } else {
            format!("{} {}", prelude, pair.user)
        };

        let bot_tokens: Vec<&str> = pair.bot.split_whitespace().collect();
        for (i, target_word) in bot_tokens.iter().chain(std::iter::once(&END_OF_BOT)).enumerate() {
            let features = build_input(&running, i, &vocab);
            let probs = net.forward_and_cache(gpu, features)?;
            if let Some(target) = vocab.lookup(target_word) {
                let p = probs[target].max(1e-9);
                total_loss += -(p as f64).ln();
                count += 1;
                net.train_step(lr, target);
            }
            if *target_word == END_OF_BOT {
                break;
            }
            running.push(' ');
            running.push_str(target_word);
        }
    }

    let avg = if count > 0 {
        total_loss / count as f64
    } else {
        0.0
    };
    Ok(EpochStats {
        avg_loss: avg,
        targets: count,
    })
}

/// Convenience wrapper used at first-run init: a handful of clean epochs over
/// the corpus with no data augmentation. The long-running training program is
/// `train_one_epoch` driven from `src/bin/train.rs`.
pub fn pretrain(
    gpu: &Gpu,
    net: &mut Network,
    data: &Data,
    words: &[String],
    lr: f32,
    epochs: usize,
) -> Result<()> {
    let mut pairs = extract_train_pairs(data);
    for epoch in 0..epochs {
        let stats = train_one_epoch(gpu, net, &mut pairs, words, lr, 0.0)?;
        println!(
            "  epoch {}/{}: avg cross-entropy = {:.4} over {} targets",
            epoch + 1,
            epochs,
            stats.avg_loss,
            stats.targets
        );
    }
    Ok(())
}
