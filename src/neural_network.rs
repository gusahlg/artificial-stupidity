use crate::dialogs::{Data, Text};
use crate::embeddings::Embedding;
use crate::gpu::{Backend, Gpu, LayerGpu};
use crate::machine_learning::teacher_response;
use crate::teacher::compute_deltas;
use crate::tokenizer::{END_OF_BOT, PAD, UNK, tokenize};
use anyhow::Result;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

pub const EMBED_DIM: usize = 64;
pub const HIDDEN_SIZE: usize = 256;
pub const NUMBER_OF_HIDDEN_LAYERS: usize = 2;
pub const CONTEXT_WINDOW: usize = 8;
pub const POSITION_FEATURES: usize = 1;
pub const GRAD_CLIP: f32 = 5.0;
pub const MAX_GENERATION_LEN: usize = 40;
pub const TOP_K_SAMPLE: usize = 5;
// SGD without momentum: cross-entropy + softmax + dense output is very
// sensitive to compounding velocity on the output biases. Pure SGD with
// a small lr-decay converges steadily; momentum can be re-added if/when
// we move to a per-parameter normalized optimizer (Adam, RMSProp).
pub const MOMENTUM: f32 = 0.0;
pub const WEIGHT_DECAY: f32 = 1e-4;

pub fn input_size_for(embed_dim: usize, context: usize) -> usize {
    embed_dim * context + POSITION_FEATURES
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Activation {
    Tanh,
    Linear,
}

pub struct Layer {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub w_velocity: Vec<f32>,
    pub b_velocity: Vec<f32>,
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
            w_velocity: vec![0.0; rows * cols],
            b_velocity: vec![0.0; rows],
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
            w_velocity: vec![0.0; rows * cols],
            b_velocity: vec![0.0; rows],
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

    /// Grow the output dimension (rows) by re-randomizing new rows. Used when
    /// the vocab has expanded between runs. Re-allocates GPU buffers.
    pub fn extend_rows(
        &mut self,
        new_rows: usize,
        gpu: &Gpu,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Result<()> {
        if new_rows <= self.rows {
            return Ok(());
        }
        let extra = new_rows - self.rows;
        let bound = (6.0f32 / (new_rows + self.cols) as f32).sqrt();
        for _ in 0..(extra * self.cols) {
            self.weights.push(rng.gen_range(-bound..bound));
        }
        self.biases.resize(new_rows, 0.0);
        self.w_velocity.resize(self.weights.len(), 0.0);
        self.b_velocity.resize(new_rows, 0.0);
        self.matmul_out = vec![0.0; new_rows];
        self.rows = new_rows;
        self.layer_gpu = Self::alloc_gpu(self.rows, self.cols, &self.weights, gpu)?;
        Ok(())
    }
}

pub struct Network {
    pub embedding: Embedding,
    pub layers: Vec<Layer>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub hidden_layers: usize,
    pub embed_dim: usize,
    pub context_window: usize,
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

fn make_input(embed: Vec<f32>, position: usize) -> Vec<f32> {
    let mut v = embed;
    v.push((position as f32 / 100.0).min(1.0));
    v
}

impl Network {
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        token_ids: &[usize],
        position: usize,
    ) -> Result<Vec<f32>> {
        let embed = self.embedding.forward(token_ids, false);
        let mut input = make_input(embed, position);
        for layer in &mut self.layers {
            input = layer.forward(gpu, &input, false)?;
        }
        softmax_inplace(&mut input);
        Ok(input)
    }

    pub fn forward_and_cache(
        &mut self,
        gpu: &Gpu,
        token_ids: &[usize],
        position: usize,
    ) -> Result<Vec<f32>> {
        let embed = self.embedding.forward(token_ids, true);
        let mut input = make_input(embed, position);
        let n = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            input = layer.forward(gpu, &input, true)?;
            if i == n - 1 {
                softmax_inplace(&mut input);
                layer.last_activations.clone_from(&input);
            }
        }
        Ok(input)
    }

    pub fn train_step(&mut self, lr: f32, target_idx: usize) {
        let bp = compute_deltas(self, target_idx);
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let d = &bp.layer_deltas[idx];
            let cols = layer.cols;
            for j in 0..layer.rows {
                let g = d[j].clamp(-GRAD_CLIP, GRAD_CLIP);
                if g == 0.0 {
                    continue;
                }
                // Clip the velocity itself so momentum can't accumulate into
                // huge effective per-step updates over the course of an epoch.
                let bv = (MOMENTUM * layer.b_velocity[j] + g).clamp(-GRAD_CLIP, GRAD_CLIP);
                layer.b_velocity[j] = bv;
                let mut b = layer.biases[j];
                b -= lr * bv;
                b -= lr * WEIGHT_DECAY * b;
                layer.biases[j] = b;

                let row_base = j * cols;
                for k in 0..cols {
                    let x = layer.last_input[k];
                    if x == 0.0 {
                        continue;
                    }
                    let wg = g * x;
                    let vidx = row_base + k;
                    let v =
                        (MOMENTUM * layer.w_velocity[vidx] + wg).clamp(-GRAD_CLIP, GRAD_CLIP);
                    layer.w_velocity[vidx] = v;
                    let mut w = layer.weights[vidx];
                    w -= lr * v;
                    w -= lr * WEIGHT_DECAY * w;
                    layer.weights[vidx] = w;
                }
            }
            layer.gpu_dirty = true;
        }
        // Apply the input-gradient slice that corresponds to the embedding
        // section of the input vector (everything before the trailing position
        // scalar). Same lr/momentum/wd.
        let embed_grad_len = self.context_window * self.embed_dim;
        self.embedding
            .apply_grad(&bp.input_grad[..embed_grad_len], lr, MOMENTUM, WEIGHT_DECAY);
    }
}

pub struct VocabIndex<'a> {
    pub words: &'a [String],
    pub idx: HashMap<&'a str, usize>,
    pub pad_id: usize,
    pub unk_id: usize,
    pub eob_id: Option<usize>,
}

impl<'a> VocabIndex<'a> {
    pub fn new(words: &'a [String]) -> Self {
        let idx: HashMap<&str, usize> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();
        let pad_id = idx.get(PAD).copied().unwrap_or(0);
        let unk_id = idx.get(UNK).copied().unwrap_or(1);
        let eob_id = idx.get(END_OF_BOT).copied();
        Self {
            words,
            idx,
            pad_id,
            unk_id,
            eob_id,
        }
    }
    pub fn lookup(&self, w: &str) -> Option<usize> {
        self.idx.get(w).copied()
    }
    pub fn id_or_unk(&self, w: &str) -> usize {
        self.lookup(w).unwrap_or(self.unk_id)
    }
    pub fn ids_or_unk(&self, tokens: &[String]) -> Vec<usize> {
        tokens.iter().map(|t| self.id_or_unk(t)).collect()
    }
}

/// Build the fixed-width token-id window for the network input. Older tokens
/// are dropped, newest token sits at the rightmost slot, missing prefix slots
/// are filled with PAD so position 0 still has a valid input.
pub fn build_token_window(context: &[String], vocab: &VocabIndex<'_>, window: usize) -> Vec<usize> {
    let mut ids = vec![vocab.pad_id; window];
    if context.is_empty() {
        return ids;
    }
    let start = context.len().saturating_sub(window);
    let slice = &context[start..];
    let dst_start = window - slice.len();
    for (i, tok) in slice.iter().enumerate() {
        ids[dst_start + i] = vocab.id_or_unk(tok);
    }
    ids
}

pub fn network_init(
    gpu: &Gpu,
    embed_dim: usize,
    context_window: usize,
    hidden_size: usize,
    hidden_layers: usize,
    vocab_size: usize,
) -> Result<Network> {
    let mut rng = rand::thread_rng();
    let embedding = Embedding::new(vocab_size, embed_dim, &mut rng);
    let input_size = input_size_for(embed_dim, context_window);

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
        vocab_size,
        hidden_size,
        Activation::Linear,
        gpu,
        &mut rng,
    )?);

    Ok(Network {
        embedding,
        layers,
        vocab_size,
        hidden_size,
        hidden_layers,
        embed_dim,
        context_window,
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

fn detokenize(tokens: &[String]) -> String {
    let mut out = String::new();
    for tok in tokens {
        let is_punct = tok.chars().all(|c| !c.is_alphanumeric() && c != '\'');
        if out.is_empty() {
            out.push_str(tok);
        } else if is_punct {
            out.push_str(tok);
        } else {
            out.push(' ');
            out.push_str(tok);
        }
    }
    out
}

pub fn generate(
    gpu: &Gpu,
    net: &mut Network,
    start: &str,
    memory: &[String],
    words: &[String],
) -> Result<String> {
    let vocab = VocabIndex::new(words);
    let mut context: Vec<String> = Vec::new();
    for m in memory {
        context.extend(tokenize(m));
    }
    context.extend(tokenize(start));

    let mut produced: Vec<String> = Vec::new();
    for i in 0..MAX_GENERATION_LEN {
        let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);
        let probs = net.forward(gpu, &window, i)?;
        let idx = sample_top_k(&probs, TOP_K_SAMPLE);
        let next = &words[idx];
        if next == END_OF_BOT {
            break;
        }
        context.push(next.clone());
        produced.push(next.clone());
    }
    Ok(detokenize(&produced))
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
    // Tokenize once.
    let user_tokens = tokenize(start);
    let mut memory_tokens: Vec<String> = Vec::new();
    for m in memory {
        memory_tokens.extend(tokenize(m));
    }
    let mut scoring_memory = memory_tokens.clone();
    scoring_memory.extend(user_tokens.iter().cloned());

    // Pull teacher response with embedding-based similarity. The closure
    // borrows the embedding immutably; that borrow ends before we mutate the
    // network below.
    let teacher: Vec<String> = {
        let vocab = VocabIndex::new(words);
        teacher_response(data, &scoring_memory, &user_tokens, |toks| {
            let ids = vocab.ids_or_unk(toks);
            net.embedding.centroid(&ids)
        })
    };

    let vocab = VocabIndex::new(words);

    let mut context: Vec<String> = memory_tokens;
    context.extend(user_tokens.iter().cloned());
    let mut produced: Vec<String> = Vec::new();

    // Teacher list + sentinel </BOT> so the model also learns to stop.
    let mut targets: Vec<&str> = teacher.iter().map(|s| s.as_str()).collect();
    if !targets.is_empty() {
        targets.push(END_OF_BOT);
    }

    for i in 0..MAX_GENERATION_LEN {
        let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);

        match targets.get(i).copied() {
            Some(t) => {
                let _probs = net.forward_and_cache(gpu, &window, i)?;
                if let Some(target_id) = vocab.lookup(t) {
                    net.train_step(lr, target_id);
                }
                if t == END_OF_BOT {
                    break;
                }
                context.push(t.to_string());
                produced.push(t.to_string());
            }
            None => {
                let probs = net.forward(gpu, &window, i)?;
                let idx = sample_top_k(&probs, TOP_K_SAMPLE);
                let next = &words[idx];
                if next == END_OF_BOT {
                    break;
                }
                context.push(next.clone());
                produced.push(next.clone());
            }
        }
    }

    Ok(detokenize(&produced))
}

#[derive(Clone)]
pub struct TrainPair {
    pub prelude: Vec<String>,
    pub user: Vec<String>,
    pub bot: Vec<String>,
}

pub fn extract_train_pairs(data: &Data) -> Vec<TrainPair> {
    let mut pairs = Vec::new();
    for section in &data.sections {
        let mut prelude_tokens: Vec<String> = Vec::new();
        let mut idx = 0;
        while idx + 1 < section.len() {
            let user = match &section[idx] {
                Text::User(t) => t.clone(),
                _ => {
                    idx += 1;
                    continue;
                }
            };
            let bot = match &section[idx + 1] {
                Text::Bot(t) => t.clone(),
                _ => {
                    idx += 1;
                    continue;
                }
            };
            pairs.push(TrainPair {
                prelude: prelude_tokens.clone(),
                user: user.clone(),
                bot: bot.clone(),
            });
            prelude_tokens.extend(user);
            prelude_tokens.extend(bot);
            idx += 2;
        }
    }
    pairs
}

pub struct EpochStats {
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_targets: u64,
    pub val_targets: u64,
}

/// One supervised epoch: shuffles `train_pairs`, runs SGD on each bot token
/// (with </BOT> appended as a stop target), then evaluates val_pairs in
/// forward-only mode. `prelude_drop_prob` randomly hides the in-section
/// prelude during training so the model also learns cold-start responses.
pub fn train_one_epoch(
    gpu: &Gpu,
    net: &mut Network,
    train_pairs: &mut [TrainPair],
    val_pairs: &[TrainPair],
    words: &[String],
    lr: f32,
    prelude_drop_prob: f32,
) -> Result<EpochStats> {
    let vocab = VocabIndex::new(words);
    let mut rng = rand::thread_rng();
    train_pairs.shuffle(&mut rng);

    let mut train_loss = 0.0f64;
    let mut train_targets = 0u64;
    for pair in train_pairs.iter() {
        let drop: f32 = rng.gen_range(0.0..1.0);
        let prelude: &[String] = if drop < prelude_drop_prob {
            &[]
        } else {
            &pair.prelude
        };
        let mut context: Vec<String> = prelude.to_vec();
        context.extend(pair.user.iter().cloned());

        let mut targets: Vec<&str> = pair.bot.iter().map(|s| s.as_str()).collect();
        targets.push(END_OF_BOT);

        for (i, target_word) in targets.iter().enumerate() {
            let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);
            let probs = net.forward_and_cache(gpu, &window, i)?;
            if let Some(target_id) = vocab.lookup(target_word) {
                let p = probs[target_id].max(1e-9);
                train_loss += -(p as f64).ln();
                train_targets += 1;
                net.train_step(lr, target_id);
            }
            if *target_word == END_OF_BOT {
                break;
            }
            context.push(target_word.to_string());
        }
    }

    let mut val_loss = 0.0f64;
    let mut val_targets = 0u64;
    for pair in val_pairs {
        let mut context: Vec<String> = pair.prelude.clone();
        context.extend(pair.user.iter().cloned());
        let mut targets: Vec<&str> = pair.bot.iter().map(|s| s.as_str()).collect();
        targets.push(END_OF_BOT);
        for (i, target_word) in targets.iter().enumerate() {
            let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);
            let probs = net.forward(gpu, &window, i)?;
            if let Some(target_id) = vocab.lookup(target_word) {
                let p = probs[target_id].max(1e-9);
                val_loss += -(p as f64).ln();
                val_targets += 1;
            }
            if *target_word == END_OF_BOT {
                break;
            }
            context.push(target_word.to_string());
        }
    }

    Ok(EpochStats {
        train_loss: if train_targets > 0 {
            train_loss / train_targets as f64
        } else {
            0.0
        },
        val_loss: if val_targets > 0 {
            val_loss / val_targets as f64
        } else {
            0.0
        },
        train_targets,
        val_targets,
    })
}

pub fn pretrain(
    gpu: &Gpu,
    net: &mut Network,
    data: &Data,
    words: &[String],
    lr: f32,
    epochs: usize,
) -> Result<()> {
    let mut pairs = extract_train_pairs(data);
    let val_n = (pairs.len() / 10).max(1).min(pairs.len().saturating_sub(1));
    let val_pairs: Vec<TrainPair> = pairs.split_off(pairs.len() - val_n);
    for epoch in 0..epochs {
        let stats = train_one_epoch(gpu, net, &mut pairs, &val_pairs, words, lr, 0.0)?;
        println!(
            "  epoch {}/{}: train xent={:.4} val xent={:.4} ({} / {} targets)",
            epoch + 1,
            epochs,
            stats.train_loss,
            stats.val_loss,
            stats.train_targets,
            stats.val_targets,
        );
    }
    Ok(())
}

// re-export cosine for callers that wire teacher_response themselves
pub use crate::embeddings::cosine as embedding_cosine;
