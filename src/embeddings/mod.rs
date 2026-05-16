//! Trainable word embedding table.
//!
//! `weights` is a `vocab_size × embed_dim` row-major matrix. `forward`
//! concatenates the rows for a given token-ID window into one flat vector,
//! which feeds the dense network. `apply_grad` updates only the rows that
//! were actually used in the last forward, with SGD + momentum + L2.

use rand::Rng;
use std::collections::HashMap;

/// Token IDs whose embedding rows are frozen (never updated). PAD=0 is a
/// padding sentinel — letting it drift makes the input layer's "no history
/// yet" signal noisy and tends to destabilize early training.
const FROZEN_IDS: &[usize] = &[0];

pub struct Embedding {
    pub weights: Vec<f32>,
    pub velocities: Vec<f32>,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub last_token_ids: Vec<usize>,
}

impl Embedding {
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Self {
        let n = vocab_size * embed_dim;
        let bound = (1.0f32 / embed_dim as f32).sqrt();
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(rng.gen_range(-bound..bound));
        }
        for &id in FROZEN_IDS {
            let start = id * embed_dim;
            for k in 0..embed_dim {
                weights[start + k] = 0.0;
            }
        }
        Self {
            velocities: vec![0.0; n],
            weights,
            vocab_size,
            embed_dim,
            last_token_ids: Vec::new(),
        }
    }

    pub fn from_parts(vocab_size: usize, embed_dim: usize, weights: Vec<f32>) -> Self {
        let n = weights.len();
        Self {
            velocities: vec![0.0; n],
            weights,
            vocab_size,
            embed_dim,
            last_token_ids: Vec::new(),
        }
    }

    pub fn lookup(&self, id: usize) -> &[f32] {
        let start = id * self.embed_dim;
        &self.weights[start..start + self.embed_dim]
    }

    /// Concatenate the rows for `token_ids` into one flat vector of length
    /// `token_ids.len() * embed_dim`. When `cache` is true, the IDs are
    /// stored so the backward pass can attribute the gradient to the right
    /// rows.
    pub fn forward(&mut self, token_ids: &[usize], cache: bool) -> Vec<f32> {
        let d = self.embed_dim;
        let mut out = Vec::with_capacity(token_ids.len() * d);
        for &id in token_ids {
            let id = if id < self.vocab_size { id } else { 1 };
            let start = id * d;
            out.extend_from_slice(&self.weights[start..start + d]);
        }
        if cache {
            self.last_token_ids = token_ids.to_vec();
        }
        out
    }

    /// `grad` is laid out exactly like the output of `forward`: one
    /// `embed_dim`-wide block per token. When the same id appears multiple
    /// times in the window (typical for PAD at the start of a sequence), we
    /// SUM the per-occurrence gradients into a single update for that row —
    /// applying them as separate momentum updates would multiply the effect
    /// by the number of occurrences and blow the row up.
    pub fn apply_grad(&mut self, grad: &[f32], lr: f32, momentum: f32, weight_decay: f32) {
        let d = self.embed_dim;
        debug_assert_eq!(grad.len(), self.last_token_ids.len() * d);

        let mut accum: HashMap<usize, Vec<f32>> = HashMap::new();
        for (i, &id) in self.last_token_ids.iter().enumerate() {
            if id >= self.vocab_size || FROZEN_IDS.contains(&id) {
                continue;
            }
            let src = i * d;
            let entry = accum.entry(id).or_insert_with(|| vec![0.0; d]);
            for k in 0..d {
                entry[k] += grad[src + k];
            }
        }
        for (id, g) in accum {
            let dst = id * d;
            for k in 0..d {
                let v = (momentum * self.velocities[dst + k] + g[k]).clamp(-5.0, 5.0);
                self.velocities[dst + k] = v;
                let mut w = self.weights[dst + k];
                w -= lr * v;
                w -= lr * weight_decay * w;
                self.weights[dst + k] = w;
            }
        }
    }

    /// Append fresh random rows so the table covers `new_vocab_size` tokens.
    /// No-op if it's already big enough; never shrinks.
    pub fn extend_to(&mut self, new_vocab_size: usize, rng: &mut rand::rngs::ThreadRng) {
        if new_vocab_size <= self.vocab_size {
            return;
        }
        let bound = (1.0f32 / self.embed_dim as f32).sqrt();
        let extra = (new_vocab_size - self.vocab_size) * self.embed_dim;
        self.weights.reserve(extra);
        for _ in 0..extra {
            self.weights.push(rng.gen_range(-bound..bound));
        }
        self.velocities.resize(self.velocities.len() + extra, 0.0);
        self.vocab_size = new_vocab_size;
    }

    /// Mean-pool embeddings for the given token IDs, skipping PAD (0) and
    /// UNK (1) so empty / unknown queries don't drown out signal. Returns a
    /// zero vector if nothing matched.
    pub fn centroid(&self, token_ids: &[usize]) -> Vec<f32> {
        let d = self.embed_dim;
        let mut out = vec![0.0f32; d];
        let mut n = 0u32;
        for &id in token_ids {
            if id <= 1 || id >= self.vocab_size {
                continue;
            }
            let start = id * d;
            for k in 0..d {
                out[k] += self.weights[start + k];
            }
            n += 1;
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            for x in out.iter_mut() {
                *x *= inv;
            }
        }
        out
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}
