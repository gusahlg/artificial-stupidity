//! Trainable word embedding table with Adam optimizer state.

use rand::Rng;
use std::collections::HashMap;

const FROZEN_IDS: &[usize] = &[0]; // PAD frozen

pub struct Embedding {
    pub weights: Vec<f32>,
    pub m: Vec<f32>, // Adam first moment
    pub v: Vec<f32>, // Adam second moment
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub last_token_ids: Vec<usize>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize, rng: &mut rand::rngs::ThreadRng) -> Self {
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
            m: vec![0.0; n],
            v: vec![0.0; n],
            weights,
            vocab_size,
            embed_dim,
            last_token_ids: Vec::new(),
        }
    }

    pub fn from_parts(vocab_size: usize, embed_dim: usize, weights: Vec<f32>) -> Self {
        let n = weights.len();
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            weights,
            vocab_size,
            embed_dim,
            last_token_ids: Vec::new(),
        }
    }

    /// Reconstruct from persisted weights AND Adam moments. Used by `persist::load`
    /// on v3 model files so a resumed training run keeps its momentum/variance
    /// estimates and skips the Adam warmup penalty.
    pub fn from_parts_with_adam(
        vocab_size: usize,
        embed_dim: usize,
        weights: Vec<f32>,
        m: Vec<f32>,
        v: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(weights.len(), vocab_size * embed_dim);
        debug_assert_eq!(m.len(), weights.len());
        debug_assert_eq!(v.len(), weights.len());
        Self {
            m,
            v,
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

    /// AdamW update for the (small set of) embedding rows that were touched
    /// during the last forward. Duplicates in the window have their gradient
    /// contributions summed into a single row update.
    pub fn apply_grad_adam(
        &mut self,
        grad: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u64,
    ) {
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

        let bc1 = 1.0 - beta1.powi(step as i32);
        let bc2 = 1.0 - beta2.powi(step as i32);
        let bc1 = bc1.max(1e-12);
        let bc2 = bc2.max(1e-12);

        for (id, g) in accum {
            let dst = id * d;
            for k in 0..d {
                let gk = g[k];
                let m_new = beta1 * self.m[dst + k] + (1.0 - beta1) * gk;
                let v_new = beta2 * self.v[dst + k] + (1.0 - beta2) * gk * gk;
                self.m[dst + k] = m_new;
                self.v[dst + k] = v_new;
                let m_hat = m_new / bc1;
                let v_hat = v_new / bc2;
                let mut w = self.weights[dst + k];
                w -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * w);
                self.weights[dst + k] = w;
            }
        }
    }

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
        self.m.resize(self.m.len() + extra, 0.0);
        self.v.resize(self.v.len() + extra, 0.0);
        self.vocab_size = new_vocab_size;
    }

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
