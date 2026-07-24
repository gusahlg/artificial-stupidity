use crate::dialogs::{Data, Turn};
use crate::embeddings::Embedding;
use crate::gpu::{Backend, Gpu, LayerGpu};
use crate::machine_learning::teacher_response;
use crate::persons::{BOT_PERSON_ID, close_tag, open_tag};
use crate::tokenizer::{PAD, UNK, tokenize};
use anyhow::Result;
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;
use wide::f32x8;

/// SIMD lane count for the Adam inner k-loop. f32x8 corresponds to AVX2 on
/// x86_64 / NEON on aarch64. The compiler will pick the right backend
/// without us specifying.
const SIMD_LANES: usize = 8;

#[inline(always)]
fn load_f32x8(s: &[f32]) -> f32x8 {
    let arr: [f32; 8] = s[..8].try_into().expect("slice shorter than f32x8");
    f32x8::from(arr)
}

#[inline(always)]
fn store_f32x8(s: &mut [f32], v: f32x8) {
    let arr = v.to_array();
    s[..8].copy_from_slice(&arr);
}

/// Vocab token used as the bot's open tag, e.g. `<PERSON_0>`.
fn bot_open_tag() -> String {
    open_tag(BOT_PERSON_ID)
}
/// Vocab token used as the bot's close tag, e.g. `</PERSON_0>` — the stop
/// signal during generation and the trailing target during training.
fn bot_close_tag() -> String {
    close_tag(BOT_PERSON_ID)
}

/// Wrap a turn with its PERSON_N open/close tags inline.
fn wrap_turn(t: &Turn) -> Vec<String> {
    let mut out = Vec::with_capacity(t.tokens.len() + 2);
    out.push(open_tag(t.person_id));
    out.extend(t.tokens.iter().cloned());
    out.push(close_tag(t.person_id));
    out
}

pub const EMBED_DIM: usize = 256;
pub const HIDDEN_SIZE: usize = 768;
pub const NUMBER_OF_HIDDEN_LAYERS: usize = 4;
pub const CONTEXT_WINDOW: usize = 32;
pub const POSITION_FEATURES: usize = 1;
/// Inference-side repetition penalty. Each time `generate` samples a
/// token, the next forward pass's softmax probabilities for that
/// token (and the few before it, up to `REPETITION_WINDOW`) are
/// multiplied by this factor before `sample_top_k`. Values < 1.0
/// downweight already-emitted tokens; 1.0 disables the penalty.
/// 0.5 was found empirically to fix the "the. the- 1." mid-reply
/// repetition pattern observed in 2026-05-22 manual testing without
/// suppressing legitimately repeating function words ("a", "the",
/// "i", etc.) too aggressively, because the penalty only sees the
/// most recent few emissions.
pub const REPETITION_PENALTY: f32 = 0.5;
/// How many of the most recent generated tokens get the repetition
/// penalty applied. Keep small (~6) so common words like "the" /
/// "a" / "i" can still appear at a natural frequency.
pub const REPETITION_WINDOW: usize = 6;
/// Per-element gradient clamp. Kept as a final safety bound after the
/// global-norm scale below, so a single pathological weight can never
/// move by more than this. Most updates are well under this magnitude
/// once the global-norm rescale has fired.
pub const GRAD_CLIP: f32 = 5.0;
/// Global L2 gradient-norm clip threshold. Set to +infinity to make
/// the clip a no-op while still letting the diagnostic measurement
/// run.
///
/// **Why disabled.** The original idea was "clip-by-global-norm
/// preserves direction better than element-wise clamp," but the
/// implementation here clips the *delta* (post-activation gradient)
/// vectors, not the actual *weight* gradients. The relationship is
/// `dL/dW = delta_out ⊗ input_in`, so scaling delta also scales the
/// weight gradient — but the scale factor that would be ideal
/// differs per layer (it depends on `‖input_in‖`, which varies
/// across the 4 layers). A single global scale on deltas does not
/// match textbook "clip-by-global-norm" semantics.
///
/// Empirically: at fresh init `‖delta‖₂ ≈ 11`, but after a few
/// thousand training steps it grows to the **thousands**. A
/// "reasonable" textbook value (1.0, or even 50.0) fires on >80%
/// of steps and erases the gradient signal, regressing val loss.
/// The element-wise `GRAD_CLIP` clamp inside the Adam loop is
/// preserved as the outlier guard.
///
/// The diagnostic statistics (mean / clip-count) are still emitted
/// under `SIGHURT_TIME_STEPS=1` for future tuning experiments.
pub const GLOBAL_GRAD_NORM_CLIP: f32 = f32::INFINITY;
pub const MAX_GENERATION_LEN: usize = 40;
pub const TOP_K_SAMPLE: usize = 5;
/// Minimum number of content (non-tag) tokens a generated reply must
/// contain before the bot's close tag `</PERSON_0>` is allowed to be
/// sampled. The close-tag probability is zeroed until this many
/// tokens have been emitted. Kills the degenerate "." / "no."
/// one-token replies observed at low val loss — the model learns
/// "close early" as a safe loss-minimizer and this floor forces it
/// to commit to an actual reply. 4 matches the survey's (§5)
/// "at least 4–6 content tokens" recommendation, at the permissive
/// end so short-but-legitimate replies ("yes , why ?") still fit.
pub const MIN_REPLY_TOKENS: usize = 4;
/// LayerNorm variance epsilon. Standard 1e-5 (PyTorch default);
/// guards the 1/√σ² against near-zero variance early in training.
pub const LN_EPS: f32 = 1e-5;
/// Cap each training example's target sequence at this many tokens. Merged
/// Discord turns can be paragraph-length; without a cap, a single example
/// can drive hundreds of forward+backward steps and stall training. The
/// model only sees the LAST `CONTEXT_WINDOW` tokens of context per step
/// anyway, so the late tokens of a very long target add little learning
/// signal relative to their cost. 20 tokens is a generous reply length.
pub const MAX_TARGET_TOKENS: usize = 20;
// Adam hyperparameters. Adam handles the per-layer gradient-scale differences
// (output layer gradients dwarf hidden-layer gradients) that broke our plain
// SGD+momentum experiments.
pub const ADAM_BETA1: f32 = 0.9;
pub const ADAM_BETA2: f32 = 0.999;
pub const ADAM_EPS: f32 = 1e-8;
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
    /// Adam first moment for weights.
    pub w_m: Vec<f32>,
    /// Adam second moment for weights.
    pub w_v: Vec<f32>,
    pub b_m: Vec<f32>,
    pub b_v: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub activation: Activation,
    pub last_input: Vec<f32>,
    pub last_activations: Vec<f32>,
    /// Per-neuron dropout scale factor for the most recent
    /// training-time forward pass. Length matches `rows`. Values are
    /// either `0.0` (neuron was dropped) or `1.0/(1-p)` (kept, scaled
    /// to preserve activation expectation). Length 0 means "no
    /// dropout was applied to this forward" — backward then leaves
    /// gradients alone. Runtime-only; never serialized.
    pub last_dropout_mask: Vec<f32>,
    /// LayerNorm learnable gain γ. **Empty = LayerNorm disabled for
    /// this layer** — the pre-v5 behavior, and always the case for
    /// the output layer. When non-empty (length == `rows`), the
    /// forward becomes `z → LN(z) → activation`: the pre-activation
    /// vector is mean/variance-normalized to x̂ and the layer emits
    /// `activation(γ·x̂ + β)`. Persisted in v5 model files.
    pub ln_gain: Vec<f32>,
    /// LayerNorm learnable bias β. Same length/emptiness contract as
    /// `ln_gain`.
    pub ln_bias: Vec<f32>,
    /// Adam first/second moments for `ln_gain` / `ln_bias`.
    /// Persisted in v5 so resumed training keeps warm optimizer
    /// state, mirroring `w_m`/`w_v`/`b_m`/`b_v`.
    pub ln_gain_m: Vec<f32>,
    pub ln_gain_v: Vec<f32>,
    pub ln_bias_m: Vec<f32>,
    pub ln_bias_v: Vec<f32>,
    /// Cached normalized pre-activations x̂ from the most recent
    /// `cache = true` forward. The LN backward needs exactly x̂ and
    /// `last_ln_inv_std`; we cache them EXPLICITLY at the point they
    /// are computed rather than reconstructing them from
    /// `last_activations` — the 2026-05-20 dropout bug
    /// (improvement-survey §5b) came from deriving backward inputs
    /// from a post-transform cache, silently corrupting the tanh
    /// derivative. Runtime-only; never serialized.
    pub last_ln_xhat: Vec<f32>,
    /// Cached `1/√(σ² + LN_EPS)` matching `last_ln_xhat`.
    pub last_ln_inv_std: f32,

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
            w_m: vec![0.0; rows * cols],
            w_v: vec![0.0; rows * cols],
            b_m: vec![0.0; rows],
            b_v: vec![0.0; rows],
            weights,
            biases,
            rows,
            cols,
            activation,
            last_input: Vec::new(),
            last_activations: Vec::new(),
            last_dropout_mask: Vec::new(),
            ln_gain: Vec::new(),
            ln_bias: Vec::new(),
            ln_gain_m: Vec::new(),
            ln_gain_v: Vec::new(),
            ln_bias_m: Vec::new(),
            ln_bias_v: Vec::new(),
            last_ln_xhat: Vec::new(),
            last_ln_inv_std: 0.0,
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
            w_m: vec![0.0; rows * cols],
            w_v: vec![0.0; rows * cols],
            b_m: vec![0.0; rows],
            b_v: vec![0.0; rows],
            weights,
            biases,
            rows,
            cols,
            activation,
            last_input: Vec::new(),
            last_activations: Vec::new(),
            last_dropout_mask: Vec::new(),
            ln_gain: Vec::new(),
            ln_bias: Vec::new(),
            ln_gain_m: Vec::new(),
            ln_gain_v: Vec::new(),
            ln_bias_m: Vec::new(),
            ln_bias_v: Vec::new(),
            last_ln_xhat: Vec::new(),
            last_ln_inv_std: 0.0,
            layer_gpu,
            matmul_out: vec![0.0; rows],
            gpu_dirty: false,
        })
    }

    /// Reconstruct from persisted weights, biases AND Adam moments. Used by
    /// `persist::load` on v3 model files so the resumed run keeps its Adam
    /// momentum / second-moment state instead of zeroing it and paying the
    /// first-step warmup tax on every restart.
    pub fn from_parts_with_adam(
        rows: usize,
        cols: usize,
        activation: Activation,
        weights: Vec<f32>,
        biases: Vec<f32>,
        w_m: Vec<f32>,
        w_v: Vec<f32>,
        b_m: Vec<f32>,
        b_v: Vec<f32>,
        gpu: &Gpu,
    ) -> Result<Self> {
        debug_assert_eq!(weights.len(), rows * cols);
        debug_assert_eq!(biases.len(), rows);
        debug_assert_eq!(w_m.len(), rows * cols);
        debug_assert_eq!(w_v.len(), rows * cols);
        debug_assert_eq!(b_m.len(), rows);
        debug_assert_eq!(b_v.len(), rows);
        let layer_gpu = Self::alloc_gpu(rows, cols, &weights, gpu)?;
        Ok(Self {
            w_m,
            w_v,
            b_m,
            b_v,
            weights,
            biases,
            rows,
            cols,
            activation,
            last_input: Vec::new(),
            last_activations: Vec::new(),
            last_dropout_mask: Vec::new(),
            ln_gain: Vec::new(),
            ln_bias: Vec::new(),
            ln_gain_m: Vec::new(),
            ln_gain_v: Vec::new(),
            ln_bias_m: Vec::new(),
            ln_bias_v: Vec::new(),
            last_ln_xhat: Vec::new(),
            last_ln_inv_std: 0.0,
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

    /// Is LayerNorm enabled for this layer? Encoded as "non-empty
    /// gain vector" so v2/v3/v4 loads (which never populate the LN
    /// fields) automatically get the legacy no-LN forward.
    pub fn ln_active(&self) -> bool {
        !self.ln_gain.is_empty()
    }

    /// Populate LN parameters at identity (γ = 1, β = 0, moments
    /// zeroed) so the layer's function is initially unchanged apart
    /// from the normalization itself. Called by
    /// `Network::enable_layer_norm` at fresh init and usable by the
    /// persist loader tests.
    pub fn init_layer_norm(&mut self) {
        self.ln_gain = vec![1.0; self.rows];
        self.ln_bias = vec![0.0; self.rows];
        self.ln_gain_m = vec![0.0; self.rows];
        self.ln_gain_v = vec![0.0; self.rows];
        self.ln_bias_m = vec![0.0; self.rows];
        self.ln_bias_v = vec![0.0; self.rows];
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

        // Everything past the matvec (bias add, LayerNorm, activation)
        // runs on the CPU regardless of backend — `gpu.matvec` is a pure
        // W@x on both paths — so LN produces IDENTICAL math on Vulkan
        // and CPU and no backend forcing is needed. Keeping the non-LN
        // arm byte-for-byte the same arithmetic as before v5 also keeps
        // v2/v3/v4 model files forwarding bit-identically.
        let mut out = Vec::with_capacity(self.rows);
        if self.ln_active() {
            // z → LN(z) → activation. Compute the full pre-activation
            // vector first (LN needs the mean/variance across neurons),
            // then normalize, scale/shift, and activate.
            for j in 0..self.rows {
                out.push(self.matmul_out[j] + self.biases[j]);
            }
            let h = self.rows as f32;
            let mean = out.iter().sum::<f32>() / h;
            let var = out.iter().map(|z| (z - mean) * (z - mean)).sum::<f32>() / h;
            let inv_std = 1.0 / (var + LN_EPS).sqrt();
            if cache {
                self.last_ln_inv_std = inv_std;
                self.last_ln_xhat.clear();
                self.last_ln_xhat.reserve(self.rows);
            }
            for j in 0..self.rows {
                let xhat = (out[j] - mean) * inv_std;
                if cache {
                    // Cache x̂ EXPLICITLY here, at the point of
                    // computation — backward must never re-derive it
                    // from post-transform values (§5b postmortem).
                    self.last_ln_xhat.push(xhat);
                }
                let y = self.ln_gain[j] * xhat + self.ln_bias[j];
                out[j] = match self.activation {
                    Activation::Tanh => y.tanh(),
                    Activation::Linear => y,
                };
            }
        } else {
            for j in 0..self.rows {
                let z = self.matmul_out[j] + self.biases[j];
                let a = match self.activation {
                    Activation::Tanh => z.tanh(),
                    Activation::Linear => z,
                };
                out.push(a);
            }
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
        self.w_m.resize(self.weights.len(), 0.0);
        self.w_v.resize(self.weights.len(), 0.0);
        self.b_m.resize(new_rows, 0.0);
        self.b_v.resize(new_rows, 0.0);
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
    /// Adam time-step counter. Persisted in v3 model files so a resumed
    /// run keeps its bias-correction warm and skips the first-step tax.
    pub adam_step: u64,
    /// Inverted-dropout probability applied to hidden-layer
    /// activations during `forward_and_cache` (the training-time
    /// forward). 0.0 disables. `forward` (val / generation) ignores
    /// this and runs full-strength. **Persisted in v5 model files**: a
    /// resumed run without `--dropout` keeps the model's saved value
    /// (so a value baked in by one experiment carries forward until an
    /// explicit `--dropout 0.0`); v2/v3/v4 files load as 0.0.
    pub dropout_p: f32,
    /// Label smoothing factor α applied to the cross-entropy target
    /// distribution in `compute_deltas_into`. Replaces the one-hot
    /// target with `(1 - α) · one_hot + α · uniform`. 0.0 disables.
    /// Set via `--label-smoothing F`; persisted in v5 model files so
    /// resumed runs default to the value they were trained with.
    pub label_smoothing: f32,
    /// Master LayerNorm switch. When true, every HIDDEN layer carries
    /// LN gain/bias parameters (see `Layer::ln_gain`) and applies
    /// `z → LN(z) → tanh` in its forward. Structural — persisted in
    /// v5 model files. False for all v2/v3/v4 loads and for
    /// `network_init` (the trainer opts in via `enable_layer_norm()`
    /// on fresh init only), so the LIVE v4 production model keeps
    /// byte-identical forward math.
    pub layer_norm: bool,
    /// Reusable backward-pass buffers. Held on the network so we don't
    /// allocate per training step. `train_step` `mem::take`s this out so
    /// it can pass `&Network` + `&mut scratch` separately past the borrow
    /// checker, then puts it back when done.
    pub backprop_scratch: crate::teacher::BackpropScratch,
    /// Accumulated per-phase wall time across training steps. Filled by
    /// `train_step` (backward / dense Adam / embedding Adam) and by
    /// `train_one_epoch` (forward). `Instant::now()` is ~tens of ns on
    /// Linux so the always-on cost over ~100k steps/epoch is well under
    /// 10 ms. Read + reset by `train_one_epoch` once per epoch; printed
    /// when `SIGHURT_TIME_STEPS=1`.
    pub profile: StepProfile,
}

/// Per-phase wall-time accumulator for one training epoch's worth of
/// `train_step` calls. Nanoseconds because `Duration::as_nanos` returns
/// u128 and a 4-hour epoch fits comfortably.
#[derive(Default, Clone, Copy, Debug)]
pub struct StepProfile {
    pub forward_ns: u128,
    pub backward_ns: u128,
    pub adam_dense_ns: u128,
    pub adam_embed_ns: u128,
    pub steps: u64,
    /// Sum of per-step pre-clip ‖grad‖₂ values (across the whole
    /// network) over the epoch. Divide by `grad_norm_observations`
    /// for the mean. Useful for tuning `GLOBAL_GRAD_NORM_CLIP`.
    pub grad_norm_sum: f64,
    /// Number of training steps whose pre-clip ‖grad‖ exceeded the
    /// global threshold and were rescaled. If this is 100% the clip
    /// is too tight; if 0% it's purely a safety guard.
    pub grad_norm_clips: u64,
    /// Number of training steps observed for the grad-norm stats.
    /// Lags `steps` by exactly 0 — every `train_step` adds one.
    pub grad_norm_observations: u64,
}

impl StepProfile {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
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
        // Sample a fresh inverted-dropout mask per forward and apply
        // it to each *hidden* layer's activations. Output layer (the
        // softmax over vocab) is never dropped — that would silence
        // the very targets we're trying to predict.
        //
        // Subtlety: `Layer::forward(.., cache=true)` caches the
        // *pre-dropout* tanh output in `last_activations`. Backward
        // (`compute_deltas_into`) reads that to compute the tanh
        // derivative `1 - a²`. The dropout *mask* is a separate
        // multiplicative factor on the gradient flowing through that
        // neuron — applied independently in backward. We must NOT
        // overwrite `last_activations` with the post-dropout values,
        // because `1 - (scale*tanh(z))²` ≠ `1 - tanh(z)²` and the
        // former collapses to zero whenever `scale*tanh(z) → ±1`,
        // crippling training (observed: 90-min run regressed val
        // 5.80 → 7.10 across 2 epochs at p=0.1). Forward propagation
        // does see the post-dropout values via the `input` variable
        // (which is what gets passed to the next layer).
        let p = self.dropout_p;
        let keep = 1.0 - p;
        let scale = if keep > 0.0 { 1.0 / keep } else { 0.0 };
        let mut rng = rand::thread_rng();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            input = layer.forward(gpu, &input, true)?;
            if i == n - 1 {
                // Final layer: softmax, no dropout. Re-cache because
                // backward reads `last_activations` for the output
                // delta `softmax - one_hot`.
                softmax_inplace(&mut input);
                layer.last_activations.clone_from(&input);
                layer.last_dropout_mask.clear();
            } else if p > 0.0 {
                // Hidden layer with dropout enabled. Sample a fresh
                // mask, scale `input` (which propagates forward) but
                // LEAVE `last_activations` at its pre-dropout value.
                layer.last_dropout_mask.clear();
                layer.last_dropout_mask.reserve(input.len());
                for v in input.iter_mut() {
                    let r: f32 = rng.gen_range(0.0..1.0);
                    let m = if r < keep { scale } else { 0.0 };
                    layer.last_dropout_mask.push(m);
                    *v *= m;
                }
            } else {
                layer.last_dropout_mask.clear();
            }
        }
        Ok(input)
    }

    /// Turn on per-hidden-layer LayerNorm. Intended for FRESH
    /// networks only (the trainer's `--layer-norm` flag) — flipping
    /// this on a model trained without LN would abruptly change its
    /// forward function. Every layer except the output gets identity
    /// LN parameters (γ = 1, β = 0) with zeroed Adam moments.
    /// Deliberately a separate method rather than a `network_init`
    /// parameter so the existing `network_init` signature (called by
    /// main.rs / serve.rs) stays stable.
    pub fn enable_layer_norm(&mut self) {
        self.layer_norm = true;
        let n = self.layers.len();
        for layer in &mut self.layers[..n - 1] {
            layer.init_layer_norm();
        }
    }

    /// Adam (AdamW-decoupled) update. Per-output-neuron parallelism via rayon
    /// — each row of every layer's weight matrix is independent so the inner
    /// loop scales linearly with cores. The embedding update happens once per
    /// call and only touches a few rows.
    pub fn train_step(&mut self, lr: f32, target_idx: usize) {
        self.adam_step = self.adam_step.saturating_add(1);
        let step = self.adam_step as i32;
        let bc1 = (1.0 - ADAM_BETA1.powi(step)).max(1e-12);
        let bc2 = (1.0 - ADAM_BETA2.powi(step)).max(1e-12);

        // Move scratch out so we can pass `&Network` (for reads) and
        // `&mut BackpropScratch` (for writes) as disjoint borrows. Returned
        // at end of function.
        let mut scratch = std::mem::take(&mut self.backprop_scratch);
        let t_back = std::time::Instant::now();
        crate::teacher::compute_deltas_into(self, target_idx, &mut scratch);
        // Global-norm clip across all gradients (every layer's delta
        // vector + the embedding input_grad). Computed via SUM-of-
        // squares then a single sqrt so it's one pass even with many
        // layers. Preserves gradient direction, unlike the per-element
        // clamp inside the inner Adam loop.
        {
            let mut sq: f64 = 0.0;
            for d in &scratch.layer_deltas {
                for &g in d.iter() {
                    sq += (g as f64) * (g as f64);
                }
            }
            for &g in scratch.input_grad.iter() {
                sq += (g as f64) * (g as f64);
            }
            // LN parameter gradients participate too (empty vecs when
            // LayerNorm is off, so this is free for legacy models).
            for d in scratch.ln_gain_grad.iter().chain(scratch.ln_bias_grad.iter()) {
                for &g in d.iter() {
                    sq += (g as f64) * (g as f64);
                }
            }
            let norm = sq.sqrt() as f32;
            let clipped = norm > GLOBAL_GRAD_NORM_CLIP && norm.is_finite();
            if clipped {
                let scale = GLOBAL_GRAD_NORM_CLIP / norm;
                for d in scratch.layer_deltas.iter_mut() {
                    for g in d.iter_mut() {
                        *g *= scale;
                    }
                }
                for g in scratch.input_grad.iter_mut() {
                    *g *= scale;
                }
                for d in scratch
                    .ln_gain_grad
                    .iter_mut()
                    .chain(scratch.ln_bias_grad.iter_mut())
                {
                    for g in d.iter_mut() {
                        *g *= scale;
                    }
                }
            }
            self.profile.grad_norm_sum += norm as f64;
            if clipped {
                self.profile.grad_norm_clips += 1;
            }
            self.profile.grad_norm_observations += 1;
        }
        self.profile.backward_ns += t_back.elapsed().as_nanos();
        let t_adam_dense = std::time::Instant::now();
        let bp = &scratch;
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            let d = &bp.layer_deltas[idx];
            let cols = layer.cols;
            let input_ref: &[f32] = &layer.last_input;
            let biases_ptr = layer.biases.as_mut_ptr() as usize;
            let bm_ptr = layer.b_m.as_mut_ptr() as usize;
            let bv_ptr = layer.b_v.as_mut_ptr() as usize;

            // Walk rows in parallel: each row owns a disjoint slice of weights,
            // w_m, w_v plus exactly one bias / b_m / b_v slot.
            layer
                .weights
                .par_chunks_mut(cols)
                .zip(layer.w_m.par_chunks_mut(cols))
                .zip(layer.w_v.par_chunks_mut(cols))
                .enumerate()
                .for_each(|(j, ((w_row, m_row), v_row))| {
                    let g = d[j].clamp(-GRAD_CLIP, GRAD_CLIP);
                    // SAFETY: each j accesses a unique index into biases / b_m / b_v,
                    // so the raw-pointer writes don't alias across threads.
                    unsafe {
                        let bptr = (biases_ptr as *mut f32).add(j);
                        let bmp = (bm_ptr as *mut f32).add(j);
                        let bvp = (bv_ptr as *mut f32).add(j);
                        let m_new = ADAM_BETA1 * *bmp + (1.0 - ADAM_BETA1) * g;
                        let v_new = ADAM_BETA2 * *bvp + (1.0 - ADAM_BETA2) * g * g;
                        *bmp = m_new;
                        *bvp = v_new;
                        let m_hat = m_new / bc1;
                        let v_hat = v_new / bc2;
                        let mut b = *bptr;
                        b -= lr * (m_hat / (v_hat.sqrt() + ADAM_EPS) + WEIGHT_DECAY * b);
                        *bptr = b;
                    }
                    if g == 0.0 {
                        return;
                    }
                    // SIMD body: process 8 weights per iteration with f32x8.
                    let chunks = cols / SIMD_LANES;
                    let tail_start = chunks * SIMD_LANES;
                    let g_v = f32x8::splat(g);
                    let beta1_v = f32x8::splat(ADAM_BETA1);
                    let omb1_v = f32x8::splat(1.0 - ADAM_BETA1);
                    let beta2_v = f32x8::splat(ADAM_BETA2);
                    let omb2_v = f32x8::splat(1.0 - ADAM_BETA2);
                    let lr_v = f32x8::splat(lr);
                    let wd_v = f32x8::splat(WEIGHT_DECAY);
                    let eps_v = f32x8::splat(ADAM_EPS);
                    let bc1_v = f32x8::splat(bc1);
                    let bc2_v = f32x8::splat(bc2);
                    for ci in 0..chunks {
                        let off = ci * SIMD_LANES;
                        let x_v = load_f32x8(&input_ref[off..]);
                        let gk_v = g_v * x_v;
                        let m_old = load_f32x8(&m_row[off..]);
                        let v_old = load_f32x8(&v_row[off..]);
                        let w_old = load_f32x8(&w_row[off..]);
                        let m_new = beta1_v * m_old + omb1_v * gk_v;
                        let v_new = beta2_v * v_old + omb2_v * gk_v * gk_v;
                        let m_hat = m_new / bc1_v;
                        let v_hat = v_new / bc2_v;
                        let w_new = w_old
                            - lr_v * (m_hat / (v_hat.sqrt() + eps_v) + wd_v * w_old);
                        store_f32x8(&mut m_row[off..], m_new);
                        store_f32x8(&mut v_row[off..], v_new);
                        store_f32x8(&mut w_row[off..], w_new);
                    }
                    // Scalar tail for cols % 8 (e.g. first layer with
                    // input_size = embed_dim * context_window + 1 = 8193,
                    // which leaves a 1-float remainder).
                    for k in tail_start..cols {
                        let x = input_ref[k];
                        let gk = g * x;
                        let m_new = ADAM_BETA1 * m_row[k] + (1.0 - ADAM_BETA1) * gk;
                        let v_new = ADAM_BETA2 * v_row[k] + (1.0 - ADAM_BETA2) * gk * gk;
                        m_row[k] = m_new;
                        v_row[k] = v_new;
                        let m_hat = m_new / bc1;
                        let v_hat = v_new / bc2;
                        let mut w = w_row[k];
                        w -= lr * (m_hat / (v_hat.sqrt() + ADAM_EPS) + WEIGHT_DECAY * w);
                        w_row[k] = w;
                    }
                });
            layer.gpu_dirty = true;
        }

        // LayerNorm gain/bias Adam updates. Frozen (skipped entirely)
        // when LayerNorm is off — `ln_active()` is false for every
        // layer of a legacy model, so v2/v3/v4 training behavior is
        // untouched. Same element-wise GRAD_CLIP clamp as the dense
        // parameters. NO weight decay here: decaying γ pulls it toward
        // 0 and fights the normalization itself (standard practice —
        // AdamW implementations exclude LN parameters from decay).
        // Serial loop: 2 × rows per hidden layer is trivial next to
        // the dense update above.
        if self.layer_norm {
            for (idx, layer) in self.layers.iter_mut().enumerate() {
                if !layer.ln_active() {
                    continue;
                }
                let gg = &bp.ln_gain_grad[idx];
                let bg = &bp.ln_bias_grad[idx];
                debug_assert_eq!(gg.len(), layer.rows);
                debug_assert_eq!(bg.len(), layer.rows);
                for j in 0..layer.rows {
                    let g = gg[j].clamp(-GRAD_CLIP, GRAD_CLIP);
                    let m_new = ADAM_BETA1 * layer.ln_gain_m[j] + (1.0 - ADAM_BETA1) * g;
                    let v_new = ADAM_BETA2 * layer.ln_gain_v[j] + (1.0 - ADAM_BETA2) * g * g;
                    layer.ln_gain_m[j] = m_new;
                    layer.ln_gain_v[j] = v_new;
                    layer.ln_gain[j] -=
                        lr * (m_new / bc1) / ((v_new / bc2).sqrt() + ADAM_EPS);

                    let g = bg[j].clamp(-GRAD_CLIP, GRAD_CLIP);
                    let m_new = ADAM_BETA1 * layer.ln_bias_m[j] + (1.0 - ADAM_BETA1) * g;
                    let v_new = ADAM_BETA2 * layer.ln_bias_v[j] + (1.0 - ADAM_BETA2) * g * g;
                    layer.ln_bias_m[j] = m_new;
                    layer.ln_bias_v[j] = v_new;
                    layer.ln_bias[j] -=
                        lr * (m_new / bc1) / ((v_new / bc2).sqrt() + ADAM_EPS);
                }
            }
        }
        self.profile.adam_dense_ns += t_adam_dense.elapsed().as_nanos();

        let t_adam_embed = std::time::Instant::now();
        let embed_grad_len = self.context_window * self.embed_dim;
        self.embedding.apply_grad_adam(
            &bp.input_grad[..embed_grad_len],
            lr,
            ADAM_BETA1,
            ADAM_BETA2,
            ADAM_EPS,
            WEIGHT_DECAY,
            self.adam_step,
        );
        self.profile.adam_embed_ns += t_adam_embed.elapsed().as_nanos();
        self.profile.steps += 1;

        // Return scratch buffers to the network for the next call to reuse.
        self.backprop_scratch = scratch;
    }
}

pub struct VocabIndex<'a> {
    pub words: &'a [String],
    pub idx: HashMap<&'a str, usize>,
    pub pad_id: usize,
    pub unk_id: usize,
    /// Vocab id of the bot's close tag (`</PERSON_0>`). Generation stops on this.
    pub bot_close_id: Option<usize>,
    /// Tokens that must never be emitted during generation: PAD, UNK, SEC,
    /// every `<PERSON_N>` open tag, and every `</PERSON_N>` close tag
    /// except the bot's own (which IS allowed — as the stop signal).
    pub forbidden_emit_ids: Vec<usize>,
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
        let bot_close_id = idx.get(bot_close_tag().as_str()).copied();
        let mut forbidden_emit_ids: Vec<usize> = Vec::new();
        for (i, w) in words.iter().enumerate() {
            let w_str = w.as_str();
            // Structural specials: PAD, UNK, SEC — never natural-language output.
            if w_str == PAD || w_str == UNK || w_str == crate::tokenizer::SEC {
                forbidden_emit_ids.push(i);
                continue;
            }
            // Training-only placeholders: corpus cleaner rules 9/10/11
            // replace URLs / Discord mentions / emoji-shortcodes with
            // these sentinel tokens so the punctuation-splitting
            // tokenizer doesn't shatter them into per-character noise.
            // The bot training-time WANTS these in the input/target
            // streams as compact stand-ins, but emitting them in a
            // generated reply is meaningless to a human reader — the
            // user has already reported seeing "__MENTION__" appear
            // mid-sentence. Mask out at generation time.
            if w_str == crate::tokenizer::URL_PLACEHOLDER
                || w_str == crate::tokenizer::MENTION_PLACEHOLDER
                || w_str == crate::tokenizer::EMOJI_PLACEHOLDER
            {
                forbidden_emit_ids.push(i);
                continue;
            }
            if crate::persons::parse_open_tag(w_str).is_some() {
                forbidden_emit_ids.push(i);
                continue;
            }
            if let Some(pid) = crate::persons::parse_close_tag(w_str) {
                if pid != BOT_PERSON_ID {
                    forbidden_emit_ids.push(i);
                }
            }
        }
        Self {
            words,
            idx,
            pad_id,
            unk_id,
            bot_close_id,
            forbidden_emit_ids,
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
        adam_step: 0,
        dropout_p: 0.0,
        label_smoothing: 0.0,
        layer_norm: false,
        backprop_scratch: crate::teacher::BackpropScratch::default(),
        profile: StepProfile::default(),
    })
}

/// Zero out the probability mass at `forbidden_ids` in place. Caller is
/// responsible for renormalizing if needed; `sample_top_k` only sorts by
/// raw value so unnormalized inputs work too.
fn mask_forbidden(probs: &mut [f32], forbidden_ids: &[usize]) {
    for &i in forbidden_ids {
        if i < probs.len() {
            probs[i] = 0.0;
        }
    }
}

/// Cached `SIGHURT_TEMPERATURE` (sampling temperature, default 1.0).
/// Read ONCE per process via `OnceLock` — generation is per-token and
/// `std::env::var` costs a lock + allocation per call, which is
/// pointless for a value that never changes mid-run. Non-numeric,
/// non-finite, or non-positive values fall back to 1.0 (off).
static SAMPLING_TEMPERATURE: OnceLock<f32> = OnceLock::new();
fn sampling_temperature() -> f32 {
    *SAMPLING_TEMPERATURE.get_or_init(|| {
        std::env::var("SIGHURT_TEMPERATURE")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|t| t.is_finite() && *t > 0.0)
            .unwrap_or(1.0)
    })
}

/// Cached `SIGHURT_TOP_P` (nucleus sampling mass, in (0, 1]). `None`
/// (unset or out of range) keeps the legacy top-k=5 behavior. Same
/// read-once rationale as `SAMPLING_TEMPERATURE`.
static SAMPLING_TOP_P: OnceLock<Option<f32>> = OnceLock::new();
fn sampling_top_p() -> Option<f32> {
    *SAMPLING_TOP_P.get_or_init(|| {
        std::env::var("SIGHURT_TOP_P")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|p| p.is_finite() && *p > 0.0 && *p <= 1.0)
    })
}

/// Temperature reshaping on an (possibly unnormalized) probability
/// vector: `p_i → p_i^(1/T)`, then renormalize to sum 1. Operating on
/// probabilities rather than logits is equivalent to the usual
/// `logits/T` formulation up to the shared softmax normalizer, and the
/// probs are what we have here (the network's forward already applied
/// softmax). T < 1 sharpens, T > 1 flattens, T = 1 is a no-op and is
/// skipped entirely so the default path costs nothing.
fn apply_temperature(probs: &mut [f32], temperature: f32) {
    if !(temperature > 0.0) || (temperature - 1.0).abs() < 1e-6 {
        return;
    }
    let inv_t = 1.0 / temperature;
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        // powf(0, x) = 0 for x > 0, so masked-out entries stay masked.
        *p = p.max(0.0).powf(inv_t);
        sum += *p;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

/// Nucleus (top-p) sampling: sort descending, take the smallest prefix
/// whose cumulative mass reaches `top_p` of the TOTAL remaining mass
/// (the vector arrives masked/penalized, hence unnormalized — using a
/// fraction of the total is equivalent to renormalizing first), then
/// sample proportionally within that prefix. Falls back like
/// `sample_top_k`: empty/degenerate input returns index 0 / the
/// highest-probability index.
fn sample_top_p(probs: &[f32], top_p: f32) -> usize {
    let mut pairs: Vec<(f32, usize)> = probs
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_finite() && **p > 0.0)
        .map(|(i, p)| (*p, i))
        .collect();
    if pairs.is_empty() {
        return 0;
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = pairs.iter().map(|(p, _)| *p).sum();
    if total <= 0.0 {
        return pairs[0].1;
    }
    let threshold = top_p * total;
    let mut cut = pairs.len();
    let mut cum = 0.0f32;
    for (i, (p, _)) in pairs.iter().enumerate() {
        cum += *p;
        if cum >= threshold {
            cut = i + 1;
            break;
        }
    }
    let nucleus = &pairs[..cut];
    let sum: f32 = nucleus.iter().map(|(p, _)| *p).sum();
    if sum <= 0.0 {
        return nucleus[0].1;
    }
    let mut r = rand::thread_rng().gen_range(0.0..sum);
    for (p, i) in nucleus {
        r -= *p;
        if r <= 0.0 {
            return *i;
        }
    }
    // Floating-point drift fallback, mirroring `sample_top_k`.
    nucleus[0].1
}

/// The ONE sampling pipeline, shared by `generate` and the
/// free-running branch of `generate_and_train`. Historically the two
/// diverged — `generate_and_train` was missing the repetition penalty
/// — and per-path tweaks kept drifting; every masking/penalty/
/// temperature/truncation step now lives here, in order:
///
/// 1. forbidden-token mask (PAD/UNK/tags/placeholders),
/// 2. minimum-reply-length mask (`</PERSON_0>` zeroed until
///    `emitted_tokens >= MIN_REPLY_TOKENS`),
/// 3. repetition penalty over `recent_ids`,
/// 4. temperature reshaping (`SIGHURT_TEMPERATURE`),
/// 5. nucleus sampling when `SIGHURT_TOP_P` is set, else top-k=5.
///
/// `emitted_tokens` counts non-tag tokens already produced this reply
/// (tags never reach the produced list, so `produced.len()` is exactly
/// that count). `probs` is consumed destructively.
fn sample_reply_token(
    probs: &mut [f32],
    vocab: &VocabIndex<'_>,
    recent_ids: &[usize],
    emitted_tokens: usize,
) -> usize {
    mask_forbidden(probs, &vocab.forbidden_emit_ids);
    // Best legal (non-forbidden) token BEFORE the destructive min-length /
    // penalty / temperature steps. This is the safety net: those steps can
    // legitimately drive the whole vector to zero — the min-length mask
    // zeroes the close tag when it holds ~all the mass early in a reply,
    // and `apply_temperature`'s p^(1/T) underflows tiny residual mass to
    // exactly 0.0 in f32 at low temperatures. Without a fallback the
    // samplers would return index 0 (`<PAD>`), bypassing the forbidden
    // mask and spraying `<PAD>` into the reply. Instead we fall back to
    // this token, which the model actually wanted and which is guaranteed
    // legal (or a stop when literally nothing is legal).
    let (best_legal_idx, best_legal_val) = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| p.is_finite())
        .fold((usize::MAX, 0.0f32), |(bi, bv), (i, p)| {
            if p > bv { (i, p) } else { (bi, bv) }
        });
    if best_legal_val <= 0.0 {
        // Model put all its mass on forbidden tokens — no legal
        // continuation exists. Signal a clean stop (out-of-range index).
        return probs.len();
    }

    if emitted_tokens < MIN_REPLY_TOKENS {
        if let Some(close_id) = vocab.bot_close_id {
            if close_id < probs.len() {
                probs[close_id] = 0.0;
            }
        }
    }
    apply_repetition_penalty(probs, recent_ids, REPETITION_PENALTY);
    apply_temperature(probs, sampling_temperature());

    // If the pipeline zeroed everything, take the pre-mask legal argmax
    // rather than letting the samplers fall through to index 0 (`<PAD>`).
    let has_mass = probs.iter().any(|p| p.is_finite() && *p > 0.0);
    if !has_mass {
        return best_legal_idx;
    }

    match sampling_top_p() {
        Some(p) => sample_top_p(probs, p),
        None => sample_top_k(probs, TOP_K_SAMPLE),
    }
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
    // Floating-point drift can leave r slightly > 0 after the loop; fall
    // back to the highest-prob token rather than the last (lowest).
    top[0].1
}

fn detokenize(tokens: &[String]) -> String {
    let mut out = String::new();
    // "@" glues to the token AFTER it: the corpus now contains
    // learnable `@name` mention tokens which the punctuation-splitting
    // tokenizer shatters into "@" + "name", and rendering them as
    // "hey@ gustav" (the old generic-punctuation rule: no space
    // before, space after) reads wrong. We want "hey @gustav" —
    // space BEFORE the "@", none after.
    let mut glue_next = false;
    for tok in tokens {
        let is_punct = tok.chars().all(|c| !c.is_alphanumeric() && c != '\'');
        if out.is_empty() {
            out.push_str(tok);
        } else if glue_next {
            out.push_str(tok);
        } else if tok == "@" {
            out.push(' ');
            out.push_str(tok);
        } else if is_punct {
            out.push_str(tok);
        } else {
            out.push(' ');
            out.push_str(tok);
        }
        glue_next = tok == "@";
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
    let bot_open = bot_open_tag();
    let bot_close = bot_close_tag();

    let mut context: Vec<String> = Vec::new();
    for m in memory {
        context.extend(tokenize(m));
    }
    context.extend(tokenize(start));
    // Prime the generator: append the bot's open tag so the next token the
    // model produces is conditioned on "now I am speaking."
    context.push(bot_open);

    let mut produced: Vec<String> = Vec::new();
    let mut recent_ids: Vec<usize> = Vec::with_capacity(REPETITION_WINDOW);
    for i in 0..MAX_GENERATION_LEN {
        let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);
        let mut probs = net.forward(gpu, &window, i)?;
        let idx = sample_reply_token(&mut probs, &vocab, &recent_ids, produced.len());
        if idx >= words.len() {
            break;
        }
        let next = &words[idx];
        if next == &bot_close {
            break;
        }
        // Slide the repetition window.
        if recent_ids.len() >= REPETITION_WINDOW {
            recent_ids.remove(0);
        }
        recent_ids.push(idx);
        context.push(next.clone());
        produced.push(next.clone());
    }
    Ok(detokenize(&produced))
}

/// Multiply `probs[id]` by `penalty` for every `id` in `recent_ids`.
/// `penalty < 1.0` downweights recently-emitted tokens, reducing the
/// chance the model emits the same token several times in a row.
/// IDs out of `probs` range are silently ignored (defensive — should
/// never happen if `recent_ids` is built from `sample_top_k` output).
fn apply_repetition_penalty(probs: &mut [f32], recent_ids: &[usize], penalty: f32) {
    if penalty >= 1.0 {
        return;
    }
    for &id in recent_ids {
        if id < probs.len() {
            probs[id] *= penalty;
        }
    }
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
    let bot_open = bot_open_tag();
    let bot_close = bot_close_tag();

    let mut context: Vec<String> = memory_tokens;
    context.extend(user_tokens.iter().cloned());
    context.push(bot_open.clone());
    let mut produced: Vec<String> = Vec::new();

    // Teacher list + sentinel </PERSON_0> so the model also learns to stop.
    let mut targets: Vec<String> = teacher.iter().cloned().collect();
    if !targets.is_empty() {
        targets.push(bot_close.clone());
    }

    // Repetition window shared across the teacher-forced and
    // free-running phases so the penalty sees a continuous stream of
    // recent emissions when the model takes over from the teacher.
    let mut recent_ids: Vec<usize> = Vec::with_capacity(REPETITION_WINDOW);
    let push_recent = |recent_ids: &mut Vec<usize>, id: usize| {
        if recent_ids.len() >= REPETITION_WINDOW {
            recent_ids.remove(0);
        }
        recent_ids.push(id);
    };

    for i in 0..MAX_GENERATION_LEN {
        let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);

        match targets.get(i) {
            Some(t) => {
                let _probs = net.forward_and_cache(gpu, &window, i)?;
                if let Some(target_id) = vocab.lookup(t) {
                    net.train_step(lr, target_id);
                    push_recent(&mut recent_ids, target_id);
                }
                if t == &bot_close {
                    break;
                }
                context.push(t.clone());
                produced.push(t.clone());
            }
            None => {
                // Free-running: same shared sampling pipeline as
                // `generate` (this branch historically lacked the
                // repetition penalty — a known divergence, now fixed
                // by construction).
                let mut probs = net.forward(gpu, &window, i)?;
                let idx =
                    sample_reply_token(&mut probs, &vocab, &recent_ids, produced.len());
                if idx >= words.len() {
                    break;
                }
                let next = &words[idx];
                if next == &bot_close {
                    break;
                }
                push_recent(&mut recent_ids, idx);
                context.push(next.clone());
                produced.push(next.clone());
            }
        }
    }

    Ok(detokenize(&produced))
}

/// One supervised example: given the prelude plus the immediately-preceding
/// turn (`query`), predict the next speaker's content. `query` is already
/// wrapped with the query speaker's `<PERSON_N>...</PERSON_N>` tags;
/// `prelude` has every earlier turn wrapped likewise. The training loop
/// prepends `<PERSON_{target_person_id}>` to the context and appends the
/// matching close tag as the stop target.
///
/// We emit an example for EVERY adjacent pair, not just bot-as-target.
/// The model learns general conversational dynamics and the speaker
/// identity is conditioned via the forced open tag at the head of the
/// generation. At inference, `generate()` forces `<PERSON_0>` and stops at
/// `</PERSON_0>`, so the bot speaks as itself even though training saw
/// many speakers.
#[derive(Clone)]
pub struct TrainExample {
    pub prelude: Vec<String>,
    pub query: Vec<String>,
    pub target_person_id: u32,
    pub target_tokens: Vec<String>,
}

pub fn extract_train_examples(data: &Data) -> Vec<TrainExample> {
    let mut examples = Vec::new();
    for section in &data.sections {
        let mut prelude_tokens: Vec<String> = Vec::new();
        let n = section.len();
        if n < 2 {
            continue;
        }
        for i in 0..n - 1 {
            let a = &section[i];
            let b = &section[i + 1];
            if !b.tokens.is_empty() {
                let mut target = b.tokens.clone();
                target.truncate(MAX_TARGET_TOKENS);
                examples.push(TrainExample {
                    prelude: prelude_tokens.clone(),
                    query: wrap_turn(a),
                    target_person_id: b.person_id,
                    target_tokens: target,
                });
            }
            // After processing this pair, turn `a` joins the prelude.
            prelude_tokens.extend(wrap_turn(a));
        }
    }
    examples
}

pub struct EpochStats {
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_targets: u64,
    pub val_targets: u64,
}

/// One supervised epoch. Each example's bot turn is presented as:
/// `[prelude] [query (wrapped)] <PERSON_0> [bot tokens] </PERSON_0>`.
/// The model is trained to predict each bot content token plus the closing
/// `</PERSON_0>` stop. `prelude_drop_prob` randomly hides the in-section
/// prelude during training so cold-start responses also stay reasonable.
pub fn train_one_epoch(
    gpu: &Gpu,
    net: &mut Network,
    train_examples: &mut [TrainExample],
    val_examples: &[TrainExample],
    words: &[String],
    lr: f32,
    prelude_drop_prob: f32,
) -> Result<EpochStats> {
    let vocab = VocabIndex::new(words);
    let mut rng = rand::thread_rng();
    train_examples.shuffle(&mut rng);

    // Per-epoch phase timing is reset here. `train_step` adds to the
    // backward / Adam buckets, this loop adds to the forward bucket.
    net.profile.reset();

    let mut train_loss = 0.0f64;
    let mut train_targets = 0u64;
    for ex in train_examples.iter() {
        let target_open = open_tag(ex.target_person_id);
        let target_close = close_tag(ex.target_person_id);
        let drop: f32 = rng.gen_range(0.0..1.0);
        let prelude: &[String] = if drop < prelude_drop_prob {
            &[]
        } else {
            &ex.prelude
        };
        let mut context: Vec<String> = prelude.to_vec();
        context.extend(ex.query.iter().cloned());
        context.push(target_open);

        let mut targets: Vec<String> = ex.target_tokens.clone();
        targets.push(target_close.clone());

        for (i, target_word) in targets.iter().enumerate() {
            let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);
            let t_fwd = std::time::Instant::now();
            let probs = net.forward_and_cache(gpu, &window, i)?;
            net.profile.forward_ns += t_fwd.elapsed().as_nanos();
            if let Some(target_id) = vocab.lookup(target_word) {
                let p = probs[target_id].max(1e-9);
                train_loss += -(p as f64).ln();
                train_targets += 1;
                net.train_step(lr, target_id);
            }
            if target_word == &target_close {
                break;
            }
            context.push(target_word.clone());
        }
    }

    if std::env::var("SIGHURT_TIME_STEPS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        let p = &net.profile;
        let total = (p.forward_ns + p.backward_ns + p.adam_dense_ns + p.adam_embed_ns) as f64;
        let steps = p.steps.max(1) as f64;
        let pct = |x: u128| -> f64 {
            if total > 0.0 {
                100.0 * (x as f64) / total
            } else {
                0.0
            }
        };
        eprintln!(
            "  timing> steps={} fwd={:.1}%/{:.0}µs back={:.1}%/{:.0}µs adam_dense={:.1}%/{:.0}µs adam_embed={:.1}%/{:.0}µs",
            p.steps,
            pct(p.forward_ns),
            (p.forward_ns as f64) / steps / 1_000.0,
            pct(p.backward_ns),
            (p.backward_ns as f64) / steps / 1_000.0,
            pct(p.adam_dense_ns),
            (p.adam_dense_ns as f64) / steps / 1_000.0,
            pct(p.adam_embed_ns),
            (p.adam_embed_ns as f64) / steps / 1_000.0,
        );
        if p.grad_norm_observations > 0 {
            let mean = p.grad_norm_sum / p.grad_norm_observations as f64;
            let clip_pct = 100.0 * (p.grad_norm_clips as f64) / (p.grad_norm_observations as f64);
            eprintln!(
                "  grad>  mean ‖g‖₂={:.4} clipped={}/{} ({:.1}%)",
                mean, p.grad_norm_clips, p.grad_norm_observations, clip_pct,
            );
        }
    }

    let mut val_loss = 0.0f64;
    let mut val_targets = 0u64;
    for ex in val_examples {
        let target_open = open_tag(ex.target_person_id);
        let target_close = close_tag(ex.target_person_id);
        let mut context: Vec<String> = ex.prelude.clone();
        context.extend(ex.query.iter().cloned());
        context.push(target_open);
        let mut targets: Vec<String> = ex.target_tokens.clone();
        targets.push(target_close.clone());
        for (i, target_word) in targets.iter().enumerate() {
            let window = build_token_window(&context, &vocab, CONTEXT_WINDOW);
            let probs = net.forward(gpu, &window, i)?;
            if let Some(target_id) = vocab.lookup(target_word) {
                let p = probs[target_id].max(1e-9);
                val_loss += -(p as f64).ln();
                val_targets += 1;
            }
            if target_word == &target_close {
                break;
            }
            context.push(target_word.clone());
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
    let mut examples = extract_train_examples(data);
    let val_n = (examples.len() / 10)
        .max(1)
        .min(examples.len().saturating_sub(1));
    let val_examples: Vec<TrainExample> = examples.split_off(examples.len() - val_n);
    for epoch in 0..epochs {
        let stats = train_one_epoch(gpu, net, &mut examples, &val_examples, words, lr, 0.0)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_index_collects_forbidden_emit_tokens() {
        // PAD=0, UNK=1, SEC=2, <PERSON_0>=3, </PERSON_0>=4, <PERSON_1>=5,
        // </PERSON_1>=6, then content tokens.
        let words: Vec<String> = vec![
            "<PAD>".into(),
            "<UNK>".into(),
            "<SEC>".into(),
            "<PERSON_0>".into(),
            "</PERSON_0>".into(),
            "<PERSON_1>".into(),
            "</PERSON_1>".into(),
            "hello".into(),
        ];
        let vi = VocabIndex::new(&words);
        // PAD/UNK/SEC, <PERSON_0> open, <PERSON_1> open, </PERSON_1> close.
        // Bot's </PERSON_0> close (id 4) is NOT forbidden — that's our stop.
        let forbidden: std::collections::HashSet<usize> =
            vi.forbidden_emit_ids.iter().copied().collect();
        assert!(forbidden.contains(&0)); // PAD
        assert!(forbidden.contains(&1)); // UNK
        assert!(forbidden.contains(&2)); // SEC
        assert!(forbidden.contains(&3)); // <PERSON_0>
        assert!(forbidden.contains(&5)); // <PERSON_1>
        assert!(forbidden.contains(&6)); // </PERSON_1>
        assert!(!forbidden.contains(&4)); // </PERSON_0> — bot's stop, allowed
        assert!(!forbidden.contains(&7)); // hello — content, allowed
    }

    #[test]
    fn placeholder_tokens_are_forbidden_at_generation() {
        // Regression: a real user reported the bot emitting literal
        // `__MENTION__` / `__URL__` mid-sentence. Those are training-
        // only sentinels added by clean_corpus and should never
        // appear in a generated reply.
        let words: Vec<String> = vec![
            "<PAD>".into(),
            "<UNK>".into(),
            "<SEC>".into(),
            "<PERSON_0>".into(),
            "</PERSON_0>".into(),
            "<PERSON_1>".into(),
            "</PERSON_1>".into(),
            "__URL__".into(),
            "__MENTION__".into(),
            "__EMOJI__".into(),
            "hello".into(),
        ];
        let vi = VocabIndex::new(&words);
        let forbidden: std::collections::HashSet<usize> =
            vi.forbidden_emit_ids.iter().copied().collect();
        // Placeholders forbidden.
        assert!(forbidden.contains(&7), "__URL__ should be forbidden");
        assert!(forbidden.contains(&8), "__MENTION__ should be forbidden");
        assert!(forbidden.contains(&9), "__EMOJI__ should be forbidden");
        // Normal content still allowed.
        assert!(!forbidden.contains(&10), "regular tokens still allowed");
    }

    #[test]
    fn mask_forbidden_zeros_out_specified_positions() {
        let mut probs = vec![0.5, 0.3, 0.1, 0.1];
        mask_forbidden(&mut probs, &[1, 3]);
        assert_eq!(probs, vec![0.5, 0.0, 0.1, 0.0]);
    }

    #[test]
    fn mask_forbidden_tolerates_out_of_range() {
        let mut probs = vec![0.5, 0.5];
        mask_forbidden(&mut probs, &[0, 5, 7]);
        assert_eq!(probs, vec![0.0, 0.5]);
    }

    #[test]
    fn repetition_penalty_downweights_recent_ids() {
        let mut probs = vec![1.0, 1.0, 1.0, 1.0];
        apply_repetition_penalty(&mut probs, &[1, 3], 0.5);
        assert_eq!(probs, vec![1.0, 0.5, 1.0, 0.5]);
    }

    #[test]
    fn repetition_penalty_is_noop_at_unity() {
        let mut probs = vec![0.4, 0.6];
        apply_repetition_penalty(&mut probs, &[0, 1], 1.0);
        assert_eq!(probs, vec![0.4, 0.6]);
    }

    #[test]
    fn repetition_penalty_tolerates_out_of_range_ids() {
        let mut probs = vec![1.0, 1.0];
        apply_repetition_penalty(&mut probs, &[0, 5, 9], 0.5);
        assert_eq!(probs, vec![0.5, 1.0]);
    }

    fn toks(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn detokenize_glues_mention_after_at_sign() {
        // The corpus contains learnable `@name` mention tokens which the
        // tokenizer splits into "@" + "name"; rendering must not insert
        // a space after the "@" (and must keep one before it).
        assert_eq!(detokenize(&toks(&["hey", "@", "gustav"])), "hey @gustav");
    }

    #[test]
    fn detokenize_at_sign_at_reply_start() {
        assert_eq!(detokenize(&toks(&["@", "gustav", "hi"])), "@gustav hi");
    }

    #[test]
    fn detokenize_ordinary_punctuation_unchanged() {
        // Regression guard: the "@" special case must not disturb the
        // existing no-space-before-punctuation rendering.
        assert_eq!(detokenize(&toks(&["hello", ",", "world", "!"])), "hello, world!");
    }

    #[test]
    fn temperature_unity_is_noop() {
        let mut probs = vec![0.8, 0.2];
        apply_temperature(&mut probs, 1.0);
        assert_eq!(probs, vec![0.8, 0.2]);
    }

    #[test]
    fn temperature_below_one_sharpens_and_renormalizes() {
        // T = 0.5 → p^2, renormalized: [0.64, 0.04] / 0.68.
        let mut probs = vec![0.8, 0.2];
        apply_temperature(&mut probs, 0.5);
        assert!((probs[0] - 0.64 / 0.68).abs() < 1e-5);
        assert!((probs[1] - 0.04 / 0.68).abs() < 1e-5);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn temperature_above_one_flattens() {
        // T = 2 → sqrt, renormalized — the gap between the two entries
        // must shrink but the ordering must hold.
        let mut probs = vec![0.9, 0.1];
        apply_temperature(&mut probs, 2.0);
        assert!(probs[0] > probs[1]);
        assert!(probs[0] < 0.9);
        assert!(probs[1] > 0.1);
    }

    #[test]
    fn top_p_takes_smallest_sufficient_prefix() {
        // Sorted mass: 0.6, 0.3, 0.1. top_p = 0.5 → nucleus is just the
        // argmax, so sampling is deterministic.
        let probs = vec![0.1, 0.6, 0.3];
        for _ in 0..20 {
            assert_eq!(sample_top_p(&probs, 0.5), 1);
        }
    }

    #[test]
    fn top_p_excludes_tail_tokens() {
        // top_p = 0.9 → nucleus {0.6, 0.3}; index 0 (0.1) must never
        // be sampled.
        let probs = vec![0.1, 0.6, 0.3];
        for _ in 0..100 {
            let idx = sample_top_p(&probs, 0.9);
            assert!(idx == 1 || idx == 2, "sampled outside nucleus: {}", idx);
        }
    }

    #[test]
    fn top_p_works_on_unnormalized_mass() {
        // Masking/penalties leave the vector unnormalized; the nucleus
        // threshold is a FRACTION of the remaining total, so scaling
        // everything by 10 must not change the outcome.
        let probs = vec![1.0, 6.0, 3.0];
        for _ in 0..20 {
            assert_eq!(sample_top_p(&probs, 0.5), 1);
        }
    }

    /// Vocab fixture with the bot's close tag at index 4 and content
    /// tokens at 5/6, mirroring the production layout.
    fn min_len_vocab() -> Vec<String> {
        toks(&[
            "<PAD>",
            "<UNK>",
            "<SEC>",
            "<PERSON_0>",
            "</PERSON_0>",
            "hello",
            "world",
        ])
    }

    #[test]
    fn min_reply_length_masks_bot_close_early() {
        let words = min_len_vocab();
        let vi = VocabIndex::new(&words);
        assert_eq!(vi.bot_close_id, Some(4));
        // The close tag dominates the distribution — without the
        // minimum-length mask it would be sampled essentially always.
        let base = vec![0.0, 0.0, 0.0, 0.0, 0.97, 0.02, 0.01];
        for emitted in 0..MIN_REPLY_TOKENS {
            for _ in 0..25 {
                let mut probs = base.clone();
                let idx = sample_reply_token(&mut probs, &vi, &[], emitted);
                assert_ne!(idx, 4, "close tag sampled at emitted={}", emitted);
            }
        }
    }

    #[test]
    fn min_reply_length_allows_bot_close_after_threshold() {
        let words = min_len_vocab();
        let vi = VocabIndex::new(&words);
        let mut probs = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let idx = sample_reply_token(&mut probs, &vi, &[], MIN_REPLY_TOKENS);
        assert_eq!(idx, 4, "close tag must be samplable once the floor is met");
    }

    #[test]
    fn degenerate_all_mass_on_close_never_emits_pad() {
        // The close tag holds ALL the mass and we're below the min-length
        // floor, so the min-length mask zeroes the only nonzero entry. The
        // sampler must NOT fall through to index 0 (`<PAD>`); it falls back
        // to the pre-mask legal argmax (the close tag = clean stop).
        let words = min_len_vocab();
        let vi = VocabIndex::new(&words);
        for _ in 0..50 {
            let mut probs = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            let idx = sample_reply_token(&mut probs, &vi, &[], 0);
            assert_ne!(idx, 0, "must never emit <PAD>");
            assert_eq!(idx, 4, "falls back to the legal argmax (close = stop)");
        }
    }

    #[test]
    fn all_mass_on_forbidden_signals_stop() {
        // Model put everything on forbidden tokens (e.g. a PERSON open
        // tag). No legal continuation exists → return an out-of-range
        // index so the generate loop stops cleanly instead of emitting junk.
        let words = min_len_vocab();
        let vi = VocabIndex::new(&words);
        // index 1 = <UNK> (forbidden), everything else zero.
        let mut probs = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let idx = sample_reply_token(&mut probs, &vi, &[], MIN_REPLY_TOKENS);
        assert!(idx >= words.len(), "no legal token -> stop sentinel");
    }
}
