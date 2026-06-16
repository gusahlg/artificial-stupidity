//! Auto-trainer for SuperSighurt LLM.

use anyhow::Result;
use rust_fun::dialogs::Data;
use std::io::Write as _;
use rust_fun::gpu::Gpu;
use rust_fun::memory;
use rust_fun::neural_network::{
    CONTEXT_WINDOW, EMBED_DIM, HIDDEN_SIZE, NUMBER_OF_HIDDEN_LAYERS, extract_train_examples,
    generate, network_init, train_one_epoch,
};
use rust_fun::persist::{self, LoadedShape};
use std::time::Instant;

const MODEL_PATH: &str = "model.bin";

/// Read a f64 from a one-line text file. Returns None on missing file,
/// I/O error, or parse failure. Used by the best-val tracking to
/// resume the running "best so far" across training sessions instead
/// of resetting to infinity at every `main()` invocation.
fn read_best_val_sidecar(path: &str) -> Option<f64> {
    let s = std::fs::read_to_string(path).ok()?;
    s.trim().parse::<f64>().ok()
}

/// Compute the LR for a given 1-indexed epoch under the
/// warmup + cosine-anneal schedule (or the legacy multiplicative decay
/// when `anneal_epochs == 0`).
///
/// Phases:
/// 1. **Warmup**: `epoch ∈ [1, warmup_epochs]` → linear ramp
///    `lr_peak * epoch / warmup_epochs`. Skipped if `warmup_epochs == 0`.
/// 2. **Cosine anneal**: counted from the *end* of warmup. Over
///    `anneal_epochs` epochs lr cosine-descends from `lr_peak` to
///    `lr_min`. Standard half-cosine: at progress=0 returns lr_peak,
///    at progress=1 returns lr_min.
/// 3. **Tail**: after anneal completes, stays at `lr_min`.
///
/// When `anneal_epochs == 0` (the legacy path), the trainer's own
/// multiplicative `--lr-decay` is used outside this function.
fn lr_for_epoch(
    epoch: usize,
    lr_peak: f32,
    lr_min: f32,
    warmup_epochs: usize,
    anneal_epochs: usize,
) -> f32 {
    if anneal_epochs == 0 {
        // Caller handles the legacy multiplicative decay path.
        return lr_peak;
    }
    if warmup_epochs > 0 && epoch <= warmup_epochs {
        // Warmup is open-on-the-left: epoch 1 starts at lr_peak/warmup
        // (so we never start with lr = 0), epoch == warmup_epochs ends
        // exactly at lr_peak.
        let frac = epoch as f32 / warmup_epochs as f32;
        return lr_peak * frac;
    }
    let post_warmup = epoch - warmup_epochs;
    if post_warmup >= anneal_epochs {
        return lr_min;
    }
    let progress = post_warmup as f32 / anneal_epochs as f32;
    let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    lr_min + (lr_peak - lr_min) * cos_factor
}

/// Write a f64 atomically to a text file (tmp + rename). One line,
/// trailing newline, full f64 precision.
fn write_best_val_sidecar(path: &str, val: f64) -> std::io::Result<()> {
    let tmp = format!("{}.tmp", path);
    {
        let mut f = std::fs::File::create(&tmp)?;
        writeln!(f, "{}", val)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

struct Args {
    epochs: usize,
    lr: f32,
    save_every: usize,
    prelude_drop: f32,
    sample_every: usize,
    lr_decay: f32,
    val_frac: f32,
    /// Optional cap on the training pool size, applied AFTER the train/val
    /// split. Useful for short benchmark runs (e.g. `--max-train-examples 50`
    /// finishes a "fake epoch" in seconds for timing).
    max_train_examples: Option<usize>,
    /// Optional cap on the validation pool size, applied AFTER the split.
    /// Same purpose as `--max-train-examples` but for the validation pass.
    max_val_examples: Option<usize>,
    /// Linear warmup over the first N epochs: epoch e (1-indexed) uses
    /// `lr * (e / lr_warmup_epochs)` until it reaches the requested
    /// peak. 0 disables warmup (the legacy behavior). Useful because
    /// Adam's first few updates from cold moments tend to overshoot;
    /// a 1–3 epoch warmup at this scale is enough.
    lr_warmup_epochs: usize,
    /// Cosine-anneal lr from peak to `lr_min` over `lr_anneal_epochs`
    /// (counted from the *end* of warmup). 0 disables the cosine path
    /// and falls back to the legacy `--lr-decay` per-epoch multiplier.
    /// When non-zero, `--lr-decay` is ignored.
    lr_anneal_epochs: usize,
    /// Floor of the cosine anneal (the LR we asymptote toward). Has
    /// no effect when `lr_anneal_epochs == 0`. After anneal completes,
    /// LR stays at `lr_min` for the rest of the run.
    lr_min: f32,
    /// Dropout rate on hidden-layer activations during training. 0
    /// disables (legacy behavior). 0.1–0.3 typical for over-fit
    /// MLPs. Always disabled in the validation pass.
    dropout: f32,
    /// Label smoothing α applied to cross-entropy targets. 0
    /// disables. 0.1 is a typical LM value; sometimes 0.05 works
    /// better when the corpus is small (avoids over-flattening the
    /// learning signal).
    label_smoothing: f32,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            epochs: 50,
            lr: 0.05,
            save_every: 1,
            prelude_drop: 0.3,
            sample_every: 5,
            lr_decay: 0.985,
            val_frac: 0.1,
            max_train_examples: None,
            max_val_examples: None,
            lr_warmup_epochs: 0,
            lr_anneal_epochs: 0,
            lr_min: 0.0,
            dropout: 0.0,
            label_smoothing: 0.0,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--epochs" => a.epochs = it.next().unwrap().parse().unwrap(),
                "--lr" => a.lr = it.next().unwrap().parse().unwrap(),
                "--save-every" => a.save_every = it.next().unwrap().parse().unwrap(),
                "--prelude-drop" => a.prelude_drop = it.next().unwrap().parse().unwrap(),
                "--sample-every" => a.sample_every = it.next().unwrap().parse().unwrap(),
                "--lr-decay" => a.lr_decay = it.next().unwrap().parse().unwrap(),
                "--val-frac" => a.val_frac = it.next().unwrap().parse().unwrap(),
                "--max-train-examples" => {
                    a.max_train_examples = Some(it.next().unwrap().parse().unwrap())
                }
                "--max-val-examples" => {
                    a.max_val_examples = Some(it.next().unwrap().parse().unwrap())
                }
                "--lr-warmup-epochs" => {
                    a.lr_warmup_epochs = it.next().unwrap().parse().unwrap()
                }
                "--lr-anneal-epochs" => {
                    a.lr_anneal_epochs = it.next().unwrap().parse().unwrap()
                }
                "--lr-min" => a.lr_min = it.next().unwrap().parse().unwrap(),
                "--dropout" => a.dropout = it.next().unwrap().parse().unwrap(),
                "--label-smoothing" => a.label_smoothing = it.next().unwrap().parse().unwrap(),
                "--help" | "-h" => {
                    println!(
                        "train [--epochs N] [--lr F] [--save-every N] [--prelude-drop F] [--sample-every N] [--lr-decay F] [--val-frac F] [--max-train-examples N] [--max-val-examples N] [--lr-warmup-epochs N] [--lr-anneal-epochs N] [--lr-min F] [--dropout F] [--label-smoothing F]"
                    );
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown flag: {other}");
                    std::process::exit(2);
                }
            }
        }
        a
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Default to CPU. The 2026-05 SIGHURT_TIME_STEPS profiling showed the
    // CPU rayon matmul beats Vulkan by ~5.5× per step on this model size:
    //   GPU:  fwd 13.5ms  back 8.5ms   adam 2.1ms   = 24.2 ms/step
    //   CPU:  fwd 1.1ms   back 2.2ms   adam 1.6ms   =  4.9 ms/step
    // The hidden layers are only 768×768 and Vulkan dispatch overhead
    // (~2-3 ms per matvec call, × 5 layers × per-token) dominates the
    // actual matmul math. GPU only pays off once we batch matmuls big
    // enough to amortize dispatch — i.e. once mini-batch training lands.
    // Opt back into GPU with `SIGHURT_TRAIN_GPU=1` for experiments.
    let force_gpu = std::env::var("SIGHURT_TRAIN_GPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let gpu = if force_gpu {
        match Gpu::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("GPU init failed ({}); falling back to CPU.", e);
                Gpu::new_cpu()
            }
        }
    } else {
        Gpu::new_cpu()
    };
    // Legacy: SIGHURT_TRAIN_CPU=1 used to mean "force CPU when GPU is
    // default". CPU is now the default, so the variable is moot. We
    // intentionally do not warn — the new default does what the old
    // override did.
    println!("Auto-trainer starting. Backend: {}", gpu.device_name());

    let dialog = Data::load()?;
    let vocab = dialog.build_vocab();
    memory::save_vocab(&vocab);
    println!(
        "Vocab size: {} (max PERSON id: {})",
        vocab.len(),
        dialog.max_person_id().map(|n| n as i64).unwrap_or(-1)
    );

    let vocab_hash = persist::compute_vocab_hash(&vocab);
    let shape = LoadedShape {
        embed_dim: EMBED_DIM,
        context_window: CONTEXT_WINDOW,
        vocab_size: vocab.len(),
        hidden_size: HIDDEN_SIZE,
        hidden_layers: NUMBER_OF_HIDDEN_LAYERS,
        vocab_hash,
    };

    let mut net = match persist::load_with_vocab(MODEL_PATH, &gpu, shape, Some(&vocab)) {
        Ok(Some(n)) => {
            println!("Loaded existing model from {MODEL_PATH}; continuing training.");
            n
        }
        Ok(None) => {
            println!("No model at {MODEL_PATH}; initializing fresh network.");
            network_init(
                &gpu,
                EMBED_DIM,
                CONTEXT_WINDOW,
                HIDDEN_SIZE,
                NUMBER_OF_HIDDEN_LAYERS,
                vocab.len(),
            )?
        }
        Err(e) => {
            println!("Saved model unusable ({e}); initializing fresh.");
            network_init(
                &gpu,
                EMBED_DIM,
                CONTEXT_WINDOW,
                HIDDEN_SIZE,
                NUMBER_OF_HIDDEN_LAYERS,
                vocab.len(),
            )?
        }
    };

    let mut examples = extract_train_examples(&dialog);
    let total = examples.len();
    if total < 2 {
        anyhow::bail!(
            "corpus has only {} training example(s); need at least 2 for train/val split",
            total
        );
    }
    let val_n = ((total as f32) * args.val_frac).round() as usize;
    let val_n = val_n.max(1).min(total.saturating_sub(1));
    let mut val_examples: Vec<_> = examples.split_off(total - val_n);
    if let Some(cap) = args.max_train_examples {
        examples.truncate(cap);
    }
    if let Some(cap) = args.max_val_examples {
        val_examples.truncate(cap);
    }
    println!(
        "Extracted {} examples total -> {} train / {} val.",
        total,
        examples.len(),
        val_examples.len()
    );

    let sample_prompts = [
        "Hey there!",
        "Can you explain what an LLM is?",
        "I'm building a chatbot in Rust.",
        "How do I make my bot smarter?",
    ];

    let mut lr = args.lr;
    let t_start = Instant::now();
    // Track the best val we've seen and stash a separate `model.bin.best`
    // snapshot whenever it improves. `model.bin` keeps the *latest*
    // weights (what the auto-reload watcher serves); `model.bin.best`
    // is a manual rollback target the user can `cp` over `model.bin`
    // if a later epoch regresses.
    //
    // The "best so far" is persisted in a sidecar `model.bin.best.val`
    // text file. Without that, every new training session starts with
    // best_val = +infinity and immediately overwrites the prior .best
    // snapshot on its first epoch — losing the work of any earlier
    // session even when its val was lower. If the user changes the
    // corpus (so val distributions are no longer comparable) they
    // should delete the sidecar manually.
    let best_path = format!("{}.best", MODEL_PATH);
    let best_val_sidecar = format!("{}.val", best_path);
    let mut best_val = match read_best_val_sidecar(&best_val_sidecar) {
        Some(v) => {
            println!(
                "Resuming best-val tracking from {} = {:.4}",
                best_val_sidecar, v
            );
            v
        }
        None => f64::INFINITY,
    };
    // Wire dropout into the network from the CLI. The field is
    // runtime-only — every fresh `train` invocation re-sets it.
    net.dropout_p = args.dropout;
    if args.dropout > 0.0 {
        println!("Dropout enabled: p = {:.2}", args.dropout);
    }
    net.label_smoothing = args.label_smoothing;
    if args.label_smoothing > 0.0 {
        println!("Label smoothing enabled: α = {:.3}", args.label_smoothing);
    }
    if args.lr_anneal_epochs > 0 {
        println!(
            "LR schedule: warmup {} epochs → cosine over {} epochs to lr_min={:.6}",
            args.lr_warmup_epochs, args.lr_anneal_epochs, args.lr_min
        );
    }
    for epoch in 1..=args.epochs {
        // Set per-epoch LR via schedule when cosine is enabled;
        // otherwise apply the legacy multiplicative decay at the
        // end of each epoch (preserved below).
        if args.lr_anneal_epochs > 0 {
            lr = lr_for_epoch(
                epoch,
                args.lr,
                args.lr_min,
                args.lr_warmup_epochs,
                args.lr_anneal_epochs,
            );
        }
        let t0 = Instant::now();
        let stats = train_one_epoch(
            &gpu,
            &mut net,
            &mut examples,
            &val_examples,
            &vocab,
            lr,
            args.prelude_drop,
        )?;
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "epoch {:>3}/{:<3}  lr={:.5}  train={:.4}  val={:.4}  ({}t/{}v)  dt={:.2}s  total={:.1}s",
            epoch,
            args.epochs,
            lr,
            stats.train_loss,
            stats.val_loss,
            stats.train_targets,
            stats.val_targets,
            dt,
            t_start.elapsed().as_secs_f64()
        );

        if epoch % args.save_every == 0 {
            persist::save(&net, MODEL_PATH, vocab_hash)?;
        }

        // Best-val tracking: write `model.bin.best` whenever we hit a new
        // global-low validation loss. `best_val` was loaded from the
        // sidecar at startup if a previous session left one behind.
        // Skip when val_targets is 0 (unlikely but defensive — the loss
        // would be 0 / 0 = NaN-like).
        if stats.val_targets > 0 && stats.val_loss < best_val {
            best_val = stats.val_loss;
            if let Err(e) = persist::save(&net, &best_path, vocab_hash) {
                eprintln!("  warn: failed to write {}: {}", best_path, e);
            } else {
                if let Err(e) = write_best_val_sidecar(&best_val_sidecar, best_val) {
                    eprintln!("  warn: failed to write {}: {}", best_val_sidecar, e);
                }
                println!("  new best val {:.4} → snapshotted to {}", best_val, best_path);
            }
        }

        if args.sample_every > 0 && epoch % args.sample_every == 0 {
            // Disable dropout for the sample generation so we see
            // the model's actual inference output, not a stochastic
            // dropout-affected one. `generate` calls `net.forward`
            // which doesn't apply dropout anyway, but be explicit.
            let saved = net.dropout_p;
            net.dropout_p = 0.0;
            let prompt = sample_prompts[epoch % sample_prompts.len()];
            match generate(&gpu, &mut net, prompt, &[], &vocab) {
                Ok(s) => println!("  sample> You: {prompt}\n  sample> Bot: {s}"),
                Err(e) => println!("  sample failed: {e}"),
            }
            net.dropout_p = saved;
        }

        if args.lr_anneal_epochs == 0 {
            // Legacy multiplicative decay path. Only active when the
            // cosine schedule isn't requested.
            lr *= args.lr_decay;
        }
    }

    persist::save(&net, MODEL_PATH, vocab_hash)?;
    println!(
        "Done. Final model saved to {MODEL_PATH}. Total wall time {:.1}s.",
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn lr_schedule_disabled_returns_peak() {
        // anneal_epochs = 0 → caller handles legacy decay, helper
        // just returns lr_peak as-is.
        for e in 1..10 {
            assert!(approx(lr_for_epoch(e, 0.001, 0.0, 0, 0), 0.001));
        }
    }

    #[test]
    fn lr_schedule_warmup_is_linear() {
        // 3-epoch warmup from 0 → lr_peak. Epoch 1 should be 1/3,
        // epoch 2 should be 2/3, epoch 3 should hit lr_peak exactly.
        let peak = 0.0003;
        assert!(approx(lr_for_epoch(1, peak, 0.0, 3, 10), peak / 3.0));
        assert!(approx(lr_for_epoch(2, peak, 0.0, 3, 10), peak * 2.0 / 3.0));
        assert!(approx(lr_for_epoch(3, peak, 0.0, 3, 10), peak));
    }

    #[test]
    fn lr_schedule_cosine_endpoints_match_convention() {
        // Convention: `post_warmup = epoch - warmup_epochs` counts
        // 1-indexed cosine steps. So epoch=warmup_epochs+1 has
        // progress = 1/anneal_epochs (already descended slightly
        // from the peak; we hit peak exactly at the *end* of warmup,
        // not at the start of cosine). At epoch=warmup+anneal,
        // progress = 1.0 and we return lr_min.
        let peak = 0.0003;
        let lr_min = 0.00001;
        // Midpoint check: post_warmup=5 of 10 → progress=0.5,
        // cos(π/2)=0 → factor=0.5 → lr is exactly midway between
        // lr_min and peak.
        let mid = lr_min + (peak - lr_min) * 0.5;
        assert!(approx(lr_for_epoch(5, peak, lr_min, 0, 10), mid));
        // End-of-cosine: post_warmup=10 hits the anneal_epochs
        // guard → lr_min exactly.
        assert!(approx(lr_for_epoch(10, peak, lr_min, 0, 10), lr_min));
    }

    #[test]
    fn lr_schedule_tail_stays_at_lr_min() {
        // Past the anneal window, lr stays pinned at lr_min.
        let peak = 0.0003;
        let lr_min = 0.00002;
        for e in 11..30 {
            assert!(approx(
                lr_for_epoch(e, peak, lr_min, 0, 10),
                lr_min,
            ));
        }
    }

    #[test]
    fn lr_schedule_warmup_then_cosine_compose() {
        // 2-epoch warmup then 4-epoch cosine. Warmup endpoints:
        // epoch 1 → peak/2, epoch 2 → peak. Cosine then runs
        // post_warmup = 1..4. Epoch 6 (post_warmup=4) → lr_min.
        let peak = 0.0002;
        let lr_min = 0.0;
        assert!(approx(lr_for_epoch(1, peak, lr_min, 2, 4), peak / 2.0));
        assert!(approx(lr_for_epoch(2, peak, lr_min, 2, 4), peak));
        // Epoch 6 is post_warmup=4=anneal → lr_min (guard branch).
        assert!(approx(lr_for_epoch(6, peak, lr_min, 2, 4), lr_min));
        // Tail.
        assert!(approx(lr_for_epoch(7, peak, lr_min, 2, 4), lr_min));
    }
}
