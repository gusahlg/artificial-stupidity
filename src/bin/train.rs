//! Standalone auto-trainer for the SuperSighurt LLM.
//!
//! Iterates supervised epochs over the dialog corpus with shuffling and
//! random prelude-dropout, logs cross-entropy per epoch, periodically saves
//! the model, and prints a sample generation so progress is visible.
//!
//! Usage:
//!   train [--epochs N] [--lr 0.05] [--save-every N] [--prelude-drop 0.3] [--sample-every N]

use anyhow::Result;
use rust_fun::dialogs::Data;
use rust_fun::gpu::Gpu;
use rust_fun::memory;
use rust_fun::neural_network::{
    HIDDEN_SIZE, NUMBER_OF_HIDDEN_LAYERS, extract_train_pairs, generate, input_size_for,
    network_init, train_one_epoch,
};
use rust_fun::persist::{self, LoadedShape};
use std::time::Instant;

const MODEL_PATH: &str = "model.bin";

struct Args {
    epochs: usize,
    lr: f32,
    save_every: usize,
    prelude_drop: f32,
    sample_every: usize,
    lr_decay: f32,
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
                "--help" | "-h" => {
                    println!(
                        "train [--epochs N] [--lr F] [--save-every N] [--prelude-drop F] [--sample-every N] [--lr-decay F]"
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

    let gpu = Gpu::new()?;
    println!("Auto-trainer starting. Backend: {}", gpu.device_name());

    let mut dialog = Data::new();
    dialog.load();
    let vocab = memory::load();
    println!("Vocab size: {}", vocab.len());

    let hidden_size = HIDDEN_SIZE;
    let hidden_layers = NUMBER_OF_HIDDEN_LAYERS;
    let output_size = vocab.len();
    let input_size = input_size_for(vocab.len());

    let shape = LoadedShape {
        input_size,
        vocab_size: output_size,
        hidden_size,
        hidden_layers,
    };

    let mut net = match persist::load(MODEL_PATH, &gpu, shape) {
        Ok(Some(n)) => {
            println!("Loaded existing model from {MODEL_PATH}; continuing training.");
            n
        }
        Ok(None) => {
            println!("No model at {MODEL_PATH}; initializing fresh network.");
            network_init(&gpu, input_size, hidden_size, hidden_layers, output_size)?
        }
        Err(e) => {
            println!("Saved model unusable ({e}); initializing fresh.");
            network_init(&gpu, input_size, hidden_size, hidden_layers, output_size)?
        }
    };

    let mut pairs = extract_train_pairs(&dialog);
    println!(
        "Extracted {} (user, bot) training pairs from corpus.",
        pairs.len()
    );

    let sample_prompts = [
        "Hey there!",
        "Can you explain what an LLM is?",
        "I'm building a chatbot in Rust.",
        "How do I make my bot smarter?",
    ];

    let mut lr = args.lr;
    let t_start = Instant::now();
    for epoch in 1..=args.epochs {
        let t0 = Instant::now();
        let stats = train_one_epoch(&gpu, &mut net, &mut pairs, &vocab, lr, args.prelude_drop)?;
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "epoch {:>3}/{:<3}  lr={:.5}  xent={:.4}  targets={}  dt={:.2}s  total={:.1}s",
            epoch,
            args.epochs,
            lr,
            stats.avg_loss,
            stats.targets,
            dt,
            t_start.elapsed().as_secs_f64()
        );

        if epoch % args.save_every == 0 {
            persist::save(&net, MODEL_PATH)?;
        }

        if args.sample_every > 0 && epoch % args.sample_every == 0 {
            let prompt = sample_prompts[epoch % sample_prompts.len()];
            match generate(&gpu, &mut net, prompt, &[], &vocab) {
                Ok(s) => println!("  sample> You: {prompt}\n  sample> Bot: {s}"),
                Err(e) => println!("  sample failed: {e}"),
            }
        }

        lr *= args.lr_decay;
    }

    persist::save(&net, MODEL_PATH)?;
    println!(
        "Done. Final model saved to {MODEL_PATH}. Total wall time {:.1}s.",
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}
