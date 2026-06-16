use rust_fun::dialogs::Data;
use rust_fun::gpu::Gpu;
use rust_fun::memory;
use rust_fun::neural_network::{
    CONTEXT_WINDOW, EMBED_DIM, HIDDEN_SIZE, NUMBER_OF_HIDDEN_LAYERS, generate, generate_and_train,
    network_init, pretrain,
};
use rust_fun::persist::{self, LoadedShape};
use std::io::{self, Write};
use std::time::Instant;

const MODEL_PATH: &str = "model.bin";
const PRETRAIN_EPOCHS: usize = 3;
const PRETRAIN_LR: f32 = 0.05;
/// Per-turn LR when interactive training is enabled. Deliberately very
/// small: a chat turn is a batch of 1 against a teacher response of
/// unverified quality, and any cumulative drift goes straight into
/// the live `model.bin`. Keep it ≪ the offline-trainer LR (currently
/// 0.0003) — a few hundred turns at 0.0001 should nudge the model,
/// not rewrite it.
const ONLINE_LR: f32 = 0.0001;
const SAVE_EVERY_N_TURNS: usize = 5;

fn main() -> anyhow::Result<()> {
    let mut talking = true;
    let mut bot_memory: Vec<String> = Vec::new();

    println!("Hey, welcome to SuperSighurt LLM-mode!");

    let gpu = Gpu::new()?;
    println!("Backend: {}", gpu.device_name());

    let dialog: Data = Data::load()?;
    let vocab = dialog.build_vocab();
    memory::save_vocab(&vocab);
    println!("Vocab size: {}", vocab.len());

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
            println!("Loaded saved model from {}", MODEL_PATH);
            n
        }
        Ok(None) => {
            println!(
                "No saved model at {}. Fresh init (embed={}, ctx={}, hidden={}x{}).",
                MODEL_PATH, EMBED_DIM, CONTEXT_WINDOW, NUMBER_OF_HIDDEN_LAYERS, HIDDEN_SIZE
            );
            let mut net = network_init(
                &gpu,
                EMBED_DIM,
                CONTEXT_WINDOW,
                HIDDEN_SIZE,
                NUMBER_OF_HIDDEN_LAYERS,
                vocab.len(),
            )?;
            println!("Pretraining on dialog corpus ({} epochs)...", PRETRAIN_EPOCHS);
            let t0 = Instant::now();
            pretrain(&gpu, &mut net, &dialog, &vocab, PRETRAIN_LR, PRETRAIN_EPOCHS)?;
            println!(
                "Pretraining done in {:.2}s. Saving initial weights.",
                t0.elapsed().as_secs_f64()
            );
            persist::save(&net, MODEL_PATH, vocab_hash)?;
            net
        }
        Err(e) => {
            println!(
                "Could not load saved model ({}). Initializing fresh and pretraining.",
                e
            );
            let mut net = network_init(
                &gpu,
                EMBED_DIM,
                CONTEXT_WINDOW,
                HIDDEN_SIZE,
                NUMBER_OF_HIDDEN_LAYERS,
                vocab.len(),
            )?;
            let t0 = Instant::now();
            pretrain(&gpu, &mut net, &dialog, &vocab, PRETRAIN_LR, PRETRAIN_EPOCHS)?;
            println!(
                "Pretraining done in {:.2}s. Saving initial weights.",
                t0.elapsed().as_secs_f64()
            );
            persist::save(&net, MODEL_PATH, vocab_hash)?;
            net
        }
    };

    let mut turns_since_save = 0usize;
    let mut output = String::new();
    let mut input = String::new();

    // Online training defaults OFF. The previous default (on, at LR 0.02)
    // could drift the model materially in a handful of turns, and a chat
    // session is a noisy training signal compared to the supervised
    // trainer. Opt in per session with `:train on`.
    println!("Type ':q' to quit, ':save' to checkpoint, ':train on|off' to toggle online learning (off by default).");
    let mut train_active = false;

    while talking {
        if !output.trim().is_empty() {
            println!("Sighurt: {}", output);
            output.clear();
        } else {
            print!("You: ");
            io::stdout().flush().unwrap();

            input.clear();
            std::io::stdin().read_line(&mut input).expect("failed to read");
            input = input.trim().to_string();

            if input == ":q" {
                talking = false;
                continue;
            }
            if input == ":save" {
                persist::save(&net, MODEL_PATH, vocab_hash)?;
                println!("[saved to {}]", MODEL_PATH);
                continue;
            }
            if input == ":train off" {
                train_active = false;
                println!("[training disabled]");
                continue;
            }
            if input == ":train on" {
                train_active = true;
                println!("[training enabled]");
                continue;
            }
            if input.is_empty() {
                continue;
            }

            let start = Instant::now();
            let sentence = if train_active {
                generate_and_train(
                    &gpu,
                    &mut net,
                    &input,
                    &bot_memory,
                    &vocab,
                    &dialog,
                    ONLINE_LR,
                )?
            } else {
                generate(&gpu, &mut net, &input, &bot_memory, &vocab)?
            };
            println!(
                "Time to get answer in seconds: {:.3}",
                start.elapsed().as_secs_f64()
            );
            output = sentence.clone();
            bot_memory.push(input.clone());
            bot_memory.push(sentence);

            turns_since_save += 1;
            if train_active && turns_since_save >= SAVE_EVERY_N_TURNS {
                persist::save(&net, MODEL_PATH, vocab_hash)?;
                turns_since_save = 0;
            }
        }
    }

    persist::save(&net, MODEL_PATH, vocab_hash)?;
    println!("Model saved to {}.", MODEL_PATH);
    Ok(())
}
