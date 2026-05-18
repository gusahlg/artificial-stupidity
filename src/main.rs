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
const ONLINE_LR: f32 = 0.02;
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

    let shape = LoadedShape {
        embed_dim: EMBED_DIM,
        context_window: CONTEXT_WINDOW,
        vocab_size: vocab.len(),
        hidden_size: HIDDEN_SIZE,
        hidden_layers: NUMBER_OF_HIDDEN_LAYERS,
    };

    let mut net = match persist::load(MODEL_PATH, &gpu, shape) {
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
            persist::save(&net, MODEL_PATH)?;
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
            persist::save(&net, MODEL_PATH)?;
            net
        }
    };

    let mut turns_since_save = 0usize;
    let mut output = String::new();
    let mut input = String::new();

    println!("Type ':q' to quit, ':save' to checkpoint, ':train off|on' to toggle.");
    let mut train_active = true;

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
                persist::save(&net, MODEL_PATH)?;
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
                persist::save(&net, MODEL_PATH)?;
                turns_since_save = 0;
            }
        }
    }

    persist::save(&net, MODEL_PATH)?;
    println!("Model saved to {}.", MODEL_PATH);
    Ok(())
}
