mod dialogs;
mod embeddings;
mod gpu;
mod machine_learning;
mod memory;
mod neural_network;
mod teacher;

use crate::dialogs::Data;
use crate::gpu::Gpu;
use crate::neural_network::{generate, generate_and_train, network_init};
use std::io::{self, Write};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let train: bool = true;
    let mut output = String::new();
    let mut input = String::new();
    let mut talking = true;
    let mut bot_memory: Vec<String> = Vec::new();

    println!("Hey, welcome to SuperSighurt LLM-mode!");

    let gpu = Gpu::new()?;
    println!("GPU backend: {}", gpu.ctx.device_name());

    // Dialogs first: loading them can extend vocab.txt with new words. Then snapshot
    // the vocab so the network's output size matches what we will ever index into.
    let mut dialog: Data = Data::new();
    dialog.load();
    let words_in_vocab = memory::load();
    println!("Vocab size: {}", words_in_vocab.len());

    let hidden_size: usize = 10000;
    let output_size: usize = words_in_vocab.len();
    let mut net = network_init(&gpu, hidden_size, output_size)?;
    let lr: f32 = 0.1;

    while talking {
        if !output.trim().is_empty() {
            println!("Sighurt: {}", output);
            output.clear();
        } else {
            print!("You: ");
            io::stdout().flush().unwrap();

            input.clear();
            std::io::stdin()
                .read_line(&mut input)
                .expect("failed to read");
            input = input.trim().to_string();

            if input == ":q" {
                talking = false;
                continue;
            }
            bot_memory.push(input.clone());

            let start = Instant::now();
            let sentence = if train {
                generate_and_train(
                    &gpu,
                    &mut net,
                    &input,
                    &bot_memory,
                    &words_in_vocab,
                    &dialog,
                    lr,
                )?
            } else {
                generate(&gpu, &mut net, &input, &bot_memory, &words_in_vocab)?
            };
            println!(
                "Time to get answer in seconds: {:?}",
                start.elapsed().as_secs_f64()
            );
            output = sentence;
        }
    }

    Ok(())
}
