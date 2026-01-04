mod memory;
mod neural_network;
mod machine_learning;
mod dialogs;
mod teacher;
use crate::neural_network::{network_init, generate, generate_and_train};
use crate::memory::{append};
use std::io::{self, Write};
use crate::dialogs::{Data};
use std::time::Instant;

fn main(){
    let train: bool = true;
    let mut output = String::new();
    let mut input = String::new();
    let mut talking = true;
    let mut bot_memory: Vec<String> = Vec::new();
    println!("Hey, welcome to SuperSighurt LLM-mode!");
    let words_in_vocab = memory::load();

    // let input_size: usize = 6;
    let hidden_size: usize = 1500;
    let output_size: usize = words_in_vocab.len();
    let mut net = network_init(hidden_size, output_size);
    let mut dialog: Data = Data::new();
    dialog.load();
    let lr = 10f32;
    while talking {
        if !output.trim().is_empty() {
            println!("Sighurt: {}", output);
            output.clear();
        } 
        else {
            print!("You: ");
            io::stdout().flush().unwrap();

            input.clear();
            std::io::stdin().read_line(&mut input).expect("failed to read");
            input = input.trim().to_string();
            bot_memory.push(input.clone());
            append(&(&input as &str));

            if input == ":q" {
                talking = false;
                continue;
            }
            let mut sentence: String = String::new();

            // Performance data gatherer:
            let start = Instant::now();
            if train {
                sentence = generate_and_train(&mut net, &input, &bot_memory, words_in_vocab.clone(), &dialog, lr);
            }
            else {
                sentence = generate(&net, &input, &bot_memory, words_in_vocab.clone());
            }
            println!("Time to get answer in seconds: {:?}", start.elapsed().as_secs());
            output = sentence;
        }
    }

}
