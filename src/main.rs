mod memory;
mod neural_network;
mod machine_learning;
mod dialogs;
use crate::neural_network::{network_init, generate};
use crate::memory::{append};
use std::io::{self, Write};

fn main(){
    let mut output = String::new();
    let mut input = String::new();
    let mut talking = true;
    let mut bot_memory: Vec<String> = Vec::new();
    println!("Hey, welcome to SuperSighurt LLM-mode!");
     
    // let input_size: usize = 6;
    let hidden_size: usize = 750;
    let output_size: usize = memory::load().len();
    let net = network_init(hidden_size, output_size);
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

            let sentence = generate(&net, &input, &bot_memory);
            output = sentence;
        }
    }

}
