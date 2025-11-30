mod neural_network;
use crate::neural_network::{interpret_input, network_init, generate};
use std::io::{self, Write};

fn main(){
    let mut memory: Vec<String> = Vec::with_capacity(100);
    let mut output = String::new();
    let mut input = String::new();
    let mut talking = true;
    println!("Hey, have fun talking to this dumb bot");

    let input_size: usize = 4;
    let hidden_size: usize = 15;
    let output_size: usize = 5;
    let net = network_init(input_size, hidden_size, output_size);
    while talking {
        if !output.trim().is_empty(){
            println!("Bot: {}", output);
            output.clear();
        }
        else{ 
            print!("You: ");
            io::stdout().flush();
            if input == ":q" {
                talking = false;
            } else if input == ":mem" {
                println!("{:?}", memory);
            }
            input.clear();
            
            std::io::stdin().read_line(&mut input).expect("failed to read");
            input = input.trim().to_string();
            memory.push(input.clone());

            let features = interpret_input(&input, &memory);
            let outputs = net.forward(features);
            let score = outputs[0];
            output = format!("Score: {:.3}", score);
        }
    }
}
