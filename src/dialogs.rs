use std::fs;
use crate::memory::{append};
// I want to read a file. Match user input with a part of the according section. I also want to
// keep track of what section it is. I also want a functionality that makes it possible to
// automatically add new words from the dialogs into the vocaby, this feature should be toggleable
// for cases where performance is essential and when all words are presumed to be known. In short I
// want to input what the user wrote (keeping track of sections is done internally) and get the bot
// response back. Correct being the response that is paired with the most similar dialog entry
// (user input).
#[derive(Clone)]
pub enum Text {
    User(String),
    Bot(String),
}

#[derive(Clone)]
pub struct Data {
    // Each outer vec is one section and the inner is a vec of words in the section
    pub Sections: Vec<Vec<Text>>,
}

impl Data {
    pub fn load(&mut self) {
        let data = fs::read_to_string("data/dialogs.txt").expect("failed to read file");
        let mut loading_user: bool = false;
        let mut loading_bot: bool = false;
        let mut current_section: usize = 0;
        for word in data.split_whitespace() {
            append(&word);
            if loading_user {
                if word == "</USER>" { loading_user = false; continue; }
                self.Sections[current_section].push(Text::User(word.to_string()));
            }
            else if loading_bot {
                if word == "</BOT>" { loading_bot = false; continue; }
                self.Sections[current_section].push(Text::Bot(word.to_string()));
            }

            else if word == "<SEC>" { 
                let new_current_section: Vec<Text> = Vec::new();
                self.Sections.push(new_current_section);
                current_section += 1;
            }
            else if word == "<USER>" { loading_user = true; }
            else if word == "<BOT>" { loading_bot = true; }
        }
    }
}

