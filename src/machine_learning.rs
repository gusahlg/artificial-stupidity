// Recap: I want to use the previous happening in a section as something to match with the memory
// to know how similar the dialog has been so far. Then I want to use the user's latest question
// and with the help of memory and latest response find the most similar teacher dialog and then
// match each of the LLM's words in its next response with the words in the teacher response.
use crate::dialogs::{Data, Text};
// For now arbitrary way of determining how much two words resemble each other
fn string_similarity(word1: &str, word2: &str) -> u8 {
    let a = word1;
    let b = word2;

    let mut result: u8 = 0;
    if a == b { return 100; }
    
    let mut len_diff: u8 = 0;
    if a.len() > b.len() { len_diff = a.len() as u8 - b.len() as u8; }
    result += len_diff; 

    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca == cb { result += 1; }
    }

    result
}
fn index_of_most_similar_section(sections: &Vec<Vec<Text>>, memory: &Vec<&str>) -> usize {
    let mut scores: Vec<(u8, usize)> = Vec::with_capacity(sections.len());

    for (section_index, section) in sections.iter().enumerate() {
        let mut section_score: u8 = 0;
        let mut words_in_section: Vec<String> = Vec::new();

        // Goes through every word in every texts and adds to a vec
        for text in section {
            match text {
                Text::User(s) => {
                    for word in s.split_whitespace() {
                        words_in_section.push(word.to_string());
                    }
                }
                Text::Bot(_) => { }
            }
        }
        
        for (i, word) in words_in_section.iter().enumerate() {
            // will panic if memory[i] is out of bounds, fix is to make it break out of loop if
            // error
            if let Some(mem_word) = memory.get(i) {
                section_score += string_similarity(word.as_str(), mem_word);
            }
            else { break; }
        }

        //scores[section_index] = (section_score, section_index);
        scores.push(section_score, section_index);
    }
    
    // Go through all scores and find the one with the highest, 
    let mut highscore: u8 = 0;
    let mut idx_of_highest: usize = 0;
    for (sec_score, sec_idx) in scores {
        if sec_score > highscore { highscore = sec_score; idx_of_highest = sec_idx; }
    }
    idx_of_highest 
}
pub fn teacher_response(dialog: &Data, bot_memory: &Vec<&str>, user_input: &str) -> String {
    let index = index_of_most_similar_section(&dialog.Sections, &bot_memory);
    let section: Vec<Text> = dialog.Sections[index].clone();

    // I want to check what question by the user inside of the chosen section best matches the
    // current user input and then I want to return the next bot response from the function
    
    let mut highscore: u8 = 0;
    let mut idx_of_highest: usize = 0;
    for (index, text) in section.iter().enumerate() {
        match text {
            Text::User(s) => {
                let score = string_similarity(&s.as_str(), &user_input);
                if score > highscore { highscore = score; idx_of_highest = index; }
            }
            Text::Bot(_) => { continue; }
        }
    }
    
    // +1 Because I want to get the Bot response to the user question that is at idx_of_highest
    let teacher: Text = section[idx_of_highest + 1].clone();
    if let Text::Bot(s) = teacher {
        return s;
    }
    else { return String::new(); }
}
