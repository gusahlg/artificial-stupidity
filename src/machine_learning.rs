// Recap: I want to use the previous happening in a section as something to match with the memory
// to know how similar the dialog has been so far. Then I want to use the user's latest question
// and with the help of memory and latest response find the most similar teacher dialog and then
// match each of the LLM's words in its next response with the words in the teacher response.
use crate::dialogs::{Data, Text};
// For now arbitrary way of determining how much two words resemble each other
pub fn string_similarity(word1: &str, word2: &str) -> u8 {
    let mut result: u8 = 0;
    if word1 == word2 { return 100; }
    
    let mut len_diff: u8 = 0;
    if word1.len() > word2.len() {
        len_diff = (word1.len() - word2.len()) as u8;
    }
    result += len_diff; 

    for (cword1, cword2) in word1.chars().zip(word2.chars()) {
        if cword1 == cword2 { result += 1; }
    }

    result
}
fn index_of_most_similar_section(sections: &[Vec<Text>], memory: &[String]) -> usize {
    // u64: the old u8 silently wrapped after ~3 word comparisons in release mode,
    // turning section scoring into noise. Per-word similarity still returns u8.
    let mut scores: Vec<(u64, usize)> = Vec::with_capacity(sections.len());

    for (section_index, section) in sections.iter().enumerate() {
        let mut section_score: u64 = 0;
        let mut words_in_section: Vec<String> = Vec::new();

        for text in section {
            if let Text::User(s) = text {
                for word in s.split_whitespace() {
                    words_in_section.push(word.to_string());
                }
            }
        }

        for (i, word) in words_in_section.iter().enumerate() {
            if let Some(mem_word) = memory.get(i) {
                section_score += string_similarity(word.as_str(), mem_word) as u64;
            } else {
                break;
            }
        }

        scores.push((section_score, section_index));
    }

    let mut highscore: u64 = 0;
    let mut idx_of_highest: usize = 0;
    for (sec_score, sec_idx) in scores {
        if sec_score > highscore {
            highscore = sec_score;
            idx_of_highest = sec_idx;
        }
    }
    idx_of_highest
}
pub fn teacher_response(dialog: &Data, bot_memory: &[String], user_input: &str) -> String {
    // Skip empty sections so we never land on dialogs::load's leading empty section.
    let candidates: Vec<&Vec<Text>> = dialog.Sections.iter().filter(|s| !s.is_empty()).collect();
    if candidates.is_empty() {
        return String::new();
    }
    let sections_owned: Vec<Vec<Text>> = candidates.iter().map(|s| (*s).clone()).collect();
    let index = index_of_most_similar_section(&sections_owned, bot_memory);
    let section: &Vec<Text> = &sections_owned[index];

    let mut found = false;
    let mut highscore: u8 = 0;
    let mut idx_of_highest: usize = 0;
    for (index, text) in section.iter().enumerate() {
        if let Text::User(s) = text {
            let score = string_similarity(s.as_str(), user_input);
            if !found || score > highscore {
                highscore = score;
                idx_of_highest = index;
                found = true;
            }
        }
    }

    if !found || idx_of_highest + 1 >= section.len() {
        return String::new();
    }
    if let Text::Bot(s) = &section[idx_of_highest + 1] {
        s.clone()
    } else {
        String::new()
    }
}
