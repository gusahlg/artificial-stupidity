use std::io::Write;

pub fn load() -> Vec<String> {
    std::fs::read_to_string("vocab.txt")
        .unwrap()
        .lines()
        .map(|s| s.to_string())
        .collect()
}

pub fn append(text: &str) {
    let lines = load();
    let mut words: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        if lines.iter().any(|line| line == words[i]) {
            words.remove(i);
        } else {
            i += 1;
        }
    }


    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("vocab.txt")
        .unwrap();

    for w in words{
        writeln!(f, "{}", w).unwrap();
    }
}

