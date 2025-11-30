pub fn load() -> Vec<String> {
    std::fs::read_to_string("vocab.txt").lines().map(|s| s.to_string()).collect()
}
pub fn append(text: &str){
    let lines: Vec<String> = load();
    for i in 0..lines.size(){
        if text.to_string() == lines[i]{
            return;
        }
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open("vocab.txt").unwrap();
    writeln!(f, "{}", text).unwrap();
}
