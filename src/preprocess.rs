use regex::Regex;
use std::collections::HashMap;

pub struct TextProcessor {
    word_index: HashMap<String, usize>,
    next_index: usize,
    max_words: usize,
}

impl TextProcessor {
    pub fn new(max_words: usize) -> Self {
        TextProcessor {
            word_index: HashMap::new(),
            next_index: 0,
            max_words,
        }
    }

    pub fn process_text(&mut self, text: &str) -> Vec<f64> {
        let cleaned = self.clean_text(text);
        let tokens = self.tokenize(&cleaned);

        let mut vector = vec![0.0; self.max_words];

        for token in tokens {
            let index = self.word_index.entry(token).or_insert_with(|| {
                let current = self.next_index;
                self.next_index += 1;
                current
            });

            if *index < self.max_words {
                vector[*index] += 1.0;
            }
        }

        let sum: f64 = vector.iter().sum();
        if sum > 0.0 {
            for item in &mut vector {
                *item /= sum;
            }
        }

        vector
    }

    fn clean_text(&self, text: &str) -> String {
        let re = Regex::new(r"[^a-zA-Z0-9\s]").unwrap();
        let cleaned = re.replace_all(text, "").to_lowercase();
        cleaned.to_string()
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}