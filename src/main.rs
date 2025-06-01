mod neural_net;
mod preprocess;
mod data;

use data::load_data;
use neural_net::NeuralNetwork;
use preprocess::TextProcessor;
use std::error::Error;

const MAX_WORDS: usize = 1000;
const HIDDEN_LAYER_SIZE: usize = 16;
const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 10;

fn main() -> Result<(), Box<dyn Error>> {
    let comments = load_data("data/train.csv")?;
    
    let mut processor = TextProcessor::new(MAX_WORDS);
    
    let mut nn = NeuralNetwork::new(MAX_WORDS, HIDDEN_LAYER_SIZE, 1, LEARNING_RATE);
    
    println!("Training network...");
    for epoch in 0..EPOCHS {
        let mut total_error = 0.0;
        
        for comment in &comments {
            let input = processor.process_text(&comment.comment_text);
            
            let target = vec![comment.toxic];
            
            nn.train(&input, &target);
            
            let output = nn.feedforward(&input);
            total_error += (output[0] - target[0]).powi(2);
        }
        
        println!("Epoch {}: MSE = {}", epoch, total_error / comments.len() as f64);
    }
    
    println!("\nTesting network...");
    let test_comments = [
        "You're an idiot and I hate you!",
        "Thanks for your help, I appreciate it.",
        "Go die in a hole!",
        "Could you please explain this to me?",
    ];
    
    for comment in test_comments {
        let input = processor.process_text(comment);
        let output = nn.feedforward(&input);
        println!("Comment: {}", comment);
        println!("Toxicity probability: {:.2}%", output[0] * 100.0);
        println!();
    }
    
    Ok(())
}