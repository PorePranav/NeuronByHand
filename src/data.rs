use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ToxicComment {
    pub comment_text: String,
    pub toxic: f64,
}

pub fn load_data(file_path: &str) -> Result<Vec<ToxicComment>, Box<dyn Error>> {
    let file = File::open(Path::new(file_path))?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut comments = Vec::new();

    for result in rdr.deserialize() {
        let record: ToxicComment = result?;
        comments.push(record);
    }

    Ok(comments)
}