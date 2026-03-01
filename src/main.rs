mod audio;
mod decoder;
mod encoder;
mod model;
mod preset;
mod tokenizer;
mod transcribe;

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model_dir> <wav_file>", args[0]);
        std::process::exit(1);
    }
    let model_dir = PathBuf::from(&args[1]);
    let wav_path  = PathBuf::from(&args[2]);

    eprintln!("Loading model from {}...", model_dir.display());
    let mut pipeline = transcribe::Pipeline::load(&model_dir)
        .expect("failed to load model");

    eprintln!("Transcribing {}...", wav_path.display());
    let text = pipeline.transcribe(&wav_path)
        .expect("transcription failed");

    println!("{text}");
}
