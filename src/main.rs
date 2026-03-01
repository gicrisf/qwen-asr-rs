mod audio;
mod decoder;
mod encoder;
mod model;
mod preset;
mod tokenizer;
mod transcribe;

use std::path::PathBuf;

use clap::Parser;

/// Qwen3-ASR speech-to-text inference (Rust port)
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Model directory (contains vocab.json, merges.txt, model.safetensors[.index.json])
    #[arg(short = 'd', long)]
    model_dir: PathBuf,

    /// Input WAV file
    #[arg(short = 'i', long)]
    input: PathBuf,

    /// Suppress progress output on stderr (transcription still printed to stdout)
    #[arg(long)]
    silent: bool,
}

fn main() {
    let args = Args::parse();

    if !args.silent {
        eprintln!("Loading model from {} ...", args.model_dir.display());
    }

    let mut pipeline = transcribe::Pipeline::load(&args.model_dir)
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });

    if !args.silent {
        eprintln!("Transcribing {} ...", args.input.display());
    }

    let text = pipeline.transcribe(&args.input)
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });

    println!("{text}");
}
