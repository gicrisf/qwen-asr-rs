use std::path::PathBuf;

use clap::Parser;
use rayon::ThreadPoolBuilder;

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

    /// Number of threads for inference (default: all logical cores)
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,

    /// Suppress progress output on stderr (transcription still printed to stdout)
    #[arg(long)]
    silent: bool,
}

fn main() {
    let args = Args::parse();

    if args.threads > 0 {
        ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap_or_else(|e| eprintln!("warning: could not set thread count: {e}"));
    }

    if !args.silent {
        let n = rayon::current_num_threads();
        eprintln!("Loading model from {} ... ({n} threads)", args.model_dir.display());
    }

    let mut pipeline = qwen_asr_rs::transcribe::Pipeline::load(&args.model_dir)
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });

    if !args.silent {
        eprintln!("Transcribing {} ...", args.input.display());
    }

    let text = pipeline.transcribe(&args.input)
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });

    println!("{text}");
}
