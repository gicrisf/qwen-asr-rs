use std::path::PathBuf;

use clap::Parser;
use rayon::ThreadPoolBuilder;

use qwen_asr_rs::transcribe::{Pipeline, TimingInfo};

/// Qwen3-ASR inference benchmark
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Model directory
    #[arg(short = 'd', long)]
    model_dir: PathBuf,

    /// Audio file (default: synthetic silence)
    #[arg(short = 'i', long)]
    input: Option<PathBuf>,

    /// Synthetic silence duration in seconds (ignored when -i is given)
    #[arg(short = 's', long, default_value_t = 5)]
    silence_sec: u32,

    /// Number of benchmark runs
    #[arg(short = 'n', long, default_value_t = 5)]
    runs: usize,

    /// Number of threads (default: all logical cores)
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,

    /// What to benchmark: 0 = full pipeline, 1 = encoder only
    #[arg(short = 'w', long, default_value_t = 0)]
    what: u8,
}

// ── Stats ─────────────────────────────────────────────────────────────────────

struct Stats { min: f64, mean: f64, max: f64 }

fn stats(values: &[f64]) -> Stats {
    let min  = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max  = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Stats { min, mean, max }
}

// ── Full pipeline benchmark ───────────────────────────────────────────────────

fn bench_full(pipeline: &mut Pipeline, n_runs: usize, args: &Args) {
    let (mel, audio_ms, src_desc) = match &args.input {
        Some(p) => {
            let (m, ms) = pipeline.mel_from_wav(p)
                .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });
            (m, ms, p.display().to_string())
        }
        None => {
            let (m, ms) = pipeline.mel_silence(args.silence_sec)
                .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });
            (m, ms, format!("synthetic silence ({}s)", args.silence_sec))
        }
    };

    eprintln!("Mode: full pipeline  |  {} run(s)  |  {:.1} s  [{}]\n",
              n_runs, audio_ms / 1000.0, src_desc);

    // Warm-up
    eprintln!("  warmup ...");
    let _ = pipeline.transcribe_mel(&mel, audio_ms);

    let mut total_ms_v  = Vec::with_capacity(n_runs);
    let mut encode_ms_v = Vec::with_capacity(n_runs);
    let mut decode_ms_v = Vec::with_capacity(n_runs);

    for i in 0..n_runs {
        let (_, TimingInfo { encode_ms, decode_ms, n_tokens, .. }) =
            pipeline.transcribe_mel(&mel, audio_ms)
                .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });

        let total_ms = encode_ms + decode_ms;
        total_ms_v.push(total_ms);
        encode_ms_v.push(encode_ms);
        decode_ms_v.push(decode_ms);

        eprintln!("  run {}/{n_runs}:  total={:6.0} ms  enc={:6.0} ms  dec={:6.0} ms  tokens={}  rt={:.2}x",
                  i + 1, total_ms, encode_ms, decode_ms, n_tokens, total_ms / audio_ms);
    }

    let tot = stats(&total_ms_v);
    let enc = stats(&encode_ms_v);
    let dec = stats(&decode_ms_v);

    eprintln!();
    eprintln!("{:<14}  {:>8}  {:>8}  {:>8}", "",       "min",    "mean",   "max");
    eprintln!("{:<14}  {:>8.1}  {:>8.1}  {:>8.1}  ms",  "total",  tot.min, tot.mean, tot.max);
    eprintln!("{:<14}  {:>8.1}  {:>8.1}  {:>8.1}  ms",  "encode", enc.min, enc.mean, enc.max);
    eprintln!("{:<14}  {:>8.1}  {:>8.1}  {:>8.1}  ms",  "decode", dec.min, dec.mean, dec.max);
    eprintln!("{:<14}  {:>8.2}  {:>8.2}  {:>8.2}  x RT", "rt_factor",
              tot.min / audio_ms, tot.mean / audio_ms, tot.max / audio_ms);
    eprintln!();
}

// ── Encoder-only benchmark ────────────────────────────────────────────────────

fn bench_encoder(pipeline: &mut Pipeline, n_runs: usize, args: &Args) {
    let (mel, audio_ms, src_desc) = match &args.input {
        Some(p) => {
            let (m, ms) = pipeline.mel_from_wav(p)
                .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });
            (m, ms, p.display().to_string())
        }
        None => {
            let (m, ms) = pipeline.mel_silence(args.silence_sec)
                .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });
            (m, ms, format!("synthetic silence ({}s)", args.silence_sec))
        }
    };
    let n_frames = mel.dims()[1];

    eprintln!("Mode: encoder only  |  {} run(s)  |  {} frames ({:.1} s)  [{}]\n",
              n_runs, n_frames, audio_ms / 1000.0, src_desc);

    // Warm-up
    let _ = pipeline.encode_timed(&mel);

    let mut elapsed_v = Vec::with_capacity(n_runs);

    for i in 0..n_runs {
        let (seq_len, enc_ms) = pipeline.encode_timed(&mel)
            .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });
        elapsed_v.push(enc_ms);
        eprintln!("  run {}/{n_runs}:  enc={:6.0} ms  seq_len={}", i + 1, enc_ms, seq_len);
    }

    let enc = stats(&elapsed_v);
    let n_layers = pipeline.encoder.cfg.layers;

    eprintln!();
    eprintln!("{:<14}  {:>8}  {:>8}  {:>8}", "",         "min",    "mean",   "max");
    eprintln!("{:<14}  {:>8.1}  {:>8.1}  {:>8.1}  ms",    "encode", enc.min, enc.mean, enc.max);
    eprintln!("{:<14}  {:>8.2}  {:>8.2}  {:>8.2}  ms/layer", "per layer",
              enc.min / n_layers as f64, enc.mean / n_layers as f64, enc.max / n_layers as f64);
    eprintln!();
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    if args.what > 1 {
        eprintln!("error: -w must be 0 or 1");
        std::process::exit(1);
    }
    if args.runs < 1 {
        eprintln!("error: -n must be >= 1");
        std::process::exit(1);
    }

    if args.threads > 0 {
        ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap_or_else(|e| eprintln!("warning: could not set thread count: {e}"));
    }

    let n_threads = rayon::current_num_threads();
    let n_cpus    = num_cpus::get();
    eprintln!("system_info: n_threads = {n_threads} / {n_cpus}\n");

    eprintln!("Loading model from {} ...", args.model_dir.display());
    let mut pipeline = Pipeline::load(&args.model_dir)
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });

    match args.what {
        0 => bench_full(&mut pipeline, args.runs, &args),
        1 => bench_encoder(&mut pipeline, args.runs, &args),
        _ => unreachable!(),
    }
}
