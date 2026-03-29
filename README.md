# Qwen3 ASR

> **Experimental branch**: This branch uses [mixed-precision BF16 GEMM](https://github.com/sarah-quinones/gemm/pull/40) via forked dependencies. Once the upstream PR is merged, these optimizations will be available in the main branch.

Rust port of [antirez/qwen-asr](https://github.com/antirez/qwen-asr), a CPU inference engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR) speech-to-text models, built on [candle](https://github.com/huggingface/candle).

The original C implementation by [antirez](https://github.com/antirez) is an almost-zero-dependency inference engine that runs in real time even on modest hardware, with support for offline, segmented, and streaming transcription modes. This port follows the same architecture and uses it as the reference for correctness.

Supports:
- `Qwen3-ASR-0.6B` (single shard, ~1.4 GB)
- `Qwen3-ASR-1.7B` (multi-shard, ~3.5 GB)

Model variant is auto-detected from the model directory.

## Model Download

```bash
bash download_model.sh            # interactive: prompts for small or large
```

Not interactive:

```bash
bash download_model.sh --model small --dir /path/to/model-dir
```

Or download directly from [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR).

## Build

```bash
cargo build --release
```

Binaries are placed in `target/release/`.

## Usage

### Transcribe a WAV file

```bash
./target/release/qwen-asr-rs -d /path/to/model -i audio.wav
```

Transcription is printed to stdout. Progress/timing is printed to stderr.

**Options:**

| Flag       | Default   | Description                              |
|------------|-----------|------------------------------------------|
| `-d`       | required  | Model directory                          |
| `-i`       | required  | Input WAV file (16 kHz mono recommended) |
| `-t N`     | all cores | Number of Rayon threads                  |
| `--silent` | off       | Suppress stderr output                   |

### Benchmark

```bash
./target/release/bench -d /path/to/model -i audio.wav -n 5
```

**Options:**

| Flag   | Default   | Description                                          |
|--------|-----------|------------------------------------------------------|
| `-d`   | required  | Model directory                                      |
| `-i`   | —         | Input WAV file (uses synthetic silence if omitted)   |
| `-s N` | `5`       | Silence duration in seconds (when `-i` is not given) |
| `-n N` | `5`       | Number of benchmark runs                             |
| `-t N` | all cores | Number of threads                                    |
| `-w 0` | —         | Benchmark full pipeline (default)                    |
| `-w 1` | —         | Benchmark encoder only                               |

Example output:

```
system_info: n_threads = 4 / 12

Loading model from qwen3-asr-0.6b ...
Mode: full pipeline  |  5 run(s)  |  11.0 s  [samples/jfk.wav]

  warmup ...
  run 1/5:  total=  6838 ms  enc=  1746 ms  dec=  5092 ms  tokens=26  rt=0.62x
  ...

                     min      mean       max
total             6463.1    6745.6    6853.1  ms
encode            1480.3    1662.8    1746.0  ms
decode            4982.8    5082.8    5137.6  ms
rt_factor           0.59      0.61      0.62  x RT
```

## Library

The crate is also usable as a library:

```rust
use qwen_asr_rs::transcribe::Pipeline;

let mut pipeline = Pipeline::load(model_dir)?;

// Transcribe a WAV file
let text = pipeline.transcribe(wav_path)?;

// With timing info
let (text, timing) = pipeline.transcribe_timed(wav_path)?;
println!("encode: {:.0} ms  decode: {:.0} ms", timing.encode_ms, timing.decode_ms);
```

## Architecture

The pipeline mirrors the original C implementation:

```
WAV → mel spectrogram → Encoder → audio embeddings
                                         |
                   prompt embeddings ────┘
                   (prefix + audio + suffix)
                                         |
                                      Decoder → text tokens → Tokenizer → text
```

The **encoder** runs a Conv2D stem (three stride-2 layers) followed by a windowed self-attention transformer. The **decoder** is a Qwen3 causal LM (GQA, RoPE, SwiGLU) with audio embeddings injected directly into the prompt.

## Performance

Benchmarked on AMD Ryzen 5 4600H (6c/12t laptop), `Qwen3-ASR-0.6B`, JFK sample (~11 s audio).

**Warm inference — bench binary (10 runs, model loaded once):**

| Implementation     | Mean    | RT factor | vs libtorch |
|--------------------|---------|-----------|-------------|
| libtorch (tch-rs)  | 6018 ms | 0.55x     | baseline    |
| candle (main)      | 6444 ms | 0.59x     | 7% slower   |
| candle (bf16-gemm) | 4698 ms | 0.43x     | **22% faster** |

This branch uses forked crates with BF16 support:
- [gemm fork](https://github.com/gicrisf/gemm/tree/bf16) ([upstream PR](https://github.com/sarah-quinones/gemm/pull/40))
- [candle fork](https://github.com/gicrisf/candle)

## Credits

- [antirez](https://github.com/antirez) for his [C implementation](https://github.com/antirez/qwen-asr): a remarkably clean, zero-dependency inference engine.
- [candle](https://github.com/huggingface/candle) by the Hugging Face team and its community, the framework that made this port practical.
