# Benchmark Results: C vs Rust Implementation

Benchmarks run on 12-thread system with OpenBLAS (C) and gemm crate (Rust).

## Test Configuration

- **Hardware**: 12 logical cores
- **C BLAS**: OpenBLAS 0.3.30
- **Rust GEMM**: gicrisf/gemm fork (bf16 branch)
- **Model**: qwen3-asr-0.6b

---

## 5 Seconds Synthetic Silence

### C Implementation

```
$ ./qwen_asr_bench -d qwen3-asr-0.6b -n 5 -s 5

Mode: full pipeline  |  5 run(s)  |  5.0 s  [synthetic silence]

  run 1/5:  total=  1945 ms  enc=   441 ms  dec=  1389 ms  tokens=0
  run 2/5:  total=  1977 ms  enc=   444 ms  dec=  1407 ms  tokens=0
  run 3/5:  total=  1683 ms  enc=   405 ms  dec=  1159 ms  tokens=0
  run 4/5:  total=  2284 ms  enc=   594 ms  dec=  1602 ms  tokens=0
  run 5/5:  total=  2285 ms  enc=   600 ms  dec=  1596 ms  tokens=0

                     min      mean       max
total             1683.5    2034.9    2284.8  ms
encode             404.9     496.7     599.7  ms
decode            1159.4    1430.8    1602.4  ms
```

### Rust Implementation

```
$ ./target/release/bench -d ../qwen3-asr-0.6b -n 5 -s 5 -w 0

Mode: full pipeline  |  5 run(s)  |  5.0 s  [synthetic silence (5s)]

  run 1/5:  total=  1519 ms  enc=   594 ms  dec=   925 ms  tokens=0
  run 2/5:  total=  1439 ms  enc=   533 ms  dec=   906 ms  tokens=0
  run 3/5:  total=  1284 ms  enc=   505 ms  dec=   779 ms  tokens=0
  run 4/5:  total=  1582 ms  enc=   568 ms  dec=  1014 ms  tokens=0
  run 5/5:  total=  1789 ms  enc=   695 ms  dec=  1094 ms  tokens=0

                     min      mean       max
total             1284.2    1522.6    1789.2  ms
encode             505.1     579.0     695.2  ms
decode             779.1     943.6    1094.0  ms
```

### Rust Encoder Breakdown

```
$ ./target/release/bench -d ../qwen3-asr-0.6b -n 5 -s 5 -w 1

Mode: encoder only  |  5 run(s)  |  500 frames (5.0 s)

  run 1/5:  enc=   567 ms  conv=   200 ms  xfmr=   367 ms  seq_len=65
  run 2/5:  enc=   559 ms  conv=   188 ms  xfmr=   370 ms  seq_len=65
  run 3/5:  enc=   567 ms  conv=   211 ms  xfmr=   356 ms  seq_len=65
  run 4/5:  enc=   540 ms  conv=   173 ms  xfmr=   368 ms  seq_len=65
  run 5/5:  enc=   537 ms  conv=   171 ms  xfmr=   366 ms  seq_len=65

                     min      mean       max
encode             537.3     554.1     567.1  ms
conv_stem          170.9     188.7     211.0  ms
transformer        355.9     365.3     370.3  ms
per layer          19.77     20.30     20.57  ms/layer
```

---

## Summary: 5s Silence

| Component  | C (mean) | Rust (mean) | Difference          |
|------------|----------|-------------|---------------------|
| **Total**  | 2035 ms  | 1523 ms     | **Rust 25% faster** |
| **Encode** | 497 ms   | 579 ms      | Rust 16% slower     |
| **Decode** | 1431 ms  | 944 ms      | **Rust 34% faster** |

### Encoder Breakdown (Rust only)

| Component               | Time    | % of Encoder |
|-------------------------|---------|--------------|
| Conv Stem               | 189 ms  | 34%          |
| Transformer (18 layers) | 365 ms  | 66%          |
| Per Layer               | 20.3 ms | -            |

---

## 11 Seconds Real Audio (jfk.wav)

### Rust Implementation

```
$ ./target/release/bench -d ../qwen3-asr-0.6b -i ../samples/jfk.wav -n 5 -w 1

Mode: encoder only  |  5 run(s)  |  1100 frames (11.0 s)

  run 1/5:  enc=  1205 ms  conv=   294 ms  xfmr=   911 ms  seq_len=143
  run 2/5:  enc=  1096 ms  conv=   279 ms  xfmr=   818 ms  seq_len=143
  run 3/5:  enc=  1153 ms  conv=   244 ms  xfmr=   909 ms  seq_len=143
  run 4/5:  enc=  1238 ms  conv=   281 ms  xfmr=   957 ms  seq_len=143
  run 5/5:  enc=  1235 ms  conv=   323 ms  xfmr=   912 ms  seq_len=143

                     min      mean       max
encode            1096.4    1185.5    1238.4  ms
conv_stem          243.8     284.0     322.6  ms
transformer        817.8     901.5     957.4  ms
per layer          45.43     50.08     53.19  ms/layer
```

### Encoder Breakdown (11s audio)

| Component               | Time    | % of Encoder |
|-------------------------|---------|--------------|
| Conv Stem               | 284 ms  | 24%          |
| Transformer (18 layers) | 902 ms  | 76%          |
| Per Layer               | 50.1 ms | -            |

---

## Analysis

### Conv Stem Optimization Success

The custom im2col + GEMM implementation with parallel chunk processing achieved:

| Audio Length | Chunks | Conv Stem Time | Per-Chunk |
|--------------|--------|----------------|-----------|
| 5s           | 5      | 189 ms         | 38 ms     |
| 11s          | 11     | 284 ms         | 26 ms     |

The parallel chunk processing scales well - more chunks means better parallelization.

### Remaining Bottleneck: Transformer

The transformer layers now dominate encoder time:
- 5s audio: 365 ms (66% of encoder)
- 11s audio: 902 ms (76% of encoder)

The transformer uses Candle's `.matmul()` which goes through the gemm crate. Potential optimizations:
1. Ensure Candle uses `Parallelism::Rayon(0)` for large matmuls
2. Consider fused attention kernels
3. Profile individual layer components (Q/K/V projections, attention, FFN)

### Decode Performance

Rust decoder is significantly faster than C (34% improvement). This is likely due to:
- BF16 weight storage with optimized bf16 GEMM kernels
- Different threading strategy

---

## Realtime Performance

| Audio      | C rt_factor | Rust rt_factor |
|------------|-------------|----------------|
| 5s silence | 0.41x       | 0.30x          |

Both implementations achieve faster-than-realtime inference, with Rust being ~27% faster overall.

---

## Hardware/Software Details

```
System: 12 logical cores
C Compiler: gcc with -O3 -march=native -ffast-math
C BLAS: OpenBLAS 0.3.30
Rust: release build with LTO
Rust GEMM: gicrisf/gemm (bf16 branch)
Model: qwen3-asr-0.6b (0.6B parameters)
```
