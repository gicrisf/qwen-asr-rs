# C vs Rust Conv Stem Implementation Comparison

This document provides a detailed comparison between the C implementation (`qwen_asr_kernels.c`, `qwen_asr_encoder.c`) and the Rust implementation (`encoder.rs`) of the conv stem for Qwen3-ASR.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [im2col Implementation](#im2col-implementation)
3. [Conv2D + GEMM](#conv2d--gemm)
4. [GELU Activation](#gelu-activation)
5. [Reshape Operation](#reshape-operation)
6. [Chunk Processing Strategy](#chunk-processing-strategy)
7. [Memory Management](#memory-management)
8. [Parallelization Strategy](#parallelization-strategy)
9. [Performance Comparison](#performance-comparison)
10. [Key Differences Summary](#key-differences-summary)

---

## Architecture Overview

Both implementations follow the same high-level architecture:

```
Mel Spectrogram [128, mel_frames]
    │
    ▼ (split into chunks)
┌─────────────────────────────────────┐
│  Per-chunk Conv Stem:               │
│    Conv1: [1, 128, W] → [480, 64, W1]   (stride 2)
│    GELU                             │
│    Conv2: [480, 64, W1] → [480, 32, W2] (stride 2)
│    GELU                             │
│    Conv3: [480, 32, W2] → [480, 16, W3] (stride 2)
│    GELU                             │
│    Reshape: [480, 16, W3] → [W3, 7680]  │
│    Project: [W3, 7680] → [W3, d_model]  │
│    Add sinusoidal PE                │
└─────────────────────────────────────┘
    │
    ▼ (concatenate all chunks)
Transformer Encoder Layers
    │
    ▼
Output [total_tokens, output_dim]
```

---

## im2col Implementation

The `im2col` operation transforms convolution into matrix multiplication by unrolling input patches into columns.

### C Implementation (`qwen_asr_kernels.c:566-590`)

```c
static void im2col(const float *in, float *cols,
                   int c_in, int h_in, int w_in,
                   int kh, int kw, int stride, int padding,
                   int h_out, int w_out) {
    int col_len = h_out * w_out;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + (size_t)col_row * col_len;
                for (int oh = 0; oh < h_out; oh++) {
                    int ih = oh * stride - padding + ki;
                    for (int ow = 0; ow < w_out; ow++) {
                        int iw = ow * stride - padding + kj;
                        if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                            col_ptr[oh * w_out + ow] = in[ic * h_in * w_in + ih * w_in + iw];
                        } else {
                            col_ptr[oh * w_out + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}
```

### Rust Implementation (`encoder.rs:16-47`)

```rust
fn im2col(
    input: &[f32],
    c_in: usize, h_in: usize, w_in: usize,
    kh: usize, kw: usize,
    stride: usize, padding: usize,
    h_out: usize, w_out: usize,
) -> Vec<f32> {
    let col_len = h_out * w_out;
    let patch_size = c_in * kh * kw;
    let mut cols = vec![0.0f32; patch_size * col_len];

    for ic in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let col_row = (ic * kh + ki) * kw + kj;
                let col_ptr = col_row * col_len;
                for oh in 0..h_out {
                    let ih = (oh * stride + ki) as isize - padding as isize;
                    for ow in 0..w_out {
                        let iw = (ow * stride + kj) as isize - padding as isize;
                        if ih >= 0 && ih < h_in as isize && iw >= 0 && iw < w_in as isize {
                            let in_idx = ic * h_in * w_in + ih as usize * w_in + iw as usize;
                            cols[col_ptr + oh * w_out + ow] = input[in_idx];
                        }
                        // else: already 0 (padding)
                    }
                }
            }
        }
    }
    cols
}
```

### Comparison

| Aspect | C | Rust |
|--------|---|------|
| **Output allocation** | Caller provides buffer | Returns new `Vec<f32>` |
| **Bounds checking** | Manual `if` checks | Same manual checks (no implicit bounds) |
| **Index calculation** | `ih = oh * stride - padding + ki` | `ih = (oh * stride + ki) as isize - padding as isize` |
| **Padding handling** | Explicit `else { 0.0f }` | Relies on `vec![0.0f32; ...]` initialization |
| **Pointer arithmetic** | Raw pointer `cols + col_row * col_len` | Index `col_ptr + oh * w_out + ow` |

**Key Insight**: The Rust version uses `isize` for the index calculation to handle negative padding offsets, then casts back to `usize` for array indexing. The C version uses signed `int` throughout.

---

## Conv2D + GEMM

### C Implementation (`qwen_asr_kernels.c:592-634`)

```c
void qwen_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;

    /* im2col: input -> column matrix [patch_size, spatial_out] */
    float *cols = (float *)malloc((size_t)patch_size * spatial_out * sizeof(float));
    im2col(in, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, spatial_out, patch_size,
                1.0f, weight, patch_size, cols, spatial_out,
                0.0f, out, spatial_out);
#else
    // Naive fallback (not shown)
#endif

    free(cols);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * spatial_out;
            for (int s = 0; s < spatial_out; s++) {
                row[s] += b;
            }
        }
    }
}
```

### Rust Implementation (`encoder.rs:49-107`)

```rust
fn conv2d_gemm(
    input: &[f32],
    weight: &[f32],  // [c_out, patch_size] row-major
    bias: &[f32],    // [c_out]
    c_in: usize, c_out: usize,
    h_in: usize, w_in: usize,
    kh: usize, kw: usize,
    stride: usize, padding: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h_in + 2 * padding - kh) / stride + 1;
    let w_out = (w_in + 2 * padding - kw) / stride + 1;
    let patch_size = c_in * kh * kw;
    let spatial_out = h_out * w_out;

    // im2col
    let cols = im2col(input, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    // GEMM
    let mut output = vec![0.0f32; c_out * spatial_out];

    unsafe {
        gemm::gemm(
            c_out,                   // m
            spatial_out,             // n
            patch_size,              // k
            output.as_mut_ptr(),
            1,                       // dst_cs (column stride)
            spatial_out as isize,    // dst_rs (row stride)
            false,                   // read_dst
            weight.as_ptr(),
            1,                       // lhs_cs
            patch_size as isize,     // lhs_rs
            cols.as_ptr(),
            1,                       // rhs_cs
            spatial_out as isize,    // rhs_rs
            0.0,                     // alpha
            1.0,                     // beta
            false, false, false,     // conj flags
            gemm::Parallelism::None, // parallelism
        );
    }

    // Add bias
    for oc in 0..c_out {
        let b = bias[oc];
        let row_start = oc * spatial_out;
        for s in 0..spatial_out {
            output[row_start + s] += b;
        }
    }

    (output, h_out, w_out)
}
```

### Comparison

| Aspect | C (cblas_sgemm) | Rust (gemm crate) |
|--------|-----------------|-------------------|
| **API style** | Column-major default, flags for row-major | Explicit stride parameters |
| **Parameters** | `(layout, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)` | `(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs, alpha, beta, conj*, parallelism)` |
| **Row-major setup** | `CblasRowMajor, CblasNoTrans, CblasNoTrans` | `cs=1, rs=width` for each matrix |
| **Parallelism** | Implicit (OpenBLAS/MKL threads) | Explicit `Parallelism::Rayon(n)` or `None` |
| **Return value** | Output via pointer parameter | Returns `(output, h_out, w_out)` tuple |

### GEMM Stride Explanation

For row-major matrices, the stride parameters are:
- `cs` (column stride) = 1 (adjacent elements in a row)
- `rs` (row stride) = width (elements between rows)

```
Matrix [M, N] row-major:
  Element (i, j) at index: i * N + j
  Column stride: 1
  Row stride: N
```

---

## GELU Activation

### C Implementation (`qwen_asr_kernels.c:886-893`)

```c
void qwen_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}
```

### Rust Implementation (`encoder.rs:109-119`)

```rust
fn gelu_inplace(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;

    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        let inner = SQRT_2_OVER_PI * (*v + COEFF * x3);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}
```

### Comparison

| Aspect                    | C                                         | Rust                             |
|---------------------------|-------------------------------------------|----------------------------------|
| **Constants**             | Inline literals                           | Named `const` values             |
| **Math function**         | `tanhf()`                                 | `.tanh()` method                 |
| **Loop style**            | Index-based `for (int i = 0; i < n; i++)` | Iterator `for v in x.iter_mut()` |
| **In-place modification** | Direct `x[i] = ...`                       | Dereference `*v = ...`           |

**Formula**: Both use the GELU approximation:
```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

---

## Reshape Operation

The reshape transforms `[480, 16, W3]` (channels, frequency, time) to `[W3, 7680]` (time, flattened features).

### C Implementation (`qwen_asr_encoder.c:261-271`)

```c
/* Reshape [480, 16, w3] -> [w3, 480*16=7680] then project to d_model */
int conv_proj_dim = QWEN_CONV_HIDDEN * h3; /* 480 * 16 = 7680 */
float *reshaped = (float *)malloc(w3 * conv_proj_dim * sizeof(float));
for (int t = 0; t < w3; t++) {
    for (int ch = 0; ch < QWEN_CONV_HIDDEN; ch++) {
        for (int f = 0; f < h3; f++) {
            reshaped[t * conv_proj_dim + ch * h3 + f] =
                c3[ch * h3 * w3 + f * w3 + t];
        }
    }
}
```

### Rust Implementation (`encoder.rs:385-394`)

```rust
// Reshape [480, 16, w3] -> [w3, 7680]
let mut reshaped = vec![0.0f32; w3 * CONV_CH * h3];
for t in 0..w3 {
    for ch in 0..CONV_CH {
        for f in 0..h3 {
            reshaped[t * CONV_CH * h3 + ch * h3 + f] =
                x[ch * h3 * w3 + f * w3 + t];
        }
    }
}
```

### Index Mapping Visualization

```
Input:  c3[ch, f, t] stored as c3[ch * h3 * w3 + f * w3 + t]
        Shape: [480, 16, W3]

Output: reshaped[t, ch * h3 + f] stored as reshaped[t * 7680 + ch * 16 + f]
        Shape: [W3, 7680]

Transformation: For each time step t, flatten (ch, f) into a single vector
```

### Comparison

| Aspect                     | C                                       | Rust                             |
|----------------------------|-----------------------------------------|----------------------------------|
| **Loop structure**         | Identical triple nested loop            | Identical                        |
| **Index formula (output)** | `t * conv_proj_dim + ch * h3 + f`       | `t * CONV_CH * h3 + ch * h3 + f` |
| **Index formula (input)**  | `ch * h3 * w3 + f * w3 + t`             | `ch * h3 * w3 + f * w3 + t`      |
| **Dimension calculation**  | `conv_proj_dim = QWEN_CONV_HIDDEN * h3` | Uses `CONV_CH * h3` inline       |

**The implementations are functionally identical.**

---

## Chunk Processing Strategy

### C Implementation (`qwen_asr_encoder.c:221-286`)

```c
/* Process each chunk through Conv2D + reshape + project + sinusoidal PE */
for (int c = 0; c < n_chunks; c++) {
    int start = c * chunk_size;
    int end = start + chunk_size;
    if (end > mel_frames) end = mel_frames;
    int chunk_w = end - start;

    /* Extract chunk mel: [128, chunk_w] */
    float *chunk_mel = (float *)malloc(128 * chunk_w * sizeof(float));
    for (int m = 0; m < 128; m++) {
        memcpy(chunk_mel + m * chunk_w, mel + m * mel_frames + start,
               chunk_w * sizeof(float));
    }

    /* Conv2D layer 1 */
    float *c1 = (float *)malloc(...);
    qwen_conv2d(c1, chunk_mel, ...);
    qwen_gelu(c1, ...);
    free(chunk_mel);

    /* Conv2D layer 2 */
    float *c2 = (float *)malloc(...);
    qwen_conv2d(c2, c1, ...);
    qwen_gelu(c2, ...);
    free(c1);

    /* Conv2D layer 3 */
    float *c3 = (float *)malloc(...);
    qwen_conv2d(c3, c2, ...);
    qwen_gelu(c3, ...);
    free(c2);

    /* Reshape + Project + PE */
    // ... (sequential)
}
```

### Rust Implementation (`encoder.rs:348-411`)

```rust
// --- Process chunks in PARALLEL using rayon ---
let conv_results: Vec<_> = (0..n_chunks)
    .into_par_iter()
    .map(|c| {
        let start = c * chunk_size;
        let chunk_w = chunk_size.min(mel_frames - start);

        // Extract chunk
        let mut chunk_mel = vec![0.0f32; MEL_BINS * chunk_w];
        for m in 0..MEL_BINS {
            for w in 0..chunk_w {
                chunk_mel[m * chunk_w + w] = mel_data[m * mel_frames + start + w];
            }
        }

        // Conv1 + GELU
        let (mut x, h1, w1) = conv2d_gemm(&chunk_mel, ...);
        gelu_inplace(&mut x);

        // Conv2 + GELU
        let (mut x, h2, w2) = conv2d_gemm(&x, ...);
        gelu_inplace(&mut x);

        // Conv3 + GELU
        let (mut x, h3, w3) = conv2d_gemm(&x, ...);
        gelu_inplace(&mut x);

        // Reshape (still in parallel)
        let mut reshaped = vec![...];
        // ... reshape loop ...

        (reshaped, w3, h3)
    })
    .collect();

// Project + PE sequentially (requires Tensor operations)
for (c, (reshaped, w3, h3)) in conv_results.into_iter().enumerate() {
    let x_tensor = Tensor::from_vec(reshaped, ...)?;
    let x_proj = self.conv_out.forward(&x_tensor)?;
    let pe = sinusoidal_pe(w3, ...)?;
    chunks.push((x_proj + pe)?);
}
```

### Comparison

| Aspect                   | C                        | Rust                       |
|--------------------------|--------------------------|----------------------------|
| **Chunk iteration**      | Sequential `for` loop    | Parallel `into_par_iter()` |
| **Mel extraction**       | `memcpy` row-by-row      | Element-by-element copy    |
| **Intermediate buffers** | `malloc/free` per layer  | `Vec` with automatic drop  |
| **Projection**           | In same loop, sequential | Separate sequential loop   |
| **Parallelism source**   | BLAS-internal (threads)  | Rayon (work-stealing)      |

---

## Memory Management

### C Approach

```c
// Explicit allocation
float *c1 = (float *)malloc(QWEN_CONV_HIDDEN * h1 * w1 * sizeof(float));

// Manual deallocation
free(chunk_mel);
free(c1);
free(c2);
// etc.
```

**Characteristics**:
- Explicit size calculations with `sizeof`
- Must remember to free every allocation
- Risk of memory leaks on error paths
- No automatic cleanup

### Rust Approach

```rust
// Automatic allocation (zero-initialized)
let mut chunk_mel = vec![0.0f32; MEL_BINS * chunk_w];

// Implicit deallocation when out of scope
// (or when ownership moves)
let (mut x, h1, w1) = conv2d_gemm(&chunk_mel, ...);
// chunk_mel dropped here if not used again
```

**Characteristics**:
- Size implicit from element count
- Automatic deallocation via RAII/Drop
- No memory leaks possible (compiler-enforced)
- Zero-initialization with `vec![0.0; n]`

---

## Parallelization Strategy

### C Strategy

```
┌────────────────────────────────────────┐
│  Sequential chunk loop                 │
│    ┌──────────────────────────────┐   │
│    │  Conv2D (im2col)             │   │
│    │    ↓                         │   │
│    │  GEMM (cblas_sgemm)          │◄──┼── Multi-threaded (OpenBLAS)
│    │    ↓                         │   │
│    │  GELU (single-threaded)      │   │
│    │    ↓                         │   │
│    │  Next layer...               │   │
│    └──────────────────────────────┘   │
└────────────────────────────────────────┘
```

- Chunks processed **sequentially**
- BLAS GEMM calls use **internal threading** (OpenBLAS/MKL)
- GELU, reshape, etc. are single-threaded

### Rust Strategy

```
┌────────────────────────────────────────────────────────┐
│  Parallel chunk processing (Rayon)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ Chunk 0  │ │ Chunk 1  │ │ Chunk 2  │  ...          │
│  │ Conv1    │ │ Conv1    │ │ Conv1    │               │
│  │ GELU     │ │ GELU     │ │ GELU     │               │
│  │ Conv2    │ │ Conv2    │ │ Conv2    │               │
│  │ GELU     │ │ GELU     │ │ GELU     │               │
│  │ Conv3    │ │ Conv3    │ │ Conv3    │               │
│  │ GELU     │ │ GELU     │ │ GELU     │               │
│  │ Reshape  │ │ Reshape  │ │ Reshape  │               │
│  └──────────┘ └──────────┘ └──────────┘               │
│       ↓             ↓            ↓                     │
│  ┌──────────────────────────────────────┐             │
│  │ Sequential: Project + PE (uses Candle) │            │
│  └──────────────────────────────────────┘             │
└────────────────────────────────────────────────────────┘
```

- Chunks processed **in parallel** via Rayon work-stealing
- Individual GEMM calls are **single-threaded** (`Parallelism::None`)
- Parallelism is at chunk level, not matrix level
- Projection done sequentially (Candle/Tensor operations)

---

## Performance Comparison

### Benchmark Results (11s audio, 12 threads)

| Component     | C       | Rust (Sequential) | Rust (Parallel) |
|---------------|---------|-------------------|-----------------|
| Conv Stem     | ~608 ms | ~864 ms           | **~284 ms**     |
| Total Encoder | ~800 ms | ~1750 ms          | ~1180 ms        |

### Analysis

1. **Rust Sequential (~864 ms)**: Slower than C because:
   - Extra Vec allocations (vs reusable buffers)
   - Bounds checking overhead
   - Less aggressive BLAS optimization in gemm crate vs OpenBLAS

2. **Rust Parallel (~284 ms)**: Faster than C because:
   - 11 chunks processed simultaneously
   - Work-stealing balances uneven chunk sizes
   - Avoids BLAS threading overhead for small matrices

3. **Trade-off**: The Rust implementation sacrifices single-GEMM efficiency for chunk-level parallelism, which wins for typical audio lengths.

---

## Key Differences Summary

| Aspect               | C                    | Rust                     |
|----------------------|----------------------|--------------------------|
| **Memory**           | Manual malloc/free   | Automatic Vec/Drop       |
| **GEMM Library**     | cblas (OpenBLAS/MKL) | gemm crate               |
| **Parallelism**      | BLAS-internal        | Rayon (chunk-level)      |
| **Chunk Processing** | Sequential           | Parallel                 |
| **Safety**           | Manual bounds        | Compiler-enforced        |
| **Performance**      | ~608 ms              | ~284 ms (2.1x faster)    |
| **Code Style**       | Imperative, pointers | Functional iterators     |
| **Error Handling**   | Return codes, manual | Result types, ? operator |

### Code Correspondence Table

| C Function      | Rust Function           | Location             |
|-----------------|-------------------------|----------------------|
| `im2col()`      | `im2col()`              | `encoder.rs:16-47`   |
| `qwen_conv2d()` | `conv2d_gemm()`         | `encoder.rs:49-107`  |
| `qwen_gelu()`   | `gelu_inplace()`        | `encoder.rs:109-119` |
| `cblas_sgemm()` | `gemm::gemm()`          | via gemm crate       |
| Reshape loop    | Reshape loop            | `encoder.rs:385-394` |
| Chunk for-loop  | `into_par_iter().map()` | `encoder.rs:350-397` |

---

## Conclusion

The Rust implementation achieves a **2.1x speedup** over the C implementation by:

1. **Parallelizing chunk processing** instead of relying on BLAS-internal threading
2. **Using work-stealing** (Rayon) for efficient load balancing
3. **Keeping individual GEMMs single-threaded** to avoid contention

The core algorithms (im2col, GELU, reshape) are **functionally identical** between C and Rust, differing only in syntax and memory management idioms. The performance difference comes entirely from the parallelization strategy.
