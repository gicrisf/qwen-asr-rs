# Transformer Layer: C vs Rust Implementation Comparison

This document analyzes why the C transformer is faster than the Rust/Candle implementation.

## Performance Gap

For 5 seconds of audio (65 tokens, 18 layers):

| Implementation | Transformer Time | Per Layer |
|----------------|------------------|-----------|
| C (estimated) | ~300 ms | ~17 ms |
| Rust | ~365 ms | ~20 ms |
| **Gap** | **~65 ms (22%)** | **~3 ms** |

---

## Architecture Overview

Both implement the same transformer layer:

```
Input [seq, d_model]
    │
    ├─► LayerNorm ─► Q,K,V Projections ─► Windowed Attention ─► O Projection ─► Residual
    │                                                                              │
    └──────────────────────────────────────────────────────────────────────────────┘
    │
    ├─► LayerNorm ─► FC1 ─► GELU ─► FC2 ─► Residual
    │                                         │
    └─────────────────────────────────────────┘
    │
Output [seq, d_model]
```

---

## Key Difference #1: Attention Algorithm

### C: Online Softmax (Memory-Efficient)

```c
// qwen_asr_kernels.c:1003-1048
void qwen_bidirectional_attention(...) {
    for (int h = 0; h < n_heads; h++) {
        for (int w = 0; w < n_windows; w++) {
            int ws = window_starts[w];
            int we = window_starts[w + 1];

            for (int i = ws; i < we; i++) {
                const float *q_row = Q + i * hidden + h * head_dim;
                float *o_row = out + i * hidden + h * head_dim;

                // Online softmax - never materializes score matrix
                float max_score = -1e30f;
                float sum_exp = 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                for (int j = ws; j < we; j++) {
                    const float *k_row = K + j * hidden + h * head_dim;
                    const float *v_row = V + j * hidden + h * head_dim;

                    float score = qwen_dot_f32(q_row, k_row, head_dim) * scale;

                    if (score > max_score) {
                        // New max: rescale accumulated output
                        float correction = expf(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        qwen_vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        // Accumulate with current max
                        float wt = expf(score - max_score);
                        sum_exp += wt;
                        qwen_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                // Final normalization
                qwen_vec_scale_inplace(o_row, 1.0f / sum_exp, head_dim);
            }
        }
    }
}
```

**Characteristics:**
- Never allocates [seq, seq] or [n_heads, seq, seq] score matrix
- O(seq × window_size × head_dim) memory per head
- Fused softmax + output accumulation in single pass
- SIMD-optimized dot product and vector ops

### Rust/Candle: Materialized Attention Matrix

```rust
// encoder.rs:230-246
fn forward(&self, x: &Tensor, mask: &Tensor, ...) -> Result<Tensor> {
    // Q, K, V projections
    let q = self.q_proj.forward(&xn)?;
    let k = self.k_proj.forward(&xn)?;
    let v = self.v_proj.forward(&xn)?;

    // Reshape: [seq, d_model] → [n_heads, seq, head_dim]
    let q = q.reshape((seq, n_heads, head_dim))?.transpose(0, 1)?;
    let k = k.reshape((seq, n_heads, head_dim))?.transpose(0, 1)?;
    let v = v.reshape((seq, n_heads, head_dim))?.transpose(0, 1)?.contiguous()?;

    // Compute full attention scores: [n_heads, seq, seq]
    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;

    // Apply mask (adds -inf outside windows)
    let scores = scores.broadcast_add(mask)?;

    // Full softmax over entire score matrix
    let weights = softmax_last_dim(&scores)?;

    // Output: [n_heads, seq, head_dim]
    let out = weights.matmul(&v)?;
    // ...
}
```

**Characteristics:**
- Allocates [n_heads, seq, seq] score matrix
- O(n_heads × seq²) memory
- Separate passes: scores → mask → softmax → output
- Multiple tensor allocations per operation

### Memory Comparison (65 tokens, 16 heads)

| Approach | Score Matrix Size | Per Layer |
|----------|-------------------|-----------|
| C (online) | 0 bytes | 0 |
| Rust (materialized) | 16 × 65 × 65 × 4 = 270 KB | 270 KB |

---

## Key Difference #2: Memory Allocation Strategy

### C: Preallocate and Reuse

```c
// qwen_asr_encoder.c:300-308
// Allocate ONCE before all layers
float *x_norm = (float *)malloc(total_tokens * d_model * sizeof(float));
float *q = (float *)malloc(total_tokens * d_model * sizeof(float));
float *k = (float *)malloc(total_tokens * d_model * sizeof(float));
float *v = (float *)malloc(total_tokens * d_model * sizeof(float));
float *attn_out = (float *)malloc(total_tokens * d_model * sizeof(float));
float *proj_out = (float *)malloc(total_tokens * d_model * sizeof(float));
float *ffn_mid = (float *)malloc(total_tokens * ffn_dim * sizeof(float));
float *ffn_out = (float *)malloc(total_tokens * d_model * sizeof(float));

// Reuse across all 18 layers
for (int layer = 0; layer < cfg->enc_layers; layer++) {
    qwen_layer_norm(x_norm, x, ...);
    qwen_linear(q, x_norm, ...);
    // ... reuse same buffers
}

// Free ONCE after all layers
free(x_norm); free(q); free(k); ...
```

### Rust/Candle: Allocate Per Operation

```rust
// Each operation creates new tensors
for layer in &self.layers {
    let xn = self.attn_norm.forward(&x)?;        // New tensor
    let q = self.q_proj.forward(&xn)?;           // New tensor
    let k = self.k_proj.forward(&xn)?;           // New tensor
    let v = self.v_proj.forward(&xn)?;           // New tensor
    let q = q.reshape(...)?.transpose(0, 1)?;    // New tensor
    let k = k.reshape(...)?.transpose(0, 1)?;    // New tensor
    let v = v.reshape(...)?.transpose(0, 1)?.contiguous()?;  // New tensor
    let scores = q.matmul(&k.transpose(1, 2)?)?; // New tensor
    // ... many more allocations
}
```

**Impact**: 18 layers × ~15 tensor allocations per layer = ~270 allocations vs 8 allocations in C.

---

## Key Difference #3: BLAS Backend

### C: OpenBLAS with cblas_sgemm

```c
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            seq_len, out_dim, in_dim,
            1.0f, x, in_dim, W, in_dim,
            0.0f, y, out_dim);
```

- Highly optimized for x86 (AVX2/AVX-512)
- Threaded internally for large matrices
- Well-tuned blocking and cache utilization

### Rust: gemm crate

```rust
// Through candle's matmul
let output = q.matmul(&k.transpose(1, 2)?)?;
```

- Good but not as mature as OpenBLAS
- May have different threading behavior
- Less architecture-specific tuning

---

## Key Difference #4: Tensor Operation Overhead

### C: Direct Pointer Arithmetic

```c
// Direct memory access
const float *q_row = Q + i * hidden + h * head_dim;
float score = qwen_dot_f32(q_row, k_row, head_dim);
```

### Rust/Candle: Abstraction Layers

```rust
// Each operation involves:
// 1. Shape validation
// 2. Stride computation
// 3. Potential contiguous() copy
// 4. Dispatch to backend

let q = q.reshape((seq, n_heads, head_dim))?  // Validate + compute strides
         .transpose(0, 1)?;                    // Validate + compute strides
```

---

## Optimization Opportunities for Rust

### 1. Implement Online Softmax Attention

Port the C online softmax algorithm to Rust:

```rust
fn windowed_attention_online(
    out: &mut [f32],
    q: &[f32], k: &[f32], v: &[f32],
    seq: usize, n_heads: usize, head_dim: usize,
    scale: f32, window_size: usize,
) {
    // Similar to C implementation
    // Never materialize score matrix
}
```

**Expected savings**: ~50-100 ms (eliminates score matrix allocation + separate softmax pass)

### 2. Preallocate Layer Buffers

```rust
struct EncoderBuffers {
    x_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_mid: Vec<f32>,
}

impl Encoder {
    fn forward_with_buffers(&self, mel: &Tensor, buffers: &mut EncoderBuffers) {
        // Reuse buffers across layers
    }
}
```

**Expected savings**: ~20-30 ms (reduces allocator pressure)

### 3. Fuse Operations

Combine reshape + transpose + matmul into single kernel:

```rust
fn fused_qkv_attention(
    q_proj: &Linear, k_proj: &Linear, v_proj: &Linear,
    x: &[f32], ...
) -> Vec<f32> {
    // Single pass: project + reshape + attention
}
```

### 4. Use Raw GEMM for Projections

Bypass Candle for linear projections:

```rust
fn linear_forward_raw(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    seq: usize, in_dim: usize, out_dim: usize,
) {
    unsafe {
        gemm::gemm(/* params for W^T @ X */);
    }
    // Add bias
}
```

---

## Summary

| Aspect | C | Rust/Candle | Impact |
|--------|---|-------------|--------|
| **Attention algorithm** | Online softmax | Materialized matrix | High |
| **Memory allocation** | Preallocate + reuse | Per-operation | Medium |
| **BLAS backend** | OpenBLAS | gemm crate | Low-Medium |
| **Abstraction overhead** | None (raw pointers) | Tensor ops | Low |

**Primary bottleneck**: The materialized attention matrix approach in Candle.

**Recommended fix priority**:
1. Implement online softmax attention (highest impact)
2. Preallocate reusable buffers
3. Consider raw GEMM for projections if needed
