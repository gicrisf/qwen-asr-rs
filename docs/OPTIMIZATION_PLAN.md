# Transformer Optimization Plan

## Current State

| Component | C | Rust | Gap |
|-----------|---|------|-----|
| Conv Stem | ~200ms | ~189ms | **Rust wins** |
| Transformer (18 layers) | ~300ms | ~365ms | C 18% faster |
| **Total Encoder** | ~500ms | ~554ms | C 10% faster |

The conv stem optimization (parallel im2col+GEMM) was successful. The remaining gap is in the transformer layers.

## Lessons Learned from Online Softmax Attempt

**What we tried**: Port C's online softmax algorithm to Rust
**Result**: 365ms → 473ms (29% slower!)

**Why it failed**:
1. Naive Rust loops without SIMD can't compete with batched GEMM
2. C version uses AVX2/NEON optimized `qwen_dot_f32`, `qwen_vec_axpy_inplace`, etc.
3. Per-element operations have more overhead than batched matmul

**Key insight**: To beat Candle's matmul-based attention, we need SIMD-optimized kernels, not just algorithmic changes.

---

## Optimization Strategies

### Strategy 1: Optimize Within Candle (Medium Effort, Medium Impact)

#### 1.1 Reduce Tensor Allocations

Current attention creates ~10 tensors per layer:
```rust
let q = self.q_proj.forward(&xn)?;           // alloc
let k = self.k_proj.forward(&xn)?;           // alloc
let v = self.v_proj.forward(&xn)?;           // alloc
let q = q.reshape(...)?.transpose(0, 1)?;    // alloc (view, but transpose needs contiguous)
let k = k.reshape(...)?.transpose(0, 1)?;    // alloc
let v = v.reshape(...)?.transpose(0, 1)?.contiguous()?;  // alloc
let scores = q.matmul(&k.transpose(1, 2)?)?; // alloc [n_heads, seq, seq]
let scores = scores.broadcast_add(mask)?;    // alloc
let weights = softmax_last_dim(&scores)?;    // alloc
let out = weights.matmul(&v)?;               // alloc
```

**Candle changes needed**:
- Add in-place operations: `matmul_into()`, `softmax_inplace()`
- Add fused reshape+transpose that returns contiguous
- Reuse score matrix buffer across layers

#### 1.2 Add Fused Attention Kernel

Add to candle a specialized attention function:
```rust
// candle-nn/src/ops/attention.rs
pub fn scaled_dot_product_attention(
    q: &Tensor,  // [batch, heads, seq_q, head_dim]
    k: &Tensor,  // [batch, heads, seq_k, head_dim]
    v: &Tensor,  // [batch, heads, seq_k, head_dim]
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor>
```

Internally fuses: `(Q @ K.T * scale + mask).softmax(-1) @ V`

---

### Strategy 2: Custom SIMD Attention Kernel (High Effort, High Impact)

Port C's online softmax with proper SIMD to a new crate or candle extension.

#### 2.1 SIMD Primitives Needed

```rust
// In gemm or new crate: simd_kernels

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD dot product (AVX2)
pub fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut sum = _mm256_setzero_ps();
        for i in (0..a.len()).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        // horizontal sum...
    }
}

/// dst = dst * scale + src (AVX2)
pub fn vec_scale_add_avx2(dst: &mut [f32], src: &[f32], scale: f32) { ... }

/// dst += alpha * src (AVX2)
pub fn vec_axpy_avx2(dst: &mut [f32], src: &[f32], alpha: f32) { ... }
```

#### 2.2 Online Softmax with SIMD

```rust
pub fn windowed_attention_simd(
    out: &mut [f32],
    q: &[f32], k: &[f32], v: &[f32],
    seq: usize, n_heads: usize, head_dim: usize,
    scale: f32, window_size: usize,
) {
    // Parallel across heads
    (0..n_heads).into_par_iter().for_each(|h| {
        // Per-head processing with SIMD kernels
        for window in windows {
            for i in window {
                let q_row = &q[...];
                for j in window {
                    let score = dot_f32_avx2(q_row, &k[...]) * scale;
                    // online softmax with SIMD vec ops
                }
            }
        }
    });
}
```

---

### Strategy 3: Optimize gemm Crate (Low-Medium Effort, Medium Impact)

#### 3.1 Verify Parallelism Settings

Check if candle is using `Parallelism::Rayon(0)` for matmul:
```rust
// In candle-core/src/cpu_backend/mod.rs
// Ensure large matmuls use threading
```

#### 3.2 Tune GEMM Parameters

The gemm crate has tunable parameters:
```rust
gemm::set_threading_threshold(threshold);
gemm::set_lhs_packing_threshold_multi_thread(threshold);
gemm::set_rhs_packing_threshold(threshold);
```

Profile to find optimal values for our workload (seq=65-143, d_model=896).

#### 3.3 Consider Prepacking Weights

For inference, weights are constant. Prepack them:
```rust
// During model load
let packed_q_weight = gemm::prepack_rhs(&q_weight_data, k, n);

// During forward (faster)
gemm::gemm_prepacked_rhs(&packed_q_weight, &input, &mut output);
```

---

### Strategy 4: Batch Operations Across Layers (Medium Effort, Medium Impact)

#### 4.1 Fuse Q/K/V Projections

Instead of 3 separate matmuls:
```rust
// Current: 3 matmuls
let q = x @ Wq;  // [seq, d_model] @ [d_model, d_model]
let k = x @ Wk;
let v = x @ Wv;

// Better: 1 matmul + split
let qkv = x @ W_qkv;  // [seq, d_model] @ [d_model, 3*d_model]
let (q, k, v) = qkv.split([d_model, d_model, d_model], -1);
```

Requires fusing weights during model load.

#### 4.2 Preallocate Buffers

```rust
struct EncoderCache {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    attn_out: Vec<f32>,
    ffn_mid: Vec<f32>,
}

impl Encoder {
    fn forward_with_cache(&self, mel: &Tensor, cache: &mut EncoderCache) {
        // Reuse cache buffers across 18 layers
    }
}
```

---

## Recommended Priority

| Priority | Strategy | Effort | Expected Gain | Notes |
|----------|----------|--------|---------------|-------|
| 1 | Verify gemm parallelism | Low | 5-15% | Quick check |
| 2 | Fuse Q/K/V projections | Medium | 10-20% | 3 matmuls → 1 |
| 3 | Add SIMD attention kernel | High | 20-40% | Match C perf |
| 4 | Prepack weights | Medium | 5-10% | One-time cost |
| 5 | Reduce allocations in candle | Medium | 10-15% | Requires candle changes |

---

## Quick Wins to Try First

### 1. Check gemm threading

```rust
// Add to encoder.rs temporarily
eprintln!("gemm threading threshold: {}", gemm::get_threading_threshold());
```

### 2. Profile individual operations

```rust
// Add timing to EncLayer::forward
let t0 = Instant::now();
let q = self.q_proj.forward(&xn)?;
eprintln!("q_proj: {:?}", t0.elapsed());
// ... etc
```

### 3. Try fused QKV in encoder.rs

Manually fuse the projection weights at load time and benchmark.

---

## Files to Modify

| File | Changes |
|------|---------|
| `qwen-asr-rs/src/encoder.rs` | Fuse QKV, preallocate buffers |
| `candle/candle-core/src/cpu_backend/mod.rs` | Check/tune GEMM parallelism |
| `candle/candle-nn/src/ops.rs` | Add fused attention op |
| `gemm/gemm/src/lib.rs` | Tune threading thresholds |
| `gemm/gemm-f32/src/` | SIMD optimizations if needed |

---

## Benchmark Commands

```bash
# Encoder only
./target/release/bench -d ../qwen3-asr-0.6b -n 10 -s 5 -w 1

# Full pipeline
./target/release/bench -d ../qwen3-asr-0.6b -n 5 -s 5 -w 0

# Compare with C
nix-shell --run "./qwen_asr_bench -d qwen3-asr-0.6b -n 5 -s 5"
```
