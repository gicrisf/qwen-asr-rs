# GEMM Pre-packing Optimization Plan for qwen-asr-rs

## Executive Summary

This document outlines a performance optimization strategy for the Rust implementation of Qwen3-ASR. The key insight is that **pre-packing weight matrices at model load time** can significantly improve inference performance by avoiding redundant packing operations during each forward pass.

---

## 1. Background

### Current Architecture

qwen-asr-rs is a pure-Rust ASR inference engine using:
- **candle** (Hugging Face's ML framework) for tensor operations
- **gemm crate** (custom fork with BF16 support) for matrix multiplication
- BF16 decoder weights, F32 encoder weights

### Performance vs C Reference

Benchmarked on AMD Ryzen 5 4600H, 0.6B model, 11s audio (jfk.wav), 4 threads:

| Implementation | Encode | Decode | Total | RT Factor |
|----------------|--------|--------|-------|-----------|
| C (OpenBLAS)   | ~1.2s  | ~3.3s  | ~4.6s | 0.42x     |
| Rust (candle)  | ~1.9s  | ~4.1s  | ~6.1s | 0.55x     |

The Rust implementation is **~33% slower overall**.

---

## 2. Key Findings

### 2.1 BF16 Decoder Success

We implemented native BF16 GEMM in a fork of the gemm crate. Results:

| Config | Warm Inference | vs Baseline |
|--------|----------------|-------------|
| libtorch (F32) | 6018ms | baseline |
| candle upstream (BF16 weights, F32 GEMM) | 6444ms | 7% slower |
| candle + BF16 GEMM fork | 4698ms | **22% faster** |

**Why BF16 GEMM wins despite slower microbenchmarks:**

Isolated gemm benchmarks show BF16 is 5-47% slower than F32 (conversion overhead). But real model inference is **memory-bandwidth bound**, not compute-bound:

- Decoder has ~168 weight matrices across 24 layers
- Total weights: ~500MB (BF16) vs ~1GB (F32)
- During autoregressive decoding, same weights are reused for every token
- BF16's half-size weights mean better cache utilization across layers
- Memory bandwidth savings outweigh conversion overhead

**Key insight from the implementation:** Candle upstream was doing a full tensor BF16→F32 conversion before every GEMM. Our fork converts during the packing phase, which is more efficient because:
1. Conversion happens in cache-sized packing buffers (L1/L2)
2. No intermediate full-size F32 tensor allocation
3. Half the bytes read from main memory

### 2.2 BF16 Encoder Failed

We tested BF16 for the encoder. Results:

| Config | Encode | Decode | Change |
|--------|--------|--------|--------|
| F32 encoder | ~1.9s | ~4.1s | baseline |
| BF16 encoder | ~2.6s | ~3.9s | encode **37% slower** |

**Why BF16 encoder is slower:**

The encoder is **compute-bound**, not memory-bound:
- Conv2D operations dominate (small matrices, high compute intensity)
- BF16→F32 conversion overhead is not amortized by bandwidth savings
- Conclusion: Keep encoder at F32

### 2.3 Encoder Bottleneck Analysis

Profiling the encoder (F32):

| Phase | Time | % of Encoder |
|-------|------|--------------|
| Conv2D stem (3 layers) | ~1300ms | **67%** |
| Transformer attention | ~620ms | 32% |
| Head projection | ~6ms | <1% |

**Conv2D is the main bottleneck.** Both C and Rust use im2col + GEMM for convolution. The difference is OpenBLAS sgemm vs gemm crate's F32 GEMM.

---

## 3. Root Cause: Redundant Packing

### How GEMM Works

Modern GEMM implementations (OpenBLAS, gemm crate, etc.) reorganize ("pack") input matrices into cache-friendly layouts before computation:

```
A × B = C

1. Pack A into A_packed (cache-friendly layout)
2. Pack B into B_packed (cache-friendly layout)
3. Compute using optimized microkernel
4. (Packing happens every GEMM call)
```

### The Problem

In neural network inference, **weight matrices are constant** after model loading. But both candle and the gemm crate re-pack weights on every forward pass:

```
Forward pass 1: pack(W) → compute
Forward pass 2: pack(W) → compute  ← same W, redundant pack
Forward pass 3: pack(W) → compute  ← same W, redundant pack
...
```

For qwen-asr-rs with ~11s audio:
- Decoder: ~26 tokens × 168 weight matrices = 4,368 redundant packing operations
- Encoder: ~11 chunks × ~20 weight matrices = ~220 redundant packing operations

### Why OpenBLAS is Faster

OpenBLAS has highly optimized packing routines (hand-tuned assembly). The gemm crate's packing is pure Rust. Even though both re-pack every call, OpenBLAS does it faster.

**Pre-packing eliminates this difference** by doing packing once at load time.

---

## 4. Proposed Solution: Weight Pre-packing

### Concept

Pack weight matrices once at model load time, store the packed representation, and use pre-packed GEMM during inference:

```
Model Load:
  W_packed = pack(W)  ← once

Forward pass 1: compute(W_packed)  ← no packing
Forward pass 2: compute(W_packed)  ← no packing
Forward pass 3: compute(W_packed)  ← no packing
```

### Expected Impact

Conservative estimate based on profiling:
- Packing typically takes 10-30% of GEMM time
- Encoder Conv2D: ~1300ms → ~1000-1100ms (15-20% improvement)
- Decoder: Already fast with BF16, but would still benefit

This could close most of the gap with the C implementation.

---

## 5. Implementation Plan

### Phase 1: gemm Crate Changes

**Location:** `github.com/gicrisf/gemm` (bf16 branch)

**New Public API:**

```rust
// 1. Calculate packed buffer size
pub fn packed_rhs_size<T>(k: usize, n: usize) -> usize;
pub fn packed_lhs_size<T>(m: usize, k: usize) -> usize;

// 2. Pack matrices (call once at load time)
pub unsafe fn pack_rhs<T>(
    k: usize, n: usize,
    src: *const T, src_rs: isize, src_cs: isize,
    dst: *mut T,  // pre-allocated buffer
    parallelism: Parallelism,
) -> PackedRhs<T>;

// 3. GEMM with pre-packed operand
pub unsafe fn gemm_with_packed_rhs<T>(
    m: usize, n: usize, k: usize,
    dst: *mut T, dst_cs: isize, dst_rs: isize,
    read_dst: bool,
    lhs: *const T, lhs_cs: isize, lhs_rs: isize,
    packed_rhs: &PackedRhs<T>,  // pre-packed weights
    alpha: T, beta: T,
    parallelism: Parallelism,
);
```

**Challenges:**
- Packed format depends on `KernelParams { kc, mc, nc }` computed at runtime
- Need to either fix these at compile time or store them with `PackedRhs`
- Must handle both F32 (encoder) and BF16 (decoder)

**Estimated effort:** 2-3 days

### Phase 2: candle Changes

**Location:** `github.com/gicrisf/candle` (bf16-gemm branch)

**New Layer Type:**

```rust
// candle-nn/src/packed_linear.rs
pub struct PackedLinear {
    weight: Tensor,           // original weight (for serialization)
    packed_weight: Vec<u8>,   // pre-packed buffer
    bias: Option<Tensor>,
    // Metadata for packed format
    m: usize, k: usize,
}

impl PackedLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let packed = pack_weight(&weight)?;
        Ok(Self { weight, packed_weight: packed, bias, ... })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Use gemm_with_packed_rhs instead of regular gemm
    }
}
```

**Estimated effort:** 1 day

### Phase 3: qwen-asr-rs Changes

**Location:** `qwen-asr-rs/src/encoder.rs`, `decoder.rs`

**Changes:**
- Replace `Linear` with `PackedLinear` for encoder layers
- Replace `Linear` with `PackedLinear` for decoder layers (already BF16)
- Conv2D: Either create `PackedConv2d` or modify candle's Conv2D to use packed weights

**Estimated effort:** 0.5 days

---

## 6. Alternative Approaches Considered

### 6.1 Enable MKL/OpenBLAS in candle

**Pros:** Minimal code changes
**Cons:**
- MKL not optimal for AMD CPUs
- OpenBLAS not exposed as candle feature
- Adds external dependency

### 6.2 Winograd Convolution

**Pros:** Reduces FLOPs by ~2.25x for 3x3 convs
**Cons:**
- Complex implementation
- Only helps Conv2D, not linear layers
- Numerical precision concerns

### 6.3 BF16 Encoder

**Tested and rejected:** 37% slower due to conversion overhead in compute-bound Conv2D.

---

## 7. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Encoder time | ~1.9s | ~1.3s (match C) |
| Decoder time | ~4.1s | ~3.5s (exceed C) |
| Total time | ~6.1s | ~4.8s |
| RT factor | 0.55x | 0.44x |

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Packed format changes between gemm versions | High | Version the packed format, include metadata |
| Increased memory usage (store both original + packed) | Medium | Packed is ~same size as original; acceptable |
| Complexity in gemm crate | Medium | Keep API minimal, good documentation |
| Thread safety of packed buffers | Low | PackedRhs is read-only after creation |

---

## 9. Open Questions

1. **Should we upstream to gemm crate?** The pre-packing API would benefit other users, but adds maintenance burden.

2. **Fixed vs runtime kernel params?** Fixing `KernelParams` at compile time simplifies the API but loses runtime optimization.

3. **Benchmark on Intel?** Pre-packing should help on Intel too, but need to verify.

---

## 10. Next Steps

1. [ ] Review this plan with colleague
2. [ ] Decide on kernel params strategy (fixed vs runtime)
3. [ ] Implement gemm crate changes
4. [ ] Add benchmark comparing packed vs unpacked
5. [ ] Implement candle PackedLinear
6. [ ] Integrate into qwen-asr-rs
7. [ ] Full benchmark suite
8. [ ] Consider upstreaming

---

## Appendix A: Benchmark Commands

```bash
# Warm inference benchmark (current)
cargo run --release --bin bench -- -d ../qwen3-asr-0.6b -i ../samples/jfk.wav -n 10 -t 4

# Cold start benchmark (requires hyperfine)
hyperfine --warmup 2 --runs 10 \
  'target/release/qwen-asr-rs -d ../qwen3-asr-0.6b -i ../samples/jfk.wav --silent'
```

## Appendix B: Key Files

- `gemm/gemm/src/gemm.rs` - Main GEMM implementation, packing functions
- `candle/candle-core/src/cpu_backend/mod.rs:1365` - Where candle calls gemm
- `candle/candle-nn/src/linear.rs` - Linear layer implementation
- `qwen-asr-rs/src/encoder.rs` - Encoder with Conv2D + attention
- `qwen-asr-rs/src/decoder.rs` - Decoder with BF16 linear layers
