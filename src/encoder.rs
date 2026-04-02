use std::fmt;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    layer_norm, linear, linear_no_bias,
    ops::softmax_last_dim,
    LayerNorm, Linear, Module, VarBuilder,
};
use rayon::prelude::*;

// --- Custom im2col + GEMM conv2d (bypass candle for perf) ---

/// im2col: Transform input [c_in, h_in, w_in] -> column matrix [patch_size, spatial_out]
/// Direct port from qwen_asr_kernels.c:566-590
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

/// Conv2d using im2col + gemm. Returns output [c_out, h_out, w_out] flattened.
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

    // im2col: input -> cols [patch_size, spatial_out]
    let cols = im2col(input, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    // GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out]
    let mut output = vec![0.0f32; c_out * spatial_out];

    // gemm expects: dst_cs, dst_rs (column stride, then row stride)
    // For row-major matrices:
    //   weight: [c_out, patch_size] -> cs=1, rs=patch_size
    //   cols:   [patch_size, spatial_out] -> cs=1, rs=spatial_out
    //   output: [c_out, spatial_out] -> cs=1, rs=spatial_out
    unsafe {
        gemm::gemm(
            c_out,                   // m (rows of A and C)
            spatial_out,             // n (cols of B and C)
            patch_size,              // k (cols of A, rows of B)
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
            0.0,                     // alpha (C = alpha*C + beta*A*B, don't read dst)
            1.0,                     // beta
            false, false, false,     // conj_dst, conj_lhs, conj_rhs
            gemm::Parallelism::None, // Single-threaded (chunks parallelized via rayon)
        );
    }

    // Add bias (broadcast across spatial dims)
    for oc in 0..c_out {
        let b = bias[oc];
        let row_start = oc * spatial_out;
        for s in 0..spatial_out {
            output[row_start + s] += b;
        }
    }

    (output, h_out, w_out)
}

/// GELU activation in-place (exact formula matching candle/C)
fn gelu_inplace(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;

    for v in x.iter_mut() {
        let x3 = *v * *v * *v;
        let inner = SQRT_2_OVER_PI * (*v + COEFF * x3);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}

/// Raw conv stem weights for direct GEMM
struct ConvStem {
    // Conv1: [1, 128, W] -> [480, 64, W1], kernel 3x3, stride 2, pad 1
    conv1_weight: Vec<f32>,  // [480, 1*3*3] = [480, 9]
    conv1_bias: Vec<f32>,    // [480]
    // Conv2: [480, H1, W1] -> [480, H2, W2]
    conv2_weight: Vec<f32>,  // [480, 480*3*3] = [480, 4320]
    conv2_bias: Vec<f32>,    // [480]
    // Conv3: [480, H2, W2] -> [480, H3, W3]
    conv3_weight: Vec<f32>,  // [480, 480*3*3] = [480, 4320]
    conv3_bias: Vec<f32>,    // [480]
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub output_dim: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub chunk_size: usize,
}

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

#[derive(Debug, Clone)]
struct EncLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    attn_norm: LayerNorm,
    mlp: Mlp,
    ffn_norm: LayerNorm,
}

pub struct Encoder {
    conv_stem: ConvStem,
    conv_out: Linear,
    layers: Vec<EncLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    pub cfg: EncoderConfig,
}

// --- forward-pass helpers ---

// Sinusoidal PE [n_pos, d_model], per-chunk starting from position 0.
// NOTE: this is NOT the standard transformer PE. Two differences matter:
//   1. log_timescale = log(10000) / (half - 1), not the usual 2i/d_model
//   2. layout: sines in row[0..half], cosines in row[half..d_model] (not interleaved)
// See qwen_sinusoidal_pe() in qwen_asr_kernels.c.
fn sinusoidal_pe(n_pos: usize, d_model: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let half = d_model / 2;
    let log_timescale = 10000f32.ln() / (half - 1) as f32;
    let mut data = vec![0f32; n_pos * d_model];
    for p in 0..n_pos {
        for d in 0..half {
            let angle = p as f32 * (-(d as f32) * log_timescale).exp();
            data[p * d_model + d]        = angle.sin();
            data[p * d_model + half + d] = angle.cos();
        }
    }
    Ok(Tensor::from_vec(data, (n_pos, d_model), dev)?)
}

// Attention bias: 0.0 within each window, -inf across windows [total, total].
// Windows are contiguous, non-overlapping, each of `window_size` tokens.
fn window_mask(total: usize, window_size: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; total * total];
    let mut start = 0;
    while start < total {
        let end = (start + window_size).min(total);
        for i in start..end {
            for j in start..end {
                data[i * total + j] = 0.0;
            }
        }
        start += window_size;
    }
    Ok(Tensor::from_vec(data, (total, total), dev)?)
}

// --- EncLayer forward ---

impl EncLayer {
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        n_heads: usize,
        head_dim: usize,
    ) -> candle_core::Result<Tensor> {
        let seq = x.dims()[0];

        // Self-attention (pre-norm)
        let xn = self.attn_norm.forward(x)?;
        let q = self.q_proj.forward(&xn)?;
        let k = self.k_proj.forward(&xn)?;
        let v = self.v_proj.forward(&xn)?;

        // [seq, d_model] → [n_heads, seq, head_dim]
        let q = q.reshape((seq, n_heads, head_dim))?.transpose(0, 1)?;
        let k = k.reshape((seq, n_heads, head_dim))?.transpose(0, 1)?;
        let v = v.reshape((seq, n_heads, head_dim))?.transpose(0, 1)?.contiguous()?;

        let scale = (head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        // mask [total, total] broadcasts over the heads dim
        let scores = scores.broadcast_add(mask)?;
        let weights = softmax_last_dim(&scores)?;

        // [n_heads, seq, head_dim] → [seq, d_model]
        let out = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .contiguous()?
            .flatten_from(1)?;

        let out = self.o_proj.forward(&out)?;
        let x = (x + out)?;

        // FFN (pre-norm)
        let xn = self.ffn_norm.forward(&x)?;
        let h = self.mlp.fc1.forward(&xn)?.gelu()?;
        let h = self.mlp.fc2.forward(&h)?;
        Ok((&x + h)?)
    }
}

// --- Encoder::load ---

impl Encoder {
    // Encoder weights loaded as F32: the mel input is F32, and Conv2D requires matching
    // dtypes. The decoder uses BF16 weights where the large linear layers benefit most.
    // SAFETY: the safetensors files must not be modified while the Encoder is live.
    pub fn load(paths: &[impl AsRef<Path>], cfg: EncoderConfig, dev: &Device) -> candle_core::Result<Self> {
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_ref()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, dev)? };
        let vb = vb.pp("thinker.audio_tower");

        // Conv stem constants (QWEN_CONV_HIDDEN=480, fixed for both model sizes):
        // three stride-2 convolutions reduce height 128→64→32→16, so c_h=16, flat=480*16=7680.
        const CONV_CH: usize = 480;
        const CONV_FLAT: usize = 480 * 16;

        // Load conv weights as raw Vec<f32> for direct GEMM.
        // Candle stores conv weights as [c_out, c_in, kh, kw], we need [c_out, c_in*kh*kw].
        let conv1_w_tensor = vb.pp("conv2d1").get((CONV_CH, 1, 3, 3), "weight")?;
        let conv1_weight: Vec<f32> = conv1_w_tensor.flatten_all()?.to_vec1()?;
        let conv1_bias: Vec<f32> = vb.pp("conv2d1").get(CONV_CH, "bias")?.to_vec1()?;

        let conv2_w_tensor = vb.pp("conv2d2").get((CONV_CH, CONV_CH, 3, 3), "weight")?;
        let conv2_weight: Vec<f32> = conv2_w_tensor.flatten_all()?.to_vec1()?;
        let conv2_bias: Vec<f32> = vb.pp("conv2d2").get(CONV_CH, "bias")?.to_vec1()?;

        let conv3_w_tensor = vb.pp("conv2d3").get((CONV_CH, CONV_CH, 3, 3), "weight")?;
        let conv3_weight: Vec<f32> = conv3_w_tensor.flatten_all()?.to_vec1()?;
        let conv3_bias: Vec<f32> = vb.pp("conv2d3").get(CONV_CH, "bias")?.to_vec1()?;

        let conv_stem = ConvStem {
            conv1_weight, conv1_bias,
            conv2_weight, conv2_bias,
            conv3_weight, conv3_bias,
        };

        let conv_out = linear_no_bias(CONV_FLAT, cfg.d_model, vb.pp("conv_out"))?;

        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let lp = vb.pp(format!("layers.{i}"));
            layers.push(EncLayer {
                q_proj:    linear(cfg.d_model, cfg.d_model, lp.pp("self_attn.q_proj"))?,
                k_proj:    linear(cfg.d_model, cfg.d_model, lp.pp("self_attn.k_proj"))?,
                v_proj:    linear(cfg.d_model, cfg.d_model, lp.pp("self_attn.v_proj"))?,
                o_proj:    linear(cfg.d_model, cfg.d_model, lp.pp("self_attn.out_proj"))?,
                attn_norm: layer_norm(cfg.d_model, 1e-5, lp.pp("self_attn_layer_norm"))?,
                mlp: Mlp {
                    fc1: linear(cfg.d_model, cfg.ffn_dim, lp.pp("fc1"))?,
                    fc2: linear(cfg.ffn_dim, cfg.d_model, lp.pp("fc2"))?,
                },
                ffn_norm: layer_norm(cfg.d_model, 1e-5, lp.pp("final_layer_norm"))?,
            });
        }

        let ln_post = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
        let proj1   = linear(cfg.d_model, cfg.d_model,    vb.pp("proj1"))?;
        let proj2   = linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;

        Ok(Encoder { conv_stem, conv_out, layers, ln_post, proj1, proj2, cfg })
    }

    // Input mel: [128, mel_frames] (128 mel bins, F32)
    // Output:    [total_tokens, output_dim]
    pub fn forward(&self, mel: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_inner(mel, false).map(|(t, _, _)| t)
    }

    // Same as forward but returns timing breakdown (conv_stem_ms, transformer_ms)
    pub fn forward_timed(&self, mel: &Tensor) -> candle_core::Result<(Tensor, f64, f64)> {
        self.forward_inner(mel, true)
    }

    fn forward_inner(&self, mel: &Tensor, timed: bool) -> candle_core::Result<(Tensor, f64, f64)> {
        use std::time::Instant;
        let t0 = if timed { Some(Instant::now()) } else { None };

        let dev = mel.device();
        let mel_frames = mel.dims()[1];
        let chunk_size = self.cfg.chunk_size;
        let n_chunks = mel_frames.div_ceil(chunk_size);

        // Extract mel as flat f32 slice [128, mel_frames] row-major
        let mel_data: Vec<f32> = mel.flatten_all()?.to_vec1()?;

        // Conv stem constants
        const CONV_CH: usize = 480;
        const MEL_BINS: usize = 128;

        // --- Conv2d stem: process chunks in parallel using custom im2col + GEMM ---
        // Process conv stem (3 conv layers) in parallel across chunks, then project sequentially
        let conv_results: Vec<_> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let start = c * chunk_size;
                let chunk_w = chunk_size.min(mel_frames - start);

                // Extract chunk: [1, 128, chunk_w] (c_in=1, h=128, w=chunk_w)
                let mut chunk_mel = vec![0.0f32; MEL_BINS * chunk_w];
                for m in 0..MEL_BINS {
                    for w in 0..chunk_w {
                        chunk_mel[m * chunk_w + w] = mel_data[m * mel_frames + start + w];
                    }
                }

                // Conv1: [1, 128, chunk_w] -> [480, h1, w1]
                let (mut x, h1, w1) = conv2d_gemm(
                    &chunk_mel, &self.conv_stem.conv1_weight, &self.conv_stem.conv1_bias,
                    1, CONV_CH, MEL_BINS, chunk_w, 3, 3, 2, 1,
                );
                gelu_inplace(&mut x);

                // Conv2: [480, h1, w1] -> [480, h2, w2]
                let (mut x, h2, w2) = conv2d_gemm(
                    &x, &self.conv_stem.conv2_weight, &self.conv_stem.conv2_bias,
                    CONV_CH, CONV_CH, h1, w1, 3, 3, 2, 1,
                );
                gelu_inplace(&mut x);

                // Conv3: [480, h2, w2] -> [480, h3, w3]
                let (mut x, h3, w3) = conv2d_gemm(
                    &x, &self.conv_stem.conv3_weight, &self.conv_stem.conv3_bias,
                    CONV_CH, CONV_CH, h2, w2, 3, 3, 2, 1,
                );
                gelu_inplace(&mut x);

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
                (reshaped, w3, h3)
            })
            .collect();

        // Project and add PE sequentially (requires Tensor operations)
        let mut chunks: Vec<Tensor> = Vec::with_capacity(n_chunks);
        let mut tokens_per_ref_chunk: usize = 0;

        for (c, (reshaped, w3, h3)) in conv_results.into_iter().enumerate() {
            if c == 0 {
                tokens_per_ref_chunk = w3;
            }
            let x_tensor = Tensor::from_vec(reshaped, (w3, CONV_CH * h3), dev)?;
            let x_proj = self.conv_out.forward(&x_tensor)?;
            let pe = sinusoidal_pe(w3, self.cfg.d_model, dev)?;
            chunks.push((x_proj + pe)?);
        }

        let conv_stem_ms = t0.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        let t1 = if timed { Some(std::time::Instant::now()) } else { None };

        // --- Transformer ---
        let mut x = Tensor::cat(&chunks, 0)?; // [total_tokens, d_model]
        let total_tokens = x.dims()[0];

        // Window = tokens_per_ref_chunk * (n_window_infer / chunk_size)
        // e.g. 13 * (800 / 100) = 104 tokens per attention window
        let window_size = tokens_per_ref_chunk * (self.cfg.n_window_infer / self.cfg.chunk_size);
        let mask = window_mask(total_tokens, window_size, dev)?;

        for layer in &self.layers {
            x = layer.forward(&x, &mask, self.cfg.heads, self.cfg.head_dim)?;
        }

        // --- Head ---
        let x = self.ln_post.forward(&x)?;
        let x = self.proj1.forward(&x)?.gelu()?;
        let out = self.proj2.forward(&x)?;

        let transformer_ms = t1.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
        Ok((out, conv_stem_ms, transformer_ms))
    }
}

impl std::fmt::Display for Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn s(t: &Tensor) -> String {
            t.dims().iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×")
        }
        fn bias(t: Option<&Tensor>) -> String {
            t.map_or(String::new(), |b| format!("  bias [{}]", s(b)))
        }
        fn ln(norm: &LayerNorm) -> String {
            format!("[{}]  bias [{}]", s(norm.weight()), s(norm.bias().unwrap()))
        }

        writeln!(f, "Encoder {{")?;
        writeln!(f, "  conv1    weight [480×9]  bias [480]")?;
        writeln!(f, "  conv2    weight [480×4320]  bias [480]")?;
        writeln!(f, "  conv3    weight [480×4320]  bias [480]")?;
        writeln!(f, "  conv_out weight [{}]{}", s(self.conv_out.weight()), bias(self.conv_out.bias()))?;
        writeln!(f, "  layers   {} ×", self.layers.len())?;
        if let Some(l) = self.layers.first() {
            writeln!(f, "    q_proj    weight [{}]{}", s(l.q_proj.weight()), bias(l.q_proj.bias()))?;
            writeln!(f, "    k_proj    weight [{}]{}", s(l.k_proj.weight()), bias(l.k_proj.bias()))?;
            writeln!(f, "    v_proj    weight [{}]{}", s(l.v_proj.weight()), bias(l.v_proj.bias()))?;
            writeln!(f, "    o_proj    weight [{}]{}", s(l.o_proj.weight()), bias(l.o_proj.bias()))?;
            writeln!(f, "    attn_norm {}", ln(&l.attn_norm))?;
            writeln!(f, "    fc1       weight [{}]{}", s(l.mlp.fc1.weight()), bias(l.mlp.fc1.bias()))?;
            writeln!(f, "    fc2       weight [{}]{}", s(l.mlp.fc2.weight()), bias(l.mlp.fc2.bias()))?;
            writeln!(f, "    ffn_norm  {}", ln(&l.ffn_norm))?;
        }
        writeln!(f, "  ln_post  {}", ln(&self.ln_post))?;
        writeln!(f, "  proj1    weight [{}]{}", s(self.proj1.weight()), bias(self.proj1.bias()))?;
        writeln!(f, "  proj2    weight [{}]{}", s(self.proj2.weight()), bias(self.proj2.bias()))?;
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::ModelPreset;
    use std::env;
    use std::path::PathBuf;

    fn smoke_shard_path() -> PathBuf {
        if let Ok(model_dir) = env::var("QWEN_ASR_MODEL_DIR") {
            let p = PathBuf::from(model_dir);
            return if p.is_dir() { p.join("model.safetensors") } else { p };
        }
        if let Ok(root) = env::var("QWEN_ASR_ROOT") {
            return PathBuf::from(root).join("qwen3-asr-0.6b").join("model.safetensors");
        }
        panic!(
            "Set QWEN_ASR_MODEL_DIR=/abs/path/to/model.safetensors \
or QWEN_ASR_ROOT=/abs/path/to/repo-root"
        );
    }

    #[test]
    #[ignore]
    fn load_0_6b_encoder_smoke() {
        let shard = smoke_shard_path();
        let cfg = ModelPreset::Qwen3Asr0_6b.config().encoder;
        let enc = Encoder::load(&[&shard], cfg.clone(), &Device::Cpu).expect("Encoder::load failed");
        assert_eq!(enc.cfg, cfg);
        println!("{}", enc);
    }

    #[test]
    #[ignore]
    fn forward_0_6b_matches_c_reference() {
        // Reference values from:
        //   QWEN_DUMP_MEL=/tmp/jfk_mel.bin QWEN_DEBUG_ENC=1 \
        //   ./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --silent
        #[rustfmt::skip]
        let reference: &[(usize, usize, f32)] = &[
            (0, 0,  0.043440323), (0, 1, -0.015488941), (0, 2, -0.032936737), (0, 3,  0.023766715),
            (1, 0,  0.005729813), (1, 1, -0.019771859), (1, 2, -0.035450991), (1, 3,  0.002760586),
            (2, 0,  0.008599119), (2, 1, -0.003811852), (2, 2, -0.023442566), (2, 3,  0.001752629),
            (3, 0,  0.016011002), (3, 1, -0.007761396), (3, 2, -0.017930379), (3, 3,  0.006804271),
        ];

        // Load mel dumped by C binary (4-byte int mel_frames, then [128, mel_frames] f32)
        let mel_bin = std::fs::read("/tmp/jfk_mel.bin")
            .expect("run: QWEN_DUMP_MEL=/tmp/jfk_mel.bin ./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --silent");
        let mel_frames = i32::from_le_bytes(mel_bin[..4].try_into().unwrap()) as usize;
        let floats: Vec<f32> = mel_bin[4..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mel = Tensor::from_vec(floats, (128, mel_frames), &Device::Cpu).unwrap();

        let shard = smoke_shard_path();
        let cfg = ModelPreset::Qwen3Asr0_6b.config().encoder;
        let enc = Encoder::load(&[&shard], cfg, &Device::Cpu).expect("Encoder::load failed");

        let out = enc.forward(&mel).expect("forward failed");
        let out_vec = out.to_vec2::<f32>().unwrap();

        assert_eq!(out_vec.len(), 143, "expected 143 total tokens");

        for &(t, d, expected) in reference {
            let got = out_vec[t][d];
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-4,
                "out[{t}][{d}]: got {got:.8} expected {expected:.8} diff {diff:.2e}"
            );
        }
    }

    #[test]
    #[ignore]
    fn forward_0_6b_shape() {
        let shard = smoke_shard_path();
        let cfg = ModelPreset::Qwen3Asr0_6b.config().encoder;
        let enc = Encoder::load(&[&shard], cfg.clone(), &Device::Cpu).expect("Encoder::load failed");

        // One full chunk of silence
        let mel = Tensor::zeros((128, cfg.chunk_size), DType::F32, &Device::Cpu).unwrap();
        let out = enc.forward(&mel).expect("forward failed");

        println!("output shape: {:?}", out.dims());
        // For chunk_size=100: w3 = 13 tokens, output_dim = 1024
        assert_eq!(out.dims()[1], cfg.output_dim);
    }
}
