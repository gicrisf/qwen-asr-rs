use std::fmt;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv2d, layer_norm, linear, linear_no_bias,
    ops::softmax_last_dim,
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder,
};

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
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
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
    // Weights loaded as F32. File stores BF16 (~356 MB for 0.6b encoder); F32 doubles
    // that to ~712 MB. CPU candle has no BF16 matmul kernel, so F32 is required for now.
    // SAFETY: the safetensors files must not be modified while the Encoder is live.
    pub fn load(paths: &[impl AsRef<Path>], cfg: EncoderConfig, dev: &Device) -> candle_core::Result<Self> {
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_ref()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, dev)? };
        let vb = vb.pp("thinker.audio_tower");

        // Conv stem constants (QWEN_CONV_HIDDEN=480, fixed for both model sizes):
        // three stride-2 convolutions reduce height 128→64→32→16, so c_h=16, flat=480*16=7680.
        const CONV_CH: usize = 480;
        const CONV_FLAT: usize = 480 * 16;
        let conv_cfg = Conv2dConfig { stride: 2, padding: 1, ..Default::default() };
        let conv1   = conv2d(1,       CONV_CH, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2   = conv2d(CONV_CH, CONV_CH, 3, conv_cfg, vb.pp("conv2d2"))?;
        let conv3   = conv2d(CONV_CH, CONV_CH, 3, conv_cfg, vb.pp("conv2d3"))?;
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

        Ok(Encoder { conv1, conv2, conv3, conv_out, layers, ln_post, proj1, proj2, cfg })
    }

    // Input mel: [128, mel_frames] (128 mel bins, F32)
    // Output:    [total_tokens, output_dim]
    pub fn forward(&self, mel: &Tensor) -> candle_core::Result<Tensor> {
        let dev = mel.device();
        let mel_frames = mel.dims()[1];
        let chunk_size = self.cfg.chunk_size;
        let n_chunks = mel_frames.div_ceil(chunk_size);

        // --- Conv2d stem: process each chunk independently ---
        let mut chunks: Vec<Tensor> = Vec::with_capacity(n_chunks);
        let mut tokens_per_ref_chunk: usize = 0;

        for c in 0..n_chunks {
            let start = c * chunk_size;
            let chunk_w = chunk_size.min(mel_frames - start);

            // [128, chunk_w] → [1, 1, 128, chunk_w]  (batch=1, in_ch=1)
            let x = mel.narrow(1, start, chunk_w)?.unsqueeze(0)?.unsqueeze(0)?;

            // Three stride-2 Conv2d + GELU  →  [1, 480, 16, w3]
            let x = self.conv1.forward(&x)?.gelu()?;
            let x = self.conv2.forward(&x)?.gelu()?;
            let x = self.conv3.forward(&x)?.gelu()?;

            let dims = x.dims(); // [1, c_ch, c_h, w3]
            let (c_ch, c_h, w3) = (dims[1], dims[2], dims[3]);
            if c == 0 {
                tokens_per_ref_chunk = w3;
            }

            // [1, 480, 16, w3] → permute → [1, w3, 480, 16] → [w3, 7680]
            let x = x
                .permute([0, 3, 1, 2])?
                .contiguous()?
                .reshape((w3, c_ch * c_h))?;

            // Project to d_model (no bias), add per-chunk sinusoidal PE
            let x = self.conv_out.forward(&x)?;
            let pe = sinusoidal_pe(w3, self.cfg.d_model, dev)?;
            chunks.push((x + pe)?);
        }

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
        Ok(self.proj2.forward(&x)?)
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
        writeln!(f, "  conv1    weight [{}]{}", s(self.conv1.weight()), bias(self.conv1.bias()))?;
        writeln!(f, "  conv2    weight [{}]{}", s(self.conv2.weight()), bias(self.conv2.bias()))?;
        writeln!(f, "  conv3    weight [{}]{}", s(self.conv3.weight()), bias(self.conv3.bias()))?;
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
