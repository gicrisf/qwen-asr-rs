use candle_core::{Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear};
use std::fmt;
use thiserror::Error;

use crate::weights::{Weights, WeightsError};

#[derive(Debug, Error)]
pub enum EncoderError {
    #[error("weights error: {0}")]
    Weights(#[from] WeightsError),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
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
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv_out: Linear,
    layers: Vec<EncLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    cfg: EncoderConfig,
}

// --- private loading helpers ---

// All encoder tensors are stored as BF16 in the file (~356 MB for 0.6b).
// Loading as F32 doubles that to ~712 MB. We accept this for now; switching
// to native BF16 would require using get_bf16 + DType::BF16 throughout.
fn load_tensor(w: &Weights, name: &str, dev: &Device) -> Result<Tensor, EncoderError> {
    let data = w.get_f32(name)?;
    let shape = w.shape_of(name)
        .ok_or_else(|| WeightsError::MissingTensor(name.to_string()))?;
    Ok(Tensor::from_vec(data, shape, dev)?)
}

fn mk_linear(w: &Weights, wname: &str, bname: Option<&str>, dev: &Device) -> Result<Linear, EncoderError> {
    let weight = load_tensor(w, wname, dev)?;
    let bias = bname.map(|n| load_tensor(w, n, dev)).transpose()?;
    Ok(Linear::new(weight, bias))
}

fn mk_layer_norm(w: &Weights, wname: &str, bname: &str, eps: f64, dev: &Device) -> Result<LayerNorm, EncoderError> {
    let weight = load_tensor(w, wname, dev)?;
    let bias = load_tensor(w, bname, dev)?;
    Ok(LayerNorm::new(weight, bias, eps))
}

fn mk_conv2d(w: &Weights, wname: &str, bname: &str, stride: usize, padding: usize, dev: &Device) -> Result<Conv2d, EncoderError> {
    let weight = load_tensor(w, wname, dev)?;
    let bias = load_tensor(w, bname, dev)?;
    let cfg = Conv2dConfig { stride, padding, ..Default::default() };
    Ok(Conv2d::new(weight, Some(bias), cfg))
}

// --- Encoder::load ---

impl Encoder {
    pub fn load(w: &Weights, cfg: EncoderConfig, dev: &Device) -> Result<Self, EncoderError> {
        const P: &str = "thinker.audio_tower.";

        let conv1    = mk_conv2d(w, &format!("{P}conv2d1.weight"), &format!("{P}conv2d1.bias"), 2, 1, dev)?;
        let conv2    = mk_conv2d(w, &format!("{P}conv2d2.weight"), &format!("{P}conv2d2.bias"), 2, 1, dev)?;
        let conv3    = mk_conv2d(w, &format!("{P}conv2d3.weight"), &format!("{P}conv2d3.bias"), 2, 1, dev)?;
        let conv_out = mk_linear(w, &format!("{P}conv_out.weight"), None, dev)?;

        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let lp = format!("{P}layers.{i}");
            layers.push(EncLayer {
                q_proj:    mk_linear(w, &format!("{lp}.self_attn.q_proj.weight"),   Some(&format!("{lp}.self_attn.q_proj.bias")),   dev)?,
                k_proj:    mk_linear(w, &format!("{lp}.self_attn.k_proj.weight"),   Some(&format!("{lp}.self_attn.k_proj.bias")),   dev)?,
                v_proj:    mk_linear(w, &format!("{lp}.self_attn.v_proj.weight"),   Some(&format!("{lp}.self_attn.v_proj.bias")),   dev)?,
                o_proj:    mk_linear(w, &format!("{lp}.self_attn.out_proj.weight"), Some(&format!("{lp}.self_attn.out_proj.bias")), dev)?,
                attn_norm: mk_layer_norm(w, &format!("{lp}.self_attn_layer_norm.weight"), &format!("{lp}.self_attn_layer_norm.bias"), 1e-5, dev)?,
                mlp: Mlp {
                    fc1: mk_linear(w, &format!("{lp}.fc1.weight"), Some(&format!("{lp}.fc1.bias")), dev)?,
                    fc2: mk_linear(w, &format!("{lp}.fc2.weight"), Some(&format!("{lp}.fc2.bias")), dev)?,
                },
                ffn_norm: mk_layer_norm(w, &format!("{lp}.final_layer_norm.weight"), &format!("{lp}.final_layer_norm.bias"), 1e-5, dev)?,
            });
        }

        let ln_post = mk_layer_norm(w, &format!("{P}ln_post.weight"), &format!("{P}ln_post.bias"), 1e-5, dev)?;
        let proj1   = mk_linear(w, &format!("{P}proj1.weight"), Some(&format!("{P}proj1.bias")), dev)?;
        let proj2   = mk_linear(w, &format!("{P}proj2.weight"), Some(&format!("{P}proj2.bias")), dev)?;

        Ok(Encoder { conv1, conv2, conv3, conv_out, layers, ln_post, proj1, proj2, cfg })
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
    use crate::weights::Weights;
    use std::env;
    use std::path::PathBuf;

    fn smoke_model_path() -> PathBuf {
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
        let shard = smoke_model_path();
        let weights = Weights::from_files(&[&shard]).expect("failed to load weights");

        let cfg = ModelPreset::from(&weights).config().encoder;
        let enc = Encoder::load(&weights, cfg.clone(), &Device::Cpu).expect("Encoder::load failed");
        assert_eq!(enc.cfg, cfg);
        println!("{}", enc);
    }
}
