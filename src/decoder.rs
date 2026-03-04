// Qwen3 causal decoder (our own implementation so we can inject encoder embeddings
// at audio-pad token positions: candle's ModelForCausalLM only accepts token IDs).
//
// Weights loaded as F32. CPU candle has no BF16 matmul kernel.
// SAFETY: the safetensors files must not be modified while the Decoder is live.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    embedding, kv_cache::ConcatKvCache, linear_no_bias, ops::softmax_last_dim,
    rms_norm, rotary_emb::rope, Embedding, Linear, RmsNorm, VarBuilder,
};
use candle_transformers::models::qwen3::Config as Qwen3Config;

// ── RoPE cache ────────────────────────────────────────────────────────────────

struct RopeCache {
    sin: Tensor,
    cos: Tensor,
}

impl RopeCache {
    fn new(cfg: &Qwen3Config, dev: &Device) -> Result<Self> {
        let half = cfg.head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / cfg.head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?; // [max_pos, half]
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq)?.contiguous()?;
        let sin = self.sin.narrow(0, offset, seq)?.contiguous()?;
        let dtype = q.dtype();
        let q = rope(&q.to_dtype(DType::F32)?.contiguous()?, &cos, &sin)?
            .to_dtype(dtype)?;
        let k = rope(&k.to_dtype(DType::F32)?.contiguous()?, &cos, &sin)?
            .to_dtype(dtype)?;
        Ok((q, k))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// Repeat KV heads to match the query head count (GQA).
fn repeat_kv(t: Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(t);
    }
    let (b, h, l, d) = t.dims4()?;
    t.unsqueeze(2)?.expand((b, h, n, l, d))?.reshape((b, h * n, l, d))
}

// Linear with optional bias (used to respect config.attention_bias).
fn linear_b(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

// Causal mask: 0 where attended, -inf where not. Shape [b, 1, tgt, tgt+offset].
fn causal_mask(b: usize, tgt: usize, offset: usize, dev: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..tgt)
        .flat_map(|i| {
            (0..(tgt + offset)).map(move |j| {
                if j <= i + offset { 0.0 } else { f32::NEG_INFINITY }
            })
        })
        .collect();
    Tensor::from_vec(mask, (b, 1, tgt, tgt + offset), dev)
}

// ── GQA Attention ─────────────────────────────────────────────────────────────

struct DecAttn {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm:  RmsNorm,
    k_norm:  RmsNorm,
    kv_cache: ConcatKvCache,
    n_heads:     usize,
    n_kv_heads:  usize,
    n_kv_groups: usize,
    head_dim:    usize,
    hidden_size: usize,
    rope: Arc<RopeCache>,
}

impl DecAttn {
    fn new(cfg: &Qwen3Config, rope: Arc<RopeCache>, vb: VarBuilder) -> Result<Self> {
        let n_heads    = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim   = cfg.head_dim;
        let bias       = cfg.attention_bias;
        Ok(Self {
            q_proj: linear_b(cfg.hidden_size, n_heads    * head_dim, bias, vb.pp("q_proj"))?,
            k_proj: linear_b(cfg.hidden_size, n_kv_heads * head_dim, bias, vb.pp("k_proj"))?,
            v_proj: linear_b(cfg.hidden_size, n_kv_heads * head_dim, bias, vb.pp("v_proj"))?,
            o_proj: linear_b(n_heads * head_dim, cfg.hidden_size,    bias, vb.pp("o_proj"))?,
            q_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            kv_cache: ConcatKvCache::new(2), // concat along sequence dim
            n_heads,
            n_kv_heads,
            n_kv_groups: n_heads / n_kv_heads,
            head_dim,
            hidden_size: n_heads * head_dim,
            rope,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // [b, l, H*D] → [b, H, l, D]
        let q = q.reshape((b, l, self.n_heads,    self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, l, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, l, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Per-head RMSNorm: flatten (b,H,l,D)→(b*H*l,D), norm, reshape back
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((b, self.n_heads,    l, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((b, self.n_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rope.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.n_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(&m.to_dtype(scores.dtype())?)?;
        }
        let probs = softmax_last_dim(&scores)?;

        probs.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_cache(&mut self) { self.kv_cache.reset(); }
}

// ── SwiGLU FFN ────────────────────────────────────────────────────────────────

struct DecMlp {
    gate_proj: Linear,
    up_proj:   Linear,
    down_proj: Linear,
}

impl DecMlp {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj:   linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up   = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Transformer layer ─────────────────────────────────────────────────────────

struct DecLayer {
    attn: DecAttn,
    mlp:  DecMlp,
    ln1:  RmsNorm,
    ln2:  RmsNorm,
}

impl DecLayer {
    fn new(cfg: &Qwen3Config, rope: Arc<RopeCache>, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: DecAttn::new(cfg, rope, vb.pp("self_attn"))?,
            mlp:  DecMlp::new(cfg, vb.pp("mlp"))?,
            ln1:  rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2:  rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let dtype = x.dtype();
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h, mask, offset)?.to_dtype(dtype)?;
        let x = (x + h)?;
        let h2 = self.mlp.forward(&self.ln2.forward(&x)?)?.to_dtype(dtype)?;
        x + h2
    }

    fn clear_cache(&mut self) { self.attn.clear_cache(); }
}

// ── Public Decoder ────────────────────────────────────────────────────────────

pub struct Decoder {
    embed_tokens: Embedding,
    layers: Vec<DecLayer>,
    norm:     RmsNorm,
    lm_head:  Linear,
    cfg: Qwen3Config,
}

impl Decoder {
    pub fn load(paths: &[impl AsRef<Path>], cfg: &Qwen3Config, dev: &Device) -> Result<Self> {
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_ref()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, dev)? };
        let vb = vb.pp("thinker");

        let rope = Arc::new(RopeCache::new(cfg, dev)?);

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        // Tied weights: lm_head shares the embed_tokens matrix (qwen3 tie_word_embeddings=true).
        let lm_head = Linear::new(embed_tokens.embeddings().clone(), None);

        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecLayer::new(cfg, rope.clone(), vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        Ok(Self { embed_tokens, layers, norm, lm_head, cfg: cfg.clone() })
    }

    /// Look up token embeddings: `[b, seq]` → `[b, seq, hidden]`.
    pub fn embed(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(token_ids)
    }

    /// Run the transformer on pre-built embeddings `[b, seq, hidden]`.
    /// `offset` = number of KV-cache tokens already stored from previous calls.
    /// Returns logits for the **last** token: `[b, vocab_size]`.
    pub fn forward_with_embeds(&mut self, embeds: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l, _) = embeds.dims3()?;
        let dev = embeds.device();
        let mask = if l == 1 { None } else { Some(causal_mask(b, l, offset, dev)?) };

        let mut h = embeds.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }
        let last = self.norm.forward(&h)?.narrow(1, l - 1, 1)?.squeeze(1)?; // [b, hidden]
        last.apply(&self.lm_head) // [b, vocab_size]
    }

    /// Autoregressive step: embed one token ID, run decoder, return argmax.
    /// KV cache is updated in-place; `offset` must equal current cache length.
    pub fn step(&mut self, token_id: u32, offset: usize) -> Result<u32> {
        let dev = self.embed_tokens.embeddings().device();
        let ids = Tensor::from_vec(vec![token_id], (1, 1), dev)?;
        let embeds = self.embed_tokens.forward(&ids)?;
        let logits = self.forward_with_embeds(&embeds, offset)?; // [1, vocab_size]
        logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::ModelPreset;
    use std::env;
    use std::path::PathBuf;

    fn smoke_shard_path() -> PathBuf {
        if let Ok(p) = env::var("QWEN_ASR_MODEL_DIR") {
            let p = PathBuf::from(p);
            return if p.is_dir() { p.join("model.safetensors") } else { p };
        }
        if let Ok(root) = env::var("QWEN_ASR_ROOT") {
            return PathBuf::from(root)
                .join("qwen3-asr-0.6b")
                .join("model.safetensors");
        }
        panic!("Set QWEN_ASR_MODEL_DIR or QWEN_ASR_ROOT");
    }

    #[test]
    #[ignore]
    fn load_0_6b_decoder_smoke() {
        let shard = smoke_shard_path();
        let cfg = ModelPreset::Qwen3Asr0_6b.config().decoder;
        let dec = Decoder::load(&[&shard], &cfg, &Device::Cpu).expect("load failed");
        assert_eq!(dec.cfg.hidden_size, cfg.hidden_size);
        assert_eq!(dec.cfg.num_hidden_layers, cfg.num_hidden_layers);
    }

    #[test]
    #[ignore]
    // Reference from: QWEN_DEBUG_DEC_TOKEN=151644 ./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav
    //   DEC_DEBUG embed[0..4] = -0.0075378418 -0.097167969 0.016113281 0.047607422
    fn embed_lookup_matches_c_reference() {
        let shard = smoke_shard_path();
        let cfg = ModelPreset::Qwen3Asr0_6b.config().decoder;
        let dec = Decoder::load(&[&shard], &cfg, &Device::Cpu).expect("load failed");

        let ids = Tensor::from_vec(vec![151644u32], (1, 1), &Device::Cpu).unwrap();
        let embeds = dec.embed(&ids).unwrap(); // [1, 1, hidden]
        let row = embeds.squeeze(0).unwrap().squeeze(0).unwrap()
            .narrow(0, 0, 4).unwrap()
            .to_vec1::<f32>().unwrap();

        let reference = [-0.0075378418f32, -0.097167969, 0.016113281, 0.047607422];
        for (i, (&got, &expected)) in row.iter().zip(reference.iter()).enumerate() {
            let diff = (got - expected).abs();
            assert!(diff < 1e-4, "embed[151644][{i}]: got {got:.8} expected {expected:.8}");
        }
    }
}
