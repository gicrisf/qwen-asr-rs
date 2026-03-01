use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM};

// Weights loaded as F32. File stores BF16 (~290 MB for 0.6b decoder); F32 doubles
// that to ~580 MB. CPU candle has no BF16 matmul kernel, so F32 is required for now.
// SAFETY: the safetensors files must not be modified while the returned model is live.
pub fn load(
    paths: &[impl AsRef<Path>],
    cfg: &Qwen3Config,
    dev: &Device,
) -> candle_core::Result<ModelForCausalLM> {
    let paths: Vec<&Path> = paths.iter().map(|p| p.as_ref()).collect();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, dev)? };
    ModelForCausalLM::new(cfg, vb.pp("thinker"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::ModelPreset;
    use crate::weights::Weights;
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

    // Full end-to-end logit comparison against C requires the threading context
    // that qwen_transcribe sets up — qwen_decoder_forward is ~100x slower without
    // it (311 MB BF16 lm_head matvec, single-threaded).
    // Real numerical verification will happen as part of the pipeline test.
    // For now we verify: (a) correct weight layout/loading, (b) embed lookup.

    #[test]
    #[ignore]
    // Reference from: QWEN_DEBUG_DEC_TOKEN=151644 ./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav
    //   DEC_DEBUG embed[0..4] = -0.0075378418 -0.097167969 0.016113281 0.047607422
    fn embed_lookup_matches_c_reference() {
        let shard = smoke_shard_path();
        let weights = Weights::from_files(&[&shard]).expect("failed to load weights");
        let cfg = ModelPreset::from(&weights).config().decoder;
        let mut dec = load(&[&shard], &cfg, &Device::Cpu).expect("load failed");

        // Feed token 151644 (<|im_start|>) and get logits — the embed is looked up
        // internally. We verify it by checking the logits are finite and the argmax
        // is a valid token, then separately assert the embed values via the weight.
        let token = Tensor::from_vec(vec![151644u32], (1, 1), &Device::Cpu).unwrap();
        let logits = dec.forward(&token, 0).expect("forward failed");
        let logits_vec = logits.squeeze(0).unwrap().squeeze(0).unwrap()
            .to_vec1::<f32>().unwrap();
        assert!(logits_vec.iter().all(|v| v.is_finite()), "logits contain NaN/inf");

        // Verify the embedding of token 151644 matches the C reference.
        // The Decoder loads embed_tokens from thinker.model.embed_tokens.weight [vocab, hidden].
        // We load that tensor independently and check the first 4 values of row 151644.
        let embed_w = weights.get_f32("thinker.model.embed_tokens.weight")
            .expect("embed_tokens weight");
        let hidden = cfg.hidden_size;
        let row = &embed_w[151644 * hidden .. 151644 * hidden + 4];
        let reference = [-0.0075378418f32, -0.097167969, 0.016113281, 0.047607422];
        for (i, (&got, &expected)) in row.iter().zip(reference.iter()).enumerate() {
            let diff = (got - expected).abs();
            assert!(diff < 1e-4, "embed[151644][{i}]: got {got:.8} expected {expected:.8}");
        }
    }

    #[test]
    #[ignore]
    fn load_0_6b_decoder_smoke() {
        let shard = smoke_shard_path();

        let weights = Weights::from_files(&[&shard]).expect("failed to load weights");
        let cfg = ModelPreset::from(&weights).config().decoder;

        let mut dec = load(&[&shard], &cfg, &Device::Cpu).expect("load failed");

        // One token forward: token 0, offset 0  →  logits [1, 1, vocab_size]
        let token = Tensor::from_vec(vec![0u32], (1, 1), &Device::Cpu).unwrap();
        let logits = dec.forward(&token, 0).expect("forward failed");

        let (b, s, v) = logits.dims3().unwrap();
        println!("logits shape: [{b}, {s}, {v}]");
        assert_eq!(b, 1);
        assert_eq!(s, 1);
        assert_eq!(v, cfg.vocab_size);
    }
}
