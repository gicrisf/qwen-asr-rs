use crate::audio::AudioConfig;
use crate::decoder::DecoderConfig;
use crate::encoder::EncoderConfig;
use crate::model::ModelConfig;
use crate::weights::Weights;

use std::convert::From;

pub enum ModelPreset {
    Qwen3Asr0_6b,
    Qwen3Asr1_7b,
}

impl From<&Weights> for ModelPreset {
    fn from(weights: &Weights) -> Self {
        if weights.has_tensor("thinker.audio_tower.layers.18.self_attn.q_proj.weight") {
            ModelPreset::Qwen3Asr1_7b
        } else {
            ModelPreset::Qwen3Asr0_6b
        }
    }
}

impl ModelPreset {
    fn encoder_config(&self) -> EncoderConfig {
        match self {
            ModelPreset::Qwen3Asr0_6b => EncoderConfig {
                d_model: 896,
                layers: 18,
                heads: 14,
                head_dim: 64,
                ffn_dim: 3584,
                output_dim: 1024,
                n_window: 50,
                n_window_infer: 800,
                chunk_size: 100,
            },
            ModelPreset::Qwen3Asr1_7b => EncoderConfig {
                d_model: 1024,
                layers: 24,
                heads: 16,
                head_dim: 64,
                ffn_dim: 4096,
                output_dim: 2048,
                n_window: 50,
                n_window_infer: 800,
                chunk_size: 100,
            },
        }
    }

    fn decoder_config(&self) -> DecoderConfig {
        match self {
            ModelPreset::Qwen3Asr0_6b => DecoderConfig {
                vocab_size: 151936,
                hidden_size: 1024,
                intermediate_size: 3072,
                num_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                rope_theta: 1e6,
                rms_norm_eps: 1e-6,
            },
            ModelPreset::Qwen3Asr1_7b => DecoderConfig {
                vocab_size: 151936,
                hidden_size: 2048,
                intermediate_size: 6144,
                num_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                rope_theta: 1e6,
                rms_norm_eps: 1e-6,
            },
        }
    }

    pub fn config(&self) -> ModelConfig {
        ModelConfig {
            encoder: self.encoder_config(),
            decoder: self.decoder_config(),
            audio: AudioConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use safetensors::Dtype;

    fn write_weights(tensors: HashMap<String, safetensors::tensor::TensorView<'_>>) -> Weights {
        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();
        Weights::from_bytes(&serialized).unwrap()
    }

    #[test]
    fn detects_1_7b_when_discriminator_tensor_present() {
        let raw = vec![0u8; 4];
        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![1], &raw).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert(
            "thinker.audio_tower.layers.18.self_attn.q_proj.weight".to_string(),
            view,
        );

        let weights = write_weights(tensors);
        let preset = ModelPreset::from(&weights);

        assert!(matches!(preset, ModelPreset::Qwen3Asr1_7b));
    }

    #[test]
    fn defaults_to_0_6b_when_discriminator_tensor_missing() {
        let raw = vec![0u8; 4];
        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![1], &raw).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("some.other.tensor".to_string(), view);

        let weights = write_weights(tensors);
        let preset = ModelPreset::from(&weights);

        assert!(matches!(preset, ModelPreset::Qwen3Asr0_6b));
    }
}
