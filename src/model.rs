use crate::audio::AudioConfig;
use crate::encoder::EncoderConfig;
use crate::preset::ModelPreset;
use candle_transformers::models::qwen3::Config as Qwen3Config;

use std::path::{Path, PathBuf};
use thiserror::Error;
use crate::weights::Weights;

#[derive(Debug, Error)]
enum ModelError {
    #[error("Failed to locate weights in {0}")]
    MissingWeights(String),
    #[error("Failed to read the json index {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse the json index {0}")]
    CorruptIndex(#[from] serde_json::Error),
    #[error("Invalid json index: {0}")]
    InvalidIndex(String),
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub decoder: Qwen3Config,
    // TODO
    // pub tokenizer: TokenizerConfig,
    pub audio: AudioConfig,
}

#[derive(Debug)]
pub struct Model {
    // TODO
    // encoder: Encoder,
    // TODO
    // decoder: Decoder,
    // TODO
    // tokenizer: Tokenizer,
    pub config: ModelConfig,
}

impl Model {
    fn load_from_dir(model_dir: &Path) -> Result<Self, ModelError> {
        println!("Loading model from {:?}", model_dir);
        // searching for a safetensors json index
        let index = model_dir.join("model.safetensors.index.json");
        if index.exists() {
            // Reading the json
            let content = std::fs::read_to_string(index)?;
            let jsonv: serde_json::Value = serde_json::from_str(&content)?;
            let weight_map = jsonv
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    ModelError::InvalidIndex("missing or invalid weight_map".to_string())
                })?;

            // println!("{:#?}", weight_map);

            let mut shards: Vec<String> =
                weight_map.values()
                          .filter_map(|v| v.as_str().map(|s| s.to_string()))
                          .collect();
            // no need to use proper sort
            shards.sort_unstable();
            // since we deduplicate right after
            shards.dedup();

            // we finally have a list of shards
            let shards: Vec<PathBuf> =
                shards
                .into_iter()
                .map(|s| model_dir.join(s))
                .collect();

            // are we sure the files are actually there?
            for shard_path in &shards {
                if !shard_path.exists() {
                    return Err(ModelError::MissingWeights(shard_path.display().to_string()));
                }
            }

            println!("{:#?}", shards);
            // fine, we can proceed
            // We could get everything
            // let weights = Weights::from_files(&shards);

            Ok(Model {
                config: ModelPreset::Qwen3Asr1_7b.config()
            })
        } else {
            // no index? let's go for a single shard
            let single_shard = model_dir.join("model.safetensors");
            if !single_shard.exists() {
                return Err(ModelError::MissingWeights(format!("{:?}", model_dir)))
            }

            Ok(Model {
                config: ModelPreset::Qwen3Asr0_6b.config()
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;

    fn smoke_model_dir() -> PathBuf {
        if let Ok(model_dir) = env::var("QWEN_ASR_MODEL_DIR") {
            let p = PathBuf::from(model_dir);
            return if p.is_file() {
                p.parent().map_or_else(|| p.clone(), PathBuf::from)
            } else {
                p
            };
        }
        if let Ok(root) = env::var("QWEN_ASR_ROOT") {
            return PathBuf::from(root).join("qwen3-asr-1.7b");
        }
        panic!(
            "Set QWEN_ASR_MODEL_DIR=/abs/path/to/model-dir \
or QWEN_ASR_ROOT=/abs/path/to/repo-root"
        );
    }

    #[test]
    #[ignore]
    fn load_from_dir_smoke() {
        let model_dir = smoke_model_dir();
        let model = Model::load_from_dir(&model_dir);
        println!("{:#?}", model);
        assert!(model.is_ok());
    }
}
