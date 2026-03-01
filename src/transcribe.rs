use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use thiserror::Error;

use crate::audio::{self, AudioConfig, AudioError};
use crate::decoder::Decoder;
use crate::encoder::Encoder;
use crate::preset::ModelPreset;
use crate::tokenizer::{
    Tokenizer, TokenizerError,
    PROMPT_PREFIX_HEAD, PROMPT_PREFIX_TAIL, PROMPT_SUFFIX_BASE,
    TOKEN_ASR_TEXT, TOKEN_IM_END,
};

#[derive(Debug, Error)]
pub enum TranscribeError {
    #[error("audio error: {0}")]
    Audio(#[from] AudioError),
    #[error("model error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing weights: {0}")]
    MissingWeights(String),
}

pub struct Pipeline {
    encoder:   Encoder,
    decoder:   Decoder,
    tokenizer: Tokenizer,
    audio_cfg: AudioConfig,
}

impl Pipeline {
    pub fn load(model_dir: &Path) -> Result<Self, TranscribeError> {
        let preset = ModelPreset::from_dir(model_dir);
        let cfg    = preset.config();
        let shards = collect_shards(model_dir)?;
        let dev    = Device::Cpu;

        let encoder   = Encoder::load(&shards, cfg.encoder, &dev)?;
        let decoder   = Decoder::load(&shards, &cfg.decoder, &dev)?;
        let tokenizer = Tokenizer::load(model_dir)?;

        Ok(Self { encoder, decoder, tokenizer, audio_cfg: cfg.audio })
    }

    /// Transcribe a WAV file to text.
    pub fn transcribe(&mut self, wav_path: &Path) -> Result<String, TranscribeError> {
        let dev = Device::Cpu;

        // 1. Audio → log-mel spectrogram [mel_bins, n_frames]
        let samples = audio::load_wav(wav_path, &self.audio_cfg)?;
        let (mel_flat, n_frames) = audio::mel_spectrogram(&samples, &self.audio_cfg);
        if n_frames == 0 {
            return Ok(String::new());
        }
        let mel = Tensor::from_vec(mel_flat, (self.audio_cfg.mel_bins, n_frames), &dev)?;

        // 2. Encoder → [n_audio, output_dim]
        let enc_out = self.encoder.forward(&mel)?;
        let (n_audio, _) = enc_out.dims2()?;
        eprintln!("encoder: {n_audio} audio tokens");

        // 3. Build full prompt embeddings:
        //    [PREFIX_HEAD | PREFIX_TAIL | audio×n | SUFFIX_BASE | <asr_text>]
        let prefix_ids: Vec<u32> = PROMPT_PREFIX_HEAD.iter()
            .chain(PROMPT_PREFIX_TAIL.iter())
            .copied()
            .collect();
        let prefix_len = prefix_ids.len();
        let prefix_t   = Tensor::from_vec(prefix_ids, (1, prefix_len), &dev)?;
        let prefix_emb = self.decoder.embed(&prefix_t)?; // [1, prefix_len, hidden]

        // Encoder output as embeddings: [n_audio, hidden] → [1, n_audio, hidden]
        let audio_emb = enc_out.unsqueeze(0)?;

        let suffix_ids: Vec<u32> = PROMPT_SUFFIX_BASE.iter()
            .copied()
            .chain(std::iter::once(TOKEN_ASR_TEXT))
            .collect();
        let suffix_len = suffix_ids.len();
        let suffix_t   = Tensor::from_vec(suffix_ids, (1, suffix_len), &dev)?;
        let suffix_emb = self.decoder.embed(&suffix_t)?; // [1, suffix_len, hidden]

        let prompt_emb = Tensor::cat(&[&prefix_emb, &audio_emb, &suffix_emb], 1)?;
        let prompt_len = prompt_emb.dims()[1];

        // 4. Prefill: run all prompt tokens through the decoder at once.
        self.decoder.clear_kv_cache();
        let logits    = self.decoder.forward_with_embeds(&prompt_emb, 0)?; // [1, vocab]
        let mut token = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;

        // 5. Autoregressive loop: collect tokens until <|im_end|> or budget.
        let max_new_tokens = 448;
        let mut output_ids: Vec<u32> = Vec::new();
        let mut offset = prompt_len;

        loop {
            if token == TOKEN_IM_END || output_ids.len() >= max_new_tokens {
                break;
            }
            output_ids.push(token);
            token   = self.decoder.step(token, offset)?;
            offset += 1;
        }

        // 6. Decode token IDs → UTF-8 text.
        Ok(self.tokenizer.decode(&output_ids, true)?)
    }
}

/// Collect weight shard paths from a model directory.
/// 0.6b: single `model.safetensors`.
/// 1.7b: shards listed in `model.safetensors.index.json`.
pub fn collect_shards(model_dir: &Path) -> Result<Vec<PathBuf>, TranscribeError> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let content = std::fs::read_to_string(&index)?;
        let jsonv: serde_json::Value = serde_json::from_str(&content)?;
        let weight_map = jsonv
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TranscribeError::MissingWeights("invalid weight_map".into()))?;
        let mut shards: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        shards.sort_unstable();
        shards.dedup();
        let paths: Vec<PathBuf> = shards.into_iter().map(|s| model_dir.join(s)).collect();
        for p in &paths {
            if !p.exists() {
                return Err(TranscribeError::MissingWeights(p.display().to_string()));
            }
        }
        Ok(paths)
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            return Err(TranscribeError::MissingWeights(model_dir.display().to_string()));
        }
        Ok(vec![single])
    }
}
