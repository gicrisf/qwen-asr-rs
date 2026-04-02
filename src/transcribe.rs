use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
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

/// Per-run timing breakdown returned by [`Pipeline::transcribe_timed`].
pub struct TimingInfo {
    pub encode_ms: f64,
    pub decode_ms: f64,
    pub n_tokens:  usize,
    pub audio_ms:  f64,
}

pub struct Pipeline {
    pub encoder:   Encoder,
    pub decoder:   Decoder,
    pub tokenizer: Tokenizer,
    pub audio_cfg: AudioConfig,
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
        let (mel, _) = self.mel_from_wav(wav_path)?;
        let (text, _) = self.transcribe_mel(&mel, 0.0)?;
        Ok(text)
    }

    /// Transcribe a WAV file and return per-phase timing.
    pub fn transcribe_timed(&mut self, wav_path: &Path) -> Result<(String, TimingInfo), TranscribeError> {
        let (mel, audio_ms) = self.mel_from_wav(wav_path)?;
        self.transcribe_mel(&mel, audio_ms)
    }

    /// Run the full pipeline on a pre-built mel tensor.
    /// Used internally and by the bench tool so audio loading isn't re-timed.
    pub fn transcribe_mel(&mut self, mel: &Tensor, audio_ms: f64) -> Result<(String, TimingInfo), TranscribeError> {
        let dev = Device::Cpu;

        // ── Encoder ───────────────────────────────────────────────────────────
        let t_enc   = Instant::now();
        let enc_out = self.encoder.forward(mel)?;
        let encode_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

        let (n_audio, _) = enc_out.dims2()?;
        eprintln!("encoder: {n_audio} audio tokens");

        // ── Build prompt embeddings ───────────────────────────────────────────
        let t_dec = Instant::now();

        let prefix_ids: Vec<u32> = PROMPT_PREFIX_HEAD.iter()
            .chain(PROMPT_PREFIX_TAIL.iter())
            .copied()
            .collect();
        let prefix_len = prefix_ids.len();
        let prefix_t   = Tensor::from_vec(prefix_ids, (1, prefix_len), &dev)?;
        let prefix_emb = self.decoder.embed(&prefix_t)?;

        let audio_emb = enc_out.unsqueeze(0)?.to_dtype(DType::BF16)?;

        let suffix_ids: Vec<u32> = PROMPT_SUFFIX_BASE.iter()
            .copied()
            .chain(std::iter::once(TOKEN_ASR_TEXT))
            .collect();
        let suffix_len = suffix_ids.len();
        let suffix_t   = Tensor::from_vec(suffix_ids, (1, suffix_len), &dev)?;
        let suffix_emb = self.decoder.embed(&suffix_t)?;

        let prompt_emb = Tensor::cat(&[&prefix_emb, &audio_emb, &suffix_emb], 1)?;
        let prompt_len = prompt_emb.dims()[1];

        // ── Prefill ───────────────────────────────────────────────────────────
        self.decoder.clear_kv_cache();
        let logits    = self.decoder.forward_with_embeds(&prompt_emb, 0)?;
        let mut token = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;

        // ── Autoregressive loop ───────────────────────────────────────────────
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

        let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        let n_tokens  = output_ids.len();

        let text = self.tokenizer.decode(&output_ids, true)?;
        Ok((text, TimingInfo { encode_ms, decode_ms, n_tokens, audio_ms }))
    }

    /// Run only the encoder on a mel tensor and return (n_audio_tokens, encode_ms).
    pub fn encode_timed(&mut self, mel: &Tensor) -> Result<(usize, f64), TranscribeError> {
        let t = Instant::now();
        let out = self.encoder.forward(mel)?;
        let ms  = t.elapsed().as_secs_f64() * 1000.0;
        Ok((out.dims()[0], ms))
    }

    /// Run only the encoder with detailed timing: (n_tokens, total_ms, conv_stem_ms, transformer_ms).
    pub fn encode_timed_breakdown(&mut self, mel: &Tensor) -> Result<(usize, f64, f64, f64), TranscribeError> {
        let t = Instant::now();
        let (out, conv_ms, xfmr_ms) = self.encoder.forward_timed(mel)?;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        Ok((out.dims()[0], total_ms, conv_ms, xfmr_ms))
    }

    /// Load a WAV file and return (mel tensor [mel_bins, n_frames], audio_duration_ms).
    pub fn mel_from_wav(&self, wav_path: &Path) -> Result<(Tensor, f64), TranscribeError> {
        let samples  = audio::load_wav(wav_path, &self.audio_cfg)?;
        let audio_ms = samples.len() as f64 / self.audio_cfg.sample_rate as f64 * 1000.0;
        let (flat, n_frames) = audio::mel_spectrogram(&samples, &self.audio_cfg);
        let mel = Tensor::from_vec(flat, (self.audio_cfg.mel_bins, n_frames), &Device::Cpu)?;
        Ok((mel, audio_ms))
    }

    /// Load WAV audio from a byte buffer and return (mel tensor, audio_duration_ms).
    pub fn mel_from_bytes(&self, bytes: &[u8]) -> Result<(Tensor, f64), TranscribeError> {
        let samples  = audio::load_wav_from_bytes(bytes, &self.audio_cfg)?;
        let audio_ms = samples.len() as f64 / self.audio_cfg.sample_rate as f64 * 1000.0;
        let (flat, n_frames) = audio::mel_spectrogram(&samples, &self.audio_cfg);
        let mel = Tensor::from_vec(flat, (self.audio_cfg.mel_bins, n_frames), &Device::Cpu)?;
        Ok((mel, audio_ms))
    }

    /// Build a zeroed (silence) mel tensor for `audio_sec` seconds.
    pub fn mel_silence(&self, audio_sec: u32) -> Result<(Tensor, f64), TranscribeError> {
        let n_frames = (audio_sec as usize * self.audio_cfg.sample_rate as usize)
            / self.audio_cfg.hop_length;
        let mel = Tensor::zeros((self.audio_cfg.mel_bins, n_frames), DType::F32, &Device::Cpu)?;
        Ok((mel, audio_sec as f64 * 1000.0))
    }
}

/// Collect weight shard paths from a model directory.
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
