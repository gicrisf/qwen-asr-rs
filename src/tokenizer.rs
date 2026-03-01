use std::path::Path;
use std::collections::HashMap;

use thiserror::Error;
use tokenizers::decoders::byte_level::ByteLevel as BLDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("encode error: {0}")]
    Encode(String),
    #[error("decode error: {0}")]
    Decode(String),
}

// ── Special token IDs (from qwen_asr.h) ──────────────────────────────────────
pub const TOKEN_IM_START:    u32 = 151644; // <|im_start|>
pub const TOKEN_IM_END:      u32 = 151645; // <|im_end|>
pub const TOKEN_ENDOFTEXT:   u32 = 151643; // <|endoftext|>
pub const TOKEN_AUDIO_START: u32 = 151669; // <|audio_start|>
pub const TOKEN_AUDIO_END:   u32 = 151670; // <|audio_end|>
pub const TOKEN_AUDIO_PAD:   u32 = 151676; // <|AUDIO|> padding token
pub const TOKEN_ASR_TEXT:    u32 = 151704; // <asr_text> — gates text accumulation

// ── Prompt template sequences (from qwen_asr.c) ──────────────────────────────
//
// Full prompt layout:
//   PREFIX_HEAD  [<|im_start|> "system" "\n"]
//   [optional system-prompt text tokens]
//   PREFIX_TAIL  [<|im_end|> "\n" <|im_start|> "user" "\n" <|audio_start|>]
//   AUDIO        [TOKEN_AUDIO_PAD] × N_audio_tokens
//   SUFFIX_BASE  [<|audio_end|> <|im_end|> "\n" <|im_start|> "assistant" "\n"]
//   [optional language tokens + TOKEN_ASR_TEXT]

pub const PROMPT_PREFIX_HEAD: &[u32] = &[TOKEN_IM_START, 8948, 198];
pub const PROMPT_PREFIX_TAIL: &[u32] = &[TOKEN_IM_END, 198, TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START];
pub const PROMPT_SUFFIX_BASE: &[u32] = &[TOKEN_AUDIO_END, TOKEN_IM_END, 198, TOKEN_IM_START, 77091, 198];

// ── Tokenizer ─────────────────────────────────────────────────────────────────

pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: HashMap<u32, &'static str>,
}

impl Tokenizer {
    /// Load from `<model_dir>/vocab.json` + `merges.txt`.
    pub fn load(model_dir: &Path) -> Result<Self, TokenizerError> {
        let vocab  = model_dir.join("vocab.json");
        let merges = model_dir.join("merges.txt");

        if !vocab.exists() {
            return Err(TokenizerError::Load(format!("vocab.json not found in {:?}", model_dir)));
        }
        if !merges.exists() {
            return Err(TokenizerError::Load(format!("merges.txt not found in {:?}", model_dir)));
        }

        let bpe = BPE::from_file(vocab.to_str().unwrap(), merges.to_str().unwrap())
            .build()
            .map_err(|e| TokenizerError::Load(e.to_string()))?;

        let mut inner = HfTokenizer::new(bpe);
        // GPT-2 style behavior: do not force a synthetic leading space.
        inner.with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)));
        inner.with_decoder(Some(BLDecoder::default()));

        let special_tokens = HashMap::from([
            (TOKEN_IM_START, "<|im_start|>"),
            (TOKEN_IM_END, "<|im_end|>"),
            (TOKEN_ENDOFTEXT, "<|endoftext|>"),
            (TOKEN_AUDIO_START, "<|audio_start|>"),
            (TOKEN_AUDIO_END, "<|audio_end|>"),
            (TOKEN_AUDIO_PAD, "<|AUDIO|>"),
            (TOKEN_ASR_TEXT, "<asr_text>"),
        ]);

        Ok(Tokenizer { inner, special_tokens })
    }

    /// Decode token IDs to a UTF-8 string.
    /// Set `skip_special` to strip special tokens (e.g. `<|im_start|>`).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Encode text to token IDs (used for prompt/language injection).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self.inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a single token ID. Known special IDs are handled explicitly.
    pub fn decode_token(&self, id: u32) -> Result<String, TokenizerError> {
        if let Some(tok) = self.special_tokens.get(&id) {
            return Ok((*tok).to_string());
        }
        self.decode(&[id], false)
    }

    /// Total vocabulary size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        if let Ok(p) = env::var("QWEN_ASR_MODEL_DIR") {
            let p = PathBuf::from(p);
            return if p.is_file() { p.parent().unwrap().to_path_buf() } else { p };
        }
        if let Ok(root) = env::var("QWEN_ASR_ROOT") {
            return PathBuf::from(root).join("qwen3-asr-0.6b");
        }
        panic!("Set QWEN_ASR_MODEL_DIR or QWEN_ASR_ROOT");
    }

    #[test]
    #[ignore]
    fn vocab_size_matches() {
        let tok = Tokenizer::load(&model_dir()).expect("load failed");
        // vocab.json contains base BPE ids only; special/reserved ids live above this range.
        assert_eq!(tok.vocab_size(), 151643);
    }

    #[test]
    #[ignore]
    fn decode_known_ids() {
        let tok = Tokenizer::load(&model_dir()).expect("load failed");
        assert_eq!(tok.decode(&[323, 8679, 5086], true).unwrap(), " and fear itself");
        assert_eq!(tok.decode(&[8948],  true).unwrap(), "system");
        assert_eq!(tok.decode(&[198],   true).unwrap(), "\n");
        assert_eq!(tok.decode(&[872],   true).unwrap(), "user");
        assert_eq!(tok.decode(&[77091], true).unwrap(), "assistant");
    }

    #[test]
    #[ignore]
    fn encode_decode_roundtrip() {
        let tok = Tokenizer::load(&model_dir()).expect("load failed");
        let original = "Hello world";
        let ids = tok.encode(original).expect("encode failed");
        let decoded = tok.decode(&ids, true).expect("decode failed");
        assert_eq!(decoded, original);
    }

    #[test]
    #[ignore]
    fn decode_special_ids() {
        let tok = Tokenizer::load(&model_dir()).expect("load failed");
        assert_eq!(tok.decode_token(TOKEN_IM_START).unwrap(), "<|im_start|>");
        assert_eq!(tok.decode_token(TOKEN_IM_END).unwrap(), "<|im_end|>");
        assert_eq!(tok.decode_token(TOKEN_AUDIO_START).unwrap(), "<|audio_start|>");
        assert_eq!(tok.decode_token(TOKEN_AUDIO_END).unwrap(), "<|audio_end|>");
        assert_eq!(tok.decode_token(TOKEN_AUDIO_PAD).unwrap(), "<|AUDIO|>");
        assert_eq!(tok.decode_token(TOKEN_ASR_TEXT).unwrap(), "<asr_text>");
    }
}
