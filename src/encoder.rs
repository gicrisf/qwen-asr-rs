#[derive(Debug, Clone)]
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

// TODO put encoder config in here
pub struct Encoder {
}
