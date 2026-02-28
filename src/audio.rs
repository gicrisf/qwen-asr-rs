#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub mel_bins: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub n_fft: usize,
    pub n_freq: usize,
    pub conv_hidden: usize,
    pub conv_proj_dim: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        let conv_hidden = 480;
        let n_fft = 400;

        AudioConfig {
            sample_rate: 16000,
            mel_bins: 128,
            hop_length: 160,
            window_size: 400,
            n_fft,
            n_freq: n_fft / 2 + 1, // 201
            conv_hidden,
            conv_proj_dim: conv_hidden * 16, // 7680
        }
    }
}
