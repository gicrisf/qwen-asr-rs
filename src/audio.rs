use std::path::Path;

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use rustfft::{FftPlanner, num_complex::Complex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("failed to open WAV: {0}")]
    Wav(#[from] hound::Error),
    #[error("resample error: {0}")]
    Resample(String),
    #[error("invalid sample rate: {0}")]
    InvalidSampleRate(u32),
}

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

/// Load a WAV file, downmix to mono, normalize to f32 [-1,1], resample to target rate.
pub fn load_wav(path: &Path, cfg: &AudioConfig) -> Result<Vec<f32>, AudioError> {
    let reader = hound::WavReader::open(path)?;
    wav_reader_into_samples(reader, cfg)
}

/// Load WAV audio from a byte buffer.  Same processing as [`load_wav`].
pub fn load_wav_from_bytes(bytes: &[u8], cfg: &AudioConfig) -> Result<Vec<f32>, AudioError> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = hound::WavReader::new(cursor)?;
    wav_reader_into_samples(reader, cfg)
}

fn wav_reader_into_samples<R: std::io::Read>(
    reader: hound::WavReader<R>,
    cfg: &AudioConfig,
) -> Result<Vec<f32>, AudioError> {
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let scale = 1.0 / (1i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
    };

    // Downmix to mono
    let mono: Vec<f32> = if channels == 1 {
        raw_samples
    } else {
        raw_samples
            .chunks_exact(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Resample if needed
    if spec.sample_rate == cfg.sample_rate {
        return Ok(mono);
    }

    resample(&mono, spec.sample_rate, cfg.sample_rate)
}

fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>, AudioError> {
    if from_hz == 0 || to_hz == 0 {
        return Err(AudioError::InvalidSampleRate(if from_hz == 0 { from_hz } else { to_hz }));
    }

    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let ratio = to_hz as f64 / from_hz as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // Fixed chunking keeps memory bounded for long recordings.
    let chunk = 4096usize;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
        .map_err(|e| AudioError::Resample(e.to_string()))?;

    let mut out_all = Vec::<f32>::new();
    let mut offset = 0usize;
    while offset < samples.len() {
        let end = (offset + chunk).min(samples.len());
        let mut frame = vec![0.0f32; chunk];
        frame[..(end - offset)].copy_from_slice(&samples[offset..end]);
        offset = end;

        let waves_in = vec![frame];
        let mut out = resampler
            .process(&waves_in, None)
            .map_err(|e| AudioError::Resample(e.to_string()))?;
        if let Some(ch0) = out.pop() {
            out_all.extend(ch0);
        }
    }

    // Flush resampler tail.
    let waves_in = vec![vec![0.0f32; chunk]];
    let mut tail = resampler
        .process_partial(Some(&waves_in), None)
        .map_err(|e| AudioError::Resample(e.to_string()))?;
    if let Some(ch0) = tail.pop() {
        out_all.extend(ch0);
    }

    Ok(out_all)
}

/// Build Slaney-style mel filterbank, matching the C implementation.
/// Returns [mel_bins, n_freq] row-major.
fn build_mel_filters(cfg: &AudioConfig) -> Vec<f32> {
    fn hz_to_mel(freq: f32) -> f32 {
        let min_log_hz = 1000.0f32;
        let min_log_mel = 15.0f32;
        let logstep = 27.0f32 / 6.4f32.ln();
        let mels = 3.0 * freq / 200.0;
        if freq >= min_log_hz {
            min_log_mel + (freq / min_log_hz).ln() * logstep
        } else {
            mels
        }
    }

    fn mel_to_hz(mels: f32) -> f32 {
        let min_log_hz = 1000.0f32;
        let min_log_mel = 15.0f32;
        let logstep = 6.4f32.ln() / 27.0f32;
        if mels >= min_log_mel {
            min_log_hz * ((mels - min_log_mel) * logstep).exp()
        } else {
            200.0 * mels / 3.0
        }
    }

    let n_freq = cfg.n_freq;
    let n_mel = cfg.mel_bins;

    // FFT bin center frequencies
    let fft_freqs: Vec<f32> = (0..n_freq)
        .map(|i| i as f32 * (cfg.sample_rate as f32 / 2.0) / (n_freq - 1) as f32)
        .collect();

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(cfg.sample_rate as f32 / 2.0);

    // Mel-spaced filter center frequencies
    let filter_freqs: Vec<f32> = (0..=n_mel + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mel + 1) as f32))
        .collect();

    let filter_diff: Vec<f32> = (0..=n_mel)
        .map(|i| {
            let d = filter_freqs[i + 1] - filter_freqs[i];
            if d == 0.0 { 1e-6 } else { d }
        })
        .collect();

    let mut filters = vec![0.0f32; n_mel * n_freq];
    for m in 0..n_mel {
        let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
        for f in 0..n_freq {
            let down = (fft_freqs[f] - filter_freqs[m]) / filter_diff[m];
            let up = (filter_freqs[m + 2] - fft_freqs[f]) / filter_diff[m + 1];
            let val = down.min(up).max(0.0);
            filters[m * n_freq + f] = val * enorm;
        }
    }
    filters
}

/// Compute log mel spectrogram.
/// Returns flat [mel_bins * n_frames] in [mel_bin, frame] (row-major: mel-bin is outer).
/// n_frames is returned separately.
pub fn mel_spectrogram(samples: &[f32], cfg: &AudioConfig) -> (Vec<f32>, usize) {
    let n_fft = cfg.n_fft;
    let n_freqs = cfg.n_freq;
    let hop = cfg.hop_length;
    let win_len = cfg.window_size;
    let mel_bins = cfg.mel_bins;
    let pad_len = n_fft / 2;

    // Reflect-pad
    let n_samples = samples.len();
    let mut padded = Vec::with_capacity(n_samples + 2 * pad_len);
    for i in 0..pad_len {
        let src = pad_len - i;
        padded.push(if src < n_samples { samples[src] } else { 0.0 });
    }
    padded.extend_from_slice(samples);
    for i in 0..pad_len {
        let src = n_samples as isize - 2 - i as isize;
        padded.push(if src >= 0 { samples[src as usize] } else { 0.0 });
    }

    let padded_len = padded.len();
    let n_frames_total = (padded_len - n_fft) / hop + 1;
    let n_frames = if n_frames_total > 1 { n_frames_total - 1 } else { 0 };

    if n_frames == 0 {
        return (vec![], 0);
    }

    // Periodic Hann window
    let window: Vec<f32> = (0..win_len)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_len as f32).cos()))
        .collect();

    let mel_filters = build_mel_filters(cfg);

    // FFT planner
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut scratch = vec![Complex::new(0.0f32, 0.0); fft.get_inplace_scratch_len()];

    // First pass: compute log-mel for each frame, track global max
    let mut mel_tmp = vec![0.0f32; n_frames * mel_bins];
    let mut global_max = f32::NEG_INFINITY;

    let mut buf = vec![Complex::new(0.0f32, 0.0); n_fft];
    let mut power = vec![0.0f32; n_freqs];
    for t in 0..n_frames {
        let start = t * hop;
        for i in 0..n_fft {
            let s = if i < win_len { padded[start + i] * window[i] } else { 0.0 };
            buf[i] = Complex::new(s, 0.0);
        }
        fft.process_with_scratch(&mut buf, &mut scratch);

        // Power spectrum (only first n_freqs bins)
        for f in 0..n_freqs {
            let c = buf[f];
            power[f] = c.re * c.re + c.im * c.im;
        }

        for m in 0..mel_bins {
            let filt = &mel_filters[m * n_freqs..(m + 1) * n_freqs];
            let sum: f32 = filt.iter().zip(power.iter()).map(|(f, p)| f * p).sum();
            let val = sum.max(1e-10).log10();
            mel_tmp[t * mel_bins + m] = val;
            if val > global_max {
                global_max = val;
            }
        }
    }

    // Second pass: clamp with dynamic max and normalize.
    // Output layout: [mel_bins, n_frames] for Conv2D compatibility.
    let min_val = global_max - 8.0;
    let mut mel = vec![0.0f32; mel_bins * n_frames];
    for t in 0..n_frames {
        for m in 0..mel_bins {
            let val = mel_tmp[t * mel_bins + m].max(min_val);
            mel[m * n_frames + t] = (val + 4.0) / 4.0;
        }
    }

    (mel, n_frames)
}
