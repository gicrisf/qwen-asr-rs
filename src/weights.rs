use std::collections::HashMap;
use std::path::Path;

use safetensors::{Dtype, SafeTensors, SafeTensorError};
use thiserror::Error;
use half::{bf16, f16};

#[derive(Debug, Error)]
enum WeightsError {
    #[error("failed to read weights file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse safetensors: {0}")]
    Parsing(#[from] SafeTensorError),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("wrong dtype for tensor {name}: {dtype:?}")]
    WrongDtype { name: String, dtype: Dtype },
}

// name -> (dtype, flat byte data)
#[derive(Default)]
struct Weights(HashMap<String, (Dtype, Vec<u8>)>);

type Result<T> = std::result::Result<T, WeightsError>;

impl Weights {
    fn from_files(paths: &[impl AsRef<Path>]) -> Result<Self> {
        paths.iter()
             .try_fold(Self::default(), |mut acc, path| {
                 let path = path.as_ref();
                 // Is the file actually there?
                 // If so, we read it...
                 let data = std::fs::read(path)?;
                 // ... and we parse it
                 let st = SafeTensors::deserialize(&data)?;

                 for (name, view) in st.tensors() {
                     acc.0.insert(name.to_string(), (view.dtype(), view.data().to_vec()));
                 }

                 Ok(acc)
             })
    }

    /// Load a tensor as flat f32 vector
    fn get_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dtype, bytes) = self
            .0
            .get(name)
            .ok_or_else(|| WeightsError::MissingTensor(name.to_string()))?;

        match dtype {
            Dtype::F32 => {
                Ok(bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            Dtype::BF16 => {
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                    .collect())
            }
            Dtype::F16 => {
                Ok(bytes
                    .chunks_exact(2)
                    .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                    .collect())
            }
            _ => Err(WeightsError::WrongDtype {
                name: name.to_string(),
                dtype: *dtype,
            }),
        }
    }

    /// Load a BF16 tensor as raw u16 bits (for decoding)
    fn get_raw_bf16(&self, name: &str) -> Result<Vec<u16>> {
        let (dtype, bytes) = self
            .0
            .get(name)
            .ok_or_else(|| WeightsError::MissingTensor(name.to_string()))?;

        if *dtype != Dtype::BF16 {
            return Err(WeightsError::WrongDtype {
                name: name.to_string(),
                dtype: *dtype,
            });
        }

        Ok(bytes
           .chunks_exact(2)
           .map(|b| u16::from_le_bytes([b[0], b[1]]))
           .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn get_f32_reads_f32_tensor() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.0f32.to_le_bytes());
        raw.extend_from_slice(&2.0f32.to_le_bytes());

        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![2], &raw).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), view);

        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();

        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("qwen-asr-test-{suffix}.safetensors"));

        std::fs::write(&path, serialized).unwrap();

        let weights = Weights::from_files(&[&path]).unwrap();

        let values = weights.get_f32("weight").unwrap();
        assert_eq!(values, vec![1.0f32, 2.0f32]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn get_raw_bf16_reads_bf16_tensor() {
        let raw: Vec<u8> = vec![0x80, 0x3f, 0x00, 0x40];
        let view = safetensors::tensor::TensorView::new(Dtype::BF16, vec![2], &raw).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), view);

        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();

        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("qwen-asr-test-{suffix}.safetensors"));

        std::fs::write(&path, serialized).unwrap();

        let weights = Weights::from_files(&[&path]).unwrap();

        let bits = weights.get_raw_bf16("weight").unwrap();
        assert_eq!(bits, vec![0x3f80u16, 0x4000u16]);

        let _ = std::fs::remove_file(path);
    }
}
