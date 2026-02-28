use std::collections::HashMap;
use std::path::Path;

use safetensors::{Dtype, SafeTensors, SafeTensorError};
use thiserror::Error;
use half::{bf16, f16};

use std::convert::TryFrom;

#[derive(Debug, Error)]
pub enum WeightsError {
    #[error("failed to read weights file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse safetensors: {0}")]
    Parsing(#[from] SafeTensorError),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("wrong dtype for tensor {name}: {dtype:?}")]
    WrongDtype { name: String, dtype: Dtype },
}

// name -> (dtype, shape, flat byte data)
#[derive(Default, Debug)]
pub struct Weights(HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>);

impl TryFrom<&[u8]> for Weights {
    type Error = WeightsError;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        Self::from_bytes(data)
    }
}

impl TryFrom<&Path> for Weights {
    type Error = WeightsError;

    fn try_from(path: &Path) -> Result<Self, Self::Error> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }
}

impl Weights {
    pub(crate) fn from_files(paths: &[impl AsRef<Path>]) -> Result<Self, WeightsError> {
        paths.iter()
             .try_fold(Self::default(), |mut acc, path| {
                 let path = path.as_ref();
                 // Is the file actually there?
                 // If so, we read it...
                 let data = std::fs::read(path)?;
                 // ... and we parse it
                 let weights = Self::from_bytes(&data)?;
                 acc.0.extend(weights.0);
                 Ok(acc)
             })
    }

    pub(crate) fn from_bytes(data: &[u8]) -> Result<Self, WeightsError> {
        let st = SafeTensors::deserialize(data)?;
        let mut acc = Self::default();
        for (name, view) in st.tensors() {
            acc.0
               .insert(name.to_string(), (view.dtype(), view.shape().to_vec(), view.data().to_vec()));
        }
        Ok(acc)
    }

    /// Load a tensor as flat f32 vector
    pub(crate) fn get_f32(&self, name: &str) -> Result<Vec<f32>, WeightsError> {
        let (dtype, _shape, bytes) = self
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
    fn get_raw_bf16(&self, name: &str) -> Result<Vec<u16>, WeightsError> {
        let (dtype, _shape, bytes) = self
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

    pub(crate) fn shape_of(&self, name: &str) -> Option<&[usize]> {
        self.0.get(name).map(|(_, shape, _)| shape.as_slice())
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn get_f32_reads_f32_tensor() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.0f32.to_le_bytes());
        raw.extend_from_slice(&2.0f32.to_le_bytes());

        let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![2], &raw).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), view);

        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();
        let weights = Weights::from_bytes(&serialized).unwrap();

        let values = weights.get_f32("weight").unwrap();
        assert_eq!(values, vec![1.0f32, 2.0f32]);

    }

    #[test]
    fn get_raw_bf16_reads_bf16_tensor() {
        let raw: Vec<u8> = vec![0x80, 0x3f, 0x00, 0x40];
        let view = safetensors::tensor::TensorView::new(Dtype::BF16, vec![2], &raw).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), view);

        let serialized = safetensors::tensor::serialize(tensors, None).unwrap();
        let weights = Weights::from_bytes(&serialized).unwrap();

        let bits = weights.get_raw_bf16("weight").unwrap();
        assert_eq!(bits, vec![0x3f80u16, 0x4000u16]);

    }
}
