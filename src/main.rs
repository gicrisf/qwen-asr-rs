use std::collections::HashMap;
use std::path::Path;

use safetensors::{Dtype, SafeTensors, SafeTensorError};

#[derive(Debug)]
enum WeightsError {
    Io(std::io::Error),
    Parsing(SafeTensorError),
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightsError::Io(err) => write!(f, "failed to read weights file: {err}"),
            WeightsError::Parsing(err) => write!(f, "failed to parse safetensors: {err}"),
        }
    }
}

impl std::error::Error for WeightsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WeightsError::Io(err) => Some(err),
            WeightsError::Parsing(err) => Some(err),
        }
    }
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
                 let data = std::fs::read(path)
                     .map_err(WeightsError::Io)?;

                 // ... and we parse it
                 let st = SafeTensors::deserialize(&data)
                     .map_err(WeightsError::Parsing)?;

                 for (name, view) in st.tensors() {
                     acc.0.insert(name.to_string(), (view.dtype(), view.data().to_vec()));
                 }

                 Ok(acc)
             })
    }
}

#[allow(dead_code)]
enum Model {
    Qwen06b,
    Qwen17b
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn reads_single_tensor_from_safetensors() {
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

        let (dtype, data) = weights.0.get("weight").expect("missing tensor");
        println!("read tensor: name=weight dtype={dtype:?} bytes={data:?}");
        assert_eq!(*dtype, Dtype::F32);
        assert_eq!(data.as_slice(), raw.as_slice());

        let _ = std::fs::remove_file(path);
    }
}
