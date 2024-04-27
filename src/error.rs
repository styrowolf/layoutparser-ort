use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("ort (onnxruntime) error: {0}")]
    Ort(#[from] ort::Error),
    #[error("hf-hub: {0}")]
    HuggingFace(#[from] hf_hub::api::sync::ApiError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
