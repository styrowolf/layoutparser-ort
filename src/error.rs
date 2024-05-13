use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("ort (onnxruntime) error: {0}")]
    Ort(#[from] ort::Error),
    #[error("hf-hub error: {0}")]
    HuggingFace(#[from] hf_hub::api::sync::ApiError),
    #[error("tesseract error: {0}")]
    #[cfg(feature = "ocr")]
    TesseractError(#[from] tesseract::TesseractError),
    #[error("hocr-parser error: {0}")]
    #[cfg(feature = "ocr")]
    HOCRParserError(#[from] hocr_parser::HOCRParserError),
    #[error("hOCR element conversion error: {0}")]
    #[cfg(feature = "ocr")]
    HOCRElementConversionError(#[from] crate::ocr::HOCRElementConversionError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
