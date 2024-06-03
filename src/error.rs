use thiserror::Error;

#[derive(Error, Debug)]
/// layoutparser-ort error variants
pub enum Error {
    #[error("ort (onnxruntime) error: {0}")]
    /// [`ort`] (onnxruntime) error
    Ort(#[from] ort::Error),
    #[error("hf-hub error: {0}")]
    /// Hugging Face API error
    HuggingFace(#[from] hf_hub::api::sync::ApiError),
    #[error("tesseract error: {0}")]
    #[cfg(feature = "ocr")]
    /// Tesseract error
    TesseractError(#[from] tesseract::TesseractError),
    #[error("hocr-parser error: {0}")]
    #[cfg(feature = "ocr")]
    /// hOCR parsing error
    HOCRParserError(#[from] hocr_parser::HOCRParserError),
    #[error("hOCR element conversion error: {0}")]
    #[cfg(feature = "ocr")]
    /// hOCR element conversion error
    HOCRElementConversionError(#[from] crate::ocr::HOCRElementConversionError),
}

/// A `Result` type alias using [`enum@Error`] instances as the error variant.
pub type Result<T, E = Error> = std::result::Result<T, E>;
