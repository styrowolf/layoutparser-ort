//! # Overview
//! 
//! A simplified port of [LayoutParser](https://github.com/Layout-Parser/layout-parser) for detecting layout elements on documents. 
//! Runs Detectron2 and YOLOX layout models from [unstructured-inference](https://github.com/Unstructured-IO/unstructured-inference/) 
//! in ONNX format through onnxruntime (bindings via [ort](https://github.com/pykeio/ort)).

mod error;
mod layout_element;
#[cfg(feature = "ocr")]
pub mod ocr;
mod utils;

pub use error::{Error, Result};

// re-exports
pub use ort;
pub use image;
pub use geo_types;

pub mod models;
#[cfg(feature = "save")]
pub use utils::save;

pub use layout_element::LayoutElement;
