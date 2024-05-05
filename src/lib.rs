mod error;
mod utils;

pub use error::{Error, Result};

// re-exports
pub use ort;

pub mod models;
#[cfg(feature = "save")]
pub use utils::save;

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayoutElement {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub element_type: String,
    pub probability: f32,
    pub source: String,
}
