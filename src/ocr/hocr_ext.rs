use hocr_parser::element::Element;
use hocr_parser::spec_definitions::elements;
use hocr_parser::spec_definitions::properties;

use thiserror::Error;

use crate::utils::vec_to_bbox;
use crate::LayoutElement;

#[derive(Debug, Error)]
pub enum HOCRElementConversionError {
    #[error("No bounding box found in element properties")]
    NoBoundingBoxFound,
    #[error("No confidence found in element properties")]
    NoConfidenceFound,
}

pub(crate) trait HOCRElementConversion {
    fn get_layout_element(&self) -> Result<LayoutElement, HOCRElementConversionError>;
    fn bbox(&self) -> Option<[u32; 4]>;
    fn confidence(&self) -> Option<f32>;
    fn extract_text(&self) -> String;
}

impl HOCRElementConversion for Element {
    fn get_layout_element(&self) -> Result<LayoutElement, HOCRElementConversionError> {
        let [x1, y1, x2, y2] = self
            .bbox()
            .ok_or(HOCRElementConversionError::NoBoundingBoxFound)?;

        Ok(LayoutElement::new_with_text(
            x1 as f32,
            y1 as f32,
            x2 as f32,
            y2 as f32,
            &self.element_type,
            self.extract_text(),
            self.confidence()
                .ok_or(HOCRElementConversionError::NoConfidenceFound)?,
            "hocr-parser",
        ))
    }

    fn confidence(&self) -> Option<f32> {
        match self.element_type.as_str() {
            elements::OCRX_WORD => self
                .properties
                .iter()
                .find(|(n, _)| n == properties::X_WCONF)?
                .1[0]
                .parse::<f32>()
                .map(|e| e / 100.0)
                .ok(),
            _ => {
                let children: Vec<f32> = self
                    .children
                    .iter()
                    .filter_map(|e| e.confidence())
                    .collect();
                match children.len() {
                    0 => None,
                    len => {
                        let sum: f32 = children.iter().sum();
                        Some(sum / len as f32)
                    }
                }
            }
        }
    }

    fn bbox(&self) -> Option<[u32; 4]> {
        let bbox_strs = &self
            .properties
            .iter()
            .find(|(n, _)| n == properties::BBOX)?
            .1;

        if bbox_strs.len() != 4 {
            return None;
        }

        let bbox = bbox_strs
            .iter()
            .map(|s| s.parse::<u32>().unwrap())
            .collect::<Vec<u32>>();

        Some(vec_to_bbox(bbox))
    }

    fn extract_text(&self) -> String {
        match self.element_type.as_str() {
            elements::OCRX_WORD => self.text.clone().unwrap_or_default(),
            /* to filter by confidence if you want
            elements::OCR_LINE | elements::OCRX_LINE => {
                let confidence = self.confidence().unwrap_or(0.0);
                if confidence < 0.5 {
                    return "".to_string();
                } else {
                    self.children
                    .iter()
                    .map(|e| e.extract_text())
                    .collect::<Vec<String>>()
                    .join(" ")
                    + "\n"
                }
            },
            */
            _ => {
                self.children
                    .iter()
                    .map(|e| e.extract_text())
                    .collect::<Vec<String>>()
                    .join(" ")
                    + "\n"
            }
        }
    }
}
