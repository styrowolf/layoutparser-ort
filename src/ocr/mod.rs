//! OCR agents and utilities for extracting text.

mod hocr_ext;

use hocr_ext::HOCRElementConversion;
pub use hocr_ext::HOCRElementConversionError;

use hocr_parser::spec_definitions::elements;
use hocr_parser::Element;
use hocr_parser::HOCRParserError;
pub use hocr_parser::roxmltree;

pub use hocr_parser;

use crate::{LayoutElement, Result};
use tesseract::{Tesseract, TesseractError};

/// The Tesseract OCR Agent.
/// 
/// Constructing the agent may require a valid `tessdata`` directory 
/// depending on the constructor.
pub struct TesseractAgent {
    arguments: TesseractInitArguments,
    inner: Option<tesseract::Tesseract>,
}

enum TesseractInitArguments {
    Data {
        data: Vec<u8>,
        lang: String,
    },
    DataPath {
        data_path: String,
        lang: String,
    },
    Default,
    Generic {
        data_path: Option<String>,
        lang: Option<String>,
    },
}

impl TesseractAgent {
    /// Construct a new [`TesseractAgent`].
    pub fn new() -> Result<Self> {
        let arguments = TesseractInitArguments::Default;

        let inner = Tesseract::new(None, Some("eng")).map_err(|err| TesseractError::from(err))?;

        Ok(Self {
            inner: Some(inner),
            arguments,
        })
    }

    /// Construct a new [`TesseractAgent`] specifying the OCR languages.
    pub fn new_with_lang(lang: &[&str]) -> Result<Self> {
        let lang = lang.join("+");
        let inner = Tesseract::new(None, Some(&lang)).map_err(|err| TesseractError::from(err))?;

        let arguments = TesseractInitArguments::Generic {
            data_path: None,
            lang: Some(lang.to_string()),
        };

        Ok(Self {
            inner: Some(inner),
            arguments,
        })
    }

    /// Construct a new [`TesseractAgent`] with the tessdata file and specifying the OCR languages.
    pub fn new_with_data(data: &[u8], lang: &[&str]) -> Result<Self> {
        let lang = lang.join("+");
        let inner = Tesseract::new_with_data(data, Some(&lang), tesseract::OcrEngineMode::Default)
            .map_err(|err| TesseractError::from(err))?;

        let arguments = TesseractInitArguments::Data {
            data: data.to_vec(),
            lang: lang,
        };

        Ok(Self {
            inner: Some(inner),
            arguments,
        })
    }

    /// Construct a new [`TesseractAgent`] with the tessdata path and specfiying the OCR languages.
    pub fn new_data_path(data_path: &str, lang: &[&str]) -> Result<Self> {
        let lang = lang.join("+");
        // data_path is tessdata, which includes the traineddata files
        // https://github.com/tesseract-ocr/tessdata_fast
        // https://github.com/tesseract-ocr/tessdata
        let inner = Tesseract::new_with_oem(
            Some(data_path),
            Some(&lang),
            tesseract::OcrEngineMode::Default,
        )
        .map_err(|err| TesseractError::from(err))?;

        let arguments = TesseractInitArguments::DataPath {
            data_path: data_path.to_string(),
            lang: lang,
        };

        Ok(Self {
            inner: Some(inner),
            arguments,
        })
    }

    fn reinit(&mut self) {
        // UNWRAP SAFETY: we constructed Tesseract before with these arguments, so it should be safe to unwrap
        let tesseract = match &self.arguments {
            TesseractInitArguments::Data { data, lang } => {
                Tesseract::new_with_data(&data, Some(&lang), tesseract::OcrEngineMode::Default)
                    .unwrap()
            }
            TesseractInitArguments::DataPath { data_path, lang } => Tesseract::new_with_oem(
                Some(&data_path),
                Some(&lang),
                tesseract::OcrEngineMode::Default,
            )
            .unwrap(),
            TesseractInitArguments::Default => Tesseract::new(None, None).unwrap(),
            TesseractInitArguments::Generic { data_path, lang } => {
                Tesseract::new(data_path.as_deref(), lang.as_deref()).unwrap()
            }
        };
        self.inner = Some(tesseract);
    }

    /// Extracts the text within a [`LayoutElement`] and adds it to the element.
    pub fn extract_text_to_lm(
        &mut self,
        lm: &mut LayoutElement,
        img: &image::DynamicImage,
    ) -> Result<()> {
        let img = lm.crop_from_image(img);
        let text = self.extract_text(&img)?;
        lm.text = Some(text);
        Ok(())
    }

    /// Extracts the text from an image.
    pub fn extract_text(&mut self, img: &image::DynamicImage) -> Result<String> {
        let img = img.to_rgba8();
        let (width, height) = img.dimensions();
        let bytes_per_line = 4 * width;
        let frame_data = img.clone().into_vec();

        let inner = self.inner.take().unwrap();

        let mut inner = match inner
            .set_frame(
                &frame_data,
                width as i32,
                height as i32,
                4,
                bytes_per_line as i32,
            )
            .map_err(|err| TesseractError::from(err))
        {
            Ok(tess) => tess,
            Err(err) => {
                self.reinit();
                return Err(err.into());
            }
        };

        let text = inner.get_text().map_err(|err| TesseractError::from(err))?;
        self.inner = Some(inner);

        Ok(text)
    }

    /// Extracts the text regions as [`LayoutElement`] according to the OCR [`FeatureType`] from an image.
    pub fn extract(
        &mut self,
        img: &image::DynamicImage,
        feature: FeatureType,
    ) -> Result<Vec<LayoutElement>> {
        let img = img.to_rgba8();
        let (width, height) = img.dimensions();
        let bytes_per_line = 4 * width;
        let frame_data = img.clone().into_vec();

        let inner = self.inner.take().unwrap();

        let mut inner = match inner
            .set_frame(
                &frame_data,
                width as i32,
                height as i32,
                4,
                bytes_per_line as i32,
            )
            .map_err(|err| TesseractError::from(err))
        {
            Ok(tess) => tess,
            Err(err) => {
                self.reinit();
                return Err(err.into());
            }
        };

        let hocr = inner
            .get_hocr_text(0)
            .map_err(|err| TesseractError::from(err))?;

        let element = Element::from_node(
            roxmltree::Document::parse(&hocr)
                .map_err(HOCRParserError::from)?
                .root_element(),
        )?;

        let mut elements = vec![&element];
        elements.extend(element.descendants());

        let extracted_features: Vec<_> = elements
            .into_iter()
            .filter_map(|e| {
                if e.element_type == feature.as_hocr_element() {
                    // SAFETY: tesseract hOCR conversion always works
                    Some(e.get_layout_element().unwrap())
                } else {
                    None
                }
            })
            .collect();

        self.inner = Some(inner);

        Ok(extracted_features)
    }
}

/// OCR Feature types. Useful for extracting text regions at a specific level.
pub enum FeatureType {
    /// Page
    Page,
    /// Block
    Block,
    /// Paragraph
    Para,
    /// Line
    Line,
    /// Word
    Word,
}

impl FeatureType {
    /// Convert to hOCR element type.
    pub fn as_hocr_element(&self) -> &str {
        match self {
            FeatureType::Page => elements::OCR_PAGE,
            FeatureType::Block => elements::OCR_CAREA,
            FeatureType::Para => elements::OCR_PAR,
            FeatureType::Line => elements::OCRX_LINE,
            FeatureType::Word => elements::OCRX_WORD,
        }
    }
}
