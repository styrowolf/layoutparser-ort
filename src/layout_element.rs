use geo_types::{coord, Coord, Rect};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// A detected element in a document's layout.
pub struct LayoutElement {
    /// Bounding box of the element.
    pub bbox: Rect<f32>,
    /// Type of element. This value depends on the labels of the model used to detect this element.
    pub element_type: String,
    // Confidence for the detection.
    pub confidence: f32,
    /// Source of the detection (the name of module which detected this element).
    pub source: String,
    /// Text within this element. This field is filled after OCR.
    pub text: Option<String>,
}

impl LayoutElement {
    /// Constructs a [`LayoutElement`] instance.
    pub fn new(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        element_type: &str,
        confidence: f32,
        source: &str,
    ) -> Self {
        let bbox = Rect::new(coord! { x: x1, y: y1 }, coord! { x: x2, y: y2 });

        Self {
            bbox,
            element_type: element_type.to_string(),
            confidence,
            text: None,
            source: source.to_string(),
        }
    }

    /// Constructs a [`LayoutElement`] instance with text.
    pub fn new_with_text(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        element_type: &str,
        text: String,
        confidence: f32,
        source: &str,
    ) -> Self {
        let bbox = Rect::new(coord! { x: x1, y: y1 }, coord! { x: x2, y: y2 });

        Self {
            bbox,
            element_type: element_type.to_string(),
            confidence,
            text: Some(text),
            source: source.to_string(),
        }
    }

    /// Pads the bounding box of a [`LayoutElement`]. Useful for OCRing the element.
    pub fn pad(&mut self, padding: f32) {
        self.bbox
            .set_min(self.bbox.min() - coord! { x: padding, y: padding });
        self.bbox
            .set_max(self.bbox.max() + coord! { x: padding, y: padding });
    }

    /// Crop the section of the image according to the [`LayoutElement`]'s bounding box.
    pub fn crop_from_image(&self, img: &image::DynamicImage) -> image::DynamicImage {
        let (x1, y1) = (self.bbox.min().x as u32, self.bbox.min().y as u32);
        let (width, height) = (self.bbox.width() as u32, self.bbox.height() as u32);

        img.clone().crop(x1, y1, width, height)
    }

    /// Apply a transformation to the bounding box points.
    pub fn transform(&mut self, transform: impl Fn(Coord<f32>) -> Coord<f32>) {
        self.bbox.set_min(transform(self.bbox.min()));
        self.bbox.set_max(transform(self.bbox.max()));
    }
}
