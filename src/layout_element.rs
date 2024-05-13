use geo_types::{coord, Coord, Rect};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayoutElement {
    pub bbox: Rect<f32>,
    pub element_type: String,
    pub score: f32,
    pub source: String,
    pub text: Option<String>,
}

impl LayoutElement {
    pub fn new(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        element_type: &str,
        score: f32,
        source: &str,
    ) -> Self {
        let bbox = Rect::new(coord! { x: x1, y: y1 }, coord! { x: x2, y: y2 });

        Self {
            bbox,
            element_type: element_type.to_string(),
            score,
            text: None,
            source: source.to_string(),
        }
    }

    pub fn new_with_text(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        element_type: &str,
        text: String,
        score: f32,
        source: &str,
    ) -> Self {
        let bbox = Rect::new(coord! { x: x1, y: y1 }, coord! { x: x2, y: y2 });

        Self {
            bbox,
            element_type: element_type.to_string(),
            score,
            text: Some(text),
            source: source.to_string(),
        }
    }

    pub fn pad(&mut self, padding: f32) {
        self.bbox
            .set_min(self.bbox.min() - coord! { x: padding, y: padding });
        self.bbox
            .set_max(self.bbox.max() + coord! { x: padding, y: padding });
    }

    pub fn crop_from_image(&self, img: &image::DynamicImage) -> image::DynamicImage {
        let (x1, y1) = (self.bbox.min().x as u32, self.bbox.min().y as u32);
        let (width, height) = (self.bbox.width() as u32, self.bbox.height() as u32);

        img.clone().crop(x1, y1, width, height)
    }

    pub fn transform(&mut self, transform: impl Fn(Coord<f32>) -> Coord<f32>) {
        self.bbox.set_min(transform(self.bbox.min()));
        self.bbox.set_max(transform(self.bbox.max()));
    }
}
