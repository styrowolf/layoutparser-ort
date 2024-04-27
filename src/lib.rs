use image::imageops;
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use ort::{Session, SessionBuilder, SessionOutputs};

mod error;

pub use error::{Error, Result};

fn vec_to_bbox<T: Copy>(v: Vec<T>) -> [T; 4] {
    return [v[0], v[1], v[2], v[3]];
}

#[derive(Debug)]
pub struct LayoutElement {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub element_type: String,
    pub probability: f32,
    pub source: ModelType,
}

pub struct Detectron2ONNXModel {
    model_type: ModelType,
    model: ort::Session,
    confidence_threshold: f32,
}

impl Detectron2ONNXModel {
    const REQUIRED_WIDTH: u32 = 800;
    const REQUIRED_HEIGHT: u32 = 1035;
    const DEFAULT_LABEL_MAP: [&'static str; 5] = ["Text", "Title", "List", "Table", "Figure"];
    const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.8;

    pub fn new(model_type: ModelType) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let filename = api
            .model(model_type.hf_repo().to_string())
            .get(model_type.hf_filename())?;

        let model = Session::builder()?.commit_from_file(filename)?;

        Ok(Self {
            model,
            model_type,
            confidence_threshold: Self::DEFAULT_CONFIDENCE_THRESHOLD,
        })
    }

    pub fn new_configured(
        model_type: ModelType,
        session_builder: SessionBuilder,
        confidence_threshold: f32,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let filename = api
            .model(model_type.hf_repo().to_string())
            .get(model_type.hf_filename())?;

        let model = session_builder.commit_from_file(filename)?;

        Ok(Self {
            model,
            model_type,
            confidence_threshold,
        })
    }

    pub fn predict(&self, img: &image::DynamicImage) -> Result<Vec<LayoutElement>> {
        let (img_width, img_height, input) = self.preprocess(img);

        let run_result = self.model.run(ort::inputs!["x.1" => input]?);
        match run_result {
            Ok(outputs) => {
                let elements = self.postprocess(&outputs, img_width, img_height)?;
                return Ok(elements);
            }
            Err(_err) => {
                tracing::warn!(
                    "Ignoring runtime error from onnx (likely due to encountering blank page)."
                );
                return Ok(vec![]);
            }
        }
    }

    fn preprocess(
        &self,
        img: &image::DynamicImage,
    ) -> (u32, u32, ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>) {
        let (img_width, img_height) = (img.width(), img.height());
        let img = img.resize_exact(
            Self::REQUIRED_WIDTH,
            Self::REQUIRED_HEIGHT,
            imageops::FilterType::Triangle,
        );
        let img_rgb8 = img.as_rgba8().unwrap();

        let mut input = Array::zeros((3, 1035, 800));

        for pixel in img_rgb8.enumerate_pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2 .0;
            input[[0, y, x]] = r as f32;
            input[[1, y, x]] = g as f32;
            input[[2, y, x]] = b as f32;
        }

        return (img_width, img_height, input);
    }

    fn postprocess<'s>(
        &self,
        outputs: &SessionOutputs<'s>,
        img_width: u32,
        img_height: u32,
    ) -> Result<Vec<LayoutElement>> {
        let bboxes = &outputs[0].try_extract_tensor::<f32>()?;
        let labels = &outputs[1].try_extract_tensor::<i64>()?;
        let confidence_scores = &outputs[self.confidence_index()].try_extract_tensor::<f32>()?;

        let width_conversion = img_width as f32 / Self::REQUIRED_WIDTH as f32;
        let height_conversion = img_height as f32 / Self::REQUIRED_HEIGHT as f32;

        let mut elements = vec![];

        for (bbox, (label, confidence_score)) in bboxes
            .rows()
            .into_iter()
            .zip(labels.iter().zip(confidence_scores))
        {
            let [x1, y1, x2, y2] = vec_to_bbox(bbox.iter().copied().collect());

            let detected_label = Self::DEFAULT_LABEL_MAP[*label as usize];

            if *confidence_score > self.confidence_threshold as f32 {
                elements.push(LayoutElement {
                    x1: (x1 * width_conversion),
                    y1: (y1 * height_conversion),
                    x2: (x2 * width_conversion),
                    y2: (y2 * height_conversion),
                    element_type: detected_label.to_string(),
                    probability: *confidence_score,
                    source: self.model_type,
                })
            }
        }

        elements.sort_by(|a, b| a.y1.total_cmp(&b.y1));

        return Ok(elements);
    }

    fn confidence_index(&self) -> usize {
        if self.model_type.hf_repo().contains("R_50") {
            2
        } else {
            3
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    FasterRCNN,
    MaskRCNN,
}

impl ModelType {
    pub fn hf_repo(&self) -> &str {
        match self {
            Self::FasterRCNN => "unstructuredio/detectron2_faster_rcnn_R_50_FPN_3x",
            Self::MaskRCNN => "unstructuredio/detectron2_mask_rcnn_X_101_32x8d_FPN_3x",
        }
    }

    pub fn hf_filename(&self) -> &str {
        match self {
            _ => "model.onnx",
        }
    }
}
