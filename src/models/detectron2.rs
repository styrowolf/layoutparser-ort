use image::imageops;
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use ort::{Session, SessionBuilder, SessionOutputs};

pub use crate::error::{Error, Result};
use crate::{utils::vec_to_bbox, LayoutElement};

pub struct Detectron2Model {
    model_name: String,
    model: ort::Session,
    confidence_threshold: f32,
    label_map: Vec<(i64, String)>,
    confidence_score_index: usize,
}

#[allow(non_camel_case_types)]
pub enum Detectron2PretrainedModel {
    FASTER_RCNN_R_50_FPN_3X,
    MASK_RCNN_X_101_32X8D_FPN_3x,
}

impl Detectron2PretrainedModel {
    pub fn name(&self) -> &str {
        match self {
            _ => self.hf_repo(),
        }
    }

    pub fn hf_repo(&self) -> &str {
        match self {
            Self::FASTER_RCNN_R_50_FPN_3X => "unstructuredio/detectron2_faster_rcnn_R_50_FPN_3x",
            Self::MASK_RCNN_X_101_32X8D_FPN_3x => {
                "unstructuredio/detectron2_mask_rcnn_X_101_32x8d_FPN_3x"
            }
        }
    }

    pub fn hf_filename(&self) -> &str {
        match self {
            Self::FASTER_RCNN_R_50_FPN_3X => "model.onnx",
            Self::MASK_RCNN_X_101_32X8D_FPN_3x => "model.onnx",
        }
    }

    pub fn label_map(&self) -> Vec<(i64, String)> {
        match self {
            Detectron2PretrainedModel::FASTER_RCNN_R_50_FPN_3X => {
                ["Text", "Title", "List", "Table", "Figure"]
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (i as i64, l.to_string()))
                    .collect()
            }
            Detectron2PretrainedModel::MASK_RCNN_X_101_32X8D_FPN_3x => {
                ["Text", "Title", "List", "Table", "Figure"]
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (i as i64, l.to_string()))
                    .collect()
            }
        }
    }

    pub fn confidence_score_index(&self) -> usize {
        match self {
            Detectron2PretrainedModel::FASTER_RCNN_R_50_FPN_3X => 2,
            Detectron2PretrainedModel::MASK_RCNN_X_101_32X8D_FPN_3x => 3,
        }
    }
}

impl Detectron2Model {
    pub const REQUIRED_WIDTH: u32 = 800;
    pub const REQUIRED_HEIGHT: u32 = 1035;
    pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.8;

    pub fn pretrained(p_model: Detectron2PretrainedModel) -> Result<Self> {
        let session_builder = Session::builder()?;
        let api = hf_hub::api::sync::Api::new()?;
        let filename = api
            .model(p_model.hf_repo().to_string())
            .get(p_model.hf_filename())?;

        let model = session_builder.commit_from_file(filename)?;

        Ok(Self {
            model_name: p_model.name().to_string(),
            model,
            label_map: p_model.label_map(),
            confidence_threshold: Self::DEFAULT_CONFIDENCE_THRESHOLD,
            confidence_score_index: p_model.confidence_score_index(),
        })
    }

    pub fn configure_pretrained(
        p_model: Detectron2PretrainedModel,
        confidence_threshold: f32,
        session_builder: SessionBuilder,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let filename = api
            .model(p_model.hf_repo().to_string())
            .get(p_model.hf_filename())?;

        let model = session_builder.commit_from_file(filename)?;

        Ok(Self {
            model_name: p_model.name().to_string(),
            model,
            label_map: p_model.label_map(),
            confidence_threshold,
            confidence_score_index: p_model.confidence_score_index(),
        })
    }

    pub fn new_from_file(
        file_path: &str,
        model_name: &str,
        label_map: &[(i64, &str)],
        confidence_threshold: f32,
        confidence_score_index: usize,
        session_builder: SessionBuilder,
    ) -> Result<Self> {
        let model = session_builder.commit_from_file(file_path)?;

        Ok(Self {
            model_name: model_name.to_string(),
            model,
            label_map: label_map.iter().map(|(i, l)| (*i, l.to_string())).collect(),
            confidence_threshold,
            confidence_score_index,
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
        let img_rgb8 = img.into_rgba8();

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
        let confidence_scores =
            &outputs[self.confidence_score_index].try_extract_tensor::<f32>()?;

        let width_conversion = img_width as f32 / Self::REQUIRED_WIDTH as f32;
        let height_conversion = img_height as f32 / Self::REQUIRED_HEIGHT as f32;

        let mut elements = vec![];

        for (bbox, (label, confidence_score)) in bboxes
            .rows()
            .into_iter()
            .zip(labels.iter().zip(confidence_scores))
        {
            let [x1, y1, x2, y2] = vec_to_bbox(bbox.iter().copied().collect());

            let detected_label = &self
                .label_map
                .iter()
                .find(|(l_i, _)| l_i == label)
                .unwrap()
                .1;

            if *confidence_score > self.confidence_threshold as f32 {
                elements.push(LayoutElement::new(
                    x1 * width_conversion,
                    y1 * height_conversion,
                    x2 * width_conversion,
                    y2 * height_conversion,
                    &detected_label,
                    *confidence_score,
                    &self.model_name,
                ))
            }
        }

        elements.sort_by(|a, b| a.bbox.max().y.total_cmp(&b.bbox.max().y));

        return Ok(elements);
    }
}
