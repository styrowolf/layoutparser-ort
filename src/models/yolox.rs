use image::imageops;
use itertools::Itertools;
use ndarray::{
    concatenate, s, stack, Array, Array1, Array2, ArrayBase, ArrayD, Axis, Dim, IxDyn, OwnedRepr,
};
use ort::{Session, SessionBuilder, SessionOutputs};

pub use crate::error::Result;
use crate::{utils, LayoutElement};

/// A [`YOLOX`](https://github.com/Megvii-BaseDetection/YOLOX)-based model.
pub struct YOLOXModel {
    model_name: String,
    model: ort::Session,
    is_quantized: bool,
    label_map: Vec<(i64, String)>,
}

#[derive(PartialEq)]
/// Pretrained YOLOX-based models from Hugging Face.
pub enum YOLOXPretrainedModels {
    Large,
    LargeQuantized,
    Tiny,
}

impl YOLOXPretrainedModels {
    /// Model name.
    pub fn name(&self) -> &str {
        match self {
            _ => self.hf_repo(),
        }
    }

    /// Hugging Face repository for this model.
    pub fn hf_repo(&self) -> &str {
        match self {
            _ => "unstructuredio/yolo_x_layout",
        }
    }

    /// Path for this model file in Hugging Face repository.
    pub fn hf_filename(&self) -> &str {
        match self {
            YOLOXPretrainedModels::Large => "yolox_l0.05.onnx",
            YOLOXPretrainedModels::LargeQuantized => "yolox_l0.05_quantized.onnx",
            YOLOXPretrainedModels::Tiny => "yolox_tiny.onnx",
        }
    }

    /// The label map for this model.
    pub fn label_map(&self) -> Vec<(i64, String)> {
        match self {
            _ => Vec::from_iter(
                [
                    (0, "Caption"),
                    (1, "Footnote"),
                    (2, "Formula"),
                    (3, "List-item"),
                    (4, "Page-footer"),
                    (5, "Page-header"),
                    (6, "Picture"),
                    (7, "Section-header"),
                    (8, "Table"),
                    (9, "Text"),
                    (10, "Title"),
                ]
                .iter()
                .map(|(i, l)| (*i as i64, l.to_string())),
            ),
        }
    }
}

impl YOLOXModel {
    /// Required input image width.
    pub const REQUIRED_WIDTH: u32 = 768;
    /// Required input image height.
    pub const REQUIRED_HEIGHT: u32 = 1024;

    /// Construct a [`YOLOXModel`] with a pretrained model downloaded from Hugging Face.
    pub fn pretrained(p_model: YOLOXPretrainedModels) -> Result<Self> {
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
            is_quantized: p_model == YOLOXPretrainedModels::LargeQuantized,
        })
    }

    /// Construct a configured [`YOLOXModel`] with a pretrained model downloaded from Hugging Face.
    pub fn configure_pretrained(
        p_model: YOLOXPretrainedModels,
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
            is_quantized: p_model == YOLOXPretrainedModels::LargeQuantized,
        })
    }

    /// Construct a [`YOLOXModel`] from a model file.
    pub fn new_from_file(
        file_path: &str,
        model_name: &str,
        label_map: &[(i64, &str)],
        is_quantized: bool,
        session_builder: SessionBuilder,
    ) -> Result<Self> {
        let model = session_builder.commit_from_file(file_path)?;

        Ok(Self {
            model_name: model_name.to_string(),
            model,
            label_map: label_map.iter().map(|(i, l)| (*i, l.to_string())).collect(),
            is_quantized,
        })
    }

    /// Predict [`LayoutElement`]s from the image provided.
    pub fn predict(&self, img: &image::DynamicImage) -> Result<Vec<LayoutElement>> {
        // UNWRAP SAFETY: shape unwraps are never a problem because we know the size of the output tensor
        let (input, r) = self.preprocess(img);

        let input_name = &self.model.inputs[0].name;

        let run_result = self.model.run(ort::inputs![input_name => input]?);
        match run_result {
            Ok(outputs) => {
                let predictions = self
                    .postprocess(&outputs, false)?
                    .slice(s![0, .., ..])
                    .to_owned();

                let boxes = predictions
                    .slice(s![.., 0..4])
                    .to_shape([16128, 4])
                    .unwrap()
                    .to_owned();
                let scores = predictions
                    .slice(s![.., 4..5])
                    .to_shape([16128, 1])
                    .unwrap()
                    .to_owned()
                    * predictions.slice(s![.., 5..]);

                let mut boxes_xyxy: Array<f32, _> = ndarray::Array::ones([16128, 4]);

                let s0 =
                    boxes.slice(s![.., 0]).to_owned() - (boxes.slice(s![.., 2]).to_owned() / 2.0);
                let s1 =
                    boxes.slice(s![.., 1]).to_owned() - (boxes.slice(s![.., 3]).to_owned() / 2.0);
                let s2 =
                    boxes.slice(s![.., 0]).to_owned() + (boxes.slice(s![.., 2]).to_owned() / 2.0);
                let s3 =
                    boxes.slice(s![.., 1]).to_owned() + (boxes.slice(s![.., 3]).to_owned() / 2.0);

                boxes_xyxy
                    .slice_mut(s![.., 0])
                    .iter_mut()
                    .zip_eq(s0.iter())
                    .for_each(|(old, new)| *old = *new);
                boxes_xyxy
                    .slice_mut(s![.., 1])
                    .iter_mut()
                    .zip_eq(s1.iter())
                    .for_each(|(old, new)| *old = *new);
                boxes_xyxy
                    .slice_mut(s![.., 2])
                    .iter_mut()
                    .zip_eq(s2.iter())
                    .for_each(|(old, new)| *old = *new);
                boxes_xyxy
                    .slice_mut(s![.., 3])
                    .iter_mut()
                    .zip_eq(s3.iter())
                    .for_each(|(old, new)| *old = *new);

                boxes_xyxy /= r;

                let mut regions = vec![];

                let (nms_thr, score_thr) = if self.is_quantized {
                    (0.0, 0.07)
                } else {
                    (0.1, 0.25)
                };

                let dets = multiclass_nms_class_agnostic(&boxes_xyxy, &scores, nms_thr, score_thr);

                for det in dets.outer_iter() {
                    let [x1, y1, x2, y2, prob, class_id] =
                        extract_bbox_etc(&det.into_iter().copied().collect());
                    let detected_class = self.get_label(class_id as i64);
                    regions.push(LayoutElement::new(
                        x1,
                        y1,
                        x2,
                        y2,
                        &detected_class,
                        prob,
                        &self.model_name,
                    ));
                }

                regions.sort_by(|a, b| a.bbox.max().y.total_cmp(&b.bbox.max().y));

                return Ok(regions);
            }
            Err(_err) => {
                eprintln!("{_err:?}");
                tracing::warn!(
                    "Ignoring runtime error from onnx (likely due to encountering blank page)."
                );
                return Ok(vec![]);
            }
        }
    }

    fn postprocess<'s>(
        &self,
        outputs: &SessionOutputs<'s>,
        p6: bool,
    ) -> Result<Array<f32, Dim<[usize; 3]>>> {
        let output_m = &outputs[0].try_extract_tensor::<f32>()?;
        let mut shaped_output = output_m.to_shape([1, 16128, 16]).unwrap().to_owned();

        let strides = if !p6 {
            vec![8, 16, 32]
        } else {
            vec![8, 16, 32, 64]
        };

        let hsizes: Vec<u32> = strides.iter().map(|s| Self::REQUIRED_HEIGHT / s).collect();
        let wsizes: Vec<u32> = strides.iter().map(|s| Self::REQUIRED_WIDTH / s).collect();

        let mut grids = vec![];
        let mut expanded_strides = vec![];

        for (stride, (hsize, wsize)) in strides.iter().zip(hsizes.iter().zip(wsizes.iter())) {
            let meshgrid_res = meshgrid(
                &[Array1::from_iter(0..*wsize), Array1::from_iter(0..*hsize)],
                Indexing::Xy,
            );
            let xv = meshgrid_res[0].to_owned();
            let yv = meshgrid_res[1].to_owned();

            let grid = stack![Axis(2), xv, yv]
                .to_shape((1, (hsize * wsize) as usize, 2))
                .unwrap()
                .to_owned();

            let shape_1 = &grid.shape()[0..2];
            expanded_strides.push(Array::from_elem((shape_1[0], shape_1[1], 1), stride));

            grids.push(grid);
        }

        let grids =
            ndarray::concatenate(Axis(1), &grids.iter().map(|g| g.view()).collect::<Vec<_>>())
                .unwrap();
        let expanded_strides = ndarray::concatenate(
            Axis(1),
            &expanded_strides
                .iter()
                .map(|g| g.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let s1 = (shaped_output.slice(s![.., .., 0..2]).to_owned() + grids.mapv(|e| e as f32))
            * expanded_strides.mapv(|e| *e as f32);
        let s2 = (shaped_output
            .slice(s![.., .., 2..4])
            .mapv(|e| e.exp())
            .to_owned())
            * expanded_strides.mapv(|e| *e as f32);

        shaped_output
            .slice_mut(s![.., .., 0..2])
            .into_iter()
            .zip_eq(s1.into_iter())
            .for_each(|(old, new)| {
                *old = new;
            });

        shaped_output
            .slice_mut(s![.., .., 2..4])
            .into_iter()
            .zip_eq(s2.into_iter())
            .for_each(|(old, new)| {
                *old = new;
            });

        Ok(shaped_output)
    }

    fn preprocess(
        &self,
        img: &image::DynamicImage,
    ) -> (ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>, f32) {
        let (img_width, img_height) = (img.width(), img.height());

        let mut padded_img: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = Array::ones((
            1,
            3,
            Self::REQUIRED_HEIGHT as usize,
            Self::REQUIRED_WIDTH as usize,
        )) * 114_f32;

        let r: f64 = f64::min(
            Self::REQUIRED_HEIGHT as f64 / img_height as f64,
            Self::REQUIRED_WIDTH as f64 / img_width as f64,
        );

        let resized_img = img.resize_exact(
            (img_width as f64 * r) as u32,
            (img_height as f64 * r) as u32,
            imageops::FilterType::Triangle,
        );

        for pixel in resized_img.into_rgba8().enumerate_pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2 .0;
            padded_img[[0, 0, y, x]] = r as f32;
            padded_img[[0, 1, y, x]] = g as f32;
            padded_img[[0, 2, y, x]] = b as f32;
        }

        (padded_img, r as f32)
    }

    fn get_label(&self, label_id: i64) -> String {
        self.label_map
            .iter()
            .find(|(l_i, _)| l_i == &label_id)
            .unwrap()
            .1
            .clone()
    }
}

fn multiclass_nms_class_agnostic(
    boxes: &Array<f32, Dim<[usize; 2]>>,
    scores: &Array<f32, Dim<[usize; 2]>>,
    nms_thr: f32,
    score_thr: f32,
) -> Array2<f32> {
    let cls_inds = Array1::from_iter(scores.axis_iter(Axis(0)).map(|e| {
        let (max_i, _max) = e.iter().enumerate().fold((0_usize, 0_f32), |acc, (i, e)| {
            let (max_i, max) = acc;
            if *e > max {
                (i, *e)
            } else {
                (max_i, max)
            }
        });
        max_i
    }));

    let cls_scores = Array1::from_iter(
        scores
            .axis_iter(Axis(0))
            .zip_eq(cls_inds.iter())
            .map(|(e, i)| e[*i]),
    );

    let valid_score_mask = cls_scores.mapv(|s| s > score_thr);
    let valid_scores = Array1::from_iter(
        cls_scores
            .iter()
            .zip_eq(valid_score_mask.iter())
            .filter(|(_, b)| **b)
            .map(|(s, _)| *s),
    );

    let valid_boxes: Array2<f32> = to_array2(
        &boxes
            .outer_iter()
            .zip_eq(valid_score_mask.iter())
            .filter(|(_, b)| **b)
            .map(|(s, _)| s.to_owned())
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let valid_cls_inds = Array1::from_iter(
        cls_inds
            .iter()
            .zip_eq(valid_score_mask.iter())
            .filter(|(_, b)| **b)
            .map(|(s, _)| s)
            .collect::<Vec<_>>(),
    );

    let keep = nms(&valid_boxes.to_owned(), &valid_scores, nms_thr);

    let valid_boxes_vec: Vec<_> = valid_boxes.outer_iter().collect();
    let valid_boxes_kept = to_array2(
        &keep
            .iter()
            .map(|i| valid_boxes_vec[*i])
            .map(|e| e.to_owned())
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let valid_scores_vec: Vec<_> = valid_scores.into_iter().collect();
    let valid_scores_kept = to_array2(
        &keep
            .iter()
            .map(|i| valid_scores_vec[*i])
            .map(|e| Array1::from_elem(1, e))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let valid_cls_inds_vec: Vec<_> = valid_cls_inds.into_iter().collect();
    let valid_cls_inds_kept = to_array2(
        &keep
            .iter()
            .map(|i| valid_cls_inds_vec[*i])
            .map(|e| Array1::from_elem(1, e))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    let dets = concatenate(
        Axis(1),
        &[
            valid_boxes_kept.view(),
            valid_scores_kept.view(),
            valid_cls_inds_kept.mapv(|e| *e as f32).view(),
        ],
    )
    .unwrap();

    return dets;
}

fn nms(
    boxes: &Array<f32, Dim<[usize; 2]>>,
    scores: &Array<f32, Dim<[usize; 1]>>,
    nms_thr: f32,
) -> Vec<usize> {
    let x1 = boxes.slice(s![.., 0]);
    let y1 = boxes.slice(s![.., 1]);
    let x2 = boxes.slice(s![.., 2]);
    let y2 = boxes.slice(s![.., 3]);

    let areas = (&x2 - &x1 + 1_f32) * (&y2 - &y1 + 1_f32);
    let mut order = {
        let mut o = utils::argsort_by(&scores, |a, b| a.partial_cmp(b).unwrap());
        o.reverse();
        o
    };

    let mut keep = vec![];

    while !order.is_empty() {
        let i = order[0];
        keep.push(i);

        let order_sliced = Array1::from_iter(order.iter().skip(1));

        let xx1 = order_sliced.mapv(|o_i| f32::max(x1[i], x1[*o_i]));
        let yy1 = order_sliced.mapv(|o_i| f32::max(y1[i], y1[*o_i]));
        let xx2 = order_sliced.mapv(|o_i| f32::min(x2[i], x2[*o_i]));
        let yy2 = order_sliced.mapv(|o_i| f32::min(y2[i], y2[*o_i]));

        let w = ((&xx2 - &xx1) + 1_f32).mapv(|v| f32::max(0.0, v));
        let h = ((&yy2 - &yy1) + 1_f32).mapv(|v| f32::max(0.0, v));
        let inter = w * h;
        let ovr = &inter / (areas[i] + order_sliced.mapv(|e| areas[*e]) - &inter);

        let inds = Array1::from_iter(
            ovr.iter()
                .map(|e| *e <= nms_thr)
                .enumerate()
                .filter(|(_, p)| *p)
                .map(|(i, _)| i),
        );

        drop(order_sliced);

        order = inds.into_iter().map(|i| order[i + 1]).collect();
    }

    return keep;
}

fn to_array2<T: Copy>(source: &[Array1<T>]) -> Result<Array2<T>, impl std::error::Error> {
    let width = source.len();
    let flattened: Array1<T> = source.into_iter().flat_map(|row| row.to_vec()).collect();
    let height = if width == 0 {
        flattened.len()
    } else {
        flattened.len() / width
    };
    flattened.into_shape((width, height))
}

/** [x1, y1, x2, y2, prob, class_id] */
fn extract_bbox_etc(v: &Vec<f32>) -> [f32; 6] {
    [v[0], v[1], v[2], v[3], v[4], v[5]]
}

// from: https://github.com/jreniel/meshgridrs (licensed under MIT)
#[derive(PartialEq)]
pub(crate) enum Indexing {
    Xy,
    Ij,
}
// from: https://github.com/jreniel/meshgridrs (licensed under MIT)
pub(crate) fn meshgrid<T>(
    xi: &[Array1<T>],
    indexing: Indexing,
) -> Vec<ArrayBase<OwnedRepr<T>, Dim<ndarray::IxDynImpl>>>
where
    T: Copy,
{
    let ndim = xi.len();
    let product = xi.iter().map(|x| x.iter()).multi_cartesian_product();

    let mut grids: Vec<ArrayD<T>> = Vec::with_capacity(ndim);

    for (dim_index, _) in xi.iter().enumerate() {
        // Generate a flat vector with the correct repeated pattern
        let values: Vec<T> = product.clone().map(|p| *p[dim_index]).collect();

        let mut grid_shape: Vec<usize> = vec![1; ndim];
        grid_shape[dim_index] = xi[dim_index].len();

        // Determine the correct repetition for each dimension
        for (j, len) in xi.iter().map(|x| x.len()).enumerate() {
            if j != dim_index {
                grid_shape[j] = len;
            }
        }

        let grid = Array::from_shape_vec(IxDyn(&grid_shape), values).unwrap();
        grids.push(grid);
    }

    // Swap axes for "xy" indexing
    if matches!(indexing, Indexing::Xy) && ndim > 1 {
        for grid in &mut grids {
            grid.swap_axes(0, 1);
        }
    }

    grids
}
