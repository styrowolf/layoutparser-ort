//! Implemented layout models.

mod detectron2;
mod yolox;

pub use detectron2::{Detectron2Model, Detectron2PretrainedModels};
pub use yolox::{YOLOXModel, YOLOXPretrainedModels};
