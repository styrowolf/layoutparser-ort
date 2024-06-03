use layoutparser_ort::{
    models::{Detectron2Model, Detectron2PretrainedModels},
    Result,
};

fn main() -> Result<()> {
    let img = image::open("examples/data/paper-example.png").unwrap();

    let model = Detectron2Model::pretrained(Detectron2PretrainedModels::FASTER_RCNN_R_50_FPN_3X)?;

    let predictions = model.predict(&img)?;

    println!("{:?}", predictions);

    Ok(())
}
