use layoutparser_ort::{
    models::detectron2::{Detectron2Model, Detectron2PretrainedModel},
    Result,
};

fn main() -> Result<()> {
    let img = image::open("examples/data/paper-example.png").unwrap();

    let model = Detectron2Model::pretrained(Detectron2PretrainedModel::FASTER_RCNN_R_50_FPN_3X)?;

    let predictions = model.predict(&img)?;

    for pred in &predictions {
        println!(
            "Label: {}, Confidence: {}",
            pred.element_type, pred.score
        );
    }

    println!("{:?}", predictions);

    Ok(())
}
