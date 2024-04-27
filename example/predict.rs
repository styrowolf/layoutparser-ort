use layoutparser_ort::{Detectron2ONNXModel, Result};

fn main() -> Result<()> {
    let img = image::open("data/paper-example.png").unwrap();

    let model = Detectron2ONNXModel::new(layoutparser_ort::ModelType::FasterRCNN)?;

    let predictions = model.predict(&img)?;

    println!("{:?}", predictions);

    Ok(())
}
