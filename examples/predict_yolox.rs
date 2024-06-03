use layoutparser_ort::{
    models::{YOLOXModel, YOLOXPretrainedModels},
    Result,
};

fn main() -> Result<()> {
    let img = image::open("examples/data/paper-example.png").unwrap();

    let model = YOLOXModel::pretrained(YOLOXPretrainedModels::Tiny)?;

    let predictions = model.predict(&img)?;

    println!("{:?}", predictions);

    Ok(())
}
