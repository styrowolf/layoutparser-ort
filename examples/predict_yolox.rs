use layoutparser_ort::{
    models::yolox::{YOLOXModel, YOLOXPretrainedModel},
    Result,
};

fn main() -> Result<()> {
    let img = image::open("examples/data/paper-example.png").unwrap();

    let model = YOLOXModel::pretrained(YOLOXPretrainedModel::Tiny)?;

    let predictions = model.predict(&img)?;

    println!("{:?}", predictions);

    Ok(())
}
