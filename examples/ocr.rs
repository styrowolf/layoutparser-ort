use layoutparser_ort::{
    models::{Detectron2Model, Detectron2PretrainedModels},
    ocr::TesseractAgent,
    Result,
};

fn main() -> Result<()> {
    let img = image::open("examples/data/paper-example.png").unwrap();

    let model = Detectron2Model::pretrained(Detectron2PretrainedModels::FASTER_RCNN_R_50_FPN_3X)?;

    let mut predictions = model.predict(&img)?;

    let mut agent = TesseractAgent::new()?;

    for pred in predictions.iter_mut().filter(|e| e.element_type == "Text") {
        pred.pad(5.0);
        agent.extract_text_to_lm(pred, &img)?;
        println!("{:?}", pred.text.as_ref().unwrap());
    }

    Ok(())
}
