# layoutparser-ort

impl support for tesseract agent etc.
impl async support
integrate geo types into LayoutElement
clarify licensing
rename library??

## libraries to port from and view
- [surya](https://github.com/VikParuchuri/surya): OCR, layout analysis, reading order, line detection in 90+ languages
    - SegFormer (transformers: SegFormer), Donut (transformers: Donut), CRAFT (pytorch)
    - License: GPLv3.0 (code), cc-by-nc-sa-4.0 (models)
        - cc-by-nc-sa-4.0: noncommerical but author "waive[s] that for any organization under $5M USD in gross revenue in the most recent 12-month period."
- [unstructured-inference](https://github.com/Unstructured-IO/unstructured-inference/): hosted model inference code for layout parsing models
    - Models: Detectron2 (LayoutParser-PubLayNet-PyTorch, LayoutParser-PubLayNet-ONNX), YOLOX (probably trained on DocLayNet, Quantized, ONNX), Table-Transformer (transformers: Table Transformer), Donut (transformers: Donut)
    - License: Apache 2.0
- [LayoutParser](https://github.com/Layout-Parser/layout-parser): A Unified Toolkit for Deep Learning Based Document Image Analysis 
    - Models: Detectron2
    - License: Apache 2.0
    - Documentation: https://layout-parser.readthedocs.io/en/latest/api_doc/elements.html