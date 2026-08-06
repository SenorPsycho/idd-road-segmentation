# Casting Defect Inspector

A binary classification pipeline for automated visual inspection of cast
industrial parts, paired with Grad-CAM-based interpretability and an
LLM-generated explanation layer, served through an interactive Streamlit app.

![App screenshot](assets/screenshots/app_demo.png)

## Overview

Cast industrial parts (submersible pump impellers) are visually inspected for
surface defects such as blowholes, cracks, and porosity. This project
implements and compares two transfer-learning strategies for automating that
classification, evaluates model behavior using Grad-CAM, and adds a
plain-language explanation layer intended to make model output legible to a
non-technical QA operator.

## Method

**Classification.** ResNet50 (ImageNet-pretrained) is evaluated under two
regimes: (1) a frozen-backbone linear probe, and (2) fine-tuning with
`layer4` unfrozen. Both are trained and evaluated on the same stratified
train/validation/test split, with the test set held out until final
evaluation.

**Interpretability.** Grad-CAM (and Grad-CAM++, compared as a secondary
ablation) is used to localize the image region driving each prediction.

**Explanation.** The predicted label, confidence, and Grad-CAM region are
passed to Claude Haiku, which generates a short natural-language description
of the flagged region — intended as a QA-readable supplement to the raw
prediction, not a replacement for the classifier's output.

**Application.** A Streamlit interface accepts an uploaded or selected image
and displays the prediction, Grad-CAM overlay, and generated explanation
together.

## Results

*(populated after the fine-tuning ablation — precision/recall/F1/ROC-AUC for
both regimes on the held-out test set)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Linear-probe baseline | – | – | – | – | – |
| Fine-tuned (layer4)   | – | – | – | – | – |

## Dataset

[Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(Kaggle) — 7,348 grayscale images of submersible pump impellers, pre-split
into train/test, labeled `ok_front` / `def_front`.

## Setup

```bash
git clone https://github.com/<you>/casting-defect-inspector
cd casting-defect-inspector
pip install -r requirements.txt
```

Dataset download instructions are in `data/README.md`.

## Limitations

- Binary classification only; does not distinguish defect type.
- Trained on a single dataset from a single imaging setup — generalization
  to other parts, lighting, or camera configurations is untested.
- LLM explanations are descriptive, not diagnostic — they summarize the
  Grad-CAM region in text but are not grounded in a domain-specific
  knowledge base.

## Roadmap

- [ ] Multi-class defect categorization
- [ ] Confidence calibration (temperature scaling, reliability diagram)
- [ ] Grad-CAM vs. Grad-CAM++ ablation, formalized
- [ ] RAG-based explanation layer grounded in QA documentation

## License

MIT
