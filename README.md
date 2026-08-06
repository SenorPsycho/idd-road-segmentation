# Casting Defect Inspector

Automated visual quality inspection for industrial casting defects using a binary
classifier (ok vs. defective) for submersible pump impellers, paired with
Grad-CAM interpretability and an LLM-generated plain-English QA explanation,
served through an interactive Streamlit app.

Built as a fast, defensible demonstration of an end-to-end CV + LLM pipeline:
transfer learning, model interpretability, and applied LLM integration in a
real-world industrial automation context.

**[Live demo →](#)** *(link once deployed)*

![App screenshot](assets/screenshots/app_demo.png)

## Problem

Manual visual inspection of cast industrial parts is slow, inconsistent, and
doesn't scale. This project automates defect detection on grayscale images of
submersible pump impellers, and critically explains *why* a part was
flagged, so the output is usable by a QA operator, not just a model score.

## Pipeline

1. **Classification** — ResNet50 backbone (ImageNet-pretrained), evaluated as
   both a frozen linear-probe baseline and a fine-tuned (`layer4` unfrozen)
   variant, compared head-to-head.
2. **Interpretability** — Grad-CAM (and Grad-CAM++) overlay showing which
   region of the part drove the prediction.
3. **Explanation** — the prediction, confidence, and Grad-CAM region are
   passed to the LLM, which generates a short, plain-English QA-style
   explanation an inspector could actually read.
4. **App** — a Streamlit interface: upload or select an image, see the
   prediction, heatmap, and explanation together in one screen.

## Results

*(filled in after Session 2 — ablation table: linear-probe baseline vs.
fine-tuned, precision/recall/F1/ROC-AUC on the held-out test set)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Linear-probe baseline | – | – | – | – | – |
| Fine-tuned (layer4)   | – | – | – | – | – |

## Dataset

[Real-life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(Kaggle) — ~7,300 grayscale images of submersible pump impellers, pre-split
train/test, labeled `ok_front` / `def_front`.

## Setup

```bash
git clone https://github.com/<you>/casting-defect-inspector
cd casting-defect-inspector
pip install -r requirements.txt
# download dataset into data/casting_data/ — see data/README.md
```

## Roadmap

- [ ] Multi-class defect categorization (beyond binary ok/defective)
- [ ] Confidence calibration (temperature scaling + reliability diagram)
- [ ] Grad-CAM vs. Grad-CAM++ ablation
- [ ] RAG-based explanation layer (grounded in a QA knowledge base rather
      than a general LLM prompt)

## License

MIT
