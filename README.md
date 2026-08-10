# RoadVision Nepal Phase 2: IDD Baseline Segmentation

A ResNet50 + U-Net baseline for binary road segmentation (drivable vs. 
non-drivable) trained on the Indian Driving Dataset (IDD), with evaluation 
on validation splits and qualitative testing on Kathmandu road imagery.

## Overview

This project addresses Phase 1's failure: classical CV methods break on 
unmarked, semi-structured Indian roads. Phase 2 establishes a deep learning 
baseline using the [Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/) 
as a structurally similar proxy, before deploying inference on Kathmandu roads.

**Key question answered in Phase 1:** Unmarked roads cause classical segmentation 
to fail. **Phase 2 approach:** Train a supervised baseline on IDD, then test 
generalization to real Kathmandu imagery.

## Method

**Model.** ResNet50 encoder (ImageNet-pretrained) + U-Net decoder, binary 
output (background=0, drivable=1). Training alternates between frozen-encoder 
warmup (stabilize decoder) and full fine-tuning (layer4+).

**Loss.** Combined CE + Dice (50/50 split, tunable via config) to handle 
class imbalance (drivable area typically 10–30% of image).

**Training Loop.** Per-epoch metrics (IoU, precision, recall), checkpointing 
by best validation IoU, reproducibility via seed + config snapshot + pip 
freeze + commit hash.

**Augmentation.** Flip, color jitter, conservative rotation (baseline-level 
conservatism to avoid overfitting early).

## Dataset

[Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/) — 10k+ images 
of Indian roads (marked and unmarked) with 34-class semantic segmentation 
masks. For this baseline, labels are collapsed to binary: 
drivable (road surface) vs. non-drivable (vegetation, sky, etc.).

Input: 3-channel RGB, 512×512.  
Output: binary mask (drivable/non-drivable).

## Setup

```bash
git clone https://github.com/SenorPsycho/Casting-defect-inspector.git
cd Road_Segmentation

# Create venv and install
python -m venv .env
.\.env\Scripts\Activate.ps1  # Windows
source .env/bin/activate      # Linux/Mac

pip install -r requirements.txt
```

Download IDD (Parts I + II) and organize under `Dataset/` as per 
`Dataset/README.md`.

## Results

*(populated after Phase 2 baseline training completes)*

| Split | IoU | Precision | Recall | Comment |
|---|---|---|---|---|
| Validation | – | – | – | – |
| Test (held-out) | – | – | – | – |

Qualitative: sample predictions on IDD val set + live inference on 
Kathmandu imagery (Phase 2's differentiator).

## Training

```bash
python Model_Train/train.py
```

Config hyperparameters in `Model_Train/config.yaml`. Training snapshots 
logged to `Model_Train/runs/<timestamp>/`.

## Limitations

- Binary only; does not distinguish road type (asphalt, dirt, etc.).
- Trained on IDD (Indian but still urban/marked-road-heavy); transfer to 
  unmarked Kathmandu roads is a key Phase 2 ablation.
- Evaluated on validation splits; full cross-dataset generalization testing 
  deferred to Phase 2 live deployment.

## Roadmap

- [ ] Phase 2a: Full training run on college PC, benchmark on IDD val
- [ ] Phase 2b: Qualitative inference on Kathmandu imagery (owns generalization test)
- [ ] Phase 2c: Streamlit/Gradio demo (real-time upload → segmentation overlay)
- [ ] Phase 3: Multi-class road-type segmentation (asphalt vs. dirt vs. pothole)
- [ ] Phase 3+: Temporal modeling (video sequences for consistency)

## License

MIT
