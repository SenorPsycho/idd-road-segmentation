"""
evaluate.py -- RoadVision Nepal Phase 2: full val-set evaluation.

Loads a trained checkpoint (best.pt) from a given run folder and runs a single
inference-only pass over the ENTIRE val split (2,036 images), producing:

  1. eval_metrics.csv in the run folder -- one row per val image (IoU,
     precision, recall, tp/fp/fn/tn), keyed by frame_id. This is a free
     byproduct of the same forward pass and is meant to be used later to pick
     best/worst/median examples for qualitative panels, without eyeballing
     images blind.

  2. A final aggregate IoU/precision/recall printed to console, computed via
     the exact same accumulation method used during training
     (MetricsAccumulator in metrics.py) -- this is the headline number for
     the writeup, NOT an average of the per-image numbers (per-image
     averaging would be skewed by images with different drivable-pixel
     content -- same reasoning as the epoch-level metric during training).

Usage (from repo root, on the machine with the val set + checkpoint):
    python evaluate.py --run_dir runs/2026-08-11_1628
    python evaluate.py --run_dir runs/2026-08-11_1628 --batch_size 32

Notes on design decisions:
- shuffle=False is required in the DataLoader below: this script maps each
  item in a batch back to val_dataset.samples[index] BY POSITION to recover
  its frame_id. Shuffling would break that mapping. dataset.py itself was
  NOT modified to also return frame_id, to avoid touching code that train.py
  already depends on and that's already tested.
- encoder_weights=None when building the model: we're about to load the
  fully-trained state_dict anyway, so there's no need to also download
  ImageNet-pretrained weights first (avoids an unnecessary network dependency
  on a machine that may not have one at eval time).
- torch.no_grad() wraps the entire loop: this is inference only, no
  backward pass, so tracking gradients would only waste memory.
"""

import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from Data_Prep.dataset import IDDBinarySegDataset, get_val_transforms
from Model_Train.model import build_model
from Model_Train.metrics import MetricsAccumulator, batch_confusion_counts, compute_ratios


def main():
    parser = argparse.ArgumentParser(description="Full val-set evaluation for RoadVision Nepal Phase 2.")
    parser.add_argument("--run_dir", type=str, required=True,
                         help="Path to the run folder containing config.yaml and best.pt "
                              "(e.g. runs/2026-08-11_1628)")
    parser.add_argument("--batch_size", type=int, default=16,
                         help="Inference batch size. No gradients are tracked, so this only "
                              "affects speed/memory, never correctness.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best.pt"
    output_csv = run_dir / "eval_metrics.csv"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best.pt not found in {run_dir}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = (config["data"]["image_size"], config["data"]["image_size"])
    ignore_index = config["loss"]["ignore_index"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: running on CPU. A full val-set pass (2,036 images) will be "
              "noticeably slower here than on the college PC's GPU.")

    # --- Dataset / loader ---
    val_dataset = IDDBinarySegDataset(
        root=config["data"]["root_dir"],
        split="val",
        img_size=img_size,
        transforms=get_val_transforms(img_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # required -- see module docstring
        num_workers=config["train"].get("num_workers", 0),
    )
    print(f"Val set size: {len(val_dataset)}")

    # --- Model ---
    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,  # loading trained weights below; no need for imagenet download
        classes=config["model"]["classes"],
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Inference loop ---
    epoch_acc = MetricsAccumulator(ignore_index=ignore_index)
    rows = []
    global_idx = 0  # tracks position in val_dataset.samples; valid because shuffle=False

    with torch.no_grad():
        for batch_num, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)  # (B, 2, H, W)

            # Whole-val-set accumulation -- same method training used per epoch.
            epoch_acc.update(logits, masks)

            # Per-image breakdown -- one row per image in this batch.
            batch_size_actual = images.shape[0]
            for i in range(batch_size_actual):
                img_path, mask_path = val_dataset.samples[global_idx + i]
                frame_id = mask_path.name.replace("_gtFine_binary.png", "")

                tp, fp, fn, tn = batch_confusion_counts(
                    logits[i:i + 1], masks[i:i + 1], ignore_index=ignore_index
                )
                result = compute_ratios(tp, fp, fn, tn)
                rows.append({
                    "frame_id": frame_id,
                    "iou": result["iou"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                })

            global_idx += batch_size_actual

            if batch_num % 20 == 0:
                print(f"  processed {global_idx}/{len(val_dataset)} images...")

    # --- Write per-image CSV ---
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_id", "iou", "precision", "recall", "tp", "fp", "fn", "tn"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-image metrics written to {output_csv} ({len(rows)} rows)")

    # --- Final aggregate (the headline number) ---
    final = epoch_acc.compute()
    print("\n=== Final val-set metrics (whole set, accumulated -- NOT a per-image average) ===")
    print(f"IoU:       {final['iou']:.4f}")
    print(f"Precision: {final['precision']:.4f}")
    print(f"Recall:    {final['recall']:.4f}")
    print(f"TP={final['tp']}  FP={final['fp']}  FN={final['fn']}  TN={final['tn']}")


if __name__ == "__main__":
    main()