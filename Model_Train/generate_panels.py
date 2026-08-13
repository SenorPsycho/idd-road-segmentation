"""
generate_panels.py -- RoadVision Nepal Phase 2: qualitative comparison panels.

Reads eval_metrics.csv (produced by evaluate.py) to select a spread of val
examples by IoU, re-runs those specific frames through the trained model, and
builds a single grid image showing input -> predicted mask -> overlay for
each selected example.

Selection (9 examples total by default):
  - N BEST   (highest IoU)
  - N WORST  (lowest IoU)
  - N MEDIAN, spread across the 25th/50th/75th percentile of the IoU
    distribution -- NOT three images clustered right at the exact median,
    which would look nearly identical to each other and tell you nothing new.
    Spreading across percentiles gives an honest look at "typical"
    performance across its actual range.

Usage (from repo root, after evaluate.py has already been run for this run_dir):
    python -m Model_Train.generate_panels --run_dir runs/2026-08-11_1628
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from Data_Prep.dataset import IDDBinarySegDataset, get_val_transforms
from Model_Train.model import build_model


def load_eval_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["iou"] = float(row["iou"])
            rows.append(row)
    return rows


def select_examples(rows, n_per_category=3):
    """
    Returns a list of (row, category_label) tuples: N best, N median
    (percentile-spread), N worst -- in that display order.
    """
    rows_sorted = sorted(rows, key=lambda r: r["iou"])
    n = len(rows_sorted)

    worst = [(r, "WORST") for r in rows_sorted[:n_per_category]]
    best = [(r, "BEST") for r in rows_sorted[-n_per_category:]]

    if n_per_category == 3:
        percentiles = [0.25, 0.50, 0.75]
    else:
        percentiles = [i / (n_per_category + 1) for i in range(1, n_per_category + 1)]

    median = []
    for p in percentiles:
        idx = min(int(p * n), n - 1)
        median.append((rows_sorted[idx], "MEDIAN"))

    return best + median + worst


def build_overlay(image_rgb, pred_mask, alpha=0.45):
    """
    image_rgb: (H, W, 3) uint8 display image.
    pred_mask: (H, W) array with values in {0, 1} (1 = predicted drivable).
    Highlights predicted drivable area in green, alpha-blended over the input.
    """
    overlay = image_rgb.copy()
    green = np.zeros_like(image_rgb)
    green[..., 1] = 255
    drivable = pred_mask == 1
    overlay[drivable] = (
        (1 - alpha) * image_rgb[drivable] + alpha * green[drivable]
    ).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative comparison panels.")
    parser.add_argument("--run_dir", type=str, required=True,
                         help="Run folder containing config.yaml, best.pt, and eval_metrics.csv "
                              "(e.g. runs/2026-08-11_1628)")
    parser.add_argument("--n_per_category", type=int, default=3,
                         help="Number of BEST / MEDIAN / WORST examples each (default 3, total 9).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best.pt"
    eval_csv_path = run_dir / "eval_metrics.csv"
    output_path = run_dir / "qualitative_panels.png"

    if not eval_csv_path.exists():
        raise FileNotFoundError(
            f"{eval_csv_path} not found -- run evaluate.py first to generate per-image metrics."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best.pt not found in {run_dir}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = (config["data"]["image_size"], config["data"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Select examples from eval_metrics.csv ---
    rows = load_eval_csv(eval_csv_path)
    selected = select_examples(rows, n_per_category=args.n_per_category)
    print(f"Selected {len(selected)} examples: "
          f"{args.n_per_category} best, {args.n_per_category} median (percentile-spread), "
          f"{args.n_per_category} worst")

    # --- Dataset lookup: map frame_id -> (img_path, mask_path) ---
    # Rebuilding the dataset here (rather than reusing evaluate.py's) keeps this
    # script fully independent -- it only needs config.yaml + eval_metrics.csv.
    val_dataset = IDDBinarySegDataset(
        root=config["data"]["root_dir"],
        split="val",
        img_size=img_size,
        transforms=get_val_transforms(img_size),
    )
    frame_id_to_paths = {}
    for img_path, mask_path in val_dataset.samples:
        frame_id = mask_path.name.replace("_gtFine_binary.png", "")
        frame_id_to_paths[frame_id] = (img_path, mask_path)

    # --- Model ---
    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,  # loading trained weights below
        classes=config["model"]["classes"],
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_transforms = get_val_transforms(img_size)

    # --- Build each row: input | predicted mask | overlay ---
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row_idx, (row, category) in enumerate(selected):
            frame_id = row["frame_id"]
            iou = row["iou"]

            if frame_id not in frame_id_to_paths:
                print(f"  [warn] frame_id {frame_id} from eval_metrics.csv not found in current "
                      f"val_dataset -- skipping (dataset may have changed since evaluate.py ran)")
                continue

            img_path, _ = frame_id_to_paths[frame_id]

            # Raw image for display, resized to match model input size.
            image_bgr = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb_resized = cv2.resize(image_rgb, (img_size[1], img_size[0]))

            # Normalized tensor for the model. Mask arg is required by the
            # Compose call signature but its content is irrelevant here --
            # we only use the returned "image".
            augmented = val_transforms(
                image=image_rgb, mask=np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            )
            input_tensor = augmented["image"].unsqueeze(0).to(device)

            logits = model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (H, W) in {0,1}

            pred_mask_display = (pred_mask * 255).astype(np.uint8)
            overlay = build_overlay(image_rgb_resized, pred_mask)

            axes[row_idx, 0].imshow(image_rgb_resized)
            axes[row_idx, 0].set_title(f"{category} (IoU={iou:.4f})\ninput", fontsize=9)
            axes[row_idx, 1].imshow(pred_mask_display, cmap="gray")
            axes[row_idx, 1].set_title("predicted mask", fontsize=9)
            axes[row_idx, 2].imshow(overlay)
            axes[row_idx, 2].set_title("overlay", fontsize=9)

            for col in range(3):
                axes[row_idx, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nQualitative panel grid saved to {output_path}")


if __name__ == "__main__":
    main()