"""
check_worst_gt.py -- RoadVision Nepal Phase 2: sanity check on the WORST examples.

The 3 worst-IoU frames from generate_panels.py all came back at EXACTLY
IoU = 0.0000, which is suspicious as a coincidence -- IoU = TP/(TP+FP+FN),
so if a frame's ground-truth mask has ZERO drivable-class (1) pixels, then
TP=0 and FN=0 by definition, and any false-positive prediction collapses
the ratio to exactly 0/(0+FP+0) = 0.0000 regardless of how "close" the
prediction was. That would be a metric floor effect, not three genuinely
catastrophic predictions.

This script just reads the raw ground-truth mask PNGs for the N worst
frame_ids (no model, no inference) and reports the pixel count for each
class ({0: non-drivable, 1: drivable, 255: ignore}), to confirm or rule
out that explanation before it goes in the writeup either way.

Usage:
    python -m Model_Train.check_worst_gt --run_dir runs/2026-08-11_1628 --n_worst 3
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from Data_Prep.dataset import IDDBinarySegDataset, get_val_transforms


def load_eval_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["iou"] = float(row["iou"])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_worst", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    eval_csv_path = run_dir / "eval_metrics.csv"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    img_size = (config["data"]["image_size"], config["data"]["image_size"])

    rows = load_eval_csv(eval_csv_path)
    rows_sorted = sorted(rows, key=lambda r: r["iou"])
    worst = rows_sorted[: args.n_worst]

    # Map frame_id -> mask_path via the dataset's own file-pairing logic,
    # same as evaluate.py / generate_panels.py, so this stays consistent
    # with how those scripts identify files.
    val_dataset = IDDBinarySegDataset(
        root=config["data"]["root_dir"],
        split="val",
        img_size=img_size,
        transforms=get_val_transforms(img_size),
    )
    frame_id_to_mask_path = {}
    for _, mask_path in val_dataset.samples:
        frame_id = mask_path.name.replace("_gtFine_binary.png", "")
        frame_id_to_mask_path[frame_id] = mask_path

    print(f"{'frame_id':<45} {'iou':>8} {'non-drivable(0)':>16} {'drivable(1)':>12} {'ignore(255)':>12}")
    print("-" * 100)

    for row in worst:
        frame_id = row["frame_id"]
        mask_path = frame_id_to_mask_path.get(frame_id)
        if mask_path is None:
            print(f"{frame_id:<45} [not found in current val_dataset]")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        values, counts = np.unique(mask, return_counts=True)
        count_map = dict(zip(values.tolist(), counts.tolist()))

        n0 = count_map.get(0, 0)
        n1 = count_map.get(1, 0)
        n255 = count_map.get(255, 0)

        print(f"{frame_id:<45} {row['iou']:>8.4f} {n0:>16} {n1:>12} {n255:>12}")

    print("\nIf drivable(1) == 0 for these frames, the 0.0000 IoU is a metric floor effect")
    print("(no ground-truth drivable pixels exist to score against), not a genuine worst-case")
    print("failure -- worth noting as a metric caveat in the writeup rather than a model weakness.")
    print("If drivable(1) is clearly nonzero, these are real failure cases and worth digging into.")


if __name__ == "__main__":
    main()
