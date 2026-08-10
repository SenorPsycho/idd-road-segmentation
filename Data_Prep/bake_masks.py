"""
bake_masks.py

One-time preprocessing step: converts IDD Segmentation's raw JSON polygon
annotations (gtFine/<split>/<seq>/<frame>_gtFine_polygons.json) into baked
binary PNG masks (drivable vs non-drivable vs ignore).

Class-collapse decision (logged):
  - level1Id == 0 (road, parking, drivable fallback)         -> 1 (drivable)
  - level1Id in {1..6} (everything else that's a real class) -> 0 (non-drivable)
  - level1Id == 255 (unlabeled, ego vehicle, rectification
    border, out of roi, license plate)                       -> 255 (ignore)

Canvas starts filled with 255 (ignore), not 0. This matters: any pixel not
covered by ANY polygon in the JSON (i.e. a gap in annotation) stays ignored
rather than silently becoming a false "non-drivable" label. Polygons are
drawn in the order they appear in the JSON, which reflects IDD's occlusion
/ z-ordering convention (later objects drawn on top of earlier ones).

Usage:
    python bake_masks.py --root /path/to/idd_root --split train
    python bake_masks.py --root /path/to/idd_root --split val
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# --- LUT: IDD class name -> level1Id, collapsed from the AutoNUE labels.py table ---
# (name -> level1Id). Only level1Id matters for the binary collapse.
NAME_TO_LEVEL1ID = {
    "road": 0, "parking": 0, "drivable fallback": 0,
    "sidewalk": 1, "rail track": 1, "non-drivable fallback": 1,
    "person": 2, "animal": 2, "rider": 2,
    "motorcycle": 3, "bicycle": 3, "autorickshaw": 3, "car": 3, "truck": 3,
    "bus": 3, "caravan": 3, "trailer": 3, "train": 3, "vehicle fallback": 3,
    "curb": 4, "wall": 4, "fence": 4, "guard rail": 4, "billboard": 4,
    "traffic sign": 4, "traffic light": 4, "pole": 4, "polegroup": 4,
    "obs-str-bar-fallback": 4,
    "building": 5, "bridge": 5, "tunnel": 5, "vegetation": 5,
    "sky": 6, "fallback background": 6,
    "unlabeled": 255, "ego vehicle": 255, "rectification border": 255,
    "out of roi": 255, "license plate": 255,
}

IGNORE_VALUE = 255


def collapse_level1_to_binary(level1id: int) -> int:
    """Maps a raw level1Id to the binary drivable-area scheme."""
    if level1id == 0:
        return 1  # drivable
    if level1id == 255:
        return IGNORE_VALUE  # ignore
    return 0  # non-drivable (level1Id 1-6)


def rasterize_polygons_to_mask(json_path: Path) -> np.ndarray:
    """Reads one *_gtFine_polygons.json and returns a (H, W) uint8 binary mask."""
    with open(json_path, "r") as f:
        data = json.load(f)

    h, w = data["imgHeight"], data["imgWidth"]
    canvas = Image.new("L", (w, h), color=IGNORE_VALUE)
    draw = ImageDraw.Draw(canvas)

    skipped_labels = set()
    for obj in data["objects"]:
        if obj.get("deleted", 0) == 1:
            continue

        label_name = obj["label"]
        # IDD sometimes suffixes instance labels with "group" (e.g. "cargroup");
        # strip that to fall back to the base class for the LUT lookup.
        if label_name not in NAME_TO_LEVEL1ID and label_name.endswith("group"):
            label_name = label_name[: -len("group")]

        if label_name not in NAME_TO_LEVEL1ID:
            skipped_labels.add(obj["label"])
            continue

        level1id = NAME_TO_LEVEL1ID[label_name]
        fill_value = collapse_level1_to_binary(level1id)

        polygon = obj["polygon"]
        if len(polygon) < 3:
            continue  # degenerate polygon, can't fill
        flat_coords = [tuple(pt) for pt in polygon]
        draw.polygon(flat_coords, fill=fill_value)

    if skipped_labels:
        # Not fatal, but worth knowing about -- means the LUT is missing a
        # class name actually present in this release of the dataset.
        print(f"  [warn] {json_path.name}: unrecognized labels skipped: {skipped_labels}")

    return np.array(canvas, dtype=np.uint8)


def bake_split(root: Path, split: str, out_subdir: str = "gtFine_binary"):
    gt_dir = root / "gtFine" / split
    out_dir = root / out_subdir / split
    json_files = sorted(gt_dir.rglob("*_gtFine_polygons.json"))

    print(f"[{split}] found {len(json_files)} polygon JSON files under {gt_dir}")
    if len(json_files) == 0:
        print(f"  [!] No JSON files found -- check that --root points at the "
              f"folder containing gtFine/, not gtFine/ itself.")
        return

    n_written, n_failed = 0, 0
    for json_path in json_files:
        try:
            mask = rasterize_polygons_to_mask(json_path)
        except Exception as e:
            print(f"  [error] failed on {json_path}: {e}")
            n_failed += 1
            continue

        seq_dir = json_path.parent.name
        frame_id = json_path.name.replace("_gtFine_polygons.json", "")
        dest_dir = out_dir / seq_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{frame_id}_gtFine_binary.png"
        Image.fromarray(mask, mode="L").save(dest_path)
        n_written += 1

        if n_written % 1000 == 0:
            print(f"  ...{n_written}/{len(json_files)} baked")

    print(f"[{split}] done: {n_written} written, {n_failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                         help="Dataset root containing gtFine/ and leftImg8bit/")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"],
                         help="gtFine has no test masks -- only train/val are baked")
    args = parser.parse_args()

    bake_split(args.root, args.split)
