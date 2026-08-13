"""
kathmandu_inference.py -- RoadVision Nepal Phase 2: Kathmandu domain-transfer spot check.

Runs the trained IDD checkpoint on a spread of real Kathmandu dashcam frames,
with NO fine-tuning and NO ground truth. There are no annotations for the
Kathmandu frames yet (labeling hasn't started), so this produces visuals only
-- input / predicted mask / overlay -- not a score. The point is showing how
the model behaves on the actual target domain the research is about, not
benchmarking it.

Frame selection: N frames per video, spread evenly across each video's frame
sequence (not consecutive), across all USABLE videos (v01-v05, v07 -- v06
excluded per the known fisheye distortion + burned-in telemetry overlay).

Meant to run on a machine that has BOTH the Kathmandu frames AND a copy of
the checkpoint (best.pt) + its config.yaml -- e.g. copy those two files from
the college PC's runs/<timestamp>/ folder to a local folder first, since the
checkpoint is git-ignored and never goes through the repo.

Usage:
    python -m Model_Train.kathmandu_inference \\
        --frames_root "D:/Dataset/OneDrive/Documents/Projects/Road-Vision-Nepal/..." \\
        --checkpoint eval_checkpoint/best.pt \\
        --config eval_checkpoint/config.yaml \\
        --n_per_video 2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from Data_Prep.dataset import get_val_transforms
from Model_Train.model import build_model

# v06 excluded: fisheye distortion + burned-in telemetry overlay (per project notes).
USABLE_VIDEOS = ["v01", "v02", "v03", "v04", "v05", "v07"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_video_folder(frames_root: Path, video_id: str) -> Path:
    """
    Looks for a folder matching this video id under frames_root. Tries the
    known "<id>_final" convention first, then falls back to any folder whose
    name starts with the video id, since the exact folder-naming convention
    wasn't confirmed ahead of writing this script.
    """
    exact = frames_root / f"{video_id}_final"
    if exact.exists():
        return exact

    candidates = [p for p in frames_root.iterdir() if p.is_dir() and p.name.startswith(video_id)]
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find a folder for {video_id} under {frames_root} "
        f"(tried '{video_id}_final' and any folder starting with '{video_id}')."
    )


def select_frames(video_folder: Path, n_per_video: int):
    """Evenly-spaced frame selection across the sorted file list -- not consecutive."""
    files = sorted(
        p for p in video_folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(files) == 0:
        print(f"  [warn] no image files found in {video_folder}")
        return []

    if len(files) <= n_per_video:
        return files

    indices = np.linspace(0, len(files) - 1, n_per_video, dtype=int)
    return [files[i] for i in indices]


def build_overlay(image_rgb, pred_mask, alpha=0.45):
    overlay = image_rgb.copy()
    green = np.zeros_like(image_rgb)
    green[..., 1] = 255
    drivable = pred_mask == 1
    overlay[drivable] = (
        (1 - alpha) * image_rgb[drivable] + alpha * green[drivable]
    ).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Kathmandu domain-transfer spot check (no fine-tuning, no GT).")
    parser.add_argument("--frames_root", type=str, required=True,
                         help="Folder containing the v01_final..v07_final subfolders.")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to best.pt (copied locally from the college PC's run folder).")
    parser.add_argument("--config", type=str, required=True,
                         help="Path to that run's config.yaml (copied alongside best.pt).")
    parser.add_argument("--n_per_video", type=int, default=2,
                         help="Frames per video, evenly spaced (default 2).")
    parser.add_argument("--output", type=str, default="kathmandu_panels.png",
                         help="Output grid image path.")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    output_path = Path(args.output)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = (config["data"]["image_size"], config["data"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Collect selected frames across all usable videos ---
    selected = []  # list of (video_id, frame_path)
    for video_id in USABLE_VIDEOS:
        try:
            video_folder = find_video_folder(frames_root, video_id)
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
            continue

        frames = select_frames(video_folder, args.n_per_video)
        for frame_path in frames:
            selected.append((video_id, frame_path))

    if len(selected) == 0:
        raise RuntimeError(f"No frames found under {frames_root} for any of {USABLE_VIDEOS}.")

    print(f"Selected {len(selected)} frames across {len(set(v for v, _ in selected))} videos.")

    # --- Model ---
    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,
        classes=config["model"]["classes"],
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_transforms = get_val_transforms(img_size)

    # --- Build grid: input | predicted mask | overlay, one row per frame ---
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for row_idx, (video_id, frame_path) in enumerate(selected):
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                print(f"  [warn] failed to read {frame_path}, skipping")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb_resized = cv2.resize(image_rgb, (img_size[1], img_size[0]))

            augmented = val_transforms(
                image=image_rgb, mask=np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            )
            input_tensor = augmented["image"].unsqueeze(0).to(device)

            logits = model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_mask_display = (pred_mask * 255).astype(np.uint8)
            overlay = build_overlay(image_rgb_resized, pred_mask)

            axes[row_idx, 0].imshow(image_rgb_resized)
            axes[row_idx, 0].set_title(f"{video_id} / {frame_path.name}\ninput", fontsize=8)
            axes[row_idx, 1].imshow(pred_mask_display, cmap="gray")
            axes[row_idx, 1].set_title("predicted mask", fontsize=9)
            axes[row_idx, 2].imshow(overlay)
            axes[row_idx, 2].set_title("overlay", fontsize=9)

            for col in range(3):
                axes[row_idx, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nKathmandu spot-check grid saved to {output_path}")
    print("Note: no ground truth exists for these frames -- this is a visual "
          "domain-transfer check only, not a scored evaluation.")


if __name__ == "__main__":
    main()
