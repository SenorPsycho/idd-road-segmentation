"""
kathmandu_full_inference.py -- RoadVision Nepal Phase 2: full-dataset Kathmandu spot check.

Runs the trained IDD checkpoint (no fine-tuning) on EVERY frame across all
usable Kathmandu videos (v01-v05, v07 -- v06 excluded per fisheye/telemetry),
and saves a 3-panel image (input | predicted mask | overlay) for each frame,
right next to the original frame file.

No ground truth exists for these frames, so this produces visuals only, not
scores -- same caveat as kathmandu_inference.py.

RESUMABLE: if a panel file already exists for a frame, that frame is skipped.
This matters because a full pass over 2,000+ frames on CPU will take a while
-- if the run gets interrupted, just rerun the same command and it picks up
where it left off instead of redoing completed frames.

Usage:
    python -m Model_Train.kathmandu_full_inference \\
        --frames_root "extracted_frames" \\
        --checkpoint "runs/2026-08-11_1628/best.pt" \\
        --config "runs/2026-08-11_1628/config.yaml"

Optional: --limit N to do a quick test run on only the first N frames per
video before committing to the full set.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")  # no display needed, just saving files -- avoids GUI backend overhead
import matplotlib.pyplot as plt

from Data_Prep.dataset import get_val_transforms
from Model_Train.model import build_model

USABLE_VIDEOS = ["v01", "v02", "v03", "v04", "v05", "v07"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
PANEL_SUFFIX = "_panel.png"


def find_video_folder(frames_root: Path, video_id: str) -> Path:
    exact = frames_root / f"{video_id}_final"
    if exact.exists():
        return exact
    candidates = [p for p in frames_root.iterdir() if p.is_dir() and p.name.startswith(video_id)]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not find a folder for {video_id} under {frames_root}")


def build_overlay(image_rgb, pred_mask, alpha=0.45):
    overlay = image_rgb.copy()
    green = np.zeros_like(image_rgb)
    green[..., 1] = 255
    drivable = pred_mask == 1
    overlay[drivable] = (
        (1 - alpha) * image_rgb[drivable] + alpha * green[drivable]
    ).astype(np.uint8)
    return overlay


def save_panel(image_rgb_resized, pred_mask, frame_path: Path, video_id: str):
    pred_mask_display = (pred_mask * 255).astype(np.uint8)
    overlay = build_overlay(image_rgb_resized, pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image_rgb_resized)
    axes[0].set_title(f"{video_id} / {frame_path.name}\ninput", fontsize=8)
    axes[1].imshow(pred_mask_display, cmap="gray")
    axes[1].set_title("predicted mask", fontsize=9)
    axes[2].imshow(overlay)
    axes[2].set_title("overlay", fontsize=9)
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    panel_path = frame_path.parent / f"{frame_path.stem}{PANEL_SUFFIX}"
    plt.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close(fig)  # important -- without this, memory climbs across thousands of frames
    return panel_path


def main():
    parser = argparse.ArgumentParser(description="Full-dataset Kathmandu panel generation (no GT, no fine-tuning).")
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only process the first N frames per video, for a quick test run.")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = (config["data"]["image_size"], config["data"]["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Running on CPU across the full frame set -- this will take a while. "
              "Safe to interrupt (Ctrl+C) and rerun later; completed frames are skipped.")

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

    total_processed = 0
    total_skipped = 0
    start_time = time.time()

    with torch.no_grad():
        for video_id in USABLE_VIDEOS:
            try:
                video_folder = find_video_folder(frames_root, video_id)
            except FileNotFoundError as e:
                print(f"  [warn] {e}")
                continue

            frame_files = sorted(
                p for p in video_folder.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if args.limit is not None:
                frame_files = frame_files[: args.limit]

            print(f"\n{video_id}: {len(frame_files)} frames")

            for frame_path in frame_files:
                panel_path = frame_path.parent / f"{frame_path.stem}{PANEL_SUFFIX}"
                if panel_path.exists():
                    total_skipped += 1
                    continue

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

                save_panel(image_rgb_resized, pred_mask, frame_path, video_id)

                total_processed += 1
                if total_processed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed
                    print(f"  processed {total_processed} frames "
                          f"({rate:.2f} frames/sec, {elapsed/60:.1f} min elapsed)")

    elapsed = time.time() - start_time
    print(f"\nDone. {total_processed} panels generated, {total_skipped} already existed and were skipped.")
    print(f"Total time: {elapsed/60:.1f} minutes.")
    print("Panels saved as <frame_name>_panel.png next to each original frame.")


if __name__ == "__main__":
    main()
