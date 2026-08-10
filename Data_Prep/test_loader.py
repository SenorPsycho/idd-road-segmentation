"""
test_loader.py

Day 1 sanity check -- NOT real training. Just confirms:
  - the Dataset finds image/mask pairs
  - shapes come out right after augmentation
  - a DataLoader batch collates without crashing
  - mask values are within the expected {0, 1, 255} set after augmentation
    (catches interpolation bugs -- e.g. if a mask got resized with the wrong
    interpolation mode, you'd see values like 2, 3, 4... appear from blending)

Usage:
    python test_loader.py --root /path/to/idd_root --split train
"""

import argparse

import torch
from torch.utils.data import DataLoader

from Data_Prep.dataset import IDDBinarySegDataset, get_train_transforms, get_val_transforms


def run_check(root, split, n_batches=2, batch_size=4, img_size=(512, 512)):
    transforms = get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size)
    ds = IDDBinarySegDataset(root, split=split, img_size=img_size, transforms=transforms)
    print(f"[{split}] dataset size: {len(ds)}")

    # CPU, small batch -- this is a sanity check, not a training run.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for i, (images, masks) in enumerate(loader):
        print(f"batch {i}: images {tuple(images.shape)} {images.dtype}, "
              f"masks {tuple(masks.shape)} {masks.dtype}")

        assert images.ndim == 4, "expected (B, C, H, W)"
        assert images.shape[1] == 3, "expected 3 channels"
        assert masks.shape == (images.shape[0], images.shape[2], images.shape[3]), \
            "mask shape must match image H,W with no channel dim"

        unique_vals = torch.unique(masks)
        allowed = {0, 1, 255}
        bad_vals = set(unique_vals.tolist()) - allowed
        assert not bad_vals, f"unexpected mask values {bad_vals} -- check augmentation interpolation"

        print(f"  mask unique values: {sorted(unique_vals.tolist())}")
        print(f"  image value range: [{images.min():.3f}, {images.max():.3f}]  (normalized, so negative is expected)")

        if i + 1 >= n_batches:
            break

    print(f"[{split}] OK -- loader, augmentation, and collation all passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    args = parser.parse_args()

    run_check(args.root, args.split)
