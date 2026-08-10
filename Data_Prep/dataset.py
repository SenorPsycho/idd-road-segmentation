"""
dataset.py

PyTorch Dataset for IDD drivable-area binary segmentation, reading the
baked masks produced by bake_masks.py (gtFine_binary/<split>/<seq>/*.png).

Two augmentation pipelines are provided:
  - get_train_transforms(): flip, mild color jitter, slight rotation.
    Kept deliberately conservative -- this is a baseline, not the place
    to be aggressive with augmentation.
  - get_val_transforms(): resize + normalize only, no augmentation.

Both use albumentations so the same geometric transform (flip, rotation)
is applied identically to image AND mask in one call -- this matters,
since transforming them separately risks misalignment.

Normalization uses ImageNet mean/std since the segmentation backbone is
ResNet50 pretrained on ImageNet.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = 255


class IDDBinarySegDataset(Dataset):
    """
    Expects a dataset root laid out as:
        root/leftImg8bit/<split>/<seq>/<frame>_leftImg8bit.png
        root/gtFine_binary/<split>/<seq>/<frame>_gtFine_binary.png

    Args:
        root: dataset root (str or Path)
        split: "train" or "val"
        img_size: (H, W) to resize both image and mask to
        transforms: an albumentations Compose (see get_train_transforms /
            get_val_transforms below). If None, only resize+normalize+ToTensor
            is applied (no augmentation) -- useful for the sanity test.
    """

    def __init__(self, root, split: str, img_size=(512, 512), transforms=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms

        img_dir = self.root / "leftImg8bit" / split
        mask_dir = self.root / "gtFine_binary" / split

        self.samples = []
        for mask_path in sorted(mask_dir.rglob("*_gtFine_binary.png")):
            seq = mask_path.parent.name
            frame_id = mask_path.name.replace("_gtFine_binary.png", "")

            # IDD ships images in mixed extensions across Part I (.png) and
            # Part II (.jpg) -- try both rather than assuming one.
            img_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = img_dir / seq / f"{frame_id}_leftImg8bit{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is not None:
                self.samples.append((img_path, mask_path))
            else:
                print(f"  [warn] mask found with no matching image, skipping: {mask_path}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image/mask pairs found under {self.root} for split={split}. "
                f"Did you run bake_masks.py for this split yet?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Failed to read pair: {img_path}, {mask_path}")

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            resize = A.Compose([
                A.Resize(*self.img_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
            augmented = resize(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = mask.long()  # CrossEntropyLoss expects long, not float
        return image, mask


def get_train_transforms(img_size=(512, 512)):
    """Conservative augmentation set for the baseline run."""
    return A.Compose([
        A.Resize(*img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.Rotate(limit=7, border_mode=cv2.BORDER_CONSTANT,
                  fill=0, fill_mask=IGNORE_INDEX, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=(512, 512)):
    """No augmentation -- resize + normalize only."""
    return A.Compose([
        A.Resize(*img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
