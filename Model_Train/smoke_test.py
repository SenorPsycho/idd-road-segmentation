"""
smoke_test.py -- quick pipeline check, NOT a training run.

Runs the real model/loss/dataloader stack from train.py, but only for a
handful of batches -- just confirms: data loads, model forward pass works,
loss computes, backward pass works, optimizer step works, all on the
college PC's actual GPU. No checkpoints written, no metrics logged, no
full epoch.

Usage (from Model_Train/):
    python smoke_test.py --config config.yaml --n_batches 3
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Data_Prep.dataset import IDDBinarySegDataset, get_train_transforms  # noqa: E402
from model import build_model  # noqa: E402
from losses import CombinedLoss  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def smoke_test(config_path: str, n_batches: int = 3):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    data_cfg = config["data"]
    img_size = (data_cfg["image_size"], data_cfg["image_size"])
    dataset_root = REPO_ROOT / data_cfg["root_dir"]

    print(f"Loading a few batches from {dataset_root} ...")
    train_ds = IDDBinarySegDataset(
        root=dataset_root, split="train", img_size=img_size,
        transforms=get_train_transforms(img_size),
    )
    # num_workers=0 deliberately -- this is a smoke test, not a perf test,
    # and 0 rules out Windows multiprocessing DataLoader issues as a variable.
    loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"],
                         shuffle=True, num_workers=0)

    print("Building model...")
    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        classes=config["model"]["classes"],
    ).to(device)

    criterion = CombinedLoss(
        ignore_index=config["loss"]["ignore_index"],
        ce_weight=config["loss"]["ce_weight"],
        dice_weight=config["loss"]["dice_weight"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    print(f"\nRunning {n_batches} batches (forward + loss + backward + optimizer step)...\n")
    model.train()
    for i, (images, masks) in enumerate(loader):
        if i >= n_batches:
            break

        t0 = time.time()
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss, breakdown = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0
        print(f"batch {i}: images {tuple(images.shape)}  "
              f"loss={breakdown['total_loss']:.4f}  "
              f"ce={breakdown['ce_loss']:.4f}  dice={breakdown['dice_loss']:.4f}  "
              f"({elapsed:.2f}s)")

    print(f"\nSMOKE TEST OK -- {n_batches} batches ran end-to-end with no errors.")
    print("(This did not check val loop, checkpointing, or run_utils artifacts -- "
          "those still need a real run.py execution to verify.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--n_batches", type=int, default=3)
    args = parser.parse_args()
    smoke_test(args.config, args.n_batches)