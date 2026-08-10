"""
train.py -- RoadVision Nepal Phase 2: IDD baseline training loop.

Ties together:
  Data_Prep/dataset.py  -- IDDBinarySegDataset + transform pipelines (Day 1)
  model.py                -- ResNet50+U-Net, freeze/unfreeze schedule (Task 1)
  losses.py                -- CombinedLoss (CE + masked Dice) (Task 2)
  metrics.py                -- IoU / precision / recall, ignore-aware (Task 3)
  run_utils.py                -- evidence-file scaffolding (Task 4)

Dataset root: Data_Prep/dataset.py's IDDBinarySegDataset expects
    root/leftImg8bit/<split>/<seq>/*_leftImg8bit.{png,jpg,jpeg}
    root/gtFine_binary/<split>/<seq>/*_gtFine_binary.png
`root` here is the repo's Dataset/ folder directly (post-flattening -- no
IDD_Segmentation subfolder), set via config["data"]["root_dir"].
"""

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Data_Prep.dataset import IDDBinarySegDataset, get_train_transforms, get_val_transforms  # noqa: E402

from model import build_model, set_encoder_frozen  # noqa: E402
from losses import CombinedLoss  # noqa: E402
from metrics import MetricsAccumulator  # noqa: E402
from run_utils import init_run, append_metrics_row  # noqa: E402


def build_dataloaders(config: dict):
    data_cfg = config["data"]
    img_size = (data_cfg["image_size"], data_cfg["image_size"])

    train_ds = IDDBinarySegDataset(
        root=data_cfg["root_dir"],
        split="train",
        img_size=img_size,
        transforms=get_train_transforms(img_size),
    )
    val_ds = IDDBinarySegDataset(
        root=data_cfg["root_dir"],
        split="val",
        img_size=img_size,
        transforms=get_val_transforms(img_size),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["train"]["batch_size"], shuffle=True,
        num_workers=config["train"].get("num_workers", 2), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["train"]["batch_size"], shuffle=False,
        num_workers=config["train"].get("num_workers", 2), pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    One pass over `loader`. If optimizer is given, runs in training mode with
    backprop; otherwise runs in eval mode under torch.no_grad(). Returns
    (avg_loss, avg_ce, avg_dice, metrics_dict).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_ce, total_dice, n_batches = 0.0, 0.0, 0.0, 0
    metrics_acc = MetricsAccumulator(ignore_index=criterion.ignore_index)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss, breakdown = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += breakdown["total_loss"]
            total_ce += breakdown["ce_loss"]
            total_dice += breakdown["dice_loss"]
            n_batches += 1
            metrics_acc.update(logits.detach(), masks)

    avg_loss = total_loss / n_batches
    avg_ce = total_ce / n_batches
    avg_dice = total_dice / n_batches
    return avg_loss, avg_ce, avg_dice, metrics_acc.compute()


def train(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(config)

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

    run_paths = init_run(config, base_dir=config["run"]["base_dir"])
    run_dir = run_paths["run_dir"]
    print(f"Run artifacts -> {run_dir}")

    best_iou = -1.0
    num_epochs = config["train"]["num_epochs"]
    freeze_until = config["train"]["freeze_until_epoch"]

    for epoch in range(num_epochs):
        encoder_frozen = set_encoder_frozen(model, epoch, freeze_until)

        train_loss, train_ce, train_dice, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_ce, val_dice, val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(f"[epoch {epoch}] frozen={encoder_frozen} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"iou={val_metrics['iou']:.4f} precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f}")

        append_metrics_row(run_paths["metrics_path"], {
            "epoch": epoch,
            "train_loss": train_loss, "train_ce": train_ce, "train_dice": train_dice,
            "val_loss": val_loss, "val_ce": val_ce, "val_dice": val_dice,
            "iou": val_metrics["iou"], "precision": val_metrics["precision"], "recall": val_metrics["recall"],
            "encoder_frozen": encoder_frozen,
        })

        # Checkpoint policy (default, flagged for review): save BOTH best-val-IoU
        # and last-epoch every epoch. Cheap in disk space, protects against losing
        # the best model to late-epoch overfitting while still letting you resume
        # from the most recent state.
        torch.save(model.state_dict(), run_dir / "last.pt")
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"  -> new best IoU ({best_iou:.4f}), saved best.pt")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}. Artifacts in {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
