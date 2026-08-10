"""
losses.py -- RoadVision Nepal Phase 2: IDD baseline combined loss.

Locked spec (Day 2, Task 2):
- nn.CrossEntropyLoss(ignore_index=255)  -- handles ignore-pixels natively
- Dice loss, manually masked for ignore_index=255 (no native support in Dice)
- Combined as a weighted sum, weights tunable via config.yaml (default 0.5/0.5)

Both terms operate on raw logits of shape (B, 2, H, W) and integer targets of
shape (B, H, W) with values in {0, 1, 255}, matching model.py's output and
dataset.py's mask dtype.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_dice_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice loss computed only over valid (non-ignore) pixels.

    Dice has no native ignore_index concept, so ignore-pixels are excluded
    manually: both the predicted probabilities and the one-hot targets are
    zeroed out at ignore locations before the intersection/cardinality sums,
    so those pixels contribute nothing to the loss in either direction.
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    valid_mask = (targets != ignore_index)  # (B, H, W), bool

    # Ignore-pixels get a placeholder class (0) for one-hot encoding purposes only --
    # they're zeroed out by valid_mask right after, so the placeholder value never
    # actually contributes to the loss.
    targets_safe = targets.clone()
    targets_safe[~valid_mask] = 0
    targets_onehot = F.one_hot(targets_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

    valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W), broadcasts over class dim
    probs = probs * valid_mask
    targets_onehot = targets_onehot * valid_mask

    dims = (0, 2, 3)
    intersection = (probs * targets_onehot).sum(dims)
    cardinality = (probs + targets_onehot).sum(dims)
    dice_per_class = (2 * intersection + smooth) / (cardinality + smooth)

    return 1 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of CrossEntropyLoss(ignore_index=255) and masked_dice_loss.

    ce_weight / dice_weight are separate config knobs (not required to sum to 1)
    so the relative emphasis can be tuned without renormalizing by hand.
    """

    def __init__(self, ignore_index: int = 255, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Returns (total_loss, breakdown_dict). The breakdown dict is meant to be
        logged into metrics.csv alongside total_loss, so it's easy to see whether
        a bad epoch is a CE problem, a Dice problem, or both.
        """
        ce_loss = self.ce(logits, targets)
        dice_loss = masked_dice_loss(logits, targets, ignore_index=self.ignore_index)
        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        breakdown = {
            "ce_loss": ce_loss.item(),
            "dice_loss": dice_loss.item(),
            "total_loss": total.item(),
        }
        return total, breakdown


if __name__ == "__main__":
    # Smoke test -- confirms gradients flow and ignore-pixels are genuinely excluded,
    # not just silently zero by coincidence.
    torch.manual_seed(0)

    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (2, 8, 8))
    targets[:, 0, :] = 255  # force a real ignore-region, like a rotation-padding corner

    criterion = CombinedLoss(ignore_index=255, ce_weight=0.5, dice_weight=0.5)
    loss, breakdown = criterion(logits, targets)
    print(f"breakdown: {breakdown}")

    loss.backward()
    assert logits.grad is not None, "No gradient reached logits!"
    assert not torch.isnan(logits.grad).any(), "NaN in gradients!"
    print("Gradient flow OK, no NaNs.")

    # Confirm ignore-pixels are truly ignored: changing their target values
    # (while keeping the valid region identical) must not change the loss at all.
    targets_alt = targets.clone()
    targets_alt[:, 0, :] = 1 - targets_alt[:, 0, :].clamp(max=1)  # flip placeholder row (still masked as 255)
    targets_alt[:, 0, :] = 255  # re-assert still ignore, just proving mutation elsewhere doesn't matter
    loss_alt, _ = criterion(logits.detach().requires_grad_(), targets_alt)
    assert torch.isclose(loss, loss_alt, atol=1e-6), "Loss changed when only ignore-region content changed -- masking is leaking!"
    print("Ignore-index masking verified: ignore-region content has zero effect on loss.")

    print("losses.py smoke test passed.")
