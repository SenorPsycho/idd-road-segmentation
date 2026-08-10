"""
metrics.py -- RoadVision Nepal Phase 2: IDD baseline evaluation metrics.

Reports IoU, precision, and recall for the drivable (class 1) region, since that's
the class of interest for this binary segmentation task. ignore_index=255 pixels
are excluded from every computation, matching the loss function's handling.

Metrics are accumulated as running confusion counts (TP/FP/FN/TN) across an entire
epoch and computed once at the end -- NOT averaged per-batch -- since per-batch
averaging is skewed by batches of different size or different drivable-pixel content
(e.g. an uneven last batch from drop_last=False).
"""

import torch


def batch_confusion_counts(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255):
    """
    Computes TP/FP/FN/TN for the drivable class (1) on a single batch, excluding
    ignore_index pixels entirely from all four counts.

    logits: (B, 2, H, W) raw model output
    targets: (B, H, W) integer labels in {0, 1, 255}
    """
    preds = torch.argmax(logits, dim=1)  # (B, H, W)

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    return tp, fp, fn, tn


class MetricsAccumulator:
    """
    Accumulates confusion counts across all batches in an epoch, then computes
    IoU / precision / recall once at the end via .compute().

    Usage per epoch:
        acc = MetricsAccumulator()
        for images, masks in loader:
            logits = model(images)
            acc.update(logits, masks)
        results = acc.compute()   # {'iou':.., 'precision':.., 'recall':..}
        acc.reset()               # before the next epoch
    """

    def __init__(self, ignore_index: int = 255, eps: float = 1e-6):
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        tp, fp, fn, tn = batch_confusion_counts(logits, targets, self.ignore_index)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute(self) -> dict:
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


if __name__ == "__main__":
    # Smoke test 1: hand-constructed case with a known, hand-computed answer.
    # 2x2 image, single batch:
    #   targets: [[1, 1], [0, 255]]   -- one ignore-pixel
    #   preds (via logits argmax):    [[1, 0], [0, ?]]  -- ignored pixel's pred doesn't matter
    # Valid pixels only: target=1,pred=1 (TP) | target=1,pred=0 (FN) | target=0,pred=0 (TN)
    # Expected: TP=1, FP=0, FN=1, TN=1 -> IoU = 1/(1+0+1) = 0.5, precision = 1/1 = 1.0, recall = 1/2 = 0.5
    logits = torch.zeros(1, 2, 2, 2)
    logits[0, 1, 0, 0] = 10.0  # pred=1 at (0,0)
    logits[0, 0, 0, 1] = 10.0  # pred=0 at (0,1)
    logits[0, 0, 1, 0] = 10.0  # pred=0 at (1,0)
    logits[0, 0, 1, 1] = 10.0  # pred=0 at (1,1) -- irrelevant, this pixel is ignored

    targets = torch.tensor([[[1, 1], [0, 255]]])

    acc = MetricsAccumulator()
    acc.update(logits, targets)
    result = acc.compute()
    print(f"Hand-verified case: {result}")

    assert result["tp"] == 1 and result["fp"] == 0 and result["fn"] == 1 and result["tn"] == 1
    assert abs(result["iou"] - 0.5) < 1e-4
    assert abs(result["precision"] - 1.0) < 1e-4
    assert abs(result["recall"] - 0.5) < 1e-4
    print("Hand-verified case PASSED.")

    # Smoke test 2: confirm epoch-level accumulation (two batches) differs correctly
    # from what you'd get by naively averaging two single-batch ratios.
    acc.reset()
    # batch A: 1 TP, 0 FP, 0 FN -> IoU = 1.0
    logits_a = torch.zeros(1, 2, 1, 1)
    logits_a[0, 1, 0, 0] = 10.0
    targets_a = torch.tensor([[[1]]])
    acc.update(logits_a, targets_a)

    # batch B: 0 TP, 0 FP, 1 FN -> IoU = 0.0
    logits_b = torch.zeros(1, 2, 1, 1)
    logits_b[0, 0, 0, 0] = 10.0
    targets_b = torch.tensor([[[1]]])
    acc.update(logits_b, targets_b)

    result2 = acc.compute()
    naive_average = (1.0 + 0.0) / 2  # what per-batch averaging would give: 0.5
    print(f"\nAccumulated (correct) IoU: {result2['iou']:.4f}")
    print(f"Naive per-batch average would give: {naive_average:.4f}")
    # Correct accumulated answer: TP=1, FP=0, FN=1 -> IoU = 1/2 = 0.5 (coincidentally equal here,
    # but the counts (tp=1, fp=0, fn=1) are the real quantities -- this matters more with uneven batch sizes)
    assert result2["tp"] == 1 and result2["fn"] == 1

    print("metrics.py smoke test passed.")
