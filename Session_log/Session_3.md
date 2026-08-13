# Session 3 — Day 3-4: Full Training Run (College PC)

**Date:** 2026-08-11 – 2026-08-12
**Scope:** RoadVision Nepal Phase 2 (IDD baseline arm) — Day 3-4 of implementation plan
**Environment:** College PC (RTX 4070 SUPER, i9-14900K, 32GB RAM), CUDA training with AMP

---

## Objectives (from Day 3-4 plan)

1. Full training run, however many epochs time/compute budget allows
2. Monitor for overfitting, adjust LR/augmentation if needed
3. Save best checkpoint by validation IoU

All three closed by end of session, after one failed attempt and a mid-session fix.

---

## 1. Pre-run code review

Before launching, `train.py`, `config.yaml`, and `losses.py` were reviewed for issues that could waste college-PC compute time if caught only after the fact:

- **Loss accumulation check:** confirmed `CombinedLoss.forward()` in `losses.py` already calls `.item()` on all three `breakdown` dict values (`ce_loss`, `dice_loss`, `total_loss`) before returning, and that `train.py` backpropagates through the separate raw-tensor `total` return value, not the dict. No memory-leak risk from graph-retention across the epoch loop.
- **Dice loss batch-level averaging noted (not a bug):** `masked_dice_loss` sums intersection/cardinality across `dims=(0, 2, 3)` — i.e. batch-level Dice, not per-image-then-averaged. Given drivable-pixel share varies 17–32% image-to-image (Session 2 finding), this means Dice loss for a batch is weighted toward images with more drivable pixels. Flagged as a known, common, defensible design choice — not corrected.
- **No LR schedule / no differential LR at the epoch-3 unfreeze point:** flat `lr: 0.0001` before and after encoder unfreeze. Decision: watch epoch 3 in the logs for a loss spike rather than pre-emptively adding schedule complexity.
- **No optimizer/epoch state in checkpoints:** `best.pt`/`last.pt` save `model.state_dict()` only, no resume capability. Decision (explicit): not needed — single uninterrupted college-PC session assumed.
- **`opencv-python==5.0.0.93` in `requirements.txt`** (flagged as an open item since Session 2, not a real PyPI release): confirmed fixed before this session's install.

---

## 2. First attempt — OOM crash at epoch 3

**Run folder:** `runs/2026-08-11_1615/`
**Config at launch:** `batch_size: 48` (per Day 2's `config.yaml`, validated only under frozen-encoder smoke-test conditions)

```
Using device: cuda
GPU: NVIDIA GeForce RTX 4070 SUPER
AMP: enabled
[epoch 0] frozen=True train_loss=0.1150 val_loss=0.0544 iou=0.9412 precision=0.9683 recall=0.9711
  -> new best IoU (0.9412), saved best.pt
[epoch 1] frozen=True train_loss=0.0472 val_loss=0.0432 iou=0.9437 precision=0.9612 recall=0.9811
  -> new best IoU (0.9437), saved best.pt
[epoch 2] frozen=True train_loss=0.0388 val_loss=0.0354 iou=0.9515 precision=0.9741 recall=0.9762
  -> new best IoU (0.9515), saved best.pt
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB. ...
```

**Root cause:** epochs 0–2 ran with the encoder frozen, so backprop only needed gradients through the decoder. At epoch 3 the encoder unfreezes, and backward-pass memory requirements jump substantially since gradients now flow through the full ResNet50 as well. `batch_size: 48` had only ever been validated under frozen-encoder conditions (Day 2's smoke test), not past the unfreeze boundary — so the crash surfaced a real gap in what the smoke test had actually covered, not a config typo.

**Outcome preserved:** `best.pt` from epoch 2 (val IoU 0.9515) was saved and intact in `runs/2026-08-11_1615/` despite the crash. No checkpoint-resume capability, so the run could not continue from epoch 3 — a fresh restart from epoch 0 was required.

**Fix applied:** `batch_size` lowered in `config.yaml` (from 48 to a value validated to survive the post-unfreeze memory footprint). No other config changes made.

---

## 3. Git push failure — large checkpoint files

While preparing to commit progress, `git push` was rejected by GitHub:

```
remote: error: File runs/2026-08-11_1628/best.pt is 124.37 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: File runs/2026-08-11_1628/last.pt is 124.37 MB; this exceeds GitHub's file size limit of 100.00 MB
```

**Fix, step 1 — `.gitignore` updated** to exclude checkpoint weights going forward while keeping the evidence-file trail (`config.yaml`, `metrics.csv`, `pip_freeze.txt`, `commit_hash.txt`) tracked per run:

```
# Model checkpoints (large binaries, not needed in git history)
runs/*/*.pt
```

**Fix, step 2 — files already committed.** The `.gitignore` rule only stops future commits; the `.pt` files were already staged and committed in an earlier commit, so `git rm --cached` + `git commit --amend` were insufficient (files were baked into more than one commit in history, not just the tip).

**Fix, step 3 — full history rewrite via `git filter-repo`:**

```
git filter-repo --path runs/2026-08-11_1628/best.pt --path runs/2026-08-11_1628/last.pt --invert-paths --force
```

This rewrote all history to remove both files from every commit. `filter-repo` removed the `origin` remote as a safety measure (standard behavior, not an error). Remote was re-added and history force-pushed:

```
git remote add origin https://github.com/SenorPsycho/idd-road-segmentation.git
git push --force -u origin main
```

**Result:** confirmed clean — repo pushes successfully, `.gitignore` rule holds for future runs, evidence files (`config.yaml`, `metrics.csv`, `pip_freeze.txt`, `commit_hash.txt`) remain tracked in git history; checkpoint binaries excluded going forward.

---

## 4. Successful full run — 30 epochs

**Run folder:** `runs/2026-08-11_1628/` (same folder as the crashed attempt — `config.yaml`/`commit_hash.txt` retain the original 4:28 PM start timestamp from `init_run()`'s first call, since the rerun reused rather than regenerated the folder; the crashed run's partial `metrics.csv`/checkpoints were overwritten by this run's output)

**Wall-clock:** started 2026-08-11 4:28 PM, finished 2026-08-12 7:04 AM — **~14h 36m total, ~29 min/epoch average**. Substantially slower than the crashed run's ~3:18/epoch at `batch_size: 48`, expected given the lower batch size requires more iterations per epoch over the same 14,027 training images.

**No crash.** Epoch-3 unfreeze transition was smooth — no loss spike in `train_loss`/`val_loss` at the frozen→unfrozen boundary.

**Best checkpoint:** val IoU **0.9649 at epoch 29** (the final epoch) — a genuine running best, not an early peak; IoU trended upward across the full run rather than plateauing early and coasting.

### Metrics summary (selected epochs)

| Epoch | train_loss | val_loss | val IoU | encoder_frozen |
|---|---|---|---|---|
| 0 | 0.0848 | 0.0427 | 0.9449 | True |
| 2 | 0.0351 | 0.0321 | 0.9543 | True |
| 3 | 0.0329 | 0.0296 | 0.9574 | False (unfreeze, no spike) |
| 8 | 0.0238 | 0.0266 | 0.9618 | False |
| 19 | 0.0178 | 0.0257 | 0.9639 | False |
| 22 | 0.0164 | 0.0297 | 0.9590 | False (local dip) |
| 29 | 0.0150 | 0.0269 | **0.9649** | False |

Full 30-row `metrics.csv` preserved in `runs/2026-08-11_1628/`.

---

## 5. Overfitting analysis

**Objective 2 ("monitor for overfitting") produced a real, mild-but-genuine finding, not a clean pass:**

- `train_loss` decreases steadily and monotonically the entire run — 0.0848 → 0.0150 by epoch 29, never plateauing.
- `val_loss` decreases until roughly epoch 8–9 (~0.0268), then **flattens and wobbles** in the ~0.026–0.030 range for the remaining ~20 epochs, with occasional upticks (epoch 22: 0.0297, epoch 24: 0.0298) before settling again.
- `val IoU` similarly plateaus in the 0.960–0.965 band from ~epoch 8 onward, bouncing rather than climbing smoothly (e.g. epoch 11 dips to 0.9596, epoch 22 dips to 0.9590).

This is the classic train/val divergence signature of mild overfitting: the model continues fitting the training set while validation performance stalls. Not severe — validation never meaningfully degrades — but real, and worth stating explicitly in the writeup rather than only reporting the final IoU number.

**No intervention applied this session** (no LR drop, no added augmentation) — the run was allowed to complete at the planned 30 epochs, per objective 3 ("save best checkpoint"), with the plateau documented as a finding for the writeup rather than chased mid-run.

---

## 6. Broader finding: fast saturation

Val IoU reached 0.94–0.95 by epoch 0–2 with the encoder still **frozen** (confirmed in both the crashed and successful runs). The full 30-epoch run, including 27 epochs of unfrozen full-model training, only improved IoU from ~0.9543 (epoch 2) to 0.9649 (epoch 29) — roughly a 0.01 gain, most of it front-loaded in the first ~8 epochs post-unfreeze.

**Flagged for the writeup:** this task saturates quickly on IDD's binary drivable-area collapse, and the marginal value of extended full-encoder training is small relative to the frozen-encoder baseline. This is a legitimate, defensible result — not a failure of the training setup — but needs to be named explicitly, since reviewers are likely to ask why validation IoU is already ~0.94–0.95 with the encoder frozen, and why 27 additional epochs of full training only add ~0.01 IoU.

---

## Status: Day 3-4 complete

| Item | Status |
|---|---|
| Full training run | ✅ 30 epochs, completed after one OOM-crash-and-fix cycle |
| Monitor for overfitting | ✅ mild train/val divergence identified from ~epoch 8 onward, documented |
| Save best checkpoint by val IoU | ✅ `best.pt`, IoU 0.9649 at epoch 29 |
| Evidence files (`config.yaml`, `metrics.csv`, `pip_freeze.txt`, `commit_hash.txt`) | ✅ confirmed present in `runs/2026-08-11_1628/` |
| Console tail | Not preserved (cleared after run) — not blocking, `metrics.csv` covers the substantive record |
| Git repo pushable | ✅ history rewritten via `git filter-repo`, large checkpoint binaries excluded going forward |

## Open items carried forward

- Writeup must explicitly address: (a) why val IoU is already 0.94–0.95 with the encoder frozen, and (b) the mild overfitting plateau from ~epoch 8 onward — both are real findings, not gaps to paper over.
- `runs/2026-08-11_1615/` (first successful-but-incomplete attempt, epochs 0–2, IoU 0.9515) remains as a secondary evidence folder alongside `runs/2026-08-11_1628/` (final 30-epoch result) — worth deciding whether to keep both in the final submission or note the first as superseded.
- No LR schedule or differential encoder/decoder LR was applied at the unfreeze point this run; not needed this time (no spike observed), but remains an option if a future run shows different behavior.
- Empirical check on IDD sequence frame-spacing for optical flow viability (relevant to the later temporal arm) — still deferred, carried over from Sessions 1–2.
