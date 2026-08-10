# Session 2 — Day 2: Model + Training Script

**Date:** 2026-08-10
**Scope:** RoadVision Nepal Phase 2 (IDD baseline arm) — Day 2 of implementation plan
**Environment:** Local Windows CPU machine (dev, code writing + structural testing only). College RTX 4070 Super PC not yet accessed — access expected 2026-08-11, real training deferred to that session.

---

## Objectives (from Day 2 plan)

1. ResNet50 encoder (pretrained ImageNet) + U-Net decoder, binary segmentation head
2. Loss function, decided based on measured class imbalance rather than assumed
3. Training loop with checkpointing and metrics logging (IoU, precision, recall) per the evidence-file convention (`metrics.csv`, `config.yaml`, `pip_freeze.txt`, commit hash, seed)
4. Smoke-test plan for 1–2 epochs on a small subset, to run on the college PC before committing to a full run

All code-side objectives (1–3) closed by end of session. Objective 4 is prepped and pending college PC access.

---

## 1. Pre-work: class balance check on real baked masks

Before deciding on a loss function, four real `gtFine_binary` masks were spot-checked programmatically (not by eye — at raw values `{0, 1, 255}`, classes 0 and 1 are visually indistinguishable in a standard image viewer, since they differ by 1/255 brightness).

**Result:** drivable-class pixel share ranged **17–32%** across the four samples (non-drivable 68–83%). This is a real, non-trivial imbalance — and in the opposite direction than assumed (dashcam framing was expected to be road-heavy; sky/buildings/urban clutter above the horizon dominate instead in these frames).

**Decision:** this measurement is what justified combining CE with Dice rather than using plain CrossEntropyLoss alone (see Section 3).

---

## 2. Model definition (`model.py`)

**Architecture-vs-loss conflict caught and resolved:** the original framing ("BCE+Dice") was inconsistent with also wanting native `ignore_index=255` support, since `nn.BCEWithLogitsLoss` is 1-channel and has no `ignore_index`, while `nn.CrossEntropyLoss` (which does support `ignore_index` natively) requires a 2-channel softmax output. Resolved by locking the 2-channel route.

**Locked spec:**
- `smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", classes=2)` — 2-channel raw-logit output
- 512×512 input, matching Day 1's loader
- Encoder freeze/unfreeze toggle, epoch-driven (`freeze_until_epoch`), so the pretrained ResNet50 features aren't scrambled by early noisy decoder gradients

**Decision:** use `segmentation_models_pytorch` (SMP) for this deliverable rather than a hand-built decoder, given the timeline. A from-scratch U-Net decoder implementation is deferred as a later, separate portfolio artifact ("built on a standard implementation, then reimplemented to show understanding of skip-connection mechanics").

**Verification:** structural smoke test (random-init weights, since the sandbox used for testing has no route to the pretrained-weight host) confirmed correct output shape `(B, 2, H, W)`, correct freeze/unfreeze parameter-count behavior (~32.5M total params, ~9M trainable when frozen), and correct epoch-driven freeze scheduling.

---

## 3. Combined loss function (`losses.py`)

**Locked spec:**
- `nn.CrossEntropyLoss(ignore_index=255)` — handles ignore-pixels natively
- Dice loss, **manually masked** for `ignore_index=255` — Dice has no native ignore-index support, and combining it with CE does not make it inherit CE's ignore-handling; each term needs the mask applied independently
- Combined as a weighted sum: `ce_weight` / `dice_weight`, both tunable via `config.yaml`, defaulted to 0.5/0.5 given the measured 17–32% imbalance isn't severe enough to justify a more aggressive split yet

**Verification:** gradient flow confirmed (no NaNs). Ignore-index masking was verified directly, not just assumed correct — mutating the *content* of a flagged ignore-region while keeping it flagged as 255 produced an identical loss value, confirming those pixels have genuinely zero effect on the loss rather than a coincidentally small one.

---

## 4. Metrics functions (`metrics.py`)

Reports IoU, precision, and recall for the drivable class, excluding `ignore_index=255` pixels from every computation.

**Decision:** metrics are accumulated as running confusion counts (TP/FP/FN/TN) across an entire epoch and computed once at the end, rather than averaged per-batch. Per-batch averaging is skewed by uneven batch sizes or batches with different drivable-pixel content; accumulating raw counts first gives the mathematically correct epoch-level number.

**Verification:** confirmed against a hand-computed 2×2 test case (with one ignore-pixel mixed in) where TP/FP/FN/TN and the resulting IoU/precision/recall were worked out manually and matched exactly.

---

## 5. Evidence-file scaffolding (`run_utils.py`)

Creates a timestamped `runs/YYYY-MM-DD_HHMM/` folder per run (timestamp-based naming, not manually tagged — zero-effort, always unique) and writes:
- `config.yaml` — full hyperparameter snapshot for that specific run
- `pip_freeze.txt` — exact installed package versions
- `commit_hash.txt` — current git commit, with an explicit dirty-working-tree warning if uncommitted changes are present at run time (a clean hash next to a dirty tree would misrepresent what code actually ran)
- `metrics.csv` — one row per epoch, appended during training

**Verification:** all four files confirmed created correctly in a smoke test; a sample metrics row was written and read back successfully.

---

## 6. Training loop (`train.py`)

Ties together `Data_Prep/dataset.py` (Day 1), `model.py`, `losses.py`, `metrics.py`, and `run_utils.py`.

**Correction made during this session:** `train.py` was initially written against an *assumed* `IDDBinarySegDataset` constructor signature (`root_dir`, `transform`), inferred only from Session 1's prose description since the real file hadn't been shared yet. Once the actual `Data_Prep/dataset.py` was provided, this was found to be wrong on two counts — the real signature is `(root, split, img_size, transforms)` (different parameter names, and `img_size` as an explicit `(H, W)` tuple argument to `get_train_transforms()`/`get_val_transforms()`, not a bare int). `train.py` was corrected to match the real file exactly.

**Checkpoint policy (flagged as a default, not an explicit user decision):** save both `best.pt` (highest val IoU so far) and `last.pt` (most recent epoch) every epoch — cheap in disk space, protects the best model from late-epoch overfitting while still allowing resume from the most recent state.

**Verification:** run end-to-end (real `dataset.py`, synthetic on-disk image/mask pairs matching the real directory layout, actual `train.py` module — not a stand-in) for 2 epochs. Confirmed: correct freeze-schedule transition (frozen at epoch 0, unfrozen at epoch 1 per a test `freeze_until_epoch=1`), correct loss/metric computation each epoch, correct checkpoint writes (`best.pt` and `last.pt`), and correct `metrics.csv` row appends.

---

## 7. Config file (`config.yaml`)

Single source of truth for hyperparameters; `run_utils.py` snapshots a copy of it into every run folder, so a given run's config is preserved even if this template file later changes.

**Locked values:** `image_size: 512`, `classes: 2`, `ignore_index: 255`, `freeze_until_epoch: 3`, `ce_weight`/`dice_weight: 0.5/0.5`.
**Default values (explicitly flagged, not locked):** `batch_size: 8`, `num_epochs: 30` — starting points only, meant to be revised once real loss curves and VRAM headroom are visible on the college PC.

`root_dir` set to `"Dataset"`, reflecting that the dataset folder was flattened during this session (no `IDD_Segmentation` subfolder — `Dataset/leftImg8bit`, `Dataset/gtFine_binary`, etc. sit directly under `Dataset/`).

---

## 8. Repo structure changes (this session)

- New `Model_Train/` folder created as a sibling to `Data_Prep/`, holding `model.py`, `losses.py`, `metrics.py`, `run_utils.py`, `train.py`, `config.yaml`
- `Dataset/kitti_archives_to_download.txt` and `Dataset/wget.exe` removed (leftovers from the dropped KITTI depth-estimation project)
- Duplicate loader file resolved: `test_loader_final.py` removed; `Data_Prep/test_loader.py` confirmed canonical (imports `Data_Prep.dataset`, matching the current package layout)
- `Dataset/IDD_Segmentation/` flattened to `Dataset/` directly
- Confirmed: no `phase2_hybrid_segmentation/` folder — repo is a standalone project, not nested under an older naming convention

---

## 9. Open item flagged, not yet resolved

`requirements.txt` lists `opencv-python==5.0.0.93`, which does not correspond to a real published PyPI release (current `opencv-python` versions are in the 4.x series). Worth checking before relying on this file for the college PC install — possibly a typo or an artifact of how the file was generated.

---

## Status: Day 2 (code-side) complete

| Item | Status |
|---|---|
| Model definition (`model.py`) | ✅ (built, tested, freeze/unfreeze verified) |
| Loss function (`losses.py`) | ✅ (built, ignore-masking verified against real mutation test) |
| Metrics (`metrics.py`) | ✅ (built, hand-verified against manual calculation) |
| Evidence-file scaffolding (`run_utils.py`) | ✅ (built, tested) |
| Training loop (`train.py`) | ✅ (built, corrected against real `dataset.py`, verified end-to-end) |
| Config (`config.yaml`) | ✅ (written, locked values separated from flagged defaults) |
| Smoke-test run | ⏳ pending college PC access (expected 2026-08-11) |

## Open items carried forward

- Smoke-test run on the college RTX 4070 Super PC: 1–2 epochs on a small subset, confirm no crashes, sane loss/IoU values, before committing to the full 30-epoch run.
- Verify `opencv-python==5.0.0.93` in `requirements.txt` before installing on the college PC.
- `batch_size` and `num_epochs` are defaults, not locked — revisit once real VRAM headroom and loss-curve shape are visible.
- `freeze_until_epoch=3` and `ce_weight`/`dice_weight=0.5/0.5` are locked-for-now decisions, explicitly flagged for revisit once real training metrics exist (currently no evidence either way, since no real training has run yet).
- Empirical check on IDD sequence frame-spacing for optical flow viability (relevant to the later temporal arm) — still deferred, carried over from Session 1.
