# Session 1 — Day 1: Setup & Data Audit

**Date:** 2026-08-09
**Scope:** RoadVision Nepal Phase 2 (IDD baseline arm) — Day 1 of implementation plan
**Environment:** Local Windows CPU machine (dev), `.env` virtual environment, torch CUDA available but not required for this session

---

## Objectives (from Day 1 plan)

1. Download/verify IDD Segmentation (Parts I+II) structure
2. Resolve open question: are IDD frames sequential video or independent stills
3. Write the PyTorch `Dataset` loader with class-collapse logic
4. Write the augmentation pipeline
5. Test the loader end-to-end on real data

All five closed by end of session.

---

## 1. Dataset structure verification

Dataset root: `D:\Dataset\OneDrive\Documents\Projects\Github Projects\Road_Segmentation\Dataset\IDD_Segmentation`

Confirmed via direct download inspection: what looked like two separate downloads (`idd-20k-II.tar` etc.) are actually **two sequential parts of one release** — Part II continues where Part I stops, not an overlapping or duplicate set. Combined:

- `leftImg8bit/train`: 14,027 images
- `leftImg8bit/val`: 2,036 images
- `gtFine/train`, `gtFine/val`: JSON polygon annotation files only (no pre-baked PNG masks) — `gtFine/test` does not exist, consistent with IDD never releasing test-split ground truth

**Key finding:** masks are **raw polygon JSON** (`<frame>_gtFine_polygons.json`), not ready-to-use label PNGs. This ruled out writing the `Dataset` class directly against PNGs and required a preprocessing/baking step first (see Section 3).

**Image file extension inconsistency (found later, Section 5):** Part I images are `.png`, Part II images are `.jpg`. Frame IDs and JSON annotation filenames are consistent across both parts — only the image extension differs. This was not visible from directory-level inspection and only surfaced once the full loader was run against real data.

---

## 2. Sequential vs. independent stills

Resolved via IDD documentation/dataset card research: every image carries sequence metadata and is drawn from one of 182 continuous drive sequences (front-facing dashcam footage from Bangalore/Hyderabad), not scattered independent captures.

**Implication for later phases:** the temporal/optical-flow arm of the Phase 2 ablation is viable to pretrain on IDD, since frames within a sequence folder are temporally adjacent. Whether consecutive-frame spacing is dense enough for meaningful flow estimation is still an open empirical question, deferred — not relevant to this week's baseline-only scope.

---

## 3. Class-collapse decision (logged)

Source: official AutoNUE `labels.py` (IDD's authors' own label definition table), which provides a 4-level class hierarchy (`level1Id` through `level4Id`) per class, from 7 coarse categories down to 30–41 fine-grained ones.

**Decision:** collapse to binary drivable-area segmentation using `level1Id`:

| `level1Id` | Classes | Collapsed to |
|---|---|---|
| `0` | road, parking, drivable fallback | **1 (drivable)** |
| `1`–`6` | sidewalk, rail track, people, vehicles, barriers, structures, construction, vegetation, sky, etc. | **0 (non-drivable)** |
| `255` | unlabeled, ego vehicle, rectification border, out of roi, license plate (all flagged `ignoreInEval=True` by the dataset authors) | **255 (ignore)** |

**Rationale:** this is not an arbitrary boundary — the dataset authors already separate "surfaces a vehicle can drive on" from "surfaces it cannot" at `level1Id` 0 vs. 1. The collapse further folds everything else (people, vehicles, sky, buildings) into the negative class, since for a binary drivable-area mask they're equally "don't drive here." Void/junk classes are mapped to `ignore_index=255` (used with `nn.CrossEntropyLoss(ignore_index=255)`) rather than folded into a real class, since e.g. `ego vehicle` is a sensor artifact (own car's hood in frame), not a meaningful drivable/non-drivable signal.

**Rejected alternative:** folding void classes into non-drivable (class 0). Rejected because it would pollute the negative class with pixels that carry no real signal, and the dataset's own `ignoreInEval` flag is an explicit signal to exclude them from loss computation rather than force a label.

---

## 4. Pipeline architecture decision: Option A vs Option B

Two ways to get from raw JSON polygons to trainable tensors were considered:

- **Option A:** rasterize polygons to a mask on-the-fly inside `Dataset.__getitem__`. No separate preprocessing artifact, but repeats polygon rasterization every epoch (slower).
- **Option B (chosen):** one-time preprocessing script bakes all JSON → binary PNG masks upfront; `Dataset` just reads PNGs like a standard segmentation dataset.

**Decision: Option B.** Faster training loop, and the AutoNUE community convention (`createLabels.py`) already does something structurally similar — precompute once, train many times. Note: the official `createLabels.py` was evaluated and **rejected as insufficient on its own** — it bakes multi-class label PNGs (level3/level4, 26–41 classes), not the binary masks needed here. A separate, purpose-built script (`bake_masks.py`) was written instead, doing the polygon→binary collapse directly in one step rather than baking multi-class first and collapsing after.

---

## 5. Files produced

### `Data_Prep/bake_masks.py`
One-time preprocessing script. Reads `gtFine/<split>/<seq>/<frame>_gtFine_polygons.json`, rasterizes each polygon (in JSON listing order, preserving occlusion/z-ordering) onto a canvas **initialized to 255 (ignore)** — not 0 — so any pixel not covered by an annotated polygon stays "unknown" rather than silently becoming a false non-drivable label. Applies the `level1Id` LUT from Section 3. Handles `deleted` objects (skipped) and `*group`-suffixed instance labels (mapped back to their base class). Writes to `gtFine_binary/<split>/<seq>/<frame>_gtFine_binary.png`.

**Validated on synthetic data** before running on real data: confirmed correct fill values at non-overlapping coordinates, confirmed deleted polygons excluded, confirmed unrecognized label names are flagged via `[warn]` rather than silently dropped.

**Run on real data:** `train` — 14,027 written, 0 failed. `val` — 2,036 written, 0 failed. Matches expected JSON file counts exactly.

### `Data_Prep/dataset.py`
`IDDBinarySegDataset` (PyTorch `Dataset`): pairs `leftImg8bit` images with `gtFine_binary` masks by sequence + frame ID, resizes, normalizes (ImageNet mean/std, since backbone is ResNet50 pretrained on ImageNet), returns `(image, mask)` tensors with mask as `long` dtype for `CrossEntropyLoss` compatibility.

`get_train_transforms()` / `get_val_transforms()`: albumentations pipelines. Train: horizontal flip (p=0.5), mild color jitter (p=0.5), slight rotation (±7°, p=0.3) — kept conservative per baseline scope. Val: resize + normalize only, no augmentation. Both use a single albumentations `Compose` call per sample so geometric transforms (flip, rotation) apply identically to image and mask — avoids misalignment from transforming them separately.

### `test_loader.py`
Sanity-check script (not training). Loads a few batches through `DataLoader`, asserts tensor shapes, checks mask values stay within `{0, 1, 255}` (catches interpolation bugs where resizing/rotating with the wrong mode would introduce blended intermediate values).

---

## 6. Bugs found and fixed during real-data testing

Two bugs surfaced only once the loader was run against the full real dataset — neither was visible from directory-level inspection or synthetic testing:

### Bug 1: Image extension mismatch (Part I `.png` vs Part II `.jpg`)
First real run of `test_loader.py` returned `[train] dataset size: 6993` — exactly the official Part I count, meaning **every Part II sequence's masks were silently skipped** for "no matching image" (roughly half the dataset lost). Root cause: `dataset.py` hardcoded `.png` when looking up the matching image file for a given mask. Confirmed via directory listing that Part II ships images as `.jpg` while frame IDs and JSON filenames are otherwise consistent between parts.

**Fix:** loader now tries `.png`, `.jpg`, `.jpeg` in order when resolving the image path for a given frame ID, rather than assuming one extension.

**Result after fix:** `[train] dataset size: 14027` — full dataset paired correctly.

### Bug 2: Albumentations `Rotate` argument rename silently no-op'd the ignore-fill
`test_loader.py` output included a `UserWarning: Argument(s) 'value, mask_value' are not valid for transform Rotate`. This wasn't cosmetic — the installed albumentations version renamed these to `fill`/`fill_mask`, meaning `mask_value=255` (intended to fill rotation-induced blank corners with the ignore label) **silently did not apply**, and those corners were instead getting the library's default fill (0 = non-drivable). Left unfixed, this would have quietly taught the model that rotation-artifact corners are real "non-drivable" signal rather than unknown — a mislabeling bug that would only show up as unexplained noise in later training metrics, not as a crash.

**Fix:** updated to `fill=0, fill_mask=IGNORE_INDEX` (current albumentations API). Re-run confirmed no warning, correct behavior.

---

## 7. Final verification (both splits, real data)

```
[train] dataset size: 14027
batch 0: images (4, 3, 512, 512) torch.float32, masks (4, 512, 512) torch.int64
  mask unique values: [0, 1, 255]
[train] OK -- loader, augmentation, and collation all passed.

[val] dataset size: 2036
batch 0: images (4, 3, 512, 512) torch.float32, masks (4, 512, 512) torch.int64
  mask unique values: [0, 1, 255]
[val] OK -- loader, augmentation, and collation all passed.
```

Both splits load, augment, and collate cleanly with no crashes and no unexpected mask values.

---

## Status: Day 1 complete

| Item | Status |
|---|---|
| Verify IDD structure | ✅ |
| Sequential vs. stills | ✅ (confirmed sequential) |
| Dataset loader + class-collapse | ✅ (built, logged, tested on 14,027 + 2,036 real pairs) |
| Augmentation pipeline | ✅ (bug found and fixed) |
| End-to-end sanity test | ✅ (both splits, real data) |

## Open items carried forward (not blocking, not this week's scope)

- Empirical check on whether consecutive-frame spacing within IDD sequences is dense enough for meaningful optical flow (relevant to the later temporal arm, not the current baseline).
- README note on the two torch install commands (CPU local vs. CUDA college PC) — still not written.
- College PC package inventory — still deferred to first training session.