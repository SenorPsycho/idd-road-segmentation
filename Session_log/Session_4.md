# Session 4 — Day 5: Evaluation + Qualitative Outputs

**Date:** 2026-08-12 – 2026-08-13
**Scope:** RoadVision Nepal Phase 2 (IDD baseline arm) — Day 5 of implementation plan
**Environment:** College PC (RTX 4070 SUPER) for val-set metrics; local Windows CPU machine for panel generation and Kathmandu inference, after transferring `runs/2026-08-11_1628/` (checkpoint + config) from the college PC

---

## Objectives (from Day 5 plan)

1. Compute final metrics on the held-out val set
2. Generate qualitative comparison panels: input → predicted mask → overlay, on a spread of IDD val examples
3. Run inference (no fine-tuning) on Kathmandu frames — the domain-transfer differentiator

All three closed by end of session, with one investigation branch (worst-case metric floor) and one scope change (Kathmandu: full-dataset panel generation, not a single "best frame" pick) along the way.

---

## 1. Val-set metrics (`Model_Train/evaluate.py`, new)

New script, not bolted onto `train.py` — imports the same `IDDBinarySegDataset`, `get_val_transforms`, `build_model`, and metrics functions rather than duplicating logic. Runs a single inference-only pass (`torch.no_grad()`) over all 2,036 val images using `runs/2026-08-11_1628/best.pt`.

**Design decisions:**
- `DataLoader(shuffle=False)` — required, since the script maps each item in a batch back to `val_dataset.samples[index]` by position to recover its frame_id for per-image logging. `dataset.py` itself was **not** modified to also return frame_id, to avoid touching code `train.py` already depends on.
- `encoder_weights=None` when building the model — the trained state_dict is loaded immediately after, so there's no need to also fetch ImageNet weights.
- Per-image IoU/precision/recall written to `eval_metrics.csv` (2,036 rows) as a byproduct of the same pass, intended to drive panel selection in step 2 rather than eyeballing images blind.

**`metrics.py` change:** extracted the IoU/precision/recall formula out of `MetricsAccumulator.compute()` into a standalone `compute_ratios(tp, fp, fn, tn)` function; `compute()` now calls it. `evaluate.py` reuses the same function for both per-image rows and the epoch-level aggregate — one formula, not two implementations that could drift apart. Existing smoke test in `metrics.py` untouched and still passes.

**Result:**
```
IoU:       0.9649
Precision: 0.9827
Recall:    0.9816
TP=168790225  FP=2968579  FN=3167529  TN=357627038
```
IoU matches the training-time epoch 29 val IoU exactly — confirms checkpoint, config, and val data are correctly aligned, not a coincidence to wave off.

---

## 2. Qualitative panels (`Model_Train/generate_panels.py`, new)

Reads `eval_metrics.csv` to select 9 examples: **3 best**, **3 worst**, and **3 "median"** — the median group deliberately spread across the 25th/50th/75th percentile of the IoU distribution rather than clustered right at the exact median, since three near-identical images would show nothing new. Re-runs inference on just those 9 frames (independent of `evaluate.py`'s pass) and assembles a single grid image (input | predicted mask | overlay per row, green alpha-blend overlay) rather than 9 separate files, so it drops into the writeup as one figure.

**Output:** `runs/2026-08-11_1628/qualitative_panels.png`. Visual review confirmed overlays are correctly oriented (green tracks the actual road surface in BEST/MEDIAN rows).

### Investigation: WORST 3 examples all exactly IoU = 0.0000

Suspicious as a coincidence, so a small standalone check (`Model_Train/check_worst_gt.py`, new) read the raw ground-truth mask PNGs for those 3 frame_ids directly (no model, no inference) and reported per-class pixel counts:

| frame_id | non-drivable(0) | drivable(1) | ignore(255) |
|---|---|---|---|
| 004415 | 0 | 0 | 921,600 (100%) |
| 005640 | 15,466 | 0 | 906,134 (98.3%) |
| 006240 | 0 | 0 | 921,600 (100%) |

**Confirmed: metric floor effect, not model failure.** All three have zero drivable-class ground-truth pixels, so `IoU = TP/(TP+FP+FN)` collapses to `0/(0+FP+0) = 0.0000` regardless of prediction quality. Two of the three (004415, 006240) are **entirely ignore-labeled** — no usable annotation exists for those frames at all, not even non-drivable. The third (005640) is a genuine dense-clutter scene where the road surface is fully occluded in ground truth. Flagged explicitly for the writeup as two distinct claims: a measurement caveat (metric floor on annotation-sparse frames), not a model weakness.

---

## 3. Kathmandu domain-transfer spot check

### Setup friction (resolved)

Checkpoint (`best.pt`) is git-ignored (124MB, exceeds GitHub's limit — see Session 3) and only existed on the college PC; Kathmandu frames only exist locally. Rather than transferring 39GB of frames to the college PC, the entire `runs/2026-08-11_1628/` folder (checkpoint + config + evidence files) was copied from the college PC to the local machine instead.

Local environment was missing `segmentation_models_pytorch` (previously untested there beyond Day 2's random-init structural smoke test) — installed via `pip install segmentation-models-pytorch` (hyphenated package name, underscored import name).

Kathmandu frames confirmed to live at `extracted_frames/v01_final` ... `v07_final` at repo root, filenames like `v01_frame_0001.jpg` — matching the naming convention already assumed by the script, no changes needed.

### Sampled spot check (`Model_Train/kathmandu_inference.py`, new)

2 frames per video, evenly spaced (not consecutive) across all 6 usable videos (v01–v05, v07 — v06 excluded per known fisheye distortion + burned-in telemetry). No ground truth exists for these frames, so this is visual-only — no scoring. Output: `kathmandu_panels.png`, 12-row grid.

**Review findings:**
- **Positive result:** the model — trained entirely on Bangalore/Hyderabad IDD footage, zero Kathmandu fine-tuning — tracks Kathmandu road boundaries reasonably well across most of the 12 frames, correctly navigating around buses, clustered pedestrians, and motorbikes without misclassifying them as drivable. Real domain-transfer signal worth stating plainly in the writeup.
- **Flagged, unresolved:** `v07_frame_0231` produced a visibly noisy/fragmented predicted mask, unlike the clean boundary in every other frame. Input image for that frame looks hazy/overexposed. Not yet confirmed whether this is a genuine model weakness under haze/glare or a downscaling artifact from the grid image — full-resolution follow-up still open.
- **Flagged, explained (domain gap, not a bug):** predicted drivable area bleeds onto sidewalks/paved shoulders in a couple of frames (v01_frame_0001, v04_frame_0006). Reasoned explanation: IDD's source cities generally have curbed, clearly separated sidewalks, while Kathmandu streets often have road and footpath running together with no hard boundary — a legitimate domain-shift finding consistent with the project's whole framing, not a defect to fix.

### Full-dataset panel generation (`Model_Train/kathmandu_full_inference.py`, new)

Scope changed mid-session: rather than computing a no-ground-truth "best frame" proxy score (confidence + mask-connectivity heuristic was proposed but not built, since the direction changed before it was needed), the decision was to run inference across **every frame** in all 6 usable video folders and save a 3-panel image (input | predicted mask | overlay) next to each original frame file.

**Design notes:**
- **Resumable:** skips any frame whose `<frame_stem>_panel.png` already exists, since a full CPU pass over 2,000+ frames is a real time cost and the run should survive an interruption without redoing completed work.
- `--limit N` flag for a small test run (N frames per video) before committing to the full set.
- `matplotlib.use("Agg")` (no GUI backend) and explicit `plt.close(fig)` per frame, to avoid memory buildup across thousands of saved figures.
- Progress logged every 50 frames with a running frames/sec rate, so total time can be projected from an early sample.

**Status:** tested successfully by the user (small-scale test run confirmed working). Full unlimited run across the complete `extracted_frames/` set not explicitly confirmed as completed in this session — see open items.

---

## Status: Day 5 complete

| Item | Status |
|---|---|
| Val-set metrics (`evaluate.py`) | ✅ IoU 0.9649, matches training-time epoch 29 exactly |
| `metrics.py` — `compute_ratios()` extracted for reuse | ✅ |
| Qualitative panels (`generate_panels.py`) | ✅ 9-example grid (best/median-spread/worst), reviewed |
| Worst-case investigation (`check_worst_gt.py`) | ✅ confirmed metric-floor effect, not model failure |
| Kathmandu sampled spot check (`kathmandu_inference.py`) | ✅ 12-frame grid, reviewed, 2 findings flagged |
| Kathmandu full-dataset panels (`kathmandu_full_inference.py`) | ✅ built and tested; full unlimited run not confirmed complete |

## Open items carried forward

- **`v07_frame_0231` anomaly** — noisy/fragmented predicted mask on a hazy-looking input frame. Not yet isolated at full resolution to confirm whether it's a genuine haze/glare weakness or a grid-downscaling artifact.
- **Sidewalk/road boundary bleed** in some Kathmandu frames — real domain-gap finding (IDD's curbed sidewalks vs. Kathmandu's often-unseparated road/footpath), needs a sentence in the writeup, no code fix implied.
- **Worst-case metric-floor finding** — two of three worst-IoU val frames have no usable ground truth at all (100% ignore-labeled); needs to be stated as a measurement caveat, not folded into the model's overall failure analysis.
- **Confirm whether the full unlimited `kathmandu_full_inference.py` run (all frames, no `--limit`) was actually completed** — last confirmed status was a successful small-scale test, not the full set.
- **No-ground-truth "best frame" proxy heuristic** (confidence + mask-connectivity/cleanliness) was proposed but not implemented — deferred when scope moved to full-dataset panel generation instead. Still an open option if a single standout Kathmandu example is needed later (e.g. for the certificate-course demo).
- **Local environment package gaps**: `segmentation_models_pytorch` was missing and had to be installed ad hoc; `requirements.txt` vs. actual local `.env` contents not fully audited — other missing packages may still surface.
- Writeup must incorporate, in addition to Session 3's carried-over items (fast saturation, mild overfitting plateau): the worst-case metric-floor explanation, and the Kathmandu domain-transfer findings (positive transfer signal + sidewalk-bleed domain gap + the unresolved haze frame).
- Certificate-course submission packaging (Streamlit/Gradio demo: input frame → predicted mask → overlay, plus Kathmandu qualitative examples) — not started this session, still pending per the broader Phase 2 plan.
