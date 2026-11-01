# NeuroVestigo pipeline — what each step does and how we know it is right

Last checked: 2026-11-02. Update this file whenever a step changes.

**Status words:** **OK** = checked with evidence. **KNOWN** = a small, understood
imperfection we keep. **FIX READY** = corrected in code, not yet run. **FIX APPLIED** = approved,
applied and tested; results being regenerated. **UNCHECKED** = nobody has
measured it yet.

## 1. The pipeline, step by step

| # | Step | File(s) | What it does, in plain words | How we know it is right | Status |
|---|---|---|---|---|---|
| 1 | Labelling | `neurons-labeling` (napari plug-in) | You draw the neuron on frame 0, propagate, correct mistakes, export a mask TIF per recording. | Every frame is looked at by a person. | OK |
| 2 | Prepare data | `cli.py prepare`, `src/prepare_data.py` | Turns each TIF into one PNG per frame at the original size (no resizing) and computes the average brightness/contrast used to normalise images. | The PNGs equal the ground-truth TIFs label for label, all 32 recordings (checked 2026-10-07). | OK. Note: the brightness statistics include the test recordings (a tiny, common leak). |
| 3 | Splits | `src/folds.py` | Decides which recordings train, validate and test in each of the three runs. Validation is the same 5 recordings in every run. | Rules enforced in code (`check_folds`) and by `tests/test_folds.py`. | OK. Caveat: in run 1, test recording A12_C1_100k_2 shares its frames with training recording A12_C1_100k_1. |
| 4 | Same data on the cluster | `src/data_fingerprints.json` | A fingerprint of every image and mask, so the cluster cannot train on stale data. | The training job refuses to start on any mismatch (it caught the old A71_C1 mask). | OK |
| 5 | Training input | `src/augmentations.py`, `src/neuron_dataset.py` | Cuts 16-frame clips, augments them, and enlarges everything to 1024×1024 for the model. | `tests/test_alignment.py`: the label enlarged to 1024 covers the image's own content (overlap 0.998; the old way 0.70–0.87 on thin lines). A 1-epoch run on the Mac trains, validates and records the setting. | FIX READY (2026-11-02), for new runs: labels and the frame-0 prompt are enlarged with centred "nearest-exact" (`data.mask_resize` in `src/config.py`, saved in each run's `training_config.json`). Every model trained before — the paper's three runs and the released one — used off-centre "nearest" (labels half a pixel at 1024, 0.07–0.3 of a native pixel, off the image) and keeps it: at inference `run_ours.py` reads a missing key as "nearest", and a resumed run keeps the mode it started with. The transforms have no default, so a forgotten setting fails at once. |
| 6 | Model | `src/trainer.py`, `src/lora.py` | SAM 2.1 tiny with LoRA adapters; 5,557,250 trainable parameters. | The count is printed and matches in every run. | OK |
| 7 | Loss | `src/loss_fns.py` | Focal 3, Dice 2, cbDice 0.3, supervoxel 0.3. The two topology terms run at 512×512. | Unit tests (`tests/test_cbdice_*`, `tests/test_supervoxel_speed.py`). cbDice resizes both sides bilinearly (centred); the supervoxel term resizes both sides with the same "nearest", so they stay aligned with each other. | OK |
| 8 | Validation and checkpoint choice | `src/evaluate.py`, `src/trainer.py` | Every 2 epochs: IoU and Betti-0 on the validation recordings; keeps the best-IoU checkpoint and the "topology" checkpoint. | Selection compares epochs with each other, so a constant offset does not change which epoch wins. | OK (corrected 2026-11-02). Prediction and label are both 1024×1024, so the "nearest" shrink in `evaluate.py` never runs; validation is scored against the labels of row 5, so new runs are validated on centred labels too. |
| 9 | Our inference (and zero-shot SAM 2) | `benchmark/run_ours.py` | Runs the model through a whole recording from the frame-0 mask and brings the result back to the recording's size. | `tests/test_alignment.py`: a perfect prediction comes back exactly. Same model output, run 1: IoU 0.769 → 0.808. | OK. Fixed 2026-11-01. Was: decide yes/no at 1024, then shrink off-centre. Now: shrink the scores bilinearly, then decide (SAM 2's own order). All 12 test recordings rerun with `benchmark/regen_aligned.sh`. The frame-0 prompt is enlarged the way the checkpoint's own training enlarged its labels (`saved_mask_resize`, row 5). |
| 10 | Baselines: Cellpose-SAM, Omnipose, NeuroQuantify | `benchmark/run_cellpose.py`, `run_omnipose.py`, `run_neuroquantify.py` | Segment frames enlarged ×2, ×3, ×4; keep the cells touching the target; shrink back. | Ideal-segmenter test (`tests/test_alignment.py`): the old shrink alone caps them at IoU 0.76 / 0.68 / 0.65; centred, 0.86 / 0.99 / 0.94. | OK. Fixed 2026-11-01: choose the cells at the enlarged size, then shrink by area (`benchmark/native_res.py`); all 12 test recordings rerun with `benchmark/regen_aligned.sh`. A pixel exactly half covered counts as foreground, so ties go to the baseline (on Cellpose-SAM's ×2 grid its masks come out 3–7% larger). |
| 11 | Baselines: µSAM, SAM 3, SAM 3.1 | `benchmark/run_usam.py`, `run_sam3_mask.py`, `run_sam3.py` | Their own libraries return masks at the recording's size. | µSAM reruns identically; SAM 3 and SAM 3.1 reproduce the paper's stacks exactly (1.0000). | OK |
| 12 | Scoring | `benchmark/eval_masks.py`, `benchmark/score.py` | IoU, Dice, cbDice and clDice per frame at the original size; frame 0 is never scored; each recording counts equally. | No resizing happens (sizes already match); cbDice(gt, gt) = 1; the paper's Table 1 reproduces exactly. | OK |
| 13 | Three-run tables | `benchmark/aggregate_folds.py` | Mean ± SD over the three runs, drift \|Q4 − Q1\|, and per-recording paired tests. | With run 1 alone it reproduces the paper's tables to three decimals. | OK. NeuroVestigo = the topology checkpoint only (the IoU-best row was dropped 2026-11-01; its predictions in `ours_23rec`, `ours_fold2`, `ours_fold3` predate the resize fix — do not use them). Drift on a single recording is sensitive to inference noise: the same model's masks differ by ~10% of pixels between the cluster and the Mac from frame 1, and per-recording drift moved by up to 0.055 between two inference runs (B11_C1_100k: 0.269 and 0.312 on two Mac runs). Every SAM 2-based prediction in the tables is from the Mac; the October tables had runs 2–3 from the cluster. |
| 14 | Release | `neurons-labeling/checkpoints-16-frames/` | The weights the plug-in ships: run 1's topology checkpoint. | Loaded through the plug-in's own code: all 152 adapter and 172 full-rank tensors exact. | OK |
| 15 | Plug-in inference | `neurons-labeling/src/gui.py` | The plug-in runs SAM 2's own video predictor. | Its output resize is SAM 2's bilinear (correct). | FIX APPLIED (approved 2026-11-01, neurons-labeling `c71f64e`). Measured with the released model on its held-out recordings: the plug-in saved frames as default-quality JPEG, which blurred thin neurites (mean IoU 0.772 vs 0.808 for our pipeline; −0.07 on the low-resolution recordings). Now saved at quality 100: 0.798 through the plug-in's own code. ImageNet vs dataset colour statistics costs only ~0.006 and is left as is. |

## 2. Every resize in the pipeline

Most problems of the kind found in November are resizes. "Centred" means each
output pixel is computed from the middle of the area it covers, not its corner.

| Where | What is resized | Method | Centred? | Effect | Status |
|---|---|---|---|---|---|
| Training and validation input | image → 1024 | bilinear (antialias) | yes | — | OK |
| Training and validation input | mask and frame-0 prompt → 1024 | "nearest" before 2026-11-02; "nearest-exact" for new runs | old: no (½ pixel at 1024); new: yes | 0.07–0.3 native pixel; on thin lines the old label overlaps its image only 70–87% at 1024 | FIX READY (new runs) |
| Augmentation (rotate, shear) | image / mask | bilinear / nearest, via affine | yes | — | OK |
| Topology loss | prediction and label → 512 | cbDice: bilinear; supervoxel: the same "nearest" on both | yes / same on both sides | — | OK |
| Validation metric | label 1024 → model output size | "nearest", only if the sizes differ | — | none: both are 1024, so it never runs | OK |
| `run_ours.py` output | 1024 → native | was yes/no then "nearest"; now bilinear on scores | was no; now yes | +0.04 IoU for ours (run 1) | OK (fixed 2026-11-01) |
| Cellpose-SAM / Omnipose / NeuroQuantify output | ×2 / ×3 / ×4 → native | was "nearest" (OpenCV); now target on the upscaled grid, shrunk by area | was no; now yes | large on thin lines (ideal test) | OK (fixed 2026-11-01) |
| µSAM / SAM 3 / SAM 3.1 output | internal → native | the libraries' own bilinear; PIL nearest | yes | — | OK |
| Plug-in input | frame → JPEG → 1024 | was default-quality JPEG; now quality 100. SAM 2's loader: PIL resize, ImageNet statistics | differs slightly from training | JPEG cost 0.03 IoU; statistics ~0.006 | FIX APPLIED (JPEG) |
| Plug-in output | → native | SAM 2's bilinear | yes | — | OK |
| Scoring | none (sizes match) | — | — | — | OK |

## 3. Rules that keep this table true

1. **Any new resize** uses a centred method: bilinear for scores, area or
   nearest-exact for labels. It gets a round-trip test like `tests/test_alignment.py`.
2. **Every number in the paper** comes from a committed script and a committed
   result file (`benchmark/results/`).
3. **After any change to inference or scoring:** run `tests/test_alignment.py`,
   and check that run 1 still reproduces its numbers within noise.
4. **After any ground-truth edit:** `python -m src.folds --write-fingerprints`, commit it,
   and sync `training_data/` to the cluster.
5. **Results from outside** (like the November tip) are checked on our own data,
   from the same model outputs, before anything changes.
