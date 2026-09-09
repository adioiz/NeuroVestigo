# Mask-prompted SAM 3.1 vs the table — for verification

Run: `run_sam3_mask.py`, log `logs/sam3_mask_*.out`, 2026-09-09.
Route: Meta's own SAM 2-task recipe (`examples/sam3_for_sam2_video_task_example.ipynb`, cell 11).

## Removing frame 0, as the paper's protocol requires

main.tex:1367 -- "Frame 0 ... is excluded from all quantitative
evaluations", 328 frames = (73-1)x2 + (93-1)x2. The runner reports the
mean INCLUDING frame 0, so each is corrected here as
`(mean_incl * N - iou_0) / (N - 1)`. Check the arithmetic:

| recording | N | mean incl. f0 | IoU f0 | **mean excl. f0** |
|---|---|---|---|---|
| A71_C1 | 73 | 0.7658 | 1.0000 | **0.7625** |
| A81_C1 | 73 | 0.7392 | 0.9990 | **0.7356** |
| A12_C3_100k | 93 | 0.3542 | 1.0000 | **0.3472** |
| A12_C1_100k_2 | 93 | 0.6718 | 1.0000 | **0.6682** |

Recording-level scores weighted equally (main.tex:1367): **0.6284**

## Against the published table

| Method | IoU | Per-frame supervision |
|---|---|---|
| **NeuroVestigo** | **0.763** | 1 annotation (frame 0) |
| SAM 2 (zero-shot) | 0.639 | 1 annotation (frame 0) |
| **SAM 3.1 (mask-prompt)** | **0.628** | **1 annotation (frame 0)** |
| mu-SAM (box-prompt) | 0.575 | GT box, every frame |
| SAM 3.1 (point-prompt) | 0.471 | 10 GT points, every frame |
| SAM 3.1 (text-prompt) | 0.200 | none (automatic) |

## What still has to be checked

- **These are IoU only.** The table also needs Dice, cbDice and clDice,
  which only `eval_masks.py --skip-first-frame` produces. Until that is
  run, only the IoU column above is usable.
- The exclusion arithmetic assumes the runner's rounded means; the
  authoritative number is whatever `eval_masks.py` reports.
- `A12_C3_100k` is the weak recording (0.347). Worth looking at before
  the number goes in a table.

## What this changes in the paper

1. **The fairness gap closes.** SAM 3.1 can now be given the same
   frame-0 mask our model and SAM 2 receive, so the protocol asymmetry
   a reviewer could attack disappears.
2. **One sentence becomes false.** main.tex:1456-1458 says SAM 3.1
   "substantially underperforms zero-shot SAM 2". Under matched
   supervision it is 0.628 against 0.639 -- essentially level, not
   substantially below. That must be rewritten.
3. **Our margin holds and gets cleaner**: 0.763 vs 0.628 = 0.135 IoU, now under identical supervision
   rather than across different prompt budgets.
4. The table caption (main.tex:1400-1407) says the mask entry point was
   "not benchmarked". That is no longer true.
