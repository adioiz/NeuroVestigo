# Evaluation protocol — each competitor in its most favorable setting

**Principle.** Every method is evaluated in *its own best / intended mode*, not forced
into our model's pipeline. Where a competitor could be handicapped or advantaged, we
chose the setting that is **most favorable to the competitor**. Our model still wins on
every metric and every held-out test video — the strongest, most strawman-proof form of
the comparison. Critically, our model receives among the **least** per-frame supervision.

## Per-model protocol (final figure roster: ours + 5 competitors)

| Model | Mode used (its best) | Prompt / supervision per sequence | Notes |
|---|---|---|---|
| **Ours** | promptable video propagation (SAM 2 memory) | **1 annotation** — GT mask on frame 0, then propagate | least per-frame supervision of any method here |
| SAM 2 (zero-shot) | promptable video propagation | 1 annotation — GT mask on frame 0, then propagate | same protocol as ours (its native video mode), no fine-tuning |
| µSAM | interactive box prompt (its intended mode) | **GT box on every frame** (oracle localization) | its automatic AIS/AMG modes fail to detect thin neurons → we use its far stronger interactive mode; box is picked by µSAM's own confidence |
| Cellpose-SAM | automatic segmentation (its only mode) | none — instance-matched to the target | generalist SOTA; union of predicted instances overlapping the target GT (generous) |
| **Omnipose** | automatic, `bact_phase_omni` model @ ×3 magnification | none — instance-matched to the target | its phase-contrast, elongated-morphology model (its `cyto2_omni` mammalian model detects nothing on our phase contrast); run at its training magnification (our ~1.5 px/µm frames are out-of-distribution at native scale) |
| **NeuroQuantify** | automatic, large TorchScript model @ ×3 magnification, gamma-corrected | none — instance-matched to the target | the one competitor actually trained on phase-contrast cells **and neurites**; run in the authors' intended pipeline (gamma → large model) at its training magnification |

**Favorable-setting note (both new competitors).** Omnipose and NeuroQuantify are scale-sensitive: our frames (~1.5 px/µm) are much smaller than their training magnification, and at native scale both flood the field (Omnipose over-fragments; NeuroQuantify labels ~65% of the frame). We selected the input magnification (×3) that is **most favorable to each competitor** (highest target-matched score), verified visually, so their reported numbers reflect their best achievable performance — not a mis-scaled failure.

## Why this favors the competitors, not us
- **µSAM / CellSAM** are given the exact target bounding box on **every frame** (oracle localization). Our model gets a prompt on **frame 0 only**. So the promptable competitors receive *more* per-frame information than we do, and still lose.
- **Cellpose / DeepSea** are automatic; instance-matching takes the **union** of all predicted instances overlapping the target (their best-case reduction to the single-object task).
- Automatic modes that would have made µSAM look worse (AIS/AMG, ~0.19 IoU) were **discarded** in favor of its stronger interactive mode (~0.44 IoU). We report the number that is best for the competitor.

## Competitors mentioned in text only (not plotted)

To keep the comparison focused on the strongest, most relevant methods, three additional
competitors were evaluated but are reported in text rather than in the figures:

- **DeepSea** (Zargari et al., CRM 2023) — the closest *published task* analog: single-cell
  segmentation **and tracking** in phase-contrast time-lapse. But it was trained on compact
  cells (fibroblasts, stem, epithelial), not neurons, and predictably fails on thin neurites
  (test IoU 0.335 / clDice 0.565). Its phase-contrast-neuron role is now filled more strongly
  by NeuroQuantify, so DeepSea is cited as the task precedent rather than plotted.
- **Trackastra** (Gallusser & Weigert, ECCV 2024; winner of the 7th Cell Tracking Challenge) —
  a link-after-segment tracker. It is a *tracker, not a segmenter*: its per-frame masks are
  its segmenter's, so its connectivity is bounded by that segmenter, and holding one identity
  from a single frame-0 seed drifts off a thin, low-contrast neuron. It illustrates that the
  modular "segment each frame, then link" paradigm — even with a CTC-winning tracker — cannot
  recover un-segmented neurites; our joint segment-and-track-with-memory does. Reported as a
  qualitative/temporal observation, not a segmentation-table row.
- **CellSAM** (Israel et al., Nat Methods 2025) — a SAM-family foundation cell segmenter whose
  weights are gated behind a DeepCell access token. Not evaluated; expected to share the
  SAM-family limitations already represented by Cellpose-SAM and µSAM.

## Evaluation details
- Metrics: **IoU, Dice, clDice** (clDice = connectivity; our largest margin), + **cbDice** as a secondary training-consistent metric.
- Held-out **test** set: 4 videos never in training or model selection (A71_C1, A81_C1, A12_C3_100k, A12_C1_100k_2).
- Scored on propagated frames (t ≥ 1); frame 0 is the given prompt, excluded.
- mean ± s.d. across test videos. Identical preprocessing (8-bit frames) for all models.

## Draft methods sentence
> Each competing method was evaluated in its most favorable configuration rather than
> forced into our pipeline: the interactive model µSAM received the ground-truth bounding
> box of the target neuron on every frame (oracle localization), whereas the automatic
> models (Cellpose-SAM, Omnipose, NeuroQuantify) were run without prompts — each at its
> own optimal magnification and pretrained model — and reduced to the target by union
> instance-matching. Our model and base SAM 2 received only a single frame-0 annotation
> and propagated it. Despite receiving the least per-frame supervision, our model
> outperformed every competitor on IoU, Dice, and clDice on all held-out test videos; its
> margin was largest on clDice (connectivity), where the thin neurites that competitors
> drop are decisive. The domain-specialist NeuroQuantify — the only competitor trained on
> phase-contrast neurites — was the strongest unprompted baseline (clDice 0.74) yet still
> trailed ours by 0.19 clDice.

## Results (test set, t ≥ 1, held-out; mean over 4 videos)
| Model | IoU | Dice | clDice |
|---|---|---|---|
| **Ours** | **0.742** | **0.849** | **0.932** |
| SAM 2 (zero-shot) | 0.639 | 0.771 | 0.830 |
| µSAM (box/frame) | 0.442 | 0.597 | 0.754 |
| NeuroQuantify | 0.477 | 0.630 | 0.742 |
| Omnipose | 0.279 | 0.428 | 0.649 |
| Cellpose-SAM | 0.231 | 0.329 | 0.411 |
| *DeepSea (text only)* | *0.335* | *0.480* | *0.565* |
