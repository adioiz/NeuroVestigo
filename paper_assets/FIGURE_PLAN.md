# Figure plan — neuron SAM 2 segmentation+tracking paper

Status legend:
- 🟢 **REAL NOW** — generated from on-disk data (`paper_assets/figures/`), ready to drop in.
- 🟡 **NEEDS BENCHMARK** — requires running competitors + `eval_masks.py` (clDice) + CTC. Cannot be real until that pipeline runs.
- 🔵 **NEEDS ASSET** — needs raw image crops / qualitative panels exported from napari or inference.

The paper (Nature-family shape) supports ~5 main figures + supplementary. Below: the 5 main + the extras, mapped to what exists.

---

## Main figures (the 5 the paper hangs on)

### Figure 1 — Hero / pipeline overview  🔵 NEEDS ASSET
**Shows:** the end-to-end story in one glance. Phase-contrast input → SAM 2 backbone with dual-rank LoRA+ adapters highlighted (encoder / mask-decoder / memory-attention) → topology-aware loss stack → per-frame propagated masks across a time-lapse → tracked neuron.
**Why:** reviewers decide the framing from Fig 1. It must say "neuron-specialized, topology-preserving, temporally-consistent" — NOT "we fine-tuned SAM 2."
**Data needed:** a clean architecture schematic (drawn) + 1 real input frame + 1 output mask crop + a 4–5 frame propagation strip. Partly drawable now; the real crops need an inference export.
**Build:** schematic in draw.io/Illustrator or matplotlib+patches; I can draft a matplotlib skeleton.

### Figure 2 — Qualitative comparison grid (THE centerpiece)  🟡 NEEDS BENCHMARK
**Shows:** rows = 3–4 representative neurons (incl. the hard touching-neurite case); columns = GT | Ours | SAM 2 zero-shot | μSAM | Cellpose-SAM | CellSAM | DeepSea. Error close-ups (insets) on fragmented neurites — red = fragmentation/broken process, showing where generalists split a single neurite.
**Why:** this is the visual proof of the connectivity failure mode. Most persuasive single figure for the "topology" claim.
**Data needed:** run every competitor on the same frames + your model's output + GT. **Blocked** until competitor inference is set up.
**Note:** design the layout now; populate after the benchmark run.

### Figure 3 — Quantitative benchmark bars  🟡 NEEDS BENCHMARK
**Shows:** grouped bars of IoU / Dice / **clDice** across models, two tiers (zero-shot generalists AND fine-tuned-on-your-data), on (a) your 20-video set and (b) CTC PhC-C2DH-U373. Mean ± std across sequences. clDice is the panel that exposes the connectivity gap IoU/Dice hide.
**Why:** the rigorous, strawman-proof number that backs the abstract's "clDice 0.9X, N-point gain over best foundation model."
**Data needed:** `eval_masks.py` (MISSING — must be recreated) computing IoU/Dice/clDice; competitor outputs; CTC eval. **Blocked.**

### Figure 4 — Ablation  🟢 REAL NOW (partial) → `fig1_training_journey.pdf`
**Shows:** cumulative contribution of each design choice: Path B (video-tracking gradient path), obj-score fix, LoRA rank, Hiera last-stage LoRA, cbDice, supervoxel, dataset growth.
**What's real now:** `fig1_training_journey.pdf` is the run-by-run cumulative IoU (0.695→0.840) with the +0.054 Path-B jump annotated — this IS a legitimate ablation-by-run.
**Gap for a rigorous ablation:** a *controlled* leave-one-out ablation (toggle each component off from the final config, holding everything else fixed) would be stronger than the historical run sequence. That needs targeted re-runs on the cluster. The run-journey version is honest and usable meanwhile.
**Supporting real figs:** `fig6_loss_decomposition.pdf` (weighted loss components), `fig5_param_budget.pdf` (LoRA budget donut).

### Figure 5 — Temporal / tracking consistency  🟢 REAL NOW (partial) → `fig2_temporal_degradation.pdf`
**Shows:** IoU stability across the full propagated sequence + identity consistency.
**What's real now:** `fig2_temporal_degradation.pdf` (per-frame IoU t0→t7, −0.124 then plateau) and `fig3_per_video.pdf` (per-video IoU). These demonstrate propagation stability.
**Gap:** a true tracking-consistency panel (identity persistence / no ID switches across ~72 frames, ideally vs a single-frame baseline like Cellpose-SAM that flickers) needs full-sequence inference export. Strong addition once inference script exists.

---

## Supplementary / supporting figures (real now)

| File | Shows | Status |
|---|---|---|
| `fig1_training_journey.pdf` | Cumulative IoU across Runs 1–7, Path-B jump annotated | 🟢 |
| `fig2_temporal_degradation.pdf` | Per-frame IoU t0→t7 (propagation stability) | 🟢 |
| `fig3_per_video.pdf` | Per-video IoU over 5 held-out videos, mean line | 🟢 |
| `fig4_learning_curves.pdf` | Val IoU & Dice vs epoch (on-disk log) | 🟢 |
| `fig5_param_budget.pdf` | Dual-rank LoRA+ donut (13.9% trainable) + adapter placement | 🟢 |
| `fig6_loss_decomposition.pdf` | Weighted val-loss components at best epoch | 🟢 |
| **Capability matrix** | Ours vs 5 competitors × capability columns (qualitative) | 🟢 (build after competitor facts confirmed) |

---

## Hard dependency: what must be built before Figs 2 & 3 can be real

1. **Recreate `eval_masks.py`** — it does not exist in either repo. Must compute IoU, Dice, **clDice** (metric, not the loss) per sequence, with mean ± std. This is the single biggest blocker for the centerpiece figures.
2. **Competitor inference harness** — run SAM 2 (zero-shot), μSAM, Cellpose-SAM, CellSAM, DeepSea on the same frames, with the documented fair protocol (identical prompts for promptable models; instance-matching for auto models; same splits/resolution).
3. **CTC evaluation** — download PhC-C2DH-U373 (+ optionally PhC-C2DL-PSC), run your model + competitors, report SEG/TRA + clDice.
4. **(Recommended) k-fold** over the 20 videos → IoU/Dice/clDice mean ± σ, so Fig 3 error bars are real.

Until (1)–(3) exist, Figs 2 and 3 are **layout templates only** — do not fabricate competitor numbers.
