# Paper plan — neuron SAM 2 segmentation + tracking

*Roadmap from "engineering report" to submitted paper. The spine is the evaluation/comparison harness; figures and prose ride on top of it. Written 2026-07-23.*

---

## Guiding principle: numbers before prose, correctness before polish

You cannot write the Results section or build the centerpiece figures until the comparison numbers exist. So we build the harness first, generate real numbers, *then* write around them.

**Decisions locked with Adi (2026-07-23):**
- **All five competitors, both tiers** — SAM 2, µSAM, Cellpose-SAM, CellSAM, DeepSea, each zero-shot AND fine-tuned on your data. Maximal, strawman-proof.
- **Polish figures continuously** — each figure is built publication-grade as it lands. A shared style module (`paper_assets/figstyle.py`) makes this cheap; when numbers shift we re-render, not redesign.

---

## Phase 1 — Metric core + data plumbing  ⟵ START HERE
**Goal: reproduce your 0.84 IoU from raw masks, and compute the first clDice number ever.**

1. **`eval_masks.py`** (the missing foundation). Given two label stacks (prediction, ground truth), compute per-frame and per-sequence: **IoU, Dice, clDice** (centerline Dice — the metric that exposes connectivity failures), plus boundary-F1 as a bonus. Return per-sequence mean ± std.
2. **Sanity test** that sells the whole paper: take one GT neurite, snip a 1-pixel gap in a copy → show **IoU/Dice barely move but clDice drops sharply**. This is Figure-2's thesis in a unit test, and proves the metric is wired right.
3. **Data adapter**: one common format (per-sequence image stack + integer-label mask stack). Loaders for (a) your held-out videos via `data_manifest.json` + `neuron-labeling/labeled_masks/`, and (b) CTC PhC-C2DH-U373.
4. **Deliverable**: `eval_masks.py your_model_masks/ GT/` reproduces IoU≈0.84 and prints the first clDice.

## Phase 2 — Competitor runners (fair protocol)
**Goal: every model produces a predicted mask stack in the same format, scored identically.**

- One runner per model, each emitting the common format:
  - `run_ours` (fine-tuned SAM 2 + dual-rank LoRA) — refactor from `src/evaluate.py`
  - `run_sam2_zeroshot` (base SAM 2, **same env — cheapest, most important baseline**)
  - `run_cellpose_sam`, `run_usam`, `run_cellsam`, `run_deepsea` — added in priority order
- **Fair protocol, codified once**: identical point/box prompts for promptable models (SAM 2, µSAM, CellSAM, ours); documented instance-matching for automatic models (Cellpose-SAM, DeepSea → reduce multi-instance output to the target neuron before scoring); same splits & resolution.
- **Orchestrator** `benchmark.py`: runs all models × all sequences, caches predicted masks, writes `results.csv` (model, sequence, frame, IoU, Dice, clDice) + `summary.json` (mean ± std).
- **Two tiers (locked: all five, both)**: SAM 2, µSAM, Cellpose-SAM, CellSAM, DeepSea — zero-shot AND fine-tuned on your data. Implication: per-model weights/environments are real setup work; each runner is isolated (own venv/conda/container) on the cluster, and the orchestrator tolerates any single model being unavailable so partial results still flow.

## Phase 3 — Public benchmark (CTC), the independent validation
- Download **PhC-C2DH-U373** (cells with processes, closest to neurons). Run ours + competitors.
- Report **GT-prompted oracle upper bound SEPARATELY from fully-automatic** (never mixed); score with official `py-ctcmetrics` (SEG / DET / TRA). Lead with **SEG** — it's the unsaturated axis (~0.74 on the tracking leaderboard) your topology losses target.

## Phase 4 — Figures (stats + visual), now with real numbers
Centerpiece figures become buildable here (see `FIGURE_PLAN.md` for the full inventory):
- **Fig 2 — qualitative comparison grid**: GT | ours | each competitor, with error close-ups on fragmented neurites.
- **Fig 3 — quantitative bars**: IoU / Dice / **clDice** × models × two tiers × (your data + CTC), mean ± std.
- **Fig 5 — temporal/tracking consistency** across a full sequence vs a single-frame baseline that flickers.
- Already real (from last session): capability matrix, training journey/ablation, temporal degradation, per-video, param budget, loss decomposition.

## Phase 5 — Writing (Nature-family shape)
- Structure: Abstract → Intro → **Results (before Methods)** → Discussion → Methods → Data/Code availability.
- Abstract leads with the **capability gap** and specific numbers ("clDice 0.9X, +N points over the best foundation model"), not "we fine-tuned SAM 2."
- **Free Nature Methods presubmission enquiry** framed as a neuron seg+track **Resource**, in parallel with drafting.
- Prep code release (GitHub + Zenodo) and decide on dataset/subset release (the ceiling-lifting move).

## Cross-cutting prerequisites (do alongside, cheap)
- **k-fold** over the 20 videos → mean ± σ (kills the split-dependency criticism).
- **Verify/replace the "MNSeg" citation** before any draft circulates.
- **Dataset license** confirmation for release.

---

## Where things live
- Harness in **this repo** (`Fine-tune-SAM2-.../benchmark/`), reading GT from the **neuron-labeling** repo. This repo has the model, `venv` with torch, and `src/`. Heavy inference runs on the **BIU B200/Slurm cluster**; metric computation + figures run locally.

## First concrete step once approved
Build `eval_masks.py` + the sanity test + the data adapter (Phase 1, items 1–4). That's a self-contained, high-value slice that unblocks everything and gives you a real clDice number this week.
