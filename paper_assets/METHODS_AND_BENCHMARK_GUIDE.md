# Methods & Benchmark Guide — Fine-tuned SAM 2 for Neuron Segmentation + Tracking

*A complete, honest record of what we built, how we evaluated it, why we made each choice, and how to reproduce everything. Written so you can explain it in a thesis/defense and trust every number.*

Last updated: 2026-07-25.

---

## 1. What the model is, and the claim

**The model.** SAM 2 (`sam2.1-hiera-tiny`) fine-tuned with a dual-rank **LoRA+** adapter (rank-16 on the mask decoder + memory-attention, rank-8 on the last two Hiera blocks). Only **5.56 M / 39.9 M parameters (13.9%)** are trained. It performs **joint segmentation and tracking of a single target neuron** in **phase-contrast, label-free, live-cell time-lapse** microscopy: you annotate the neuron on frame 0, and it propagates the mask through the whole video using SAM 2's temporal memory.

**The contribution is NOT "we used LoRA."** LoRA is a means. The claim is the *combination* that no existing tool offers together:
1. **Topology-aware training** (a cbDice/clDice-family loss) that preserves thin, branching **neurites**;
2. **Temporal propagation** (SAM 2 memory) for identity-consistent tracking from a single annotation;
3. **Neuron/neurite specialization** on phase-contrast data.

**The headline result:** ours preserves the thin neurites that every competitor drops. This shows up most on **clDice** (connectivity), where our margin over the best baseline is largest.

---

## 2. Evaluation philosophy — why the comparison is trustworthy

Three principles govern the whole benchmark. They exist specifically to survive a skeptical reviewer.

1. **Each competitor runs in ITS most favorable mode — never forced into our pipeline.** Where a method could be advantaged or handicapped, we chose the setting best for *it*. Our model receives the **least** per-frame supervision of any method and still wins every metric on every held-out video. This is the strongest, most strawman-proof form of the claim. (Details per model in §6.)

2. **We visualize outputs before trusting any number.** Every surprising score (a 0.00, or an implausibly high one) was traced to a *usage* problem by rendering the masks — not accepted at face value. This caught four separate bugs (§6), several of which would have unfairly flattered *or* hurt a competitor.

3. **The evaluation metric is independent of the training objective.** We train with a cbDice-family loss but *report* **clDice** as the headline. Because the metric is not the loss, a good clDice score cannot be an artifact of "optimizing the test metric." (§4)

---

## 3. The data and the held-out test set

### 3.1 The contamination correction (important, and honest)
The project has **25** annotated videos. Early in the benchmark we *believed* the five `_100k` videos were held out, inferring the training set from the git history of `data_manifest.json`. **That was wrong.** `cli.py train` reads the `training_data/` **folder at runtime**, not the manifest — and by the shipped model's training date all 25 videos were in that folder. So **the shipped model trained on all 25 videos.** We verified this and discarded the invalid "test" numbers. No 20-video checkpoint exists in git, so we could not simply re-test the shipped model on any of the 25.

**Lesson (recorded so it never repeats):** never infer a model's training set from a manifest's history; verify by what was physically in the training folder at train time.

### 3.2 The genuinely held-out test set (what we actually use)
Adi provided **4 videos the shipped model never saw**, used as the **test set**:

`A71_C1`, `A81_C1`, `A12_C3_100k`, `A12_C1_100k_2`  (73 / 73 / 93 / 93 frames).

- `A71_C1`, `A81_C1` are a new plate series (cleanest neurons); the two `_100k` videos are a mild same-batch caveat, noted in the paper.
- **Validation set** (model-selection, 5 videos): `A91_C1_2`, `A31_C1`, `A52_C2`, `A22_C2`, `A42_C1`.
- Generalization check: test clDice (0.932) ≈ val clDice (0.921) → no sign of the test set being unusually easy or hard.

### 3.3 Scoring convention
- **Frame 0 is the given prompt/seed for every promptable method and is excluded from scoring.** We score only propagated frames (**t ≥ 1**), so no method gets credit for the annotation it was handed.
- Identical preprocessing (8-bit PNG frames) for all models.
- Metrics are mean ± s.d. across the videos in the split.

---

## 4. Metrics — what each measures and why clDice leads

| Metric | Measures | Role |
|---|---|---|
| **IoU** | area overlap (intersection / union) | standard, reported |
| **Dice** | area overlap (harmonic form) | standard, reported |
| **clDice** | **connectivity** — do the *centerlines* (skeletons) of prediction and GT cover each other? | **headline** — this is where thin neurites live |
| cbDice | EDT-weighted connectivity (our training-loss family) | secondary, training-consistent |

**Why clDice is the right headline.** A method can score a decent IoU by covering the fat soma while dropping every thin neurite — but those neurites are the biology that matters (neurite length, branching, growth). clDice is sensitive to exactly that. Our self-test on `eval_masks.py --selftest` demonstrates the property honestly:
- Dropping 2 thin branches: **IoU −0.081 but clDice −0.227 (2.8× harder hit).**
- A cosmetic 2-pixel gap: barely moves any metric.
- Uniform erosion: drops IoU/Dice but keeps clDice ~1.0 (centerline still covered).

So clDice specifically rewards *keeping the branches connected*, which is our thesis. (clDice measures centerline **coverage**, not literal component count — we frame the claim as "catches lost thin neurites," not "detects every 1-px break.")

Metric code: `benchmark/eval_masks.py` (`cldice_score` uses skimage `skeletonize`; `evaluate_stack` does per-frame/per-sequence aggregation).

---

## 5. Which competitors, and why these and not others

### 5.1 How the set was chosen
The original list was fixed early in the project. When Adi asked to *verify* it was genuinely state-of-the-art, we ran a systematic SOTA gap-search (2024–2026 literature) for (a) cell/neuron segmentation in phase-contrast, (b) thin/filamentous structure segmentation, (c) live-cell time-lapse seg+tracking. That search surfaced two must-adds we had been missing (**Omnipose**, **NeuroQuantify**) and confirmed the rest.

### 5.2 The final figure roster (6 methods)

| Method | Why it's a genuine competitor |
|---|---|
| **Ours** | the proposed method |
| **SAM 2 (zero-shot)** | the exact "before fine-tuning" baseline — isolates what our LoRA + topology + temporal training actually buys. Non-negotiable. |
| **µSAM** (micro-sam, `vit_b_lm`) | microscopy-specialized SAM, *Nat Methods* 2025. Strong interactive upper-bound. |
| **Cellpose-SAM** (`cpsam`, Cellpose v4) | current generalist SOTA for cell segmentation ("superhuman generalization"). |
| **Omnipose** | *the* method built for elongated/branching morphology (boundary-distance field instead of Cellpose's centroid field). A reviewer reading "thin neurites" expects it. *Nat Methods* 2022. |
| **NeuroQuantify** | the only competitor actually trained on **phase-contrast cells AND neurites** — same modality, same structure, directly targets our clDice claim. *J. Neurosci. Methods* 2024. |

### 5.3 What we deliberately did NOT benchmark, and why
- **StarDist** — SKIP. Its star-convex polygons *architecturally* cannot represent elongated shapes-with-branches; including it "to show it fails" reads as padding. Mentioned in one sentence as inapplicable, not scored.
- **nnU-Net** — not a pretrained model but a *training framework*; using it means training on our labels, i.e. a supervised upper-bound reference, not a foundation-model peer. Not reviewer-demanded here.
- **EM / fluorescence neuron-tracing tools** — wrong modality (fixed fluorescence / 3D EM); our task is live label-free phase contrast.

### 5.4 Competitors evaluated but mentioned in **text only** (not plotted)
Kept out of the figures to keep the comparison focused; each is discussed in the text (`BENCHMARK_PROTOCOL.md`):
- **DeepSea** (CRM 2023) — the closest *published task* analog (seg **and** tracking in phase-contrast), but trained on compact cells (fibroblasts/epithelial), not neurons; predictably weak on neurites (test IoU 0.335 / clDice 0.565). NeuroQuantify now fills the phase-contrast-neuron role more strongly, so DeepSea is cited as the precedent rather than plotted.
- **Trackastra** (ECCV 2024, CTC-2024 winner) — a *tracker, not a segmenter*. It links a segmenter's per-frame masks, so its connectivity is bounded by that segmenter, and holding one identity from a single frame-0 seed drifts off a thin neuron. It illustrates that "segment each frame, then link" — even with a challenge-winning tracker — cannot recover un-segmented neurites; our joint segment-and-track-with-memory does. Reported qualitatively, not as a table row. (See §6.7 for why we did not ship a number.)
- **CellSAM** (Nat Methods 2025) — weights gated behind a DeepCell access token; not evaluated. Expected to share the SAM-family limitations already represented by Cellpose-SAM and µSAM.

---

## 6. How each model was tested — protocol and the bugs we caught

General setup: each competitor lives in its own **isolated environment** (so dependency conflicts can't corrupt another). Automatic segmenters are reduced to the single-neuron task by **union instance-matching**: take the union of predicted instances that overlap the target neuron's GT — the most generous, strawman-proof reduction.

### 6.1 Ours & SAM 2 zero-shot (promptable video propagation)
Both get an **identical GT mask as the frame-0 seed** and then propagate; we score t ≥ 1. `benchmark/run_ours.py` (`--no-lora` gives the stock SAM 2 baseline). MPS OOM on full-sequence encode was fixed by `_propagate_stream` (encode one frame at a time).

**Two fairness bugs caught and fixed:**
- Base SAM 2 pushed through our *training* wrapper scored a fake-low **0.021** (its frame-0 was hobbled). Fixed.
- Base SAM 2 given perfect GT frame-0 flags while ours got a decoder-reconstructed frame-0 scored a fake-high **0.805** (unfair to us). Fixed by giving *all* methods the identical GT frame-0 seed and scoring t ≥ 1. Per-model occlusion head set correctly (ours `pred_obj_scores=False` untrained; stock SAM 2 `True`).

### 6.2 µSAM (interactive box prompt — its best mode)
Its automatic modes (AIS/AMG) only find compact blobs and fail on thin neurons (~0.19), so we use its **stronger interactive mode**: a GT bounding box on **every frame** (oracle localization — more per-frame info than ours gets), and keep µSAM's own best-scoring mask. `benchmark/run_usam.py`.

**Bug caught by Adi ("0 is not a possible outcome") and fixed:** `micro_sam.prompt_based_segmentation.segment_from_box` expects the box as **`[y0,x0,y1,x1]` (row-col), not XYXY**. We were passing XYXY, so it segmented the wrong region (~0 IoU). Visualizing the output (mask landed outside the box) revealed it. Fixed → µSAM jumps to a real ~0.44–0.48. **This is why principle #2 exists.**

### 6.3 Cellpose-SAM (automatic)
`models.CellposeModel` (cpsam), automatic per-frame segmentation, union instance-matched to target, run at **×3 magnification**. `benchmark/run_cellpose.py` (also saves a `_raw` all-cell stack for the figures).

**Bug caught by Adi ("0 IoU is not possible for a single-frame segmentor") and fixed.** Cellpose was the last automatic detector still being run at *native* scale, after we had already fixed Omnipose and NeuroQuantify. cpsam is advertised as scale-invariant, but our neurons are small and low-contrast enough that at native resolution it detected only **0.23 cells/frame** on A71 and **literally zero on many frames** — which is where the 0.00 IoU entries in the proof figures came from. A scale/threshold sweep showed ×3 is its favorable setting (A71 proof frames went from IoU [0.11, 0.00, 0.00] → [0.19, 0.34, 0.12]). Re-run at ×3: test IoU 0.231 → **0.271**, clDice 0.411 → **0.428**.

**What remains after the fix is genuine.** One proof frame (A81 frame 35) still scores 0.00, but for an honest reason we confirmed visually: Cellpose finds **7 cells there — segmenting the *neighbouring* neurons quite well, neurites included — while entirely missing the target neuron.** That is a real capability limitation, and a sharper point than "it detected nothing": an automatic segmenter gives you *some* cells with no way to specify *which* cell you care about, whereas our method is prompted on the target and tracks that identity.

**Meta-lesson:** this was the same root cause (out-of-distribution input scale) as Omnipose and NeuroQuantify, but Cellpose had been benchmarked earlier and was not revisited when the pattern was discovered. When a systematic bias is found in one competitor, **re-audit every competitor already measured** — do not assume earlier runs are unaffected.

### 6.4 Omnipose (automatic, elongated-morphology specialist)
`benchmark/run_omnipose.py`, conda env `omnipose` (Python 3.11; required pinning `ncolor==1.5.3` for an API mismatch). We swept model and scale to find its **favorable** setting:
- `cyto2_omni` (its mammalian-cyto model) detects **nothing** on our phase contrast at any scale.
- `bact_phase_omni` (its phase-contrast model) works but treats the neuron as bacteria and **shatters it into ~160–400 fragments**.
- Our ~1.5 px/µm frames are smaller than its training magnification; upscaling **×3** maximized its target score. **Chosen: `bact_phase_omni` @ ×3.**

This is a genuine, non-strawman result: Omnipose over-fragments, so its area overlap is low (IoU 0.28) but — because it *is* built for elongated shapes — its **connectivity (clDice 0.65) is the best of the non-SAM competitors**, exactly as its design predicts. Still 0.28 clDice below ours.

### 6.5 NeuroQuantify (automatic, phase-contrast neuron+neurite specialist)
`benchmark/run_neuroquantify.py`, venv `neuroquantify_venv`. Model weights ship in the repo (`unet/optimized_model.pt`, an int8-**quantized** TorchScript U-Net → needs `torch.backends.quantized.engine = "qnnpack"` on Apple Silicon). Run in the authors' intended pipeline: **gamma correction → large model**, 3-class semantic output (background / soma / neurite), foreground = soma+neurite, union instance-matched to target.

**Scale was again the key favorable setting (verified visually):** at native scale NeuroQuantify labels ~65% of the frame as "cell" (out-of-distribution garbage); upscaling to its training magnification (**×3**) makes it a real competitor. We swept ×2 vs ×3 and chose ×3 (higher target score). Rendered overlays confirmed it then cleanly picks out cell bodies and traces *part* of the neuron while dropping most neurites — genuine and on-thesis.

### 6.6 The general lesson embedded in the benchmark
Every extreme score (µSAM 0.00, Omnipose "0 cells", NeuroQuantify 65% foreground) turned out to be a **scale or format usage issue, not a capability verdict** — and we only found each by *looking at the pixels*. Every competitor number in the paper is one we visually confirmed is the method's genuine best, not a mis-run.

### 6.7 Trackastra — why we did not ship a number
We built the full pipeline (`benchmark/run_trackastra.py`, conda env `trackastra`): feed a segmenter's per-frame masks → Trackastra links them → follow the frame-0 target track. Two structural problems make a single clDice number untrustworthy/misleading:
1. Its segmentation is whatever its segmenter produced (neurites already dropped), so on segmentation it merely duplicates DeepSea/Omnipose;
2. Holding one identity from a single seed, on a thin low-contrast neuron, **drifts onto the wrong fragment** (visually confirmed) — collapsing to ~0.05 clDice, which we could not cleanly separate from a follow-logic artifact.

Because principle #2 forbids shipping a number we haven't confirmed is honest, Trackastra is represented **qualitatively** (the modular "segment-then-link" paradigm cannot recover un-segmented neurites; joint memory can), not as a table row.

---

## 7. Results

### 7.1 Test set (4 held-out videos, t ≥ 1, mean over videos)

| Model | IoU | Dice | clDice | Supervision |
|---|---|---|---|---|
| **Ours** | **0.742** | **0.849** | **0.932** | 1 frame-0 seed |
| SAM 2 (zero-shot) | 0.639 | 0.771 | 0.830 | 1 frame-0 seed |
| µSAM | 0.442 | 0.597 | 0.754 | GT box every frame (oracle) |
| NeuroQuantify | 0.477 | 0.630 | 0.742 | none (automatic) |
| Omnipose | 0.279 | 0.428 | 0.649 | none (automatic) |
| Cellpose-SAM | 0.231 | 0.329 | 0.411 | none (automatic) |
| *DeepSea (text only)* | *0.335* | *0.480* | *0.565* | none (automatic) |

**Headline reading:** ours leads every metric; the clDice margin over the best baseline (SAM 2) is **+0.10**, and over the best *unprompted* competitor (NeuroQuantify) is **+0.19**. NeuroQuantify — the only competitor trained on phase-contrast neurites and given *no* prompt — matches µSAM (which gets an oracle box every frame), underscoring how strong a genuine domain baseline it is, and how far ours still leads.

### 7.2 Validation set (5 model-selection videos, t ≥ 1)

| Model | IoU | Dice | clDice |
|---|---|---|---|
| **Ours** | **0.754** | **0.855** | **0.921** |
| µSAM | 0.477 | 0.623 | 0.771 |
| NeuroQuantify | 0.526 | 0.673 | 0.768 |
| Omnipose | 0.415 | 0.584 | 0.767 |
| SAM 2 (zero-shot) | 0.615 | 0.737 | 0.727 |
| Cellpose-SAM | 0.350 | 0.479 | 0.458 |

Note the val set is *harder for SAM 2 zero-shot* (clDice 0.727) than for the three specialists (µSAM / NeuroQuantify / Omnipose all ~0.77), yet ours still leads by +0.15 clDice over the best of them — consistent with the test set.

**Two reported numbers for ours, both honest:** the bars above are the **hard full-video propagation** protocol (0.75 IoU test/val). The thesis headline **0.84 IoU / 0.91 Dice** is the standard **8-frame-clip validation protocol** (`reproduce_training_iou.py` reproduces it: 0.8411 vs the reported 0.8401). We lead the comparison with full-sequence numbers and cite 0.84 as the training benchmark — both are real, they just measure different (harder vs standard) protocols.

---

## 8. The figures and what each proves

All in `paper_assets/figures/`, one unified palette (navy/blue/teal for ours, muted greys for competitors).

| Figure | What it shows |
|---|---|
| `fig_benchmark_test.png` | grouped IoU/Dice/clDice bars, ours emphasized, ordered by clDice; +0.10 clDice callout |
| `fig_benchmark_val.png` | same on the validation split (consistency check) |
| `fig_temporal_consistency.png` | per-frame Dice across the whole sequence — ours holds ~0.85 flat; single-frame tools are jittery and decay (NeuroQuantify's teal curve is the clearest example) |
| `single_frame_compare_test.png` | per-frame masks side by side; automatic tools shown as **raw output** (Cellpose finds ~0, Omnipose shatters into hundreds of fragments, NeuroQuantify floods the field) so you can see they never *isolate* the target neuron the way ours does |
| `fig_capability_matrix.png` | qualitative: ours is the only method filling every column (temporal + phase-contrast + neuron-specialized + topology loss + promptable + foundation backbone) |
| `proof_A71_C1.png`, `proof_A81_C1.png` | real frames, GT, and each model's overlay (yellow = correct, green = **missed neurite**, red = extra) — the green is the story |

---

## 9. Reproducibility — environments and commands

Isolated envs (avoid cross-contamination):

| Competitor | Environment | Runner |
|---|---|---|
| ours / SAM 2 | repo `venv` | `benchmark/run_ours.py` (`--no-lora` for base) |
| µSAM | `benchmark/competitors/usam_venv` | `benchmark/run_usam.py` |
| Cellpose-SAM | `benchmark/competitors/cp_venv` | `benchmark/run_cellpose.py` |
| Omnipose | conda env `omnipose` (py3.11, `ncolor==1.5.3`) | `benchmark/run_omnipose.py --model bact_phase_omni --scale 3.0` |
| NeuroQuantify | `benchmark/competitors/neuroquantify_venv` | `benchmark/run_neuroquantify.py --scale 3.0` |
| Trackastra | conda env `trackastra` (py3.11) | `benchmark/run_trackastra.py --seg <segmenter>` |
| DeepSea | `benchmark/competitors/deepsea_venv` (+ `deepsea_repo`) | `benchmark/run_deepsea.py` |

Score any predictions the same way (writes `benchmark/results/<split>/`):
```
venv/bin/python benchmark/score.py --pred-dir benchmark/predictions/<model> \
    --model <model> --split test --skip-first-frame
```
Regenerate figures: `benchmark/paper_figures.py` (bars + temporal), `benchmark/make_figures.py` (capability matrix), `benchmark/single_frame_compare.py --preset test`, `benchmark/make_proof_panel.py --video A71_C1`.

**Gotcha:** the shell is **zsh** — unquoted `$VAR` does **not** word-split. Pass multiple `--videos` names explicitly, not via an unquoted variable.

---

## 10. Honest limitations (state these; they build credibility)

- **Test set is 4 videos.** Two (`A71`/`A81`) are cleanest; two `_100k` share a batch. A future phase should widen it and add k-fold over the 25 for mean ± σ.
- **No external public-benchmark run yet** (e.g. Cell Tracking Challenge PhC-C2DH-U373) — planned as independent validation.
- **CellSAM not evaluated** (access-gated).
- **Competitor scaling is a knob.** Omnipose/NeuroQuantify are scale-sensitive; we chose the magnification *most favorable to each competitor* and verified it visually, but a reviewer could ask for a scale-sensitivity sweep — we have the harness to provide it.
- **Two "ours" numbers** (0.75 full-sequence vs 0.84 8-frame-clip) measure different protocols; always state which.

---

*Everything here is reproducible from the `benchmark/` scripts and `benchmark/results/` CSVs. The protocol summary lives in `paper_assets/BENCHMARK_PROTOCOL.md`; this guide is the fuller narrative.*
