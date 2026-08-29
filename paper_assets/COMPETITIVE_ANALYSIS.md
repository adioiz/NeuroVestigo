# Competitive analysis & positioning — neuron time-lapse segmentation + tracking

*Prepared for the paper. Every non-obvious factual claim is followed by a primary-source URL. Numbers about your own model come from `gpu-checkpoints/training_log.json` and `REPORT.tex` (Run 7). This document is the research backbone for the Introduction, Related Work, and Discussion sections, and for the figure/benchmark plan.*

---

## 0. TL;DR — the one-paragraph positioning

Your contribution is **not** "we LoRA-fine-tuned SAM 2" (crowded — see §4) and **not** "a specialist beats generalists on its niche" (expected). It is: **the first neuron-specialized, topology-preserving, temporally-consistent segmentation-and-tracking pipeline for live phase-contrast neuron time-lapse** — a setting where (a) foundation models fragment thin branching neurites and (b) no public dataset simultaneously covers phase-contrast + genuine neurons + tracking. The capability matrix (`fig7_capability_matrix.pdf`) makes this concrete: across six competitors, **only your method fills the "neuron/neurite specialized" and "topology-aware loss" columns**, and it is one of only two that also does native temporal tracking. Lead the abstract and Fig 1 with that capability gap, not with the model.

---

## 1. The honest novelty statement (put a version of this in the paper)

> **Loss vs. metric — keep them distinct.** Your training **loss** is **cbDice** (`SoftCBDiceLoss`, weight 5.0) + a **supervoxel connectivity loss**; clDice is only the predecessor cbDice extends (you do not train on clDice). Your reported **connectivity metric** is **clDice** — the independent, widely-recognized standard. Train-on-cbDice / report-clDice is deliberate and reviewer-proof: never headline the exact quantity you optimized. (cbDice's own MICCAI paper evaluates with clDice/Dice/Betti, not "cbDice-as-metric.")

**Genuinely novel (defensible):** To our knowledge, no prior work brings a **connectivity/topology loss (cbDice, which extends clDice) together with a supervoxel connectivity loss to bear on *live neuron time-lapse* segmentation coupled with a foundation model's temporal memory (SAM 2's memory-attention tracking)**. The topology losses that exist were all designed and validated on **static** targets — retinal/cerebral vessels, roads, EM membranes, and fixed 3D connectomics volumes — whose ground-truth topology is fixed. Neuron time-lapse changes the problem: connectivity must be preserved **across frames** as neurites extend, retract, and branch, and a "broken" vs. "genuinely retracted" neurite is a temporally-defined distinction. Enforcing centerline/connectivity topology **jointly with** SAM 2 cross-frame memory propagation — so a topologically-correct mask at frame *t* conditions frame *t+1* — is the specific contribution. Dual-rank LoRA+ is the delivery mechanism that makes this trainable on scarce annotated live-imaging data.

**NOT novel — credit it, don't claim it:**
- **LoRA/PEFT fine-tuning of SAM/SAM2 is crowded** — SAM2LoRA (fundus), SAM-OCTA2 (OCTA), SAMed (medical), µSAM, and a 2025 SAM2-microscopy bioRxiv all cover this ground, several on thin-vessel targets with near-identical composite losses (§4).
- **Each individual topology loss is prior art** — clDice (CVPR 2021), cbDice (MICCAI 2024), supervoxel connectivity loss (AAAI 2025). Cite, don't claim.
- **"Foundation model for microscopy segmentation + tracking" already exists** — µSAM (Nature Methods 2025).

**Framing rule:** claim the *combination + domain shift*, phrased as "to the best of our knowledge, based on a systematic search" — a search cannot exhaustively prove a negative.

---

## 2. Competitors — two tiers

### Tier A — generalist / foundation segmentation models

#### SAM 2 (base, Meta) — the backbone you build on
- "SAM 2: Segment Anything in Images and Videos," Ravi et al., Meta FAIR, arXiv:2408.00714 (Aug 2024). https://arxiv.org/abs/2408.00714 · code (Apache-2.0): https://github.com/facebookresearch/sam2
- **Task:** promptable, class-agnostic instance segmentation, images **and** video, 2D. Native **temporal tracking** via streaming memory (memory bank + memory attention + memory encoder).
- **Domain:** trained on natural images/video (SA-1B + SA-V) — **no microscopy, no phase-contrast, no neurons**. Clean domain gap.
- **Topology:** none. Loss = focal + dice + IoU + occlusion; single connected blob per prompt. The paper itself admits SAM 2 "struggles with accurately tracking objects with very thin or fine details." https://arxiv.org/html/2408.00714v1
- **Reported (natural-video VOS, SAM 2.1 Large J&F):** SA-V test 79.5, MOSE val 74.6, LVOS v2 80.6. https://github.com/facebookresearch/sam2
- **Differentiation:** SAM 2 supplies the tracking backbone but has zero microscopy training and no connectivity constraint → on low-contrast phase-contrast neurites it collapses thin arbors into a blob or drops faint distal processes, and needs a prompt per object. Your work adds domain fine-tuning + topology objectives **on top of** its memory propagation.

#### µSAM / micro-sam — closest microscopy-native promptable competitor
- "Segment Anything for Microscopy," Archit et al., **Nature Methods 22:579–591 (2025)**, DOI 10.1038/s41592-024-02580-4. https://www.nature.com/articles/s41592-024-02580-4 · code (MIT): https://github.com/computational-cell-analytics/micro-sam
- **Task:** interactive + automatic instance segmentation (light & EM), 2D/3D, napari plugin. **Has a tracking annotator.**
- **Critical nuance for your differentiation:** it is built on **SAM 1, not SAM 2**; tracking = "each frame is first segmented individually and then tracked with **Trackastra**" (a separate graph linker), **not** native video mask propagation. SAM2 support is roadmap-only. https://computational-cell-analytics.github.io/micro-sam/micro_sam.html
- **Domain:** trained on >17k microscopy images incl. **LIVECell (phase-contrast, incl. SH-SY5Y neuronal-*like* cells)** and 8 CTC datasets — so it *has* seen phase-contrast, but its automatic head (UNETR + distance-to-centroid + watershed) assumes **blob-like objects**, not neurite arbors. No topology loss.
- **Reported:** primary metric mean segmentation accuracy (mSA, IoU 0.5–0.95). LIVECell AIS specialist mSA ≈ 0.39; on par/better than Cellpose on several LM sets, below Cellpose on TissueNet. (Exact per-dataset interactive tables behind paywall.) https://www.nature.com/articles/s41592-024-02580-4
- **Differentiation:** blob-oriented automatic head; tracking is per-frame-segment-then-link on SAM 1; **no topology preservation**. Your method: native SAM 2 temporal propagation + neurite topology losses + neuron specialization.

#### Cellpose-SAM (+ Cellpose 3) — strongest single-frame generalist
- "Cellpose-SAM: superhuman generalization for cellular segmentation," Pachitariu, Rariden, Stringer (Janelia), **bioRxiv 2025**, DOI 10.1101/2025.04.28.651001. https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1 · code (BSD-3; data CC-BY-NC): https://github.com/MouseLand/cellpose
- Cellpose 3: **Nature Methods 22:592–599 (2025)**, DOI 10.1038/s41592-025-02595-5. https://www.nature.com/articles/s41592-025-02595-5
- **Task:** automatic (prompt-free) instance segmentation via flow fields; keeps only SAM's **ViT-L encoder**, discards SAM's decoder; 256×256 tiles. **Single-frame — no tracking.**
- **Domain:** generalist across label-free/fluorescence/histology incl. phase-contrast-like data. **"Superhuman"** = error 0.163 vs inter-human 0.257, approaching consensus bound 0.128 (verify decimals against the bioRxiv PDF).
- **Topology & the ONE correction you must apply:** ⚠️ **Do NOT claim Cellpose has a "round-cell bias."** The Cellpose paper explicitly segments "cells with complex protuberances such as elongated dendrites" (https://www.nature.com/articles/s41592-020-01018-x). The correct, defensible framing: the **center-convergent flow representation degrades on *very* elongated/branched cells (long neurons, muscle fibers)** — cite **Omnipose** (Nature Methods 2022, https://www.nature.com/articles/s41592-022-01639-4), which was built precisely because of that failure. Cellpose-SAM's own docs list "long neurons or skeletal muscle fibers" as a remaining struggle. No topology loss.
- **Differentiation:** prompt-free, non-temporal, topology-agnostic. You add tracking + connectivity + neuron specialization.

### Tier B — task relatives (cell-specific / seg+track)

#### DeepSea — your CLOSEST task relative (benchmark against this)
- Zargari et al., **Cell Reports Methods 3(6):100500 (2023)**, DOI 10.1016/j.crmeth.2023.100500. https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(23)00129-7 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10326378/ · code (MIT): https://github.com/abzargar/DeepSea · dataset (⚠️ **CC-BY-NC-ND**): https://deepseas.org/datasets/
- **Task:** single-cell instance **segmentation + tracking** in **2D phase-contrast time-lapse** — same imaging regime as you, and it landed in **Cell Reports Methods** (your realistic target tier).
- **Architecture:** small residual U-Net (~1.9M params) + auxiliary edge branch + a **separate tracker U-Net** (appearance-matching), *not* a foundation model, *not* promptable, *not* memory-propagation.
- **Domain:** mESC, bronchial epithelial, C2C12 muscle — **compact near-convex bodies, NO neurons, no neurites**. No topology/connectivity loss (edge branch splits touching blobs, doesn't preserve elongated connectivity).
- **Reported (its own held-out phase-contrast test set):** seg IoU **0.90 ± 0.2**, tracking precision **0.98**, mitosis precision **0.89**, full cell-cycle **MOTA 0.94** (vs TrackMate 0.29). Released a public benchmark: **3,624 images / 47 sequences + 228 tracking trajectories**.
- **Differentiation & use:** it is the **strongest published phase-contrast seg+track system with a public benchmark** → cite as nearest prior art, and **run your pipeline on its dataset** for an apples-to-apples number (note the domain gap: their cells are non-neuronal blobs). You differ on: neurites, topology loss, foundation-model transfer, prompt-based propagation.

#### CellSAM — closest architectural relative (SAM foundation model for cells)
- Israel/Marks et al., Van Valen Lab, **Nature Methods (2025)**, DOI 10.1038/s41592-025-02879-w (bioRxiv 2023, arXiv:2311.11004). https://www.nature.com/articles/s41592-025-02879-w · https://pmc.ncbi.nlm.nih.gov/articles/PMC12695629/ · code: https://github.com/vanvalenlab/cellsam
- **Task:** automatic 2D instance segmentation. **Single-frame — no built-in tracking** (delegated to DeepCell tracking downstream).
- **Architecture:** SAM **ViT-B** + **CellFinder** (Anchor-DETR detector) emitting bbox prompts; also manually promptable.
- **Domain:** fluorescence/brightfield/phase-contrast/mass-cytometry — phase-contrast is **one of many** modalities, non-neuronal; neurons only as aspirational future work. No topology loss.
- **Reported:** F1 @ IoU≥0.5; LIVECell zero-shot F1 0.13→0.40; parity with human annotators on 4 categories. Public dataset at https://cellsam.deepcell.org.
- **Differentiation:** single-frame, blob-instance via bbox prompts (geometry ill-suited to thin branching neurites), not neuron-specialized. Cite as the "SAM-transfers-to-microscopy" baseline; you add temporal tracking + neurite topology + phase-contrast/neuron specialization.

---

## 3. Topology / connectivity loss cluster — prior art to CREDIT

| Loss | Paper | What it does | Validated on |
|---|---|---|---|
| **clDice** | Shit et al., **CVPR 2021**, arXiv:2003.07311 | Soft-skeleton centerline Dice; guarantees topology preservation up to homotopy; O(kn) | vessels, roads, **neurons** (2D/3D, static) |
| **cbDice** | Shi et al., **MICCAI 2024**, arXiv:2407.01517 | Extends clDice with **EDT radius weighting** for geometry/diameter awareness | vascular (Circle of Willis / TopCoW) |
| **Supervoxel connectivity loss** | Grim, Chandrashekar, Sumbul (Allen Inst.), **AAAI 2025**, arXiv:2501.01022 | Penalizes **split/merge** via "critical components"; extends "simple point" to multi-voxel sets; O(n) | fixed 3D LM mouse brain (EXASPIM), DRIVE, ISBI12, CrackTree |

Sources: https://arxiv.org/abs/2003.07311 · https://arxiv.org/abs/2407.01517 · https://arxiv.org/abs/2501.01022 · also cite **clDice** as both a *loss* and the *metric* that exposes connectivity errors IoU/Dice miss.

⚠️ **"MNSeg" could not be verified** — repeated searches found no paper by that acronym for neurite segmentation. **Do not cite it without the exact title/authors/venue.** If you meant a verifiable neurite benchmark, candidates: **FISBe** (arXiv:2404.00130, thin filamentous neuron instances), **Fluorescent Neuronal Cells v2** (Sci Data 2024), **SNEMI3D / CREMI** (EM neurite). Confirm which you intended.

**Gap confirmed:** none of these losses has been applied to *live neuron time-lapse* + a foundation model's temporal memory. That is your whitespace.

---

## 4. Why "LoRA-fine-tune SAM2" is not novel (be upfront)

- **SAM2LoRA** — LoRA on SAM2 encoder+decoder for **retinal fundus** vessels/optic-disc, composite BCE+SoftDice+FocalTversky, <5% params. arXiv:2510.10288. https://arxiv.org/abs/2510.10288
- **SAM-OCTA2** — LoRA/adapter fine-tune of SAM2 for **OCTA retinal vessels**. arXiv:2409.09286. https://arxiv.org/pdf/2409.09286
- **SAMed** — canonical LoRA-on-SAM medical baseline (your lineage).
- **Lightweight SAM2 microscopy fine-tuning** — bioRxiv 2025.11.08.687405; notably reports **LoRA underperformed full fine-tuning** for their cell task (useful nuance). https://www.biorxiv.org/content/10.1101/2025.11.08.687405v1.full

→ Position LoRA+ and the loss functions as **adopted, integrated components**. Reserve the novelty claim for the topology + temporal + neuron combination.

---

## 5. Datasets — the gap IS an opportunity (dataset/Resource angle)

**Key finding: no public dataset is simultaneously phase-contrast + genuine neurons + tracking.**
- **LIVECell** (Nature Methods 2021): phase-contrast, includes **SH-SY5Y** (neuronal-*like*, the hardest class), but **segmentation only — no tracking**. https://www.nature.com/articles/s41592-021-01249-6
- **DeepSea / Ker et al. 2018** (Sci Data): phase-contrast **seg+track**, but non-neuronal cells. https://www.nature.com/articles/sdata2018237
- **CTC**: has phase-contrast time-lapse tracking (PhC-C2DH-U373, PhC-C2DL-PSC) but **no cultured-neuron set**; "Fluo-N2DH…" = *fluorescent Nuclei*, not neurons.
- **Neuron datasets** (DeepNeuron, BigNeuron, DIADEM, FISBe): mostly **static 3D morphology reconstruction**, not live tracking.

→ **This validates two things:** (1) your Introduction's motivation ("a truly neuronal live-imaging seg+track benchmark does not exist"), and (2) the **dataset-as-Resource strategy** that raises your journal ceiling (§7). Releasing even a curated subset of your ~20→150 sequences fills a real gap.

---

## 6. Benchmark plan — strawman-proof, two-tier, with a fair CTC protocol

### 6.1 Validate on an independent PUBLIC benchmark (non-negotiable)
Use **CTC PhC-C2DH-U373** (glioblastoma-astrocytoma, **cells WITH processes — closest to neurons**): 520×696, 115 frames, 2 train + 2 test sequences, gold+silver truth. **Watch the two leaderboards:** the segmentation-only benchmark (CSB) top SEG reaches ≈ 0.93, but on the **tracking benchmark (CTB) top SEG saturates around 0.74** while TRA is near-solved (≈ 0.98). So on the neuron-like set, **SEG is the hard, unsaturated axis your topology losses target** — lead with it. Optionally add **PhC-C2DL-PSC** (dense, hard SEG ≈ 0.84) and **BF-C2DL-MuSC** (elongated). Note CTC SEG (Jaccard, >0.5 match, averaged over sparse GT) is **stricter and not directly comparable** to your in-house val IoU 0.84. https://celltrackingchallenge.net/2d-datasets/ · https://celltrackingchallenge.net/latest-ctb-results/
- Metrics: **SEG** (Jaccard, match rule `|R∩S| > 0.5|R|`, mean over reference objects), **DET**, **TRA** (AOGM). Use the official evaluator **py-ctcmetrics**: https://github.com/CellTrackingChallenge/py-ctcmetrics

### 6.2 The fair protocol for a promptable single-object model (report BOTH)
1. **Oracle / GT-prompted upper bound** — iterate over GT instances, derive a box (and/or point) prompt per object, run one prediction each, paste single-object masks into a labeled image, score SEG. **Label it explicitly "GT-derived prompts (oracle localization)"** — it isolates *mask quality*, not detection, and is NOT comparable to CTC leaderboard entries (which see only raw images). This is µSAM's interactive regime.
2. **Fully-automatic pipeline** — no GT leakage (grid/AMG prompts, or a detector→prompt→SAM pipeline); score DET/SEG/TRA so it *is* leaderboard-comparable.
- Always separate the two; report per-dataset (not just averages); pair sparse U373 with a dense set so SEG reflects detection difficulty, not just boundary quality. (Template: µSAM's regime split; SAM2-on-CTC arXiv:2509.09943.)

### 6.3 The two-tier competitor comparison (Fig 2 + Fig 3)
- **Tier 1 — zero-shot generalists:** base SAM 2, µSAM, Cellpose-SAM, CellSAM, DeepSea.
- **Tier 2 — same models fine-tuned on your data** (where feasible).
- **Metrics:** IoU, Dice, **clDice** (clDice is the panel that exposes connectivity failures IoU/Dice hide — your headline advantage).
- **Protocol:** identical prompts for promptable models; documented instance-matching for auto models (reduce multi-instance output to the target neuron before scoring); same splits/resolution; **mean ± std across sequences**; evaluate on both your data and CTC.

---

## 7. Journal ladder — honest tiering (submit top-down, cascade on rejection)

Calibration: Nature Methods is the home of Cellpose ×3, µSAM, CellSAM, LIVECell, TrackMate 7, Ultrack. **DeepSea — the closest scope match to your tool — landed in Cell Reports Methods, not Nature Methods.** SAM2-for-cell-video is still **preprint-only** (an open field).

| Journal | ~IF (verify) | Presub? | Realistic for this tool | Notes |
|---|---|---|---|---|
| **Nature Methods** | ~28 (conflicting) | **Yes, encouraged** | Reach — best odds as a **Resource/dataset** | Do a free presubmission enquiry; needs broad multi-lab utility |
| Nature Communications | ~15.7 | No | Unlikely as narrow tool | Needs broad-impact framing |
| Nature Machine Intelligence | ~18.4 | No | No | Wants ML novelty, not application |
| **Cell Reports Methods** | ~4.5 | **Yes, encouraged** | **Yes — aspirational, do a presub** | Where DeepSea (your task family) landed |
| Medical Image Analysis | ~10–14 | Not standard | Competitive; scope risk | Medical-leaning; needs method novelty |
| **Communications Biology** | ~5.1–5.8 | No | **Yes — strong realistic target** | Rewards soundness+utility over leap |
| IEEE TMI | ~9.8 | No | No | Medical + high method-novelty bar |
| **Bioinformatics (App. Note)** | ~5.5 | Not formal | **Yes — cleanest software home** | Needs robust public release |
| **Scientific Data** | ~6.9 | Not formal | **Yes — for the dataset only** | Data Descriptor; split off the dataset |

**Recommended sequence:** (1) **Nature Methods presubmission enquiry** framed as a neuron seg+track **Resource** (free scope read); then submit top-down → (2) **Cell Reports Methods** (presub) → (3) **Communications Biology** → (4) **Bioinformatics Application Note**. In parallel, **split the annotated dataset into a Scientific Data Data Descriptor** — an easy, high-value paper you cite from the tool paper, and the single biggest lever on your ceiling.

⚠️ Impact factors above come from third-party aggregators that disagree — **re-verify on each journal's own metrics page / Clarivate JCR before quoting in a cover letter.**

---

## 8. Critical gaps — what must exist BEFORE the impressive figures are real

The centerpiece figures (Fig 2 qualitative grid, Fig 3 clDice bars) **cannot be real today.** Prerequisites, in priority order:

1. 🔴 **`eval_masks.py` does not exist** in either repo. You believe you have "a working eval_masks.py that computes IoU/Dice/clDice" — it is not on disk, and clDice appears only as a *loss* in `src/loss_fns.py`, never as a computed *metric*. **No clDice number exists anywhere yet.** Recreating this is the #1 blocker.
2. 🔴 **No competitor has been run.** REPORT.tex has zero baselines. The two-tier comparison is entirely unbuilt.
3. 🔴 **No CTC evaluation** set up.
4. 🟠 **k-fold** over the 20 videos for IoU/Dice/clDice **mean ± σ** (currently one fixed split; the report flags split dependency on A91_C1_2).
5. 🟠 **MNSeg phantom citation** — verify or replace before any draft circulates.
6. 🟠 **Dataset license** — confirm you can release your data; note DeepSea data is CC-BY-NC-ND and Cellpose data CC-BY-NC (don't redistribute derivatives).

**Until (1)–(3) exist, do not fabricate competitor numbers.** Figs 2 & 3 stay as layout templates.

---

## 9. Figures — see `FIGURE_PLAN.md`
Real now (from your data): `fig1`–`fig7` in `paper_assets/figures/`. The capability matrix (`fig7`) and training-journey/ablation (`fig1`) are paper-ready; the qualitative grid and clDice-benchmark bars need the §8 pipeline first.
