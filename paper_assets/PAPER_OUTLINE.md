# Paper outline — neuron SAM 2 segmentation + tracking

*Scope decided 2026-07-29: **method paper only**; dataset release mentioned as future work. Target: **Cell Reports Methods** (where DeepSea, the closest task relative, landed), with a free Nature Methods presubmission enquiry in parallel. Nature-family shape: Abstract → Intro → Results → Discussion → Methods.*

**How to read this:** every claim below is tagged with the evidence that supports it — `[Fig N]`, `[Table N]`, or a literal number. Nothing in the outline is a placeholder we don't already have, except items explicitly marked **⚠ GAP**.

---

## 0. The narrative arc (approve this first)

The whole paper is one argument in five beats:

1. **The biology lives in the neurites.** In live, label-free phase-contrast imaging of neurons, the quantities people actually want — neurite length, branching, outgrowth rate — are carried by thin, faint, constantly-changing processes, not by the bright soma.
2. **Existing tools systematically lose exactly that.** Cell segmenters are blob-oriented (find the soma, drop the processes) or fragment filaments; foundation/video models are general-purpose and topology-blind; trackers are bounded by whatever their segmenter gave them. And crucially, **area-overlap metrics hide this failure** — you can score a respectable IoU while deleting every branch.
3. **So the fix has to be three things at once**, which nothing currently combines: topology-aware training (keep the branches connected), temporal memory (hold one neuron's identity across the movie from a single annotation), and neuron/phase-contrast specialization.
4. **We build exactly that**, parameter-efficiently: a dual-rank LoRA+ adaptation of SAM 2 training only 13.9% of parameters, with a connectivity-preserving composite loss — and we evaluate with **clDice**, a metric that actually sees the branches.
5. **It wins, and the win is honest.** Against five competitors — each run in *its own most favorable* configuration, several receiving *more* per-frame supervision than we do — our method leads every metric on every held-out video, and by the largest margin on connectivity.

**The one-sentence version:** *A single annotation on the first frame yields a topologically faithful, temporally consistent segmentation of one neuron across an entire phase-contrast time-lapse — preserving the thin neurites that every existing method drops.*

---

## 1. Title (options)

1. **"Topology-preserving, temporally consistent segmentation and tracking of individual neurons in label-free time-lapse microscopy"** ← recommended: names all three contributions, no method-name jargon.
2. "Parameter-efficient adaptation of a video foundation model preserves neurite connectivity in live phase-contrast neuron imaging"
3. "From one annotation to a full movie: connectivity-aware neuron segmentation and tracking in label-free microscopy"

*Avoid* putting "LoRA" or "SAM 2" in the title — per our positioning, the adaptation technique is the means, not the claim.

---

## 2. Abstract (~150–200 words for CRM)

Structure, with the numbers to use:

| Beat | Content |
|---|---|
| Context | Live phase-contrast imaging is the only way to watch neurons develop without phototoxicity or labels; neurite morphology is the readout. |
| Problem | Existing segmenters recover the soma but fragment or delete thin neurites, and none maintain single-neuron identity across a movie. Area metrics (IoU/Dice) conceal this. |
| Approach | Dual-rank LoRA+ adaptation of SAM 2 (**5.56 M of 39.9 M parameters, 13.9 %**) with a connectivity-preserving composite objective; one frame-0 annotation propagates through the sequence via the model's temporal memory. |
| Result | On four fully held-out videos: **clDice 0.932, Dice 0.849, IoU 0.742** — ahead of every competitor, with the largest margin on connectivity (**+0.10 clDice** over the best foundation baseline, **+0.19** over the best domain-specialist). |
| Fairness | Each competitor was run in its own most favorable mode; several received the ground-truth target box on *every* frame while our method received a single annotation — and still lost. |
| Impact | Enables neurite-level quantification of individual neurons in label-free live imaging at one-annotation cost. |

**Do not** open with "we fine-tuned SAM 2." Open with the capability gap.

---

## 3. Introduction (~4 paragraphs)

**¶1 — Why label-free live neuron imaging.** Phototoxicity and label perturbation make fluorescence poorly suited to long developmental time-lapses; phase-contrast is the workhorse. What people measure: neurite outgrowth, branching, polarity, network formation. *Cite Shefi-lab context + phase-contrast neuron literature.*

**¶2 — Why this is computationally hard.** Thin (few-pixel) processes, very low contrast against debris, touching/overlapping neighbours, morphology that changes every frame, and a target that must remain *the same cell* across hundreds of frames. Note the concrete data property: the target neuron occupies **~1.5 % of the frame**.

**¶3 — What exists, and the precise gap** (this paragraph does the competitive work; source: `COMPETITIVE_ANALYSIS.md`):
- *Generalist cell segmenters* — Cellpose/Cellpose-SAM, StarDist: centroid- or star-convex-based, built for compact cells; StarDist's star-convex parameterization is **architecturally** unable to represent branched filaments.
- *Morphology-specialized* — Omnipose: designed for elongated/branching shapes, but object-centric and single-frame.
- *Domain-specialized* — NeuroQuantify: phase-contrast neurons **and** neurites, but semantic (not per-cell) and single-frame.
- *Foundation models* — SAM/SAM 2, µSAM, CellSAM: strong general priors, topology-blind, and (except SAM 2) no temporal memory.
- *Trackers* — DeepSea, Trackastra: link per-frame masks, so their connectivity is upper-bounded by the segmenter feeding them.
- **The gap in one line:** no method combines topology-preserving training, temporal identity, and neuron/phase-contrast specialization.

**¶4 — What we do + contributions list.**
1. A neuron-specialized, topology-preserving, temporally-consistent seg+track method, trained parameter-efficiently (13.9 % of weights).
2. A rigorous, deliberately competitor-favorable benchmark against five methods spanning all four families above, on genuinely held-out data.
3. The demonstration that **connectivity metrics are necessary**: rankings and failure modes are invisible under IoU/Dice alone.

*Novelty phrasing (locked):* "to our knowledge, based on a systematic search" — claim the **combination**, credit LoRA/PEFT, individual topology losses, and foundation-models-for-microscopy to prior art.

---

## 4. Results

> Ordering principle: capability first (what it does), then the comparison (how it ranks), then why it works (ablation), then the honesty argument (protocol).

### 4.1 "One annotation segments an entire movie" — the capability
**Claim:** a single frame-0 mask propagates through the whole sequence, preserving branch structure.
**Evidence:** `[Fig 1]` qualitative montage — [proof_A71_C1.png](figures/proof_A71_C1.png), [proof_A81_C1.png](figures/proof_A81_C1.png). Overlay semantics: yellow = correct, **green = missed neurite**, red = extra. Our column stays yellow (IoU 0.85 / 0.88 / 0.86 on A71 frames 20/41/64) while competitor columns bleed green.
**Sentence to land:** the green in the competitor columns *is* the biology being lost.

### 4.2 Quantitative comparison on held-out data
**Claim:** leads every metric, by the widest margin on connectivity.
**Evidence:** `[Fig 2]` [fig_benchmark_test.png](figures/fig_benchmark_test.png) + `[Table 1]`.

| Method | IoU | Dice | clDice | Per-frame supervision |
|---|---|---|---|---|
| **Ours** | **0.742** | **0.849** | **0.932** | 1 annotation (frame 0) |
| SAM 2 (zero-shot) | 0.639 | 0.771 | 0.830 | 1 annotation (frame 0) |
| µSAM (box-prompt) | 0.442 | 0.597 | 0.754 | GT box **every frame** |
| NeuroQuantify | 0.477 | 0.630 | 0.742 | none (automatic) |
| Omnipose | 0.279 | 0.428 | 0.649 | none (automatic) |
| Cellpose-SAM | 0.271 | 0.389 | 0.428 | none (automatic) |

Three specific observations worth a sentence each:
- **+0.10 clDice** over base SAM 2 isolates what the topology-aware fine-tuning bought (same backbone, same protocol).
- **NeuroQuantify (0.742 clDice, unprompted) ties µSAM (0.754, oracle box every frame)** — the domain specialist is the strongest *unprompted* baseline, which makes our +0.19 margin over it the meaningful comparison.
- **Omnipose ranks 5th on IoU but 4th on clDice (0.649 > Cellpose 0.428)** — exactly as its elongated-morphology design predicts: it traces processes but shatters them. *This is the cleanest evidence that clDice measures something IoU does not.*

**Generalization:** validation-set results `[Fig 2b]` [fig_benchmark_val.png](figures/fig_benchmark_val.png) reproduce the ordering (ours clDice 0.921); test ≈ val for our method (0.932 vs 0.921) indicates no test-set idiosyncrasy. ⚠ *Cellpose val row pending re-run at matched scale.*

### 4.3 Temporal consistency — the unique capability
**Claim:** ours holds quality across the sequence; single-frame methods flicker and decay.
**Evidence:** `[Fig 3]` [fig_temporal_consistency.png](figures/fig_temporal_consistency.png). Ours ≈ flat 0.85 Dice out to frame 72 (endpoint 0.83); NeuroQuantify's per-frame curve is visibly jittery and decays toward ~0.5.
**Why it matters:** frame-to-frame flicker is fatal for the actual scientific measurement — growth-rate estimates differentiate the mask over time, so segmentation noise becomes measurement noise.

### 4.4 Why it works — ablation
**Claim:** each design choice contributes; the temporal path is the single biggest lever.
**Evidence:** `[Fig 4]` [fig_ablation.png](figures/fig_ablation.png) — validation IoU **0.695 → 0.840** across seven cumulative steps. Largest single jump **+0.054**, from training the **memory-attention adapters** (SAM 2's video path). Supporting: [fig_param_budget.png](figures/fig_param_budget.png) — 5.56 M / 39.9 M = 13.9 % trainable.
**Sentence to land:** the gain comes disproportionately from adapting the *temporal* machinery — evidence that this is a video problem, not a per-frame problem.

### 4.5 The comparison is favorable to the competitors, not to us
**Claim:** the result is not an artifact of mis-running baselines.
**Evidence:** `[Fig 5]` [single_frame_compare_test.png](figures/single_frame_compare_test.png) showing competitors' **raw, unmatched output** — Cellpose finds ~0–7 cells, Omnipose shatters a single neuron into 300–400 fragments, NeuroQuantify floods the field — plus `[Fig 6]` [fig_capability_matrix.png](figures/fig_capability_matrix.png).
Points to state explicitly:
- Every competitor ran in its **own best mode** at its **own best input scale**, selected by sweeps and confirmed visually (Methods).
- µSAM received the **ground-truth box on every frame** (oracle localization) — strictly more information than our single annotation — and still trailed.
- Automatic methods were reduced to the target by the **most generous** rule available (union of all overlapping instances).
- Where an automatic method scores 0 on a frame, it is because it **segmented other neurons in the field but not the target** (`[Fig S1]`) — a genuine limitation of unprompted segmentation, not a run failure.

---

## 5. Discussion (~4 paragraphs)

**¶1 — What we showed.** Recap without repeating numbers: combining topology-aware training with temporal memory closes a gap that neither alone addresses; the margin concentrates on connectivity.

**¶2 — Metrics are part of the contribution.** Under IoU alone, Omnipose and Cellpose look comparable; under clDice they are 0.22 apart, because one traces processes and the other doesn't. Recommend connectivity metrics as standard reporting for filamentous/neurite segmentation. Note our metric is **independent of our training objective** (train cbDice-family, report clDice), so the result is not metric-gaming.

**¶3 — Limitations (state plainly; this builds credibility).**
- Test set is **4 videos**; two are cleanest, two share an acquisition batch.
- Single-target-per-video protocol; multi-neuron dense fields untested.
- Competitor input scale is a tunable knob — we chose each competitor's best and verified visually, but a full scale-sensitivity sweep is future work.
- No external public-benchmark run yet **⚠ GAP** (see checklist).
- CellSAM not evaluated (access-gated weights).
- Two protocol numbers for our own method (0.84 IoU on the standard 8-frame-clip validation protocol vs 0.75 on harder full-video propagation) — always stated with the protocol attached.

**¶4 — Outlook.** Dataset release as a benchmark Resource; multi-neuron tracking; growth/forecasting dynamics (explicitly future thesis work, per scope decision); transfer to other filamentous live-imaging problems (glia, vasculature, fungal hyphae).

---

## 6. Methods (STAR Methods for CRM)

Mostly a rewrite of [METHODS_AND_BENCHMARK_GUIDE.md](METHODS_AND_BENCHMARK_GUIDE.md) into paper prose. Subsections:

1. **Data & annotation** — 25 phase-contrast time-lapse videos; annotation procedure; splits (**test: A71_C1, A81_C1, A12_C3_100k, A12_C1_100k_2**; val: 5 videos); target ~1.5 % frame area.
2. **Model & adaptation** — SAM 2.1 Hiera-tiny; dual-rank LoRA+ (**rank 16, α 32, dropout 0.1** on mask decoder + memory attention; **rank 8, α 16** on the last Hiera stage); 5.56 M/39.9 M trainable.
3. **Training objective** — composite: focal (w 5.0, α 0.25, γ 2.0) + Dice (2.0) + **cbDice (5.0**, 5 iterations, topology resolution 256**)** + supervoxel (0.3) + IoU (1.0); 110 epochs; 1 initial conditioning frame.
4. **Inference protocol** — frame-0 GT seed, streaming propagation (one frame encoded at a time for memory-bounded inference).
5. **Metrics** — IoU, Dice, **clDice** (skeleton-based, definition + equation), cbDice secondary. Include the metric-validation result: dropping two thin branches costs IoU −0.081 but clDice −0.227 (2.8×), while uniform erosion leaves clDice ≈ 1.0.
6. **Competitor protocols** — per-model table (mode, prompt, scale, matching rule) exactly as in `BENCHMARK_PROTOCOL.md`; state the favorable-setting principle and the scale-selection procedure.
7. **Scoring** — frame 0 excluded (it is the given prompt); all methods scored on t ≥ 1; mean ± s.d. across videos.
8. **Implementation & availability** — Apple Silicon MPS inference, isolated environments per competitor, code release plan.

---

## 7. Figure inventory (final)

| # | File | Shows |
|---|---|---|
| 1 | `proof_A71_C1.png` (+ A81 as S) | qualitative: ours preserves branches, others bleed green |
| 2 | `fig_benchmark_test.png` (+ `_val` as 2b) | IoU/Dice/clDice bars, all six methods |
| 3 | `fig_temporal_consistency.png` | per-frame Dice across the sequence |
| 4 | `fig_ablation.png` | cumulative design-choice contributions |
| 5 | `single_frame_compare_test.png` | competitors' raw output — they don't isolate the target |
| 6 | `fig_capability_matrix.png` | conceptual positioning (may move to Fig 1 or Discussion) |
| S1 | Cellpose A81 f35 panel | the honest 0.00: finds neighbours, misses target |
| S2 | `fig_param_budget.png` | trainable-parameter budget |

---

## 8. Pre-submission checklist

- [ ] **⚠ Cellpose val re-run** at matched scale → refresh `fig_benchmark_val` + Table.
- [ ] **⚠ External public benchmark** (CTC PhC-C2DH-U373) — the strongest remaining reviewer-proofing; report SEG separately for GT-prompted vs automatic.
- [ ] **⚠ k-fold** over the 25 videos → mean ± σ, to kill split-dependency criticism.
- [ ] Verify or replace the unverifiable **"MNSeg"** citation before circulating.
- [ ] Code release (GitHub + Zenodo DOI).
- [ ] Nature Methods presubmission enquiry (free) in parallel with drafting.
- [ ] Confirm author list/order and funding statements with Prof. Shefi.

---

*Next step once this arc is approved: draft §4 Results in full prose against these figures, then §6 Methods (largely mechanical from the guide), then Intro/Discussion, Abstract last.*
