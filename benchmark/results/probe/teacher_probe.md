# Teacher probe — does the distillation teacher hold transferable knowledge?

Produced by `benchmark/probe_teacher.py`. Each frame is decoded three ways in
the frame-0 conditioning regime, to separate two abilities the distillation
motivation conflated:

- **correct GT** — the teacher exactly as distillation uses it
- **WRONG mask** — a different frame's GT. If the score follows it, the decoder
  is refining what it is handed rather than segmenting.
- **NO prompt** — standalone ability. This is the number that matters: near
  zero means there is no knowledge to transfer, at any distillation weight.

## checkpoints-16-frames — the shipped baseline (no prompt dropout, no distillation)

| video | frame | correct GT | WRONG mask | NO prompt |
|---|---|---|---|---|
| A91_C1_2 | 0 | 0.9052 | 0.2002 | 0.0023 |
| A91_C1_2 | 20 | 0.8756 | 0.4155 | 0.0141 |
| A91_C1_2 | 40 | 0.8663 | 0.1786 | 0.0119 |
| A31_C1 | 0 | 0.9544 | 0.3159 | 0.0029 |
| A31_C1 | 20 | 0.9617 | 0.6192 | 0.0051 |
| A31_C1 | 40 | 0.9614 | 0.3100 | 0.0126 |
| A52_C2 | 0 | 0.9760 | 0.5232 | 0.0033 |
| A52_C2 | 20 | 0.9678 | 0.4132 | 0.0109 |
| A52_C2 | 40 | 0.9721 | 0.5073 | 0.0205 |
| **MEAN** | | **0.9378** | **0.3870** | **0.0093** |

**Reading.** Unprompted the decoder scores 0.0093 — it cannot find the neuron
on its own. Handed a wrong mask it segments the wrong thing (0.387). So its
0.938 with the correct mask is the ground truth being handed to it and cleaned
up: the teacher is a mask REFINER, not a segmenter, and its entire advantage is
the answer itself. There is nothing for the student to acquire, which is why
the t0-t1 gap stayed flat across runs at alpha 36% and alpha 10%.

This checkpoint trained WITHOUT prompt dropout, so it was never required to
segment unprompted — that is part of why the NO-prompt column is so low.

## checkpoints-distill-v2 — trained WITH prompt dropout p=0.4

| video | frame | correct GT | WRONG mask | NO prompt |
|---|---|---|---|---|
| A91_C1_2 | 0 | 0.9081 | 0.1945 | 0.0000 |
| A91_C1_2 | 20 | 0.8845 | 0.4170 | 0.0000 |
| A91_C1_2 | 40 | 0.8751 | 0.1861 | 0.0003 |
| A31_C1 | 0 | 0.9521 | 0.3129 | 0.7211 |
| A31_C1 | 20 | 0.9537 | 0.6134 | 0.8613 |
| A31_C1 | 40 | 0.9539 | 0.3098 | 0.8416 |
| A52_C2 | 0 | 0.9740 | 0.5206 | 0.8666 |
| A52_C2 | 20 | 0.9648 | 0.4163 | 0.7448 |
| A52_C2 | 40 | 0.9686 | 0.5028 | 0.7270 |
| **MEAN** | | **0.9372** | **0.3859** | **0.5292** |

## The comparison

| NO-prompt IoU | baseline | v2 | factor |
|---|---|---|---|
| A91_C1_2 | 0.0094 | 0.0001 | — |
| A31_C1 | 0.0069 | **0.8080** | 118× |
| A52_C2 | 0.0116 | **0.7795** | 67× |
| **MEAN** | **0.0093** | **0.5292** | **57×** |

The `correct GT` column is unchanged (0.9378 vs 0.9372) and `WRONG mask` is
unchanged (0.3870 vs 0.3859), so the two models are equally good at refining a
mask they are handed. The entire difference is in the ability to segment with
**no prompt at all**, which standard training never acquires and which prompt
dropout produces.

**The result is bimodal, not uniform.** On A31_C1 and A52_C2 the model segments
unprompted at 0.72-0.87, close to its prompted score. On A91_C1_2 it stays at
zero. A91_C1_2 is the hardest recording in the validation split (per-video IoU
~0.761 against ~0.874 and ~0.845), so the capability appears where the imaging
is clean and not where it is not.

## What this does and does not show

**Does:** prompt dropout gives SAM 2's decoder a standalone segmentation
ability it otherwise entirely lacks — a 57x change in the mean, from 0.0093 to
0.5292, with the prompted pathway untouched.

**Does not:** improve prompted propagation accuracy. Over matched epochs 98-110
v2 scores val IoU 0.8304 against the baseline's 0.8373 (t = -19.6), with
Betti-0 and Betti-1 both indistinguishable. The new capability did not
translate into a better model on the metric the benchmark reports.

**Attribution caveat.** v2 trained with prompt dropout AND distillation, so
strictly this measures the pair. Distillation's target is the model's own
PROMPTED output, which cannot teach prompt-free segmentation, while dropout
explicitly forces it -- so the mechanism points at dropout. Establishing it
cleanly needs a dropout-only run, which has never been done: v1 and v2 both
had both mechanisms.
