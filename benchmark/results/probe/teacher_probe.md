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

Not yet run; the checkpoint lives on the cluster. Command:

    python benchmark/probe_teacher.py --ckpt checkpoints-distill-v2/lora_weights.pt

**What it decides.** Prompt dropout is the one mechanism never tested in
isolation, and the NO-prompt column measures exactly what it was meant to
teach:

- **~0.01, like the baseline** — dropout did not work either. Both mechanisms
  failed; the honest write-up is a negative result for the pair.
- **materially higher (0.2, 0.4, ...)** — dropout taught the decoder to
  segment without a prompt, something the baseline never learned. That is a
  real, separable finding that stands regardless of distillation, and it
  changes what the paper can claim.
