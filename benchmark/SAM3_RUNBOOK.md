# SAM 3 benchmark — runbook

Resume guide. Assumes you have been away and remember nothing. Every stage has a
check; if a check fails, stop there rather than continuing.

---

## Where you left off

| done | |
|---|---|
| ✅ | `sam3-env` conda env created (Python 3.12) and SAM 3 installed into it |
| ✅ | `benchmark/run_sam3.py` written, committed, and the split/GT bugs fixed |
| ✅ | `hf auth login` as **adioizerovich** |
| ⏳ | **Blocked:** `facebook/sam3` weights pending license approval on Hugging Face |

Nothing else is blocked. The moment access lands, start at Stage 0.

---

## Two rules that will cost you a day if broken

**1. Never `pip install` anything while `neurotrack-env` is active.**
SAM 3 requires `numpy<2`; the training env has numpy 2.2.6. Installing SAM 3
there downgrades numpy and breaks training. SAM 3 lives in `sam3-env`, alone.
This already happened once into the system Python 3.9 — harmless there, fatal
in `neurotrack-env`.

**2. Two environments, and they are not interchangeable.**
`run_sam3.py` runs in `sam3-env` (it needs SAM 3, plus `tifffile`).
`score.py` runs in `neurotrack-env` — it does not need SAM 3, and clDice/cbDice
go through `skimage.morphology.skeletonize`, so it must run in the SAME
environment that scored every other competitor or the SAM 3 row is not
comparable to them.

**3. The SAM 3 API calls in `run_video()` are INFERRED.**
The public README documents only the text-prompt video path, so
`start_session` / `add_prompt` / `propagate_in_video` are educated guesses.
Stage 1 (`--probe`) exists to check them. Do not skip to Stage 2 — a wrong
guess produces numbers rather than an error, and you would not know.

---

## Stage 0 — verify access and environment

```bash
conda activate sam3-env
hf download facebook/sam3
du -sh ~/.cache/huggingface/hub/models--facebook--sam3
```

**Check:** multi-GB. Kilobytes means the config files downloaded but the weights
are still gated — access has not been granted yet, so stop.

```bash
python -c "import sys, sam3, torch, tifffile; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"
```

**Check:** Python **3.12.x**, torch ≥2.7, CUDA ≥12.6, numpy **1.26.x**.
If Python says 3.9, the env is not active.

If anything is missing, install the four extras SAM 3 does not pull in itself —
**pinned, all in one resolve**:

```bash
pip install "numpy>=1.26,<2" "tifffile<2025" "setuptools<81" einops
```

Do NOT `pip install tifffile` unpinned. The 2026 build hard-requires
`numpy>=2.1`, so pip silently upgrades numpy and breaks SAM 3's own `numpy<2`
pin — the package installs, then `import sam3` fails or misbehaves. A 2024
build reads exactly the same TIFFs. `setuptools<81` is for `pkg_resources`,
which `sam3/model_builder.py` imports: Python 3.12 venvs ship no setuptools and
81+ removed it. `einops` is imported by `sam3/sam/rope.py` and is genuinely
absent from SAM 3's own dependency list — `import sam3` fails with
`ModuleNotFoundError: No module named 'einops'` without it. It is unpinned
because it is pure Python with no numpy floor.

And confirm the SCORING env has its own deps, without installing anything into it:

```bash
conda run -n neurotrack-env python -c "import tifffile, skimage, numpy; print(tifffile.__version__, skimage.__version__, numpy.__version__)"
```

If `sam3-env` is gone (cluster rebuild, purged home):

```bash
conda create -y -n sam3-env python=3.12
conda activate sam3-env
pip install -e ~/NeuroVestigo/sam3
```

Wheel names must read `cp312`. If you see `cp39`, the env is not active.

---

## Stage 1 — probe the real API  ⟵ STOP AND REPORT

The login node has no GPU. Get a GPU node first:

```bash
srun --partition=B200-4h --gres=gpu:1 --pty bash
conda activate sam3-env
cd ~/NeuroVestigo
python benchmark/run_sam3.py --probe
```

This prints the request types and prompt keys your installed build actually
accepts. **Send that output to Claude before going further.**

- If it matches what `run_video()` calls → proceed to Stage 2.
- If it differs → one edit to the three `handle_request` calls and possibly
  `_iter_results()`. Cheap now; expensive after a cluster run.

---

## Stage 2 — sweep prompt modes on VALIDATION only

`Sam3VideoPredictor.add_prompt` accepts text, points and boxes but not a mask,
so on this path SAM 3 cannot receive the frame-0 mask our model and zero-shot
SAM 2 get. Its closest analogue is the box prompt, which is exactly what
micro-SAM receives.

> **Unresolved, and it matters.** `--probe` also found
> `Sam3TrackerPredictor.add_new_mask` and
> `Sam3VideoTrackingMultiplexDemo.add_new_masks` — a SAM 2-style tracking path
> that looks mask-promptable. If it can be driven standalone (`build_tracker`
> takes `with_backbone=False`, so this is not obviously easy), SAM 3.1 could be
> benchmarked with the same frame-0 mask as everything else and the protocol
> asymmetry below disappears. Settle this before the paper claims SAM 3 cannot
> take a mask.

Sweep on val, adopt the winner, apply it unchanged to test — the same rule every
other competitor here was tuned under.

```bash
for m in box-every box-first points-every points-first text; do
  python benchmark/run_sam3.py --mode $m --no-cudnn \
    --videos A91_C1_2 A31_C1 A52_C2 A22_C2 A42_C1
done
```

Writes `benchmark/predictions/sam3_<mode>/<video>.tif` (hyphens stripped:
`box-every` → `sam3_boxevery`). About a minute per mode, model load included.

`--no-cudnn` is required on dgx-b200-02: see the cuDNN row in Troubleshooting.

**Check:** five directories, five `.tif` each. The runner prints a foreground
fraction per video — `0.0000` everywhere means prompts are not reaching the
model; near `1.0000` means a thresholding problem.

**Calibration.** On A71_C1, `box-first` gave foreground 0.0009 against a GT
foreground of 0.0330 — precision 0.89, recall 0.024. SAM 3.1 segments the soma
accurately and misses the arbour, which is what a box around a thin branching
structure invites: the box is mostly background, so the model takes the compact
object inside it. Expect the point modes to do better, and do not read a verdict
off `box-first` alone.

---

## Stage 3 — score the sweep and pick the winner

Scoring runs in **`neurotrack-env`** (see rule 2). `conda run` avoids
switching your shell and cannot accidentally pip-install into it:

```bash
for m in boxevery boxfirst pointsevery pointsfirst text; do
  conda run -n neurotrack-env python benchmark/score.py \
    --pred-dir benchmark/predictions/sam3_$m \
    --model sam3_$m --split val --cbdice --skip-first-frame
done
cat benchmark/results/val/summary.csv
```

`--skip-first-frame` matters: in `box-first` and `text` mode frame 0 is prompted,
so scoring it would flatter those modes. Every competitor is scored the same way.

`--split val` needs no `--videos` — it resolves the five recordings from
`checkpoints/training_config.json`. If it errors that the val split is empty,
that config lost its `val_videos`.

**Pick the winning mode by val IoU.** Write it down; you need it twice below.

---

## Stage 4 — run the winner on TEST

Substitute the winning mode for `MODE`:

```bash
python benchmark/run_sam3.py --mode MODE \
  --videos A71_C1 A81_C1 A12_C3_100k A12_C1_100k_2
```

These four are the true held-out set — never trained on, never validated on.
Run **one** mode here. Sweeping on test is what the val split exists to prevent.

---

## Stage 5 — score TEST

`MODE` here is the stripped form (`boxevery`, not `box-every`):

```bash
conda run -n neurotrack-env python benchmark/score.py \
  --pred-dir benchmark/predictions/sam3_MODE \
  --model sam3 --split test --cbdice --skip-first-frame
cat benchmark/results/test/summary.csv
```

---

## What to send back

`benchmark/results/test/summary.csv` — that is the row for Tables 2, 3 and 4 and
the sentences in §4.1.

Also worth sending: `benchmark/results/val/summary.csv`, so the sweep and the
choice of prompt mode are on the record rather than asserted.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `Access denied. This repository requires approval` | license not granted yet; the small configs download regardless, so check the directory SIZE |
| wheels say `cp39` | `sam3-env` not active |
| `TypeError: 'NoneType' object is not iterable` in a split | `data.val_videos` is null in `checkpoints/training_config.json` |
| `--split test` scores nothing, no error | the four test recordings are missing from `data_manifest.json` |
| GT file not found when scoring | manifest holds absolute Mac paths; `load_gt` falls back to `training_data/masks/<name>/`, so that recording needs `cli.py prepare` |
| `unexpected propagate payload` | mask key differs on this build; add it to `_MASK_KEYS` |
| `propagation returned no frames` | the session loaded 0 frames — SAM 3's loader may not accept a `.png` directory |
| foreground fraction `0.0000` everywhere | prompts not reaching the model; re-run `--probe` |
| foreground fraction near `1.0000` | masks came back as probabilities and were thresholded at 0; `_binarize` should catch this |
| `ModuleNotFoundError: tifffile` | install the four pinned extras in Stage 0 |
| `ModuleNotFoundError: pkg_resources` | `setuptools<81` missing in `sam3-env` |
| `ModuleNotFoundError: einops` | SAM 3 does not declare it; install the four pinned extras in Stage 0 |
| `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` | two cuDNN installs on the library path; pass `--no-cudnn` (conv **and** SDPA), or fix `LD_LIBRARY_PATH` |
| `expand(-1, B, -1)` ... `existing size (0)` | truncated propagation empties the tracker; do not set `max_frame_num_to_track` |
| `init_state() got an unexpected keyword` | handled by the `_filter_init_state_kwargs` shim; the `[shim]` line confirms it engaged |
| `AssertionError` at `sam3_multiplex_tracking.py:1717` | prompts must be normalised to [0,1] |
| `sam3 0.1.0 requires numpy<2 ... you have numpy 2.x` | something pulled numpy up; re-run the Stage 0 pinned install |
| `ModuleNotFoundError: skimage` when scoring | you are scoring in `sam3-env`; score in `neurotrack-env` |
| `huggingface-cli` prints a deprecation notice and exits | use `hf` (`hf auth login`, `hf download`) |

## Confirmed SAM 3 API (verified by `--probe`, 2026-08-26)

Not guesses any more — read off `Sam3BasePredictor` on the installed build.

| | |
|---|---|
| builder | `sam3.model_builder.build_sam3_predictor(version="sam3.1")` |
| avoid | `build_sam3_video_predictor` — returns `Sam3VideoPredictorMultiGPU`, wants an NCCL group |
| start | `handle_request({"type":"start_session","resource_path":<dir>})` |
| prompt | `handle_request({"type":"add_prompt","session_id":..,"frame_index":..,..})` |
| propagate | `handle_stream_request({...})` — a **generator**, yields `{"frame_index","outputs"}` |
| masks | `out_binary_masks`, already thresholded at `output_prob_thresh` (0.5) |

Two defaults that produce wrong numbers rather than errors, both now overridden
explicitly in `run_video()`:

- `add_prompt(rel_coordinates=True)` — pixel boxes would be read as normalised
  coordinates, putting every prompt off-frame.
- `propagate_in_video(propagation_direction="both")` — walks forward then
  backward, writing every frame twice.

## RESULTS (complete, 2026-08-27)

**Reported configuration:** SAM 3.1, `--mode points-every --npoints 10`.
Chosen on validation, applied unchanged to test.

**Test (n=4):** IoU **0.4712** · Dice 0.5953 · clDice 0.5921 · cbDice 0.6018
Eighth of fourteen methods; ours is 0.7599.

**Language-only row.** `--mode text --text cells --max-num-objects 64`,
also chosen on validation (0.4192): test IoU **0.2002** · Dice 0.2779 ·
clDice 0.2901 · cbDice 0.2600. Twelfth of fourteen.

### Fairness sweep — everything that was tried

Every other competitor here gets a hyperparameter sweep before its number is
reported. This is SAM 3.1's.

| knob | values tried | result |
|---|---|---|
| prompt mode | box-every, box-first, points-every, points-first, text | points-every wins by 20x |
| `npoints` | 3, 5, 10, 15, 20, 30 | interior optimum at **10** (0.6341); degrades past it, 30 collapses to 0.3684 |
| `output_prob_thresh` | 0.3, 0.5, 0.7 | **no effect on either path** — verified plumbed link-by-link, see below |
| `use_rope_real` | True / False | 0.5786 vs 0.5795 — noise; the 64 missing `freqs_cis` keys are benign |
| version | sam3.1 (stronger) | used throughout |
| text + points | — | **forbidden by the model** (`sam3_multiplex_tracking.py:2710`) |
| text + box | "neuron" | 0.0275 vs 0.0272 without — no effect |
| text phrase (12) | **cells 0.3938**, cell 0.2920, "a cell" 0.1667; 0.0000 for neuron, nerve cell, gray neuron, bright cell, fluorescent cell, white worm, branching structure, thin white line, bright spot, tree, root, blob, object |
| `max_num_objects` | 16 → 0.3938, 32 → 0.4132, **64 → 0.4192**; the default cap was discarding the target |

Validation IoU by prompt mode (npoints=3): points-every 0.5786, box-every
0.0272, box-first 0.0248, points-first 0.0000, text 0.0000.

### Why `output_prob_thresh` is inert, and why that is a finding

Not an unswept knob. The chain was checked at every link: `handle_stream_request`
forwards it to `propagate_in_video`, which forwards it to the model, and both
`add_prompt` and `propagate_in_video` on `Sam3MultiplexTracking*` declare it.
It still produces bit-identical masks at 0.3/0.5/0.7 on the concept path AND
the tracker path. With a visual prompt and an `obj_id`, masks come from the
tracker decoder at its own logit boundary; the threshold gates concept-detection
confidence, and on this data the detector finds nothing (hence text = 0.0000).

### Vocabulary, not blindness

"neuron" produces zero detections; "cells" reaches 0.3938 on validation.
Crucially **"blob" and "object" also produce zero**, so this is not a generic
"something is here" detector firing on any prompt -- SAM 3.1 has a learned
*cell* concept that transfers to fluorescence microscopy and lacks a *neuron*
one. Modifiers destroy it ("bright cell" -> 0.0000) and even the article
matters ("a cell" -> 0.1667).

`raw_fg` and `inst/frame` (printed by the runner) separate "the detector found
nothing" from "it found things, none of them the target". Both occur: every
zero-scoring phrase has `inst/frame=0.0`, while A42_C1 under "cell" has the
highest raw foreground of any recording (0.1099) and IoU 0.0011.

Language-prompted performance is also wildly unstable across recordings of the
same preparation: on test, A81_C1 scores 0.5430 while A12_C1_100k_2 scores
0.0000 with essentially no detections at all (raw_fg 0.0004). That variance is
itself the finding.

### Two results that matter more than the ranking

**Zero-shot SAM 2 beats SAM 3.1 by 0.24** (0.6392 vs 0.4712). Architectural,
not a tuning artefact: SAM 2's video predictor accepts a mask and starts on the
exact target; SAM 3.1's video path is concept-driven and refuses masks, so the
most it can be given is points.

**`points-first` scores 0.0000.** That is the like-for-like comparison against
our protocol — one prompt on frame 0. SAM 3.1's 0.4712 required ground-truth
points on all 73-93 frames. The supervision asymmetry runs against SAM 3.1's
favour, not ours; say so explicitly or a reviewer will assume the opposite.

**Caveat to state:** clDice peaks at 5 points (0.8182) and falls at 10 (0.7813)
while IoU rises. Selection was on IoU, consistent with every other competitor —
state that rule rather than letting it look chosen after the fact.

### SETTLED 2026-08-29: the mask path is not usable in the released build

`Sam3TrackerPredictor.add_new_mask` exists, and the tracker can be built
standalone -- `build_tracker(apply_temporal_disambiguation=False,
with_backbone=True)` works, frames load once mirrored as JPEG (the loader
globs `.jpg` and reports "no images found" on our `.png`), and the tracker
moves to the GPU. But `add_new_mask` itself then fails:

    AttributeError: 'Sam3TrackerPredictor' object has no attribute
                    'teacher_force_obj_scores_for_mem'

That is inside SAM 3's own code, with a correct CPU 2-D tensor, cuDNN
disabled and autocast on -- not an input problem. The entry point is present in
the source but not wired for objects built by the only public factory.

`build_tracker` also takes no checkpoint argument, and the weights it returns
have std 0.0197-0.0238 across 40 tensors -- a fresh-init signature. So even a
working call would not have produced a benchmarkable row without a separate
way to load the released weights.

Four earlier runs stopped for reasons that were OURS, not SAM 3's: a missing
required positional on `build_tracker`, the `.png` loader, a CPU/CUDA dtype
mismatch, and the cuDNN sublibrary mismatch that `run_sam3.py` has had a flag
for all along. Each time the probe announced the paper's claim was supported.
`benchmark/test_probe_sam3_mask.py` now exercises the probe against a stub with
these signatures before it is run, and the probe reports a stop as evidence
only when the refusal is about masks rather than plumbing.

**Written into the paper as:** the interface accepts text, point and box
prompts and no working path was found for a mask -- a statement about what we
could drive, not about the architecture, which does contain the machinery.

## Note for the paper

Whatever the number turns out to be, state the protocol asymmetry plainly:
SAM 3 could not be given a mask prompt because its video predictor does not
accept one. Reported under its best box/points/text regime, chosen on
validation. A reviewer who notices the difference unaided will assume the
comparison was rigged.


## Is the area blow-up ours or SAM 3.1's? -- SETTLED, it is SAM 3.1's

Asked 2026-08-29 after the red-everywhere panel in `proof_A81_C1`. On late
frames SAM 3.1's point-prompted mask covers 42-78% of the field, up to 72x the
target area. Six checks, all pointing the same way:

| check | result |
|---|---|
| `inst/frame` in point mode | **1.0** -- one object per frame, so `np.any(masks,0)` is a no-op |
| connected components | 95-98% of the area is ONE component |
| keep only GT-overlapping components | IoU 0.106 -> 0.112, i.e. the blob CONTAINS the target |
| polarity | as-is 0.554 vs inverted 0.007 -- not flipped |
| blow-up vs `--npoints` | already 45x at 3 points; worse at 3 than at 5 |
| area over time | non-monotonic (49.9% -> 13.1% -> 76.1%) -- not accumulation |

The `inst/frame=1.0` measurement is the decisive one and is worth re-running
if the runner's mask handling ever changes:

```bash
python benchmark/run_sam3.py --mode points-every --npoints 10 --no-cudnn \
    --limit-frames 73 --out-dir /tmp/sam3_check --videos A81_C1
```

It printed `fg=0.1476`, bit-identical to the committed prediction, so the run
reproduces. Written up in the paper's appendix; the same quantity is Table 3's
-0.418 decay.


## Validation numbers re-verified from artifacts (2026-08-29)

Every SAM 3.1 validation claim in the paper's appendix was re-scored from the
retained prediction directories. 17 of 18 reproduced to 4 decimal places.

One did not, and it is a real correction: the paper attributed **0.634** to the
prompt-mode comparison, but `sam3_pointsevery` scores **0.5786** -- bit-identical
to `sam3_p3_t0.5`, i.e. that stage ran at the DEFAULT 3 points. 0.634 is the
post-tuning value from the later npoints sweep, so quoting it at the
prompt-mode stage was anachronistic. Corrected to 0.579; the conclusion is
unchanged, since points-every beats box-every 0.579 vs 0.027 either way.

Also confirmed: clDice 0.8182 at 5 points vs 0.7813 at 10 (the selection-rule
caveat), ours on validation IoU 0.7618 / cbDice 0.8429, and the two collapse
recordings are 14.7x (A91_C1_2, val) and 12.3x (A12_C3_100k, test)
foreground over-predictions -- "roughly tenfold" is fair.

**Not verifiable from artifacts:** the pre-SAM-3 competitor sweep values
(uSAM vit_b 0.42 vs vit_l 0.56; Cellpose cellprob 0.38 -> 0.47 and flow
0.44 -> 0.56; uSAM multimask 0.574 vs 0.595). Those sweeps did not retain
their predictions -- only the final shipped configuration was kept, which
scores usam 0.5752 and cellpose_sam 0.5768 on validation. The sweep numbers
rest on the contemporaneous audit trail in main.tex's header block, not on
anything re-derivable. Retain sweep predictions in future.
