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
pip install -e ~/Fine-tune-SAM2-model-for-Neurons-segmentation/sam3
```

Wheel names must read `cp312`. If you see `cp39`, the env is not active.

---

## Stage 1 — probe the real API  ⟵ STOP AND REPORT

The login node has no GPU. Get a GPU node first:

```bash
srun --partition=B200-4h --gres=gpu:1 --pty bash
conda activate sam3-env
cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation
python benchmark/run_sam3.py --probe
```

This prints the request types and prompt keys your installed build actually
accepts. **Send that output to Claude before going further.**

- If it matches what `run_video()` calls → proceed to Stage 2.
- If it differs → one edit to the three `handle_request` calls and possibly
  `_iter_results()`. Cheap now; expensive after a cluster run.

---

## Stage 2 — sweep prompt modes on VALIDATION only

SAM 3 cannot take a mask prompt (its predictor accepts text, points and boxes —
see `sam3/model/sam3_base_predictor.py`), so it cannot receive the frame-0 mask
our model and zero-shot SAM 2 get. That is a property of the model, not a
handicap imposed on it. Its closest analogue is the box prompt, which is exactly
what micro-SAM receives.

Sweep on val, adopt the winner, apply it unchanged to test — the same rule every
other competitor here was tuned under.

```bash
for m in box-every box-first points-every text; do
  python benchmark/run_sam3.py --mode $m \
    --videos A91_C1_2 A31_C1 A52_C2 A22_C2 A42_C1
done
```

Writes `benchmark/predictions/sam3_<mode>/<video>.tif` (hyphens stripped:
`box-every` → `sam3_boxevery`).

**Check:** four directories, five `.tif` each. The runner prints a foreground
fraction per video — if it is `0.0000` everywhere, the prompts are not reaching
the model and Stage 1 was wrong.

---

## Stage 3 — score the sweep and pick the winner

Scoring runs in **`neurotrack-env`** (see rule 2). `conda run` avoids
switching your shell and cannot accidentally pip-install into it:

```bash
for m in boxevery boxfirst pointsevery text; do
  conda run -n neurotrack-env python benchmark/score.py \
    --pred-dir benchmark/predictions/sam3_$m \
    --model sam3_$m --split val --cbdice --skip-first-frame
done
cat benchmark/results/val/summary.csv
```

`--skip-first-frame` matters: in `box-first` and `text` mode frame 0 is prompted,
so scoring it would flatter those modes. Every competitor is scored the same way.

`--split val` needs no `--videos` — it resolves the five recordings from
`gpu-checkpoints/training_config.json`. If it errors that the val split is empty,
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
| `TypeError: 'NoneType' object is not iterable` in a split | `data.val_videos` is null in `gpu-checkpoints/training_config.json` |
| `--split test` scores nothing, no error | the four test recordings are missing from `data_manifest.json` |
| GT file not found when scoring | manifest holds absolute Mac paths; `load_gt` falls back to `training_data/masks/<name>/`, so that recording needs `cli.py prepare` |
| `unexpected propagate response` | Stage 1 guesses were wrong; fix `_iter_results()` |
| foreground fraction `0.0000` everywhere | prompts not reaching the model; re-run Stage 1 |
| `ModuleNotFoundError: tifffile` | install the four pinned extras in Stage 0 |
| `ModuleNotFoundError: pkg_resources` | `setuptools<81` missing in `sam3-env` |
| `ModuleNotFoundError: einops` | SAM 3 does not declare it; install the four pinned extras in Stage 0 |
| `sam3 0.1.0 requires numpy<2 ... you have numpy 2.x` | something pulled numpy up; re-run the Stage 0 pinned install |
| `ModuleNotFoundError: skimage` when scoring | you are scoring in `sam3-env`; score in `neurotrack-env` |
| `huggingface-cli` prints a deprecation notice and exits | use `hf` (`hf auth login`, `hf download`) |

## Note for the paper

Whatever the number turns out to be, state the protocol asymmetry plainly:
SAM 3 could not be given a mask prompt because its video predictor does not
accept one. Reported under its best box/points/text regime, chosen on
validation. A reviewer who notices the difference unaided will assume the
comparison was rigged.
