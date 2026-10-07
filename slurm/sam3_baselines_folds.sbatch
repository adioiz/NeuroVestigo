#!/bin/bash
# ---------------------------------------------------------------------------
# The two SAM 3 baselines of Table 1 on the new test recordings of folds 2 and
# 3, with the settings behind the paper's rows:
#   SAM 3 (mask)    run_sam3_mask.py                       -> sam3_maskfirst/
#   SAM 3.1 (text)  run_sam3.py --mode text --text cells
#                   --max-num-objects 64 --no-cudnn         -> sam3_text_final/
# Neither is trained on this data, so one run serves both folds, and fold 1's
# recordings already have predictions.
#
#   sbatch slurm/sam3_baselines_folds.sbatch
#
# It first reruns both on fold 1's A71_C1 and compares with the paper's stacks,
# so a changed setting shows up before eight recordings are spent on it.
# ---------------------------------------------------------------------------
#SBATCH --job-name=nv_sam3_folds
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sam3_folds_%j.out
#SBATCH --error=logs/sam3_folds_%j.out
#SBATCH --no-requeue

echo "=== sam3 baselines for folds 2-3, $(date) ==="; hostname
source ~/.bashrc
conda activate sam3-env
cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs
echo "Code: $(git log -1 --format='%h %s')"

VIDEOS=$(python -c "from src.folds import FOLDS; print(' '.join(v for f in (2, 3) for v in FOLDS[f]))")
echo "new test recordings: ${VIDEOS}"
TEXT_ARGS="--mode text --text cells --max-num-objects 64 --no-cudnn"
CHECK=$(mktemp -d)

echo; echo "##### reproduction check on A71_C1"
python -u benchmark/run_sam3_mask.py --videos A71_C1 --out-dir "$CHECK/sam3_maskfirst" || exit 1
python -u benchmark/run_sam3.py $TEXT_ARGS --videos A71_C1 --out-dir "$CHECK/sam3_text_final" || exit 1
python - "$CHECK" <<'PY' || exit 1
import os, sys
import numpy as np, tifffile
check = sys.argv[1]
for d in ("sam3_maskfirst", "sam3_text_final"):
    ref = os.path.join("benchmark", "predictions", d, "A71_C1.tif")
    if not os.path.exists(ref):
        print(f"  {d}: no paper stack at {ref} to compare with -- unchecked")
        continue
    a = tifffile.imread(ref) > 0
    b = tifffile.imread(os.path.join(check, d, "A71_C1.tif")) > 0
    union = np.logical_or(a, b).sum()
    agree = 1.0 if union == 0 else np.logical_and(a, b).sum() / union
    print(f"  {d}: IoU with the paper's stack {agree:.4f}")
    if agree < 0.90:
        sys.exit(f"{d} does not reproduce the paper's run -- settings changed; stopping")
PY

echo; echo "##### SAM 3 (mask) on the new recordings"
python -u benchmark/run_sam3_mask.py --videos $VIDEOS --out-dir benchmark/predictions/sam3_maskfirst
echo; echo "##### SAM 3.1 (text) on the new recordings"
python -u benchmark/run_sam3.py $TEXT_ARGS --videos $VIDEOS --out-dir benchmark/predictions/sam3_text_final

rm -rf "$CHECK"
echo; ls benchmark/predictions/sam3_maskfirst benchmark/predictions/sam3_text_final
echo "=== done at $(date) ==="
