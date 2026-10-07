#!/bin/bash
# ---------------------------------------------------------------------------
# Test-set predictions for one finished fold of slurm/train_fold.sbatch: both
# checkpoints (IoU-best and topology tie-break) on that fold's four held-out
# recordings.
#
#   sbatch slurm/score_fold.sbatch 2
#
# Refuses a checkpoint whose own training_config.json did not hold out this
# fold. Copies that config next to the prediction stacks, where
# benchmark/aggregate_folds.py checks it again, and scores into
# benchmark/results/test_fold<N>/ (rows ours_best, ours_topo).
# ---------------------------------------------------------------------------
#SBATCH --job-name=nv_score_fold
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/score_fold_%j.out
#SBATCH --error=logs/score_fold_%j.out
#SBATCH --no-requeue

FOLD="$1"
if [ "$FOLD" != "2" ] && [ "$FOLD" != "3" ]; then
    echo "usage: sbatch slurm/score_fold.sbatch <2|3>   (fold 1 is scored: ours_23rec, ours_23rec_topo)"
    exit 2
fi

echo "=== score fold ${FOLD} $(date) ==="; hostname
source ~/.bashrc
conda activate neurotrack-env
cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs
echo "Code: $(git log -1 --format='%h %s')"

CKPT_DIR=checkpoints-fold${FOLD}
VIDEOS=$(python -c "from src.folds import test_videos; print(' '.join(test_videos(${FOLD})))")
echo "fold ${FOLD} test recordings: ${VIDEOS}"

python - "$CKPT_DIR" "$FOLD" <<'PY' || exit 1
import json, sys
from src.folds import test_videos
ckpt, fold = sys.argv[1], int(sys.argv[2])
held_out = json.load(open(f"{ckpt}/training_config.json"))["data"]["test_videos"]
if sorted(held_out) != sorted(test_videos(fold)):
    sys.exit(f"{ckpt} held out {held_out}, not fold {fold}'s {test_videos(fold)}")
print(f"{ckpt} held out exactly fold {fold}'s test set")
PY

for tag in "" "_topo"; do
    ck="${CKPT_DIR}/lora_weights${tag}.pt"
    name="ours_fold${FOLD}${tag}"
    echo; echo "##### ${name}  (${ck})"
    python benchmark/run_ours.py --ckpt "$ck" --videos $VIDEOS --out-dir "benchmark/predictions/${name}" \
        || { echo "run_ours failed for ${name}"; continue; }
    cp "${CKPT_DIR}/training_config.json" "benchmark/predictions/${name}/training_config.json"
    python benchmark/score.py --pred-dir "benchmark/predictions/${name}" --model "ours${tag:-_best}" \
        --fold "$FOLD" --split test --cbdice --skip-first-frame
done

echo; echo "##### fold ${FOLD} summary (paper fold 1, topo checkpoint: IoU 0.7685 / cbDice 0.8602)"
column -s, -t "benchmark/results/test_fold${FOLD}/summary.csv"
echo "=== done at $(date) ==="
