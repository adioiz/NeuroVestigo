#!/bin/bash
# ---------------------------------------------------------------------------
# One cross-validation fold (src/folds.py), chained on the 4-hour partition.
#
#   sbatch slurm/train_fold.sbatch 2
#   sbatch slurm/train_fold.sbatch 3
#
# The paper's 23-recording recipe, unchanged except for the held-out test set:
# --fold swaps in that fold's four test recordings and pins the shared
# validation set, and the rest of these flags are the ones the paper run used
# (its resolved config was diffed against checkpoints-23rec/training_config.json:
# only test_videos and checkpoint_dir differ).
#
# Folds 2 and 3 write to different directories, so both arrays can be queued
# at once. Within one array, %1 keeps exactly one task running: two tasks
# sharing train_state.pt would interleave epochs without crashing.
#
# Every task first checks training_data/ against the fingerprints committed in
# src/data_fingerprints.json. It is not in git and is synced by hand; in folds
# 2 and 3, fold 1's test recordings (A71_C1 among them) are TRAINING data, so a
# stale copy here would be trained on silently.
# ---------------------------------------------------------------------------
#SBATCH --job-name=nv_fold
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%1
#SBATCH --output=logs/fold_%A_%a.out
#SBATCH --error=logs/fold_%A_%a.err
#SBATCH --no-requeue

FOLD="$1"
if [ "$FOLD" != "2" ] && [ "$FOLD" != "3" ]; then
    echo "usage: sbatch slurm/train_fold.sbatch <2|3>   (fold 1 is the paper's run, already trained)"
    scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}" 2>/dev/null
    exit 2
fi

echo "=== Fold ${FOLD}: task ${SLURM_ARRAY_TASK_ID} of array ${SLURM_ARRAY_JOB_ID} started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source ~/.bashrc
conda activate neurotrack-env

cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs
echo "Code: $(git log -1 --format='%h %s')"

EPOCHS=110
CKPT_DIR=checkpoints-fold${FOLD}
COMMON_ARGS="--fold $FOLD --epochs $EPOCHS --batch-size 2 --num-frames 16 --val-num-frames 16 --patience 90 --checkpoint-dir $CKPT_DIR"

echo "--- Preflight ---"
if ! python -m src.folds --check-data; then
    echo "[preflight] training_data/ does not match the laptop's -- not training."
    scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
    exit 1
fi
if [ -f "$CKPT_DIR/training_config.json" ]; then
    # A directory left by another split would be resumed as if it were this one.
    if ! python - "$CKPT_DIR" "$FOLD" <<'PY'
import json, sys
from src.folds import test_videos
ckpt, fold = sys.argv[1], int(sys.argv[2])
got = json.load(open(f"{ckpt}/training_config.json"))["data"]["test_videos"]
if sorted(got) != sorted(test_videos(fold)):
    sys.exit(f"[preflight] {ckpt} holds a run with test_videos={got}, not fold {fold}")
PY
    then
        scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
        exit 1
    fi
fi

STATE_FILE=$CKPT_DIR/train_state.pt

if [ -f "$STATE_FILE" ]; then
    # The on-disk `epoch` key is 0-indexed and already COMPLETED, so a
    # finished 110-epoch run stores 109 -- hence epoch+1 >= EPOCHS.
    DONE_EPOCHS=$(python - <<PY
import torch
try:
    s = torch.load("$STATE_FILE", map_location="cpu", weights_only=False)
    print(int(s.get("epoch", -1)) + 1)
except Exception:
    print(-1)
PY
    )
    echo "[chain] $DONE_EPOCHS/$EPOCHS epochs completed"

    if [ "$DONE_EPOCHS" -ge "$EPOCHS" ] 2>/dev/null; then
        echo "[chain] Target reached -- cancelling remaining pending tasks."
        scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
        echo "=== Job ended at $(date) ==="
        exit 0
    fi

    echo "[resume] Found $STATE_FILE - continuing previous run"
    echo "--- Starting training (resuming) ---"
    python -u cli.py train $COMMON_ARGS --resume "$STATE_FILE"
    RC=$?
else
    echo "[resume] No saved state - starting fresh"
    echo "--- Starting training (fresh) ---"
    python -u cli.py train $COMMON_ARGS
    RC=$?
fi

echo "=== Fold ${FOLD}: task ${SLURM_ARRAY_TASK_ID} exited with $RC at $(date) ==="

# 143 = SIGTERM and 137 = SIGKILL, i.e. the 4-hour wall clock stopping us
# mid-epoch. That is normal here: the next task resumes from the last epoch
# boundary. Anything else is a real failure, and without this the array would
# retry the same bug nine more times, each after queueing for a slot.
if [ $RC -ne 0 ] && [ $RC -ne 143 ] && [ $RC -ne 137 ]; then
    echo "[chain] Non-timeout failure -- cancelling the rest of the array."
    scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
fi
exit $RC
