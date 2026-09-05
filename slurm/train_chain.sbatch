#!/bin/bash
# ---------------------------------------------------------------------------
# Chained training on a 4-hour partition, one job at a time.
#
#   sbatch slurm/train_chain.sbatch
#
# `--array=1-10%1` queues ten tasks but lets only ONE run at a time, which is
# the whole point: two tasks running concurrently would both load the same
# train_state.pt and both write it back, interleaving epochs and corrupting
# optimiser state. The %1 throttle makes that impossible.
#
# Each task resumes from checkpoints-distill/train_state.pt, written at the end
# of EVERY epoch. A task killed mid-epoch by the wall clock loses at most that
# one epoch, not the run.
#
# Task 1 starts fresh (--resume is omitted when no state file exists, because
# trainer.py raises FileNotFoundError on a --resume path that is not there).
# Once the run reaches EPOCHS the next task cancels the remaining PENDING
# tasks and exits, so leftover slots are released rather than wasted.
# ---------------------------------------------------------------------------
#SBATCH --job-name=nv-distill
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=1-10%1
#SBATCH --output=logs/nv-distill_%A_%a.out
#SBATCH --error=logs/nv-distill_%A_%a.out

set -u

REPO="$HOME/Fine-tune-SAM2-model-for-Neurons-segmentation"
CKPT_DIR="$REPO/checkpoints-distill"
EPOCHS=110

cd "$REPO" || exit 1
mkdir -p logs "$CKPT_DIR"

# --- environment -----------------------------------------------------------
# REPLACE THIS BLOCK with the module/conda lines from your existing sbatch
# script. Left deliberately empty rather than guessed: a wrong `module load`
# fails 8 queued tasks in a row, each after waiting for a slot.
# e.g.  module load cuda/12.4
#       source ~/miniconda3/etc/profile.d/conda.sh && conda activate neurotrack-env
# ---------------------------------------------------------------------------

STATE="$CKPT_DIR/train_state.pt"

if [ -f "$STATE" ]; then
    # `epoch` on disk is 0-indexed and already COMPLETED, so a finished
    # 110-epoch run stores 109. Hence epoch+1 >= EPOCHS means done.
    STATUS=$(python - <<PY
import torch
try:
    s = torch.load("$STATE", map_location="cpu", weights_only=False)
    ep   = int(s.get("epoch", -1)) + 1
    stop = int(s.get("epochs_without_improvement", 0))
    print(f"{ep} {stop}")
except Exception as e:
    print(f"-1 0  # {e}")
PY
    )
    DONE_EPOCHS=$(echo "$STATUS" | awk '{print $1}')
    NO_IMPROVE=$(echo "$STATUS" | awk '{print $2}')
    echo "=== task ${SLURM_ARRAY_TASK_ID}: ${DONE_EPOCHS}/${EPOCHS} epochs done, ${NO_IMPROVE} without improvement ==="

    if [ "$DONE_EPOCHS" -ge "$EPOCHS" ] 2>/dev/null; then
        echo "Target reached. Cancelling remaining pending tasks in array ${SLURM_ARRAY_JOB_ID}."
        scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
        exit 0
    fi
    RESUME_ARG="--resume $STATE"
else
    echo "=== task ${SLURM_ARRAY_TASK_ID}: no train_state.pt, starting from scratch ==="
    RESUME_ARG=""
fi

python cli.py train \
    --epochs "$EPOCHS" \
    --batch-size 2 \
    --num-frames 16 \
    --val-num-frames 16 \
    --patience 90 \
    --val-videos A91_C1_2,A31_C1,A52_C2,A22_C2,A42_C1 \
    --checkpoint-dir "$CKPT_DIR" \
    $RESUME_ARG
RC=$?

echo "=== task ${SLURM_ARRAY_TASK_ID} exited with $RC ==="
# A clean exit here means either the wall clock stopped us mid-run (the next
# task resumes) or training finished (the next task sees it and cancels the
# rest). A CRASH would otherwise be retried nine times, so stop the array.
if [ $RC -ne 0 ] && [ $RC -ne 143 ] && [ $RC -ne 137 ]; then
    echo "Non-timeout failure. Cancelling the rest of the array so it does not retry a bug 9 times."
    scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
fi
exit $RC
