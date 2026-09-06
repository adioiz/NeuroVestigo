#!/bin/bash
# ---------------------------------------------------------------------------
# Chained training on the 4-hour partition: one job at a time, no babysitting.
#
#   sbatch slurm/train_chain.sbatch
#
# This is run.sh plus three things:
#   1. --array=1-10%1  -- ten queued tasks, exactly ONE running at a time.
#      The %1 throttle is the point. Two concurrent tasks would both load and
#      both write train_state.pt, interleaving epochs and silently corrupting
#      optimiser moments and scheduler position. It would not crash; it would
#      just train badly.
#   2. A completion check, so once the run reaches EPOCHS the next task
#      cancels the remaining pending tasks instead of wasting slots.
#   3. A crash guard, so a genuine bug is not retried nine times.
#
# CKPT_DIR is checkpoints-distill-v2, NOT checkpoints/. The latter holds the
# 22-recording model that both repos' weight lookup resolves to; writing there
# would destroy it. It is also not checkpoints-distill, which holds the first
# distillation run -- reusing that directory would resume it at epoch 110 and
# the completion check would cancel the array immediately.
# ---------------------------------------------------------------------------
#SBATCH --job-name=sam2_lora_neurons
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%1
#SBATCH --output=logs/sam2_%A_%a.out
#SBATCH --error=logs/sam2_%A_%a.err
#SBATCH --no-requeue

echo "=== Task ${SLURM_ARRAY_TASK_ID} of array ${SLURM_ARRAY_JOB_ID} started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source ~/.bashrc
conda activate neurotrack-env

echo "--- Env check ---"
which python
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0))"

cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs

EPOCHS=110
CKPT_DIR=checkpoints-distill-v2
COMMON_ARGS="--epochs $EPOCHS --batch-size 2 --num-frames 16 --val-num-frames 16 --patience 90 --val-videos A91_C1_2,A31_C1,A52_C2,A22_C2,A42_C1 --checkpoint-dir $CKPT_DIR"

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

echo "=== Task ${SLURM_ARRAY_TASK_ID} exited with $RC at $(date) ==="

# 143 = SIGTERM and 137 = SIGKILL, i.e. the 4-hour wall clock stopping us
# mid-epoch. That is normal here: the next task resumes from the last epoch
# boundary. Anything else is a real failure, and without this the array would
# retry the same bug nine more times, each after queueing for a slot.
if [ $RC -ne 0 ] && [ $RC -ne 143 ] && [ $RC -ne 137 ]; then
    echo "[chain] Non-timeout failure -- cancelling the rest of the array."
    scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
fi
exit $RC
