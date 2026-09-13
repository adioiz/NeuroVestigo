#!/bin/bash
# ---------------------------------------------------------------------------
# The SAM 2 chain, verbatim, with SAM 3 as the starting weights.
#
#   sbatch slurm/train_sam3_chain.sbatch
#
# Same recipe as train_chain.sbatch: 110 epochs, batch 2, 16 frames, dual-rank
# LoRA+ on decoder + memory attention + the last two encoder blocks, one
# 4-hour task at a time, resume-aware, crash-guarded. The only differences:
#   --backbone sam3     build_sam3_video_model().tracker + shared backbone,
#                       facebook/sam3/sam3.pt (SAM 3, not 3.1 -- 3.1 has no
#                       standalone tracker checkpoint); image_size 1008 and
#                       mean/std 0.5 follow automatically
#   sam3-env            SAM 3's dependency set (+ scipy, installed if absent)
#   checkpoints-sam3    fresh dir; these weights are NOT interchangeable with
#                       any SAM 2 checkpoint
#
# The encoder is ~20x Hiera-tiny, so expect a slower step. Read s/step from
# the first log: if it is far above the SAM 2 chain's ~3.2 s, the fallback is
# --precompute --no-hiera-last-stage (frozen features cached once; flips only).
# ---------------------------------------------------------------------------
#SBATCH --job-name=sam3_lora_neurons
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%1
#SBATCH --output=logs/sam3_%A_%a.out
#SBATCH --error=logs/sam3_%A_%a.err
#SBATCH --no-requeue

echo "=== Task ${SLURM_ARRAY_TASK_ID} of array ${SLURM_ARRAY_JOB_ID} started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

source ~/.bashrc
conda activate sam3-env
# the training path needs scipy; sam3-env ships without it
python -c "import scipy" 2>/dev/null || pip install -q scipy

echo "--- Env check ---"
which python
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0))"

cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs

EPOCHS=110
CKPT_DIR=checkpoints-sam3
COMMON_ARGS="--backbone sam3 --epochs $EPOCHS --batch-size 2 --num-frames 16 --val-num-frames 16 --patience 90 --val-videos A91_C1_2,A31_C1,A52_C2,A22_C2,A42_C1 --checkpoint-dir $CKPT_DIR"

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
