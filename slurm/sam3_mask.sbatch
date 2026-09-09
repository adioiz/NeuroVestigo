#!/bin/bash
# ---------------------------------------------------------------------------
# Drive SAM 3.1 with a frame-0 mask prompt. One job tries all three routes.
#
#   sbatch slurm/sam3_mask.sbatch
#   tail -f logs/sam3_mask_<jobid>.out
#
# 40 minutes: route B may transplant weights and rebuild a tracker, and a
# failed route still runs its diagnostics before the next one starts.
# ---------------------------------------------------------------------------
#SBATCH --job-name=sam3_mask
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sam3_mask_%j.out
#SBATCH --error=logs/sam3_mask_%j.out
#SBATCH --no-requeue

echo "=== sam3_mask started at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/.bashrc
conda activate sam3-env
cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs

python -u benchmark/run_sam3_mask.py --videos A81_C1
echo "=== exit $? at $(date) ==="
