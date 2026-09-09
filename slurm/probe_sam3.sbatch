#!/bin/bash
# ---------------------------------------------------------------------------
# One-shot: does SAM 3's tracker accept a mask as a visual prompt?
#
#   sbatch slurm/probe_sam3.sbatch
#   tail -f logs/probe_sam3_<jobid>.out
#
# Batch, not interactive srun. B200-4h is congested and an srun makes you sit
# and watch the queue; this is 20 minutes on one GPU, which is the profile the
# backfill scheduler slots in first.
#
# Needs a GPU: build_sam3_predictor() has no device argument and goes to CUDA
# inside sam3's own builder, so a login-node run dies at construction with
# "Found no NVIDIA driver" before it reaches the mask question at all.
#
# sam3-env, NOT neurotrack-env -- SAM 3 has its own dependency set.
# ---------------------------------------------------------------------------
#SBATCH --job-name=probe_sam3_mask
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/probe_sam3_%j.out
#SBATCH --error=logs/probe_sam3_%j.out
#SBATCH --no-requeue

echo "=== probe_sam3_mask started at $(date) ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source ~/.bashrc
conda activate sam3-env

cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs

python -u benchmark/probe_sam3_mask.py
echo "=== exit $? at $(date) ==="
