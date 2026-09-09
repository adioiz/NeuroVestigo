#!/bin/bash
#SBATCH --job-name=sam3_preflight
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sam3_preflight_%j.out
#SBATCH --error=logs/sam3_preflight_%j.out
#SBATCH --no-requeue
# Can our LoRA recipe train the SAM 3.1 tracker? ~2 minutes, run it before
# queueing anything long. sam3-env: SAM 3 has its own dependency set.
echo "=== sam3_preflight $(date) ==="; hostname
source ~/.bashrc
conda activate sam3-env
cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs
python -u benchmark/sam3_lora_preflight.py
echo "=== exit $? at $(date) ==="
