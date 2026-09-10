#!/bin/bash
#SBATCH --job-name=score_23rec
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/score_23rec_%j.out
#SBATCH --error=logs/score_23rec_%j.out
#SBATCH --no-requeue
# Test-set rows for the 23-recording run: the IoU-best checkpoint and the
# topology tie-break one, scored by the same score.py as every other row.
echo "=== score_23rec $(date) ==="; hostname
source ~/.bashrc
conda activate neurotrack-env
cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs

for tag in "" "_topo"; do
  ck="checkpoints/lora_weights${tag}.pt"
  name="ours_23rec${tag}"
  echo; echo "##### $name  ($ck)"
  python benchmark/run_ours.py --ckpt "$ck" --out-dir "benchmark/predictions/$name" || { echo "run_ours failed for $name"; continue; }
  python benchmark/score.py --pred-dir "benchmark/predictions/$name" --model "$name" --split test --cbdice --skip-first-frame
done

echo; echo "##### summary"
column -s, -t benchmark/results/test/summary.csv | grep -E "^model|^ours"
echo "=== exit $? at $(date) ==="
