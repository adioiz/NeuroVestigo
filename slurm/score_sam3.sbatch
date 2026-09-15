#!/bin/bash
#SBATCH --job-name=score_sam3
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/score_sam3_%j.out
#SBATCH --error=logs/score_sam3_%j.out
#SBATCH --no-requeue
# Test-set rows for the SAM 3 LoRA run: the IoU-best checkpoint (epoch 38) and
# the topology tie-break (epoch 54, Betti-0 0.750). Inference needs sam3-env;
# score.py needs skimage, which lives in neurotrack-env -- so two envs.
# run_ours.py reads training_config.json beside the checkpoint to know it is
# building SAM 3 at 1008 px with 0.5 normalization.
echo "=== score_sam3 $(date) ==="; hostname
source ~/.bashrc
cd ~/Fine-tune-SAM2-model-for-Neurons-segmentation || exit 1
mkdir -p logs
VIDEOS="A71_C1 A81_C1 A12_C3_100k A12_C1_100k_2"

conda activate sam3-env
for tag in "" "_topo"; do
  name="ours_sam3${tag}"
  echo; echo "##### inference $name"
  python -u benchmark/run_ours.py --ckpt "checkpoints-sam3/lora_weights${tag}.pt" --backbone sam3 \
      --videos $VIDEOS --out-dir "benchmark/predictions/$name" || echo "run_ours failed for $name"
done

conda activate neurotrack-env
for tag in "" "_topo"; do
  name="ours_sam3${tag}"
  echo; echo "##### score $name"
  python benchmark/score.py --pred-dir "benchmark/predictions/$name" --model "$name" \
      --split test --cbdice --skip-first-frame
done

echo; echo "##### summary (ours rows)"
column -s, -t benchmark/results/test/summary.csv | grep -E "^model|^ours"
echo "=== exit $? at $(date) ==="
