#!/bin/bash
#SBATCH --job-name=slot_smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

module load miniconda
conda activate slot_attention

mkdir -p logs checkpoints_test results_test

cd src
echo "=== Smoke Test: 2-epoch training ==="
python train.py --config ../configs/smoke_test.yaml

echo ""
echo "=== Smoke Test: Evaluation ==="
python evaluate.py \
    --checkpoint ../checkpoints_test/best_model.pt \
    --data_path ../data/RevisitPinkerAndPrince/experiment_1/english_merged.txt \
    --wug_dir ../data/RevisitPinkerAndPrince/experiment_1_wugs/

echo ""
echo "=== Smoke test complete ==="
