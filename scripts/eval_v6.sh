#!/bin/bash
#SBATCH --job-name=eval_v6
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
DATA_PATH="${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt"
WUG_DIR="${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1_wugs/"

module load miniconda
conda activate slot_attention
cd "${PROJECT_DIR}/src"

for run in v6_monotonic_no_slots v6_monotonic_slots; do
    CKPT="${PROJECT_DIR}/checkpoints/${run}/best_model.pt"
    echo ""
    echo "=========================================="
    echo "  Evaluating: ${run}"
    echo "=========================================="
    python evaluate.py --checkpoint "$CKPT" --data_path "$DATA_PATH" --wug_dir "$WUG_DIR"
done
