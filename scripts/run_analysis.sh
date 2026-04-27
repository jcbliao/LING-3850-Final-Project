#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
DATA_PATH="${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt"

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"

python analyze_slots.py \
    --checkpoint "${PROJECT_DIR}/checkpoints/v13b_bottleneck_balanced/best_model.pt" \
    --data_path "$DATA_PATH" \
    --out_dir "${PROJECT_DIR}/results/v13b_analysis"
