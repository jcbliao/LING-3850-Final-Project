#!/bin/bash
#SBATCH --job-name=eval_wug
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
CKPT=${1:-checkpoints/v6_monotonic_no_slots/best_model.pt}

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python evaluate.py \
    --checkpoint "${PROJECT_DIR}/${CKPT}" \
    --data_path "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt" \
    --wug_dir "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1_wugs/"
