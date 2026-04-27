#!/bin/bash
#SBATCH --job-name=inspect
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Usage: sbatch scripts/inspect_predictions.sh [checkpoint_path]
# Default: checkpoints/v4_copy_multihead/best_model.pt

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
CKPT=${1:-checkpoints/v4_copy_multihead/best_model.pt}

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python "${PROJECT_DIR}/scripts/inspect_predictions.py" --checkpoint "${PROJECT_DIR}/${CKPT}"
