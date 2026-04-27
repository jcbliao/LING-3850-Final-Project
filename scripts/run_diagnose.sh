#!/bin/bash
#SBATCH --job-name=diagnose_copy
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/diagnose_%j.log

# Diagnose copy mechanism on v5 no-slot baseline
# Usage: sbatch scripts/run_diagnose.sh [checkpoint_path]

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
CHECKPOINT=${1:-${PROJECT_DIR}/checkpoints/v5_baseline_no_slots/best_model.pt}

mkdir -p "${PROJECT_DIR}/logs"

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python ../scripts/diagnose_copy.py --checkpoint "${CHECKPOINT}" 2>&1
