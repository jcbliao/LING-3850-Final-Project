#!/bin/bash
#SBATCH --job-name=slot_eval
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
CHECKPOINT=${1:-checkpoints/best_model.pt}

mkdir -p "${PROJECT_DIR}/logs"

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python evaluate.py \
    --checkpoint "${PROJECT_DIR}/${CHECKPOINT}" \
    --data_path "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt" \
    --wug_dir "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1_wugs/" \
    2>&1 | tee "${PROJECT_DIR}/logs/eval_${SLURM_JOB_ID}.log"
