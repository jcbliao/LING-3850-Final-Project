#!/bin/bash
#SBATCH --job-name=eval_moe
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Usage: sbatch scripts/eval_only.sh <checkpoint_path>
PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
CKPT=${1}

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python evaluate.py \
    --checkpoint "${CKPT}" \
    --data_path "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt" \
    --wug_dir "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1_wugs/" \
    2>&1 | tee "${PROJECT_DIR}/logs/eval_$(basename $(dirname ${CKPT}))_${SLURM_JOB_ID}.log"
