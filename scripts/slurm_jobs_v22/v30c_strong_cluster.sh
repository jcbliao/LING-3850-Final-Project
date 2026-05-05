#!/bin/bash
#SBATCH --job-name=v30c_strong_cluster
#SBATCH --partition=gpu,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/radev/project/kuan/jl3795/slot_attention/logs/v22/%x_%j.out

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
mkdir -p "${PROJECT_DIR}/logs/v22"

unset CONDA_ENVS_PATH CONDA_PKGS_DIRS CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python train.py --config "${PROJECT_DIR}/configs/v30c_strong_cluster.yaml" 2>&1
