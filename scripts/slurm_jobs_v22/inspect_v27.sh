#!/bin/bash
#SBATCH --job-name=inspect_v27
#SBATCH --partition=gpu,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs/radev/project/kuan/jl3795/slot_attention/logs/v22/%x_%j.out

unset CONDA_ENVS_PATH CONDA_PKGS_DIRS CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL
module load miniconda
conda activate slot_attention

cd /gpfs/radev/project/kuan/jl3795/slot_attention/src
python ../scripts/inspect_v27.py
