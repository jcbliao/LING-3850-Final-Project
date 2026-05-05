#!/bin/bash
#SBATCH --job-name=v21_smoke
#SBATCH --partition=gpu,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/radev/project/kuan/jl3795/slot_attention/logs/v21_smoke_%j.out

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python train.py --config "${PROJECT_DIR}/configs/v21_smoke.yaml"
