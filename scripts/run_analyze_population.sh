#!/bin/bash
#SBATCH --job-name=v17a_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/project/kuan/jl3795/slot_attention/logs/analyze_v17a_%j.log

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python "${PROJECT_DIR}/scripts/analyze_population.py"
