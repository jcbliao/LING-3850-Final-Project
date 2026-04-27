#!/bin/bash
#SBATCH --job-name=slot_attn
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Usage (from any directory): sbatch scripts/run_experiment.sh [config_file]
# Default: configs/default.yaml

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
CONFIG=${1:-configs/default.yaml}

mkdir -p "${PROJECT_DIR}"/{logs,checkpoints,results}

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python train.py --config "${PROJECT_DIR}/${CONFIG}" 2>&1 | tee "${PROJECT_DIR}/logs/train_${SLURM_JOB_ID}.log"

# Extract save_dir from config (default: ../checkpoints)
SAVE_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${PROJECT_DIR}/${CONFIG}')); print(c.get('save_dir', '../checkpoints'))")
# Resolve relative to src/
CKPT_PATH="${PROJECT_DIR}/src/${SAVE_DIR}/best_model.pt"

echo ""
echo "=== Evaluation ==="
python evaluate.py \
    --checkpoint "${CKPT_PATH}" \
    --data_path "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt" \
    --wug_dir "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1_wugs/" \
    2>&1 | tee -a "${PROJECT_DIR}/logs/train_${SLURM_JOB_ID}.log"
