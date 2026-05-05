#!/bin/bash
#SBATCH --job-name=v20_oversample_noslot_d128_dr0p1_a1_s2
#SBATCH --partition=gpu,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/project/kuan/jl3795/slot_attention/logs/v20_sweep_oversample/%x_%j.out

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
mkdir -p "${PROJECT_DIR}/logs/v20_sweep"

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python train.py --config "${PROJECT_DIR}/configs/sweep_v20_oversample/v20_oversample_noslot_d128_dr0p1_a1_s2.yaml" 2>&1

# Evaluation
SAVE_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${PROJECT_DIR}/configs/sweep_v20_oversample/v20_oversample_noslot_d128_dr0p1_a1_s2.yaml')); print(c.get('save_dir', '../checkpoints'))")
CKPT_PATH="${PROJECT_DIR}/src/${SAVE_DIR}/best_model.pt"

echo ""
echo "=== Evaluation ==="
python evaluate.py \
    --checkpoint "${CKPT_PATH}" \
    --data_path "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt" \
    --wug_dir "${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1_wugs/"
