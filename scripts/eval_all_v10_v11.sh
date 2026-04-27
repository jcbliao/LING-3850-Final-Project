#!/bin/bash
#SBATCH --job-name=eval_v10_v11
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
DATA_PATH="${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt"
WUG_DIR="${PROJECT_DIR}/data/RevisitPinkerAndPrince/experiment_1_wugs/"

module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"

for run in v10a_bilstm_no_slots v10b_bilstm_slots v11a_weak_decoder v11b_tight_bottleneck v11c_pretrain v11d_cls_auxiliary; do
    CKPT="${PROJECT_DIR}/checkpoints/${run}/best_model.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=========================================="
        echo "  Evaluating: ${run}"
        echo "=========================================="
        python evaluate.py --checkpoint "$CKPT" --data_path "$DATA_PATH" --wug_dir "$WUG_DIR"
    else
        echo "SKIP ${run}: checkpoint not found at ${CKPT}"
    fi
done
