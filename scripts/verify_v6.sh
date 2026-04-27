#!/bin/bash
#SBATCH --job-name=verify_v6
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00

PROJECT_DIR=/gpfs/radev/project/kuan/jl3795/slot_attention
module load miniconda
conda activate slot_attention

cd "${PROJECT_DIR}/src"
python -c "
import warnings; warnings.filterwarnings('ignore')
from model import SlotAttentionTransducer
import torch

# Test monotonic copy (no slots)
m1 = SlotAttentionTransducer(vocab_size=43, d_model=64, num_slots=4, enc_layers=2, dec_layers=2, d_ff=128, use_copy=True, use_slots=False)
src = torch.randint(1, 43, (2, 8))
tgt = torch.randint(1, 43, (2, 10))
out1 = m1(src, tgt)
out1['loss'].backward()
print(f'v6 no-slots: loss={out1[\"loss\"].item():.4f}')
print(f'  align_log_alpha={m1.decoder.copy_align_log_alpha.item():.4f}')
print(f'  align_log_alpha.grad={m1.decoder.copy_align_log_alpha.grad.item():.6f}')

# Test monotonic copy (with slots)
m2 = SlotAttentionTransducer(vocab_size=43, d_model=64, num_slots=4, enc_layers=2, dec_layers=2, d_ff=128, use_copy=True, use_slots=True, slot_nhead=4)
out2 = m2(src, tgt)
out2['loss'].backward()
print(f'v6 slots: loss={out2[\"loss\"].item():.4f}')

# Test greedy decode
p1 = m1.greedy_decode(src, max_len=12)
p2 = m2.greedy_decode(src, max_len=12)
print(f'Decode: no-slots={p1.shape}, slots={p2.shape}')

print('All OK')
"
