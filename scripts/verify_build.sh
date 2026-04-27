#!/bin/bash
#SBATCH --job-name=verify
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

# Test v4-style (copy + multihead + slots) with fixed copy mechanism
m1 = SlotAttentionTransducer(vocab_size=43, d_model=64, num_slots=4, enc_layers=2, dec_layers=2, d_ff=128, use_copy=True, slot_nhead=4)
src = torch.randint(1, 43, (2, 8))
tgt = torch.randint(1, 43, (2, 10))
out1 = m1(src, tgt)
out1['loss'].backward()
print(f'v4 (copy+slots): loss={out1[\"loss\"].item():.4f} OK')

# Test no-slot baseline with copy
m2 = SlotAttentionTransducer(vocab_size=43, d_model=64, num_slots=4, enc_layers=2, dec_layers=2, d_ff=128, use_copy=True, use_slots=False)
out2 = m2(src, tgt)
out2['loss'].backward()
print(f'v5 (copy, no slots): loss={out2[\"loss\"].item():.4f} OK')

# Test greedy decode for both
p1 = m1.greedy_decode(src, max_len=12)
p2 = m2.greedy_decode(src, max_len=12)
print(f'Decode shapes: v4={p1.shape}, v5={p2.shape} OK')

# Verify p_gen bias initialization
print(f'p_gen bias: {m1.decoder.p_gen_linear.bias.item():.1f} (should be -2.0)')

# Verify copy key projection exists
print(f'Has copy_key_proj: {hasattr(m1.decoder, \"copy_key_proj\")}')

print('All OK')
"
