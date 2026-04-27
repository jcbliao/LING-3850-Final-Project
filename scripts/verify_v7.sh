#!/bin/bash
#SBATCH --job-name=verify_v7
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

# v7a: LSTM decoder, no slots
m1 = SlotAttentionTransducer(
    vocab_size=43, d_model=128, num_slots=4, enc_layers=2, dec_layers=1,
    d_ff=256, use_slots=False, decoder_type='lstm',
)
src = torch.randint(1, 43, (2, 8))
tgt = torch.randint(1, 43, (2, 10))
out1 = m1(src, tgt)
out1['loss'].backward()
print(f'v7a (LSTM, no slots): loss={out1[\"loss\"].item():.4f}, params={sum(p.numel() for p in m1.parameters()):,}')

# Greedy decode
p1 = m1.greedy_decode(src, max_len=12)
print(f'  decode shape: {p1.shape}')

# v7b: LSTM decoder, with slots
m2 = SlotAttentionTransducer(
    vocab_size=43, d_model=128, num_slots=4, enc_layers=2, dec_layers=1,
    d_ff=256, use_slots=True, decoder_type='lstm', slot_nhead=4,
)
out2 = m2(src, tgt)
out2['loss'].backward()
print(f'v7b (LSTM, slots): loss={out2[\"loss\"].item():.4f}, params={sum(p.numel() for p in m2.parameters()):,}')

p2 = m2.greedy_decode(src, max_len=12)
print(f'  decode shape: {p2.shape}')

# Verify backward compat: Transformer decoder still works
m3 = SlotAttentionTransducer(
    vocab_size=43, d_model=128, num_slots=4, enc_layers=2, dec_layers=2,
    d_ff=256, use_slots=False, decoder_type='transformer', use_copy=True,
)
out3 = m3(src, tgt)
print(f'Transformer+copy (backward compat): loss={out3[\"loss\"].item():.4f}')

print('All OK')
"
