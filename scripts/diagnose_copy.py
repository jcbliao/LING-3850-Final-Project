"""Diagnostic script for the copy mechanism.

Loads a checkpoint and runs a few examples through the model,
printing detailed diagnostics at each decoding step:
- p_gen (generate vs copy gate) values
- copy attention weights over source positions
- which source character has the highest copy weight
- the predicted token vs what copy would have produced
- whether the model is generating from vocab or copying

Supports both slot-based models (v4) and no-slot baselines (v5).

Usage (needs GPU):
    cd src && python ../scripts/diagnose_copy.py [--checkpoint PATH]

Or via SLURM:
    sbatch scripts/run_diagnose.sh
"""

import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from data.vocab import CharVocab
from data.dataset import load_english_merged, split_data, PastTenseDataset
from model import SlotAttentionTransducer


DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "v4_copy_multihead" / "best_model.pt"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "RevisitPinkerAndPrince" / "experiment_1" / "english_merged.txt"


def load_checkpoint(path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    config = ckpt["config"]

    model = SlotAttentionTransducer(
        vocab_size=len(vocab),
        d_model=config.get("d_model", 128),
        nhead=config.get("nhead", 4),
        enc_layers=config.get("enc_layers", 3),
        dec_layers=config.get("dec_layers", 3),
        d_ff=config.get("d_ff", 256),
        num_slots=config.get("num_slots", 8),
        slot_iters=config.get("slot_iters", 3),
        dropout=config.get("dropout", 0.1),
        pad_idx=vocab.pad_idx,
        lambda_l0=config.get("lambda_l0", 0.01),
        alpha_recon=config.get("alpha_recon", 0.0),
        l0_beta=config.get("l0_beta", 0.66),
        l0_mode=config.get("l0_mode", "conditional"),
        target_l0=config.get("target_l0", 2.0),
        lagrangian_lr=config.get("lagrangian_lr", 0.01),
        use_copy=config.get("use_copy", False),
        slot_nhead=config.get("slot_nhead", 1),
        use_slots=config.get("use_slots", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, config


def decode_with_diagnostics(model, src, vocab, device, max_len=32):
    """Run greedy decoding with full copy mechanism diagnostics at each step."""
    model.eval()
    decoder = model.decoder

    B = src.size(0)
    assert B == 1, "Diagnostics designed for single examples"

    # Encode -> (Slot Attention -> L0Drop) or direct
    H = model.encoder(src)                # (1, n, d)

    if model.use_slots:
        slots = model.slot_attention(H)       # (1, K, d)
        memory = model.l0drop(slots)          # (1, K, d) sparse
    else:
        memory = H                            # (1, n, d) — no-slot baseline

    src_chars = vocab.decode(src[0].tolist())
    n_src = src.size(1)

    print(f"\n{'='*80}")
    print(f"Source tokens: {src_chars}")
    print(f"Source indices: {src[0].tolist()}")
    print(f"Encoder output shape: {H.shape}")
    print(f"Memory shape: {memory.shape}")
    print(f"use_slots: {model.use_slots}")

    if model.use_slots:
        # Check slot magnitudes after L0Drop
        slot_norms = memory[0].norm(dim=-1)
        print(f"\nSlot norms after L0Drop: {slot_norms.tolist()}")
        print(f"  (near-zero = pruned slot)")
    else:
        print(f"\nNo-slot mode: decoder cross-attends directly to encoder output")

    # Step-by-step greedy decoding with diagnostics
    generated = torch.full((1, 1), vocab.sos_idx, dtype=torch.long, device=device)
    step_diagnostics = []

    for step in range(max_len - 1):
        # --- Replicate decoder forward pass to extract internals ---
        tgt = generated
        m = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(m, m, device=device, dtype=torch.bool), diagonal=1
        )
        tgt_pad_mask = (tgt == decoder.pad_idx)

        tgt_emb = decoder.embedding(tgt) * math.sqrt(decoder.d_model)
        x = decoder.pos_enc(tgt_emb)
        x = decoder.transformer(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        # Generate distribution
        gen_logits = decoder.output_proj(x)  # (1, m, V)
        p_vocab = F.softmax(gen_logits, dim=-1)

        # Copy attention
        copy_query = decoder.copy_query_proj(x)  # (1, m, d)
        copy_key = decoder.copy_key_proj(H)  # (1, n, d)
        copy_energy = torch.bmm(copy_query, copy_key.transpose(1, 2))  # (1, m, n)
        src_pad_mask = (src == decoder.pad_idx).unsqueeze(1)  # (1, 1, n)
        copy_energy = copy_energy.masked_fill(src_pad_mask, -1e9)
        copy_attn = F.softmax(copy_energy, dim=-1)  # (1, m, n)

        # Scatter copy attention
        p_copy = torch.zeros_like(p_vocab)  # (1, m, V)
        src_expanded = src.unsqueeze(1).expand(-1, m, -1)  # (1, m, n)
        p_copy.scatter_add_(2, src_expanded, copy_attn)

        # Copy context for gate
        copy_context = torch.bmm(copy_attn, H)  # (1, m, d)

        # p_gen gate
        gate_input = torch.cat([x, copy_context, tgt_emb], dim=-1)
        p_gen = torch.sigmoid(decoder.p_gen_linear(gate_input))  # (1, m, 1)

        # Final distribution
        p_final = p_gen * p_vocab + (1 - p_gen) * p_copy

        # Extract diagnostics for the LAST position (current step)
        p_gen_val = p_gen[0, -1, 0].item()
        copy_attn_last = copy_attn[0, -1]  # (n,)
        top_copy_pos = copy_attn_last.argmax().item()
        top_copy_weight = copy_attn_last[top_copy_pos].item()

        # What would copy produce?
        copy_token_idx = src[0, top_copy_pos].item()
        copy_token = vocab.idx2token[copy_token_idx]

        # What does vocab generate?
        vocab_token_idx = p_vocab[0, -1].argmax().item()
        vocab_token = vocab.idx2token[vocab_token_idx]
        vocab_prob = p_vocab[0, -1, vocab_token_idx].item()

        # Final prediction
        final_token_idx = p_final[0, -1].argmax().item()
        final_token = vocab.idx2token[final_token_idx]
        final_prob = p_final[0, -1, final_token_idx].item()

        # Copy attention distribution over source
        copy_attn_list = copy_attn_last.tolist()

        diag = {
            "step": step,
            "p_gen": p_gen_val,
            "top_copy_pos": top_copy_pos,
            "top_copy_weight": top_copy_weight,
            "copy_token": copy_token,
            "vocab_token": vocab_token,
            "vocab_prob": vocab_prob,
            "final_token": final_token,
            "final_prob": final_prob,
            "copy_attn": copy_attn_list,
        }
        step_diagnostics.append(diag)

        # Greedy next token
        next_token = p_final[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == vocab.eos_idx:
            break

    return generated, step_diagnostics


def print_diagnostics(step_diagnostics, src_chars):
    """Pretty-print decoding diagnostics."""
    print(f"\n{'='*80}")
    print(f"STEP-BY-STEP DECODING DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"{'Step':>4} | {'p_gen':>6} | {'Final':>8} | {'Vocab':>8} | {'Copy':>8} | "
          f"{'CopyPos':>7} | {'CopyWt':>6} | {'CopySrc':>7}")
    print(f"{'-'*4:>4}-+-{'-'*6:>6}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-"
          f"{'-'*7:>7}-+-{'-'*6:>6}-+-{'-'*7:>7}")

    for d in step_diagnostics:
        src_char_at_pos = src_chars[d["top_copy_pos"]] if d["top_copy_pos"] < len(src_chars) else "?"
        print(f"{d['step']:>4} | {d['p_gen']:>6.3f} | {d['final_token']:>8} | "
              f"{d['vocab_token']:>8} | {d['copy_token']:>8} | "
              f"{d['top_copy_pos']:>7} | {d['top_copy_weight']:>6.3f} | "
              f"{src_char_at_pos:>7}")

    # Summary statistics
    p_gens = [d["p_gen"] for d in step_diagnostics]
    print(f"\n--- Summary ---")
    print(f"p_gen mean: {sum(p_gens)/len(p_gens):.4f}")
    print(f"p_gen min:  {min(p_gens):.4f}")
    print(f"p_gen max:  {max(p_gens):.4f}")
    print(f"p_gen > 0.5 (generate-dominant): {sum(1 for p in p_gens if p > 0.5)}/{len(p_gens)} steps")
    print(f"p_gen < 0.5 (copy-dominant):     {sum(1 for p in p_gens if p < 0.5)}/{len(p_gens)} steps")

    if all(p > 0.9 for p in p_gens):
        print("\n*** WARNING: p_gen is near 1.0 at all steps — model is NOT using copy! ***")
        print("    The copy mechanism is effectively disabled.")
    elif all(p < 0.1 for p in p_gens):
        print("\n*** WARNING: p_gen is near 0.0 at all steps — model is ONLY copying! ***")
        print("    The model never generates from vocabulary.")


def print_copy_attention_heatmap(step_diagnostics, src_chars):
    """Print a text-based heatmap of copy attention over source positions."""
    print(f"\n{'='*80}")
    print(f"COPY ATTENTION HEATMAP (rows=decode steps, cols=source positions)")
    print(f"{'='*80}")

    # Header: source characters
    header = "     | " + " ".join(f"{c:>5}" for c in src_chars)
    print(header)
    print("-" * len(header))

    for d in step_diagnostics:
        attn = d["copy_attn"][:len(src_chars)]
        row = f"t={d['step']:>2} | " + " ".join(f"{a:>5.2f}" for a in attn)
        print(row)


def check_copy_gate_params(model):
    """Inspect the learned parameters of the copy gate."""
    decoder = model.decoder
    print(f"\n{'='*80}")
    print("COPY GATE PARAMETERS")
    print(f"{'='*80}")

    # p_gen_linear: weight and bias
    w = decoder.p_gen_linear.weight.data  # (1, 3*d_model)
    b = decoder.p_gen_linear.bias.data    # (1,)
    print(f"p_gen_linear weight shape: {w.shape}")
    print(f"p_gen_linear bias: {b.item():.4f}")
    print(f"  (positive bias pushes p_gen toward 1 = always generate)")
    print(f"  (negative bias pushes p_gen toward 0 = always copy)")

    d = decoder.d_model
    w_decoder_state = w[0, :d]
    w_copy_context = w[0, d:2*d]
    w_tgt_emb = w[0, 2*d:]
    print(f"\nWeight norms by input component:")
    print(f"  decoder_state: {w_decoder_state.norm():.4f}")
    print(f"  copy_context:  {w_copy_context.norm():.4f}")
    print(f"  tgt_embedding: {w_tgt_emb.norm():.4f}")

    # copy_query_proj
    w_copy = decoder.copy_query_proj.weight.data
    print(f"\ncopy_query_proj weight norm: {w_copy.norm():.4f}")
    print(f"copy_query_proj weight shape: {w_copy.shape}")


def check_encoder_slot_alignment(model, src, vocab, device):
    """Check whether encoder output and slot memory live in similar spaces."""
    model.eval()
    H = model.encoder(src)

    if model.use_slots:
        slots = model.slot_attention(H)
        memory = model.l0drop(slots)
    else:
        memory = H

    print(f"\n{'='*80}")
    print("REPRESENTATION ALIGNMENT CHECK")
    print(f"{'='*80}")

    # Cosine similarity between encoder states and decoder copy query projection
    # The copy mechanism does: query = copy_query_proj(decoder_state)
    # Then: energy = query @ encoder_out.T
    # For this to work, copy_query_proj must project decoder states into
    # the SAME space as encoder outputs.
    H_norm = H[0].norm(dim=-1)
    print(f"Encoder output norms (per position): {H_norm.tolist()}")
    print(f"  mean: {H_norm.mean():.4f}, std: {H_norm.std():.4f}")

    mem_norm = memory[0].norm(dim=-1)
    print(f"Memory norms: {mem_norm.tolist()}")
    if model.use_slots:
        print(f"  (these are slot norms after L0Drop)")
    else:
        print(f"  (these are encoder position norms — no slot bottleneck)")


def main():
    parser = argparse.ArgumentParser(description="Diagnose copy mechanism")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint).resolve()
    print(f"Loading checkpoint from: {checkpoint_path}")
    model, vocab, config = load_checkpoint(str(checkpoint_path), device)
    print(f"Vocab size: {len(vocab)}")
    print(f"use_copy: {config.get('use_copy', False)}")
    print(f"use_slots: {config.get('use_slots', True)}")
    print(f"d_model: {config.get('d_model', 128)}")
    print(f"num_slots: {config.get('num_slots', 8)}")

    # Verify copy mechanism is present
    assert model.decoder.use_copy, "Model does not have copy mechanism enabled!"
    print("Copy mechanism: ENABLED")

    # Check gate parameters
    check_copy_gate_params(model)

    # Load data
    entries = load_english_merged(str(DATA_PATH))
    train_entries, val_entries, test_entries = split_data(entries, seed=config.get("seed", 42))

    use_phon = config.get("use_phonological", True)

    # Select a mix of test examples: some regular, some irregular
    reg_examples = [e for e in test_entries if e["regularity"] == "reg"][:3]
    irreg_examples = [e for e in test_entries if e["regularity"] == "irreg"][:3]
    examples = reg_examples + irreg_examples

    for i, entry in enumerate(examples):
        if use_phon:
            src_seq = entry["phon_src"]
            tgt_seq = entry["phon_tgt"]
        else:
            src_seq = list(entry["orth_src"])
            tgt_seq = list(entry["orth_tgt"])

        src_ids = vocab.encode(src_seq, add_sos=False, add_eos=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        regularity = entry["regularity"]
        print(f"\n\n{'#'*80}")
        print(f"EXAMPLE {i+1} ({regularity}): {src_seq} -> {tgt_seq}")
        print(f"{'#'*80}")

        # Check representation alignment
        check_encoder_slot_alignment(model, src_tensor, vocab, device)

        # Decode with diagnostics
        generated, diagnostics = decode_with_diagnostics(
            model, src_tensor, vocab, device
        )

        pred_chars = vocab.decode(generated[0].tolist())
        print(f"\nPrediction: {pred_chars}")
        print(f"Target:     {tgt_seq}")
        print(f"Match: {'YES' if pred_chars == tgt_seq else 'NO'}")

        src_chars_decoded = vocab.decode(src_tensor[0].tolist())
        print_diagnostics(diagnostics, src_chars_decoded)
        print_copy_attention_heatmap(diagnostics, src_chars_decoded)

    # Final summary: check if there's a systematic issue
    print(f"\n\n{'#'*80}")
    print("OVERALL DIAGNOSIS")
    print(f"{'#'*80}")
    print("""
Things to check in the output above:

1. p_gen VALUES:
   - If p_gen ~ 1.0 everywhere: copy mechanism is dead, model only generates.
     Fix: initialize p_gen bias negative, or add copy pretraining.
   - If p_gen ~ 0.5 everywhere: gate hasn't learned, model is confused.
   - Ideal: p_gen low (copy) for stem chars, p_gen high (generate) for suffix.

2. COPY ATTENTION:
   - Should show diagonal-ish pattern (position i copies from ~position i).
   - If attention is uniform/random: copy_query_proj isn't learning alignment.
   - If attention is always on position 0 or the last position: degenerate.

3. SCRAMBLED OUTPUT (the reported bug):
   - If copy attention is correct but p_gen ~ 1.0: model ignores copy weights.
   - If copy attention is wrong (non-monotonic): alignment is broken.
     This could mean the copy query projection and encoder output are
     in different representation spaces (encoder output goes through
     slot attention which transforms it, but copy attends to PRE-slot H).

4. SLOT NORMS:
   - If some slots have near-zero norms: L0Drop is pruning (good).
   - If all slots have similar norms: L0Drop may not be active.
""")


if __name__ == "__main__":
    main()
