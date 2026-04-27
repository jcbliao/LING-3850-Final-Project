"""Error analysis and slot activation visualization for v13b.

Produces:
1. Per-verb error analysis (which irregulars are correct/wrong + predictions)
2. Slot gate activation heatmap (regular vs irregular verbs)
3. Per-slot activation statistics
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.vocab import CharVocab
from data.dataset import (
    load_english_merged, split_data, PastTenseDataset, collate_fn,
)
from model import SlotAttentionTransducer


def load_checkpoint(path, device):
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
        decoder_type=config.get("decoder_type", "transformer"),
        encoder_type=config.get("encoder_type", "transformer"),
        use_slots=config.get("use_slots", True),
        alpha_cls=config.get("alpha_cls", 0.0),
        dec_bottleneck=config.get("dec_bottleneck", 0),
        lstm_hidden=config.get("lstm_hidden", 0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, config


@torch.no_grad()
def extract_gates_and_predictions(model, entries, vocab, device):
    """Run inference on entries, extract predictions and L0 gate values."""
    results = []
    for entry in entries:
        src_ids = vocab.encode(entry["phon_src"], add_sos=False, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)

        # Forward through encoder + slot attention + L0
        H = model.encoder(src)
        slots = model.slot_attention(H)
        memory = model.l0drop(slots)

        # Extract gate values — method depends on l0drop type
        l0 = model.l0drop
        if hasattr(l0, '_last_log_alpha'):
            # InputConditionalL0Drop
            log_alpha = l0._last_log_alpha
            s = torch.sigmoid(log_alpha)
            z = s * (l0.zeta - l0.gamma) + l0.gamma
            gate_values = z.clamp(0.0, 1.0).squeeze(0).cpu().numpy()
        elif hasattr(l0, '_last_scores'):
            # TopKDrop — use the hard mask (which slots were selected)
            scores = l0._last_scores.squeeze(0)  # (K,)
            _, topk_idx = scores.topk(l0.k)
            gate_values = np.zeros(scores.shape[0])
            gate_values[topk_idx.cpu().numpy()] = 1.0
        elif hasattr(l0, '_last_logits'):
            # GumbelSlotRouter — use hard top-k at eval
            logits = l0._last_logits.squeeze(0)
            _, topk_idx = logits.topk(l0.k)
            gate_values = np.zeros(logits.shape[0])
            gate_values[topk_idx.cpu().numpy()] = 1.0
        else:
            gate_values = np.ones(memory.shape[1])

        # Greedy decode
        preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)
        pred_tokens = vocab.decode(preds[0].tolist())
        tgt_tokens = entry["phon_tgt"]

        results.append({
            "src": "".join(entry["phon_src"]),
            "tgt": "".join(tgt_tokens),
            "pred": "".join(pred_tokens),
            "correct": pred_tokens == tgt_tokens,
            "regularity": entry["regularity"],
            "gate_values": gate_values,
        })
    return results


def error_analysis(results, out_dir):
    """Print and save detailed error analysis."""
    reg = [r for r in results if r["regularity"] == "reg"]
    irreg = [r for r in results if r["regularity"] == "irreg"]

    lines = []
    lines.append("=" * 60)
    lines.append("ERROR ANALYSIS")
    lines.append("=" * 60)

    # Irregular analysis (the interesting part)
    lines.append(f"\n--- IRREGULAR VERBS ({sum(r['correct'] for r in irreg)}/{len(irreg)} correct) ---\n")
    for r in sorted(irreg, key=lambda x: x["correct"], reverse=True):
        status = "CORRECT" if r["correct"] else "WRONG"
        lines.append(f"  [{status:7s}] {r['src']:12s} → {r['tgt']:12s} (pred: {r['pred']})")
        lines.append(f"           gates: [{', '.join(f'{g:.3f}' for g in r['gate_values'])}]")

    # Regular errors
    reg_wrong = [r for r in reg if not r["correct"]]
    lines.append(f"\n--- REGULAR VERBS ({sum(r['correct'] for r in reg)}/{len(reg)} correct) ---")
    lines.append(f"    {len(reg_wrong)} errors:\n")
    for r in reg_wrong[:20]:  # show first 20
        lines.append(f"  [WRONG  ] {r['src']:12s} → {r['tgt']:12s} (pred: {r['pred']})")

    if len(reg_wrong) > 20:
        lines.append(f"  ... and {len(reg_wrong) - 20} more")

    text = "\n".join(lines)
    print(text)

    with open(out_dir / "error_analysis.txt", "w") as f:
        f.write(text)


def slot_visualization(results, out_dir):
    """Visualize slot activation patterns."""
    reg = [r for r in results if r["regularity"] == "reg"]
    irreg = [r for r in results if r["regularity"] == "irreg"]
    K = len(results[0]["gate_values"])

    reg_gates = np.array([r["gate_values"] for r in reg])    # (N_reg, K)
    irreg_gates = np.array([r["gate_values"] for r in irreg])  # (N_irreg, K)

    # --- Plot 1: Mean gate values per slot, regular vs irregular ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    reg_mean = reg_gates.mean(axis=0)
    irreg_mean = irreg_gates.mean(axis=0)
    x = np.arange(K)
    width = 0.35

    axes[0].bar(x - width/2, reg_mean, width, label="Regular", color="steelblue")
    axes[0].bar(x + width/2, irreg_mean, width, label="Irregular", color="coral")
    axes[0].set_xlabel("Slot")
    axes[0].set_ylabel("Mean Gate Value")
    axes[0].set_title("Mean Slot Activation by Verb Type")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Slot {i}" for i in range(K)])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Gate value distributions (violin/box) ---
    all_gates = np.array([r["gate_values"] for r in results])
    all_labels = [r["regularity"] for r in results]

    for slot_idx in range(K):
        reg_vals = reg_gates[:, slot_idx]
        irreg_vals = irreg_gates[:, slot_idx]
        pos = slot_idx
        bp_reg = axes[1].boxplot([reg_vals], positions=[pos - 0.2], widths=0.3,
                                  patch_artist=True, showfliers=False)
        bp_irreg = axes[1].boxplot([irreg_vals], positions=[pos + 0.2], widths=0.3,
                                    patch_artist=True, showfliers=False)
        bp_reg["boxes"][0].set_facecolor("steelblue")
        bp_irreg["boxes"][0].set_facecolor("coral")

    axes[1].set_xlabel("Slot")
    axes[1].set_ylabel("Gate Value")
    axes[1].set_title("Gate Value Distribution by Slot")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Slot {i}" for i in range(K)])
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Correct vs incorrect irregulars ---
    irreg_correct = np.array([r["gate_values"] for r in irreg if r["correct"]])
    irreg_wrong = np.array([r["gate_values"] for r in irreg if not r["correct"]])

    if len(irreg_correct) > 0 and len(irreg_wrong) > 0:
        axes[2].bar(x - width/2, irreg_correct.mean(axis=0), width,
                    label="Correct irreg", color="green", alpha=0.7)
        axes[2].bar(x + width/2, irreg_wrong.mean(axis=0), width,
                    label="Wrong irreg", color="red", alpha=0.7)
        axes[2].set_xlabel("Slot")
        axes[2].set_ylabel("Mean Gate Value")
        axes[2].set_title("Slot Activation: Correct vs Wrong Irregulars")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f"Slot {i}" for i in range(K)])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "slot_activations.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: Heatmap of gate values for all irregular verbs ---
    fig, ax = plt.subplots(figsize=(8, max(4, len(irreg) * 0.4)))
    irreg_sorted = sorted(irreg, key=lambda r: r["correct"], reverse=True)
    gate_matrix = np.array([r["gate_values"] for r in irreg_sorted])
    labels = [f"{'OK' if r['correct'] else 'X ':s} {r['src']}→{r['tgt']}"
              for r in irreg_sorted]

    im = ax.imshow(gate_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"Slot {i}" for i in range(K)])
    ax.set_title("L0 Gate Values per Irregular Verb")
    plt.colorbar(im, ax=ax, label="Gate value")
    fig.tight_layout()
    fig.savefig(out_dir / "irregular_heatmap.png", dpi=150)
    plt.close(fig)

    # --- Stats ---
    stats = {
        "reg_mean_gates": reg_mean.tolist(),
        "irreg_mean_gates": irreg_mean.tolist(),
        "gate_diff_per_slot": (irreg_mean - reg_mean).tolist(),
    }
    if len(irreg_correct) > 0:
        stats["irreg_correct_mean_gates"] = irreg_correct.mean(axis=0).tolist()
    if len(irreg_wrong) > 0:
        stats["irreg_wrong_mean_gates"] = irreg_wrong.mean(axis=0).tolist()

    with open(out_dir / "slot_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSlot activation stats:")
    print(f"  Regular mean gates:   [{', '.join(f'{g:.3f}' for g in reg_mean)}]")
    print(f"  Irregular mean gates: [{', '.join(f'{g:.3f}' for g in irreg_mean)}]")
    print(f"  Difference (irr-reg): [{', '.join(f'{g:+.3f}' for g in (irreg_mean - reg_mean))}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="../results/analysis")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocab, config = load_checkpoint(args.checkpoint, device)
    entries = load_english_merged(args.data_path)
    _, _, test_entries = split_data(entries, seed=config.get("seed", 42))

    print(f"Analyzing {len(test_entries)} test entries...")
    results = extract_gates_and_predictions(model, test_entries, vocab, device)

    error_analysis(results, out_dir)
    slot_visualization(results, out_dir)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
