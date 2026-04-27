"""Analyze v17a population decoder neuron confidences and error patterns.

Produces:
  - neuron_confidence_heatmap.png: mean neuron weights for reg vs irreg
  - irregular_predictions.txt: all irregular verb predictions with confidence dists
  - confidence_stats.txt: summary statistics
"""

import sys
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Run from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.vocab import CharVocab
from data.dataset import (
    load_english_merged, load_wug_data, split_data,
    PastTenseDataset, collate_fn,
)
from evaluate import load_checkpoint


def compute_input_confidences(model, src):
    """Compute neuron confidences from encoder output (input mode).

    This avoids relying on _last_confidences which reflects greedy decode
    loop state. Instead, we manually run encoder + each neuron's confidence head.

    Returns:
        confidences: (B, K) raw logits
        weights: (B, K) softmax-normalized
    """
    H = model.encoder(src)  # (B, n, d)

    all_conf = []
    for neuron in model.decoder.neurons:
        # Replicate the input-mode confidence computation from PopulationNeuron
        pad_mask = (src == neuron.pad_idx)
        real_mask = ~pad_mask
        lengths = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (H * real_mask.unsqueeze(-1).float()).sum(dim=1) / lengths
        conf = neuron.confidence_proj(pooled).squeeze(-1)  # (B,)
        all_conf.append(conf)

    confidences = torch.stack(all_conf, dim=1)  # (B, K)
    tau = model.decoder.confidence_tau
    weights = F.softmax(confidences / tau, dim=-1)  # (B, K)
    return confidences, weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_dir = Path("/gpfs/radev/project/kuan/jl3795/slot_attention")
    ckpt_path = project_dir / "checkpoints" / "v17a_population" / "best_model.pt"
    data_path = project_dir / "data" / "RevisitPinkerAndPrince" / "experiment_1" / "english_merged.txt"
    wug_dir = project_dir / "data" / "RevisitPinkerAndPrince" / "experiment_1_wugs"
    out_dir = project_dir / "results" / "v17a_population"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, vocab, config = load_checkpoint(str(ckpt_path), device)
    model.eval()
    num_neurons = model.decoder.num_experts
    print(f"Loaded model with {num_neurons} neurons, confidence_mode={model.decoder.confidence_mode}")

    # Load data
    entries = load_english_merged(str(data_path))
    train_entries, val_entries, test_entries = split_data(entries, seed=config.get("seed", 42))

    use_phon = config.get("use_phonological", True)
    collate_fn_padded = partial(collate_fn, pad_idx=vocab.pad_idx)

    # ----------------------------------------------------------------
    # Analyze test set
    # ----------------------------------------------------------------
    test_ds = PastTenseDataset(test_entries, vocab, use_phon)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn_padded)

    all_results = []
    entry_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            src, tgt = batch[0].to(device), batch[1].to(device)
            B = src.size(0)

            # Get predictions via greedy decode
            preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)

            # Get neuron confidences from encoder (input mode)
            confidences, weights = compute_input_confidences(model, src)

            for i in range(B):
                if entry_idx >= len(test_entries):
                    break
                entry = test_entries[entry_idx]
                src_tokens = vocab.decode(src[i].tolist())
                tgt_tokens = vocab.decode(tgt[i].tolist())
                pred_tokens = vocab.decode(preds[i].tolist())

                src_str = "".join(src_tokens)
                tgt_str = "".join(tgt_tokens)
                pred_str = "".join(pred_tokens)

                is_correct = tgt_str == pred_str
                conf_weights = weights[i].cpu().numpy()  # (K,)
                top_neuron = int(conf_weights.argmax())
                regularity = entry["regularity"]

                all_results.append({
                    "src": src_str,
                    "tgt": tgt_str,
                    "pred": pred_str,
                    "correct": is_correct,
                    "weights": conf_weights,
                    "top_neuron": top_neuron,
                    "regularity": regularity,
                })
                entry_idx += 1

    print(f"Analyzed {len(all_results)} test examples")

    # ----------------------------------------------------------------
    # 1. Heatmap: mean neuron confidence weights for reg vs irreg
    # ----------------------------------------------------------------
    reg_weights = np.array([r["weights"] for r in all_results if r["regularity"] == "reg"])
    irreg_weights = np.array([r["weights"] for r in all_results if r["regularity"] == "irreg"])

    reg_mean = reg_weights.mean(axis=0) if len(reg_weights) > 0 else np.zeros(num_neurons)
    irreg_mean = irreg_weights.mean(axis=0) if len(irreg_weights) > 0 else np.zeros(num_neurons)

    heatmap_data = np.stack([reg_mean, irreg_mean])  # (2, K)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(num_neurons))
    ax.set_xticklabels([f"Neuron {i}" for i in range(num_neurons)])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Regular", "Irregular"])
    ax.set_title("Mean Neuron Confidence Weights by Verb Regularity")

    # Annotate cells
    for i in range(2):
        for j in range(num_neurons):
            ax.text(j, i, f"{heatmap_data[i, j]:.3f}",
                    ha="center", va="center", fontsize=12,
                    color="white" if heatmap_data[i, j] > 0.4 else "black")

    plt.colorbar(im, ax=ax, label="Mean confidence weight")
    plt.tight_layout()
    heatmap_path = out_dir / "neuron_confidence_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    # ----------------------------------------------------------------
    # 2. Irregular verb predictions with neuron confidence distributions
    # ----------------------------------------------------------------
    irreg_results = [r for r in all_results if r["regularity"] == "irreg"]
    irreg_path = out_dir / "irregular_predictions.txt"
    with open(irreg_path, "w") as f:
        f.write(f"{'Source':<15} {'Target':<15} {'Prediction':<15} {'Correct':<8} "
                + " ".join([f"N{i:<6}" for i in range(num_neurons)])
                + f" {'Top':>4}\n")
        f.write("-" * (15 * 3 + 8 + num_neurons * 8 + 6) + "\n")
        for r in sorted(irreg_results, key=lambda x: x["src"]):
            w_str = " ".join([f"{w:.4f}" for w in r["weights"]])
            f.write(f"{r['src']:<15} {r['tgt']:<15} {r['pred']:<15} "
                    f"{'YES' if r['correct'] else 'NO':<8} "
                    f"{w_str}  N{r['top_neuron']}\n")
    print(f"Saved irregular predictions to {irreg_path}")

    # ----------------------------------------------------------------
    # 3. Stats: confidence distribution for correct vs incorrect
    # ----------------------------------------------------------------
    correct_weights = np.array([r["weights"] for r in all_results if r["correct"]])
    incorrect_weights = np.array([r["weights"] for r in all_results if not r["correct"]])

    stats_path = out_dir / "confidence_stats.txt"
    with open(stats_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("POPULATION DECODER NEURON ANALYSIS — v17a\n")
        f.write("=" * 70 + "\n\n")

        # Overall counts
        n_reg = sum(1 for r in all_results if r["regularity"] == "reg")
        n_irreg = sum(1 for r in all_results if r["regularity"] == "irreg")
        n_correct = sum(1 for r in all_results if r["correct"])
        n_total = len(all_results)
        f.write(f"Test set: {n_total} total ({n_reg} regular, {n_irreg} irregular)\n")
        f.write(f"Overall accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}\n")
        reg_correct = sum(1 for r in all_results if r["regularity"] == "reg" and r["correct"])
        irreg_correct = sum(1 for r in all_results if r["regularity"] == "irreg" and r["correct"])
        f.write(f"Regular accuracy: {reg_correct}/{n_reg} = {reg_correct/n_reg:.1%}\n")
        f.write(f"Irregular accuracy: {irreg_correct}/{n_irreg} = {irreg_correct/n_irreg:.1%}\n")
        balanced = (reg_correct / n_reg + irreg_correct / n_irreg) / 2 if n_irreg > 0 else 0
        f.write(f"Balanced accuracy: {balanced:.1%}\n\n")

        # Mean confidence by correctness
        f.write("-" * 50 + "\n")
        f.write("Mean neuron confidence weights: CORRECT predictions\n")
        f.write("-" * 50 + "\n")
        if len(correct_weights) > 0:
            for j in range(num_neurons):
                f.write(f"  Neuron {j}: {correct_weights[:, j].mean():.4f} "
                        f"(std {correct_weights[:, j].std():.4f})\n")
        else:
            f.write("  No correct predictions.\n")

        f.write(f"\n")
        f.write("-" * 50 + "\n")
        f.write("Mean neuron confidence weights: INCORRECT predictions\n")
        f.write("-" * 50 + "\n")
        if len(incorrect_weights) > 0:
            for j in range(num_neurons):
                f.write(f"  Neuron {j}: {incorrect_weights[:, j].mean():.4f} "
                        f"(std {incorrect_weights[:, j].std():.4f})\n")
        else:
            f.write("  No incorrect predictions.\n")

        # Mean confidence by regularity
        f.write(f"\n")
        f.write("-" * 50 + "\n")
        f.write("Mean neuron confidence weights: REGULAR verbs\n")
        f.write("-" * 50 + "\n")
        for j in range(num_neurons):
            f.write(f"  Neuron {j}: {reg_mean[j]:.4f}\n")

        f.write(f"\n")
        f.write("-" * 50 + "\n")
        f.write("Mean neuron confidence weights: IRREGULAR verbs\n")
        f.write("-" * 50 + "\n")
        for j in range(num_neurons):
            f.write(f"  Neuron {j}: {irreg_mean[j]:.4f}\n")

        # ----------------------------------------------------------------
        # 4. Per-neuron: what % of its "top-confidence" verbs are reg vs irreg
        # ----------------------------------------------------------------
        f.write(f"\n")
        f.write("=" * 50 + "\n")
        f.write("Per-neuron top-confidence verb breakdown\n")
        f.write("=" * 50 + "\n")
        for j in range(num_neurons):
            top_for_j = [r for r in all_results if r["top_neuron"] == j]
            n_top = len(top_for_j)
            if n_top == 0:
                f.write(f"\nNeuron {j}: 0 verbs (never top)\n")
                continue
            n_reg_j = sum(1 for r in top_for_j if r["regularity"] == "reg")
            n_irreg_j = sum(1 for r in top_for_j if r["regularity"] == "irreg")
            n_correct_j = sum(1 for r in top_for_j if r["correct"])
            f.write(f"\nNeuron {j}: {n_top} verbs where it has highest confidence\n")
            f.write(f"  Regular:   {n_reg_j}/{n_top} = {n_reg_j/n_top:.1%}\n")
            f.write(f"  Irregular: {n_irreg_j}/{n_top} = {n_irreg_j/n_top:.1%}\n")
            f.write(f"  Correct:   {n_correct_j}/{n_top} = {n_correct_j/n_top:.1%}\n")

        # Also show neuron confidence entropy (how peaked the distribution is)
        f.write(f"\n")
        f.write("=" * 50 + "\n")
        f.write("Confidence distribution entropy (per example)\n")
        f.write("=" * 50 + "\n")
        all_w = np.array([r["weights"] for r in all_results])
        entropies = -(all_w * np.log(all_w + 1e-10)).sum(axis=1)
        max_entropy = np.log(num_neurons)
        f.write(f"Mean entropy: {entropies.mean():.4f} (max possible: {max_entropy:.4f})\n")
        f.write(f"Std entropy:  {entropies.std():.4f}\n")
        f.write(f"Min entropy:  {entropies.min():.4f}\n")
        f.write(f"Max entropy:  {entropies.max():.4f}\n")

        reg_ent = entropies[[i for i, r in enumerate(all_results) if r["regularity"] == "reg"]]
        irreg_ent = entropies[[i for i, r in enumerate(all_results) if r["regularity"] == "irreg"]]
        f.write(f"\nRegular mean entropy:   {reg_ent.mean():.4f}\n")
        f.write(f"Irregular mean entropy: {irreg_ent.mean():.4f}\n")

    print(f"Saved stats to {stats_path}")

    # Print summary to stdout
    with open(stats_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
