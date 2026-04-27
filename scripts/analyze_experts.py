"""Analyze expert routing and error patterns for the v16e MoE model.

Produces:
- expert_heatmap.png: expert assignment distribution for reg vs irreg
- irregular_by_expert.txt: all irregular verb predictions grouped by expert
- expert_stats.txt: utilization stats and specific correct/incorrect irregulars
"""

import sys
from functools import partial
from pathlib import Path
from collections import defaultdict

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.vocab import CharVocab
from data.dataset import (
    load_english_merged, load_wug_data, split_data,
    PastTenseDataset, collate_fn,
)
from model import SlotAttentionTransducer


# -- Paths --
PROJECT_DIR = Path("/gpfs/radev/project/kuan/jl3795/slot_attention")
CKPT_PATH = PROJECT_DIR / "checkpoints" / "v16e_moe_gumbel_diversity" / "best_model.pt"
DATA_PATH = PROJECT_DIR / "data" / "RevisitPinkerAndPrince" / "experiment_1" / "english_merged.txt"
WUG_DIR = PROJECT_DIR / "data" / "RevisitPinkerAndPrince" / "experiment_1_wugs"
OUT_DIR = PROJECT_DIR / "results" / "v16e_moe_gumbel_diversity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint(path, device):
    """Load model from checkpoint (mirrors evaluate.py logic with key remapping)."""
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
        num_experts=config.get("num_experts", 4),
        expert_hidden=config.get("expert_hidden", 64),
        routing_mode=config.get("routing_mode", "soft"),
        gumbel_tau=config.get("gumbel_tau", 1.0),
        lambda_balance=config.get("lambda_balance", 0.01),
        alpha_cls_router=config.get("alpha_cls_router", 0.0),
        lambda_diversity=config.get("lambda_diversity", 0.0),
        confidence_mode=config.get("confidence_mode", "input"),
        neuron_dropout=config.get("neuron_dropout", 0.0),
        confidence_tau=config.get("confidence_tau", 1.0),
    ).to(device)

    # Remap old MoE router keys
    state = ckpt["model_state_dict"]
    remap = {
        "decoder.router.router.0.weight": "decoder.router.pool_proj.weight",
        "decoder.router.router.0.bias": "decoder.router.pool_proj.bias",
        "decoder.router.router.2.weight": "decoder.router.router.weight",
        "decoder.router.router.2.bias": "decoder.router.router.bias",
    }
    for old_key, new_key in remap.items():
        if old_key in state and new_key not in state:
            state[new_key] = state.pop(old_key)

    model.load_state_dict(state)
    model.eval()
    return model, vocab, config


def analyze_test_set(model, test_entries, vocab, config, device):
    """Run inference on test set, recording predictions and expert assignments."""
    use_phon = config.get("use_phonological", True)
    ds = PastTenseDataset(test_entries, vocab, use_phon)
    collate = partial(collate_fn, pad_idx=vocab.pad_idx)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)

    results = []
    entry_idx = 0

    for src, tgt, reg_labels in loader:
        src, tgt = src.to(device), tgt.to(device)
        B = src.size(0)

        # Encode
        H = model.encoder(src)

        # Get expert assignments
        if model.use_slots:
            slots = model.slot_attention(H)
            memory = model.l0drop(slots)
        else:
            memory = H

        expert_idx, expert_probs = model.decoder.get_expert_assignments(
            memory, src_tokens=src
        )

        # Greedy decode
        preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)

        for i in range(B):
            pred_tokens = vocab.decode(preds[i].tolist())
            tgt_tokens = vocab.decode(tgt[i].tolist())
            src_tokens = vocab.decode(src[i].tolist())
            entry = test_entries[entry_idx]

            results.append({
                "src": "".join(src_tokens),
                "tgt": "".join(tgt_tokens),
                "pred": "".join(pred_tokens),
                "correct": pred_tokens == tgt_tokens,
                "expert": expert_idx[i].item(),
                "expert_probs": expert_probs[i].cpu().numpy(),
                "regularity": entry["regularity"],
            })
            entry_idx += 1

    return results


def analyze_wugs(model, wug_entries, vocab, device):
    """Analyze expert routing for wug (nonce) verbs."""
    results = []
    for entry in wug_entries:
        src_ids = torch.tensor(
            [vocab.encode(entry["phon_src"], add_eos=True)],
            dtype=torch.long, device=device,
        )
        H = model.encoder(src_ids)
        if model.use_slots:
            slots = model.slot_attention(H)
            memory = model.l0drop(slots)
        else:
            memory = H

        expert_idx, expert_probs = model.decoder.get_expert_assignments(
            memory, src_tokens=src_ids
        )
        preds = model.greedy_decode(src_ids, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)
        pred_tokens = vocab.decode(preds[0].tolist())

        results.append({
            "src": " ".join(entry["phon_src"]),
            "pred": " ".join(pred_tokens),
            "tgt_regular": " ".join(entry["phon_tgt_regular"]),
            "tgt_irregular": " ".join(entry["phon_tgt_irregular"]),
            "matches_regular": pred_tokens == entry["phon_tgt_regular"],
            "matches_irregular": pred_tokens == entry["phon_tgt_irregular"],
            "expert": expert_idx[0].item(),
            "expert_probs": expert_probs[0].cpu().numpy(),
        })
    return results


def make_heatmap(results, num_experts, out_path):
    """Create heatmap of expert assignment counts for reg vs irreg."""
    counts = np.zeros((2, num_experts), dtype=int)  # rows: [reg, irreg]
    for r in results:
        row = 0 if r["regularity"] == "reg" else 1
        counts[row, r["expert"]] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(counts, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"Expert {k}" for k in range(num_experts)])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Regular", "Irregular"])
    ax.set_xlabel("Expert")
    ax.set_ylabel("Verb Class")
    ax.set_title("v16e MoE Expert Assignment: Regular vs Irregular")

    # Annotate cells with counts
    for i in range(2):
        for j in range(num_experts):
            text = ax.text(j, i, str(counts[i, j]),
                           ha="center", va="center", fontsize=14,
                           color="white" if counts[i, j] > counts.max() / 2 else "black")

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {out_path}")


def write_irregular_by_expert(results, num_experts, out_path):
    """Write all irregular verb predictions grouped by expert."""
    irreg = [r for r in results if r["regularity"] == "irreg"]

    with open(out_path, "w") as f:
        f.write("IRREGULAR VERB PREDICTIONS BY EXPERT\n")
        f.write("=" * 60 + "\n\n")

        for k in range(num_experts):
            group = [r for r in irreg if r["expert"] == k]
            f.write(f"--- Expert {k} ({len(group)} irregular verbs) ---\n")
            for r in sorted(group, key=lambda x: x["src"]):
                status = "CORRECT" if r["correct"] else "WRONG"
                f.write(f"  [{status}] {r['src']} -> {r['tgt']}  "
                        f"(pred: {r['pred']})  "
                        f"probs: [{', '.join(f'{p:.3f}' for p in r['expert_probs'])}]\n")
            f.write("\n")

    print(f"Saved irregular analysis to {out_path}")


def write_stats(results, wug_results, num_experts, out_path):
    """Write expert utilization stats and specific irregular results."""
    with open(out_path, "w") as f:
        f.write("v16e MoE EXPERT ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        # -- Expert utilization --
        f.write("EXPERT UTILIZATION (test set)\n")
        f.write("-" * 40 + "\n")
        for k in range(num_experts):
            group = [r for r in results if r["expert"] == k]
            n_reg = sum(1 for r in group if r["regularity"] == "reg")
            n_irreg = sum(1 for r in group if r["regularity"] == "irreg")
            n_correct = sum(1 for r in group if r["correct"])
            total = len(group)
            acc = n_correct / total * 100 if total > 0 else 0
            f.write(f"  Expert {k}: {total} verbs "
                    f"({n_reg} reg, {n_irreg} irreg), "
                    f"accuracy {n_correct}/{total} ({acc:.1f}%)\n")
        f.write("\n")

        # -- Overall stats --
        reg_results = [r for r in results if r["regularity"] == "reg"]
        irreg_results = [r for r in results if r["regularity"] == "irreg"]
        reg_acc = sum(r["correct"] for r in reg_results) / len(reg_results) * 100 if reg_results else 0
        irreg_acc = sum(r["correct"] for r in irreg_results) / len(irreg_results) * 100 if irreg_results else 0
        balanced = (reg_acc + irreg_acc) / 2

        f.write("OVERALL ACCURACY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Regular:   {sum(r['correct'] for r in reg_results)}/{len(reg_results)} ({reg_acc:.1f}%)\n")
        f.write(f"  Irregular: {sum(r['correct'] for r in irreg_results)}/{len(irreg_results)} ({irreg_acc:.1f}%)\n")
        f.write(f"  Balanced:  {balanced:.1f}%\n\n")

        # -- Correct irregular predictions --
        correct_irreg = [r for r in irreg_results if r["correct"]]
        f.write(f"CORRECT IRREGULAR PREDICTIONS ({len(correct_irreg)} total)\n")
        f.write("-" * 40 + "\n")
        for r in sorted(correct_irreg, key=lambda x: x["expert"]):
            f.write(f"  Expert {r['expert']}: {r['src']} -> {r['tgt']}  "
                    f"probs: [{', '.join(f'{p:.3f}' for p in r['expert_probs'])}]\n")
        f.write("\n")

        # -- Incorrect irregular predictions --
        wrong_irreg = [r for r in irreg_results if not r["correct"]]
        f.write(f"INCORRECT IRREGULAR PREDICTIONS ({len(wrong_irreg)} total)\n")
        f.write("-" * 40 + "\n")
        for r in sorted(wrong_irreg, key=lambda x: x["expert"]):
            f.write(f"  Expert {r['expert']}: {r['src']} -> {r['pred']}  "
                    f"(expected: {r['tgt']})  "
                    f"probs: [{', '.join(f'{p:.3f}' for p in r['expert_probs'])}]\n")
        f.write("\n")

        # -- Wug results by expert --
        if wug_results:
            f.write("WUG VERB EXPERT ROUTING\n")
            f.write("-" * 40 + "\n")
            for k in range(num_experts):
                wug_group = [r for r in wug_results if r["expert"] == k]
                n_reg_match = sum(r["matches_regular"] for r in wug_group)
                n_irreg_match = sum(r["matches_irregular"] for r in wug_group)
                f.write(f"  Expert {k}: {len(wug_group)} wugs "
                        f"({n_reg_match} match regular, {n_irreg_match} match irregular)\n")
            f.write(f"\n  Total wugs: {len(wug_results)}, "
                    f"regular matches: {sum(r['matches_regular'] for r in wug_results)}, "
                    f"irregular matches: {sum(r['matches_irregular'] for r in wug_results)}\n")

    print(f"Saved stats to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading checkpoint...")
    model, vocab, config = load_checkpoint(CKPT_PATH, device)
    num_experts = config.get("num_experts", 4)
    print(f"Model loaded. num_experts={num_experts}, "
          f"routing_mode={config.get('routing_mode', 'soft')}")

    # Load and split data
    entries = load_english_merged(str(DATA_PATH))
    _, _, test_entries = split_data(entries, seed=config.get("seed", 42))
    print(f"Test set: {len(test_entries)} entries "
          f"({sum(1 for e in test_entries if e['regularity']=='reg')} reg, "
          f"{sum(1 for e in test_entries if e['regularity']=='irreg')} irreg)")

    # Analyze test set
    print("Analyzing test set...")
    with torch.no_grad():
        results = analyze_test_set(model, test_entries, vocab, config, device)

    # Analyze wugs
    wug_results = []
    if WUG_DIR.exists():
        print("Analyzing wug verbs...")
        wug_entries = load_wug_data(str(WUG_DIR))
        with torch.no_grad():
            wug_results = analyze_wugs(model, wug_entries, vocab, device)

    # Generate outputs
    make_heatmap(results, num_experts, OUT_DIR / "expert_heatmap.png")
    write_irregular_by_expert(results, num_experts, OUT_DIR / "irregular_by_expert.txt")
    write_stats(results, wug_results, num_experts, OUT_DIR / "expert_stats.txt")

    print("Done!")


if __name__ == "__main__":
    main()
