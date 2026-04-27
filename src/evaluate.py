"""Evaluation on seen verbs, unseen verbs, and nonce (wug) verbs."""

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.vocab import CharVocab
from data.dataset import (
    load_english_merged, load_wug_data, split_data,
    PastTenseDataset, collate_fn,
)
from model import SlotAttentionTransducer


def load_checkpoint(path: str, device: torch.device):
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
    # Remap old MoE router keys (nn.Sequential → split pool_proj + router)
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


def evaluate_exact_match(model, dataloader, vocab, device):
    """Compute per-example predictions and exact-match accuracy."""
    results = []
    correct = 0
    for src, tgt, *_ in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)
        for i in range(src.size(0)):
            pred_tokens = vocab.decode(preds[i].tolist())
            tgt_tokens = vocab.decode(tgt[i].tolist())
            src_tokens = vocab.decode(src[i].tolist())
            is_correct = pred_tokens == tgt_tokens
            correct += int(is_correct)
            results.append({
                "src": "".join(src_tokens),
                "tgt": "".join(tgt_tokens),
                "pred": "".join(pred_tokens),
                "correct": is_correct,
            })
    acc = correct / len(results) if results else 0.0
    return acc, results


def evaluate_wugs(model, wug_entries, vocab, device):
    """Evaluate on nonce verbs: decode each and compare to regular/irregular targets."""
    results = []
    for entry in wug_entries:
        src_ids = torch.tensor(
            [vocab.encode(entry["phon_src"], add_eos=True)],
            dtype=torch.long, device=device,
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
        })

    n_reg = sum(r["matches_regular"] for r in results)
    n_irreg = sum(r["matches_irregular"] for r in results)
    print(f"Wug results: {n_reg}/{len(results)} match regular, "
          f"{n_irreg}/{len(results)} match irregular")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to english_merged.txt")
    parser.add_argument("--wug_dir", type=str, default=None,
                        help="Path to experiment_1_wugs/ directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, config = load_checkpoint(args.checkpoint, device)
    collate = partial(collate_fn, pad_idx=vocab.pad_idx)

    # Seen / unseen verb evaluation
    entries = load_english_merged(args.data_path)
    train_entries, val_entries, test_entries = split_data(
        entries, seed=config.get("seed", 42)
    )

    # Separate test set by regularity
    reg_test = [e for e in test_entries if e["regularity"] == "reg"]
    irreg_test = [e for e in test_entries if e["regularity"] == "irreg"]

    use_phon = config.get("use_phonological", True)
    accs = {}
    for label, subset in [("all_test", test_entries), ("regular", reg_test),
                          ("irregular", irreg_test)]:
        ds = PastTenseDataset(subset, vocab, use_phon)
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
        acc, _ = evaluate_exact_match(model, loader, vocab, device)
        accs[label] = acc
        print(f"{label}: {acc:.4f} ({len(subset)} examples)")

    # Balanced accuracy: equal weight to regular and irregular
    if "regular" in accs and "irregular" in accs:
        balanced = (accs["regular"] + accs["irregular"]) / 2
        print(f"balanced_acc: {balanced:.4f} (avg of regular + irregular)")

    # Wug evaluation
    if args.wug_dir:
        wug_entries = load_wug_data(args.wug_dir)
        evaluate_wugs(model, wug_entries, vocab, device)


if __name__ == "__main__":
    main()
