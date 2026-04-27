"""Diagnose wug (nonce verb) evaluation failures.

Loads a checkpoint, runs all wug verbs through the model, and prints
detailed diagnostics including raw token encodings to reveal
tokenization/vocab mismatches between training data and wug data.

Usage:
    cd src && python ../scripts/diagnose_wugs.py \
        --checkpoint ../checkpoints/v6_monotonic_no_slots/best_model.pt
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path so imports work when run from project root or src/
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

import torch
from data.vocab import CharVocab
from data.dataset import load_english_merged, load_wug_data, split_data
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
        use_slots=config.get("use_slots", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, config


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose wug evaluation: check vocab, encoding, and predictions"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--data_path", type=str,
        default=str(src_dir.parent / "data" / "RevisitPinkerAndPrince"
                     / "experiment_1" / "english_merged.txt"),
        help="Path to english_merged.txt (for comparison verbs)"
    )
    parser.add_argument(
        "--wug_dir", type=str,
        default=str(src_dir.parent / "data" / "RevisitPinkerAndPrince"
                     / "experiment_1_wugs"),
        help="Path to experiment_1_wugs/ directory"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model and vocab ---
    model, vocab, config = load_checkpoint(args.checkpoint, device)
    print(f"Config use_slots: {config.get('use_slots', True)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Vocab tokens: {vocab.idx2token}")
    print()

    # --- Vocab analysis ---
    print("=" * 70)
    print("SECTION 1: VOCAB COVERAGE ANALYSIS")
    print("=" * 70)

    # Collect training charset
    entries = load_english_merged(args.data_path)
    train_chars = set()
    for e in entries:
        for ch in e["phon_src"]:
            train_chars.add(ch)
        for ch in e["phon_tgt"]:
            train_chars.add(ch)

    # Collect wug charset
    wug_entries = load_wug_data(args.wug_dir)
    wug_chars = set()
    for w in wug_entries:
        for ch in w["phon_src"]:
            wug_chars.add(ch)
        for ch in w["phon_tgt_regular"]:
            wug_chars.add(ch)
        for ch in w["phon_tgt_irregular"]:
            wug_chars.add(ch)

    oov_chars = wug_chars - train_chars
    shared_chars = wug_chars & train_chars
    print(f"Training charset ({len(train_chars)}): {sorted(train_chars)}")
    print(f"Wug charset ({len(wug_chars)}): {sorted(wug_chars)}")
    print(f"Shared: {sorted(shared_chars)}")
    print(f"OOV in wugs ({len(oov_chars)}): {sorted(oov_chars)}")
    print()

    # Check which OOV chars map to UNK
    print("OOV char -> vocab index mapping:")
    for ch in sorted(oov_chars):
        idx = vocab.token2idx.get(ch, vocab.unk_idx)
        label = "UNK" if idx == vocab.unk_idx else f"idx={idx}"
        print(f"  '{ch}' (U+{ord(ch):04X}) -> {label}")
    print()

    # --- Tokenization comparison ---
    print("=" * 70)
    print("SECTION 2: TOKENIZATION COMPARISON (training vs wug)")
    print("=" * 70)

    # Show a few training examples
    print("\nTraining examples (phon_src -> phon_tgt):")
    for e in entries[:5]:
        src_chars = e["phon_src"]
        src_ids = vocab.encode(src_chars, add_eos=True)
        tgt_chars = e["phon_tgt"]
        tgt_ids = vocab.encode(tgt_chars, add_sos=True, add_eos=True)
        unk_count = sum(1 for i in src_ids if i == vocab.unk_idx)
        print(f"  src chars: {src_chars}")
        print(f"  src ids:   {src_ids}  (UNK count: {unk_count})")
        print(f"  tgt chars: {tgt_chars}")
        print(f"  tgt ids:   {tgt_ids}")
        print()

    print("Wug examples (phon_src):")
    for w in wug_entries[:5]:
        src_chars = w["phon_src"]
        src_ids = vocab.encode(src_chars, add_eos=True)
        reg_chars = w["phon_tgt_regular"]
        reg_ids = vocab.encode(reg_chars, add_sos=True, add_eos=True)
        unk_in_src = sum(1 for i in src_ids if i == vocab.unk_idx)
        unk_in_tgt = sum(1 for i in reg_ids if i == vocab.unk_idx)
        print(f"  src chars: {src_chars}")
        print(f"  src ids:   {src_ids}  (UNK count: {unk_in_src})")
        print(f"  reg chars: {reg_chars}")
        print(f"  reg ids:   {reg_ids}  (UNK count: {unk_in_tgt})")
        print()

    # --- Count total UNK tokens across all wugs ---
    total_src_tokens = 0
    total_src_unk = 0
    total_tgt_tokens = 0
    total_tgt_unk = 0
    for w in wug_entries:
        src_ids = vocab.encode(w["phon_src"], add_eos=True)
        reg_ids = vocab.encode(w["phon_tgt_regular"], add_eos=True)
        total_src_tokens += len(src_ids) - 1  # exclude EOS
        total_src_unk += sum(1 for i in src_ids if i == vocab.unk_idx)
        total_tgt_tokens += len(reg_ids) - 1
        total_tgt_unk += sum(1 for i in reg_ids if i == vocab.unk_idx)

    print(f"Across all {len(wug_entries)} wug entries:")
    print(f"  Source: {total_src_unk}/{total_src_tokens} tokens are UNK "
          f"({100*total_src_unk/total_src_tokens:.1f}%)")
    print(f"  Target (regular): {total_tgt_unk}/{total_tgt_tokens} tokens are UNK "
          f"({100*total_tgt_unk/total_tgt_tokens:.1f}%)")
    print()

    # --- Model predictions ---
    print("=" * 70)
    print("SECTION 3: MODEL PREDICTIONS ON WUG VERBS")
    print("=" * 70)

    n_match_reg = 0
    n_match_irreg = 0
    with torch.no_grad():
        for i, w in enumerate(wug_entries):
            src_ids = vocab.encode(w["phon_src"], add_eos=True)
            src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

            preds = model.greedy_decode(
                src_tensor,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
            )
            pred_tokens = vocab.decode(preds[0].tolist())

            reg_match = (pred_tokens == w["phon_tgt_regular"])
            irreg_match = (pred_tokens == w["phon_tgt_irregular"])
            n_match_reg += int(reg_match)
            n_match_irreg += int(irreg_match)

            status = ""
            if reg_match:
                status = " [MATCH REG]"
            elif irreg_match:
                status = " [MATCH IRREG]"

            unk_in_src = sum(1 for x in src_ids if x == vocab.unk_idx)

            print(f"\nWug {i:3d}: src={' '.join(w['phon_src'])}")
            print(f"         src_ids={src_ids} (UNK: {unk_in_src})")
            print(f"         pred={' '.join(pred_tokens)}{status}")
            print(f"         tgt_reg={' '.join(w['phon_tgt_regular'])}")
            print(f"         tgt_irreg={' '.join(w['phon_tgt_irregular'])}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total wug verbs: {len(wug_entries)}")
    print(f"Match regular:   {n_match_reg}/{len(wug_entries)} "
          f"({100*n_match_reg/len(wug_entries):.1f}%)")
    print(f"Match irregular: {n_match_irreg}/{len(wug_entries)} "
          f"({100*n_match_irreg/len(wug_entries):.1f}%)")
    print(f"OOV characters in wug data: {sorted(oov_chars)}")
    print(f"Fraction of wug source tokens that are UNK: "
          f"{total_src_unk}/{total_src_tokens} ({100*total_src_unk/total_src_tokens:.1f}%)")


if __name__ == "__main__":
    main()
