"""Per-example breakdown of v24a's failures on test irregulars + wugs.

Loads the v24a best checkpoint and dumps:
    src  ->  pred  vs  target   [✓/✗]   retrieved exemplars used
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functools import partial  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.dataset import (  # noqa: E402
    load_english_merged, load_wug_data, split_data,
    PastTenseDataset, collate_fn, _pad_retrieved,
)
from data.retrieval import RetrievalIndex, classify_inflection, INFLECTION_CLASSES  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402

CKPT = "/gpfs/radev/project/kuan/jl3795/slot_attention/checkpoints/v24a_class_retrieval/best_model.pt"
DATA = "/gpfs/radev/project/kuan/jl3795/slot_attention/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt"
WUGS = "/gpfs/radev/project/kuan/jl3795/slot_attention/data/RevisitPinkerAndPrince/experiment_1_wugs/"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, config = load_checkpoint(CKPT, device)
    model.eval()
    use_phon = config.get("use_phonological", True)

    entries = load_english_merged(DATA)
    train, val, test = split_data(entries, seed=config.get("seed", 42))

    # Rebuild the same retrieval index used at training time
    seen = set()
    pairs = []
    for e in train:
        key = tuple(e["phon_src"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((e["phon_src"], e["phon_tgt"]))
    idx = RetrievalIndex(pairs, k=config.get("retrieval_k", 5))

    test_irreg = [e for e in test if e["regularity"] == "irreg"]

    print(f"=== Test irregulars (n={len(test_irreg)}) ===")
    print(f"{'src':<14}{'pred':<14}{'tgt':<14}{'class':<14}retrieved (top-3 of 5)")
    print("-" * 100)

    correct = 0
    for e in test_irreg:
        src_chars = e["phon_src"]
        tgt_chars = e["phon_tgt"]
        true_cls = classify_inflection(src_chars, tgt_chars)
        # Class-conditional retrieval (matches what dataset does)
        nbrs = idx.query_in_class(src_chars, true_cls)
        nbr_tgts = idx.get_targets(nbrs)
        nbr_srcs = [pairs[j][0] for j in nbrs]
        # Encode for forward pass
        src_ids = torch.tensor([vocab.encode(src_chars, add_eos=True)],
                               dtype=torch.long, device=device)
        retr_id_tensors = [
            torch.tensor(vocab.encode(t, add_sos=False, add_eos=False),
                         dtype=torch.long)
            for t in nbr_tgts
        ]
        # Pad to k
        while len(retr_id_tensors) < idx.k:
            retr_id_tensors.append(retr_id_tensors[-1] if retr_id_tensors
                                   else torch.tensor([vocab.pad_idx], dtype=torch.long))
        retr_ids, retr_mask = _pad_retrieved([retr_id_tensors], vocab.pad_idx)
        retr_ids, retr_mask = retr_ids.to(device), retr_mask.to(device)

        preds = model.greedy_decode(src_ids, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                    retrieval_ids=retr_ids,
                                    retrieval_pad_mask=retr_mask)
        pred_chars = vocab.decode(preds[0].tolist())
        ok = pred_chars == tgt_chars
        correct += int(ok)
        mark = "✓" if ok else "✗"

        retr_summary = "  ".join(
            f"{''.join(s)}→{''.join(t)}"
            for s, t in zip(nbr_srcs[:3], nbr_tgts[:3])
        )
        print(f"{''.join(src_chars):<14}"
              f"{''.join(pred_chars):<14}"
              f"{''.join(tgt_chars):<14}"
              f"{INFLECTION_CLASSES[true_cls]:<14}{mark} | {retr_summary}")

    print(f"\nIrregular accuracy: {correct}/{len(test_irreg)} = {correct/len(test_irreg):.1%}")

    # Wugs analysis
    print()
    print(f"=== Wug failures (sample) ===")
    print("Wugs use plain edit-distance retrieval (no true class), unlike training.")
    print(f"{'src':<14}{'pred':<14}{'expected_reg':<14}{'expected_irr':<14}retrieved")
    print("-" * 110)
    wug_entries = load_wug_data(WUGS)
    n_reg = n_irr = 0
    failures_shown = 0
    for entry in wug_entries:
        src_chars = entry["phon_src"]
        nbrs = idx.query(src_chars)  # plain edit distance, no class
        nbr_tgts = idx.get_targets(nbrs)
        nbr_srcs = [pairs[j][0] for j in nbrs]
        src_ids = torch.tensor([vocab.encode(src_chars, add_eos=True)],
                               dtype=torch.long, device=device)
        retr_id_tensors = [
            torch.tensor(vocab.encode(t, add_sos=False, add_eos=False),
                         dtype=torch.long)
            for t in nbr_tgts
        ]
        while len(retr_id_tensors) < idx.k:
            retr_id_tensors.append(retr_id_tensors[-1])
        retr_ids, retr_mask = _pad_retrieved([retr_id_tensors], vocab.pad_idx)
        retr_ids, retr_mask = retr_ids.to(device), retr_mask.to(device)

        preds = model.greedy_decode(src_ids, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                    retrieval_ids=retr_ids,
                                    retrieval_pad_mask=retr_mask)
        pred_chars = vocab.decode(preds[0].tolist())
        matches_reg = pred_chars == entry["phon_tgt_regular"]
        matches_irr = pred_chars == entry["phon_tgt_irregular"]
        n_reg += int(matches_reg)
        n_irr += int(matches_irr)
        if not (matches_reg or matches_irr) and failures_shown < 12:
            retr_summary = "  ".join(
                f"{''.join(s)}→{''.join(t)}"
                for s, t in zip(nbr_srcs[:3], nbr_tgts[:3])
            )
            print(f"{''.join(src_chars):<14}"
                  f"{''.join(pred_chars):<14}"
                  f"{''.join(entry['phon_tgt_regular']):<14}"
                  f"{''.join(entry['phon_tgt_irregular']):<14}"
                  f"{retr_summary}")
            failures_shown += 1
    print(f"\nWug totals: {n_reg}/90 reg, {n_irr}/90 irr")


if __name__ == "__main__":
    main()
