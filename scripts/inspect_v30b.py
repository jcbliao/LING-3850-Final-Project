"""Per-verb prediction analysis for v30b (biLSTM+LSTM + cluster retrieval +
LEARNED predictor). Shows every test irregular with:
  - source (present-tense form)
  - true past form (target)
  - model's predicted past form
  - predicted cluster ID + name
  - top-3 exemplars retrieved from the predicted cluster
  - true (oracle) cluster for comparison

Critically uses PREDICTED cluster (no leak via tgt) for retrieval — this is
the honest evaluation path that the v30b training+test loop uses.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from data.dataset import (  # noqa: E402
    load_english_merged, split_data, _pad_retrieved,
)
from data.retrieval import (  # noqa: E402
    RetrievalIndex, derive_pattern_signature,
)
from evaluate import load_checkpoint  # noqa: E402

CKPT = "/gpfs/radev/project/kuan/jl3795/slot_attention/checkpoints/v30b_learned_cluster_retrieval/best_model.pt"
DATA = "/gpfs/radev/project/kuan/jl3795/slot_attention/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, config = load_checkpoint(CKPT, device)
    model.eval()
    use_phon = config.get("use_phonological", True)

    entries = load_english_merged(DATA)
    train, val, test = split_data(entries, seed=config.get("seed", 42))

    # Rebuild retrieval index from unique train pairs
    seen = set()
    pairs = []
    for e in train:
        key = tuple(e["phon_src"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((e["phon_src"], e["phon_tgt"]))
    idx = RetrievalIndex(pairs, k=config.get("retrieval_k", 5))

    # Map cluster ID -> human-readable signature
    id_to_sig = {v: k for k, v in idx.cluster_registry.items()}

    test_irreg = [e for e in test if e["regularity"] == "irreg"]

    print("=" * 130)
    print(f"v30b TEST IRREGULARS ({len(test_irreg)}) — PREDICTED-CLUSTER EVAL (honest, no leak)")
    print("=" * 130)
    print(f"{'src':<14}{'tgt':<14}{'pred':<14}OK  pred_cluster                 true_cluster                 retrieved (top-3)")
    print("-" * 130)

    n_correct = 0
    n_cluster_correct = 0
    for e in test_irreg:
        src_chars = e["phon_src"] if use_phon else list(e["orth_src"])
        tgt_chars = e["phon_tgt"] if use_phon else list(e["orth_tgt"])
        src_ids = vocab.encode(src_chars, add_eos=True)
        src_t = torch.tensor([src_ids], dtype=torch.long, device=device)

        # Predict cluster from src ALONE (no tgt)
        pred_cid = model.predict_clusters(src_t).item()
        true_cid = idx.cluster_id_for(src_chars, tgt_chars)
        cluster_match = (pred_cid == true_cid)
        if cluster_match:
            n_cluster_correct += 1

        # Class-conditional retrieval using predicted cluster
        nbrs = idx.query_in_cluster(src_chars, pred_cid)
        nbr_tgts = idx.get_targets(nbrs)
        nbr_srcs = [pairs[j][0] for j in nbrs]

        # Build retrieval tensors
        fallback = torch.tensor([vocab.unk_idx], dtype=torch.long)
        retr_id_tensors = [
            torch.tensor(vocab.encode(t, add_sos=False, add_eos=False),
                         dtype=torch.long)
            for t in nbr_tgts
        ]
        retr_id_tensors = [t if t.numel() > 0 else fallback for t in retr_id_tensors]
        while len(retr_id_tensors) < idx.k:
            retr_id_tensors.append(retr_id_tensors[-1] if retr_id_tensors else fallback)
        retr_ids, retr_mask = _pad_retrieved([retr_id_tensors], vocab.pad_idx)
        retr_ids = retr_ids.to(device)
        retr_mask = retr_mask.to(device)

        # Greedy decode
        preds = model.greedy_decode(src_t, sos_idx=vocab.sos_idx,
                                     eos_idx=vocab.eos_idx,
                                     retrieval_ids=retr_ids,
                                     retrieval_pad_mask=retr_mask)
        pred_chars = vocab.decode(preds[0].tolist())
        ok = pred_chars == tgt_chars
        if ok:
            n_correct += 1

        sig_pred = id_to_sig.get(pred_cid, ("UNK",))
        sig_true = id_to_sig.get(true_cid, ("UNK",))
        sig_pred_str = ",".join(sig_pred) if sig_pred else "IDENTITY"
        sig_true_str = ",".join(sig_true) if sig_true else "IDENTITY"
        match_str = "✓" if cluster_match else "✗"
        retr = "  ".join(
            f"{''.join(s)}->{''.join(t)}" for s, t in zip(nbr_srcs[:3], nbr_tgts[:3])
        )

        mark = "✓" if ok else "✗"
        print(f"{''.join(src_chars):<14}"
              f"{''.join(tgt_chars):<14}"
              f"{''.join(pred_chars):<14}"
              f"{mark}   "
              f"{sig_pred_str:<28} {sig_true_str:<22} {match_str} | {retr}")

    print()
    print(f"Irregular accuracy:    {n_correct}/{len(test_irreg)} = "
          f"{n_correct/len(test_irreg):.1%}")
    print(f"Cluster-prediction accuracy on irregulars: "
          f"{n_cluster_correct}/{len(test_irreg)} = "
          f"{n_cluster_correct/len(test_irreg):.1%}")


if __name__ == "__main__":
    main()
