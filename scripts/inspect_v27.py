"""Per-verb success/failure analysis for v27 (biLSTM+LSTM+class retrieval).

Lists every test irregular (success + failure with predicted form), every
failing regular, plus what was retrieved for each failure. Useful for
qualitative discussion in the writeup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functools import partial  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.dataset import (  # noqa: E402
    load_english_merged, split_data, PastTenseDataset, collate_fn, _pad_retrieved,
)
from data.retrieval import (  # noqa: E402
    RetrievalIndex, classify_inflection, INFLECTION_CLASSES,
)
from evaluate import load_checkpoint  # noqa: E402

CKPT = "/gpfs/radev/project/kuan/jl3795/slot_attention/checkpoints/v27_bilstm_lstm_retrieval/best_model.pt"
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

    use_class_retrieval = config.get("use_class_retrieval", True)

    ds = PastTenseDataset(test, vocab, use_phon,
                          retrieval_index=idx,
                          use_class_retrieval=use_class_retrieval)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx))

    results = []
    for batch in loader:
        src = batch[0].to(device)
        tgt = batch[1]
        if len(batch) == 5 and batch[3].dim() == 3:
            retr_ids = batch[3].to(device)
            retr_mask = batch[4].to(device)
        else:
            retr_ids = retr_mask = None
        preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                    retrieval_ids=retr_ids,
                                    retrieval_pad_mask=retr_mask)
        for i in range(src.size(0)):
            pred = vocab.decode(preds[i].tolist())
            tgt_tokens = vocab.decode(tgt[i].tolist())
            src_tokens = vocab.decode(src[i].tolist())
            results.append({
                "src": "".join(src_tokens),
                "tgt": "".join(tgt_tokens),
                "pred": "".join(pred),
                "correct": pred == tgt_tokens,
            })

    for r, e in zip(results, test):
        r["regularity"] = e["regularity"]
        r["cls"] = INFLECTION_CLASSES[classify_inflection(e["phon_src"], e["phon_tgt"])]
        r["retrieved"] = []
        if use_class_retrieval:
            cls_id = classify_inflection(e["phon_src"], e["phon_tgt"])
            nbrs = idx.query_in_class(e["phon_src"], cls_id)
        else:
            nbrs = idx.query(e["phon_src"])
        for j in nbrs[:3]:
            r["retrieved"].append((
                "".join(idx.train_pairs[j][0]),
                "".join(idx.train_pairs[j][1]),
            ))

    irr = [r for r in results if r["regularity"] == "irreg"]
    reg = [r for r in results if r["regularity"] == "reg"]

    # ALL irregulars (success + failure)
    print("=" * 110)
    print(f"TEST IRREGULARS ({len(irr)})")
    print("=" * 110)
    print(f"{'src':<14}{'tgt':<14}{'pred':<14}{'class':<14}OK  retrieved (top-3)")
    print("-" * 110)
    for r in sorted(irr, key=lambda x: (not x["correct"], x["cls"])):
        ok = "✓" if r["correct"] else "✗"
        retr = "  ".join(f"{s}->{t}" for s, t in r["retrieved"])
        print(f"{r['src']:<14}{r['tgt']:<14}{r['pred']:<14}{r['cls']:<14}{ok}  {retr}")
    n_ok = sum(r["correct"] for r in irr)
    print(f"\nIrregular accuracy: {n_ok}/{len(irr)} = {n_ok/len(irr):.1%}")

    # FAILING regulars only
    fail_reg = [r for r in reg if not r["correct"]]
    print()
    print("=" * 110)
    print(f"FAILING REGULARS ({len(fail_reg)} of {len(reg)})")
    print("=" * 110)
    print(f"{'src':<14}{'tgt':<14}{'pred':<14}{'class':<14}retrieved (top-3)")
    print("-" * 110)
    for r in fail_reg:
        retr = "  ".join(f"{s}->{t}" for s, t in r["retrieved"])
        print(f"{r['src']:<14}{r['tgt']:<14}{r['pred']:<14}{r['cls']:<14}{retr}")
    print(f"\nRegular accuracy: {len(reg)-len(fail_reg)}/{len(reg)} = "
          f"{(len(reg)-len(fail_reg))/len(reg):.1%}")

    # Per-class breakdown of irregulars
    print()
    print("=" * 60)
    print("Per-class breakdown (irregulars)")
    print("=" * 60)
    by_cls = {}
    for r in irr:
        by_cls.setdefault(r["cls"], []).append(r)
    for cls, items in sorted(by_cls.items(), key=lambda x: -len(x[1])):
        n = sum(it["correct"] for it in items)
        print(f"  {cls:<14}  {n:>2}/{len(items):>2}  "
              f"({n/len(items):.0%})")


if __name__ == "__main__":
    main()
