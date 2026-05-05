"""Dump what the kNN retrieval index actually returns for test irregulars.

Mirrors the build pipeline in train.py: same seed, same split, same index,
so the retrievals shown are exactly what the model saw at training/eval time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.dataset import load_english_merged, split_data  # noqa: E402
from data.retrieval import RetrievalIndex, edit_distance  # noqa: E402


def main():
    data_path = "/gpfs/radev/project/kuan/jl3795/slot_attention/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt"
    entries = load_english_merged(data_path)
    train, val, test = split_data(entries, seed=42)
    print(f"Splits: train={len(train)} val={len(val)} test={len(test)}")

    train_irreg = [e for e in train if e["regularity"] == "irreg"]
    print(f"Train irregulars: {len(train_irreg)}")

    test_irreg = [e for e in test if e["regularity"] == "irreg"]
    test_reg = [e for e in test if e["regularity"] == "reg"]
    print(f"Test: {len(test_reg)} regular + {len(test_irreg)} irregular")

    # Build index exactly like train.py does
    seen = set()
    pairs = []
    for e in train:
        key = tuple(e["phon_src"])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((e["phon_src"], e["phon_tgt"]))
    idx = RetrievalIndex(pairs, k=5)

    print(f"\nIndex over {len(pairs)} unique train pairs.\n")

    print("=" * 90)
    print(f"{'TEST IRREGULAR':<25} -> top-5 retrieved (src -> past)  [reg/irr; edit_dist]")
    print("=" * 90)
    for e in test_irreg:
        src = "".join(e["phon_src"])
        tgt = "".join(e["phon_tgt"])
        nbrs = idx.query(e["phon_src"])
        retrieved = []
        for j in nbrs:
            n_src, n_tgt = pairs[j]
            # Lookup regularity of this train entry by src match
            reg = next((te["regularity"] for te in train
                        if te["phon_src"] == n_src), "?")
            d = edit_distance(e["phon_src"], n_src)
            retrieved.append(f"{''.join(n_src)}->{''.join(n_tgt)} [{reg};d={d}]")
        n_irreg_in_top5 = sum(1 for s in retrieved if "irreg" in s)
        print(f"{src:<10} -> {tgt:<14} ({n_irreg_in_top5}/5 irreg) | "
              + "  ".join(retrieved))

    print()
    print("=" * 90)
    print(f"{'TEST REGULAR (sample 10)':<25} -> top-5 retrieved")
    print("=" * 90)
    import random
    rng = random.Random(0)
    for e in rng.sample(test_reg, 10):
        src = "".join(e["phon_src"])
        tgt = "".join(e["phon_tgt"])
        nbrs = idx.query(e["phon_src"])
        retrieved = []
        for j in nbrs:
            n_src, n_tgt = pairs[j]
            reg = next((te["regularity"] for te in train
                        if te["phon_src"] == n_src), "?")
            d = edit_distance(e["phon_src"], n_src)
            retrieved.append(f"{''.join(n_src)}->{''.join(n_tgt)} [{reg};d={d}]")
        n_irreg_in_top5 = sum(1 for s in retrieved if "irreg" in s)
        print(f"{src:<10} -> {tgt:<14} ({n_irreg_in_top5}/5 irreg) | "
              + "  ".join(retrieved))

    # Aggregate stats: of test irregulars, how many have ANY irregular in top-5?
    print()
    print("=" * 90)
    print("Summary: retrieval coverage for test irregulars")
    print("=" * 90)
    n_with_any = 0
    n_with_useful = 0   # at least one retrieved past form ENDS in same trigram
    for e in test_irreg:
        nbrs = idx.query(e["phon_src"])
        any_irreg = False
        useful = False
        tgt_suffix = "".join(e["phon_tgt"][-3:])
        for j in nbrs:
            n_src, n_tgt = pairs[j]
            reg = next((te["regularity"] for te in train
                        if te["phon_src"] == n_src), "?")
            if reg == "irreg":
                any_irreg = True
            if "".join(n_tgt[-3:]) == tgt_suffix and reg == "irreg":
                useful = True
        if any_irreg:
            n_with_any += 1
        if useful:
            n_with_useful += 1
    n = len(test_irreg)
    print(f"  {n_with_any}/{n} test irregulars have AT LEAST ONE irregular in top-5")
    print(f"  {n_with_useful}/{n} test irregulars have a 'useful' irregular "
          "(same final-trigram past form) in top-5")
    print(f"  -> theoretical retrieval-only ceiling on irreg ≈ {n_with_useful/n:.1%}")


if __name__ == "__main__":
    main()
