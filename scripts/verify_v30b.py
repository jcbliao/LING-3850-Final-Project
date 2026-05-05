"""Smoke test: cluster retrieval + learned predictor (v30b)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functools import partial  # noqa: E402
import torch  # noqa: E402

from data.vocab import CharVocab  # noqa: E402
from data.dataset import PastTenseDataset, collate_fn  # noqa: E402
from data.retrieval import RetrievalIndex  # noqa: E402
from model import SlotAttentionTransducer  # noqa: E402


def main():
    vocab = CharVocab()
    vocab.build_from_sequences([list("playedsingranwlk")])
    entries = [
        {"phon_src": list("play"), "phon_tgt": list("played"), "regularity": "reg",
         "orth_src": "play", "orth_tgt": "played"},
        {"phon_src": list("sing"), "phon_tgt": list("sang"), "regularity": "irreg",
         "orth_src": "sing", "orth_tgt": "sang"},
        {"phon_src": list("ring"), "phon_tgt": list("rang"), "regularity": "irreg",
         "orth_src": "ring", "orth_tgt": "rang"},
        {"phon_src": list("walk"), "phon_tgt": list("walked"), "regularity": "reg",
         "orth_src": "walk", "orth_tgt": "walked"},
    ]
    train_pairs = [(e["phon_src"], e["phon_tgt"]) for e in entries]
    idx = RetrievalIndex(train_pairs, k=2)
    print(f"Num clusters: {idx.num_clusters}")

    ds = PastTenseDataset(entries, vocab, use_phonological=True,
                          retrieval_index=idx, retrieval_mode="cluster")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    batch = next(iter(loader))
    print(f"Batch fields: {len(batch)} (expecting 6: src, tgt, label, retr_ids, retr_mask, cluster_id)")
    assert len(batch) == 6, f"Expected 6 fields with cluster IDs, got {len(batch)}"
    src, tgt, reg_labels, retr_ids, retr_mask, cluster_targets = batch
    print(f"  src={src.shape}  retr_ids={retr_ids.shape}  cluster_targets={cluster_targets}")

    model = SlotAttentionTransducer(
        vocab_size=len(vocab),
        d_model=32,
        enc_layers=1, dec_layers=1, d_ff=64,
        dropout=0.0, pad_idx=vocab.pad_idx,
        encoder_type="bilstm", decoder_type="lstm",
        use_slots=False, use_retrieval=True,
        num_clusters=idx.num_clusters,
        lambda_cluster=0.5,
    )
    print(f"Params (with cluster predictor): {sum(p.numel() for p in model.parameters()):,}")
    model.train()
    result = model(src, tgt, reg_labels=reg_labels,
                   retrieval_ids=retr_ids, retrieval_pad_mask=retr_mask,
                   cluster_targets=cluster_targets)
    print(f"Forward: loss={result['loss'].item():.4f}  "
          f"loss_cluster={result.get('loss_cluster', 0):.4f}")
    result["loss"].backward()
    print("Backward OK")

    model.eval()
    pred_clusters = model.predict_clusters(src)
    print(f"Predicted clusters: {pred_clusters.tolist()}")
    decoded = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                  retrieval_ids=retr_ids, retrieval_pad_mask=retr_mask)
    print(f"Greedy decode shape: {decoded.shape}")
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
