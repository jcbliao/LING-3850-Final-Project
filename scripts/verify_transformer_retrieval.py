"""Build check: TransformerCharDecoder + copy + monotonic + retrieval."""

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
    V = len(vocab)
    print(f"Vocab size: {V}")

    entries = [
        {"orth_src": "play", "orth_tgt": "played",
         "phon_src": list("play"), "phon_tgt": list("played"),
         "regularity": "reg"},
        {"orth_src": "sing", "orth_tgt": "sang",
         "phon_src": list("sing"), "phon_tgt": list("sang"),
         "regularity": "irreg"},
        {"orth_src": "ring", "orth_tgt": "rang",
         "phon_src": list("ring"), "phon_tgt": list("rang"),
         "regularity": "irreg"},
        {"orth_src": "walk", "orth_tgt": "walked",
         "phon_src": list("walk"), "phon_tgt": list("walked"),
         "regularity": "reg"},
    ]
    train_pairs = [(e["phon_src"], e["phon_tgt"]) for e in entries]
    idx = RetrievalIndex(train_pairs, k=2)

    ds = PastTenseDataset(entries, vocab, use_phonological=True,
                          retrieval_index=idx)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    batch = next(iter(loader))
    print(f"Batch fields: {len(batch)}")
    src, tgt, reg_labels, retr_ids, retr_mask = batch
    print(f"  src={src.shape}  retr_ids={retr_ids.shape}")

    model = SlotAttentionTransducer(
        vocab_size=V,
        d_model=32,
        nhead=4,
        enc_layers=1,
        dec_layers=1,
        d_ff=64,
        encoder_type="transformer",
        decoder_type="transformer",
        use_slots=False,
        use_copy=True,
        mono_alpha_init=0.5,
        dropout=0.0,
        pad_idx=vocab.pad_idx,
        use_retrieval=True,
    )
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    result = model(src, tgt, reg_labels=reg_labels,
                   retrieval_ids=retr_ids, retrieval_pad_mask=retr_mask)
    print(f"Forward: loss={result['loss'].item():.4f}  logits={result['logits'].shape}")
    result["loss"].backward()
    print("Backward OK")

    model.eval()
    decoded = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                  max_len=32,
                                  retrieval_ids=retr_ids,
                                  retrieval_pad_mask=retr_mask)
    print(f"Greedy decode shape: {decoded.shape}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
