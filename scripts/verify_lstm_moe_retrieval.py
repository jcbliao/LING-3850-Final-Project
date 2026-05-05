"""Build check: biLSTM encoder + (LSTM | MoE) decoder + retrieval."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from functools import partial  # noqa: E402
import torch  # noqa: E402

from data.vocab import CharVocab  # noqa: E402
from data.dataset import PastTenseDataset, collate_fn  # noqa: E402
from data.retrieval import RetrievalIndex  # noqa: E402
from model import SlotAttentionTransducer  # noqa: E402


def _build_data():
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
    ds = PastTenseDataset(entries, vocab, use_phonological=True, retrieval_index=idx)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx),
    )
    batch = next(iter(loader))
    return vocab, batch


def _exercise(model_kwargs, label):
    print(f"\n--- {label} ---")
    vocab, batch = _build_data()
    src, tgt, reg_labels, retr_ids, retr_mask = batch
    model = SlotAttentionTransducer(
        vocab_size=len(vocab),
        d_model=32,
        enc_layers=1, dec_layers=1, d_ff=64,
        dropout=0.0, pad_idx=vocab.pad_idx,
        use_slots=False, use_retrieval=True,
        **model_kwargs,
    )
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    model.train()
    result = model(src, tgt, reg_labels=reg_labels,
                   retrieval_ids=retr_ids, retrieval_pad_mask=retr_mask)
    print(f"Forward: loss={result['loss'].item():.4f}")
    result["loss"].backward()
    print("Backward OK")
    model.eval()
    decoded = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                  retrieval_ids=retr_ids, retrieval_pad_mask=retr_mask)
    print(f"Greedy decode shape: {decoded.shape}")


def main():
    _exercise(dict(encoder_type="bilstm", decoder_type="lstm"),
              "biLSTM + LSTM + retrieval (v27)")
    _exercise(dict(encoder_type="bilstm", decoder_type="moe_lstm",
                   num_experts=3, expert_hidden=16,
                   routing_mode="gumbel", lambda_diversity=0.1),
              "biLSTM + MoE + retrieval (v28)")
    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
