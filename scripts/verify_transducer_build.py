"""Static build check for the HMNT TransducerDecoder integration.

Constructs the full model with decoder_type='transducer' and runs one
forward+backward pass on a tiny dummy batch (CPU only — no training).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from data.vocab import CharVocab  # noqa: E402
from data.dataset import PastTenseDataset, collate_fn  # noqa: E402
from model import SlotAttentionTransducer  # noqa: E402
from model.transducer_actions import (  # noqa: E402
    align_to_actions, apply_actions, action_vocab_size,
)
from functools import partial  # noqa: E402


def main():
    # Tiny vocab: 5 chars
    vocab = CharVocab()
    vocab.build_from_sequences([["p", "l", "a", "y", "e", "d", "s", "i", "n", "g"]])
    V = len(vocab)
    print(f"Vocab size: {V}, action vocab size: {action_vocab_size(V)}")

    entries = [
        {"orth_src": "play", "orth_tgt": "played",
         "phon_src": list("play"), "phon_tgt": list("played"),
         "regularity": "reg"},
        {"orth_src": "sing", "orth_tgt": "sang",
         "phon_src": list("sing"), "phon_tgt": list("sang"),
         "regularity": "irreg"},
        {"orth_src": "play", "orth_tgt": "played",
         "phon_src": list("play"), "phon_tgt": list("played"),
         "regularity": "reg"},
    ]
    ds = PastTenseDataset(entries, vocab, use_phonological=True, use_transducer=True)
    item = ds[0]
    print(f"Sample item: src.shape={item[0].shape}, tgt.shape={item[1].shape}, "
          f"action.shape={item[3].shape}")
    print(f"  actions: {item[3].tolist()}")

    # Verify roundtrip via dataset action target
    src_raw = vocab.encode(list("play"), add_sos=False, add_eos=False)
    tgt_raw = vocab.encode(list("played"), add_sos=False, add_eos=False)
    actions = align_to_actions(src_raw, tgt_raw)
    out = apply_actions(src_raw, actions)
    assert out == tgt_raw, f"Roundtrip mismatch: {out} vs {tgt_raw}"
    print("Roundtrip on real vocab: OK")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=3, shuffle=False,
                        collate_fn=partial(collate_fn, pad_idx=vocab.pad_idx))
    batch = next(iter(loader))
    src, tgt, reg_labels, action_tgt = batch
    print(f"Batched: src={src.shape} tgt={tgt.shape} actions={action_tgt.shape}")

    # Build model with HMNT
    model = SlotAttentionTransducer(
        vocab_size=V,
        d_model=32,
        enc_layers=1,
        dec_layers=1,
        d_ff=64,
        encoder_type="bilstm",
        decoder_type="transducer",
        use_slots=False,
        dropout=0.0,
        pad_idx=vocab.pad_idx,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Phase 1: pure teacher forcing
    model.train()
    result = model(src, tgt, reg_labels=reg_labels, action_targets=action_tgt)
    print(f"TF forward: loss={result['loss'].item():.4f}  "
          f"logits={result['logits'].shape}")
    result["loss"].backward()
    print("TF backward OK")

    # Phase 2: DAgger β-mixing
    model.zero_grad()
    src_no_eos = [[int(x) for x in s.tolist()
                   if int(x) not in {vocab.pad_idx, vocab.sos_idx, vocab.eos_idx}]
                  for s in src]
    tgt_no_special = [[int(x) for x in t.tolist()
                       if int(x) not in {vocab.pad_idx, vocab.sos_idx, vocab.eos_idx}]
                      for t in tgt]
    result = model(src, tgt, reg_labels=reg_labels, action_targets=action_tgt,
                   dagger_beta=0.5,
                   src_no_eos=src_no_eos, tgt_no_special=tgt_no_special)
    print(f"DAgger forward: loss={result['loss'].item():.4f}")
    result["loss"].backward()
    print("DAgger backward OK")

    # Greedy decode
    model.eval()
    decoded = model.greedy_decode(src, sos_idx=vocab.sos_idx,
                                   eos_idx=vocab.eos_idx, max_len=32)
    print(f"Decoded shape: {decoded.shape}")
    for i in range(decoded.size(0)):
        toks = vocab.decode(decoded[i].tolist())
        print(f"  sample {i}: {''.join(toks)}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
