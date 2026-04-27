"""Inspect model predictions on test set — shows actual outputs for error analysis."""
import sys
sys.path.insert(0, '.')
import argparse
import torch
from functools import partial
from data.vocab import CharVocab
from data.dataset import load_english_merged, split_data, PastTenseDataset, collate_fn
from model import SlotAttentionTransducer
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = ckpt['vocab']
    config = ckpt['config']

    model = SlotAttentionTransducer(
        vocab_size=len(vocab),
        d_model=config.get('d_model', 128), nhead=config.get('nhead', 4),
        enc_layers=config.get('enc_layers', 3), dec_layers=config.get('dec_layers', 3),
        d_ff=config.get('d_ff', 256), num_slots=config.get('num_slots', 8),
        slot_iters=config.get('slot_iters', 3), dropout=config.get('dropout', 0.1),
        pad_idx=vocab.pad_idx, l0_mode=config.get('l0_mode', 'conditional'),
        target_l0=config.get('target_l0', 2.0), lagrangian_lr=config.get('lagrangian_lr', 0.01),
        use_copy=config.get('use_copy', False), slot_nhead=config.get('slot_nhead', 1),
        l0_beta=config.get('l0_beta', 0.66), use_slots=config.get('use_slots', True),
        decoder_type=config.get('decoder_type', 'transformer'),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    entries = load_english_merged(config['data_path'])
    _, _, test_entries = split_data(entries, seed=42)

    reg_test = [e for e in test_entries if e['regularity'] == 'reg']
    irreg_test = [e for e in test_entries if e['regularity'] == 'irreg']

    collate_f = partial(collate_fn, pad_idx=vocab.pad_idx)

    def show_predictions(label, subset, max_wrong=30):
        ds = PastTenseDataset(subset, vocab, use_phonological=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=collate_f)
        all_preds = []
        correct = 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            preds = model.greedy_decode(src, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx)
            for i in range(src.size(0)):
                s = ''.join(vocab.decode(src[i].tolist()))
                t = ''.join(vocab.decode(tgt[i].tolist()))
                p = ''.join(vocab.decode(preds[i].tolist()))
                is_correct = (p == t)
                correct += int(is_correct)
                all_preds.append((s, t, p, is_correct))

        print(f'\n=== {label} ({correct}/{len(all_preds)} = {correct/len(all_preds)*100:.1f}%) ===')
        shown = 0
        for s, t, p, ok in all_preds:
            if not ok:
                print(f'  {s:15s} -> {t:15s} | pred: {p:15s} [WRONG]')
                shown += 1
                if shown >= max_wrong:
                    remaining = sum(1 for _, _, _, ok in all_preds if not ok) - shown
                    if remaining > 0:
                        print(f'  ... and {remaining} more wrong')
                    break
        print(f'  --- Correct examples (up to 10) ---')
        shown_c = 0
        for s, t, p, ok in all_preds:
            if ok:
                print(f'  {s:15s} -> {t:15s} | pred: {p:15s} [OK]')
                shown_c += 1
                if shown_c >= 10:
                    break

    show_predictions('REGULAR TEST', reg_test)
    show_predictions('IRREGULAR TEST', irreg_test, max_wrong=100)

if __name__ == '__main__':
    main()
