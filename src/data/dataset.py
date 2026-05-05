"""Dataset and data loading for past-tense transduction."""

import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .vocab import CharVocab
from .retrieval import RetrievalIndex  # noqa: F401  (re-exported for callers)


def load_english_merged(filepath: str) -> List[dict]:
    """Load english_merged.txt from Kirov & Cotterell.

    Format per line: orth_present  orth_past  phon_present  phon_past  reg/irreg
    """
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            entries.append({
                "orth_src": parts[0],
                "orth_tgt": parts[1],
                "phon_src": list(parts[2]),
                "phon_tgt": list(parts[3]),
                "regularity": parts[4],
            })
    return entries


def _wug_to_disc(chars: List[str]) -> List[str]:
    """Convert wug/CELEXmod phonological encoding to DISC encoding.

    The wug data uses a different transcription system than the training data:
    - È = primary stress marker (strip)
    - Y = aI diphthong (PRICE), Ã = V (STRUT), o = @U (GOAT)
    - Õ = 3: (NURSE), W = aU (MOUTH)
    - C = tS (affricate), J = dZ (affricate), Q = & (TRAP)
    """
    MULTI_CHAR_MAP = {
        "Y": ["a", "I"],    # PRICE diphthong
        "W": ["a", "U"],    # MOUTH diphthong
        "o": ["@", "U"],    # GOAT diphthong
        "Õ": ["3", ":"],    # NURSE vowel
        "C": ["t", "S"],    # voiceless postalveolar affricate
        "J": ["d", "Z"],    # voiced postalveolar affricate
    }
    SINGLE_CHAR_MAP = {
        "Ã": "V",   # STRUT vowel
        "Q": "&",   # TRAP vowel
    }
    result = []
    for ch in chars:
        if ch == "È":
            continue  # strip stress marker
        elif ch in MULTI_CHAR_MAP:
            result.extend(MULTI_CHAR_MAP[ch])
        elif ch in SINGLE_CHAR_MAP:
            result.append(SINGLE_CHAR_MAP[ch])
        else:
            result.append(ch)
    return result


def load_wug_data(wug_dir: str) -> List[dict]:
    """Load Albright & Hayes nonce verbs from experiment_1_wugs/.

    Files: src.txt (present), tgt_regular.txt, tgt_irregular.txt
    CELEXmod.txt has header + frequency info.
    Characters are space-separated with stress markers (È).
    Converted to DISC encoding to match training data.
    """
    wug_dir = Path(wug_dir)
    entries = []

    src_lines = (wug_dir / "src.txt").read_text(encoding="utf-8").strip().splitlines()
    tgt_reg_lines = (wug_dir / "tgt_regular.txt").read_text(encoding="utf-8").strip().splitlines()
    tgt_irreg_lines = (wug_dir / "tgt_irregular.txt").read_text(encoding="utf-8").strip().splitlines()

    for src, tgt_reg, tgt_irreg in zip(src_lines, tgt_reg_lines, tgt_irreg_lines):
        src_chars = _wug_to_disc(src.strip().split())
        reg_chars = _wug_to_disc(tgt_reg.strip().split())
        irreg_chars = _wug_to_disc(tgt_irreg.strip().split())
        entries.append({
            "phon_src": src_chars,
            "phon_tgt_regular": reg_chars,
            "phon_tgt_irregular": irreg_chars,
        })
    return entries


def split_data(entries: List[dict], train_ratio: float = 0.8,
               val_ratio: float = 0.1, seed: int = 42
               ) -> Tuple[List[dict], List[dict], List[dict]]:
    """Deterministic train/val/test split."""
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def apply_training_regime(train_entries: List[dict], regime: str,
                          seed: int = 42) -> List[dict]:
    """Reorder/resample training data according to a training regime.

    Regimes:
        natural:         Use all data as-is (default).
        balanced:        Subsample regulars to match irregular count.
        oversample:      Keep all data, repeat irregulars to match regular count.
        irregular_first: Irregulars in first half, then all data shuffled.
    """
    if regime == "natural":
        return train_entries

    rng = random.Random(seed)
    reg = [e for e in train_entries if e["regularity"] == "reg"]
    irreg = [e for e in train_entries if e["regularity"] == "irreg"]

    if regime == "balanced":
        rng.shuffle(reg)
        sampled_reg = reg[:len(irreg)]
        combined = sampled_reg + irreg
        rng.shuffle(combined)
        return combined

    if regime == "oversample":
        # Repeat irregulars so they appear as often as regulars
        # Keeps all regular examples — no data loss
        repeats = len(reg) // len(irreg)
        remainder = len(reg) % len(irreg)
        rng.shuffle(irreg)
        oversampled_irreg = irreg * repeats + irreg[:remainder]
        combined = reg + oversampled_irreg
        rng.shuffle(combined)
        return combined

    if regime == "irregular_first":
        rng.shuffle(irreg)
        rng.shuffle(reg)
        # Irregulars first, then full dataset shuffled
        second_half = reg + irreg
        rng.shuffle(second_half)
        return irreg + second_half

    raise ValueError(f"Unknown training regime: {regime}")


class PastTenseDataset(Dataset):
    """PyTorch dataset for (source, target) character-level verb pairs."""

    def __init__(self, entries: List[dict], vocab: CharVocab,
                 use_phonological: bool = True, use_edits: bool = False,
                 use_edit_labels: bool = False, use_transducer: bool = False,
                 retrieval_index=None, use_class_retrieval: bool = False,
                 retrieval_mode: str = "knn"):
        self.entries = entries
        self.vocab = vocab
        self.use_phonological = use_phonological
        self.use_edits = use_edits
        self.use_edit_labels = use_edit_labels
        self.use_transducer = use_transducer
        self.retrieval_index = retrieval_index
        self.use_class_retrieval = use_class_retrieval
        # Precompute the retrieved-target token-id sequences once per item.
        # Each entry: list of k tensors (1D) of varying length.
        # When use_class_retrieval is on, use query_in_class(src, true_class)
        # instead of plain edit-distance kNN. NOTE: true_class is derived from
        # (src, tgt), so for val/test items this is a slight information leak.
        # v24a measures the *ceiling* of class-conditional retrieval; a future
        # variant should swap in a learned class predictor for eval-time queries.
        self._retrieved: Optional[list] = None
        if retrieval_index is not None:
            from .retrieval import classify_inflection
            self._retrieved = []
            self._cluster_ids: list[int] = []
            for e in entries:
                src_chars = e["phon_src"] if use_phonological else list(e["orth_src"])
                tgt_chars = e["phon_tgt"] if use_phonological else list(e["orth_tgt"])
                # Track true cluster ID for each item (used by the learned
                # cluster-predictor head's auxiliary CE loss when enabled).
                self._cluster_ids.append(
                    retrieval_index.cluster_id_for(src_chars, tgt_chars))
                if retrieval_mode == "random":
                    # Negative control: random k pairs (deterministic per src).
                    neighbors = retrieval_index.query_random(src_chars)
                elif retrieval_mode == "cluster":
                    # Unsupervised cluster retrieval: language-agnostic
                    # alternative to use_class_retrieval. Uses TRUE cluster
                    # at val/test (small leak, same as use_class_retrieval).
                    cid = retrieval_index.cluster_id_for(src_chars, tgt_chars)
                    neighbors = retrieval_index.query_in_cluster(src_chars, cid)
                elif use_class_retrieval:
                    cls = classify_inflection(src_chars, tgt_chars)
                    neighbors = retrieval_index.query_in_class(src_chars, cls)
                else:
                    neighbors = retrieval_index.query(src_chars)
                tgts = retrieval_index.get_targets(neighbors)
                # Encode each retrieved tgt to ids (no SOS/EOS — flatten as memory).
                tgt_ids = [
                    torch.tensor(self.vocab.encode(t, add_sos=False, add_eos=False),
                                 dtype=torch.long)
                    for t in tgts
                ]
                # Class-conditional retrieval may return < k items if the class
                # has few members. Pad with copies of the closest available item
                # so collate_fn's (B, k, L) shape stays consistent across the batch.
                # Use UNK (not PAD) as the absolute-empty fallback — BiLSTMEncoder
                # uses pack_padded_sequence which crashes on length-0 sequences,
                # so retrieved memory entries must contain at least one non-PAD
                # token even when no real neighbor exists.
                fallback = torch.tensor([vocab.unk_idx], dtype=torch.long)
                # Replace any zero-length encodings with the UNK fallback.
                tgt_ids = [t if t.numel() > 0 else fallback for t in tgt_ids]
                while len(tgt_ids) < retrieval_index.k:
                    tgt_ids.append(tgt_ids[-1] if tgt_ids else fallback)
                self._retrieved.append(tgt_ids)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.entries[idx]
        if self.use_phonological:
            src = entry["phon_src"]
            tgt = entry["phon_tgt"]
        else:
            src = list(entry["orth_src"])
            tgt = list(entry["orth_tgt"])

        src_ids = self.vocab.encode(src, add_sos=False, add_eos=True)
        tgt_ids = self.vocab.encode(tgt, add_sos=True, add_eos=True)
        # Regularity label: 0=regular, 1=irregular
        reg_label = 0 if entry.get("regularity", "reg") == "reg" else 1

        if self.use_edit_labels:
            from model.edit_labeler import align_to_position_labels
            src_raw = self.vocab.encode(src, add_sos=False, add_eos=False)
            tgt_raw = self.vocab.encode(tgt, add_sos=False, add_eos=False)
            labels, suffix = align_to_position_labels(src_raw, tgt_raw, len(self.vocab))
            # Pad labels to match src_ids length (which has EOS appended)
            # Use -1 for the EOS position (ignored in cross_entropy)
            labels_padded = labels + [-1] * (len(src_ids) - len(labels))
            # Suffix: append EOS (= char_vocab_size, same as in EditLabeler)
            suffix_with_eos = suffix + [len(self.vocab)]
            return (torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                    reg_label,
                    torch.tensor(labels_padded, dtype=torch.long),
                    torch.tensor(suffix_with_eos, dtype=torch.long))

        if self.use_edits:
            from model.edit_decoder import align_to_edits
            src_raw = self.vocab.encode(src, add_sos=False, add_eos=False)
            tgt_raw = self.vocab.encode(tgt, add_sos=False, add_eos=False)
            edit_ids = align_to_edits(src_raw, tgt_raw, len(self.vocab))
            return (torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                    reg_label,
                    torch.tensor(edit_ids, dtype=torch.long))

        if self.use_transducer:
            from model.transducer_actions import align_to_actions
            src_raw = self.vocab.encode(src, add_sos=False, add_eos=False)
            tgt_raw = self.vocab.encode(tgt, add_sos=False, add_eos=False)
            action_ids = align_to_actions(src_raw, tgt_raw)
            base = (torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                    reg_label,
                    torch.tensor(action_ids, dtype=torch.long))
            if self._retrieved is not None:
                return base + (self._retrieved[idx],)
            return base

        base = (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
                reg_label)
        if self._retrieved is not None:
            # Non-transducer + retrieval (Transformer/LSTM): append the k
            # retrieved tgt id tensors as a 4th element. If a cluster ID is
            # available (whenever an index is provided), pass it as a 5th
            # element for the optional learned-cluster-predictor's aux loss.
            extra = (self._retrieved[idx],)
            if self._cluster_ids:
                extra = extra + (int(self._cluster_ids[idx]),)
            return base + extra
        return base


def _pad_retrieved(retrieved_per_sample, pad_idx: int):
    """Pad a list-of-lists-of-1D-tensors into a (B, k, L_max) tensor + mask.

    `retrieved_per_sample` has length B; each element is a list of k 1D
    tensors of varying length. Returns:
        retr_ids:  (B, k, L_max)  long, padded with pad_idx
        retr_mask: (B, k, L_max)  bool — True where padded
    """
    B = len(retrieved_per_sample)
    k = len(retrieved_per_sample[0])
    L_max = max(t.size(0) for sample in retrieved_per_sample for t in sample)
    retr_ids = torch.full((B, k, L_max), pad_idx, dtype=torch.long)
    retr_mask = torch.ones((B, k, L_max), dtype=torch.bool)
    for i, sample in enumerate(retrieved_per_sample):
        for j, t in enumerate(sample):
            L = t.size(0)
            retr_ids[i, j, :L] = t
            retr_mask[i, j, :L] = False
    return retr_ids, retr_mask


def collate_fn(batch, pad_idx: int = 0):
    """Pad variable-length sequences into a batch.

    Returns:
        src: (batch, max_src_len)
        tgt: (batch, max_tgt_len)
        reg_labels: (batch,) — 0=regular, 1=irregular
        edits: (batch, max_edit_len) — only if edit sequences present
        retr_ids, retr_mask: (B, k, L) and (B, k, L) — only if retrieval is on
    """
    n_fields = len(batch[0])
    # has_retrieval_5 is the OLD transducer + retrieval shape:
    # (src, tgt, label, action_ids, retrieved_list) — batch[3] is a 1D Tensor.
    has_retrieval_5 = (n_fields == 5 and isinstance(batch[0][4], list)
                       and not isinstance(batch[0][3], list))
    has_retrieval_4 = (n_fields == 4 and isinstance(batch[0][3], list))
    # 5-field with int cluster_id at position 4: (src, tgt, label, retrieved_list, cluster_id)
    has_retrieval_4_plus_cluster = (n_fields == 5
                                     and isinstance(batch[0][3], list)
                                     and isinstance(batch[0][4], int))

    if has_retrieval_4_plus_cluster:
        # Non-transducer + retrieval + cluster ID
        srcs, tgts, labels, retrieved, cluster_ids = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
        reg_labels = torch.tensor(labels, dtype=torch.long)
        retr_ids, retr_mask = _pad_retrieved(retrieved, pad_idx)
        cluster_t = torch.tensor(cluster_ids, dtype=torch.long)
        return (src_padded, tgt_padded, reg_labels, retr_ids, retr_mask, cluster_t)

    if has_retrieval_4:
        # Non-transducer + retrieval (Transformer/LSTM): (src, tgt, label, retrieved_list)
        srcs, tgts, labels, retrieved = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
        reg_labels = torch.tensor(labels, dtype=torch.long)
        retr_ids, retr_mask = _pad_retrieved(retrieved, pad_idx)
        return src_padded, tgt_padded, reg_labels, retr_ids, retr_mask

    if n_fields == 5 and not has_retrieval_5:
        # Edit labeler: (src, tgt, label, edit_labels, suffix)
        srcs, tgts, labels, edit_labels, suffixes = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
        edit_label_padded = pad_sequence(edit_labels, batch_first=True, padding_value=-1)
        suffix_padded = pad_sequence(suffixes, batch_first=True, padding_value=pad_idx)
        reg_labels = torch.tensor(labels, dtype=torch.long)
        return src_padded, tgt_padded, reg_labels, edit_label_padded, suffix_padded
    elif has_retrieval_5:
        # Transducer + retrieval: (src, tgt, label, actions, retrieved_list)
        srcs, tgts, labels, actions, retrieved = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
        action_padded = pad_sequence(actions, batch_first=True, padding_value=0)
        reg_labels = torch.tensor(labels, dtype=torch.long)
        retr_ids, retr_mask = _pad_retrieved(retrieved, pad_idx)
        return src_padded, tgt_padded, reg_labels, action_padded, retr_ids, retr_mask
    elif n_fields == 4:
        # Edit decoder OR transducer w/o retrieval: (src, tgt, label, edits/actions)
        srcs, tgts, labels, edits = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
        edit_padded = pad_sequence(edits, batch_first=True, padding_value=0)
        reg_labels = torch.tensor(labels, dtype=torch.long)
        return src_padded, tgt_padded, reg_labels, edit_padded
    else:
        srcs, tgts, labels = zip(*batch)
        src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
        reg_labels = torch.tensor(labels, dtype=torch.long)
        return src_padded, tgt_padded, reg_labels
