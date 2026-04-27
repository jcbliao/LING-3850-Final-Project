"""Character-level vocabulary for phonological transcription."""

from typing import List, Optional


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class CharVocab:
    """Maps characters (phonological or orthographic) to integer indices."""

    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: list[str] = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.idx2token)

    def _add(self, token: str) -> int:
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)
        return self.token2idx[token]

    def build_from_sequences(self, sequences: List[List[str]]):
        """Build vocabulary from a list of character sequences."""
        for seq in sequences:
            for ch in seq:
                self._add(ch)

    def encode(self, sequence: List[str], add_sos: bool = False,
               add_eos: bool = False) -> List[int]:
        """Convert character list to index list."""
        indices = []
        if add_sos:
            indices.append(self.sos_idx)
        for ch in sequence:
            indices.append(self.token2idx.get(ch, self.unk_idx))
        if add_eos:
            indices.append(self.eos_idx)
        return indices

    def decode(self, indices: List[int], strip_special: bool = True) -> List[str]:
        """Convert index list back to characters."""
        tokens = []
        for idx in indices:
            tok = self.idx2token[idx]
            if strip_special and tok in SPECIAL_TOKENS:
                if tok == EOS_TOKEN:
                    break
                continue
            tokens.append(tok)
        return tokens
