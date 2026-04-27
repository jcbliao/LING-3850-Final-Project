"""Per-position edit labeler for morphological transduction.

Non-autoregressive: classifies each source position as COPY, DELETE, or SUB(c)
in parallel. A small suffix head predicts 0-3 appended characters.

Avoids pointer drift by eliminating the sequential pointer entirely.
Each source character independently receives an edit label.

Works for morphology because DISC phonological encoding makes characters atomic:
  speak (s,p,i,:,k) → spoke (s,p,@,U,k) = all 1→1 mappings + suffix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Per-position edit labels:
#   0 = COPY (emit source char)
#   1 = DELETE (emit nothing)
#   2..2+V-1 = SUB_c (emit char c instead)
LABEL_COPY = 0
LABEL_DELETE = 1
LABEL_SUB_START = 2  # SUB for char index c → label = 2 + c


def align_to_position_labels(src_ids: list[int], tgt_ids: list[int],
                             char_vocab_size: int) -> tuple[list[int], list[int]]:
    """Convert (source, target) to per-position labels + suffix.

    Uses Needleman-Wunsch alignment, then extracts:
    - per-position label for each source char
    - suffix chars appended after the last source position

    Args:
        src_ids: source character indices (no SOS/EOS)
        tgt_ids: target character indices (no SOS/EOS)
        char_vocab_size: size of character vocabulary

    Returns:
        labels: list of per-position edit labels (len = len(src_ids))
        suffix: list of char indices appended at end (may be empty)
    """
    n, m = len(src_ids), len(tgt_ids)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if src_ids[i - 1] == tgt_ids[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # SUB
                    dp[i - 1][j] + 1,       # DELETE
                    dp[i][j - 1] + 1,       # INSERT
                )

    # Traceback
    ops = []  # list of (op, src_idx_or_None, tgt_char_or_None)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_ids[i - 1] == tgt_ids[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("COPY", i - 1, None))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("SUB", i - 1, tgt_ids[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("INSERT", None, tgt_ids[j - 1]))
            j -= 1
        else:
            ops.append(("DELETE", i - 1, None))
            i -= 1

    ops.reverse()

    # Convert to per-position labels + suffix
    labels = [LABEL_COPY] * n  # default: COPY
    suffix = []

    for op, src_idx, tgt_char in ops:
        if op == "COPY":
            labels[src_idx] = LABEL_COPY
        elif op == "DELETE":
            labels[src_idx] = LABEL_DELETE
        elif op == "SUB":
            labels[src_idx] = LABEL_SUB_START + tgt_char
        elif op == "INSERT":
            suffix.append(tgt_char)

    return labels, suffix


def apply_position_edits(src_ids: list[int], labels: list[int],
                         suffix: list[int], char_vocab_size: int) -> list[int]:
    """Apply per-position edits and suffix to produce output."""
    output = []
    for i, label in enumerate(labels):
        if i >= len(src_ids):
            break
        if label == LABEL_COPY:
            output.append(src_ids[i])
        elif label == LABEL_DELETE:
            continue
        elif label >= LABEL_SUB_START:
            output.append(label - LABEL_SUB_START)
    output.extend(suffix)
    return output


class EditLabeler(nn.Module):
    """Non-autoregressive per-position edit classifier + suffix predictor.

    Two heads:
    1. Edit head: classifies each source position → COPY/DELETE/SUB(c)
    2. Suffix head: small LSTM predicting 0-3 appended characters

    No sequential pointer — each position classified independently.
    """

    def __init__(self, char_vocab_size: int, d_model: int = 128,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, max_suffix: int = 3, **kwargs):
        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_suffix = max_suffix

        # Number of edit labels: COPY + DELETE + SUB for each char
        self.num_labels = 2 + char_vocab_size

        # Edit classification head (per-position)
        self.edit_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.num_labels),
        )

        # Suffix predictor: small LSTM
        # Input: encoder final hidden state → predict suffix chars autoregressively
        self.suffix_embedding = nn.Embedding(char_vocab_size + 1, d_model,
                                             padding_idx=0)  # +1 for SOS token
        self.suffix_sos_idx = char_vocab_size  # use last index as SOS
        self.suffix_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.suffix_proj = nn.Linear(d_model, char_vocab_size + 1)  # +1 for EOS
        self.suffix_eos_idx = char_vocab_size  # EOS = same as SOS (reuse)

    def forward(self, memory: torch.Tensor, src_tokens: torch.Tensor,
                edit_labels: torch.Tensor | None = None,
                suffix_targets: torch.Tensor | None = None):
        """
        Args:
            memory: (B, N, d_model) encoder output
            src_tokens: (B, N) source token indices
            edit_labels: (B, N) per-position edit labels (for training)
            suffix_targets: (B, S) suffix char indices with EOS (for training)
        Returns:
            dict with 'edit_logits', 'suffix_logits', 'loss'
        """
        B, N, _ = memory.shape
        src_pad_mask = (src_tokens == self.pad_idx)

        # Per-position edit classification
        edit_logits = self.edit_head(memory)  # (B, N, num_labels)

        # Suffix prediction: use encoder's mean-pooled output as initial state
        real_mask = ~src_pad_mask
        lengths = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (memory * real_mask.unsqueeze(-1).float()).sum(dim=1) / lengths  # (B, d)

        # Initialize suffix LSTM with pooled encoder output
        h0 = pooled.unsqueeze(0)  # (1, B, d)
        c0 = torch.zeros_like(h0)

        result = {"edit_logits": edit_logits}

        if edit_labels is not None:
            # Training: compute edit loss
            edit_loss = F.cross_entropy(
                edit_logits.reshape(-1, self.num_labels),
                edit_labels.reshape(-1),
                ignore_index=-1,  # ignore padding
            )
            result["edit_loss"] = edit_loss

        if suffix_targets is not None:
            # Teacher-forced suffix prediction
            S = suffix_targets.size(1)
            sos = torch.full((B, 1), self.suffix_sos_idx, dtype=torch.long,
                             device=memory.device)
            suffix_input = torch.cat([sos, suffix_targets[:, :-1]], dim=1)  # (B, S)
            suffix_emb = self.suffix_embedding(suffix_input)  # (B, S, d)
            suffix_out, _ = self.suffix_lstm(suffix_emb, (h0, c0))  # (B, S, d)
            suffix_logits = self.suffix_proj(suffix_out)  # (B, S, V+1)
            result["suffix_logits"] = suffix_logits

            suffix_loss = F.cross_entropy(
                suffix_logits.reshape(-1, self.char_vocab_size + 1),
                suffix_targets.reshape(-1),
                ignore_index=self.pad_idx,
            )
            result["suffix_loss"] = suffix_loss

        if "edit_loss" in result and "suffix_loss" in result:
            result["loss"] = result["edit_loss"] + result["suffix_loss"]

        return result

    @torch.no_grad()
    def greedy_decode(self, memory: torch.Tensor,
                      src_tokens: torch.Tensor) -> list[list[int]]:
        """Non-autoregressive decoding: classify positions + generate suffix."""
        B, N, _ = memory.shape
        src_pad_mask = (src_tokens == self.pad_idx)

        # Classify each position
        edit_logits = self.edit_head(memory)  # (B, N, num_labels)
        edit_preds = edit_logits.argmax(dim=-1)  # (B, N)

        # Generate suffix
        real_mask = ~src_pad_mask
        lengths = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (memory * real_mask.unsqueeze(-1).float()).sum(dim=1) / lengths
        h = pooled.unsqueeze(0)
        c = torch.zeros_like(h)

        suffix_token = torch.full((B, 1), self.suffix_sos_idx,
                                  dtype=torch.long, device=memory.device)
        suffixes = [[] for _ in range(B)]
        for _ in range(self.max_suffix + 1):
            emb = self.suffix_embedding(suffix_token)
            out, (h, c) = self.suffix_lstm(emb, (h, c))
            logits = self.suffix_proj(out.squeeze(1))  # (B, V+1)
            next_token = logits.argmax(dim=-1)  # (B,)
            for i in range(B):
                t = next_token[i].item()
                if t != self.suffix_eos_idx:
                    suffixes[i].append(t)
            suffix_token = next_token.unsqueeze(1)
            if all(next_token[i].item() == self.suffix_eos_idx for i in range(B)):
                break

        # Apply edits + suffix to source
        outputs = []
        for i in range(B):
            src = [t for t in src_tokens[i].tolist() if t != self.pad_idx]
            labels = edit_preds[i, :len(src)].tolist()
            out = apply_position_edits(src, labels, suffixes[i], self.char_vocab_size)
            outputs.append(out)

        return outputs
