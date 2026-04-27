"""Edit transducer decoder for morphological transduction.

Instead of generating output characters from scratch, the model predicts
a sequence of edit operations (COPY, DELETE, INSERT, SUBSTITUTE) applied
to the source string. This bakes in the inductive bias that morphology
is mostly copying with local modifications.

Regular rule: COPY COPY ... COPY INSERT(d) → trivially learnable
Irregular:    COPY COPY SUB(ɒ) COPY        → vowel change as explicit edit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import BahdanauAttention


# Edit operation types
EDIT_PAD = 0
EDIT_END = 1
EDIT_COPY = 2
EDIT_DELETE = 3
# INSERT_c starts at 4: INSERT char with index c → edit_id = 4 + c
# SUB_c starts at 4 + vocab_size: SUB char with index c → edit_id = 4 + vocab_size + c


class EditVocab:
    """Maps edit operations to integer indices.

    Layout: [PAD, END, COPY, DELETE, INSERT_0, ..., INSERT_V-1, SUB_0, ..., SUB_V-1]
    Total size: 4 + 2 * char_vocab_size
    """

    def __init__(self, char_vocab_size: int):
        self.char_vocab_size = char_vocab_size
        self.size = 4 + 2 * char_vocab_size

    def insert_id(self, char_idx: int) -> int:
        return 4 + char_idx

    def sub_id(self, char_idx: int) -> int:
        return 4 + self.char_vocab_size + char_idx

    def decode_edit(self, edit_id: int):
        """Convert edit ID to (operation, char_idx or None).

        Returns:
            (op_name, char_idx): op_name in {PAD, END, COPY, DELETE, INSERT, SUB}
        """
        if edit_id == EDIT_PAD:
            return ("PAD", None)
        elif edit_id == EDIT_END:
            return ("END", None)
        elif edit_id == EDIT_COPY:
            return ("COPY", None)
        elif edit_id == EDIT_DELETE:
            return ("DELETE", None)
        elif edit_id < 4 + self.char_vocab_size:
            return ("INSERT", edit_id - 4)
        else:
            return ("SUB", edit_id - 4 - self.char_vocab_size)


def align_to_edits(src_ids: list[int], tgt_ids: list[int],
                   char_vocab_size: int) -> list[int]:
    """Convert (source, target) character sequences to an edit script.

    Uses Needleman-Wunsch alignment with costs:
        COPY (match): 0
        SUB (mismatch): 1
        INSERT (gap in src): 1
        DELETE (gap in tgt): 1

    Args:
        src_ids: source character indices (no SOS/EOS)
        tgt_ids: target character indices (no SOS/EOS)
        char_vocab_size: size of character vocabulary

    Returns:
        edit_ids: list of edit operation indices (ending with EDIT_END)
    """
    ev = EditVocab(char_vocab_size)
    n, m = len(src_ids), len(tgt_ids)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i  # deletions
    for j in range(1, m + 1):
        dp[0][j] = j  # insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if src_ids[i - 1] == tgt_ids[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # COPY (free)
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # SUB
                    dp[i - 1][j] + 1,       # DELETE
                    dp[i][j - 1] + 1,       # INSERT
                )

    # Traceback
    edits = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_ids[i - 1] == tgt_ids[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            edits.append(EDIT_COPY)
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            edits.append(ev.sub_id(tgt_ids[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            edits.append(ev.insert_id(tgt_ids[j - 1]))
            j -= 1
        else:
            edits.append(EDIT_DELETE)
            i -= 1

    edits.reverse()
    edits.append(EDIT_END)
    return edits


def apply_edits(src_ids: list[int], edit_ids: list[int],
                char_vocab_size: int) -> list[int]:
    """Apply edit operations to source to produce output.

    Args:
        src_ids: source character indices
        edit_ids: edit operation indices
        char_vocab_size: size of character vocabulary

    Returns:
        output_ids: resulting character indices
    """
    ev = EditVocab(char_vocab_size)
    output = []
    src_ptr = 0

    for edit_id in edit_ids:
        op, char_idx = ev.decode_edit(edit_id)
        if op == "END":
            break
        elif op == "COPY":
            if src_ptr < len(src_ids):
                output.append(src_ids[src_ptr])
                src_ptr += 1
        elif op == "DELETE":
            src_ptr += 1
        elif op == "INSERT":
            output.append(char_idx)
        elif op == "SUB":
            output.append(char_idx)
            src_ptr += 1

    return output


class EditDecoder(nn.Module):
    """LSTM decoder that predicts edit operations over the source string.

    At each step, the decoder sees:
    - Its previous hidden state
    - The previous edit operation embedding
    - Attention context over encoder output
    - The current source character at the read pointer

    And predicts the next edit operation.
    """

    def __init__(self, char_vocab_size: int, d_model: int = 128,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, lstm_hidden: int = 0,
                 scheduled_sampling: float = 0.0):
        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.lstm_dim = lstm_hidden if lstm_hidden > 0 else d_model
        self.scheduled_sampling = scheduled_sampling  # probability of using own prediction

        self.edit_vocab = EditVocab(char_vocab_size)
        edit_vocab_size = self.edit_vocab.size

        # Embeddings
        self.edit_embedding = nn.Embedding(edit_vocab_size, d_model, padding_idx=EDIT_PAD)
        self.char_embedding = nn.Embedding(char_vocab_size, d_model, padding_idx=pad_idx)

        # Attention over encoder output
        self.attention = BahdanauAttention(self.lstm_dim, d_model, attn_dim=d_model)

        # LSTM: input = [edit_emb; context; current_src_char_emb]
        self.lstm = nn.LSTM(
            input_size=d_model * 3,
            hidden_size=self.lstm_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Output: predict edit operation
        self.output_proj = nn.Linear(self.lstm_dim + d_model, edit_vocab_size)

    def forward(self, edit_targets: torch.Tensor, memory: torch.Tensor,
                src_tokens: torch.Tensor,
                encoder_out: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            edit_targets: (B, T) edit operation indices (with EDIT_END)
            memory: (B, N, d_model) encoder output
            src_tokens: (B, N) source character indices (no EOS)
            encoder_out: unused, kept for API compatibility
        Returns:
            logits: (B, T-1, edit_vocab_size)
        """
        B, T = edit_targets.shape
        device = edit_targets.device
        N = src_tokens.size(1)

        mem_pad_mask = (src_tokens == self.pad_idx)  # (B, N)
        src_embs = self.char_embedding(src_tokens)  # (B, N, d)

        # Initialize
        h = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        context = torch.zeros(B, self.d_model, device=device)

        # Source read pointer per sample (starts at 0)
        src_ptr = torch.zeros(B, dtype=torch.long, device=device)

        outputs = []
        prev_edit = edit_targets[:, 0]  # first token (teacher-forced start)
        for t in range(T - 1):
            # Scheduled sampling: use own prediction with probability p
            if self.training and self.scheduled_sampling > 0 and t > 0:
                use_own = (torch.rand(B, device=device) < self.scheduled_sampling)
                own_pred = outputs[-1].argmax(dim=-1)  # (B,)
                edit_input = torch.where(use_own, own_pred, edit_targets[:, t])
            else:
                edit_input = edit_targets[:, t]

            # Current edit embedding
            edit_emb = self.edit_embedding(edit_input)  # (B, d)

            # Current source character at read pointer
            clamped_ptr = src_ptr.clamp(max=N - 1)  # (B,)
            src_char_emb = src_embs[torch.arange(B, device=device), clamped_ptr]  # (B, d)

            # LSTM step
            lstm_input = torch.cat([edit_emb, context, src_char_emb], dim=-1).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            lstm_out = lstm_out.squeeze(1)  # (B, lstm_dim)

            # Attention over encoder
            context, _ = self.attention(lstm_out, memory, mask=mem_pad_mask)

            # Predict next edit
            out = self.output_proj(
                self.dropout(torch.cat([lstm_out, context], dim=-1))
            )  # (B, edit_vocab_size)
            outputs.append(out)

            # Advance source pointer based on the used edit (teacher or own)
            # COPY, DELETE, and SUB all advance the pointer
            advances = (
                (edit_input == EDIT_COPY)
                | (edit_input == EDIT_DELETE)
                | (edit_input >= 4 + self.char_vocab_size)  # SUB range
            ).long()
            src_ptr = src_ptr + advances

        return torch.stack(outputs, dim=1)  # (B, T-1, edit_vocab_size)

    @torch.no_grad()
    def greedy_decode(self, memory: torch.Tensor, src_tokens: torch.Tensor,
                      max_len: int = 64) -> list[list[int]]:
        """Greedy autoregressive decoding of edit operations.

        Returns:
            List of output character index lists (one per batch element)
        """
        B = memory.size(0)
        N = src_tokens.size(1)
        device = memory.device

        mem_pad_mask = (src_tokens == self.pad_idx)
        src_embs = self.char_embedding(src_tokens)

        h = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        context = torch.zeros(B, self.d_model, device=device)
        src_ptr = torch.zeros(B, dtype=torch.long, device=device)

        # Start with EDIT_COPY as first input (arbitrary but natural)
        prev_edit = torch.full((B,), EDIT_COPY, dtype=torch.long, device=device)

        # Collect edits per sample
        all_edits = [[] for _ in range(B)]
        done = [False] * B

        for _ in range(max_len):
            edit_emb = self.edit_embedding(prev_edit)
            clamped_ptr = src_ptr.clamp(max=N - 1)
            src_char_emb = src_embs[torch.arange(B, device=device), clamped_ptr]

            lstm_input = torch.cat([edit_emb, context, src_char_emb], dim=-1).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            lstm_out = lstm_out.squeeze(1)

            context, _ = self.attention(lstm_out, memory, mask=mem_pad_mask)

            logits = self.output_proj(
                torch.cat([lstm_out, context], dim=-1)
            )
            next_edit = logits.argmax(dim=-1)  # (B,)

            for i in range(B):
                if not done[i]:
                    e = next_edit[i].item()
                    if e == EDIT_END:
                        done[i] = True
                    else:
                        all_edits[i].append(e)

            if all(done):
                break

            # Advance pointers
            advances = (
                (next_edit == EDIT_COPY)
                | (next_edit == EDIT_DELETE)
                | (next_edit >= 4 + self.char_vocab_size)
            ).long()
            src_ptr = src_ptr + advances
            prev_edit = next_edit

        # Apply edits to source to get output characters
        outputs = []
        for i in range(B):
            src_ids = src_tokens[i].tolist()
            # Remove padding and EOS from source
            src_ids = [x for x in src_ids if x != self.pad_idx]
            out = apply_edits(src_ids, all_edits[i], self.char_vocab_size)
            outputs.append(out)

        return outputs
