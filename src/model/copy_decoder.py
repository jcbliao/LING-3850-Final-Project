"""Monotonic copy-biased LSTM decoder for morphological transduction.

At each output step, the model decides between:
  1. COPY: emit the source character at the current pointer, advance pointer
  2. GENERATE: emit a character from vocabulary (don't advance pointer)
  3. SKIP+GENERATE: advance pointer past a source char, then generate

This gives the model the same expressive power as the edit transducer
but within a standard autoregressive framework that doesn't suffer from
pointer drift — the pointer advances are part of the generation process,
not a separate mechanism.

The architecture biases toward copying (p_copy initialized high), making
the regular rule trivially learnable: copy everything, then generate suffix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import BahdanauAttention


# Action types for the decoder at each step
ACTION_COPY = 0      # emit src[ptr], advance ptr
ACTION_GEN = 1       # emit from vocab, don't advance ptr
ACTION_SKIP_GEN = 2  # advance ptr (skip src char), then emit from vocab


class MonotonicCopyDecoder(nn.Module):
    """LSTM decoder with built-in monotonic copy mechanism.

    Instead of soft copy attention, uses a hard monotonic pointer that
    advances left-to-right through the source. At each step:
    - action_logits decide COPY vs GENERATE vs SKIP+GENERATE
    - If COPY: output = src[ptr], ptr += 1
    - If GENERATE: output = argmax(vocab_logits)
    - If SKIP+GENERATE: ptr += 1, output = argmax(vocab_logits)

    During training, teacher-forced on target characters. The action
    is derived from alignment (known at training time).
    During inference, the model decides actions autoregressively.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, lstm_hidden: int = 0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.lstm_dim = lstm_hidden if lstm_hidden > 0 else d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.char_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.attention = BahdanauAttention(self.lstm_dim, d_model, attn_dim=d_model)

        # LSTM input: [prev_output_emb; context; current_src_char_emb]
        self.lstm = nn.LSTM(
            input_size=d_model * 3,
            hidden_size=self.lstm_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Vocab projection for GENERATE actions
        self.vocab_proj = nn.Linear(self.lstm_dim + d_model, vocab_size)

        # Action gate: decides COPY vs GENERATE
        # Output: scalar logit, sigmoid → p_copy
        # Initialized to bias toward copying (sigmoid(2) ≈ 0.88)
        self.copy_gate = nn.Linear(self.lstm_dim + d_model + d_model, 1)
        nn.init.constant_(self.copy_gate.bias, 2.0)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                encoder_out: torch.Tensor | None = None,
                src_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced forward pass.

        At each step, computes a full vocab distribution that blends
        copy probability and generate probability — same as pointer-generator
        but with monotonic alignment instead of attention-based copy.

        Returns:
            logits: (B, m, vocab_size) — log-probs blending copy + generate
        """
        B, m = tgt.shape
        device = tgt.device
        N = src_tokens.size(1) if src_tokens is not None else memory.size(1)

        mem_pad_mask = None
        if src_tokens is not None and src_tokens.size(1) == memory.size(1):
            mem_pad_mask = (src_tokens == self.pad_idx)

        src_embs = self.char_embedding(src_tokens) if src_tokens is not None else None

        emb = self.dropout(self.embedding(tgt))  # (B, m, d)

        h = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        context = torch.zeros(B, self.d_model, device=device)

        # Monotonic pointer: tracks position in source
        # Derived from alignment between tgt and src (teacher-forced)
        src_ptr = torch.zeros(B, dtype=torch.long, device=device)

        # Pre-compute alignment: for each target position, determine if
        # the target char matches src[ptr]. If yes → COPY (advance ptr).
        # If no → GENERATE (don't advance). This is a greedy left-to-right alignment.
        # We compute this alignment for the whole sequence upfront for efficiency.
        if src_tokens is not None:
            # For each sample, compute ptr positions for each target step
            ptr_positions = torch.zeros(B, m, dtype=torch.long, device=device)
            is_copy = torch.zeros(B, m, dtype=torch.bool, device=device)
            for i in range(B):
                ptr = 0
                src_list = src_tokens[i].tolist()
                tgt_list = tgt[i].tolist()
                src_len = sum(1 for x in src_list if x != self.pad_idx)
                for t in range(m):
                    ptr_positions[i, t] = min(ptr, N - 1)
                    if ptr < src_len and tgt_list[t] == src_list[ptr]:
                        is_copy[i, t] = True
                        ptr += 1

        outputs = []
        for t in range(m):
            # Current source char at pointer
            cur_ptr = ptr_positions[:, t] if src_tokens is not None else torch.zeros(B, dtype=torch.long, device=device)
            clamped_ptr = cur_ptr.clamp(max=N - 1)
            src_char_emb = src_embs[torch.arange(B, device=device), clamped_ptr] if src_embs is not None else torch.zeros(B, self.d_model, device=device)

            # LSTM step
            lstm_input = torch.cat([emb[:, t], context, src_char_emb], dim=-1).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            lstm_out = lstm_out.squeeze(1)

            # Attention over encoder
            context, _ = self.attention(lstm_out, memory, mask=mem_pad_mask)

            # Generate distribution
            gen_logits = self.vocab_proj(
                self.dropout(torch.cat([lstm_out, context], dim=-1))
            )  # (B, V)
            p_vocab = F.softmax(gen_logits, dim=-1)

            # Copy gate
            gate_input = torch.cat([lstm_out, context, src_char_emb], dim=-1)
            p_copy = torch.sigmoid(self.copy_gate(gate_input))  # (B, 1)

            # Copy distribution: one-hot on the source char at pointer
            p_copy_dist = torch.zeros_like(p_vocab)  # (B, V)
            if src_tokens is not None:
                src_char_idx = src_tokens[torch.arange(B, device=device), clamped_ptr]  # (B,)
                p_copy_dist.scatter_(1, src_char_idx.unsqueeze(1), 1.0)

            # Blend
            p_final = p_copy * p_copy_dist + (1 - p_copy) * p_vocab  # (B, V)
            outputs.append(torch.log(p_final + 1e-10))

        self._last_hidden = h
        return torch.stack(outputs, dim=1)  # (B, m, V)

    @torch.no_grad()
    def greedy_decode_mono(self, memory: torch.Tensor, src_tokens: torch.Tensor,
                           sos_idx: int, eos_idx: int, max_len: int = 32):
        """Greedy decode with monotonic copy."""
        B = memory.size(0)
        N = src_tokens.size(1)
        device = memory.device

        mem_pad_mask = (src_tokens == self.pad_idx)
        src_embs = self.char_embedding(src_tokens)
        src_lens = (src_tokens != self.pad_idx).sum(dim=1)  # (B,)

        h = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        context = torch.zeros(B, self.d_model, device=device)
        src_ptr = torch.zeros(B, dtype=torch.long, device=device)

        prev_token = torch.full((B,), sos_idx, dtype=torch.long, device=device)
        generated = [prev_token.unsqueeze(1)]
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            emb = self.dropout(self.embedding(prev_token))
            clamped_ptr = src_ptr.clamp(max=N - 1)
            src_char_emb = src_embs[torch.arange(B, device=device), clamped_ptr]

            lstm_input = torch.cat([emb, context, src_char_emb], dim=-1).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            lstm_out = lstm_out.squeeze(1)

            context, _ = self.attention(lstm_out, memory, mask=mem_pad_mask)

            gen_logits = self.vocab_proj(torch.cat([lstm_out, context], dim=-1))
            p_vocab = F.softmax(gen_logits, dim=-1)

            gate_input = torch.cat([lstm_out, context, src_char_emb], dim=-1)
            p_copy = torch.sigmoid(self.copy_gate(gate_input))  # (B, 1)

            p_copy_dist = torch.zeros_like(p_vocab)
            src_char_idx = src_tokens[torch.arange(B, device=device), clamped_ptr]
            p_copy_dist.scatter_(1, src_char_idx.unsqueeze(1), 1.0)

            p_final = p_copy * p_copy_dist + (1 - p_copy) * p_vocab
            next_token = p_final.argmax(dim=-1)  # (B,)

            # Advance pointer if we copied (output matches src[ptr])
            copied = (next_token == src_char_idx) & (src_ptr < src_lens)
            src_ptr = src_ptr + copied.long()

            generated.append(next_token.unsqueeze(1))
            done = done | (next_token == eos_idx)
            if done.all():
                break
            prev_token = next_token

        return torch.cat(generated, dim=1)
