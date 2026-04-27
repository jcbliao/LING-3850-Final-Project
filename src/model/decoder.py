"""Decoders for character-level transduction.

Two decoder types:
- TransformerCharDecoder: Transformer decoder with optional copy mechanism
- LSTMDecoder: LSTM decoder with Bahdanau attention (Kirov & Cotterell style)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PositionalEncoding


class TransformerCharDecoder(nn.Module):
    """Transformer decoder with cross-attention over slot memory.

    When use_copy=True, adds a pointer/copy mechanism (See et al. 2017):
    at each step the model chooses p_gen (generate from vocab) vs 1-p_gen
    (copy from source). The source encoder hidden states are used for
    copy attention, and the final distribution blends both.

    Input:  target prefix indices (batch, t), slot memory M' (batch, K, d)
    Output: logits (batch, t, vocab_size)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, use_copy: bool = False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_copy = use_copy
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

        if use_copy:
            # Copy attention: separate query and key projections
            # (decoder states live in slot-conditioned space; encoder states
            # live in raw encoder space — separate projections bridge the gap)
            self.copy_query_proj = nn.Linear(d_model, d_model, bias=False)
            self.copy_key_proj = nn.Linear(d_model, d_model, bias=False)
            # Gate: p_gen = sigma(w · [decoder_state; context; embedding])
            self.p_gen_linear = nn.Linear(d_model * 3, 1)
            # Bias toward copying (sigmoid(-2) ≈ 0.12 → p_gen starts low)
            nn.init.constant_(self.p_gen_linear.bias, -2.0)
            # Monotonic alignment bias: Gaussian centered on diagonal
            # log_alpha controls sharpness (higher = sharper peak on diagonal)
            # Init: alpha=1.0 (log_alpha=0) — moderate bias, learnable
            self.copy_align_log_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                encoder_out: torch.Tensor | None = None,
                src_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            tgt: (batch, m) target token indices (teacher-forced, includes <sos>)
            memory: (batch, K, d_model) pruned slot memory M'
            encoder_out: (batch, n, d_model) encoder hidden states (needed for copy)
            src_tokens: (batch, n) source token indices (needed for copy)
        Returns:
            logits: (batch, m, vocab_size) — log-probs if use_copy, raw logits otherwise
        """
        m = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(m, m, device=tgt.device, dtype=torch.bool), diagonal=1
        )
        tgt_pad_mask = (tgt == self.pad_idx)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(tgt_emb)
        x = self.transformer(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        if not self.use_copy or encoder_out is None or src_tokens is None:
            return self.output_proj(x)

        # --- Copy mechanism ---
        # Generate distribution
        gen_logits = self.output_proj(x)  # (B, m, V)
        p_vocab = F.softmax(gen_logits, dim=-1)

        # Copy attention over source positions (separate Q/K projections)
        copy_query = self.copy_query_proj(x)  # (B, m, d)
        copy_key = self.copy_key_proj(encoder_out)  # (B, n, d)
        copy_energy = torch.bmm(copy_query, copy_key.transpose(1, 2))  # (B, m, n)

        # Monotonic alignment bias: Gaussian centered on diagonal
        # Output position t should copy from source position t (no scaling needed
        # for morphological transduction where output ≈ input + suffix)
        n_src = encoder_out.size(1)
        alpha = torch.exp(self.copy_align_log_alpha)  # learnable sharpness
        t_pos = torch.arange(m, device=x.device, dtype=x.dtype)  # (m,)
        n_pos = torch.arange(n_src, device=x.device, dtype=x.dtype)  # (n,)
        # Gaussian bias: peaked when t ≈ n (diagonal alignment)
        align_bias = -alpha * (t_pos.unsqueeze(1) - n_pos.unsqueeze(0)) ** 2  # (m, n)
        copy_energy = copy_energy + align_bias.unsqueeze(0)  # (B, m, n)

        # Mask padding in source
        src_pad_mask = (src_tokens == self.pad_idx).unsqueeze(1)  # (B, 1, n)
        copy_energy = copy_energy.masked_fill(src_pad_mask, -1e9)
        copy_attn = F.softmax(copy_energy, dim=-1)  # (B, m, n)

        # Scatter copy attention into vocab-sized distribution
        p_copy = torch.zeros_like(p_vocab)  # (B, m, V)
        src_expanded = src_tokens.unsqueeze(1).expand(-1, m, -1)  # (B, m, n)
        p_copy.scatter_add_(2, src_expanded, copy_attn)

        # Compute copy context for p_gen gate
        copy_context = torch.bmm(copy_attn, encoder_out)  # (B, m, d)

        # p_gen gate: blend generate vs copy
        gate_input = torch.cat([x, copy_context, tgt_emb], dim=-1)  # (B, m, 3d)
        p_gen = torch.sigmoid(self.p_gen_linear(gate_input))  # (B, m, 1)

        # Final distribution
        p_final = p_gen * p_vocab + (1 - p_gen) * p_copy  # (B, m, V)

        # Return log-probs (caller uses NLL loss instead of cross-entropy)
        return torch.log(p_final + 1e-10)


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention: score(h_dec, h_enc) = v^T tanh(W1·h_dec + W2·h_enc)."""

    def __init__(self, dec_dim: int, enc_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_state: torch.Tensor, encoder_out: torch.Tensor,
                mask: torch.Tensor | None = None):
        """
        Args:
            decoder_state: (B, dec_dim) current decoder hidden state
            encoder_out: (B, N, enc_dim) encoder outputs
            mask: (B, N) True for positions to ignore (padding)
        Returns:
            context: (B, enc_dim) weighted sum of encoder outputs
            attn_weights: (B, N) attention distribution
        """
        # (B, 1, attn_dim) + (B, N, attn_dim) → (B, N, attn_dim)
        energy = torch.tanh(
            self.W_dec(decoder_state).unsqueeze(1) + self.W_enc(encoder_out)
        )
        scores = self.v(energy).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)  # (B, N)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_out).squeeze(1)  # (B, enc_dim)
        return context, attn_weights


class LSTMDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention over encoder/slot memory.

    Replicates the Kirov & Cotterell (2018) decoder architecture:
    - Unidirectional LSTM with additive attention
    - Every character generated from vocabulary (no copy mechanism)
    - Sequential hidden state enables pattern memorization for irregulars
    - Bahdanau attention naturally learns monotonic alignment

    Input:  target prefix indices (batch, t), memory (batch, K_or_N, d)
    Output: logits (batch, t, vocab_size)
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, use_copy: bool = False,
                 dec_bottleneck: int = 0, lstm_hidden: int = 0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.use_copy = use_copy
        # lstm_hidden=0 means use d_model (default behavior)
        self.lstm_dim = lstm_hidden if lstm_hidden > 0 else d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.attention = BahdanauAttention(self.lstm_dim, d_model, attn_dim=d_model)
        # LSTM input: embedding + attention context
        self.lstm = nn.LSTM(
            input_size=d_model + d_model,  # [embedding(d_model); context(d_model)]
            hidden_size=self.lstm_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        # Output projection: [lstm_output; context] → vocab
        # lstm_output is lstm_dim, context is d_model (full slot dimension)
        # Optional bottleneck constrains decoder's independent capacity
        if dec_bottleneck > 0:
            self.output_proj = nn.Sequential(
                nn.Linear(self.lstm_dim + d_model, dec_bottleneck),
                nn.ReLU(),
                nn.Linear(dec_bottleneck, vocab_size),
            )
        else:
            self.output_proj = nn.Linear(self.lstm_dim + d_model, vocab_size)

        if use_copy:
            # Copy attention over encoder states (separate from memory attention)
            self.copy_attention = BahdanauAttention(self.lstm_dim, d_model, attn_dim=d_model)
            # p_gen gate: [lstm_out; memory_context; copy_context; embedding] → 1
            self.p_gen_linear = nn.Linear(self.lstm_dim + d_model * 3, 1)
            nn.init.constant_(self.p_gen_linear.bias, -2.0)  # bias toward copy

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                encoder_out: torch.Tensor | None = None,
                src_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            tgt: (B, m) target token indices (with <sos> prefix)
            memory: (B, K_or_N, d_model) encoder/slot memory for attention
            encoder_out: ignored (kept for API compatibility)
            src_tokens: ignored (kept for API compatibility)
        Returns:
            logits: (B, m, vocab_size)
        """
        B, m = tgt.shape
        device = tgt.device

        # Memory padding mask: for encoder output use src_tokens, for slots use norms
        mem_len = memory.size(1)
        if src_tokens is not None and src_tokens.size(1) == mem_len:
            mem_pad_mask = (src_tokens == self.pad_idx)  # (B, N)
        else:
            # Slot memory or mismatched sizes: no padding in slots
            mem_pad_mask = None

        emb = self.dropout(self.embedding(tgt))  # (B, m, d)

        # Initialize LSTM state
        h = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)

        # Initial context (zeros) — always d_model (full slot dimension)
        context = torch.zeros(B, self.d_model, device=device)

        # Source padding mask for copy attention
        src_pad_mask = None
        if src_tokens is not None:
            src_pad_mask = (src_tokens == self.pad_idx)  # (B, N)

        outputs = []
        for t in range(m):
            # Input: [embedding_t; context_t-1]
            lstm_input = torch.cat([emb[:, t], context], dim=-1).unsqueeze(1)  # (B, 1, 2d)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))  # (B, 1, d)
            lstm_out = lstm_out.squeeze(1)  # (B, d)

            # Attention over memory (slots or encoder output)
            context, _ = self.attention(lstm_out, memory, mask=mem_pad_mask)  # (B, d)

            if not self.use_copy or encoder_out is None or src_tokens is None:
                # Pure generation
                out = self.output_proj(
                    self.dropout(torch.cat([lstm_out, context], dim=-1))
                )  # (B, vocab)
                outputs.append(out)
            else:
                # Copy mechanism (Pointer-Generator for LSTM)
                gen_logits = self.output_proj(
                    self.dropout(torch.cat([lstm_out, context], dim=-1))
                )  # (B, vocab)
                p_vocab = F.softmax(gen_logits, dim=-1)

                # Copy attention over encoder states
                copy_context, copy_attn = self.copy_attention(
                    lstm_out, encoder_out, mask=src_pad_mask
                )  # (B, d), (B, N)

                # Scatter copy attention into vocab distribution
                p_copy = torch.zeros_like(p_vocab)  # (B, V)
                p_copy.scatter_add_(1, src_tokens, copy_attn)

                # p_gen gate
                gate_input = torch.cat(
                    [lstm_out, context, copy_context, emb[:, t]], dim=-1
                )  # (B, 4d)
                p_gen = torch.sigmoid(self.p_gen_linear(gate_input))  # (B, 1)

                p_final = p_gen * p_vocab + (1 - p_gen) * p_copy
                outputs.append(torch.log(p_final + 1e-10))

        self._last_hidden = h  # (num_layers, B, lstm_dim) — expose for population decoder
        return torch.stack(outputs, dim=1)  # (B, m, vocab)
