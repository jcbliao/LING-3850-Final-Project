"""Character encoders: Transformer and BiLSTM.

x = (x₁,...,xₙ) → H ∈ ℝⁿˣᵈ
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerCharEncoder(nn.Module):
    """Embeds characters and encodes with a Transformer.

    Input:  token indices (batch, n)
    Output: H ∈ (batch, n, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pad_idx = pad_idx

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, n) token indices
        Returns:
            H: (batch, n, d_model)
        """
        pad_mask = (src == self.pad_idx)  # (batch, n)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        H = self.transformer(x, src_key_padding_mask=pad_mask)
        return H


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder (Kirov & Cotterell 2018 style).

    Input:  token indices (batch, n)
    Output: H ∈ (batch, n, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_layers: int = 2, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,  # concat fwd+bwd = d_model
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, n) token indices
        Returns:
            H: (batch, n, d_model)
        """
        # Pack padded sequences for efficient LSTM processing
        lengths = (src != self.pad_idx).sum(dim=1).cpu()  # (batch,)
        emb = self.dropout(self.embedding(src))  # (batch, n, d_model)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False,
        )
        output, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=src.size(1),
        )
        return self.dropout(H)  # (batch, n, d_model)
