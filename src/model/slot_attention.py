"""Slot Attention module: H ∈ ℝⁿˣᵈ → S ∈ ℝᴷˣᵈ

Iterative competitive attention where K slots compete to explain input.
Adapted for language following Behjati & Henderson: each slot has a learnable mean
with fixed initialization variance.

Two modes:
- Single-head (default, nhead=1): slots compete over positions (original Slot Attention)
- Multi-head (nhead>1): slots compete independently within each feature subspace,
  allowing a slot to attend to different positions in different feature groups.
  More suitable for morphology where rules don't partition by position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttentionModule(nn.Module):
    """Multi-head Slot Attention with learnable slot means.

    Each iteration:
        1. Compute attention per head: softmax over slot dim so slots compete
        2. Aggregate: weighted sum of values per head, concatenate
        3. Update: GRU recurrence + MLP residual

    With nhead=1, this is identical to original Slot Attention.
    With nhead>1, each head operates on a d_slot/nhead feature subspace,
    giving slots the ability to attend to different positions for different
    feature groups (feature-level competition).
    """

    def __init__(self, d_input: int, d_slot: int, num_slots: int,
                 num_iterations: int = 3, mlp_hidden: int = 256,
                 init_std: float = 0.1, nhead: int = 1):
        super().__init__()
        assert d_slot % nhead == 0, f"d_slot ({d_slot}) must be divisible by nhead ({nhead})"
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.d_slot = d_slot
        self.nhead = nhead
        self.d_head = d_slot // nhead

        # Learnable slot initialization (per-slot mean, fixed variance)
        self.slot_mu = nn.Parameter(torch.randn(num_slots, d_slot))
        self.init_std = init_std

        # Projections for attention
        self.proj_k = nn.Linear(d_input, d_slot, bias=False)
        self.proj_v = nn.Linear(d_input, d_slot, bias=False)
        self.proj_q = nn.Linear(d_slot, d_slot, bias=False)

        # Layer norms
        self.norm_input = nn.LayerNorm(d_input)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_mlp = nn.LayerNorm(d_slot)

        # GRU for slot update
        self.gru = nn.GRUCell(d_slot, d_slot)

        # MLP residual
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, d_slot),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (batch, n, d_input) encoder output
        Returns:
            slots: (batch, K, d_slot)
        """
        B, N, _ = H.shape
        K = self.num_slots
        nH = self.nhead
        d_h = self.d_head

        # Initialize slots from learnable means + noise
        slots = self.slot_mu.unsqueeze(0).expand(B, -1, -1)
        if self.training:
            slots = slots + torch.randn_like(slots) * self.init_std

        # Pre-compute keys and values (don't change across iterations)
        H_norm = self.norm_input(H)
        k = self.proj_k(H_norm)  # (B, N, d_slot)
        v = self.proj_v(H_norm)  # (B, N, d_slot)

        # Reshape for multi-head: (B, N, d_slot) → (B*nH, N, d_h)
        k = k.view(B, N, nH, d_h).permute(0, 2, 1, 3).reshape(B * nH, N, d_h)
        v = v.view(B, N, nH, d_h).permute(0, 2, 1, 3).reshape(B * nH, N, d_h)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.proj_q(slots_norm)  # (B, K, d_slot)

            # Reshape queries for multi-head: (B, K, d_slot) → (B*nH, K, d_h)
            q = q.view(B, K, nH, d_h).permute(0, 2, 1, 3).reshape(B * nH, K, d_h)

            # Attention per head: (B*nH, K, N), softmax over K (slots compete)
            scale = d_h ** 0.5
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / scale
            attn = F.softmax(attn_logits, dim=1)  # slots compete within each head

            # Weighted mean of values per head
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_norm, v)  # (B*nH, K, d_h)

            # Reshape back: (B*nH, K, d_h) → (B, K, d_slot)
            updates = updates.reshape(B, nH, K, d_h).permute(0, 2, 1, 3).reshape(B, K, self.d_slot)

            # GRU update
            slots = self.gru(
                updates.reshape(B * K, self.d_slot),
                slots_prev.reshape(B * K, self.d_slot),
            ).reshape(B, K, self.d_slot)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots
