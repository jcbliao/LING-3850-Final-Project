"""Mixture-of-Experts decoder for morphological transduction.

Each expert is a small LSTMDecoder that learns a different transformation
pattern (e.g., regular suffixation, vowel change, suppletive). A router
network examines the encoder output and selects which expert(s) to apply.

This replaces Slot Attention's role: instead of slots competing over input
positions, experts compete to be the transformation applied to the whole input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import LSTMDecoder


class ExpertRouter(nn.Module):
    """Routes encoder output to experts via pooled representation.

    Supports soft (softmax), hard (argmax + straight-through), and
    Gumbel-softmax selection modes.
    """

    def __init__(self, d_model: int, num_experts: int,
                 routing_mode: str = "soft", gumbel_tau: float = 1.0,
                 use_cls_aux: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.routing_mode = routing_mode
        self.tau = gumbel_tau
        self.pool_proj = nn.Linear(d_model, d_model // 2)
        self.router = nn.Linear(d_model // 2, num_experts)

        # Optional auxiliary classifier: predict reg/irreg from the same
        # pooled representation the router uses. Shapes the representation
        # to be class-aware, helping the router distinguish verb types.
        self.cls_head = None
        if use_cls_aux:
            self.cls_head = nn.Linear(d_model // 2, 2)

    def forward(self, encoder_out: torch.Tensor,
                pad_mask: torch.Tensor | None = None):
        """
        Args:
            encoder_out: (B, n, d_model) encoder hidden states
            pad_mask: (B, n) True for padding positions
        Returns:
            expert_weights: (B, K) per-expert weights
            router_logits: (B, K) raw logits for load balancing loss
        """
        # Mean-pool over non-padding positions
        if pad_mask is not None:
            real_mask = ~pad_mask  # (B, n)
            lengths = real_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
            pooled = (encoder_out * real_mask.unsqueeze(-1).float()).sum(dim=1) / lengths
        else:
            pooled = encoder_out.mean(dim=1)  # (B, d_model)

        hidden = F.relu(self.pool_proj(pooled))  # (B, d_model//2)
        logits = self.router(hidden)  # (B, K)
        self._last_hidden = hidden  # cached for cls_aux

        if self.routing_mode == "soft":
            weights = F.softmax(logits, dim=-1)
        elif self.routing_mode == "hard":
            # Straight-through: argmax forward, softmax backward
            hard = F.one_hot(logits.argmax(dim=-1), self.num_experts).float()
            soft = F.softmax(logits, dim=-1)
            weights = hard - soft.detach() + soft
        elif self.routing_mode == "gumbel":
            if self.training:
                weights = F.gumbel_softmax(logits, tau=self.tau, hard=True)
            else:
                weights = F.one_hot(logits.argmax(dim=-1), self.num_experts).float()
        else:
            raise ValueError(f"Unknown routing_mode: {self.routing_mode}")

        return weights, logits

    def cls_loss(self, reg_labels: torch.Tensor) -> torch.Tensor:
        """Auxiliary classification loss on the router's pooled representation."""
        if self.cls_head is None or not hasattr(self, '_last_hidden'):
            return torch.tensor(0.0)
        cls_logits = self.cls_head(self._last_hidden)  # (B, 2)
        return F.cross_entropy(cls_logits, reg_labels)


class MoEDecoder(nn.Module):
    """Mixture-of-Experts decoder: K small LSTMDecoder experts with a router.

    Each expert is a full LSTMDecoder with Bahdanau attention. The router
    selects which expert(s) handle each input. In soft mode, all experts
    run and outputs are blended. In hard/gumbel mode, only the selected
    expert runs (sparse execution).
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_experts: int = 4, expert_hidden: int = 64,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, use_copy: bool = False,
                 dec_bottleneck: int = 0,
                 routing_mode: str = "soft", gumbel_tau: float = 1.0,
                 lambda_balance: float = 0.01,
                 alpha_cls_router: float = 0.0,
                 use_retrieval: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.pad_idx = pad_idx
        self.lambda_balance = lambda_balance
        self.alpha_cls_router = alpha_cls_router
        self.routing_mode = routing_mode

        self.router = ExpertRouter(
            d_model, num_experts, routing_mode, gumbel_tau,
            use_cls_aux=(alpha_cls_router > 0),
        )

        self.use_retrieval = use_retrieval
        self.experts = nn.ModuleList([
            LSTMDecoder(
                vocab_size=vocab_size, d_model=d_model,
                num_layers=num_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx, use_copy=use_copy,
                dec_bottleneck=dec_bottleneck,
                lstm_hidden=expert_hidden,
                use_retrieval=use_retrieval,
            )
            for _ in range(num_experts)
        ])

        self._last_router_logits = None
        self._last_expert_logits = None  # (B, K, m, V) cached for diversity loss

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                encoder_out: torch.Tensor | None = None,
                src_tokens: torch.Tensor | None = None,
                retrieval_memory: torch.Tensor | None = None,
                retrieval_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Same signature as LSTMDecoder.forward() — drop-in replacement.

        Args:
            tgt: (B, m) target token indices
            memory: (B, N, d_model) encoder output (no slots in MoE mode)
            encoder_out: (B, N, d_model) or None (for copy mechanism)
            src_tokens: (B, N) source tokens (for copy/padding)
            retrieval_memory: (B, M, d_model) retrieved-target memory, or None
            retrieval_pad_mask: (B, M) True where padded, or None
        Returns:
            logits: (B, m, V)
        """
        retr_kwargs = {}
        if self.use_retrieval:
            retr_kwargs["retrieval_memory"] = retrieval_memory
            retr_kwargs["retrieval_pad_mask"] = retrieval_pad_mask
        # Compute padding mask from src_tokens
        pad_mask = None
        if src_tokens is not None:
            pad_mask = (src_tokens == self.pad_idx)

        # Route
        expert_weights, router_logits = self.router(memory, pad_mask)  # (B, K)
        self._last_router_logits = router_logits

        # Sparse execution for hard/gumbel routing
        if self.routing_mode in ("hard", "gumbel") and not self.training:
            # At eval: run only the selected expert per sample
            expert_idx = expert_weights.argmax(dim=-1)  # (B,)
            B, m = tgt.shape
            V = self.experts[0].vocab_size
            logits = torch.zeros(B, m, V, device=tgt.device)
            for k in range(self.num_experts):
                mask_k = (expert_idx == k)  # (B,)
                if not mask_k.any():
                    continue
                # Slice retrieval memory by sample subset too
                expert_retr_kwargs = {}
                if self.use_retrieval and retrieval_memory is not None:
                    expert_retr_kwargs["retrieval_memory"] = retrieval_memory[mask_k]
                    expert_retr_kwargs["retrieval_pad_mask"] = (
                        retrieval_pad_mask[mask_k] if retrieval_pad_mask is not None else None
                    )
                logits_k = self.experts[k](
                    tgt[mask_k], memory[mask_k],
                    encoder_out=encoder_out[mask_k] if encoder_out is not None else None,
                    src_tokens=src_tokens[mask_k] if src_tokens is not None else None,
                    **expert_retr_kwargs,
                )
                logits[mask_k] = logits_k
            return logits

        # Dense execution: run all experts, weighted sum
        all_logits = []
        for expert in self.experts:
            logits_k = expert(tgt, memory, encoder_out=encoder_out,
                              src_tokens=src_tokens, **retr_kwargs)  # (B, m, V)
            all_logits.append(logits_k)

        # Stack: (B, K, m, V), weight: (B, K, 1, 1)
        stacked = torch.stack(all_logits, dim=1)
        self._last_expert_logits = stacked
        weights = expert_weights.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        logits = (stacked * weights).sum(dim=1)  # (B, m, V)

        return logits

    def load_balancing_loss(self) -> torch.Tensor:
        """Switch Transformer load balancing loss.

        L_balance = K * sum_k(f_k * P_k)
        where f_k = fraction of tokens routed to expert k
              P_k = mean router probability for expert k
        Minimum when all experts get equal traffic.
        """
        if self._last_router_logits is None:
            return torch.tensor(0.0)

        router_probs = F.softmax(self._last_router_logits, dim=-1)  # (B, K)
        # For soft routing, f_k = P_k; for hard, f_k = actual assignment fraction
        if self.routing_mode == "soft":
            f = router_probs.mean(dim=0)  # (K,)
        else:
            # Use hard assignments
            assignments = router_probs.argmax(dim=-1)  # (B,)
            f = torch.zeros(self.num_experts, device=router_probs.device)
            for k in range(self.num_experts):
                f[k] = (assignments == k).float().mean()

        P = router_probs.mean(dim=0)  # (K,)
        return self.num_experts * (f * P).sum()

    def diversity_loss(self) -> torch.Tensor:
        """Lateral inhibition: penalize experts for producing similar outputs.

        For each input, compute mean pairwise cosine similarity between expert
        output distributions (averaged over timesteps). High similarity means
        experts are redundant — the loss pushes them apart.

        Returns:
            scalar: mean pairwise cosine similarity (0 = orthogonal, 1 = identical)
        """
        if self._last_expert_logits is None:
            return torch.tensor(0.0)

        # (B, K, m, V) → (B, K, m*V) — flatten time and vocab
        B, K, m, V = self._last_expert_logits.shape
        flat = self._last_expert_logits.reshape(B, K, m * V)

        # Normalize each expert's output vector
        flat_norm = F.normalize(flat, dim=-1)  # (B, K, m*V)

        # Pairwise cosine similarity: (B, K, K)
        sim = torch.bmm(flat_norm, flat_norm.transpose(1, 2))  # (B, K, K)

        # Mean of upper triangle (exclude diagonal = self-similarity of 1.0)
        mask = torch.triu(torch.ones(K, K, device=sim.device, dtype=torch.bool), diagonal=1)
        pairwise_sim = sim[:, mask].mean()

        return pairwise_sim

    def router_cls_loss(self, reg_labels: torch.Tensor) -> torch.Tensor:
        """Auxiliary router classification loss (reg/irreg prediction)."""
        return self.router.cls_loss(reg_labels)

    def get_expert_assignments(self, memory: torch.Tensor,
                               src_tokens: torch.Tensor | None = None):
        """Return which expert each input is routed to (for analysis).

        Returns:
            expert_idx: (B,) index of selected expert
            expert_probs: (B, K) soft routing probabilities
        """
        pad_mask = None
        if src_tokens is not None:
            pad_mask = (src_tokens == self.pad_idx)
        weights, logits = self.router(memory, pad_mask)
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1), probs
