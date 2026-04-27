"""Slot sparsification layers: S ∈ ℝᴷˣᵈ → M' ∈ ℝᴷˣᵈ (sparse).

Modes:
- InputConditionalL0Drop (default): Soft hard-concrete gates with Lagrangian.
- L0Drop (legacy): Static per-slot gates.
- TopKDrop: Hard top-k selection with straight-through gradients.
- GumbelSlotRouter: Gumbel-softmax categorical selection of slot subsets.

Reference: Louizos, Welling, Kingma (2018) "Learning Sparse Neural Networks
through L0 Regularization"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L0Drop(nn.Module):
    """Static hard-concrete gates (input-independent).

    Each slot k has a single learnable log_alpha_k — same gate for every input.
    Kept for comparison but InputConditionalL0Drop is preferred.
    """

    def __init__(self, num_slots: int, beta: float = 0.66,
                 gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.num_slots = num_slots
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.log_alpha = nn.Parameter(torch.zeros(num_slots))

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        z = self._sample_z(slots)  # (B, K)
        return slots * z.unsqueeze(-1)

    def _sample_z(self, slots: torch.Tensor) -> torch.Tensor:
        B, K, _ = slots.shape
        if self.training:
            u = torch.rand(B, K, device=slots.device).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha).unsqueeze(0).expand(B, -1)
        z = s * (self.zeta - self.gamma) + self.gamma
        return z.clamp(0.0, 1.0)

    def l0_loss(self) -> torch.Tensor:
        return torch.sigmoid(
            self.log_alpha - self.beta * torch.log(torch.tensor(-self.gamma / self.zeta))
        ).sum()


class InputConditionalL0Drop(nn.Module):
    """Input-conditional hard-concrete gates.

    Gate values are computed from slot contents: log_alpha_k = MLP(slot_k),
    so different inputs activate different slot subsets. This enables the
    "variable number of active slots depending on the input" described in
    the proposal.

    Uses a Lagrangian constraint to target a specific expected L0 value,
    avoiding the fixed-lambda scale mismatch problem.
    """

    def __init__(self, d_slot: int, num_slots: int, beta: float = 0.66,
                 gamma: float = -0.1, zeta: float = 1.1,
                 target_l0: float = 2.0, lagrangian_lr: float = 0.01):
        super().__init__()
        self.num_slots = num_slots
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.target_l0 = target_l0
        self.lagrangian_lr = lagrangian_lr

        # MLP to compute per-slot gate logits from slot content
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_slot, d_slot // 2),
            nn.ReLU(),
            nn.Linear(d_slot // 2, 1),
        )

        # Lagrangian multiplier (not a model parameter — updated manually)
        self.register_buffer("lagrangian_lambda", torch.tensor(0.0))

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Apply input-conditional hard-concrete gates.

        Args:
            slots: (B, K, d_slot)
        Returns:
            gated_slots: (B, K, d_slot)
        """
        z = self._sample_z(slots)  # (B, K)
        return slots * z.unsqueeze(-1)

    def _sample_z(self, slots: torch.Tensor) -> torch.Tensor:
        """Compute input-conditional gates via hard-concrete."""
        # log_alpha depends on slot content
        log_alpha = self.gate_mlp(slots).squeeze(-1)  # (B, K)

        if self.training:
            u = torch.rand_like(log_alpha).clamp(1e-8, 1 - 1e-8)
            s = torch.sigmoid((torch.log(u / (1 - u)) + log_alpha) / self.beta)
        else:
            s = torch.sigmoid(log_alpha)

        z = s * (self.zeta - self.gamma) + self.gamma
        # Store for l0_loss computation
        self._last_log_alpha = log_alpha
        return z.clamp(0.0, 1.0)

    def l0_loss(self) -> torch.Tensor:
        """Expected L0 norm averaged over the batch.

        Returns the Lagrangian loss: lambda * (E[L0] - target_l0)
        """
        # Per-example expected L0
        gate_probs = torch.sigmoid(
            self._last_log_alpha
            - self.beta * torch.log(torch.tensor(-self.gamma / self.zeta,
                                                  device=self._last_log_alpha.device))
        )  # (B, K)
        expected_l0 = gate_probs.sum(dim=-1).mean()  # scalar: avg over batch

        # Lagrangian: lambda * (L0 - target)
        constraint = expected_l0 - self.target_l0
        return self.lagrangian_lambda * constraint + constraint ** 2

    @torch.no_grad()
    def update_lagrangian(self):
        """Update Lagrangian multiplier after each epoch.

        Dual gradient ascent: increase lambda if L0 > target, decrease if L0 < target.
        """
        gate_probs = torch.sigmoid(
            self._last_log_alpha
            - self.beta * torch.log(torch.tensor(-self.gamma / self.zeta,
                                                  device=self._last_log_alpha.device))
        )
        expected_l0 = gate_probs.sum(dim=-1).mean()
        constraint = expected_l0 - self.target_l0
        self.lagrangian_lambda = torch.clamp(
            self.lagrangian_lambda + self.lagrangian_lr * constraint,
            min=0.0,
        )


class TopKDrop(nn.Module):
    """Hard top-k slot selection with straight-through gradients.

    For each input, scores all K slots via an MLP, keeps the top-k by score,
    and zeros out the rest. Guarantees exactly k active slots per input.

    Training: straight-through estimator — forward uses hard top-k mask,
    backward passes gradients through the soft scores.
    Eval: deterministic top-k selection.
    """

    def __init__(self, d_slot: int, num_slots: int, k: int = 2):
        super().__init__()
        self.num_slots = num_slots
        self.k = k
        # target_l0 for compatibility with training loop logging
        self.target_l0 = float(k)

        # MLP to score each slot
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_slot, d_slot // 2),
            nn.ReLU(),
            nn.Linear(d_slot // 2, 1),
        )

        # Dummy buffer for API compatibility
        self.register_buffer("lagrangian_lambda", torch.tensor(0.0))

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: (B, K, d_slot)
        Returns:
            gated_slots: (B, K, d_slot) with exactly k non-zero slots per example
        """
        B, K, d = slots.shape
        # Score each slot based on its content
        scores = self.gate_mlp(slots).squeeze(-1)  # (B, K)
        self._last_scores = scores

        # Hard top-k mask
        _, topk_indices = scores.topk(self.k, dim=-1)  # (B, k)
        mask = torch.zeros(B, K, device=slots.device)
        mask.scatter_(1, topk_indices, 1.0)  # (B, K) binary

        if self.training:
            # Straight-through: forward uses hard mask, backward uses soft scores
            soft_scores = torch.sigmoid(scores)
            mask = mask - soft_scores.detach() + soft_scores  # ST trick

        return slots * mask.unsqueeze(-1)

    def l0_loss(self) -> torch.Tensor:
        """No L0 loss needed — top-k is exact. Return 0 for compatibility."""
        return torch.tensor(0.0, device=self._last_scores.device)

    def update_lagrangian(self):
        """No-op for compatibility."""
        pass


class GumbelSlotRouter(nn.Module):
    """Gumbel-softmax slot routing: learns a categorical distribution over
    which slots to activate for each input.

    Each slot gets a score from an MLP. During training, Gumbel-softmax
    produces soft-but-peaked weights that approximate discrete selection.
    During eval, hard argmax selection.

    With k>1, applies Gumbel-softmax k times without replacement (using
    the successive-rejection trick) to select exactly k slots.
    """

    def __init__(self, d_slot: int, num_slots: int, k: int = 2,
                 temperature: float = 1.0, temperature_min: float = 0.1,
                 anneal_rate: float = 0.003):
        super().__init__()
        self.num_slots = num_slots
        self.k = k
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.anneal_rate = anneal_rate
        self.target_l0 = float(k)

        # MLP to score each slot
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_slot, d_slot // 2),
            nn.ReLU(),
            nn.Linear(d_slot // 2, 1),
        )

        self.register_buffer("lagrangian_lambda", torch.tensor(0.0))
        self.register_buffer("step_count", torch.tensor(0))

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: (B, K, d_slot)
        Returns:
            gated_slots: (B, K, d_slot) with ~k active slots
        """
        B, K, d = slots.shape
        logits = self.gate_mlp(slots).squeeze(-1)  # (B, K)
        self._last_logits = logits

        if self.training:
            # Current temperature (annealed over training)
            tau = max(
                self.temperature_min,
                self.temperature * (1 - self.anneal_rate) ** self.step_count.item(),
            )
            self.step_count += 1

            # Sample k slots via repeated Gumbel-softmax
            mask = torch.zeros(B, K, device=slots.device)
            remaining_logits = logits.clone()
            for _ in range(self.k):
                soft = F.gumbel_softmax(remaining_logits, tau=tau, hard=True, dim=-1)
                mask = mask + soft
                # Remove selected slot from future rounds
                remaining_logits = remaining_logits - soft * 1e9
            mask = mask.clamp(0.0, 1.0)
        else:
            # Hard top-k at eval
            _, topk_indices = logits.topk(self.k, dim=-1)
            mask = torch.zeros(B, K, device=slots.device)
            mask.scatter_(1, topk_indices, 1.0)

        return slots * mask.unsqueeze(-1)

    def l0_loss(self) -> torch.Tensor:
        """Entropy regularization — encourage peaked distributions."""
        probs = F.softmax(self._last_logits, dim=-1)  # (B, K)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        # Minimize entropy = encourage discrete selection
        return entropy * 0.1

    def update_lagrangian(self):
        """No-op for compatibility."""
        pass
