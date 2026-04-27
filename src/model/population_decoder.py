"""Population-coded decoder for morphological transduction.

Inspired by population coding in neuroscience: K decoder "neurons" each
process the full input and produce both an output (logits) and a confidence
(firing rate). The final prediction is the confidence-weighted combination
of all neurons' outputs. No external router — each neuron self-determines
its contribution based on how well the input matches its learned tuning.

Lateral inhibition (clamped diversity loss) prevents neurons from converging
to identical tuning curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import LSTMDecoder, BahdanauAttention
from .edit_decoder import EditDecoder, EditVocab, apply_edits, EDIT_PAD, EDIT_END


class PopulationNeuron(nn.Module):
    """A single decoder neuron: LSTM decoder + self-gating confidence.

    Three confidence modes determine how the neuron computes its "firing rate":

    - "input": Confidence from a learned projection of encoder output.
      The neuron scans the input and decides its relevance before decoding.

    - "hidden": Confidence from the LSTM's final hidden state after decoding.
      The neuron's confidence reflects its actual processing experience —
      how well it "understood" the transformation. More faithful: a neuron's
      firing rate emerges from computation, not from a separate input scan.

    - "certainty": Confidence = mean max probability across output timesteps.
      Parameter-free. The most biologically faithful: firing rate IS output
      certainty. A neuron that produces peaked, confident predictions
      naturally contributes more. No learned confidence head needed.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, use_copy: bool = False,
                 dec_bottleneck: int = 0, lstm_hidden: int = 0,
                 confidence_mode: str = "input"):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.confidence_mode = confidence_mode
        lstm_dim = lstm_hidden if lstm_hidden > 0 else d_model

        self.decoder = LSTMDecoder(
            vocab_size=vocab_size, d_model=d_model,
            num_layers=num_layers, d_ff=d_ff, dropout=dropout,
            pad_idx=pad_idx, use_copy=use_copy,
            dec_bottleneck=dec_bottleneck, lstm_hidden=lstm_hidden,
        )

        if confidence_mode == "input":
            # Confidence from encoder output (v17a style)
            self.confidence_proj = nn.Sequential(
                nn.Linear(d_model, lstm_dim),
                nn.ReLU(),
                nn.Linear(lstm_dim, 1),
            )
        elif confidence_mode == "hidden":
            # Confidence from LSTM final hidden state (v17b)
            self.confidence_proj = nn.Sequential(
                nn.Linear(lstm_dim, lstm_dim // 2),
                nn.ReLU(),
                nn.Linear(lstm_dim // 2, 1),
            )
        # "certainty" mode needs no extra parameters

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                encoder_out: torch.Tensor | None = None,
                src_tokens: torch.Tensor | None = None):
        """
        Returns:
            logits: (B, m, V) decoder output
            confidence: (B,) scalar confidence for this input
        """
        logits = self.decoder(tgt, memory, encoder_out=encoder_out,
                              src_tokens=src_tokens)  # (B, m, V)

        if self.confidence_mode == "input":
            # Confidence from mean-pooled encoder output
            if src_tokens is not None:
                pad_mask = (src_tokens == self.pad_idx)
                real_mask = ~pad_mask
                lengths = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (memory * real_mask.unsqueeze(-1).float()).sum(dim=1) / lengths
            else:
                pooled = memory.mean(dim=1)
            confidence = self.confidence_proj(pooled).squeeze(-1)  # (B,)

        elif self.confidence_mode == "hidden":
            # Confidence from LSTM's final hidden state (top layer)
            h = self.decoder._last_hidden[-1]  # (B, lstm_dim)
            confidence = self.confidence_proj(h).squeeze(-1)  # (B,)

        elif self.confidence_mode == "certainty":
            # Confidence = mean max probability across timesteps
            # Parameter-free: firing rate IS prediction certainty
            probs = F.softmax(logits, dim=-1)  # (B, m, V)
            max_probs = probs.max(dim=-1).values  # (B, m)
            # Mask padding positions in target
            if tgt is not None:
                tgt_mask = (tgt != self.pad_idx).float()  # (B, m)
                lengths = tgt_mask.sum(dim=1).clamp(min=1)
                confidence = (max_probs * tgt_mask).sum(dim=1) / lengths  # (B,)
            else:
                confidence = max_probs.mean(dim=1)  # (B,)
            # Scale to logit space for softmax with other neurons
            confidence = torch.log(confidence + 1e-10)

        return logits, confidence


class PopulationDecoder(nn.Module):
    """Population-coded decoder: K self-gating neurons with lateral inhibition.

    Each neuron independently processes the input and self-determines its
    contribution weight (confidence). The final output is the confidence-
    weighted combination of all neurons' outputs. Diversity loss prevents
    neurons from learning identical tuning curves.

    No external router — routing emerges from each neuron's learned tuning.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 num_experts: int = 4, expert_hidden: int = 64,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, use_copy: bool = False,
                 dec_bottleneck: int = 0,
                 lambda_balance: float = 0.0,
                 lambda_diversity: float = 0.1,
                 confidence_mode: str = "input",
                 neuron_dropout: float = 0.0,
                 confidence_tau: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.lambda_balance = lambda_balance
        self.lambda_diversity = lambda_diversity
        self.confidence_mode = confidence_mode
        self.neuron_dropout = neuron_dropout
        self.confidence_tau = confidence_tau
        # Keep these for compatibility with full_model.py checks
        self.alpha_cls_router = 0.0
        self.routing_mode = "population"

        self.neurons = nn.ModuleList([
            PopulationNeuron(
                vocab_size=vocab_size, d_model=d_model,
                num_layers=num_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx, use_copy=use_copy,
                dec_bottleneck=dec_bottleneck, lstm_hidden=expert_hidden,
                confidence_mode=confidence_mode,
            )
            for _ in range(num_experts)
        ])

        self._last_expert_logits = None  # (B, K, m, V) for diversity loss
        self._last_confidences = None    # (B, K) for analysis

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                encoder_out: torch.Tensor | None = None,
                src_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """
        Same signature as LSTMDecoder.forward() — drop-in replacement.
        """
        all_logits = []
        all_confidences = []

        for neuron in self.neurons:
            logits_k, conf_k = neuron(tgt, memory, encoder_out=encoder_out,
                                      src_tokens=src_tokens)
            all_logits.append(logits_k)
            all_confidences.append(conf_k)

        # Stack: (B, K, m, V) and (B, K)
        stacked = torch.stack(all_logits, dim=1)
        confidences = torch.stack(all_confidences, dim=1)  # (B, K)
        self._last_expert_logits = stacked
        self._last_confidences = confidences

        # Neuron dropout: randomly mask neurons during training.
        # Analogous to synaptic pruning — forces each neuron to be
        # independently competent, prevents co-adaptation.
        if self.training and self.neuron_dropout > 0:
            mask = torch.bernoulli(
                torch.full((confidences.size(0), self.num_experts),
                           1.0 - self.neuron_dropout, device=confidences.device)
            )  # (B, K), 1 = keep, 0 = drop
            # Ensure at least one neuron survives per sample
            all_dropped = (mask.sum(dim=1) == 0)
            if all_dropped.any():
                random_idx = torch.randint(self.num_experts, (all_dropped.sum(),),
                                           device=mask.device)
                mask[all_dropped, random_idx] = 1.0
            # Mask confidences: dropped neurons get -inf before softmax
            confidences = confidences.masked_fill(mask == 0, -1e9)

        # Softmax over neurons — temperature controls sharpness
        # Low τ = peaky (specialist), high τ = uniform (collaborative)
        weights = F.softmax(confidences / self.confidence_tau, dim=-1)  # (B, K)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        logits = (stacked * weights).sum(dim=1)  # (B, m, V)

        return logits

    def diversity_loss(self) -> torch.Tensor:
        """Lateral inhibition: penalize neurons for producing similar outputs.

        Clamped at 0 — suppresses redundancy but doesn't reward anti-correlation.
        Analogous to lateral inhibition in neural circuits.
        """
        if self._last_expert_logits is None:
            return torch.tensor(0.0)

        B, K, m, V = self._last_expert_logits.shape
        flat = self._last_expert_logits.reshape(B, K, m * V)
        flat_norm = F.normalize(flat, dim=-1)  # (B, K, m*V)

        # Pairwise cosine similarity: (B, K, K)
        sim = torch.bmm(flat_norm, flat_norm.transpose(1, 2))

        # Upper triangle only (exclude self-similarity diagonal)
        mask = torch.triu(torch.ones(K, K, device=sim.device, dtype=torch.bool), diagonal=1)
        pairwise_sim = sim[:, mask]

        # Clamp: only penalize positive similarity, don't reward anti-correlation
        return pairwise_sim.clamp(min=0).mean()

    def load_balancing_loss(self) -> torch.Tensor:
        """Encourage all neurons to participate (optional).

        Penalizes when one neuron dominates all confidences.
        Uses entropy of mean confidence distribution.
        """
        if self._last_confidences is None:
            return torch.tensor(0.0)

        # Mean confidence weights across batch
        weights = F.softmax(self._last_confidences, dim=-1)  # (B, K)
        mean_weights = weights.mean(dim=0)  # (K,)

        # Negative entropy (lower = more uniform = better)
        # Max entropy = log(K), so normalize
        entropy = -(mean_weights * torch.log(mean_weights + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=entropy.device))
        return 1.0 - entropy / max_entropy  # 0 = uniform, 1 = collapsed

    def router_cls_loss(self, reg_labels: torch.Tensor) -> torch.Tensor:
        """Not used in population decoder — no router."""
        return torch.tensor(0.0, device=reg_labels.device)

    def get_neuron_confidences(self, memory: torch.Tensor,
                               tgt: torch.Tensor,
                               src_tokens: torch.Tensor | None = None):
        """Return confidence values for analysis.

        Returns:
            confidences: (B, K) raw confidence logits
            weights: (B, K) softmax-normalized weights
        """
        if self._last_confidences is not None:
            weights = F.softmax(self._last_confidences, dim=-1)
            return self._last_confidences, weights
        # If no cached values, run forward
        self.forward(tgt, memory, src_tokens=src_tokens)
        weights = F.softmax(self._last_confidences, dim=-1)
        return self._last_confidences, weights


class PopulationEditDecoder(nn.Module):
    """Population-coded edit transducer: K self-gating EditDecoder neurons.

    Combines population coding (v17a) with edit transduction (v19a).
    Each neuron predicts edit operations with self-determined confidence.
    Final output = confidence-weighted blend of neuron edit logits.
    """

    def __init__(self, char_vocab_size: int, d_model: int = 128,
                 num_experts: int = 4, num_layers: int = 1,
                 d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, lstm_hidden: int = 0,
                 lambda_diversity: float = 0.1,
                 confidence_tau: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.pad_idx = pad_idx
        self.char_vocab_size = char_vocab_size
        self.lambda_diversity = lambda_diversity
        self.confidence_tau = confidence_tau
        self.alpha_cls_router = 0.0
        self.routing_mode = "population_edit"
        self.lambda_balance = 0.0

        expert_dim = lstm_hidden if lstm_hidden > 0 else d_model

        self.neurons = nn.ModuleList([
            EditDecoder(
                char_vocab_size=char_vocab_size, d_model=d_model,
                num_layers=num_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx, lstm_hidden=lstm_hidden,
            )
            for _ in range(num_experts)
        ])

        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, 1),
            )
            for _ in range(num_experts)
        ])

        self._last_expert_logits = None
        self._last_confidences = None

    def _pool_encoder(self, memory, src_tokens):
        if src_tokens is not None:
            pad_mask = (src_tokens == self.pad_idx)
            real_mask = ~pad_mask
            lengths = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return (memory * real_mask.unsqueeze(-1).float()).sum(dim=1) / lengths
        return memory.mean(dim=1)

    def forward(self, edit_targets: torch.Tensor, memory: torch.Tensor,
                src_tokens: torch.Tensor,
                encoder_out: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced forward. Returns edit logits (B, T-1, edit_vocab)."""
        pooled = self._pool_encoder(memory, src_tokens)

        all_logits = []
        all_confidences = []

        for neuron, conf_head in zip(self.neurons, self.confidence_heads):
            logits_k = neuron(edit_targets, memory, src_tokens=src_tokens)
            conf_k = conf_head(pooled).squeeze(-1)
            all_logits.append(logits_k)
            all_confidences.append(conf_k)

        stacked = torch.stack(all_logits, dim=1)  # (B, K, T-1, edit_vocab)
        confidences = torch.stack(all_confidences, dim=1)  # (B, K)
        self._last_expert_logits = stacked
        self._last_confidences = confidences

        weights = F.softmax(confidences / self.confidence_tau, dim=-1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        logits = (stacked * weights).sum(dim=1)  # (B, T-1, edit_vocab)

        return logits

    @torch.no_grad()
    def greedy_decode(self, memory: torch.Tensor,
                      src_tokens: torch.Tensor,
                      max_len: int = 64) -> list[list[int]]:
        """Greedy decode: pick most confident neuron's output per sample."""
        pooled = self._pool_encoder(memory, src_tokens)

        confidences = []
        for conf_head in self.confidence_heads:
            conf = conf_head(pooled).squeeze(-1)
            confidences.append(conf)
        conf_stack = torch.stack(confidences, dim=1)  # (B, K)
        best_neuron = conf_stack.argmax(dim=1)  # (B,)

        B = memory.size(0)
        all_outputs = []
        for neuron in self.neurons:
            outputs_k = neuron.greedy_decode(memory, src_tokens, max_len=max_len)
            all_outputs.append(outputs_k)

        results = []
        for i in range(B):
            k = best_neuron[i].item()
            results.append(all_outputs[k][i])

        return results

    def diversity_loss(self) -> torch.Tensor:
        """Clamped lateral inhibition on edit logits."""
        if self._last_expert_logits is None:
            return torch.tensor(0.0)
        B, K, T, V = self._last_expert_logits.shape
        flat = self._last_expert_logits.reshape(B, K, T * V)
        flat_norm = F.normalize(flat, dim=-1)
        sim = torch.bmm(flat_norm, flat_norm.transpose(1, 2))
        mask = torch.triu(torch.ones(K, K, device=sim.device, dtype=torch.bool), diagonal=1)
        return sim[:, mask].clamp(min=0).mean()

    def load_balancing_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def router_cls_loss(self, reg_labels: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=reg_labels.device)
