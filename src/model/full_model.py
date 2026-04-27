"""End-to-end Slot Attention Transducer.

Pipeline: Encoder → SlotAttention → L0Drop → Decoder
Combines all components and computes the full training objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerCharEncoder, BiLSTMEncoder
from .slot_attention import SlotAttentionModule
from .l0drop import L0Drop, InputConditionalL0Drop, TopKDrop, GumbelSlotRouter
from .decoder import TransformerCharDecoder, LSTMDecoder
from .moe_decoder import MoEDecoder
from .population_decoder import PopulationDecoder, PopulationEditDecoder
from .edit_decoder import EditDecoder, EDIT_PAD
from .edit_labeler import EditLabeler
from .copy_decoder import MonotonicCopyDecoder


class SlotAttentionTransducer(nn.Module):
    """Full model: present-tense → past-tense character transduction via slot attention.

    Supports two L0 modes:
    - 'static': Original L0Drop with fixed per-slot gates (lambda_l0 weighting)
    - 'conditional': Input-conditional gates with Lagrangian constraint (target_l0)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 enc_layers: int = 3, dec_layers: int = 3, d_ff: int = 256,
                 num_slots: int = 8, slot_iters: int = 3, mlp_hidden: int = 256,
                 dropout: float = 0.1, pad_idx: int = 0,
                 lambda_l0: float = 0.01, alpha_recon: float = 0.0,
                 l0_beta: float = 0.66, l0_mode: str = "conditional",
                 target_l0: float = 2.0, lagrangian_lr: float = 0.01,
                 use_copy: bool = False, slot_nhead: int = 1,
                 use_slots: bool = True, decoder_type: str = "transformer",
                 encoder_type: str = "transformer",
                 alpha_cls: float = 0.0,
                 dec_bottleneck: int = 0,
                 lstm_hidden: int = 0,
                 num_experts: int = 4,
                 expert_hidden: int = 64,
                 routing_mode: str = "soft",
                 gumbel_tau: float = 1.0,
                 lambda_balance: float = 0.01,
                 alpha_cls_router: float = 0.0,
                 lambda_diversity: float = 0.0,
                 confidence_mode: str = "input",
                 neuron_dropout: float = 0.0,
                 confidence_tau: float = 1.0):
        super().__init__()
        self.pad_idx = pad_idx
        self.lambda_l0 = lambda_l0
        self.alpha_recon = alpha_recon
        self.alpha_cls = alpha_cls
        self.l0_mode = l0_mode
        self.use_copy = use_copy
        self.use_slots = use_slots

        if encoder_type == "bilstm":
            self.encoder = BiLSTMEncoder(
                vocab_size=vocab_size, d_model=d_model,
                num_layers=enc_layers, dropout=dropout, pad_idx=pad_idx,
            )
        else:
            self.encoder = TransformerCharEncoder(
                vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                num_layers=enc_layers, d_ff=d_ff, dropout=dropout, pad_idx=pad_idx,
            )

        if use_slots:
            self.slot_attention = SlotAttentionModule(
                d_input=d_model, d_slot=d_model, num_slots=num_slots,
                num_iterations=slot_iters, mlp_hidden=mlp_hidden,
                nhead=slot_nhead,
            )

            if l0_mode == "topk":
                self.l0drop = TopKDrop(
                    d_slot=d_model, num_slots=num_slots,
                    k=int(target_l0),
                )
            elif l0_mode == "gumbel":
                self.l0drop = GumbelSlotRouter(
                    d_slot=d_model, num_slots=num_slots,
                    k=int(target_l0),
                )
            elif l0_mode == "conditional":
                self.l0drop = InputConditionalL0Drop(
                    d_slot=d_model, num_slots=num_slots, beta=l0_beta,
                    target_l0=target_l0, lagrangian_lr=lagrangian_lr,
                )
            else:
                self.l0drop = L0Drop(num_slots=num_slots, beta=l0_beta)
        else:
            self.slot_attention = None
            self.l0drop = None

        if decoder_type == "mono_copy":
            self.decoder = MonotonicCopyDecoder(
                vocab_size=vocab_size, d_model=d_model,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx, lstm_hidden=lstm_hidden,
            )
            self.use_copy = True  # uses NLL loss (returns log-probs)
        elif decoder_type == "edit_labeler":
            self.decoder = EditLabeler(
                char_vocab_size=vocab_size, d_model=d_model,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx,
            )
        elif decoder_type == "population_edit":
            self.decoder = PopulationEditDecoder(
                char_vocab_size=vocab_size, d_model=d_model,
                num_experts=num_experts, num_layers=dec_layers,
                d_ff=d_ff, dropout=dropout, pad_idx=pad_idx,
                lstm_hidden=expert_hidden,
                lambda_diversity=lambda_diversity,
                confidence_tau=confidence_tau,
            )
            self.lambda_diversity = lambda_diversity
        elif decoder_type == "edit":
            self.decoder = EditDecoder(
                char_vocab_size=vocab_size, d_model=d_model,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx, lstm_hidden=lstm_hidden,
                scheduled_sampling=0.0,  # annealed by train.py
            )
        elif decoder_type == "population":
            self.decoder = PopulationDecoder(
                vocab_size=vocab_size, d_model=d_model,
                num_experts=num_experts, expert_hidden=expert_hidden,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx, use_copy=use_copy,
                dec_bottleneck=dec_bottleneck,
                lambda_balance=lambda_balance,
                lambda_diversity=lambda_diversity,
                confidence_mode=confidence_mode,
                neuron_dropout=neuron_dropout,
                confidence_tau=confidence_tau,
            )
            self.lambda_diversity = lambda_diversity
        elif decoder_type == "moe_lstm":
            self.decoder = MoEDecoder(
                vocab_size=vocab_size, d_model=d_model,
                num_experts=num_experts, expert_hidden=expert_hidden,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout,
                pad_idx=pad_idx, use_copy=use_copy,
                dec_bottleneck=dec_bottleneck,
                routing_mode=routing_mode, gumbel_tau=gumbel_tau,
                lambda_balance=lambda_balance,
                alpha_cls_router=alpha_cls_router,
            )
            self.lambda_diversity = lambda_diversity
        elif decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                vocab_size=vocab_size, d_model=d_model,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout, pad_idx=pad_idx,
                use_copy=use_copy, dec_bottleneck=dec_bottleneck,
                lstm_hidden=lstm_hidden,
            )
        else:
            self.decoder = TransformerCharDecoder(
                vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout, pad_idx=pad_idx,
                use_copy=use_copy,
            )

        # Optional reconstruction decoder (multi-task variant)
        self.recon_decoder = None
        if alpha_recon > 0:
            self.recon_decoder = TransformerCharDecoder(
                vocab_size=vocab_size, d_model=d_model, nhead=nhead,
                num_layers=dec_layers, d_ff=d_ff, dropout=dropout, pad_idx=pad_idx,
            )

        # Optional verb-class classifier on slot representations
        self.slot_classifier = None
        if alpha_cls > 0 and use_slots:
            # Pool slots → predict regular(0) vs irregular(1)
            self.slot_classifier = nn.Sequential(
                nn.Linear(d_model * num_slots, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
            )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_for_recon: torch.Tensor | None = None,
                reg_labels: torch.Tensor | None = None,
                edit_targets: torch.Tensor | None = None,
                edit_labels: torch.Tensor | None = None,
                suffix_targets: torch.Tensor | None = None):
        """
        Args:
            src: (batch, n) source character indices
            tgt: (batch, m) target character indices (with <sos> prefix)
            src_for_recon: (batch, m') source with <sos> prefix for reconstruction
            reg_labels: (batch,) 0=regular, 1=irregular (for auxiliary classifier)
        Returns:
            dict with 'loss', 'logits', 'l0_loss', and optionally 'recon_loss', 'cls_loss'
        """
        # Encode
        H = self.encoder(src)                    # (B, n, d)

        if self.use_slots:
            # Slot Attention → L0Drop
            slots = self.slot_attention(H)       # (B, K, d)
            memory = self.l0drop(slots)          # (B, K, d) sparse
        else:
            # No-slot baseline: decoder cross-attends directly to encoder output
            memory = H                           # (B, n, d)

        # Decode: teacher-forced
        if isinstance(self.decoder, EditLabeler) and edit_labels is not None:
            # Per-position edit labeler
            labeler_result = self.decoder(memory, src, edit_labels=edit_labels,
                                          suffix_targets=suffix_targets)
            loss_transduce = labeler_result["loss"]
            logits = labeler_result["edit_logits"]
        elif isinstance(self.decoder, (EditDecoder, PopulationEditDecoder)) and edit_targets is not None:
            # Edit transducer: predict edit operations
            logits = self.decoder(edit_targets, memory, src_tokens=src)
            # Target is edit_targets shifted by 1 (predict next edit from current)
            edit_target = edit_targets[:, 1:]
            loss_transduce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                edit_target.reshape(-1),
                ignore_index=EDIT_PAD,
            )
        else:
            # Standard character-level decoding
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]
            logits = self.decoder(dec_input, memory,
                                  encoder_out=H if self.use_copy else None,
                                  src_tokens=src)
            if self.use_copy:
                loss_transduce = F.nll_loss(
                    logits.reshape(-1, logits.size(-1)),
                    dec_target.reshape(-1),
                    ignore_index=self.pad_idx,
                )
            else:
                loss_transduce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    dec_target.reshape(-1),
                    ignore_index=self.pad_idx,
                )

        # L0 regularization
        if self.use_slots:
            loss_l0 = self.l0drop.l0_loss()
            if self.l0_mode == "conditional":
                total_loss = loss_transduce + loss_l0
            else:
                total_loss = loss_transduce + self.lambda_l0 * loss_l0
        else:
            loss_l0 = torch.tensor(0.0, device=src.device)
            total_loss = loss_transduce

        # MoE load balancing loss
        loss_balance = torch.tensor(0.0, device=src.device)
        if hasattr(self.decoder, 'load_balancing_loss'):
            loss_balance = self.decoder.load_balancing_loss()
            total_loss = total_loss + self.decoder.lambda_balance * loss_balance

        # MoE router auxiliary classification loss
        loss_cls_router = torch.tensor(0.0, device=src.device)
        if (hasattr(self.decoder, 'router_cls_loss')
                and self.decoder.alpha_cls_router > 0
                and reg_labels is not None):
            loss_cls_router = self.decoder.router_cls_loss(reg_labels)
            total_loss = total_loss + self.decoder.alpha_cls_router * loss_cls_router

        # MoE expert diversity loss (lateral inhibition)
        loss_diversity = torch.tensor(0.0, device=src.device)
        if (hasattr(self, 'lambda_diversity') and self.lambda_diversity > 0
                and hasattr(self.decoder, 'diversity_loss')):
            loss_diversity = self.decoder.diversity_loss()
            total_loss = total_loss + self.lambda_diversity * loss_diversity

        result = {
            "loss": total_loss,
            "loss_transduce": loss_transduce,
            "loss_l0": loss_l0,
            "loss_balance": loss_balance,
            "loss_cls_router": loss_cls_router,
            "loss_diversity": loss_diversity,
            "logits": logits,
        }

        # Optional reconstruction loss
        if self.recon_decoder is not None and src_for_recon is not None:
            recon_input = src_for_recon[:, :-1]
            recon_target = src_for_recon[:, 1:]
            recon_logits = self.recon_decoder(recon_input, memory)
            loss_recon = F.cross_entropy(
                recon_logits.reshape(-1, recon_logits.size(-1)),
                recon_target.reshape(-1),
                ignore_index=self.pad_idx,
            )
            result["loss"] = result["loss"] + self.alpha_recon * loss_recon
            result["loss_recon"] = loss_recon

        # Optional verb-class auxiliary loss on slot representations
        if self.slot_classifier is not None and reg_labels is not None and self.use_slots:
            # Flatten slots: (B, K, d) → (B, K*d)
            slot_flat = slots.reshape(slots.size(0), -1)
            cls_logits = self.slot_classifier(slot_flat)  # (B, 2)
            loss_cls = F.cross_entropy(cls_logits, reg_labels)
            result["loss"] = result["loss"] + self.alpha_cls * loss_cls
            result["loss_cls"] = loss_cls

        return result

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 32,
                      sos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        """Greedy autoregressive decoding for inference."""
        B = src.size(0)
        H = self.encoder(src)

        if self.use_slots:
            slots = self.slot_attention(H)
            memory = self.l0drop(slots)
        else:
            memory = H

        if isinstance(self.decoder, MonotonicCopyDecoder):
            return self.decoder.greedy_decode_mono(
                memory, src, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)

        if isinstance(self.decoder, EditLabeler):
            output_ids = self.decoder.greedy_decode(memory, src)
            tensors = []
            for ids in output_ids:
                t = [sos_idx] + ids + [eos_idx]
                tensors.append(torch.tensor(t, dtype=torch.long, device=src.device))
            max_out_len = max(len(t) for t in tensors)
            padded = torch.zeros(B, max_out_len, dtype=torch.long, device=src.device)
            for i, t in enumerate(tensors):
                padded[i, :len(t)] = t
            return padded

        if isinstance(self.decoder, (EditDecoder, PopulationEditDecoder)):
            # Edit transducer: returns list of char ID lists
            output_ids = self.decoder.greedy_decode(memory, src)
            # Convert to padded tensor with SOS/EOS for compatibility
            tensors = []
            for ids in output_ids:
                t = [sos_idx] + ids + [eos_idx]
                tensors.append(torch.tensor(t, dtype=torch.long, device=src.device))
            # Pad to same length
            max_out_len = max(len(t) for t in tensors)
            padded = torch.zeros(B, max_out_len, dtype=torch.long, device=src.device)
            for i, t in enumerate(tensors):
                padded[i, :len(t)] = t
            return padded

        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            logits = self.decoder(generated, memory,
                                  encoder_out=H if self.use_copy else None,
                                  src_tokens=src)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_idx).all():
                break

        return generated
