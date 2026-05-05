"""Hard Monotonic Neural Transducer decoder.

LSTM that conditions on:
    [prev_action_emb; src_char_at_pointer; encoder_state_at_pointer; attn_context]
and predicts the next action over the HMNT action vocab.

Pointer is explicit: STEP advances by 1, WRITE leaves it unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import BahdanauAttention
from .transducer_actions import (
    ACTION_PAD, ACTION_STEP, ACTION_END, WRITE_OFFSET,
    action_vocab_size, is_write, write_char, oracle_next_action,
)


class TransducerDecoder(nn.Module):
    """HMNT decoder with explicit source pointer.

    Args:
        char_vocab_size: number of character types (for src_char embedding & WRITE actions).
        d_model: hidden / embedding size.
        num_layers: LSTM layers.
        lstm_hidden: LSTM hidden size; 0 = use d_model.
        dropout: dropout rate.
        pad_idx: char vocab pad index (for src_char embedding).
    """

    def __init__(self, char_vocab_size: int, d_model: int = 128,
                 num_layers: int = 1, d_ff: int = 256, dropout: float = 0.1,
                 pad_idx: int = 0, lstm_hidden: int = 0,
                 use_retrieval: bool = False):
        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.lstm_dim = lstm_hidden if lstm_hidden > 0 else d_model
        self.use_retrieval = use_retrieval

        self.action_vocab_size = action_vocab_size(char_vocab_size)
        self.action_embedding = nn.Embedding(self.action_vocab_size, d_model,
                                             padding_idx=ACTION_PAD)
        self.char_embedding = nn.Embedding(char_vocab_size, d_model,
                                           padding_idx=pad_idx)

        self.attention = BahdanauAttention(self.lstm_dim, d_model, attn_dim=d_model)

        # When retrieval is enabled, a second Bahdanau head attends over the
        # flattened encoded retrieved-target memory. Its context is appended
        # to the LSTM input AND to the output projection input.
        retr_dim = d_model if use_retrieval else 0
        if use_retrieval:
            self.retrieval_attention = BahdanauAttention(self.lstm_dim, d_model,
                                                         attn_dim=d_model)

        # LSTM input: [prev_action_emb; src_char_at_ptr_emb; enc_state_at_ptr;
        #             src_attn_context; (retr_attn_context if use_retrieval)]
        self.lstm = nn.LSTM(
            input_size=d_model * 4 + retr_dim,
            hidden_size=self.lstm_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Output: [lstm_out; enc_state_at_ptr; src_attn_context;
        #         (retr_attn_context if use_retrieval)] -> action_vocab
        self.output_proj = nn.Linear(self.lstm_dim + d_model * 2 + retr_dim,
                                     self.action_vocab_size)

    def _step(self, prev_action: torch.Tensor, ptr: torch.Tensor,
              src_tokens: torch.Tensor, encoder_out: torch.Tensor,
              src_pad_mask: torch.Tensor,
              h: torch.Tensor, c: torch.Tensor,
              retrieval_memory: torch.Tensor | None = None,
              retrieval_pad_mask: torch.Tensor | None = None):
        """Single decoding step.

        Args:
            prev_action: (B,) previous action ids
            ptr: (B,) source pointer per sample
            src_tokens: (B, N) source char ids
            encoder_out: (B, N, d) encoder hidden states
            src_pad_mask: (B, N) True where padded
            h, c: LSTM state (num_layers, B, lstm_dim)
            retrieval_memory: (B, M, d) flattened retrieved-target encodings, or None
            retrieval_pad_mask: (B, M) True where padded, or None
        Returns:
            logits (B, action_vocab), new (h, c)
        """
        B, N = src_tokens.shape
        device = src_tokens.device
        batch_idx = torch.arange(B, device=device)
        clamped_ptr = ptr.clamp(max=N - 1)

        action_emb = self.action_embedding(prev_action)                  # (B, d)
        src_char_emb = self.char_embedding(src_tokens[batch_idx, clamped_ptr])  # (B, d)
        enc_at_ptr = encoder_out[batch_idx, clamped_ptr]                 # (B, d)

        attn_query = h[-1]                                                # (B, lstm_dim)
        context, _ = self.attention(attn_query, encoder_out, mask=src_pad_mask)  # (B, d)

        parts = [action_emb, src_char_emb, enc_at_ptr, context]
        out_parts = [enc_at_ptr, context]
        if self.use_retrieval:
            assert retrieval_memory is not None
            retr_ctx, _ = self.retrieval_attention(
                attn_query, retrieval_memory, mask=retrieval_pad_mask)   # (B, d)
            parts.append(retr_ctx)
            out_parts.append(retr_ctx)

        lstm_in = torch.cat(parts, dim=-1).unsqueeze(1)                  # (B, 1, 4d or 5d)
        lstm_out, (h, c) = self.lstm(lstm_in, (h, c))
        lstm_out = lstm_out.squeeze(1)                                   # (B, lstm_dim)

        out_in = torch.cat([lstm_out] + out_parts, dim=-1)
        logits = self.output_proj(self.dropout(out_in))                  # (B, A)
        return logits, h, c

    def forward(self, action_targets: torch.Tensor,
                memory: torch.Tensor, src_tokens: torch.Tensor,
                encoder_out: torch.Tensor | None = None,
                use_dagger: bool = False, beta: float = 0.0,
                src_no_eos: list[list[int]] | None = None,
                tgt_no_special: list[list[int]] | None = None,
                retrieval_memory: torch.Tensor | None = None,
                retrieval_pad_mask: torch.Tensor | None = None):
        """Teacher-forced (or DAgger β-mixed) forward pass.

        Args:
            action_targets: (B, T) precomputed oracle action sequence (with END).
                Used as the per-step input action under teacher forcing.
            memory: (B, N, d) decoder cross-attention memory (= encoder_out for HMNT;
                slot-attention path is not supported here).
            src_tokens: (B, N) source char ids (with EOS appended).
            encoder_out: passed through; unused (memory is used).
            use_dagger: if True, with prob `beta` substitute the model's own argmax
                action for the previous-action input AND for pointer-state updates;
                always train against the oracle re-queried at the (possibly
                divergent) state.
            src_no_eos / tgt_no_special: per-sample raw lists (no SOS/EOS) used by
                the oracle when DAgger is on. Required iff use_dagger.
        Returns:
            dict with:
                'logits': (B, T-1, A)
                'targets': (B, T-1) — oracle target actions (re-queried when DAgger)
                'mask': (B, T-1) bool — True at valid positions to include in loss
        """
        B, T = action_targets.shape
        device = action_targets.device
        N = src_tokens.size(1)
        src_pad_mask = (src_tokens == self.pad_idx)

        h = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)

        ptr = torch.zeros(B, dtype=torch.long, device=device)
        out_len = torch.zeros(B, dtype=torch.long, device=device)
        valid = torch.ones(B, dtype=torch.bool, device=device)
        # `diverged[i]` is True once sample i has consumed any model-chosen
        # action (so its (ptr, out_len) state may no longer be reachable from
        # the precomputed script). Only diverged samples need oracle re-query;
        # on-script samples reuse `action_targets[:, t+1]` exactly so the
        # training distribution doesn't drift on tie-broken alignments.
        diverged = torch.zeros(B, dtype=torch.bool, device=device)

        prev_action = action_targets[:, 0]  # bootstrap with the first oracle action
        # Note: the *target* at position 0 is action_targets[:, 1] (predict next).

        all_logits = []
        all_targets = []
        all_mask = []

        # Pre-extract Python lists for oracle re-querying (cheap, B small)
        if use_dagger:
            assert src_no_eos is not None and tgt_no_special is not None

        for t in range(T - 1):
            logits, h, c = self._step(prev_action, ptr, src_tokens, memory,
                                      src_pad_mask, h, c,
                                      retrieval_memory=retrieval_memory,
                                      retrieval_pad_mask=retrieval_pad_mask)
            all_logits.append(logits)

            oracle_target = action_targets[:, t + 1].clone()  # precomputed by default
            mask_t = valid.clone()

            if use_dagger:
                # Only re-query for samples that have actually diverged from
                # the precomputed trajectory. On-script samples keep the
                # precomputed target — no tie-break drift.
                if diverged.any():
                    for i in range(B):
                        if not (valid[i] and diverged[i]):
                            continue
                        oracle_target[i] = oracle_next_action(
                            src_no_eos[i], tgt_no_special[i],
                            ptr[i].item(), out_len[i].item(),
                        )

                # With prob beta, use the model's predicted action as the next
                # input AND for state updates (β-mixing).
                use_own = (torch.rand(B, device=device) < beta) & valid
                model_action = logits.argmax(dim=-1)
                next_action = torch.where(use_own, model_action, oracle_target)
                # A sample becomes diverged only when the model's action
                # *actually differs* from the oracle target. If `use_own`
                # fires but the model picks the same action, no real
                # divergence occurred — keep using precomputed targets to
                # avoid the suffix-vs-full-alignment tie-break drift that
                # would otherwise contaminate ~half of "diverged" samples.
                just_diverged = use_own & (model_action != oracle_target)
                diverged = diverged | just_diverged

                # Update state based on next_action
                is_step = (next_action == ACTION_STEP)
                is_end = (next_action == ACTION_END)
                is_w = (next_action >= WRITE_OFFSET)
                ptr = ptr + (is_step & valid).long()
                ptr = ptr.clamp(max=N - 1)

                # WRITE: only count toward output_len if it matches the expected
                # tgt char at the current output_len position. Otherwise mark
                # rollout invalid (output is now off-trajectory; oracle would be
                # giving advice for a state we can no longer reach).
                if is_w.any():
                    write_chars = next_action - WRITE_OFFSET
                    for i in range(B):
                        if not (valid[i] and is_w[i]):
                            continue
                        tgt_i = tgt_no_special[i]
                        if (out_len[i].item() < len(tgt_i)
                                and write_chars[i].item() == tgt_i[out_len[i].item()]):
                            out_len[i] = out_len[i] + 1
                        else:
                            valid[i] = False

                # END terminates the rollout for that sample
                valid = valid & ~is_end

                prev_action = next_action
            else:
                # Pure teacher forcing: state and prev_action both follow the oracle
                action = oracle_target  # = action_targets[:, t+1]
                prev_action = action
                # Update pointer along oracle trajectory so encoder lookups stay aligned
                is_step = (action == ACTION_STEP)
                ptr = ptr + (is_step & valid).long()
                ptr = ptr.clamp(max=N - 1)
                # Also drop positions past END from the loss
                is_end = (action == ACTION_END)
                # Loss valid at this step (we still predict END here);
                # subsequent positions get masked.
                # Update validity AFTER recording mask for this step.
                # mask_t already captured valid pre-end.
                valid = valid & ~is_end

            all_targets.append(oracle_target)
            all_mask.append(mask_t)

        logits_seq = torch.stack(all_logits, dim=1)            # (B, T-1, A)
        targets_seq = torch.stack(all_targets, dim=1)          # (B, T-1)
        mask_seq = torch.stack(all_mask, dim=1)                # (B, T-1)
        return {"logits": logits_seq, "targets": targets_seq, "mask": mask_seq}

    @torch.no_grad()
    def greedy_decode(self, memory: torch.Tensor, src_tokens: torch.Tensor,
                      max_actions: int = 64,
                      retrieval_memory: torch.Tensor | None = None,
                      retrieval_pad_mask: torch.Tensor | None = None,
                      ) -> list[list[int]]:
        """Greedy action decoding; returns one list of output char ids per sample."""
        B, N = src_tokens.shape
        device = memory.device
        src_pad_mask = (src_tokens == self.pad_idx)

        h = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        c = torch.zeros(self.num_layers, B, self.lstm_dim, device=device)
        ptr = torch.zeros(B, dtype=torch.long, device=device)

        # Bootstrap with a sentinel "first action" — use STEP as a neutral start
        # (the model never sees a meaningful PAD during training, but STEP appears
        # frequently and won't shift weights toward a bad init).
        prev_action = torch.full((B,), ACTION_STEP, dtype=torch.long, device=device)

        outputs: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B

        for _ in range(max_actions):
            logits, h, c = self._step(prev_action, ptr, src_tokens, memory,
                                      src_pad_mask, h, c,
                                      retrieval_memory=retrieval_memory,
                                      retrieval_pad_mask=retrieval_pad_mask)
            action = logits.argmax(dim=-1)                                # (B,)

            for i in range(B):
                if done[i]:
                    continue
                a = action[i].item()
                if a == ACTION_END:
                    done[i] = True
                elif a == ACTION_STEP:
                    if ptr[i].item() < N - 1:
                        ptr[i] = ptr[i] + 1
                elif a >= WRITE_OFFSET:
                    outputs[i].append(a - WRITE_OFFSET)

            if all(done):
                break
            prev_action = action

        return outputs
