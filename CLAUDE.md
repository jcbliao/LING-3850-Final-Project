# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Instructions

- **Always update `progress.md` and `CLAUDE.md` proactively** after every code change, experiment submission, or result analysis — without being asked.
- **Never run training or inference on the login node.** Always submit via `sbatch`. The login node is only for editing, git, and job submission.
- **Always use GPU** for model inference/evaluation jobs.
- **Job scheduling**: Submit all jobs to `gpu` partition first. After 20 seconds, check with `squeue` — if any are still queued (ST=PD), cancel one and resubmit it to `gpu_devel`. Only one job can run on `gpu_devel` at a time.

## Project Overview

"Applying Slot Attention to Past Tense" — a LING 3850 research project (Alan Xie, Henry Zhang, Jacob Liao). Applies Slot Attention to English past-tense morphology as a character-level transduction task (present → past verb forms, e.g. "play" → "played", "go" → "went"). Tests whether Slot Attention learns abstract past-tense rules better than prior RNN/Transformer approaches.

## Framework & Environment

- **PyTorch** (>=2.0) for all model code
- Runs on **Yale HPC** (GPFS filesystem) with SLURM for GPU jobs
- Python dependencies in `requirements.txt`
- **IMPORTANT**: Never run training or inference on the login node. Always submit via `sbatch`. The login node is only for editing, git, and job submission.

## Commands

```bash
# Train with default config
cd src && python train.py --config ../configs/default.yaml

# Train with custom config
cd src && python train.py --config ../configs/my_experiment.yaml

# Evaluate a checkpoint
cd src && python evaluate.py --checkpoint ../checkpoints/best_model.pt \
    --data_path ../data/RevisitPinkerAndPrince/experiment_1/english_merged.txt \
    --wug_dir ../data/RevisitPinkerAndPrince/experiment_1_wugs/

# Submit to SLURM
sbatch scripts/run_experiment.sh                    # uses default config
sbatch scripts/run_experiment.sh configs/sweep.yaml # custom config
```

## Dataset

**Source**: Kirov & Cotterell (2018) `RevisitPinkerAndPrince` repo, cloned to `data/RevisitPinkerAndPrince/`.

- `experiment_1/english_merged.txt` — 4,039 verb pairs (orthographic + phonological forms, reg/irreg labels). Tab-separated: `orth_present  orth_past  phon_present  phon_past  reg|irreg`
- `experiment_1_wugs/` — 58 Albright & Hayes nonce verbs for generalization evaluation. Space-separated phonological chars with stress markers (È). Files: `src.txt`, `tgt_regular.txt`, `tgt_irregular.txt`

The data uses **phonological transcription** (DISC encoding), not orthographic characters. The vocab module handles both modes, controlled by `use_phonological` in the config.

Split: 80% train / 10% val / 10% test (deterministic via seed).

## Architecture (PyTorch modules)

Pipeline: `Encoder → [SlotAttentionModule → L0Drop] → Decoder`

| Stage | Module | File | Input → Output |
|---|---|---|---|
| 1a | `TransformerCharEncoder` | `src/model/encoder.py` | char indices (B,n) → H (B,n,d) — default |
| 1b | `BiLSTMEncoder` | `src/model/encoder.py` | char indices (B,n) → H (B,n,d) — K&C style |
| 2 | `SlotAttentionModule` | `src/model/slot_attention.py` | H (B,n,d) → S (B,K,d) — optional (`use_slots`) |
| 3 | `L0Drop` | `src/model/l0drop.py` | S (B,K,d) → M' (B,K,d) sparse — optional |
| 4a | `TransformerCharDecoder` | `src/model/decoder.py` | (tgt, M') → logits — with optional copy mechanism |
| 4b | `LSTMDecoder` | `src/model/decoder.py` | (tgt, M') → logits — Bahdanau attention, no copy needed |
| 4c | `MoEDecoder` | `src/model/moe_decoder.py` | (tgt, H) → logits — K expert LSTMDecoders with routing |
| All | `SlotAttentionTransducer` | `src/model/full_model.py` | Combines all + loss computation |

Encoder selected via `encoder_type: "transformer"` (default) or `"bilstm"`. Decoder selected via `decoder_type: "transformer"` (default), `"lstm"`, or `"moe_lstm"`. LSTM decoder uses Bahdanau attention, generates every character (no copy mechanism), naturally learns monotonic alignment.

### MoE Decoder (Mixture of Expert Decoders)
- **`decoder_type: "moe_lstm"`** with **`use_slots: false`** — replaces slot attention entirely
- `ExpertRouter`: mean-pools encoder output → MLP(d→d/2→K) → expert weights
- K independent `LSTMDecoder` experts, each with its own Bahdanau attention and parameters
- Routing modes: `"soft"` (weighted blend), `"hard"` (argmax + straight-through), `"gumbel"` (Gumbel-softmax, temperature-annealed)
- Load balancing loss (Switch Transformer): `K * sum(f_k * P_k)`, weighted by `lambda_balance`
- Sparse execution at eval for hard/Gumbel (only selected expert runs per sample)
- Config: `num_experts`, `expert_hidden`, `routing_mode`, `gumbel_tau`, `lambda_balance`
- Inductive bias: experts compete to be the *transformation rule*, not to explain input *positions*

### Slot Attention specifics
- Each slot has a **learnable mean** (μₖ) with **fixed init variance** (σ=0.1)
- Softmax is over the **slot dimension** (slots compete to explain each input position)
- T iterations of: attention → GRU update → MLP residual (with LayerNorm)
- **Multi-head mode** (`slot_nhead` config, default 1): splits features into H independent subspaces. Slots compete per-head, allowing different position attention in different feature groups. Addresses the position-vs-feature competition issue for morphology.

### Copy Mechanism (decoder)
- **`use_copy: true`** enables pointer/copy mechanism (See et al. 2017)
- At each step: `p_gen * vocab_dist + (1-p_gen) * copy_dist`
- Copy attention uses **separate Q/K projections** (`copy_query_proj` for decoder, `copy_key_proj` for encoder) to bridge the representation space gap between slot-conditioned decoder states and raw encoder outputs
- `p_gen` bias initialized to **-2.0** (sigmoid ≈ 0.12), biasing toward copying
- Critical for morphology: ~90% of output chars are copied from input
- When enabled, decoder returns log-probs; loss uses NLL instead of cross-entropy

### No-slot baseline mode
- **`use_slots: false`** skips slot attention and L0Drop entirely
- Decoder cross-attends directly to encoder output H (B, N, d_model)
- Standard encoder-decoder with copy — tests whether slot attention helps or hurts

### L0Drop specifics (two modes)

**`l0_mode: conditional`** (default, recommended) — `InputConditionalL0Drop`:
- Gate values computed from slot contents: `log_alpha_k = MLP(slot_k)`
- Different inputs activate different slot subsets (variable capacity)
- Uses **Lagrangian constraint**: loss = `lambda * (E[L0] - target_l0) + (E[L0] - target_l0)²`
- Lambda updated automatically via dual gradient ascent — no manual balancing needed
- Config: `target_l0` (desired active slot count), `lagrangian_lr` (multiplier update rate)

**`l0_mode: static`** (legacy) — `L0Drop`:
- Static per-slot `log_alpha` — same gates for every input
- Uses fixed `lambda_l0` weighting — prone to scale mismatch with transduction loss
- Effectively equivalent to training with fewer slots

### Training objective
```
L = L_transduce + λ · L_L0                        # standard
L = L_transduce + α · L_recon + λ · L_L0           # multi-task variant
```
- `L_transduce`: cross-entropy on decoder output (teacher-forced), ignoring pad
- `L_L0`: sum of gate-open probabilities (encourages slot pruning)
- `L_recon`: optional present→present reconstruction via separate decoder sharing same slot memory

## Config

YAML files in `configs/`. Key hyperparameters:

| Param | Default | Best (v3_bigger) | v4 (latest) | Role |
|---|---|---|---|---|
| `num_slots` | 8 | 4 | 4 | Max slot count K |
| `slot_iters` | 3 | 3 | 3 | Slot attention iterations T |
| `d_model` | 128 | 128 | 128 | Hidden dimension |
| `enc_layers` | 3 | 2 | 2 | Encoder depth |
| `dec_layers` | 3 | 2 | 2 | Decoder depth |
| `d_ff` | 256 | 256 | 256 | FFN hidden dim |
| `dropout` | 0.1 | 0.3 | 0.3 | Dropout rate |
| `l0_mode` | conditional | conditional | conditional | L0 gate mode |
| `target_l0` | 2.0 | 3.0 | 3.0 | Target active slot count |
| `alpha_recon` | 0.0 | 0.0 | 0.0 | Reconstruction weight (0 = disabled) |
| `use_copy` | false | false | **true** | Copy/pointer mechanism |
| `slot_nhead` | 1 | 1 | **4** | Multi-head slot attention heads |
| `l0_beta` | 0.66 | 0.66 | 0.66 | Hard-concrete temperature |
| `patience` | 0 | 25 | 25 | Early stopping patience (0 = disabled) |

## Hyperparameter Sweep

### Parameters to sweep

| Parameter | Values | Why |
|---|---|---|
| `num_slots` | 4, 8, 16 | Core question: how many latent components does past-tense need? |
| `lambda_l0` | 0.001, 0.01, 0.1 | Controls how aggressively slots get pruned |
| `alpha_recon` | 0.0, 0.5, 1.0 | Whether multi-task reconstruction helps |

Fixed for sweep: `d_model=128`, `slot_iters=3`. Total: **27 runs**, each ~10-15 min on A100.

### Running sweeps

```bash
# Launch all 27 sweep jobs
python scripts/sweep.py

# Dry run (preview without submitting)
python scripts/sweep.py --dry-run
```

Results land in `results/sweep_<name>/` with loss curves and `history.json` per run.

Note: Training regime experiments (balanced, irregular_first) are implemented in `src/data/dataset.py` via `apply_training_regime()` but are **not planned** for this project. Focus is on the hyperparameter sweep with natural data distribution only.

## Expected Training Dynamics

### Loss curves

- **`train_loss` / `val_loss`** (total loss): Should decrease steadily. Expect rapid drop in the first 10-20 epochs, then gradual improvement. Initial value ~3.5-4.0 (random predictions over 43-char vocab: −log(1/43) ≈ 3.76). A well-trained model should reach <0.5. If val_loss plateaus while train_loss keeps dropping, the model is overfitting.
- **`loss_transduce`** (NLL): Dominates total loss. Tracks how well the decoder predicts the correct next character. Should follow the same trajectory as total loss. Near zero means the model is producing correct past-tense forms with high confidence.
- **`loss_l0`** (expected active slots): Starts near `num_slots` (all gates open, e.g. ~8.0 for K=8). With `lambda_l0 > 0`, this should gradually decrease as the model learns to prune unnecessary slots. The final value indicates how many slots the model actually uses — this is a key result. If L0 stays at `num_slots` throughout, `lambda_l0` is too low. If it drops to ~1 very early, `lambda_l0` is too high and the bottleneck is too aggressive.

### Initial run results (default config, 100 epochs)

| Metric | Epoch 1 | Epoch 20 | Epoch 36 (best) | Epoch 100 |
|---|---|---|---|---|
| train_loss | 2.43 | 0.47 | 0.35 | 0.15 |
| val_loss | 1.87 | 0.84 | **0.75** | 0.78 |
| L0 | 6.64 | 6.65 | 6.66 | 6.65 |
| test accuracy | — | — | 8.6% (reg 8.6%, irreg 0%) | 14.3% |

**Key problems identified:**
1. **Severe overfitting**: train/val gap of 0.62 by epoch 100. Val loss plateaued at ~0.75 around epoch 36; everything after was wasted.
2. **L0 is frozen**: Stayed at ~6.65/8.0 throughout. `lambda_l0=0.01` is far too weak — the L0 penalty (~0.07) is negligible compared to transduction loss (~0.15-2.4). The sparsity mechanism is effectively disabled.
3. **Very low accuracy**: 8.6% exact-match on test (best checkpoint). 0% on irregulars. 0% on nonce verbs. The model memorizes some regular training forms but doesn't generalize.
4. **No slot specialization**: With all slots always open and no sparsity pressure, slots have no incentive to specialize. The bottleneck is not functioning as designed.

### What healthy training should look like

| Metric | Epoch 1 | Epoch 20 | Epoch 50 | Epoch 100 |
|---|---|---|---|---|
| train_loss | ~2.0-2.5 | ~0.5-1.0 | ~0.3-0.5 | <0.3 |
| val_loss | ~2.0-2.5 | ~0.5-1.0 | ~0.3-0.6 | ~0.3-0.5 |
| L0 | ~K (all open) | starting to drop | dropping toward final value | stabilized (e.g. 3-5 for K=8) |
| test accuracy | 0% | 30-60% | 70-90% | 85-95%+ |

### Warning signs

- **Val loss plateaus while train loss keeps dropping** (OBSERVED): Overfitting. Need stronger regularization, less model capacity, or auxiliary objectives.
- **L0 never decreases from K** (OBSERVED): lambda_l0 too low relative to transduction loss. Increase by 10-100x or use a schedule.
- **L0 collapses to ~1 early**: lambda_l0 too high. The model is forced through too narrow a bottleneck and can't represent the input.
- **0% irregular and nonce accuracy** (OBSERVED): Model has not learned any abstract rule — just memorizing frequent regular patterns.
- **Character scrambling in predictions** (OBSERVED): Copy attention is non-monotonic — attends to wrong source positions. Root cause: Transformer decoder has no inherent monotonic alignment bias (unlike LSTM decoders). With 3,231 training examples, can't learn alignment from data alone. Fix: add positional bias to copy attention.

### Comparison to prior work

| Approach | Regular | Irregular | Key mechanism |
|---|---|---|---|
| Kirov & Cotterell 2018 (biLSTM) | ~95% | ~85-90% | Bahdanau attention (naturally monotonic) |
| Ma & Gao 2022 (Transformer) | ~90-97% | ~80-90% | Standard Transformer seq2seq |
| Our best (v5, no slots) | 32.4% | 0% | Transformer + copy (no alignment bias) |

**Fixed by monotonic alignment bias** (v6): Gaussian positional bias `bias(t,n) = -alpha*(t-n)²` added to copy attention. `alpha` is learnable (init 1.0). No length scaling needed for morphology (output ≈ input + suffix).

| Approach | Regular | Irregular | Balanced Acc | Key mechanism |
|---|---|---|---|---|
| Kirov & Cotterell 2018 (biLSTM) | ~95% | ~85-90% | ~90% | biLSTM+LSTM, Bahdanau attention |
| **Our v6 (no slots, monotonic copy)** | **96.4%** | **15.8%** | **56.1%** | Transformer + monotonic copy |
| **Our v10a (biLSTM+LSTM, no slots)** | **99.5%** | **5.3%** | **52.4%** | biLSTM+LSTM, Bahdanau attention |
| Our v6 (with slots, monotonic copy) | 95.6% | 5.3% | 50.4% | Slots still don't help |

**Balanced accuracy** = `(regular_acc + irregular_acc) / 2`. This is the primary evaluation metric — irregulars are 50% of the problem even though they're only 19/405 test examples. Slot attention provides no benefit on any architecture.

### Wug evaluation fix

Wug verbs used a **different phonological encoding** (CELEXmod) than the training data (DISC). 10 characters were OOV, including stress marker `È` in every word. Fixed in `src/data/dataset.py` with `_wug_to_disc()` converter: strips stress markers, maps vowels (`Ã→V`, `Q→&`), expands diphthongs (`Y→aI`, `W→aU`, `o→@U`, `Õ→3:`, `C→tS`, `J→dZ`).

After fix: **51/90 wug regular matches (56.7%)**, 2/90 irregular. The model has learned the regular past-tense rule and generalizes to novel verbs.

### Best results summary

| Run | Architecture | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v16e** | **biLSTM + 4 MoE (gumbel+diversity)** | **95.1%** | **47.4%** | **71.2%** | **38/90** | **12/90** |
| v16c | biLSTM + 4 MoE (gumbel+guided) | 97.4% | 42.1% | 69.8% | 48/90 | 9/90 |
| v16a | biLSTM + 4 MoE (soft) | 95.3% | 31.6% | 63.5% | 40/90 | 14/90 |
| v13b | biLSTM+LSTM+slots (bottleneck+os) | 85.5% | 31.6% | 58.5% | 41/90 | 10/90 |
| v6 no-slots | Transformer+copy+monotonic | 96.4% | 15.8% | 56.1% | 51/90 | 2/90 |

**v16e is the current best** — 71.2% balanced accuracy, nearly tripling irregular accuracy vs the original 15.8% ceiling. Key: Gumbel hard selection + diversity loss (lateral inhibition) with small experts (hidden=32). 9/19 irregulars correct.

### Next: LSTM decoder with Bahdanau attention (v7)

Planned implementation to close the irregular gap:
- `LSTMDecoder` with Bahdanau attention — replicates K&C's decoder
- No copy mechanism needed (every char generated from vocab, no regular/irregular asymmetry)
- No monotonic alignment bias needed (LSTM + Bahdanau naturally monotonic)
- Config: `decoder_type: "lstm"` vs `"transformer"` (default)
- v7-v9: LSTM decoder variants tested (1-2 layers, d=128/256, with/without copy, with/without slots)
- **LSTM maxes at ~41%** — far below Transformer+copy+monotonic (92.4%)
- LSTM needs to learn alignment from data; Transformer gets it free via Gaussian bias
- Copy mechanism doesn't help LSTM (Bahdanau already provides implicit copying)
- **Slot attention hurts in every LSTM variant** — consistent finding across all decoder types
- Root cause of LSTM underperformance (v7-v9): Transformer encoder + LSTM decoder is a mismatched pairing. K&C used biLSTM encoder whose hidden states are naturally compatible with Bahdanau attention.

### v10: BiLSTM encoder + LSTM decoder (true K&C replication)

**Confirmed**: encoder-decoder mismatch was the issue. `BiLSTMEncoder` in `src/model/encoder.py` — bidirectional LSTM, hidden_size=d_model/2 per direction, packed sequences for padding.

| Config | Encoder | Decoder | Slots | Regular | Irregular | Balanced |
|---|---|---|---|---|---|---|
| **v10a** | **biLSTM** | **LSTM(2L)** | **No** | **99.5%** | **5.3%** | **52.4%** |
| v10b | biLSTM | LSTM(2L) | Yes(K=4) | 60.6% | 5.3% | 32.9% |

v10a matches K&C on regulars (99.5% vs ~95%). Slots drop regular accuracy by 39 points (v10a→v10b). biLSTM+LSTM still can't crack irregulars (5.3% = 1/19).

### v11: Forcing slot utilization (4 strategies)

All use biLSTM encoder + LSTM decoder + slots.

| Config | Strategy | Regular | Irregular | Balanced | Wug reg |
|---|---|---|---|---|---|
| v11a | Weak decoder (1L) | 62.4% | 0% | 31.2% | 40/90 |
| **v11b** | **Tight bottleneck (2 slots)** | **77.7%** | **15.8%** | **46.8%** | **30/90** |
| **v11c** | **Pretrain slots (30 ep recon)** | **84.5%** | **15.8%** | **50.1%** | **41/90** |
| v11d | Class auxiliary (alpha_cls=1.0) | 50.8% | 5.3% | 28.0% | 25/90 |

**Findings:**
- **Pretraining slots is the best strategy** (v11c, 50.1% balanced). Autoencoding stabilizes slot representations before transduction.
- **Tight bottleneck second** (v11b, 46.8%). 2 slots forces specialization — tied for best irregular (15.8%).
- **Weak decoder doesn't help** (v11a). Reducing capacity doesn't force slot reliance.
- **Class auxiliary backfired** (v11d, worst). Optimizing for class prediction competes with transduction.
- **15.8% irregular ceiling** (3/19) shared by v6, v11b, v11c — likely same 3 semi-regular verbs.
- **No slot config beats any no-slot baseline on balanced accuracy.**

**Features added for v11:**
- `pretrain_epochs` config: Phase 0 trains encoder+slots+recon_decoder on present→present autoencoding before transduction.
- `alpha_cls` config: Verb-class classifier on pooled slot representations (K*d → 2 classes).
- Data pipeline: `PastTenseDataset` returns `(src, tgt, reg_label)` 3-tuples.

### v12: Developmental L0 annealing (learn first, prune later)

Inspired by child language development: children learn rules and exceptions together with full capacity, then gradually consolidate. Implementation: `l0_anneal_start` and `l0_anneal_epochs` config params. Phase 1 sets `target_l0 = num_slots` (all open), Phase 2 linearly anneals to final `target_l0`, Phase 3 holds. Early stopping is disabled until after `l0_anneal_start`, and `best_val_loss` resets when annealing begins.

| Config | Data | Phase 1 | Anneal over | Job | Balanced Acc |
|---|---|---|---|---|---|
| v12a | natural | 50 ep | 50-100 | 1446406 | 50.3% (reg 95.3%, irreg 5.3%) |
| v12b (old) | balanced (256 examples) | 50 ep | 50-100 | 1446407 | 22.4% (reg 18.4%, irreg **26.3%**) |
| v12b (new) | oversample (6,206 ex) | 50 ep | 50-100 | 1446437 | 52.2% (reg 88.6%, irreg 15.8%) |

**v12a result**: Annealing didn't help. Best checkpoint from epoch 51 (right at anneal start). Model reverted to Phase 1 state.

**v12b (balanced) result**: Low overall (22.4%) but **highest irregular ever: 26.3%, 11/90 wug irregular**. However, the "balanced" regime subsampled regulars to 128 — only 256 total training examples, causing severe underfitting.

**v12b (oversample) result**: Fixed data size (6,206 examples). Regulars recovered to 88.6% but irregulars back to 15.8% ceiling. Best checkpoint still from epoch 51 (Phase 1). **Annealing consistently fails to improve over Phase 1** — the pruning phase only hurts.

**Conclusion**: Developmental L0 annealing doesn't work. The model converges in Phase 1 (all slots open = effectively no slot bottleneck), and the subsequent pruning phase never recovers. The approach is fundamentally flawed: by the time annealing starts, the model has already learned a solution that uses all slots as pass-through, and there's no gradient signal to reorganize them.

**`oversample` training regime**: Added to `dataset.py`. Keeps all regulars, repeats irregulars ~24x to match. Full dataset with equal class frequency.

### v13: Decoder bottleneck + developmental annealing

Combines v12 annealing with a constrained decoder output head. `dec_bottleneck` config param adds a narrow layer before output projection: `[lstm_out; context](256) → ReLU(32) → vocab(43)`. The LSTM keeps full d_model=128 hidden state for alignment, but its ability to independently compute the answer is limited — forces reliance on slot content.

| Config | Data | dec_bottleneck | Anneal | Job |
|---|---|---|---|---|
| v13a | natural | 32 | 50→100 | 1446410 |
| v13b | oversample | 32 | 50→100 | 1446438 |

**v13 results**:

| Config | Data | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| v13a | natural | 91.2% | 5.3% | 48.2% | 45/90 | 2/90 |
| **v13b** | **oversample** | **85.5%** | **31.6%** | **58.5%** | **41/90** | **10/90** |

**v13b is the new best model.** First slot-based model to beat the no-slot baseline (v6, 56.1%) on balanced accuracy. 31.6% irregular (6/19) doubles the previous ceiling. Decoder bottleneck + oversampled irregulars is the winning combination.

**Theoretical issue with v13**: The bottleneck at `[lstm_out; context] → 32 → vocab` constrains both the LSTM's own state and the slot-derived context equally. It doesn't selectively weaken the decoder — it crushes the slot information too. Yet it works empirically — likely because the constraint forces the model to rely on attention patterns rather than raw information passing.

### v14: Small LSTM hidden (correct asymmetric bottleneck)

Instead of bottlenecking the output, reduce the LSTM hidden size itself (`lstm_hidden=32`). Bahdanau attention still queries full d_model=128 slot representations. Output projection: `[lstm_out(32); context(128)] → vocab(43)`. The decoder has minimal independent state (32-dim) but full-resolution access to slot content (128-dim). This correctly creates asymmetric capacity: slots are the high-bandwidth pathway, LSTM is the low-bandwidth one.

| Config | Data | lstm_hidden | Anneal | target_l0 | Regular | Irregular | Balanced |
|---|---|---|---|---|---|---|---|
| v14a | natural | 32 | 50→100 | 2.0 | 89.4% | 5.3% | 47.3% |
| v14b | oversample | 32 | 50→100 | 2.0 | 87.6% | 26.3% | 56.9% |
| v14c | oversample | 32 | none | 2.0 | 76.4% | 31.6% | 54.0% |
| v14d | oversample | 32 | none | 4.0 | 91.2% | 15.8% | 53.5% |

**Findings**: L0 pruning matters — v14c (target_l0=2) gets 31.6% irregular vs v14d (target_l0=4) 15.8%. Annealing helps regulars (v14b 87.6% vs v14c 76.4%) but not irregulars. v13b (output bottleneck, 58.5%) still beats all v14 variants despite being theoretically less sound.

### v15: Forced slot specialization (TopK and Gumbel-softmax)

Analysis of v13b revealed **all L0 gates = 1.0** at eval — no slot specialization despite target_l0=2.0. The hard-concrete's soft gates satisfy the L0 budget during training without actually closing. Two new mechanisms that guarantee discrete selection:

**`l0_mode: topk`** — `TopKDrop`: MLP scores each slot, keeps top-k by score, zeros rest. Straight-through gradient for training. Guarantees exactly k active slots.

**`l0_mode: gumbel`** — `GumbelSlotRouter`: Gumbel-softmax categorical selection. Samples k slots without replacement. Temperature anneals 1.0→0.1 for progressively harder selection. Entropy regularization encourages peaked distributions.

| Config | Selection | k | dec_bottleneck | Data | Job |
|---|---|---|---|---|---|
| v15a | top-k | 2 | 32 | oversample | 1446527 |
| v15b | gumbel | 2 | 32 | oversample | 1446542 |

**v15 results**:

| Config | Selection | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| v15a | TopK (k=2) | 75.1% | 15.8% | 45.5% | 40/90 | 6/90 |
| v15b | Gumbel (k=2) | 72.5% | 21.1% | 46.8% | 33/90 | 5/90 |

**Both significantly worse than v13b (58.5%).** Forced discrete slot selection hurts — drops balanced accuracy by 12-13 points. The decoder bottleneck was doing the work in v13b, not slot specialization.

**Slot stats (v15a)**: Correctly predicted irregulars used gates `[1,1,0,0]` (slots 0+1 only). Wrong irregulars activated slot 2 more (0.44 vs 0.0). Some differentiation exists but doesn't translate to performance. v15b (Gumbel) had noisy training (val_loss spiked to 2.41 at epoch 8) and higher best val_loss (0.297 vs 0.226).

**Conclusion**: Forced specialization is counterproductive. Morphological transduction isn't naturally decomposable into K independent slot-sized components the way visual scenes are. v13b succeeded because its soft gates stayed open (all = 1.0) — the bottleneck, not slot selection, drove the gains.

### v16: Mixture of Expert Decoders (replacing Slot Attention)

Replaces slot attention entirely with K small LSTM decoder experts and a router. Each expert learns a different transformation rule (e.g., regular suffixation vs vowel change). Router examines encoder output holistically, selects expert(s) per input.

| Config | Routing | Experts | expert_hidden | Data | Job |
|---|---|---|---|---|---|
| v16a | soft | 4 | 64 | oversample | 1449787 |
| v16b | gumbel (tau 1.0→0.1) | 4 | 64 | oversample | 1449793 |
| v16c | gumbel + guided | 4 | 32 | oversample | 1449818 |
| v16d | soft + diversity | 4 | 64 | oversample | 1449838 |
| v16e | gumbel + diversity | 4 | 32 | oversample | 1449844 |

**v16c key changes**: smaller experts (32 vs 64), very low load balancing (0.001), auxiliary reg/irreg classifier on router (annealed to 0 over 50 epochs). Balance=4.0 throughout — no differentiation achieved.

**v16d key changes**: based on v16a (best) + diversity loss (`lambda_diversity=0.1`). Penalizes pairwise cosine similarity between expert outputs — lateral inhibition. No labels used for routing. Forces experts apart purely by output divergence.

**v16e key changes**: combines v16c's Gumbel routing + small experts (32) with diversity loss. Tests whether hard selection + lateral inhibition achieves discrete rule-like specialization.

### v17: Population-coded decoder (no router)

Replaces the external router with **self-gating neurons** inspired by population coding. Each `PopulationNeuron` (`src/model/population_decoder.py`) computes both output logits AND a scalar confidence from its own learned projection of the encoder output. The confidence acts as a firing rate — neurons fire strongly for inputs matching their tuning. Final output = softmax(confidences) weighted sum of neuron outputs.

**Key architectural difference from MoE**: no separate router network. Each neuron determines its own contribution weight, analogous to how biological neurons' firing rates are intrinsic to the neuron, not assigned by an external controller.

Clamped diversity loss (lateral inhibition) prevents neurons from converging to identical tuning curves. Clamped at 0: penalizes positive cosine similarity between neuron outputs, does not reward anti-correlation.

| Config | confidence_mode | Neurons | expert_hidden | lambda_div | Job |
|---|---|---|---|---|---|
| v17a | input | 4 | 64 | 0.1 | 1449875 |
| v17b | hidden | 4 | 64 | 0.1 | 1449895 |
| v17c | certainty | 4 | 64 | 0.1 | 1449896 |
| v17d | input | 8 | 64 | 0.1 | 1449922 |

**Confidence modes**: `input` = from encoder pooled output (v17a), `hidden` = from LSTM final hidden state after decoding (v17b), `certainty` = mean max prediction probability, parameter-free (v17c).

**v17 results**:

| Config | Mode | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v17a** | **input** | **96.4%** | **42.1%** | **69.2%** | **43/90** | **9/90** |
| v17c | certainty | 92.2% | 31.6% | 61.9% | 42/90 | 12/90 |
| v17b | hidden | 29.3% | 36.8% | 33.1% | 18/90 | 10/90 |

**v17b failed** — hidden-state confidence overfits to teacher-forced hidden states, which diverge from greedy decode hidden states at inference. Train/eval mismatch.

**v17c collapsed** — certainty-based confidence creates winner-take-all (bal=0.01, one neuron dominates). No learned head means no gradient path to rebalance. Still works okay (61.9%) because the dominant neuron is decent.

**v17a wins** — input-based confidence avoids train/eval mismatch (encoder output is identical at both times) and the learned head allows balanced neuron participation.

### v18: Population decoder improvements

Based on learnings: small experts (32) helped in v16e, diversity loss was 0.0000 in v17a (neurons already diverge naturally). New idea: **neuron dropout** — randomly mask neurons during training (p=0.5), forcing each to be independently competent. Analogous to synaptic pruning during neural development. Prevents co-adaptation.

| Config | Neurons | expert_hidden | neuron_dropout | Job |
|---|---|---|---|---|
| v18a | 4 | 32 | 0.0 | 1449931 |
| v18b | 4 | 64 | 0.5 | 1449932 |

**v17d/v18 results**:

| Config | Change | Regular | Irregular | Balanced |
|---|---|---|---|---|
| v17a (baseline) | 4n, h=64 | 96.4% | 42.1% | **69.2%** |
| v18b | + neuron dropout | 96.6% | 36.8% | 66.7% |
| v18a | small neurons (32) | 97.2% | 31.6% | 64.4% |
| v17d | 8 neurons | 98.2% | 10.5% | 54.4% |

None beat v17a. More neurons dilute irregular signal via majority voting. Small neurons reduce capacity without forcing specialization (unlike hard routing). Neuron dropout fights the architecture's collaborative strength.

v18c (confidence tau annealing 1.0→0.1) also didn't help — 66.7% balanced. Best checkpoint at epoch 25 (tau≈0.775), meaning sharpening hurts. Every attempt to push population coding toward specialization degrades it.

**v17a (4 neurons, h=64, τ=1.0) is the population coding ceiling** at 69.2% balanced.

### v17a vs v10a stability comparison (5 seeds each)

| Model | Params | Mean Balanced | Std |
|---|---|---|---|
| v17a (population) | 721K | 68.8% | ±9.7% |
| v10a-large (single LSTM, 3L) | 715K | 67.7% | ±7.3% |
| v10a (single LSTM, 2L) | 583K | 67.5% | ±6.6% |

**No significant difference.** Population coding provides no advantage over a single LSTM decoder, even when controlling for parameter count. High variance (±7-10%) comes from seed-dependent train/test splits with small irregular test sets (16-20 verbs). v16e's earlier 71.2% was within the normal range for a single LSTM.

### Expert/neuron analysis (v16e + v17a)

**v16e**: Complete expert collapse — all 405 test verbs route to Expert 0 (prob=1.000). Experts 1-3 are dead. The 71.2% comes from a single LSTM decoder, not modular specialization.

**v17a**: Only neurons 0 and 2 active (N1, N3 dead). N2 handles ~70% of verbs (98.9% regular). N0 handles ~30% (87% regular, but captures most irregulars). N0 fires more strongly for irregulars (0.65 vs 0.46). Some differentiation, but not discrete rules.

**Correct irregulars** (both models get similar sets): vowel-change patterns where stem structure is preserved (speak→spoke, forget→forgot, keep→kept, spit→spat, overcome→overcame). **Incorrect irregulars**: suppletive or complex changes where the model applies regular suffixation (dig→digd, sell→seld, break→breakt, lie→laid).

### v16e stability test (5 seeds)

| Seed | Regular | Irregular | Balanced |
|---|---|---|---|
| 1 | 96.9% | 50.0% | 73.4% |
| 3 | 96.9% | 50.0% | 73.5% |
| 4 | 96.1% | 45.0% | 70.6% |
| 42 | 95.1% | 47.4% | 71.2% |
| 5 | 94.3% | 38.9% | 66.6% |

**Mean: 69.9% ± 3.5%** — result is robust across 6 seeds. Range 64.4%–73.5%. Different seeds change the train/test split, so irregular counts vary (16-20 test irregulars).

### v19: Edit transducer (factored transduction)

Instead of generating output characters from scratch, the decoder predicts **edit operations** (COPY, DELETE, INSERT(c), SUB(c)) applied to the source string. Bakes in the inductive bias that morphology is mostly copying with local modifications.

**Architecture** (`src/model/edit_decoder.py`):
- `EditVocab`: maps edit operations to indices. Size = 4 + 2*V (PAD, END, COPY, DELETE, INSERT_c, SUB_c)
- `align_to_edits()`: Needleman-Wunsch alignment converts (src, tgt) → edit script for training
- `EditDecoder`: LSTM decoder with Bahdanau attention + source read pointer. Input = [edit_emb; context; current_src_char_emb]. Predicts next edit operation.
- `apply_edits()`: deterministically applies edit script to source at inference

**Why this should work**: Regular rule = COPY COPY ... INSERT(d) — trivially learnable pattern. Irregular = COPY COPY SUB(vowel) COPY — explicit vowel change. The edit vocabulary forces the model to frame inflection as "modify the input" rather than "generate from scratch."

| Config | Architecture | Fix | Job |
|---|---|---|---|
| v19a | single EditDecoder | none | 1450179 |
| v19b | 4 PopulationEditNeurons | none | 1450186 |
| v19c | EditLabeler (non-autoregressive) | no pointer | 1450210 |
| v19d | EditDecoder + scheduled sampling | anneal ss 0→0.5 | 1450211 |

**v19a/b: 0.5% accuracy — failed.** Source pointer drifts at inference (exposure bias). Teacher-forced pointer works fine but greedy decode misaligns.

**v19c (edit labeler)**: Per-position classification (COPY/DELETE/SUB) in parallel — no pointer at all. Small LSTM suffix head predicts appended characters. Non-autoregressive. `src/model/edit_labeler.py`.

**v19d (scheduled sampling)**: Fixes v19a by gradually replacing teacher-forced edits with model's own predictions during training (probability 0→0.5 over 100 epochs).

**v19 results — all failed:**

| Config | Approach | Regular | Irregular | Balanced |
|---|---|---|---|---|
| v19a | sequential edits | 0.5% | 0% | 0.3% |
| v19b | population + sequential | 0.3% | 0% | 0.1% |
| v19c | per-position labeling | 0% | 26.3% | 13.2% |
| v19d | sequential + sched. sampling | 0% | 0% | 0% |

Sequential versions: pointer drift at inference. Scheduled sampling doesn't fix it — error compounds. Per-position version: overfits (train 0.008, val 0.35), suffix head too weak to generalize. Edit transduction concept is sound but implementation needs fundamental rework.

**v16 results**:

| Config | Routing | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v16e** | **gumbel + diversity** | **95.1%** | **47.4%** | **71.2%** | **38/90** | **12/90** |
| v16c | gumbel + guided | 97.4% | 42.1% | 69.8% | 48/90 | 9/90 |
| v16a | soft | 95.3% | 31.6% | 63.5% | 40/90 | 14/90 |
| v16d | soft + diversity | 96.1% | 26.3% | 61.2% | 40/90 | 13/90 |
| v16b | gumbel | 93.8% | 15.8% | 54.8% | 40/90 | 13/90 |

**v16e is the new best model** — 71.2% balanced accuracy, 47.4% irregular (9/19). Gumbel + diversity loss is the winning combination. Clamped diversity loss bug (v16d) caused negative training loss but didn't prevent learning.

### Revised v1 results (static L0, stronger lambda — FAILED)

Config: d_model=64, 4 slots, 2 layers, dropout=0.3, lambda_l0=1.0 with 20-epoch warmup, alpha_recon=0.5.

L0 worked (dropped 3.4→2.9) but overwhelmed transduction: total loss *increased* as warmup progressed. Best checkpoint at epoch 3 (before L0 kicked in). **0% accuracy.** Root cause: static gates + fixed lambda can't balance two loss terms at different scales across training.

### Revised v2 results (input-conditional gates)

Config: `configs/revised_v2.yaml` — d_model=64, 4 slots, 2 layers, dropout=0.3, `l0_mode=conditional`, `target_l0=2.0`, `alpha_recon=0.5`, patience=25.

- Val loss: **0.69**, train loss: 0.99. Test accuracy: **16.1%** (reg 14.5%, irreg 0%, wug 0/90)
- L0 constraint works perfectly. Train/val gap 0.30.
- Key issue: underfitting — train loss ~1.0 means model can't fit training data.

### v3 ablation results (d_model is the key factor)

| Config | d_model | target_l0 | Val loss | Train loss | Test acc |
|---|---|---|---|---|---|
| v3_no_recon | 64 | 2.0 | 0.770 | 0.71 | 7.9% |
| v3_relaxed_l0 | 64 | 3.0 | 0.868 | 0.89 | 7.7% |
| **v3_bigger** | **128** | **3.0** | **0.602** | **0.47** | **21.0%** |

**Findings**: d_model=64 was severely underpowered. d_model=128 with conditional L0 + dropout=0.3 achieves best results. Removing reconstruction without adding capacity hurts. Relaxing L0 alone doesn't help. All runs: 0% irregular, 0% wug.

### v4b results (fixed copy mechanism — current best with slots)

Config: `configs/v4b_copy_fixed.yaml` — d_model=128, 4 slots, `slot_nhead=4`, `use_copy=true`, `target_l0=3.0`, fixed copy Q/K projections + p_gen bias.

- Val loss: **0.467**, test accuracy: **30.6%** (regular 32.4%, irregular 0%, wug 0/90)

### v5 results (no-slot baseline — matches slot-based models)

Config: `configs/v5_baseline_no_slots.yaml` — same as v4b but `use_slots=false`.

- Val loss: **0.466**, test accuracy: **31.4%** (regular 32.4%, irregular 0%, wug 0/90)
- **KEY FINDING: Slot attention provides NO benefit.** v5 matches v4b with 24% fewer params and faster training. The character scrambling in predictions is caused by the copy mechanism/decoder, not the slot bottleneck.
- 0% on irregulars and nonce verbs across ALL runs — consistent across all architectures

### Effect of hyperparameters on training

- **Higher `num_slots`**: More capacity, risk of not pruning enough
- **Higher `target_l0`** (conditional mode): More active slots, more capacity, less compression
- **Lower `target_l0`**: Fewer active slots, stronger bottleneck — forces more abstract representations but may hurt accuracy
- **Higher `alpha_recon`**: Stabilizes slot representations, may slow transduction learning but improve generalization

## Evaluation Strategy

Three verb categories, evaluated separately in `src/evaluate.py`:
1. **Seen verbs** — training data (sanity check)
2. **Unseen real verbs** — held-out test split (regular vs irregular reported separately)
3. **Nonce verbs (wugs)** — Albright & Hayes 58 nonce verbs, compared against both regular and irregular candidate forms

Key comparison: vary training regimes (regular/irregular verb distributions) to assess abstract rule learning vs pattern memorization.

## Key References

- Locatello et al. (2020) — Original Slot Attention
- Behjati & Henderson — Dynamic Capacity Slot Attention for character sequences (primary architecture reference)
- Kirov & Cotterell (2018) — Encoder-decoder RNN baseline + dataset
- Corkery et al. (2019) — Showed nonce-verb model-human correlation is weak/unstable
- Ma & Gao (2022) — Transformer evaluation on past-tense inflection
