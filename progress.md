# Progress Log

## 2026-03-29: Project Setup & Smoke Tests

### 1. Dataset

**Source**: Kirov & Cotterell (2018) — `github.com/ckirov/RevisitPinkerAndPrince`

Cloned to `data/RevisitPinkerAndPrince/`. Contains:
- `experiment_1/english_merged.txt` — **4,039 verb pairs** (3,871 regular, 168 irregular)
  - Format: `orth_present \t orth_past \t phon_present \t phon_past \t reg|irreg`
  - Phonological characters use DISC encoding (e.g., `f O : n` → `f O : n d`)
- `experiment_1_wugs/` — **90 nonce verbs** (Albright & Hayes) for generalization testing
  - Space-separated phonological chars with stress markers (È)

This is the same dataset used by Kirov & Cotterell (2018), Corkery et al. (2019), and Ma & Gao (2022), ensuring direct comparability.

Data split (deterministic, seed=42): **3,231 train / 403 val / 405 test**

Character vocabulary size: **43** (39 phonological characters + 4 special tokens: pad, sos, eos, unk)

### 2. Environment Setup

```bash
module load miniconda
conda create -n slot_attention python=3.11
conda activate slot_attention
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn tqdm pyyaml
```

- **PyTorch 2.4.0+cu121** — compatible with Yale HPC CUDA driver (v12.8)
- Note: `torch==2.11.0` (latest) was incompatible — CUDA driver too old (requires >=13.0). Downgraded to 2.4.0+cu121.
- Login nodes have no GPU; CUDA only available via SLURM GPU partition.

### 3. Code Structure Created

```
src/
├── model/
│   ├── __init__.py
│   ├── encoder.py          # TransformerCharEncoder
│   ├── slot_attention.py   # SlotAttentionModule (iterative competitive attention)
│   ├── l0drop.py           # L0Drop (hard-concrete gates)
│   ├── decoder.py          # TransformerCharDecoder (autoregressive)
│   └── full_model.py       # SlotAttentionTransducer (end-to-end + loss)
├── data/
│   ├── __init__.py
│   ├── vocab.py            # CharVocab (encode/decode, special tokens)
│   └── dataset.py          # Data loading, splitting, PastTenseDataset, collate_fn
├── train.py                # Training loop (AdamW, cosine LR, checkpointing)
└── evaluate.py             # Evaluation (exact-match, reg/irreg breakdown, wugs)
configs/
├── default.yaml            # Full training config (100 epochs, d_model=128)
└── smoke_test.yaml         # Quick test config (2 epochs, d_model=64)
scripts/
├── run_experiment.sh       # SLURM job for full training
└── smoke_test.sh           # SLURM job for smoke test
```

### 4. Smoke Test Results

All tests run on GPU via SLURM (job 1444910, node r818u33n06).

#### 4a. Data Loading
- Loaded 4,039 verb pairs from `english_merged.txt` — OK
- Split into 3,231 / 403 / 405 — OK
- Built vocab of 43 characters — OK
- Loaded 90 nonce verbs from `experiment_1_wugs/` — OK

#### 4b. Model Forward/Backward Pass (CPU, batch=4)
- Model params: **246,703** (d_model=64, 2 enc layers, 2 dec layers, 4 slots)
- Forward pass: loss=3.93, logits shape (4, 11, 43) — OK
- Backward pass: 83 parameters received gradients, norm range [0.000000, 2.7025] — OK
- Greedy decode: produced output (garbage, as expected untrained) — OK

#### 4c. Multi-task Variant (CPU, batch=4)
- With `alpha_recon=0.5`: total loss=6.27 (transduce=4.22, recon=4.02, L0=3.32) — OK
- Backward pass through both decoders — OK

#### 4d. 2-Epoch GPU Training (SLURM job 1444910)
```
Epoch   1 | train_loss 3.2306 | val_loss 2.8156 | L0 3.32 | 6.1s
Epoch   2 | train_loss 2.7171 | val_loss 2.5654 | L0 3.32 | 2.6s
```
- Loss decreasing as expected — OK
- Test exact-match accuracy: 0.0% (expected after only 2 epochs) — OK
- Evaluation script ran successfully on checkpoint — OK
- Wug evaluation: 0/90 matches (expected) — OK
- No errors in stderr (only a harmless PyTorch nested tensor warning)

### 5. Status Summary

| Component | Status |
|---|---|
| Dataset downloaded | Done |
| Conda environment | Done (slot_attention, Python 3.11, PyTorch 2.4.0+cu121) |
| Data loading + vocab | Verified |
| Encoder | Verified |
| Slot Attention | Verified |
| L0Drop | Verified |
| Decoder | Verified |
| Full model (forward + backward) | Verified |
| Multi-task variant | Verified |
| Training loop (GPU) | Verified (2 epochs, loss decreasing) |
| Evaluation pipeline | Verified |
| Wug nonce evaluation | Verified |

### 6. Loss Plotting (added 2026-03-29)

`train.py` now tracks and plots losses every `plot_every` epochs (default: 5) to `results_dir`:
- `results/loss_curves.png` — 3-panel figure: total loss, transduction loss, L0 regularization
- `results/recon_loss.png` — reconstruction loss (only when multi-task `alpha_recon > 0`)
- `results/history.json` — raw loss values for further analysis

Config keys: `results_dir` (default `../results`), `plot_every` (default `5`).

### 7. Hyperparameter Sweep Infrastructure (added 2026-03-29)

Added `scripts/sweep.py` — generates YAML configs and submits SLURM jobs for all combinations.

**Hyperparameter grid** (27 jobs):
- `num_slots`: [4, 8, 16]
- `lambda_l0`: [0.001, 0.01, 0.1]
- `alpha_recon`: [0.0, 0.5, 1.0]

**Training regime experiments** (3 jobs, run after finding best hyperparams):
- `natural`: all data as-is
- `balanced`: subsample regulars to match irregular count
- `irregular_first`: irregulars first, then full dataset (mimics child acquisition)

Usage:
```bash
python scripts/sweep.py                  # launch 27-job sweep
python scripts/sweep.py --dry-run        # preview without submitting
python scripts/sweep.py --regime-only --num-slots 8 --lambda-l0 0.01 --alpha-recon 0.0
```

Each job writes results to `results/sweep_<name>/` and checkpoints to `checkpoints/sweep_<name>/`.

Also added `training_regime` config key and `apply_training_regime()` in `src/data/dataset.py`.

### 8. Initial Training Run Results (2026-03-29, SLURM job 1444952)

**Config**: default.yaml — d_model=128, 8 slots, 3 enc/dec layers, lambda_l0=0.01, 100 epochs, A40 GPU. Completed in 3m38s (~1.9s/epoch).

**Training log**:
```
Epoch   1 | train_loss 2.4321 | val_loss 1.8698 | L0 6.64
Epoch  36 | train_loss 0.3116 | val_loss 0.7542 | L0 6.66   ← best val loss
Epoch 100 | train_loss 0.1561 | val_loss 0.7774 | L0 6.65
```

**Evaluation (best checkpoint, epoch 36)**:
```
all_test:  8.64% (405 examples)
regular:   8.55% (386 examples)
irregular: 0.00% (19 examples)
Wug:       0/90 regular, 0/90 irregular
```

**Diagnosis — three major problems**:

1. **Severe overfitting**: Train loss reached 0.15 but val loss plateaued at 0.75 from epoch ~36. Train/val gap of 0.62. The 1.2M-parameter model is too large for 3,231 training examples.

2. **L0 sparsity completely ineffective**: L0 stayed at 6.64-6.66 (out of 8.0) for the entire run. At lambda_l0=0.01, the L0 penalty contributes ~0.07 to total loss — negligible next to transduction loss. All 8 slots remain open; no pruning occurs.

3. **Near-zero generalization**: 8.6% accuracy on regular test verbs, 0% on irregulars, 0% on nonce verbs. The model memorizes some frequent training patterns but has not learned the regular "-ed" rule or any irregular mappings.

**Root cause**: The slot attention bottleneck is not functioning. With all slots open and no sparsity pressure, the model acts like a standard encoder-decoder that overfits the small training set.

### 9. Planned Model Changes (priority order)

1. **Increase lambda_l0 to 0.5-2.0** (or add a warmup schedule: 0 for first 20 epochs, then ramp to 1.0). Current 0.01 is ~100x too weak.
2. **Reduce model capacity**: d_model 128→64, layers 3→2, dropout 0.1→0.3. Dataset is too small for 1.2M params.
3. **Enable reconstruction** (alpha_recon=0.5): Forces slots to preserve input info, should reduce overfitting.
4. **Add early stopping**: Val loss plateaued at epoch 36. Training to 100 was pure overfitting.
5. **Reduce num_slots from 8 to 4**: Past-tense likely needs only 2-3 components.
6. **Consider lambda_l0 warmup**: Let model learn basic transduction first, then increase sparsity pressure.

### 10. Revised Model Implementation (2026-03-29)

All changes from section 9 have been implemented:

**Code changes to `src/train.py`:**
- **L0 warmup schedule**: New config key `l0_warmup_epochs`. Lambda_l0 ramps linearly from 0 to target over first N epochs. Effective value logged each epoch.
- **Early stopping**: New config key `patience`. Stops training when val loss hasn't improved for N epochs. Loads best checkpoint before final evaluation.
- **Reconstruction in training loop**: When `alpha_recon > 0`, constructs `src_for_recon` (with SOS/EOS) and passes to model. Both decoders share the same pruned slot memory.
- **All `print` → `log`**: Flushed output for real-time monitoring via `tee`.
- **Evaluation appended to SLURM job**: `run_experiment.sh` now runs `evaluate.py` after training completes.

**New config `configs/revised_v1.yaml`:**

| Parameter | Default | Revised v1 | Rationale |
|---|---|---|---|
| d_model | 128 | 64 | Reduce from 1.2M → ~250K params for 3k dataset |
| enc/dec_layers | 3 | 2 | Less capacity |
| d_ff | 256 | 128 | Less capacity |
| dropout | 0.1 | 0.3 | Stronger regularization |
| num_slots | 8 | 4 | Past-tense needs ~2-3 components |
| lambda_l0 | 0.01 | 1.0 | 100x stronger sparsity pressure |
| l0_warmup_epochs | 0 | 20 | Ramp L0 to avoid premature pruning |
| alpha_recon | 0.0 | 0.5 | Multi-task to improve slot quality |
| patience | 0 | 20 | Early stopping |
| epochs | 100 | 200 | More room (early stopping will cut short) |

**Submitted**: SLURM job 1444994 with `configs/revised_v1.yaml`.

### 11. Revised v1 Run Results (2026-03-29, SLURM job 1444994)

**Result: FAILURE** — L0 worked but overwhelmed transduction learning.

```
Epoch   1 | train_loss 2.7654 | val_loss 2.6063 | L0 3.40 | λ_l0 0.0500
Epoch   3 | train_loss 3.3127 | val_loss 2.5826 | L0 3.30 | λ_l0 0.1500   ← best val
Epoch  20 | train_loss 5.1940 | val_loss 4.3079 | L0 2.94 | λ_l0 1.0000
Epoch  23 | early stopped | L0 2.88
```

Evaluation: **0% accuracy** across all categories.

**What happened**: L0 sparsity now works (dropped from 3.4 → 2.9 slots), but the L0 loss term (2.9 × 1.0 = 2.9) completely dominates the transduction loss (~0.15-2.4). As lambda_l0 ramps up, total loss *increases* because the model is penalized more for keeping slots open than rewarded for correct predictions. Best checkpoint is epoch 3 — before L0 kicks in meaningfully.

**Fundamental problem identified**: The L0 gates are **input-independent** (static `log_alpha` per slot). This means:
- Same gates for every verb — can't use 2 slots for regulars and 3 for irregulars
- L0Drop just learns a static mask, equivalent to training with fewer slots
- The proposal says "variable number of active slots depending on the input" but the implementation can't do this

Additionally, **loss scale mismatch**: transduction loss ranges 0.15-3.7 while L0 is ~3-7. No fixed lambda works across the full training trajectory.

### 12. Strategy Change: Input-Conditional Gates

Replacing static L0Drop with **input-conditional gates** following Behjati & Henderson's Dynamic Capacity approach:

**Architecture change**: `z_k = sigmoid(MLP(slot_k))` — gate values are computed from slot contents, so different inputs activate different slot subsets.

**Loss change**: Replace fixed `lambda_l0 * L_L0` with **Lagrangian constraint** targeting a specific L0 budget:
```
L = L_transduce + lambda * (L0 - target_l0)
```
Where `lambda` is updated dynamically to maintain the target. This eliminates the scale mismatch problem.

### 13. Input-Conditional Gates Implementation (2026-03-29)

Implemented `InputConditionalL0Drop` in `src/model/l0drop.py`:
- Gate MLP: `log_alpha_k = MLP(slot_k)` — gates depend on slot content, not static
- Lagrangian constraint: `loss = lambda * (E[L0] - target) + (E[L0] - target)²`
- Lambda auto-updated via dual gradient ascent each epoch
- `full_model.py` updated with `l0_mode` parameter (`"conditional"` or `"static"`)
- `train.py` updated to call `update_lagrangian()` after each epoch, log lambda value

### 14. Revised v2 Results (COMPLETED — SLURM job 1445006)

**Config**: `configs/revised_v2.yaml` — d_model=64, 4 slots, 2 layers, dropout=0.3, `l0_mode=conditional`, `target_l0=2.0`, `alpha_recon=0.5`, patience=25.

**Training log** (early stopped at epoch 189, best at epoch 164):
```
Epoch   1 | train_loss 4.8238 | val_loss 2.5953 | L0_loss 0.0052 | λ_lagr 0.0007
Epoch  28 | train_loss 1.8470 | val_loss 1.0950 | L0_loss 0.0000 | λ_lagr 0.0013
Epoch  88 | train_loss 1.1769 | val_loss 0.7266 | L0_loss 0.0000 | λ_lagr 0.0013
Epoch 164 | train_loss 0.9963 | val_loss 0.6884 | L0_loss 0.0000 | λ_lagr 0.0013  ← best
Epoch 189 | early stopped (patience=25)
```

**Evaluation (best checkpoint, epoch 164)**:
```
all_test:  16.05% (405 examples)
regular:   14.51% (386 examples)
irregular:  0.00% (19 examples)
Wug:       0/90 regular, 0/90 irregular
```

**Analysis — improved but still fundamentally limited**:

1. **L0 constraint works perfectly**: L0_loss ≈ 0 throughout, Lagrangian lambda stable at 0.0013. The model uses exactly 2.0 active slots as targeted. Input-conditional gates + Lagrangian approach solved the loss scale mismatch from v1.

2. **Overfitting reduced**: Train/val gap of 0.30 (was 0.62 in initial run). Dropout=0.3 + reconstruction helping. But val loss still plateaued at 0.69 — far from the <0.3 needed for 80%+ accuracy.

3. **Train loss itself is high (~0.99)**: The model hasn't even fit the training data well. This is NOT just overfitting — the model lacks capacity or the optimization is suboptimal. Possible causes:
   - **Reconstruction dilutes learning**: With alpha_recon=0.5, roughly half the gradient signal goes to present→present reconstruction, leaving less capacity for the actual transduction task.
   - **target_l0=2.0 too aggressive**: Only 2 active slots may not provide enough capacity to represent the verb transformation.
   - **Model too small**: d_model=64 with 2 layers may be underpowered for this task.

4. **Accuracy doubled but still poor**: 16.1% vs initial 8.6%. Still 0% on irregulars and nonce verbs. The model has not learned abstract rules.

5. **Comparison across runs**:

| Run | Val loss | Train loss | L0 | Test acc | Reg acc | Irreg acc | Wug |
|---|---|---|---|---|---|---|---|
| Initial (default) | 0.75 | 0.15 | 6.65/8 | 8.6% | 8.6% | 0% | 0% |
| Revised v1 (static) | 2.58 | 3.31 | 2.88/4 | 0% | 0% | 0% | 0% |
| **Revised v2 (cond.)** | **0.69** | **0.99** | **2.0/4** | **16.1%** | **14.5%** | **0%** | **0%** |

### 15. Diagnosis & Next Steps

**Core problem**: The model is underfitting (train loss ~1.0). Before worrying about generalization, we need the model to actually learn the training data well. The reconstruction objective and tight bottleneck are likely preventing this.

**Planned experiments** (priority order):

1. **Ablate reconstruction** — Run v2 config but with `alpha_recon=0.0`. If train loss drops significantly, reconstruction is the bottleneck. This is the single most important experiment to run next.

2. **Relax L0 target** — Try `target_l0=3.0` (75% of slots instead of 50%). More capacity might be needed before compression helps.

3. **Increase model capacity** — If train loss is still high without reconstruction and with relaxed L0, try d_model=128 with 2 layers (more width, same depth). Keep dropout=0.3 to control overfitting.

4. **Staged training** — First train to convergence without L0/reconstruction, then fine-tune with sparsity. This separates "can the model learn the task?" from "can it learn it with slots?"

5. **Hyperparameter sweep** — Only after finding a config that achieves >50% test accuracy. Current sweep grid needs updating based on these results.

### 16. v3 Ablation Experiments (2026-03-29)

Created three configs to diagnose the underfitting problem:

| Config | Key change from v2 | Goal |
|---|---|---|
| `configs/v3_no_recon.yaml` | `alpha_recon=0.0` | Does reconstruction hurt transduction? |
| `configs/v3_relaxed_l0.yaml` | `target_l0=3.0`, `alpha_recon=0.0` | Is 2 slots too tight? |
| `configs/v3_bigger.yaml` | `d_model=128`, `target_l0=3.0`, `alpha_recon=0.0` | Is d_model=64 underpowered? |

Each saves to separate `checkpoints/v3_*/` and `results/v3_*/` dirs. Updated `run_experiment.sh` to extract `save_dir` from config so evaluation finds the right checkpoint.

**Submitted** (2026-03-29):
- **Job 1445043**: `v3_no_recon.yaml`
- **Job 1445044**: `v3_relaxed_l0.yaml`
- **Job 1445045**: `v3_bigger.yaml`

**Results** (all completed 2026-03-29):

| Config | Job | Params | Best epoch | Val loss | Train loss | Test acc | Reg acc | Irreg |
|---|---|---|---|---|---|---|---|---|
| v3_no_recon | 1445043 | 249K | 148 | 0.770 | 0.71 | 7.9% | 7.8% | 0% |
| v3_relaxed_l0 | 1445044 | 249K | 149 | 0.868 | 0.89 | 7.7% | 7.5% | 0% |
| **v3_bigger** | **1445045** | **903K** | **116** | **0.602** | **0.47** | **21.0%** | **20.7%** | **0%** |

**Analysis — model capacity is the dominant factor**:

1. **v3_no_recon vs v2**: Dropping reconstruction *hurt* — accuracy dropped from 16.1% to 7.9%. Val loss went from 0.69 to 0.77. Reconstruction was actually helping by regularizing the slot representations, not hurting. Without it, the small model overfits more (train 0.71 vs val 0.77 = 0.06 gap, but much worse quality).

2. **v3_relaxed_l0 vs v3_no_recon**: Relaxing L0 from 2.0 to 3.0 made things *worse* — val loss 0.87 vs 0.77. Interesting: the Lagrangian lambda is much higher (0.20 vs 0.002), meaning the L0 constraint is fighting harder. With 3 active slots and d_model=64, the model has more capacity through the bottleneck but still can't learn well. The extra slot capacity adds noise rather than helping.

3. **v3_bigger is the clear winner**: d_model=128 (903K params) achieved 21.0% accuracy with val loss 0.60 — best so far by a wide margin. Train loss 0.47 shows the model can actually fit the training data now. Train/val gap of 0.13 is healthy with dropout=0.3.

**Key takeaway**: d_model=64 was severely underpowered. The initial run with d_model=128 (no L0, no recon) had 8.6% accuracy and massively overfit. v3_bigger with d_model=128 + conditional L0 + dropout=0.3 achieves 21.0% — the regularization mechanisms are working, they just needed enough base capacity to work with.

**But 21% is still far from useful.** All runs show 0% on irregulars and 0% on nonce verbs. The model learns some regular patterns from training data but has not learned abstract rules. This may be a fundamental limitation of the architecture for this task (see section 18).

### 17. Architecture Improvements: Copy Mechanism + Multi-Head Slots (2026-03-29)

Implemented two changes to address fundamental inductive bias problems identified during analysis:

**Problem 1: Position-level competition is wrong for morphology**

Original slot attention: `softmax(attn, dim=slot_dim)` — slots compete for character positions. But morphological rules don't partition by position. The "-ed" rule needs the whole stem.

**Solution: Multi-head slot attention** (`slot_nhead` config key). Split d_slot into H heads. Within each head (feature subspace), slots compete independently. Slot k can attend to different positions in different feature groups — captures distributed morphological information.

Implementation in `src/model/slot_attention.py`. With `nhead=1`, identical to original. With `nhead=4`, attention is `(B*4, K, N)` with independent softmax per head.

**Problem 2: Model wastes capacity re-learning to copy characters**

~90% of output characters in past-tense are identical to the input (the stem is copied). Without a copy mechanism, every character must be generated through the slot bottleneck. The model spends most of its capacity on the trivial copy operation instead of learning the actual morphological rule.

**Solution: Pointer/copy mechanism** (`use_copy` config key) in `src/model/decoder.py`, following See et al. (2017):
- At each decoding step, compute `p_gen` (generate from vocab) vs `1-p_gen` (copy from source)
- Copy attention: decoder state queries encoder hidden states directly (bypasses slot bottleneck for copying)
- Final distribution: `p_gen * vocab_dist + (1-p_gen) * copy_dist`
- Gate: `p_gen = sigmoid(W · [decoder_state; copy_context; target_embedding])`

This provides the strongest inductive bias: the model only needs to learn WHEN to generate vs copy, and WHAT to generate for the non-copied parts. Copying is "free."

**Config**: `configs/v4_copy_multihead.yaml` — d_model=128, 4 slots, `slot_nhead=4`, `use_copy=true`, `target_l0=3.0`, no reconstruction.

Verified: forward pass, backward pass, and greedy decode all work on CPU.

**Submitted**: SLURM job 1445064.

**Results** (completed, early stopped at epoch 95, best at epoch 70):
```
Epoch   1 | train_loss 2.7421 | val_loss 1.9447 | L0_loss 0.0000
Epoch  10 | train_loss 1.0995 | val_loss 0.8222 | L0_loss 0.0059
Epoch  30 | train_loss 0.6721 | val_loss 0.6028 | L0_loss 0.0142
Epoch  62 | train_loss 0.4985 | val_loss 0.4950 | L0_loss 0.0007
Epoch  70 | train_loss 0.4527 | val_loss 0.4743 | L0_loss -0.0006  ← best
Epoch  95 | early stopped (patience=25)
```

**Evaluation**:
```
all_test:  28.9% (405 examples)
regular:   30.3% (386 examples)
irregular:  0.0% (19 examples)
Wug:       0/90 regular, 0/90 irregular
```

**Analysis — copy mechanism is a clear win**:

1. **Best result by far**: 28.9% test accuracy vs 21.0% (v3_bigger). Val loss 0.474 vs 0.602. The copy mechanism provides a substantial improvement.

2. **Much faster convergence**: Reached val_loss=0.60 by epoch 30 (took v3_bigger 116 epochs). Early stopped at 95. The copy mechanism dramatically reduces the learning burden.

3. **Training dynamics healthier**: Train/val gap at best: 0.45 - 0.47 = -0.02 (essentially no gap!). Later overfitting resumes (0.37 vs 0.49 at epoch 95). Dropout=0.3 + copy mechanism are providing good regularization.

4. **L0 behavior**: L0_loss stays near 0 throughout (constraint satisfied), with Lagrangian lambda rising slowly (0→0.05). The model is using ~3 active slots as targeted.

5. **Still 0% on irregulars and nonce verbs**: The model has learned to copy stems and append regular suffixes, but hasn't learned irregular mappings or abstract rules. This is consistent with the concern that the copy mechanism makes regular patterns easy but doesn't help with irregulars (which require generation, not copying).

**Comparison across all runs**:

| Run | Val loss | Test acc | Reg acc | Irreg | Wug | Key feature |
|---|---|---|---|---|---|---|
| Initial (default) | 0.75 | 8.6% | 8.6% | 0% | 0% | d=128, no L0 |
| v2 (conditional) | 0.69 | 16.1% | 14.5% | 0% | 0% | d=64, recon |
| v3_bigger | 0.60 | 21.0% | 20.7% | 0% | 0% | d=128, cond L0 |
| **v4 (copy+MH)** | **0.47** | **28.9%** | **30.3%** | **0%** | **0%** | **copy, multihead** |

### 18. Prediction Error Analysis (2026-03-29)

Inspected v4 predictions on GPU (SLURM job 1445098). The model is NOT just "appending -ed" — it's **scrambling character order**:

```
lOk     -> lOkt      | pred: kOlkt    [reversed!]
S3:k    -> S3:kt     | pred: k3:St    [reversed!]
tSVk    -> tSVkt     | pred: kVtSt    [reversed!]
sprInt  -> sprIntId  | pred: prInstId [shuffled]
nVrIS   -> nVrISt    | pred: rVnISt   [swapped]
```

Correct predictions tend to be shorter, simpler words (`laInd`, `wEdZd`, `klINkt`). Longer words get scrambled.

**Root cause**: Slot attention compresses N positions into K=4 unordered slots, **destroying positional information**. The decoder reconstructs from unordered slot vectors. For short words it works; for longer words ordering breaks down.

The copy mechanism SHOULD fix this (it can point to specific source positions), but it's not being used effectively due to a representation space mismatch (see section 19).

For irregulars, the model applies regularization-like suffixes (`spIt` → `spItId`) rather than learning stem changes (`sp&t`). It has not learned any irregular patterns.

### 19. Copy Mechanism Bug Fix + No-Slot Baseline (2026-03-30)

**Bug identified in copy mechanism**: The copy attention used a single projection (`copy_attn_proj`) to map decoder states into encoder space. But after cross-attending to slot memory, decoder states live in a slot-conditioned representation space — a single linear projection can't bridge this gap well. This caused poor copy attention alignment → character scrambling.

**Two fixes applied to `src/model/decoder.py`**:

1. **Separate Q/K projections for copy attention**: Added `copy_query_proj` (decoder side) and `copy_key_proj` (encoder side). Both map into a shared alignment space. This is standard practice in attention mechanisms.

2. **p_gen bias initialization to -2.0**: `sigmoid(-2.0) ≈ 0.12`, biasing the model toward copying from the start. Since ~90% of output chars are copied, this provides the right inductive bias. Previously used default init (~0.5 = equal weight).

**No-slot baseline implemented**: Added `use_slots: false` config option to `SlotAttentionTransducer`. When disabled, decoder cross-attends directly to encoder output H instead of slot memory M'. This tests whether slot attention helps or hurts.

**Submitted experiments**:
- **Job 1445142**: `configs/v4b_copy_fixed.yaml` — v4 + copy mechanism fixes (separate Q/K, p_gen bias)
- **Job 1445143**: `configs/v5_baseline_no_slots.yaml` — no slot attention, direct encoder-decoder with copy

**Key comparison**:
- v4b vs v4: Did the copy mechanism fix improve accuracy?
- v5 vs v4b: Does slot attention help or hurt? If v5 >> v4b, slot attention is hurting.

### 20. v4b + v5 Results (2026-03-30)

| Run | Params | Best epoch | Val loss | Train loss | Test acc | Reg acc | Irreg | Wug |
|---|---|---|---|---|---|---|---|---|
| v4 (copy, orig) | 920K | 70 | 0.474 | 0.45 | 28.9% | 30.3% | 0% | 0% |
| **v4b (copy, fixed)** | **936K** | **78** | **0.467** | **0.38** | **30.6%** | **32.4%** | **0%** | **0%** |
| **v5 (no slots)** | **712K** | **80** | **0.466** | **0.39** | **31.4%** | **32.4%** | **0%** | **0%** |

**Critical finding: Slot attention provides NO benefit.**

v5 (no slots, 712K params) matches v4b (with slots, 936K params) exactly on regular accuracy (32.4%) and slightly beats on total accuracy (31.4% vs 30.6%). It's also 24% smaller and trains faster (1.1s vs 1.6s per epoch).

This means:
1. **The copy mechanism fix helped slightly** (v4b 30.6% vs v4 28.9%)
2. **Slot attention is NOT contributing** — the decoder gets equally good results cross-attending directly to encoder states
3. **The character scrambling persists equally in both** — it's NOT caused by slot attention destroying positional info. The scrambling happens in the copy mechanism or decoder itself.

**Character scrambling analysis** (same errors in v4b and v5):

Both models show identical scrambling patterns:
```
lOk  -> lOkt | v4b: kOlkt  | v5: klOkt   [both reversed]
S3:k -> S3:kt | v4b: k3:St  | v5: k3:St   [identical error!]
sprInt -> sprIntId | v4b: prIstId | v5: prInstId [similar]
```

Since v5 has NO slot bottleneck (decoder sees all N encoder positions directly), the scrambling is NOT caused by slot attention compressing into K slots. It's caused by either:
1. **The copy mechanism not learning proper alignment** — copy attention may be attending to wrong positions
2. **The Transformer decoder itself** struggling with character-level autoregressive generation from phonological sequences
3. **Insufficient training data** — 3,231 examples may not be enough for a Transformer to learn positional correspondence

### 21. Revised understanding

The slot attention bottleneck is neither helping nor hurting — it's irrelevant. The real bottleneck is:
1. **Copy mechanism alignment** — not learning to copy from the right positions
2. **Data scarcity** — 3,231 training examples for a Transformer-based model
3. **Phonological encoding** — DISC characters may be harder to align than orthographic

**Next steps to improve accuracy**:
1. **Diagnose copy mechanism** — run `scripts/diagnose_copy.py` to check p_gen values and copy attention alignment
2. **Try orthographic mode** — `use_phonological: false` may be easier to learn (simpler character set, more intuitive patterns)
3. **Add positional bias to copy attention** — monotonic alignment prior (morphological transduction is roughly left-to-right)
4. **Increase training data** — consider data augmentation or using all phonological + orthographic pairs

### 22. Copy Mechanism Diagnostic Results (2026-03-30, SLURM 1445177)

Ran `scripts/diagnose_copy.py` on v5 (no-slot baseline). Findings:

**1. p_gen is near zero — the model IS copying, not generating.**

Average p_gen across examples: 0.06-0.16. The model overwhelmingly uses the copy path. The p_gen bias initialization to -2.0 worked — the model learned to copy. But it's copying from **wrong positions**.

**2. Copy attention is NOT monotonic — this is the root cause of scrambling.**

Example: `laIn` → target `laInd`, predicted `InlaInd`:
```
     |     l     a     I     n
t= 0 |  0.34  0.03  0.49  0.14   ← copies I (pos 2) instead of l (pos 0)
t= 1 |  0.03  0.00  0.00  0.97   ← copies n (pos 3) instead of a (pos 1)
t= 2 |  0.95  0.01  0.00  0.03   ← NOW copies l (pos 0)
t= 3 |  0.00  1.00  0.00  0.00   ← copies a (pos 1)
```
The model copies the right characters but in scrambled order. It starts mid-word, then loops back.

**3. Repeated characters cause infinite loops.**

Example: `bIl` → target `bIld`, predicted `bIlIbIlIbIlIbd` (14 chars!):
The copy attention cycles b→I→l→I→b→I→l... because the model can't distinguish "I've already copied this position." Without monotonic constraints, the model has no concept of progress through the source.

**4. Duplicate source characters confuse copy attention.**

Example: `krItIsaIz` has `I` at positions 2, 4, 7. Copy attention distributes evenly across all three:
```
t= 2 |  0.00  0.00  0.34  0.00  0.32  0.00  0.02  0.32  0.00  ← splits across 3 I's
```
With softmax attention, identical characters at different positions get similar scores, making positional alignment impossible.

**5. The model has NO mechanism for tracking decoding progress.**

The Transformer decoder has no recurrent state — it relies on self-attention over previously generated tokens. But for copy attention, there's no signal telling it "I've already copied positions 0-3, now copy position 4." The copy query is based on the decoder's hidden state, which doesn't encode progress through the source.

### 23. Comparison to Prior Work

**Prior work achieves 93-99% on regulars, 75-92% on irregulars. We get 32%.**

| Approach | Regular | Irregular | Key mechanism |
|---|---|---|---|
| Kirov & Cotterell 2018 (biLSTM+attn) | ~95% | ~85-90% | Bahdanau attention, recurrent decoder |
| Ma & Gao 2022 (Transformer) | ~90-97% | ~80-90% | Standard Transformer encoder-decoder |
| **Our v5 (Transformer+copy)** | **32.4%** | **0%** | Copy mechanism, no alignment bias |

**Why prior work succeeds and we fail:**

1. **Bahdanau attention in RNN seq2seq naturally learns monotonic alignment.** The LSTM decoder state carries implicit position information — at each step, the hidden state encodes "I'm at position k in the output." This makes the attention tend toward diagonal alignment.

2. **Our Transformer decoder has no such bias.** Transformer self-attention is permutation-equivariant. The sinusoidal positional encoding provides SOME position info, but the copy attention has no inherent monotonic tendency. With only 3,231 training examples, the model can't learn proper alignment from data alone.

3. **Hard monotonic attention models** (Aharoni & Goldberg 2017) explicitly enforce left-to-right alignment. These are particularly effective for morphological transduction where input-output alignment is nearly monotonic.

**Root cause: We need a monotonic alignment bias in the copy attention.**

### 24. Proposed Fix: Monotonic Positional Bias

Add a **relative position bias** to copy attention that encourages diagonal (monotonic) alignment:

```
copy_energy[t, n] = query_t · key_n + bias(t, n)
```

Where `bias(t, n)` is highest when `n ≈ t` (diagonal) and decreases away from the diagonal. This can be:
- **Gaussian bias**: `bias(t,n) = -α * (t - n)²` — peaked at diagonal, decays quadratically
- **Learned relative position embedding** — let the model learn the bias

This is a standard technique in speech recognition (monotonic attention) and machine translation (local attention). It would be trivially addable to the copy attention computation.

### 25. Monotonic Alignment Bias Implementation (2026-03-30)

Added learnable Gaussian positional bias to copy attention in `src/model/decoder.py`:

```
copy_energy[t, n] += -exp(log_alpha) * (t * (n_src/m) - n)²
```

- `log_alpha` is a learnable parameter (initialized to 1.0), controlling sharpness
- `t * (n_src/m)` scales target positions to source length (handles different lengths)
- Bias peaks on the diagonal and decays quadratically away from it
- At init: `alpha=e^1≈2.7`, so positions ±1 from diagonal get bias of `-2.7`, positions ±2 get `-10.8`
- This strongly encourages monotonic (left-to-right) alignment while still allowing the model to learn non-monotonic patterns where needed (alpha can decrease during training)

**Submitted experiments**:
- **Job 1445185**: `v6_monotonic_no_slots.yaml` — no slots, copy + monotonic bias
- **Job 1445186**: `v6_monotonic_slots.yaml` — with slots, copy + monotonic bias

**First attempt (jobs 1445185/1445186)**: FAILED — 0.5% accuracy despite val_loss=0.089. The length-scaled alignment `t*(n_src/m)` breaks during autoregressive decoding because `m` changes at each step, causing unstable alignment jumps.

**Fix**: Removed length scaling. For past-tense morphology, output ≈ input + suffix, so raw positions work: output position t should copy from source position t. Also reduced initial alpha from e^1=2.7 to e^0=1.0 (gentler bias).

**Second attempt (jobs 1445201/1445202) — SUCCESS:**

| Run | Params | Best epoch | Val loss | Train loss | Test acc | Reg acc | Irreg | Wug |
|---|---|---|---|---|---|---|---|---|
| v6 no-slots | 712K | 95 | 0.063 | 0.02 | **92.4%** | **96.4%** | **15.8%** | 0% |
| v6 slots | 936K | 74 | 0.078 | 0.03 | **91.1%** | **95.6%** | **5.3%** | 0% |

**This is a breakthrough.** From 32% to 92% with a single change (monotonic alignment bias).

**Analysis**:

1. **Regular accuracy 96.4%** — in line with prior work (Kirov & Cotterell ~95%). The model has learned the past-tense rule. Remaining errors are mostly verbs ending in consonant clusters where the suffix choice (d/t/Id) is tricky.

2. **Irregular accuracy 15.8% (3/19)** — the model correctly handles `VnbaInd→VnbaUnd`, `sni:k→sni:kt`, `spi:k→sp@Uk`. These are irregulars that happen to follow sub-regular patterns (vowel change). Most irregulars are still regularized (e.g., `go→god` instead of `went`).

3. **Slots vs no-slots**: No-slots is slightly better again (92.4% vs 91.1%, 15.8% vs 5.3% irregular). Slot attention provides no benefit even with proper alignment.

4. **Error patterns** (from prediction inspection):
   - Some verbs get an extra "-Id" suffix: `tr&fIk→tr&fIktId` (should be `tr&fIkt`)
   - Some verbs get truncated: `b&n→b&n` (missing the `d`)
   - Irregulars mostly regularized: `spIt→spItId` (should be `sp&t`)

5. **Wug verbs still 0%** — need to investigate why. The model should generalize the regular rule to nonce verbs.

**Full comparison across all runs**:

| Run | Val loss | Test acc | Reg | Irreg | Key change |
|---|---|---|---|---|---|
| Initial (default) | 0.75 | 8.6% | 8.6% | 0% | Baseline |
| v2 (cond L0) | 0.69 | 16.1% | 14.5% | 0% | Input-conditional gates |
| v3_bigger | 0.60 | 21.0% | 20.7% | 0% | d_model=128 |
| v4 (copy) | 0.47 | 28.9% | 30.3% | 0% | Copy mechanism |
| v5 (no slots) | 0.47 | 31.4% | 32.4% | 0% | Removed slots |
| **v6 (monotonic)** | **0.063** | **92.4%** | **96.4%** | **15.8%** | **Monotonic alignment** |

### 26. Wug 0% Root Cause: Phonological Encoding Mismatch (2026-03-30)

**The wug data uses a completely different phonological transcription than the training data.** This is why wug accuracy is 0% — the model literally cannot read the input.

| Feature | Training data (DISC) | Wug data (CELEXmod) |
|---|---|---|
| Stress markers | None | `È` (present in every word) |
| PRICE diphthong | `aI` (2 chars) | `Y` (1 char) |
| MOUTH diphthong | `aU` (2 chars) | `W` (1 char) |
| GOAT diphthong | `@U` (2 chars) | `o` (1 char) |
| NURSE vowel | `3:` (2 chars) | `Õ` (1 char) |
| STRUT vowel | `V` | `Ã` |
| TRAP vowel | `&` | `Q` |
| CH affricate | `tS` (2 chars) | `C` (1 char) |
| J affricate | `dZ` (2 chars) | `J` (1 char) |

10 characters in wug data (`C J Q W Y o Ã È Õ «`) are completely absent from the training vocabulary. Every wug verb has at least `È`, so **every single wug input has OOV tokens** mapped to `<unk>`.

**Fix implemented** in `src/data/dataset.py`: Added `_wug_to_disc()` function that converts CELEXmod encoding to DISC:
- Strips `È` stress markers
- Maps single chars to DISC equivalents (e.g., `Ã` → `V`, `Q` → `&`)
- Expands diphthongs/affricates to multi-char DISC sequences (e.g., `Y` → `aI`, `C` → `tS`)

**Re-evaluation with fixed encoding** (SLURM job 1445231):
```
all_test:  92.4% (405 examples)
regular:   96.4% (386 examples)
irregular: 15.8% (19 examples)
Wug:       51/90 regular (56.7%), 2/90 irregular (2.2%)
```

**Wug accuracy jumped from 0% to 56.7%** — the encoding fix was the entire issue. The model HAS learned the regular past-tense rule and can generalize it to novel verbs. 51/90 nonce verbs receive the correct regular past-tense form.

### 27. Balanced + Orthographic Experiments (2026-03-30)

| Run | Config | Test acc | Regular | Irregular | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v6 phonological** | natural | **92.4%** | **96.4%** | **15.8%** | **51/90** | **2/90** |
| v6 balanced | balanced | 30.4% | 31.9% | 21.1% | 21/90 | 5/90 |
| v6 orthographic | natural, orth | 89.9% | 93.3% | 10.5% | 1/90 | 0/90 |

**Analysis**:

1. **Balanced training** (job 1445226): Overall accuracy dropped sharply (30.4%) because subsampling regulars to match the 168 irregulars leaves only ~336 training examples. However, irregular accuracy improved (21.1% vs 15.8%) and wug results show 21/90 regular + 5/90 irregular — the model learns SOME irregular patterns. The tradeoff is too extreme: losing 95% of training data for marginal irregular gains.

2. **Orthographic** (job 1445227): Slightly lower than phonological (89.9% vs 92.4%). Regular accuracy 93.3% vs 96.4%. Wug: only 1/90 — the wug evaluation uses phonological data, so orthographic mode can't evaluate wugs properly (wug files have no orthographic forms). Wug comparison is not valid here.

3. **Phonological is clearly better**: 96.4% regular, 51/90 wug generalizations. The DISC encoding captures the phonological regularities needed for past-tense rule learning.

### 28. Comparison to Kirov & Cotterell (2018)

| Metric | K&C 2018 (biLSTM) | Our v6 (Transformer+copy) |
|---|---|---|
| Architecture | biLSTM encoder-decoder | Transformer + copy + monotonic bias |
| Params | ~100-200K | 712K |
| Regular accuracy | ~95% | 96.4% |
| Irregular accuracy | ~85-90% | 15.8% |
| Wug regularization | Strong | 51/90 (56.7%) |

Our model matches K&C on regulars but lags significantly on irregulars. Key architectural differences:
- K&C used Bahdanau attention which naturally learns monotonic alignment (we needed explicit bias)
- K&C's biLSTM decoder carries sequential state, helping with positional tracking
- K&C had no explicit copy mechanism but the attention mechanism provides implicit copying
- Our low irregular accuracy (15.8% vs ~85-90%) is primarily a memorization issue — with only 168 irregular verbs in training, the model needs more exposure or explicit memorization capacity

### 29. Next: LSTM Decoder with Bahdanau Attention (planned)

**Why our Transformer+copy model fails on irregulars (15.8% vs K&C's ~85-90%)**:

1. **Copy bias suppresses generation**: p_gen ≈ 0.05 means the model copies 95% of characters. For irregulars that require vowel changes (e.g., `sINk→s&Nk`), the model must override this strong copy bias — hard to learn from only 168 examples.

2. **Transformer decoder lacks sequential state**: LSTM decoder's hidden state accumulates context step-by-step, enabling pattern memorization ("I'm generating an irregular vowel change"). Transformer decoder relies on self-attention, which is less effective for memorizing specific mappings from few examples.

**Plan: Implement LSTM decoder with Bahdanau attention**

This replicates Kirov & Cotterell's core architecture. Key properties:
- **No copy mechanism needed** — every character is generated from the vocabulary, giving equal capacity for regular and irregular patterns
- **No monotonic alignment bias needed** — Bahdanau attention with LSTM naturally learns monotonic alignment
- **Sequential hidden state** — enables better memorization of irregular patterns

**Experiments**:

| Config | Encoder | Decoder | Slots | Purpose |
|---|---|---|---|---|
| v7a | Transformer | LSTM + Bahdanau | No | K&C-style baseline — isolates decoder effect |
| v7b | Transformer | LSTM + Bahdanau | Yes | Does slot attention help with LSTM decoder? |

**Implementation needed**:
- New `LSTMDecoder` class in `src/model/decoder.py` with Bahdanau attention
- Config key `decoder_type: "lstm"` vs `"transformer"` (default) in `SlotAttentionTransducer`
- Update `train.py` and `evaluate.py` to pass `decoder_type` from config

**Expected outcome**: v7a should match K&C's ~85-90% irregular accuracy while maintaining ~95% regular accuracy. v7b tests whether slot attention provides any benefit with a more capable decoder.

### 30. LSTM Decoder Implementation (2026-03-30)

Implemented `LSTMDecoder` with Bahdanau attention in `src/model/decoder.py`:

**Architecture**:
- `BahdanauAttention`: additive attention `score = v^T tanh(W1·h_dec + W2·h_enc)`
- `LSTMDecoder`: unidirectional LSTM, input = `[embedding_t; context_{t-1}]`, output projection from `[lstm_out; context]`
- No copy mechanism — every character generated from vocabulary
- Sequential hidden state carries position/pattern information naturally

**Integration**:
- New config key: `decoder_type: "lstm"` (default: `"transformer"`)
- `SlotAttentionTransducer` in `full_model.py` selects decoder based on config
- When `decoder_type: "lstm"`, `use_copy` is forced to `False`
- Updated `train.py`, `evaluate.py`, `inspect_predictions.py` to pass `decoder_type`
- `dec_layers` controls LSTM layers (1 recommended, matches K&C)

**Model sizes**:
- v7a (LSTM, no slots): 518K params
- v7b (LSTM, slots): 741K params

**Submitted**:
- **Job 1445246**: `v7a_lstm_no_slots.yaml` — LSTM decoder, no slots (K&C baseline)
- **Job 1445247**: `v7b_lstm_slots.yaml` — LSTM decoder + slot attention

### 31. LSTM Decoder Results (2026-03-30)

Iterated through v7-v9 with various LSTM decoder configurations:

| Run | Decoder | d_model | Layers | Copy | Slots | Test acc | Reg | Irreg | Wug reg |
|---|---|---|---|---|---|---|---|---|---|
| v7a | LSTM | 128 | 1 | No | No | 13.6% | 13.7% | 0% | 4/90 |
| v7b | LSTM | 128 | 1 | No | Yes | 11.9% | 11.9% | 0% | 4/90 |
| v7c | LSTM | 128 | 2 | No | No | **41.2%** | **43.3%** | **0%** | **27/90** |
| v7d | LSTM | 128 | 2 | No | Yes | 17.3% | 18.1% | 0% | 8/90 |
| v8a | LSTM | 256 | 2 | No | No | 36.5% | 38.3% | 0% | 26/90 |
| v8b | LSTM | 256 | 2 | No | Yes | 16.1% | 16.6% | 0% | 20/90 |
| v9a | LSTM | 128 | 2 | Yes | No | 41.0% | 43.0% | 0% | 28/90 |
| v9b | LSTM | 128 | 2 | Yes | Yes | 10.1% | 11.1% | 0% | 2/90 |

**Key findings**:

1. **LSTM decoder maxes out at ~41%** — far below Transformer+copy+monotonic (92.4%). The LSTM cannot learn the alignment from 3K examples as well as the explicit monotonic bias.

2. **Copy mechanism doesn't help LSTM** (v9a ≈ v7c). Bahdanau attention already provides implicit copying; the explicit copy mechanism is redundant.

3. **Slot attention consistently hurts** — every slot variant underperforms its no-slot counterpart. The slot bottleneck compresses N encoder positions into K=4 slots, losing the positional information that Bahdanau attention needs for alignment.

4. **Bigger model doesn't help** (v8a < v7c). Overfits faster on the small dataset.

5. **Why LSTM << Transformer+copy+monotonic**: The LSTM decoder must learn alignment from data (via Bahdanau attention). The Transformer+copy decoder gets alignment for free (via the Gaussian bias). With only 3K examples, the explicit bias is worth ~50 percentage points.

6. **Why we don't match K&C**: K&C likely had different training details (scheduled sampling, optimizer, learning rate schedule) and their biLSTM encoder may produce more alignment-friendly representations. Our Transformer encoder is likely fine but the LSTM decoder needs more than 3K examples to learn proper attention patterns.

**Conclusion**: The Transformer decoder + copy mechanism + monotonic alignment bias (v6, 92.4%) remains the best architecture. The LSTM decoder is not competitive in our setup despite being successful in K&C's original work.

### 32. Full Results Summary (with balanced accuracy)

**Balanced accuracy** = `(regular_acc + irregular_acc) / 2`. Primary metric — irregulars are 50% of the problem.

| Run | Architecture | Regular | Irregular | **Balanced** | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v13b** | **biLSTM+LSTM+slots (bottleneck+os)** | **85.5%** | **31.6%** | **58.5%** | **41/90** | **10/90** |
| v6 no-slots | Transformer+copy+monotonic | 96.4% | 15.8% | 56.1% | 51/90 | 2/90 |
| v10a | biLSTM+LSTM (K&C style) | 99.5% | 5.3% | 52.4% | 48/90 | 6/90 |
| v12b (oversample) | biLSTM+LSTM+slots (anneal+os) | 88.6% | 15.8% | 52.2% | 47/90 | 5/90 |
| v11c | biLSTM+LSTM+slots (pretrain) | 84.5% | 15.8% | 50.1% | 41/90 | 5/90 |
| v14b | biLSTM+LSTM+slots (smallLSTM+os+anneal) | 87.6% | 26.3% | 56.9% | 35/90 | 5/90 |
| v14c | biLSTM+LSTM+slots (smallLSTM+os) | 76.4% | 31.6% | 54.0% | 43/90 | 7/90 |
| v14d | biLSTM+LSTM+slots (smallLSTM+os+noPrune) | 91.2% | 15.8% | 53.5% | 42/90 | 8/90 |
| v6 slots | Transformer+copy+monotonic+slots | 95.6% | 5.3% | 50.4% | 55/90 | 1/90 |
| v12a | biLSTM+LSTM+slots (anneal) | 95.3% | 5.3% | 50.3% | 48/90 | 4/90 |
| v13a | biLSTM+LSTM+slots (bottleneck) | 91.2% | 5.3% | 48.2% | 45/90 | 2/90 |
| v14a | biLSTM+LSTM+slots (smallLSTM) | 89.4% | 5.3% | 47.3% | 39/90 | 6/90 |
| v11b | biLSTM+LSTM+slots (tight) | 77.7% | 15.8% | 46.8% | 30/90 | 5/90 |
| v10b | biLSTM+LSTM+slots | 60.6% | 5.3% | 32.9% | 34/90 | 6/90 |
| v11a | biLSTM+LSTM+slots (weak dec) | 62.4% | 0% | 31.2% | 40/90 | 3/90 |
| v11d | biLSTM+LSTM+slots (cls aux) | 50.8% | 5.3% | 28.0% | 25/90 | 1/90 |
| v12b (balanced) | biLSTM+LSTM+slots (anneal+256ex) | 18.4% | **26.3%** | 22.4% | 4/90 | **11/90** |
| v7c | Transformer+LSTM | 43.3% | 0% | 21.6% | 27/90 | 0/90 |

**Key takeaways**:
1. **v13b is the best** (balanced 58.5%) — first slot model to beat no-slot baselines. Key: decoder bottleneck + oversampled irregulars.
2. **Oversampling is essential**: every natural-data run gets 5.3% irregular regardless of architecture.
3. **L0 pruning helps**: v14d (no prune, all slots open) gets 15.8% irreg; v14c (target_l0=2) gets 31.6%. Slot pruning forces specialization.
4. **Annealing helps regulars, not irregulars**: v14b (anneal) 87.6% reg vs v14c (no anneal) 76.4% reg, but irregular similar.
5. **Output bottleneck > LSTM bottleneck**: v13b (58.5%) beats all v14 variants despite being theoretically less sound. The more aggressive compression works better empirically.
6. **Three ingredients for best performance**: constrained decoder + oversampled irregulars + L0 pruning.

### 33. v10: BiLSTM Encoder Results (2026-03-30)

**Confirmed**: encoder-decoder mismatch was the LSTM bottleneck. biLSTM+LSTM (v10a) hits 99.5% regular — matching K&C.

| Config | Regular | Irregular | Balanced | Wug reg | Val loss | Best epoch |
|---|---|---|---|---|---|---|
| **v10a (no slots)** | **99.5%** | **5.3%** | **52.4%** | **48/90** | **0.039** | **47** |
| v10b (slots K=4) | 60.6% | 5.3% | 32.9% | 34/90 | 0.358 | 139 |

Slots drop regular accuracy by 39 points. Neither config cracks irregulars.

### 34. v11: Forcing Slot Utilization — Results (2026-03-30)

| Config | Strategy | Regular | Irregular | Balanced | Wug reg | Val loss |
|---|---|---|---|---|---|---|
| v11a | Weak decoder (1L) | 62.4% | 0% | 31.2% | 40/90 | 0.317 |
| **v11b** | **Tight bottleneck (2 slots)** | **77.7%** | **15.8%** | **46.8%** | **30/90** | **0.195** |
| **v11c** | **Pretrain (30 ep recon)** | **84.5%** | **15.8%** | **50.1%** | **41/90** | **0.161** |
| v11d | Class auxiliary | 50.8% | 5.3% | 28.0% | 25/90 | 0.828 |

**Findings:**
1. **Pretraining is the best slot-forcing strategy** (v11c, balanced 50.1%). Autoencoding stabilizes slots before transduction, preventing the decoder from learning to ignore them.
2. **Tight bottleneck is second** (v11b, balanced 46.8%). 2 slots forces specialization; tied for best irregular at 15.8%.
3. **Weak decoder doesn't help** (v11a, 31.2%). Reducing capacity isn't enough if slots aren't carrying useful info.
4. **Class auxiliary backfired** (v11d, 28.0% — worst). Auxiliary objective competes with transduction.
5. **15.8% irregular ceiling** (3/19) appears in v6, v11b, v11c — likely the same 3 semi-regular verbs every time.
6. **No slot config beats any no-slot baseline on balanced accuracy.** Slots remain a net negative.

**Code changes made:**
- `src/data/dataset.py`: Returns `(src, tgt, reg_label)` 3-tuples.
- `src/model/full_model.py`: Added `alpha_cls` param, `slot_classifier` head, `pretrain_epochs`.
- `src/train.py`: Phase 0 pretrain loop, passes `reg_labels` to model forward.
- `src/evaluate.py`: Reports balanced accuracy `(reg + irreg) / 2`.

### 35. v12: Developmental L0 Annealing — Learn First, Prune Later (2026-03-30)

**Motivation**: All prior slot experiments impose sparsity from the start, forcing compression before the model knows what it's compressing. Children don't do this — they learn rules and exceptions together first, then gradually consolidate (U-shaped development: memorize → overregularize → recover).

**Implementation**: Added `l0_anneal_start` and `l0_anneal_epochs` to `train.py`. The training loop updates `model.l0drop.target_l0` per epoch:
- **Phase 1** (epoch < anneal_start): `target_l0 = num_slots` (all slots open, no pruning pressure)
- **Phase 2** (anneal_start to anneal_start + anneal_epochs): linear interpolation from `num_slots` → final `target_l0`
- **Phase 3** (after anneal): hold at final `target_l0`

**Experiment design**: 2×2 factorial — data distribution × anneal timing:

| Config | Data | Phase 1 | Anneal over | Job |
|---|---|---|---|---|
| v12a | natural (~90% reg) | 100 epochs | 100 epochs | 1446385 |
| v12b | balanced (50/50) | 100 epochs | 100 epochs | 1446386 |
First attempt used phase 1 = 100 epochs — failed because early stopping triggered at epoch 112 (best at epoch 62) before annealing began. Fixed: early stopping disabled until `l0_anneal_start`, best_val_loss resets at anneal start. Resubmitted with phase 1 = 50 epochs (model overfits by ~epoch 50 anyway):

**Results:**

| Config | Data | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| v12a | natural | 95.3% | 5.3% | 50.3% | 48/90 | 4/90 |
| v12b (old) | balanced (256 ex) | 18.4% | **26.3%** | 22.4% | 4/90 | **11/90** |
| v12b (new) | oversample (6,206 ex) | 88.6% | 15.8% | 52.2% | 47/90 | 5/90 |

**v12a**: Best checkpoint epoch 51 (right at anneal start). Annealing only hurt.

**v12b (old, balanced)**: Highest irregular ever (26.3%, 11/90 wug irreg) but only 256 training examples → severe underfitting.

**v12b (new, oversample)**: Fixed data size with `oversample` regime (keeps all regulars, repeats irregulars ~24x). Regulars recovered to 88.6% but irregulars back to 15.8% ceiling. Best checkpoint still epoch 51.

**`oversample` training regime** added to `dataset.py`: keeps all regulars, repeats irregulars to match count. ~6.2K total.

**Conclusion**: Developmental L0 annealing doesn't work. All three v12 runs pick the Phase 1 checkpoint (before annealing). The model converges with all slots open as pass-through, and pruning only degrades. The approach fails because by the time annealing starts, the model has no incentive to reorganize slot usage — the pass-through solution is already optimal from the decoder's perspective.

Interesting signal: v12b-old's 26.3% irregular with downsampled data suggests **data distribution matters more than architecture** for irregular learning. The model can learn irregular patterns when they're 50% of the training data, even with very few examples.

### 36. v13: Decoder Bottleneck + Developmental Annealing (2026-03-30)

Combines v12 annealing with a constrained decoder output head. `dec_bottleneck=32` adds a narrow layer before output projection: `[lstm_out; context](256) → ReLU(32) → vocab(43)`. The LSTM keeps full d_model=128 hidden state for alignment/recurrence, but its ability to independently compute the output is constrained — forces it to rely on information from slot attention context.

| Config | Data | dec_bottleneck | Anneal | Job |
|---|---|---|---|---|
| v13a | natural | 32 | 50→100 | 1446410 |
| v13b | oversample | 32 | 50→100 | 1446438 |

**Results:**

| Config | Data | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| v13a | natural | 91.2% | 5.3% | 48.2% | 45/90 | 2/90 |
| **v13b** | **oversample** | **85.5%** | **31.6%** | **58.5%** | **41/90** | **10/90** |

**v13b is the new best model** — first slot-based model to beat the no-slot baseline (v6, 56.1%) on balanced accuracy. 31.6% irregular (6/19) doubles the previous 15.8% ceiling. Key ingredients: decoder bottleneck forces slot reliance + oversampling gives irregulars enough exposure.

**Annealing still ineffective**: Best checkpoints at epoch 50 (v13a) and 51 (v13b) — right at anneal start, same as v12. The bottleneck is what's helping, not the annealing.

**Oversampling is essential**: v13a (natural) gets 5.3% irregular; v13b (oversample) gets 31.6%. Same architecture, 6x difference in irregular accuracy.

**Theoretical issue**: The bottleneck at `[lstm_out; context] → 32 → vocab` crushes slot-derived context equally with LSTM output. It doesn't selectively weaken the decoder's independence — it weakens everything, including the slot info we want preserved. Yet it works empirically.

### 37. v14: Small LSTM Hidden — Correct Asymmetric Bottleneck (2026-03-30)

Fixes v13's theoretical problem. Instead of bottlenecking the output projection, reduce the **LSTM hidden size itself** (`lstm_hidden=32`). Bahdanau attention still queries full d_model=128 slot representations — the attention mechanism's key/value projections operate on full-dim slots. Output: `[lstm_out(32); context(128)] → vocab(43)`.

Information flow:
```
slots (4 × 128) ──attention──→ context (128) ──┐
                                                ├── [32 + 128] → vocab
embedding ──→ LSTM(hidden=32) → lstm_out (32) ──┘
```

The LSTM has only 32 dims of independent state — barely enough for position tracking and basic sequencing. All the heavy lifting (character identity, morphological pattern) must come through the 128-dim attention context over slots. This is the theoretically correct way to force slot reliance.

**Results:**

| Config | Data | Anneal | target_l0 | Regular | Irregular | Balanced | Wug irreg |
|---|---|---|---|---|---|---|---|
| v14a | natural | yes | 2.0 | 89.4% | 5.3% | 47.3% | 6/90 |
| v14b | oversample | yes | 2.0 | 87.6% | 26.3% | 56.9% | 5/90 |
| v14c | oversample | no | 2.0 | 76.4% | 31.6% | 54.0% | 7/90 |
| v14d | oversample | no | 4.0 | 91.2% | 15.8% | 53.5% | 8/90 |

**Findings:**
1. **L0 pruning matters**: v14c (target_l0=2) gets 31.6% irreg vs v14d (target_l0=4, no pruning) 15.8%. Pruning forces slot specialization even with a constrained decoder.
2. **Annealing helps regulars**: v14b (anneal) 87.6% reg vs v14c (no anneal) 76.4%. Phase 1 learning gives a better starting point for regular patterns.
3. **v13b still wins** (58.5% vs v14b's 56.9%). The output bottleneck is more aggressive — compresses entire 256-dim to 32 — which empirically outperforms the theoretically sounder asymmetric approach. Possible explanation: the output bottleneck forces the model to learn a more compressed/abstract representation overall, which happens to generalize better.

### 38. Known Issues

### 39. v13b Error Analysis and Slot Activation (2026-03-30)

**All L0 gates = 1.0 for every verb** (regular and irregular). No slot specialization whatsoever. The hard-concrete gates satisfy the L0 budget during training (with noise) but are fully open at eval (without noise). The decoder bottleneck is doing all the work — slots are a pass-through.

**Irregular error analysis** (5/19 correct):

Correct irregulars — all involve vowel changes with preserved stem structure:
- f@gEt→f@gOt (forget→forgot, E→O)
- VnbaInd→VnbaUnd (unbind→unbound, aI→aU)
- ri:raIt→ri:r@Ut (rewrite→rewrote, aI→@U)
- @Uv@kVm→@Uv@keIm (overcome→overcame, V→eI)
- sni:k→sni:kt (sneak→sneaked — actually regular-like)

Wrong irregulars — model applies regular rule (adds -t/-d/-Id):
- ki:p→ki:pt (keep, pred: should be kEpt — no vowel change learned)
- spi:k→spi:kt (speak, pred: should be sp@Uk)
- breIk→breIkt (break, pred: should be br@Uk)
- fi:l→fEld (feel, pred: close! got vowel but wrong ending, should be fElt)

Pattern: model learns vowel changes when stem is otherwise identical. Fails when both vowel AND consonant change, or when the transformation is suppletive.

### 40. v15: Forced Slot Specialization — TopK and Gumbel-softmax (2026-03-30)

Two new slot selection mechanisms that guarantee discrete selection:

**`TopKDrop`**: MLP scores each slot, hard top-k mask, straight-through gradient. Guarantees exactly k slots active. No soft gates to game.

**`GumbelSlotRouter`**: Gumbel-softmax categorical selection, samples k slots without replacement. Temperature anneals 1.0→0.1. Entropy regularization encourages peaked distributions.

| Config | Selection | k | dec_bottleneck | Data | Job |
|---|---|---|---|---|---|
| v15a | top-k | 2 | 32 | oversample | 1446527 |
| v15b | gumbel | 2 | 32 | oversample | 1446528 |

### 41. v15 Results: Forced Slot Specialization Failed (2026-03-31)

v15b was submitted as job 1446542 (original 1446528 was cancelled).

| Config | Selection | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| v15a (TopK) | top-k k=2 | 75.1% | 15.8% | 45.5% | 40/90 | 6/90 |
| v15b (Gumbel) | gumbel k=2 | 72.5% | 21.1% | 46.8% | 33/90 | 5/90 |

Both significantly worse than v13b (58.5% balanced). Forced discrete slot selection hurts — drops balanced accuracy by 12-13 points.

**Slot stats (v15a)**: Correctly predicted irregulars used gates `[1,1,0,0]` (slots 0+1). Wrong irregulars activated slot 2 more (0.44 vs 0.0). Some differentiation but doesn't help.

**Conclusion**: Forced specialization is counterproductive. Morphology isn't decomposable into K independent slot-sized components like visual scenes. The decoder bottleneck (not slot selection) drove v13b's gains. Slot attention's spatial-decomposition inductive bias fundamentally mismatches morphological transduction.

### 42. v16: Mixture of Experts Decoder (2026-03-31)

**Motivation**: Instead of slots competing over input positions, use K small decoder experts each learning a different transformation rule. A router network examines the encoder output and selects which expert(s) to apply. This directly encodes the "slot = rule" inductive bias.

**Architecture** (`src/model/moe_decoder.py`):
- `ExpertRouter`: mean-pools encoder output → MLP → expert weights (B, K)
- `MoEDecoder`: K independent `LSTMDecoder` experts, each with Bahdanau attention
- Supports soft (weighted blend), hard (straight-through argmax), and Gumbel-softmax routing
- Load balancing loss (Switch Transformer formulation) prevents expert collapse
- Sparse execution at eval time for hard/Gumbel modes

**Key differences from Slot Attention**:
1. Experts compete to be the *transformation*, not to *explain input positions*
2. Each expert is a full decoder (can learn a complete rule), not a feature vector
3. Router examines the whole input holistically, not position-by-position

| Config | Routing | Experts | expert_hidden | Data | Job |
|---|---|---|---|---|---|
| v16a | soft | 4 | 64 | oversample | 1449787 |
| v16b | gumbel (tau 1.0→0.1) | 4 | 64 | oversample | 1449793 |
| v16c | gumbel + guided cls | 4 | 32 | oversample | 1449818 |
| v16d | soft + diversity loss | 4 | 64 | oversample | 1449838 |
| v16e | gumbel + diversity loss | 4 | 32 | oversample | 1449844 |

**v16c design**: Gumbel routing + small experts (32-dim, forces abstraction) + low load balancing (0.001, allows natural 95/5 reg/irreg split) + auxiliary reg/irreg classifier on router pooled representation (alpha=1.0, annealed to 0 over 50 epochs to guide early routing without constraining final solution).

**v16 results**:

| Config | Routing | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v16a** | **soft** | **95.3%** | **31.6%** | **63.5%** | **40/90** | **14/90** |
| v16b | gumbel | 93.8% | 15.8% | 54.8% | 40/90 | 13/90 |

**v16 full results**:

| Config | Routing | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v16e** | **gumbel + diversity** | **95.1%** | **47.4%** | **71.2%** | **38/90** | **12/90** |
| v16c | gumbel + guided | 97.4% | 42.1% | 69.8% | 48/90 | 9/90 |
| v16a | soft | 95.3% | 31.6% | 63.5% | 40/90 | 14/90 |
| v16d | soft + diversity | 96.1% | 26.3% | 61.2% | 40/90 | 13/90 |
| v16b | gumbel | 93.8% | 15.8% | 54.8% | 40/90 | 13/90 |

**v16e is the new best** — 71.2% balanced, 47.4% irregular (9/19). v16d had negative training loss bug (diversity loss not clamped, cosine sim went negative).

### 44. v17: Population-coded decoder (2026-03-31)

Replaces external router with self-gating neurons (population coding). Each neuron computes output + confidence. Clamped diversity loss for lateral inhibition.

| Config | confidence_mode | Neurons | expert_hidden | lambda_div | Job |
|---|---|---|---|---|---|
| v17a | input | 4 | 64 | 0.1 | 1449875 |
| v17b | hidden | 4 | 64 | 0.1 | 1449895 |
| v17c | certainty | 4 | 64 | 0.1 | 1449896 |

**v17 results**:

| Config | Mode | Regular | Irregular | Balanced | Wug reg | Wug irreg |
|---|---|---|---|---|---|---|
| **v17a** | **input** | **96.4%** | **42.1%** | **69.2%** | **43/90** | **9/90** |
| v17c | certainty | 92.2% | 31.6% | 61.9% | 42/90 | 12/90 |
| v17b | hidden | 29.3% | 36.8% | 33.1% | 18/90 | 10/90 |

v17b: hidden-state confidence suffers train/eval mismatch. v17c: certainty collapses to one neuron (bal=0.01). Input-based confidence wins. v17d (8 neurons) hurt irregulars (10.5%) via majority dilution.

### 45. v18: Population decoder improvements (2026-03-31)

Key learnings applied: small experts (32) force abstraction (v16e finding), neuron dropout for synaptic pruning.

| Config | Neurons | expert_hidden | neuron_dropout | Job |
|---|---|---|---|---|
| v18a | 4 | 32 | 0.0 | 1449931 |
| v18b | 4 | 64 | 0.5 | 1449932 |

**v17d/v18 results**: v17a still best population model (69.2%). All variations hurt:
- v17d (8 neurons): 54.4% — majority dilution
- v18a (small neurons): 64.4% — less capacity without specialization benefit
- v18b (neuron dropout): 66.7% — fights collaborative strength
- v18c (tau annealing): 66.7% — sharpening hurts, best at collaborative phase

**Conclusion**: v17a (4 neurons, h=64, τ=1.0) is the population coding ceiling. Soft blending IS the strength. v16e (Gumbel MoE, 71.2%) remains overall best.

### 46. Expert/neuron analysis (2026-03-31)

v16e: complete expert collapse — all verbs to Expert 0. MoE is effectively a single decoder.
v17a: 2 of 4 neurons active. N0 fires stronger for irregulars (0.65 vs 0.46). Some differentiation but not rules.

### 47. v16e stability test — 5 seeds (2026-03-31)

Jobs 1450164-1450168, seeds 1-5. Verifying 71.2% is robust.

### 48. v19: Edit transducer (2026-03-31)

Predicts edit operations (COPY, DELETE, INSERT, SUB) instead of characters. Inductive bias: morphology = mostly copying with local edits. No phases needed.

| Config | Architecture | Job |
|---|---|---|
| v19a | single EditDecoder | 1450179 |
| v19b | 4 PopulationEditNeurons | 1450186 |

**All v19 failed:**
- v19a/b (sequential edits): pointer drift at inference, 0% accuracy
- v19c (per-position labeling): severe overfitting, 0% regular, 26.3% irregular
- v19d (scheduled sampling): didn't fix pointer drift, 0% accuracy

Edit transduction concept is sound but needs fundamental rework.

### 49. v20a: Monotonic copy decoder (2026-03-31)

Explicit copy gate + hard monotonic pointer. Result: 13.3% balanced — worse than implicit Bahdanau copy. Explicit copy mechanism adds complexity without benefit.

### 50. v17a vs v10a stability comparison — 5 seeds each (2026-03-31)

The key scientific comparison: population coding (v17a) vs K&C baseline (v10a), same data (oversample), 5 seeds each. Tests whether population coding's inductive bias provides a statistically significant advantage.

**Results (5 seeds each):**

| Model | Params | Mean Balanced | Std |
|---|---|---|---|
| v17a (population) | 721K | 68.8% | ±9.7% |
| v10a-large (3L LSTM) | 715K | 67.7% | ±7.3% |
| v10a (2L LSTM) | 583K | 67.5% | ±6.6% |

**No significant difference.** Population coding, MoE, and single LSTM all perform comparably when properly controlled. The architectural innovations don't improve over the K&C baseline — oversampled irregulars is the key ingredient, not the decoder architecture.

**v16e stability**: Mean 71.1% ± 2.8% across 5 seeds. Robust.

### 43. Known Issues

- **PyTorch version**: Must use `torch==2.4.0+cu121`, not latest. The HPC CUDA driver (12.8) is too old for PyTorch >=2.11.
- **Nested tensor warning**: Harmless `UserWarning` from `nn.TransformerEncoder` when using `src_key_padding_mask`. Can be suppressed with `warnings.filterwarnings` if desired.
- **CPU training is very slow**: ~7+ minutes per epoch on login node. Always use SLURM GPU partition for training.
