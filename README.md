# Applying Slot Attention to Past Tense

LING 3850 final project — Alan Xie, Henry Zhang, Jacob Liao.

We apply **Slot Attention** (and several alternatives) to English past-tense
morphology, framed as a character-level transduction task: given a present-tense
verb, generate its past-tense form (e.g. `play → played`, `go → went`, `wug → wugged`).
The central question is whether a slot-style bottleneck encourages a model to
learn an *abstract* past-tense rule (the regular `-ed` suffixation) better than
prior RNN/Transformer baselines.

The dataset and evaluation protocol follow Kirov & Cotterell (2018), including
the Albright & Hayes nonce ("wug") verbs for generalization.

## TL;DR results

We tested ~30 architectural variants. The headline findings:

1. **Slot attention does not help on this task.** A 288-run paired sweep (v20)
   over `{slots, no-slots} × seed × dropout × d_model × mono_alpha_init × {natural,
   oversample}` finds slots hurt the v6 architecture by 1–2 pp balanced accuracy,
   robust across hparams and seeds (paired t-test p < 0.001 in both regimes).
2. **Oversampling irregulars matters more than the architecture.** +4.5 pp
   balanced from regime change alone, holding architecture fixed.
3. **The single-LSTM, MoE-LSTM, and population-decoder ceilings all agree** at
   ~67–71% balanced — no architectural innovation has cleared the irregular bar
   on this corpus by a wide margin.

| Model | Architecture | Reg | Irreg | Balanced |
|---|---|---|---|---|
| **v16e** | biLSTM enc + 4 MoE LSTM decoders (Gumbel + diversity) | 95.1% | 47.4% | **71.2%** |
| v21 best (single run) | v6 + R&M phase-1=30 oversample, no slots | 94.4% | 43.8% | 69.05% |
| v17a | biLSTM enc + 4-neuron population decoder | 96.4% | 42.1% | 69.2% |
| v20 oversample best | Transformer + copy + monotonic, no slots | 93.8% | 43.8% | 68.8% |
| v13b | biLSTM + LSTM + slots + bottleneck (oversampled) | 85.5% | 31.6% | 58.5% |
| v6 | Transformer + copy + monotonic alignment bias | 96.4% | 15.8% | 56.1% |
| v10a | biLSTM + LSTM (no slots) | 99.5% | 5.3% | 52.4% |

Balanced accuracy = `(regular_acc + irregular_acc) / 2`. See `progress.md` and
`CLAUDE.md` for the full experiment log.

### Recent directions (v22–v30)

After the v20 sweep settled the slot-vs-no-slot question, subsequent work
explores stronger inductive biases for irregulars:

- **v21 — Rumelhart & McClelland two-phase curriculum.** Train on a small
  balanced ~270-verb vocabulary first, then expand. The classic overregularization
  dip *is* observable in individual runs at the phase transition, but is washed
  out by per-epoch variance on the small (~16-irregular) val set. Short phase-1
  (30 ep) on oversample is the only cell that beats baseline.
- **v22 — Hard Monotonic Neural Transducer (HMNT)** after Aharoni & Goldberg
  2017: discrete `{STEP, END, WRITE(c)}` actions over an explicit source pointer,
  trained with **DAgger β-mixing** (Makarov & Clematide 2018) to fix exposure
  bias. New `decoder_type: "transducer"`.
- **v23–v30 — Retrieval-augmented decoders.** Bolt a k-NN exemplar memory onto
  the decoder (Pinker dual-route hypothesis). Phonological-edit-distance kNN
  over training (src, tgt) pairs, with class-conditional and cluster-conditional
  variants (`use_retrieval`, `use_class_retrieval`, learned cluster predictor).

## Repository layout

```
configs/                 YAML configs for every run (v1 ... v30)
  sweep_v20{,_oversample}/   v20 paired-sweep configs (288 runs total)
  sweep_v21/                  R&M two-phase curriculum sweep
  sweep_v29/                  TF + retrieval ablation
data/                    RevisitPinkerAndPrince dataset (Kirov & Cotterell 2018)
src/
  data/
    vocab.py             DISC + CELEXmod char vocab (with wug→DISC remap)
    dataset.py           Dataset, regime utils, multi-shape collate_fn
    retrieval.py         Phonological-edit-distance kNN over train pairs
                         (class-conditional + cluster-conditional variants)
  model/
    encoder.py           TransformerCharEncoder, BiLSTMEncoder
    slot_attention.py    Multi-head slot attention bottleneck
    l0drop.py            Static / conditional L0, TopK, Gumbel routers
    decoder.py           Transformer decoder + copy + monotonic bias; LSTM+Bahdanau
    moe_decoder.py       Mixture of expert LSTM decoders
    population_decoder.py Self-gating population-coded decoder
    edit_decoder.py      v19 edit transducer (legacy)
    transducer_actions.py HMNT action vocab + NW oracle aligner + DAgger oracle
    transducer_decoder.py HMNT LSTM decoder with explicit source pointer
    full_model.py        SlotAttentionTransducer — wires everything together
  train.py               Training entry (TF / DAgger / R&M curriculum / retrieval)
  evaluate.py            Test + wug eval, retrieval-aware
scripts/
  sweep_v20.py / aggregate_v20.py     v20 sweep generator + aggregator
  sweep_v21.py / aggregate_v21.py     v21 R&M sweep + U-shape plotter
  slurm_jobs_v{20,20_oversample,21,22,29}/  generated SBATCH scripts
  verify_*.py                         CPU smoke tests for each new pathway
  inspect_*.py                        per-run error analysis
results/                Loss curves + history.json per run (no checkpoints)
progress.md             Detailed experiment log (chronological)
CLAUDE.md               Architecture reference + design notes
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch >= 2.0 is required; a GPU is strongly recommended for training.

## Quickstart

```bash
# Train with the default config
cd src && python train.py --config ../configs/default.yaml

# Train the best MoE model (v16e — 71.2% balanced)
cd src && python train.py --config ../configs/v16e_moe_gumbel_diversity.yaml

# Train the HMNT transducer with DAgger
cd src && python train.py --config ../configs/v22b_transducer_dagger.yaml

# Train the retrieval-augmented HMNT
cd src && python train.py --config ../configs/v23a_retrieval_hmnt.yaml

# Evaluate a checkpoint
cd src && python evaluate.py \
    --checkpoint ../checkpoints/best_model.pt \
    --data_path ../data/RevisitPinkerAndPrince/experiment_1/english_merged.txt \
    --wug_dir ../data/RevisitPinkerAndPrince/experiment_1_wugs/
```

On a SLURM cluster:

```bash
sbatch scripts/run_experiment.sh                              # default config
sbatch scripts/run_experiment.sh configs/v16e_moe_gumbel_diversity.yaml

# Generate + submit a full sweep
python scripts/sweep_v20.py        # v20: 288-run paired slot-vs-no-slot sweep
python scripts/sweep_v21.py        # v21: R&M two-phase curriculum
python scripts/aggregate_v20.py    # aggregate results into CSVs
```

## Architecture summary

The full pipeline is:

```
Encoder → [SlotAttention → L0Drop] → Decoder
                                       ↑
                  optional: kNN retrieval memory (v23+)
```

| Stage | Module | File |
|---|---|---|
| Encoder | `TransformerCharEncoder` / `BiLSTMEncoder` | `src/model/encoder.py` |
| Slot bottleneck (optional) | `SlotAttentionModule` | `src/model/slot_attention.py` |
| L0 sparsity (optional) | `L0Drop` / `InputConditionalL0Drop` / `TopKDrop` / `GumbelSlotRouter` | `src/model/l0drop.py` |
| Decoder | `TransformerCharDecoder` / `LSTMDecoder` / `MoEDecoder` / `PopulationDecoder` / `EditDecoder` / `TransducerDecoder` | `src/model/{decoder,moe_decoder,population_decoder,edit_decoder,transducer_decoder}.py` |
| Retrieval memory (optional) | `RetrievalIndex` (kNN, class-/cluster-conditional) | `src/data/retrieval.py` |
| Glue | `SlotAttentionTransducer` | `src/model/full_model.py` |

Decoder variants are switched via `decoder_type` in the YAML
(`transformer` / `lstm` / `moe_lstm` / `population` / `edit` / `transducer`);
slot attention is toggled with `use_slots`. Retrieval is toggled with
`use_retrieval` (+ optional `use_class_retrieval`, `retrieval_mode: cluster`,
or a learned cluster predictor with `lambda_cluster > 0`).

The HMNT (`transducer`) decoder is trained with optional **DAgger β-mixing**:
config keys `dagger_start_epoch`, `dagger_beta_max`, `dagger_anneal_epochs`.
The R&M two-phase curriculum is enabled with `phase1_epochs` and
`phase1_regime`; per-epoch reg/irreg val accuracy is dumped with
`track_val_split: true`.

See `CLAUDE.md` for the design rationale behind each variant (slot multi-head,
copy mechanism + monotonic alignment bias, MoE routing modes, population coding,
edit transducer, HMNT + DAgger, retrieval).

## Dataset

We use the `experiment_1` split from
[Kirov & Cotterell (2018)](https://github.com/ckirov/RevisitPinkerAndPrince):
4,039 (present, past) verb pairs with regular/irregular labels, encoded in
DISC phonological transcription. The 58 Albright & Hayes nonce verbs from
`experiment_1_wugs/` are used as a held-out generalization probe; we convert
their CELEXmod transcription to DISC at load time
(`_wug_to_disc` in `src/data/dataset.py`).

Split: 80% train / 10% val / 10% test (deterministic, seeded).

## References

- Locatello et al. (2020), *Object-Centric Learning with Slot Attention*
- Behjati & Henderson, *Dynamic Capacity Slot Attention*
- Kirov & Cotterell (2018), *Recurrent Neural Networks in Linguistic Theory*
- Albright & Hayes (2003), wug-test stimuli
- Ma & Gao (2022), Transformer evaluation on past-tense inflection
- Corkery et al. (2019), nonce-verb correlation analysis
- Rumelhart & McClelland (1986), *On Learning the Past Tenses of English Verbs*
  — two-phase curriculum + the U-shape overregularization phenomenon (v21)
- Aharoni & Goldberg (2017), *Morphological Inflection Generation with Hard
  Monotonic Attention* — HMNT action set + pointer (v22)
- Makarov & Clematide (2018), *Imitation Learning for Neural Morphological
  String Transduction* — DAgger β-mixing (v22)
- Pinker & Prince (1988), dual-route theory of inflection — motivation for
  retrieval-augmented decoders (v23+)
- See et al. (2017), *Get To The Point* — pointer / copy mechanism (v4+)

## Authors

Alan Xie, Henry Zhang, Jacob Liao — LING 3850, Yale University.
