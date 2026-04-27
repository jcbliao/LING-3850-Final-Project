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

We tested ~20 architectural variants. The headline finding is that slot
attention by itself provides no benefit on this task, but a **mixture of expert
LSTM decoders** (which replaces the slot bottleneck with K competing decoders)
delivers the best balanced accuracy.

| Model | Architecture | Reg | Irreg | Balanced |
|---|---|---|---|---|
| **v16e** | biLSTM enc + 4 MoE LSTM decoders (Gumbel + diversity) | 95.1% | 47.4% | **71.2%** |
| v17a | biLSTM enc + 4-neuron population decoder | 96.4% | 42.1% | 69.2% |
| v13b | biLSTM + LSTM + slots + bottleneck (oversampled) | 85.5% | 31.6% | 58.5% |
| v6 | Transformer + copy + monotonic alignment bias | 96.4% | 15.8% | 56.1% |
| v10a | biLSTM + LSTM (no slots) | 99.5% | 5.3% | 52.4% |

Balanced accuracy = `(regular_acc + irregular_acc) / 2`. See `progress.md` and
`CLAUDE.md` for the full experiment log.

## Repository layout

```
configs/         YAML configs for every run (v1 ... v20)
data/            RevisitPinkerAndPrince dataset (Kirov & Cotterell 2018)
src/
  data/          Vocabulary, dataset, training-regime utilities
  model/         Encoders, decoders, slot attention, MoE, population, edit
  train.py       Training entry point
  evaluate.py    Test + wug evaluation
scripts/         SLURM submission scripts and analysis utilities
results/         Loss curves and history.json per run (no checkpoints)
progress.md      Detailed experiment log (chronological)
CLAUDE.md        Architecture reference + design notes
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

# Train the current best model
cd src && python train.py --config ../configs/v16e_moe_gumbel_diversity.yaml

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
```

## Architecture summary

The full pipeline is:

```
Encoder → [SlotAttention → L0Drop] → Decoder
```

| Stage | Module | File |
|---|---|---|
| Encoder | `TransformerCharEncoder` / `BiLSTMEncoder` | `src/model/encoder.py` |
| Slot bottleneck (optional) | `SlotAttentionModule` | `src/model/slot_attention.py` |
| L0 sparsity (optional) | `L0Drop` / `InputConditionalL0Drop` / `TopKDrop` / `GumbelSlotRouter` | `src/model/l0drop.py` |
| Decoder | `TransformerCharDecoder` / `LSTMDecoder` / `MoEDecoder` / `PopulationDecoder` / `EditDecoder` | `src/model/{decoder,moe_decoder,population_decoder,edit_decoder}.py` |
| Glue | `SlotAttentionTransducer` | `src/model/full_model.py` |

Decoder variants are switched via `decoder_type` in the YAML
(`transformer` / `lstm` / `moe_lstm` / `population` / `edit`); slot attention
is toggled with `use_slots`.

See `CLAUDE.md` for the design rationale behind each variant (slot multi-head,
copy mechanism + monotonic alignment bias, MoE routing modes, population coding,
edit transducer, etc.).

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

## Authors

Alan Xie, Henry Zhang, Jacob Liao — LING 3850, Yale University.
