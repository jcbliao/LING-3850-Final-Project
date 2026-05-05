"""Generate the v29 sweep — the headline experiments for the retrieval paper.

Three groups:
  (1) 4 cells x 5 seeds = 20 stability runs:
        Architecture in {Transformer+copy+mono, biLSTM+LSTM}
        x Retrieval in {none, class-conditional}
  (2) 2 random-retrieval negative controls:
        Each of the two retrieval cells gets a random-retrieval variant.
  (3) 2 ablations on biLSTM+LSTM+retrieval (the strongest cell):
        - edit-distance (no class) retrieval
        - natural training regime (vs oversample)

Run with `python scripts/sweep_v29.py` (also generates SLURM scripts and sbatches them).
Use `--dry-run` to preview without submitting.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path

import yaml

PROJECT_DIR = Path("/gpfs/radev/project/kuan/jl3795/slot_attention")
CFG_DIR = PROJECT_DIR / "configs" / "sweep_v29"
SLURM_DIR = PROJECT_DIR / "scripts" / "slurm_jobs_v29"
LOG_DIR = PROJECT_DIR / "logs" / "v29"

SEEDS = [1, 2, 3, 4, 5]

CELLS = {
    "tf_no":   dict(label="Transformer + copy + mono, no retrieval",
                    encoder_type="transformer", decoder_type="transformer",
                    use_copy=True, mono_alpha_init=0.5,
                    use_retrieval=False, lr=3.0e-4),
    "tf_retr": dict(label="Transformer + copy + mono + class retrieval",
                    encoder_type="transformer", decoder_type="transformer",
                    use_copy=True, mono_alpha_init=0.5,
                    use_retrieval=True, use_class_retrieval=True,
                    retrieval_mode="knn", lr=3.0e-4),
    "lstm_no":   dict(label="biLSTM + LSTM, no retrieval",
                      encoder_type="bilstm", decoder_type="lstm",
                      use_copy=False,
                      use_retrieval=False, lr=1.0e-3),
    "lstm_retr": dict(label="biLSTM + LSTM + class retrieval",
                      encoder_type="bilstm", decoder_type="lstm",
                      use_copy=False,
                      use_retrieval=True, use_class_retrieval=True,
                      retrieval_mode="knn", lr=1.0e-3),
}

EXTRA_RUNS = [
    # Random-retrieval controls (one per retrieval-on architecture)
    ("tf_rand_s1", dict(
        encoder_type="transformer", decoder_type="transformer",
        use_copy=True, mono_alpha_init=0.5,
        use_retrieval=True, use_class_retrieval=False,
        retrieval_mode="random", lr=3.0e-4, seed=1,
        label="Transformer + RANDOM retrieval (control)")),
    ("lstm_rand_s1", dict(
        encoder_type="bilstm", decoder_type="lstm",
        use_copy=False,
        use_retrieval=True, use_class_retrieval=False,
        retrieval_mode="random", lr=1.0e-3, seed=1,
        label="biLSTM + LSTM + RANDOM retrieval (control)")),
    # Edit-distance retrieval (no class) on the strongest cell
    ("lstm_edit_s1", dict(
        encoder_type="bilstm", decoder_type="lstm",
        use_copy=False,
        use_retrieval=True, use_class_retrieval=False,
        retrieval_mode="knn", lr=1.0e-3, seed=1,
        label="biLSTM + LSTM + edit-distance retrieval (no class)")),
    # Natural regime (no oversample) on the strongest cell
    ("lstm_retr_natural_s1", dict(
        encoder_type="bilstm", decoder_type="lstm",
        use_copy=False,
        use_retrieval=True, use_class_retrieval=True,
        retrieval_mode="knn", lr=1.0e-3, seed=1,
        training_regime="natural",
        label="biLSTM + LSTM + class retrieval, natural regime")),
]


BASE_CONFIG = {
    "data_path": "../data/RevisitPinkerAndPrince/experiment_1/english_merged.txt",
    "use_phonological": True,
    "training_regime": "oversample",
    "use_slots": False,
    "d_model": 128,
    "nhead": 4,
    "enc_layers": 2,
    "dec_layers": 2,
    "d_ff": 256,
    "dropout": 0.3,
    "lstm_hidden": 0,
    "batch_size": 32,
    "weight_decay": 1.0e-4,
    "epochs": 200,
    "patience": 30,
    "retrieval_k": 5,
    "plot_every": 10,
    "track_val_split": True,
}


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=v29_{name}
#SBATCH --partition=gpu,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output={log_dir}/%x_%j.out

PROJECT_DIR={project_dir}
mkdir -p "${{PROJECT_DIR}}/logs/v29"

unset CONDA_ENVS_PATH CONDA_PKGS_DIRS CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_SHLVL

module load miniconda
conda activate slot_attention

cd "${{PROJECT_DIR}}/src"
python train.py --config "${{PROJECT_DIR}}/{config_rel}" 2>&1

SAVE_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${{PROJECT_DIR}}/{config_rel}')); print(c.get('save_dir', '../checkpoints'))")
CKPT_PATH="${{PROJECT_DIR}}/src/${{SAVE_DIR}}/best_model.pt"

echo ""
echo "=== Evaluation ==="
python evaluate.py \\
    --checkpoint "${{CKPT_PATH}}" \\
    --data_path "${{PROJECT_DIR}}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt" \\
    --wug_dir "${{PROJECT_DIR}}/data/RevisitPinkerAndPrince/experiment_1_wugs/"
"""


def build_run(name: str, overrides: dict) -> dict:
    cfg = dict(BASE_CONFIG)
    cfg.update({k: v for k, v in overrides.items() if k != "label"})
    cfg["save_dir"] = f"../checkpoints/v29_{name}"
    cfg["results_dir"] = f"../results/v29_{name}"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    # Stability cells × seeds
    for cell_name, cell_overrides in CELLS.items():
        for seed in SEEDS:
            name = f"{cell_name}_s{seed}"
            overrides = {**cell_overrides, "seed": seed}
            runs.append((name, overrides))
    # Extra ablation runs
    for name, overrides in EXTRA_RUNS:
        runs.append((name, overrides))

    print(f"Total runs: {len(runs)}")

    submitted = []
    for name, overrides in runs:
        cfg = build_run(name, overrides)
        cfg_path = CFG_DIR / f"{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
        slurm_path = SLURM_DIR / f"{name}.sh"
        slurm_path.write_text(SLURM_TEMPLATE.format(
            name=name,
            project_dir=str(PROJECT_DIR),
            log_dir=str(LOG_DIR),
            config_rel=str(cfg_path.relative_to(PROJECT_DIR)),
        ))

        if args.dry_run:
            print(f"  [dry] {name}  ({overrides.get('label', '')})")
            continue

        r = subprocess.run(["sbatch", str(slurm_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            jid = r.stdout.strip().split()[-1]
            print(f"  submitted {name}: {jid}")
            submitted.append((name, jid))
        else:
            print(f"  FAILED {name}: {r.stderr.strip()}")

    if submitted:
        manifest = SLURM_DIR / "manifest.txt"
        with open(manifest, "w") as f:
            for name, jid in submitted:
                f.write(f"{jid}\t{name}\n")
        print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    main()
