"""Hyperparameter sweep launcher.

Generates YAML configs and submits SLURM jobs for each combination.

Usage:
    # Full hyperparameter sweep (27 jobs)
    python scripts/sweep.py

    # Training regime experiments only (with specified best hyperparams)
    python scripts/sweep.py --regime-only --num-slots 8 --lambda-l0 0.01 --alpha-recon 0.0

    # Dry run (print configs without submitting)
    python scripts/sweep.py --dry-run
"""

import argparse
import itertools
import subprocess
import yaml
from pathlib import Path


BASE_CONFIG = {
    "data_path": "../data/RevisitPinkerAndPrince/experiment_1/english_merged.txt",
    "use_phonological": True,
    "seed": 42,
    "d_model": 128,
    "nhead": 4,
    "enc_layers": 3,
    "dec_layers": 3,
    "d_ff": 256,
    "slot_iters": 3,
    "dropout": 0.1,
    "l0_beta": 0.66,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "plot_every": 10,
}

SWEEP_GRID = {
    "num_slots": [4, 8, 16],
    "lambda_l0": [0.001, 0.01, 0.1],
    "alpha_recon": [0.0, 0.5, 1.0],
}

TRAINING_REGIMES = ["natural", "balanced", "irregular_first"]

PROJECT_DIR = "/gpfs/radev/project/kuan/jl3795/slot_attention"

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00

PROJECT_DIR=""" + PROJECT_DIR + """

mkdir -p "${{PROJECT_DIR}}/logs"

module load miniconda
conda activate slot_attention

cd "${{PROJECT_DIR}}/src"
python train.py --config "${{PROJECT_DIR}}/{config_path}" 2>&1 | tee "${{PROJECT_DIR}}/logs/{job_name}_${{SLURM_JOB_ID}}.log"
"""


def generate_sweep_configs(output_dir: Path):
    """Generate configs for full hyperparameter grid."""
    configs = []
    keys = list(SWEEP_GRID.keys())
    for values in itertools.product(*SWEEP_GRID.values()):
        params = dict(zip(keys, values))
        name = f"s{params['num_slots']}_l{params['lambda_l0']}_a{params['alpha_recon']}"

        config = BASE_CONFIG.copy()
        config.update(params)
        config["save_dir"] = f"../checkpoints/sweep_{name}"
        config["results_dir"] = f"../results/sweep_{name}"
        config["training_regime"] = "natural"

        config_path = output_dir / f"sweep_{name}.yaml"
        configs.append((name, config, config_path))

    return configs


def generate_regime_configs(output_dir: Path, num_slots: int,
                            lambda_l0: float, alpha_recon: float):
    """Generate configs for training regime experiments."""
    configs = []
    for regime in TRAINING_REGIMES:
        name = f"regime_{regime}"

        config = BASE_CONFIG.copy()
        config["num_slots"] = num_slots
        config["lambda_l0"] = lambda_l0
        config["alpha_recon"] = alpha_recon
        config["training_regime"] = regime
        config["save_dir"] = f"../checkpoints/{name}"
        config["results_dir"] = f"../results/{name}"

        config_path = output_dir / f"{name}.yaml"
        configs.append((name, config, config_path))

    return configs


def submit_job(name: str, config_path: Path, dry_run: bool = False):
    """Write SLURM script and submit."""
    slurm_dir = Path("scripts/slurm_jobs")
    slurm_dir.mkdir(parents=True, exist_ok=True)

    slurm_script = slurm_dir / f"{name}.sh"
    slurm_script.write_text(SLURM_TEMPLATE.format(
        job_name=name,
        config_path=config_path,
    ))

    if dry_run:
        print(f"  [dry-run] Would submit: {slurm_script}")
        return None

    result = subprocess.run(
        ["sbatch", str(slurm_script)],
        capture_output=True, text=True,
    )
    job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else "FAILED"
    print(f"  Submitted {name}: job {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Launch hyperparameter sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without submitting")
    parser.add_argument("--regime-only", action="store_true",
                        help="Only run training regime experiments")
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--lambda-l0", type=float, default=0.01)
    parser.add_argument("--alpha-recon", type=float, default=0.0)
    args = parser.parse_args()

    config_dir = Path("configs/sweep")
    config_dir.mkdir(parents=True, exist_ok=True)

    if args.regime_only:
        configs = generate_regime_configs(
            config_dir, args.num_slots, args.lambda_l0, args.alpha_recon,
        )
        print(f"Training regime experiments: {len(configs)} jobs")
    else:
        configs = generate_sweep_configs(config_dir)
        print(f"Hyperparameter sweep: {len(configs)} jobs")

    for name, config, config_path in configs:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        submit_job(name, config_path, dry_run=args.dry_run)

    print("\nDone. Monitor with: squeue -u $USER")


if __name__ == "__main__":
    main()
