"""v21 two-phase (Rumelhart & McClelland 1986) training sweep.

Phase 1: small balanced vocabulary (~270 verbs, ~50% irregular).
Phase 2: full training set (natural ~3,231 or oversample ~6,200).

Grid:
    use_slots        : True, False                       (2)
    phase1_epochs    : 0 (baseline), 30, 60, 100         (4)
    phase2_regime    : natural, oversample               (2)
    seed             : 1, 2, 3                           (3)
Total: 48 runs.

Architecture frozen at v20 best: Transformer + copy + monotonic bias,
d_model=128, dropout=0.3, mono_alpha_init=0.5.

Usage:
    python scripts/sweep_v21.py             # write configs + submit
    python scripts/sweep_v21.py --dry-run   # write configs only
"""

import argparse
import itertools
import subprocess
from pathlib import Path

import yaml

PROJECT_DIR = Path("/gpfs/radev/project/kuan/jl3795/slot_attention")

GRID = {
    "use_slots": [True, False],
    "phase1_epochs": [0, 30, 60, 100],
    "phase2_regime": ["natural", "oversample"],
    "seed": [1, 2, 3],
}

BASE = {
    "data_path": "../data/RevisitPinkerAndPrince/experiment_1/english_merged.txt",
    "use_phonological": True,
    # frozen v20 best
    "d_model": 128,
    "nhead": 4,
    "enc_layers": 2,
    "dec_layers": 2,
    "d_ff": 256,
    "dropout": 0.3,
    "use_copy": True,
    "mono_alpha_init": 0.5,
    # slot defaults (only used when use_slots: true)
    "num_slots": 4,
    "slot_iters": 3,
    "slot_nhead": 4,
    "l0_mode": "conditional",
    "target_l0": 3.0,
    "lagrangian_lr": 0.01,
    "l0_beta": 0.66,
    "alpha_recon": 0.0,
    # two-phase
    "phase1_regime": "balanced",
    # training
    "batch_size": 32,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-4,
    "epochs": 200,
    "patience": 25,
    "plot_every": 10,
    "track_val_split": True,
}

SLURM = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output={project_dir}/logs/v21_sweep/%x_%j.out

PROJECT_DIR={project_dir}
mkdir -p "${{PROJECT_DIR}}/logs/v21_sweep"

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


def make_name(p):
    sl = "slots" if p["use_slots"] else "noslot"
    if p["phase1_epochs"] == 0:
        return f"v21_baseline_{p['phase2_regime']}_{sl}_s{p['seed']}"
    return (f"v21_p1{p['phase1_epochs']}_{p['phase2_regime']}_"
            f"{sl}_s{p['seed']}")


def build(p):
    cfg = dict(BASE)
    cfg["seed"] = p["seed"]
    cfg["use_slots"] = p["use_slots"]
    cfg["phase1_epochs"] = p["phase1_epochs"]
    cfg["training_regime"] = p["phase2_regime"]  # phase 2
    name = make_name(p)
    cfg["save_dir"] = f"../checkpoints/{name}"
    cfg["results_dir"] = f"../results/{name}"
    return name, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg_dir = PROJECT_DIR / "configs" / "sweep_v21"
    slurm_dir = PROJECT_DIR / "scripts" / "slurm_jobs_v21"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "logs" / "v21_sweep").mkdir(parents=True, exist_ok=True)

    keys = list(GRID.keys())
    runs = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(keys, vals))
        name, cfg = build(p)
        runs.append((name, cfg))
    print(f"Total runs: {len(runs)}")
    if args.limit:
        runs = runs[: args.limit]
        print(f"Limited to first {len(runs)}")

    submitted = []
    for name, cfg in runs:
        cfg_path = cfg_dir / f"{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
        slurm_path = slurm_dir / f"{name}.sh"
        slurm_path.write_text(SLURM.format(
            job_name=name,
            project_dir=str(PROJECT_DIR),
            config_rel=str(cfg_path.relative_to(PROJECT_DIR)),
        ))
        if args.dry_run:
            print(f"  [dry] {name}")
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
        manifest = slurm_dir / "manifest.txt"
        with open(manifest, "w") as f:
            for name, jid in submitted:
                f.write(f"{jid}\t{name}\n")
        print(f"\nManifest: {manifest}")
        print(f"Submitted {len(submitted)} jobs.")


if __name__ == "__main__":
    main()
