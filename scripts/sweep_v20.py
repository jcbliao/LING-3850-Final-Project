"""v20 final-architecture sweep launcher.

Architecture: Transformer + copy + monotonic bias (v6 family).
Grid:
    slots          : with_slots, no_slots          (2)
    seed           : 1, 2, 3                       (3)
    mono_alpha_init: 0.25, 0.5, 1.0, 2.0           (4)
    dropout        : 0.1, 0.2, 0.3                 (3)
    d_model        : 64, 128                       (2)

Total: 144 runs.

Usage:
    python scripts/sweep_v20.py                       # natural regime, submit
    python scripts/sweep_v20.py --regime oversample   # oversample regime
    python scripts/sweep_v20.py --dry-run             # write configs only
    python scripts/sweep_v20.py --limit 10            # only submit first N
    python scripts/sweep_v20.py --depend-on-manifest scripts/slurm_jobs_v20/manifest.txt
                                                       # chain after another sweep
"""

import argparse
import itertools
import subprocess
from pathlib import Path

import yaml


PROJECT_DIR = Path("/gpfs/radev/project/kuan/jl3795/slot_attention")

GRID = {
    "use_slots": [True, False],
    "seed": [1, 2, 3],
    "mono_alpha_init": [0.25, 0.5, 1.0, 2.0],
    "dropout": [0.1, 0.2, 0.3],
    "d_model": [64, 128],
}

BASE = {
    "data_path": "../data/RevisitPinkerAndPrince/experiment_1/english_merged.txt",
    "use_phonological": True,

    # architecture
    "nhead": 4,
    "enc_layers": 2,
    "dec_layers": 2,
    "d_ff": 256,
    "use_copy": True,

    # slot defaults (only used when use_slots: true)
    "num_slots": 4,
    "slot_iters": 3,
    "slot_nhead": 4,
    "l0_mode": "conditional",
    "target_l0": 3.0,
    "lagrangian_lr": 0.01,
    "l0_beta": 0.66,

    "alpha_recon": 0.0,

    # training
    "batch_size": 32,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-4,
    "epochs": 200,
    "patience": 25,
    "plot_every": 10,
}

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu,gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output={project_dir}/logs/v20_sweep/%x_%j.out

PROJECT_DIR={project_dir}
mkdir -p "${{PROJECT_DIR}}/logs/v20_sweep"

module load miniconda
conda activate slot_attention

cd "${{PROJECT_DIR}}/src"
python train.py --config "${{PROJECT_DIR}}/{config_rel}" 2>&1

# Evaluation
SAVE_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${{PROJECT_DIR}}/{config_rel}')); print(c.get('save_dir', '../checkpoints'))")
CKPT_PATH="${{PROJECT_DIR}}/src/${{SAVE_DIR}}/best_model.pt"

echo ""
echo "=== Evaluation ==="
python evaluate.py \\
    --checkpoint "${{CKPT_PATH}}" \\
    --data_path "${{PROJECT_DIR}}/data/RevisitPinkerAndPrince/experiment_1/english_merged.txt" \\
    --wug_dir "${{PROJECT_DIR}}/data/RevisitPinkerAndPrince/experiment_1_wugs/"
"""


def fmt(v):
    if isinstance(v, float):
        # keep 0.25 -> "0p25", 1.0 -> "1p0", 0.1 -> "0p1"
        s = f"{v:g}".replace(".", "p")
        return s
    return str(v)


def make_run_name(p, regime):
    slots_tag = "slots" if p["use_slots"] else "noslot"
    regime_tag = "" if regime == "natural" else f"_{regime}"
    return (
        f"v20{regime_tag}_{slots_tag}"
        f"_d{p['d_model']}"
        f"_dr{fmt(p['dropout'])}"
        f"_a{fmt(p['mono_alpha_init'])}"
        f"_s{p['seed']}"
    )


def build_config(p, regime):
    cfg = dict(BASE)
    cfg["seed"] = p["seed"]
    cfg["use_slots"] = p["use_slots"]
    cfg["d_model"] = p["d_model"]
    cfg["dropout"] = p["dropout"]
    cfg["mono_alpha_init"] = p["mono_alpha_init"]
    cfg["training_regime"] = regime

    name = make_run_name(p, regime)
    cfg["save_dir"] = f"../checkpoints/{name}"
    cfg["results_dir"] = f"../results/{name}"
    return name, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["natural", "oversample", "balanced",
                                          "irregular_first"],
                    default="natural",
                    help="Training regime (default: natural).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only generate/submit first N runs (for smoke testing).")
    ap.add_argument("--no-submit", action="store_true",
                    help="Write configs+slurm scripts but skip sbatch.")
    ap.add_argument("--depend-on-manifest", default=None,
                    help="Path to a manifest.txt; oversample sweep won't start "
                         "until those jobids finish (afterany).")
    args = ap.parse_args()

    regime = args.regime
    suffix = "" if regime == "natural" else f"_{regime}"
    cfg_dir = PROJECT_DIR / "configs" / f"sweep_v20{suffix}"
    slurm_dir = PROJECT_DIR / "scripts" / f"slurm_jobs_v20{suffix}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "logs" / f"v20_sweep{suffix}").mkdir(parents=True, exist_ok=True)

    # Build dependency string from a prior manifest if requested
    depend_arg = []
    if args.depend_on_manifest:
        manifest_path = Path(args.depend_on_manifest)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_DIR / manifest_path
        jids = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                jids.append(line.split()[0])
        depend_arg = [f"--dependency=afterany:{':'.join(jids)}"]
        print(f"Depending on {len(jids)} jobids from {manifest_path}")

    keys = list(GRID.keys())
    runs = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(keys, vals))
        name, cfg = build_config(p, regime)
        runs.append((name, cfg))

    print(f"Regime: {regime} | Total runs: {len(runs)}")
    if args.limit:
        runs = runs[: args.limit]
        print(f"Limited to first {len(runs)} runs")

    # Patch the slurm template to include the v20 sweep log subdir for this regime
    log_subdir = f"v20_sweep{suffix}"
    slurm_template = SLURM_TEMPLATE.replace("logs/v20_sweep/", f"logs/{log_subdir}/")

    submitted = []
    for name, cfg in runs:
        cfg_path = cfg_dir / f"{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)

        slurm_path = slurm_dir / f"{name}.sh"
        slurm_path.write_text(slurm_template.format(
            job_name=name,
            project_dir=str(PROJECT_DIR),
            config_rel=str(cfg_path.relative_to(PROJECT_DIR)),
        ))

        if args.dry_run or args.no_submit:
            print(f"  [dry] {name}")
            continue

        sbatch_cmd = ["sbatch"] + depend_arg + [str(slurm_path)]
        result = subprocess.run(
            sbatch_cmd,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            jid = result.stdout.strip().split()[-1]
            print(f"  submitted {name}: {jid}")
            submitted.append((name, jid))
        else:
            print(f"  FAILED {name}: {result.stderr.strip()}")

    if submitted:
        manifest = slurm_dir / "manifest.txt"
        with open(manifest, "w") as f:
            for name, jid in submitted:
                f.write(f"{jid}\t{name}\n")
        print(f"\nManifest written: {manifest}")
        print(f"Submitted {len(submitted)} jobs.")
        print("Monitor: squeue -u $USER")


if __name__ == "__main__":
    main()
