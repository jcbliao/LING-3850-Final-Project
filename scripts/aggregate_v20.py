"""Aggregate v20 sweep results into a single CSV + summary table."""

import csv
import json
import re
from pathlib import Path

PROJECT = Path("/gpfs/radev/project/kuan/jl3795/slot_attention")
RESULTS = PROJECT / "results"

# eval prints (in slurm log):
#   regular: 0.9221 (385 examples)
#   irregular: 0.2000 (20 examples)
#   balanced_acc: 0.5610
#   Wug results: 52/90 match regular, 5/90 match irregular
RE_REG = re.compile(r"^regular:\s+([0-9.]+)\s+\((\d+)\s+examples\)", re.M)
RE_IRR = re.compile(r"^irregular:\s+([0-9.]+)\s+\((\d+)\s+examples\)", re.M)
RE_BAL = re.compile(r"^balanced_acc:\s+([0-9.]+)", re.M)
RE_ALL = re.compile(r"^all_test:\s+([0-9.]+)", re.M)
RE_WUG = re.compile(r"^Wug results:\s+(\d+)/(\d+)\s+match regular,\s+(\d+)/(\d+)\s+match irregular", re.M)

# parse run name like v20_slots_d128_dr0p1_a1_s1 or v20_oversample_noslot_d64_dr0p3_a2_s3
RE_RUN = re.compile(
    r"^v20(?:_(?P<regime>oversample|balanced|irregular_first))?"
    r"_(?P<slots>slots|noslot)"
    r"_d(?P<d_model>\d+)"
    r"_dr(?P<dropout>[0-9p]+)"
    r"_a(?P<alpha>[0-9p]+)"
    r"_s(?P<seed>\d+)$"
)


def unfmt(s):
    return float(s.replace("p", "."))


def find_log(run_name):
    # search both v20_sweep and v20_sweep_oversample
    for sub in ("v20_sweep", "v20_sweep_oversample"):
        d = PROJECT / "logs" / sub
        if not d.exists():
            continue
        matches = list(d.glob(f"{run_name}_*.out"))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def parse_run(run_dir):
    name = run_dir.name
    m = RE_RUN.match(name)
    if not m:
        return None
    parts = m.groupdict()
    rec = {
        "run": name,
        "regime": parts["regime"] or "natural",
        "slots": parts["slots"] == "slots",
        "d_model": int(parts["d_model"]),
        "dropout": unfmt(parts["dropout"]),
        "alpha": unfmt(parts["alpha"]),
        "seed": int(parts["seed"]),
    }

    # best val_loss from history.json
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        try:
            h = json.loads(hist_path.read_text())
            val_losses = h.get("val_loss", [])
            rec["best_val_loss"] = min(val_losses) if val_losses else None
            rec["epochs_run"] = len(h.get("train_loss", []))
        except Exception:
            rec["best_val_loss"] = None

    # eval metrics from slurm log
    log_path = find_log(name)
    if log_path:
        text = log_path.read_text(errors="ignore")
        if (m := RE_REG.search(text)):
            rec["reg_acc"] = float(m.group(1))
            rec["reg_n"] = int(m.group(2))
        if (m := RE_IRR.search(text)):
            rec["irr_acc"] = float(m.group(1))
            rec["irr_n"] = int(m.group(2))
        if (m := RE_BAL.search(text)):
            rec["balanced_acc"] = float(m.group(1))
        if (m := RE_ALL.search(text)):
            rec["all_acc"] = float(m.group(1))
        if (m := RE_WUG.search(text)):
            rec["wug_reg"] = int(m.group(1))
            rec["wug_reg_n"] = int(m.group(2))
            rec["wug_irr"] = int(m.group(3))
            rec["wug_irr_n"] = int(m.group(4))

    return rec


def main():
    rows = []
    for d in sorted(RESULTS.iterdir()):
        if d.is_dir() and d.name.startswith("v20_"):
            rec = parse_run(d)
            if rec:
                rows.append(rec)

    print(f"Parsed {len(rows)} runs")

    # write full CSV
    keys = ["run", "regime", "slots", "d_model", "dropout", "alpha", "seed",
            "best_val_loss", "epochs_run", "all_acc", "reg_acc", "irr_acc",
            "balanced_acc", "reg_n", "irr_n",
            "wug_reg", "wug_reg_n", "wug_irr", "wug_irr_n"]
    out_csv = PROJECT / "results" / "v20_sweep_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"Wrote {out_csv}")

    # summary: aggregate over seeds
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        if "balanced_acc" not in r:
            continue
        key = (r["regime"], r["slots"], r["d_model"], r["dropout"], r["alpha"])
        cells[key].append(r)

    summary_rows = []
    for key, runs in cells.items():
        regime, slots, d_model, dropout, alpha = key
        bals = [r["balanced_acc"] for r in runs]
        regs = [r["reg_acc"] for r in runs]
        irrs = [r["irr_acc"] for r in runs]
        wug_regs = [r.get("wug_reg", 0) for r in runs]
        wug_irrs = [r.get("wug_irr", 0) for r in runs]
        n = len(bals)
        mean = lambda xs: sum(xs) / len(xs)
        std = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
        summary_rows.append({
            "regime": regime, "slots": slots, "d_model": d_model,
            "dropout": dropout, "alpha": alpha, "n_seeds": n,
            "bal_mean": mean(bals), "bal_std": std(bals),
            "reg_mean": mean(regs), "reg_std": std(regs),
            "irr_mean": mean(irrs), "irr_std": std(irrs),
            "wug_reg_mean": mean(wug_regs),
            "wug_irr_mean": mean(wug_irrs),
        })

    summary_rows.sort(key=lambda r: -r["bal_mean"])
    out_csv2 = PROJECT / "results" / "v20_sweep_summary.csv"
    with open(out_csv2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"Wrote {out_csv2}")

    # Top-line print
    print("\n=== Top 10 cells by balanced_acc (mean ± std over 3 seeds) ===")
    print(f"{'regime':<11} {'slots':<6} {'d':<4} {'dr':<5} {'a':<5} "
          f"{'bal':>14} {'reg':>14} {'irr':>14}")
    for r in summary_rows[:10]:
        sl = "slots" if r["slots"] else "noslot"
        print(f"{r['regime']:<11} {sl:<6} {r['d_model']:<4} "
              f"{r['dropout']:<5} {r['alpha']:<5} "
              f"{r['bal_mean']:6.3f}±{r['bal_std']:6.3f}  "
              f"{r['reg_mean']:6.3f}±{r['reg_std']:6.3f}  "
              f"{r['irr_mean']:6.3f}±{r['irr_std']:6.3f}")

    # Slot vs no-slot grand summary, by regime
    print("\n=== Grand mean by regime × slots ===")
    grouped = defaultdict(list)
    for r in rows:
        if "balanced_acc" not in r:
            continue
        grouped[(r["regime"], r["slots"])].append(r)
    print(f"{'regime':<11} {'slots':<6} {'n':>3} {'bal_mean':>10} {'bal_std':>10} "
          f"{'reg':>10} {'irr':>10} {'wug_irr':>9}")
    for (regime, slots), runs in sorted(grouped.items()):
        bals = [r["balanced_acc"] for r in runs]
        regs = [r["reg_acc"] for r in runs]
        irrs = [r["irr_acc"] for r in runs]
        wug_irrs = [r.get("wug_irr", 0) for r in runs]
        n = len(bals)
        mean = lambda xs: sum(xs) / len(xs)
        std = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0
        sl = "slots" if slots else "noslot"
        print(f"{regime:<11} {sl:<6} {n:>3} {mean(bals):>10.3f} {std(bals):>10.3f} "
              f"{mean(regs):>10.3f} {mean(irrs):>10.3f} {mean(wug_irrs):>9.1f}")


if __name__ == "__main__":
    main()
