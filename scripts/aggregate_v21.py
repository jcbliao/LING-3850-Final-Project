"""Aggregate v21 two-phase sweep results + render U-shape plots."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path("/gpfs/radev/project/kuan/jl3795/slot_attention")
RESULTS = PROJECT / "results"

RE_REG = re.compile(r"^regular:\s+([0-9.]+)\s+\((\d+)\s+examples\)", re.M)
RE_IRR = re.compile(r"^irregular:\s+([0-9.]+)\s+\((\d+)\s+examples\)", re.M)
RE_BAL = re.compile(r"^balanced_acc:\s+([0-9.]+)", re.M)
RE_ALL = re.compile(r"^all_test:\s+([0-9.]+)", re.M)
RE_WUG = re.compile(r"^Wug results:\s+(\d+)/(\d+)\s+match regular,\s+(\d+)/(\d+)\s+match irregular", re.M)

# v21_baseline_natural_slots_s1 OR v21_p160_oversample_noslot_s3
RE_RUN = re.compile(
    r"^v21_(?:baseline|p1(?P<p1>\d+))_(?P<regime>natural|oversample)"
    r"_(?P<slots>slots|noslot)_s(?P<seed>\d+)$"
)


def find_log(name):
    matches = list((PROJECT / "logs" / "v21_sweep").glob(f"{name}_*.out"))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def parse_run(d):
    m = RE_RUN.match(d.name)
    if not m:
        return None
    g = m.groupdict()
    rec = {
        "run": d.name,
        "phase1_epochs": int(g["p1"]) if g["p1"] else 0,
        "regime": g["regime"],
        "slots": g["slots"] == "slots",
        "seed": int(g["seed"]),
    }
    hist = d / "history.json"
    if hist.exists():
        try:
            h = json.loads(hist.read_text())
            rec["history"] = h
            if h.get("val_loss"):
                rec["best_val_loss"] = min(h["val_loss"])
            rec["epochs_run"] = len(h.get("train_loss", []))
        except Exception:
            pass
    log = find_log(d.name)
    if log:
        text = log.read_text(errors="ignore")
        if (m := RE_REG.search(text)):
            rec["reg_acc"] = float(m.group(1)); rec["reg_n"] = int(m.group(2))
        if (m := RE_IRR.search(text)):
            rec["irr_acc"] = float(m.group(1)); rec["irr_n"] = int(m.group(2))
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
        if d.is_dir() and d.name.startswith("v21_"):
            r = parse_run(d)
            if r:
                rows.append(r)
    print(f"Parsed {len(rows)} runs")

    # CSV
    keys = ["run", "phase1_epochs", "regime", "slots", "seed",
            "best_val_loss", "epochs_run", "all_acc", "reg_acc", "irr_acc",
            "balanced_acc", "wug_reg", "wug_reg_n", "wug_irr", "wug_irr_n"]
    out = PROJECT / "results" / "v21_sweep_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"Wrote {out}")

    # Summary by (phase1_epochs, regime, slots) over seeds
    cells = defaultdict(list)
    for r in rows:
        if "balanced_acc" not in r:
            continue
        cells[(r["phase1_epochs"], r["regime"], r["slots"])].append(r)

    rows_sum = []
    for key, runs in sorted(cells.items()):
        p1, regime, slots = key
        bals = [r["balanced_acc"] for r in runs]
        regs = [r["reg_acc"] for r in runs]
        irrs = [r["irr_acc"] for r in runs]
        wug_regs = [r.get("wug_reg", 0) for r in runs]
        wug_irrs = [r.get("wug_irr", 0) for r in runs]
        n = len(bals)
        mean = lambda xs: sum(xs) / len(xs)
        std = lambda xs: (sum((x - mean(xs)) ** 2 for x in xs) / max(1, len(xs)-1)) ** 0.5 if len(xs) > 1 else 0.0
        rows_sum.append({
            "phase1_epochs": p1, "regime": regime, "slots": slots, "n": n,
            "bal_mean": mean(bals), "bal_std": std(bals),
            "reg_mean": mean(regs), "reg_std": std(regs),
            "irr_mean": mean(irrs), "irr_std": std(irrs),
            "wug_reg_mean": mean(wug_regs), "wug_irr_mean": mean(wug_irrs),
        })
    out2 = PROJECT / "results" / "v21_sweep_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sum[0].keys()))
        w.writeheader()
        for r in rows_sum:
            w.writerow(r)
    print(f"Wrote {out2}")

    # Print summary table
    print(f"\n{'p1':>4} {'regime':<11} {'slots':<6} {'n':>2} "
          f"{'bal':>14} {'reg':>14} {'irr':>14} {'wug_irr':>9}")
    for r in rows_sum:
        sl = "slots" if r["slots"] else "noslot"
        print(f"{r['phase1_epochs']:>4} {r['regime']:<11} {sl:<6} {r['n']:>2} "
              f"{r['bal_mean']:6.3f}±{r['bal_std']:6.3f}  "
              f"{r['reg_mean']:6.3f}±{r['reg_std']:6.3f}  "
              f"{r['irr_mean']:6.3f}±{r['irr_std']:6.3f}  "
              f"{r['wug_irr_mean']:>5.1f}/90")

    # ===== U-shape plots =====
    # For each (regime, slots), plot val_irr_acc vs epoch averaged over 3 seeds,
    # one line per phase1_epochs setting.
    plot_dir = PROJECT / "results" / "v21_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    by_combo = defaultdict(list)
    for r in rows:
        if "history" not in r:
            continue
        by_combo[(r["regime"], r["slots"])].append(r)

    for (regime, slots), combo_runs in by_combo.items():
        # Group by phase1 setting
        by_p1 = defaultdict(list)
        for r in combo_runs:
            by_p1[r["phase1_epochs"]].append(r)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        cmap = plt.get_cmap("viridis")
        p1_keys = sorted(by_p1.keys())
        for i, p1 in enumerate(p1_keys):
            color = cmap(i / max(1, len(p1_keys) - 1))
            for metric, ax, label in [("val_reg_acc", axes[0], "regular"),
                                       ("val_irr_acc", axes[1], "irregular")]:
                # Average curves over seeds (truncate to shortest)
                series = [r["history"].get(metric, []) for r in by_p1[p1]]
                series = [s for s in series if s]
                if not series:
                    continue
                L = min(len(s) for s in series)
                if L < 2:
                    continue
                mat = [[s[i] for s in series] for i in range(L)]
                avg = [sum(col) / len(col) for col in mat]
                epochs = list(range(1, L + 1))
                lab = "baseline" if p1 == 0 else f"p1={p1}"
                ax.plot(epochs, avg, color=color, label=lab, linewidth=1.5)
                if p1 > 0:
                    ax.axvline(p1, color=color, linestyle=":", alpha=0.4)

        for ax, title in zip(axes, ["regular", "irregular"]):
            ax.set_xlabel("epoch")
            ax.set_ylabel(f"val {title} accuracy")
            ax.set_title(f"val {title} (regime={regime}, {'slots' if slots else 'noslot'})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(-0.02, 1.02)

        fig.tight_layout()
        sl = "slots" if slots else "noslot"
        out_png = plot_dir / f"u_shape_{regime}_{sl}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
