"""Aggregate results/summary.jsonl into report tables and figures.

Usage: python src/analyze.py [--task fine] [--cell vanilla]
Outputs: results/analysis/{tables.md, effects.png, learning_curves.png, per_type_f1.png}

Seeds fix the data order and the embedding initialisation, so a given seed is "lucky" or "unlucky"
for every configuration alike. Each configuration is therefore compared to the baseline trained
with the SAME seed (paired difference Δ), which removes most of the seed noise. The best
configuration is chosen on the validation Δ among configurations that have all seeds.
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eda import INK_2, SERIES  # shared report style (rcParams set on import)
from sweep import BASELINE, STUDIES, run_name
from train import RESULTS

OUT = RESULTS / "analysis"
SEEDS = (1, 2, 3)


def load_summary() -> dict[str, dict]:
    rows = [json.loads(l) for l in open(RESULTS / "summary.jsonl") if l.strip()]
    return {r["run_name"]: r for r in rows}  # last write wins if a run was repeated


def setting(override: dict) -> str:
    diff = {k: v for k, v in override.items() if BASELINE.get(k) != v}
    return ", ".join(f"{k}={v}" for k, v in diff.items()) or "baseline"


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def config_stats(summary, task, cell, override) -> dict | None:
    runs = {s: summary.get(run_name(task, override, s, cell)) for s in SEEDS}
    base = {s: summary.get(run_name(task, {}, s, cell)) for s in SEEDS}
    done = [s for s in SEEDS if runs[s]]
    if not done:
        return None
    paired = [s for s in done if base[s]]
    pick = lambda key, src, seeds: [100 * src[s][key] for s in seeds]  # noqa: E731
    return {
        "setting": setting(override),
        "is_baseline": setting(override) == "baseline",
        "seeds": len(done),
        "val": mean_std(pick("val_f1", runs, done)),
        "test": mean_std(pick("test_f1", runs, done)),
        "macro": mean_std(pick("test_macro_f1", runs, done)),
        "d_val": mean_std([a - b for a, b in zip(pick("val_f1", runs, paired), pick("val_f1", base, paired))]),
        "d_test": mean_std([a - b for a, b in zip(pick("test_f1", runs, paired), pick("test_f1", base, paired))]),
        "d_test_seeds": [a - b for a, b in zip(pick("test_f1", runs, paired), pick("test_f1", base, paired))],
        "params": runs[done[0]]["params_total"],
        "params_recurrent": runs[done[0]]["params_recurrent"],
        "epoch_time": st.mean(runs[s]["mean_epoch_time_s"] for s in done),
        "best_epoch": st.mean(runs[s]["best_epoch"] for s in done),
        "override": override,
    }


def fmt(ms, signed=False):
    m, s = ms
    return f"{m:+.2f} ± {s:.2f}" if signed else f"{m:.2f} ± {s:.2f}"


def write_tables(tables, best, path):
    lines = [
        "# Hyperparameter studies — vanilla LSTM, FloraNER fine-grained",
        "",
        "Entity-level F1 (%), mean ± std over seeds. **Δ** = difference with the baseline trained with the",
        "same seed (paired), mean ± std over seeds. Choice on **validation**, **test** only reported.",
        "† = incomplete (fewer than 3 seeds): not eligible as best configuration.",
        "",
        f"**Best configuration (highest mean Δ val among complete ones): {best['setting']}** — "
        f"test F1 {fmt(best['test'])} (Δ {fmt(best['d_test'], True)}), macro F1 {fmt(best['macro'])}.",
        "",
    ]
    for study, rows in tables.items():
        if not rows:
            continue
        lines += [f"## {study}", "",
                  "| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) "
                  "| s/epoch | best epoch |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            name = r["setting"] + (" †" if r["seeds"] < len(SEEDS) else "") + (" **(best)**" if r is best else "")
            dv = "—" if r["is_baseline"] else fmt(r["d_val"], True)
            dt = "—" if r["is_baseline"] else fmt(r["d_test"], True)
            lines.append(f"| {name} | {r['seeds']} | {fmt(r['val'])} | {dv} | {fmt(r['test'])} | {dt} | "
                         f"{fmt(r['macro'])} | {r['params']:,} ({r['params_recurrent']:,}) | "
                         f"{r['epoch_time']:.1f} | {r['best_epoch']:.1f} |")
        lines.append("")
    path.write_text("\n".join(lines))


def fig_effects(tables, path):
    """Forest plot: paired Δ test F1 vs baseline for every non-baseline configuration."""
    rows = [(study, r) for study, rs in tables.items() for r in rs if not r["is_baseline"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(rows) + 1.4))
    ys = list(range(len(rows)))
    for y, (study, r) in zip(ys, rows):
        m, s = r["d_test"]
        ax.plot([m - s, m + s], [y, y], color=SERIES[0], lw=2, solid_capstyle="round")
        ax.plot(m, y, "o", color=SERIES[0], ms=7, mec="white", mew=1.5)
        for d in r["d_test_seeds"]:
            ax.plot(d, y, "|", color=INK_2, ms=8, alpha=0.6)
        ax.text(1.95, y, f"{m:+.2f}" + (" †" if r["seeds"] < len(SEEDS) else ""), va="center", ha="right",
                fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(ys, [r["setting"] for _, r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ test F1 vs baseline with the same seed (points)\n"
                  "dot: mean · bar: ± 1 std · ticks: individual seeds · †: 2 seeds only")
    ax.set_title("Effect of each hyperparameter (paired with the baseline)")
    ax.grid(axis="y", visible=False)
    lo = min(min(r["d_test_seeds"] or [0]) for _, r in rows)
    ax.set_xlim(min(lo, 0) - 0.5, 2.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_learning_curves(task, cell, best, path):
    configs = [("baseline", {}), (best["setting"], best["override"])] if not best["is_baseline"] else [
        ("baseline", {})]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, (label, override) in enumerate(configs):
        for s in SEEDS:
            p = RESULTS / "runs" / run_name(task, override, s, cell) / "history.json"
            if not p.exists():
                continue
            h = json.load(open(p))
            ep = [x["epoch"] for x in h]
            a.plot(ep, [100 * x["val_f1"] for x in h], color=SERIES[k], lw=1.6, alpha=0.9,
                   label=label if s == 1 else None)
    a.set_title("Validation entity F1 (%) — one line per seed")
    a.set_xlabel("epoch")
    a.set_ylim(80, 96)
    a.legend(frameon=False, loc="lower right")
    h = json.load(open(RESULTS / "runs" / run_name(task, {}, 1, cell) / "history.json"))
    ep = [x["epoch"] for x in h]
    b.plot(ep, [x["train_loss"] for x in h], color=SERIES[0], lw=2, label="train")
    b.plot(ep, [x["val_loss"] for x in h], color=SERIES[1], lw=2, label="validation")
    b.set_title("Baseline (seed 1): cross-entropy loss")
    b.set_xlabel("epoch")
    b.set_ylim(0, 0.4)
    b.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_per_type(task, cell, best, path):
    def mean_type(override):
        per = [json.load(open(RESULTS / "runs" / run_name(task, override, s, cell) / "metrics.json"))["test"]["per_type"]
               for s in SEEDS if (RESULTS / "runs" / run_name(task, override, s, cell) / "metrics.json").exists()]
        return {t: (st.mean(p[t]["f1-score"] for p in per), per[0][t]["support"]) for t in per[0]}

    base = mean_type({})
    configs = [("baseline", base)] + ([] if best["is_baseline"] else [(best["setting"], mean_type(best["override"]))])
    types = sorted(base, key=lambda t: -base[t][1])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(types) + 1.3))
    w = 0.38
    for k, (label, vals) in enumerate(configs):
        ax.barh([y + (k - (len(configs) - 1) / 2) * (w + 0.03) for y in range(len(types))],
                [100 * vals[t][0] for t in types], height=w, color=SERIES[k], label=label)
    ax.set_yticks(range(len(types)), [f"{t} (n={int(base[t][1])})" for t in types], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(50, 100)
    ax.set_xlabel("test F1 (%), mean over seeds")
    ax.set_title("Test F1 per entity type (n = test entities)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="fine")
    p.add_argument("--cell", default="vanilla")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    tables, seen = {}, set()
    for study, overrides in STUDIES.items():
        rows = []
        for o in overrides:
            r = config_stats(summary, args.task, args.cell, o)
            if r:
                rows.append(r)
        tables[study] = rows
    complete = {r["setting"]: r for rs in tables.values() for r in rs if r["seeds"] == len(SEEDS)}
    best = max(complete.values(), key=lambda r: (r["d_val"][0] if not r["is_baseline"] else 0.0))
    write_tables(tables, best, OUT / "tables.md")
    fig_effects(tables, OUT / "effects.png")
    if run_name(args.task, {}, 1, args.cell) in summary:
        fig_learning_curves(args.task, args.cell, best, OUT / "learning_curves.png")
        fig_per_type(args.task, args.cell, best, OUT / "per_type_f1.png")
    (OUT / "studies.png").unlink(missing_ok=True)  # replaced by effects.png
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
