"""Aggregate results/summary.jsonl into report tables and figures.

Usage: python src/analyze.py [--task fine] [--cell vanilla]
Outputs: results/analysis/{tables.md, studies.png, learning_curves.png, per_type_f1.png}
"""
from __future__ import annotations

import argparse
import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eda import GRID, INK_2, SERIES, SURFACE  # shared report style (rcParams set on import)
from sweep import BASELINE, STUDIES, run_name
from train import RESULTS

OUT = RESULTS / "analysis"


def load_summary() -> dict[str, dict]:
    rows = [json.loads(l) for l in open(RESULTS / "summary.jsonl")]
    return {r["run_name"]: r for r in rows}  # last write wins if a run was repeated


def label(override: dict) -> str:
    diff = {k: v for k, v in override.items() if BASELINE.get(k) != v}
    if not diff:
        return "baseline"
    return ", ".join(f"{k}={v}" for k, v in diff.items())


def agg(values):
    return (st.mean(values), st.stdev(values) if len(values) > 1 else 0.0) if values else (float("nan"), 0.0)


def study_table(summary, task, cell, study):
    rows = []
    for override in STUDIES[study]:
        runs = [summary[n] for s in range(1, 11) if (n := run_name(task, override, s, cell)) in summary]
        if not runs:
            continue
        rows.append({
            "variant": label({**BASELINE, **override}) if override else "baseline",
            "setting": ", ".join(f"{k}={v}" for k, v in override.items()),
            "seeds": len(runs),
            "val_f1": agg([r["val_f1"] for r in runs]),
            "test_f1": agg([r["test_f1"] for r in runs]),
            "test_macro_f1": agg([r["test_macro_f1"] for r in runs]),
            "params": runs[0]["params_total"],
            "params_recurrent": runs[0]["params_recurrent"],
            "epoch_time": st.mean(r["mean_epoch_time_s"] for r in runs),
            "best_epoch": st.mean(r["best_epoch"] for r in runs),
        })
    return rows


def fmt(ms):
    return f"{100 * ms[0]:.2f} ± {100 * ms[1]:.2f}"


def write_tables(tables, path):
    lines = ["# Hyperparameter studies (vanilla LSTM)", "",
             "Entity-level F1 (seqeval, %) — mean ± std over seeds. Choose on **val**, report **test**.", ""]
    for study, rows in tables.items():
        if not rows:
            continue
        best = max(rows, key=lambda r: r["val_f1"][0])
        lines += [f"## {study}", "",
                  "| setting | seeds | val F1 | test F1 | test macro F1 | params (recurrent) | s/epoch | best epoch |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            mark = " **(best val)**" if r is best else ""
            lines.append(f"| {r['setting']}{mark} | {r['seeds']} | {fmt(r['val_f1'])} | {fmt(r['test_f1'])} | "
                         f"{fmt(r['test_macro_f1'])} | {r['params']:,} ({r['params_recurrent']:,}) | "
                         f"{r['epoch_time']:.1f} | {r['best_epoch']:.1f} |")
        lines.append("")
    path.write_text("\n".join(lines))


def fig_studies(tables, path):
    studies = [s for s, rows in tables.items() if rows]
    if not studies:
        return
    cols = 3
    nrows = -(-len(studies) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(13, 3.2 * nrows), squeeze=False)
    for ax, study in zip(axes.flat, studies):
        rows = tables[study]
        ys = range(len(rows))
        means = [100 * r["val_f1"][0] for r in rows]
        stds = [100 * r["val_f1"][1] for r in rows]
        ax.errorbar(means, ys, xerr=stds, fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2,
                    capsize=0, markersize=7)
        for y, m in zip(ys, means):
            ax.text(m, y + 0.22, f"{m:.2f}", ha="center", fontsize=8, color=INK_2)
        ax.set_yticks(list(ys), [r["setting"] for r in rows], fontsize=8)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.invert_yaxis()
        ax.set_title(study)
        ax.set_xlabel("val F1 (%)")
        ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[len(studies):]:
        ax.axis("off")
    fig.suptitle("Vanilla LSTM — one-factor-at-a-time studies (mean ± std over seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_learning_curves(run, path):
    hist = json.load(open(RESULTS / "runs" / run / "history.json"))
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.plot(ep, [h["train_loss"] for h in hist], color=SERIES[0], lw=2, label="train")
    a.plot(ep, [h["val_loss"] for h in hist], color=SERIES[1], lw=2, label="validation")
    a.set_title("Cross-entropy loss")
    a.set_xlabel("epoch")
    a.legend(frameon=False)
    b.plot(ep, [100 * h["val_f1"] for h in hist], color=SERIES[0], lw=2, label="micro F1")
    b.plot(ep, [100 * h["val_macro_f1"] for h in hist], color=SERIES[1], lw=2, label="macro F1")
    b.set_title("Validation entity-level F1 (%)")
    b.set_xlabel("epoch")
    b.legend(frameon=False)
    fig.suptitle(run, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_per_type(run, path):
    m = json.load(open(RESULTS / "runs" / run / "metrics.json"))["test"]["per_type"]
    items = sorted(m.items(), key=lambda kv: kv[1]["support"], reverse=True)
    names = [f"{k} (n={int(v['support'])})" for k, v in items]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(items) + 1.2))
    ys = range(len(items))
    w = 0.26
    for k, metric in enumerate(("precision", "recall", "f1-score")):
        ax.barh([y + (k - 1) * (w + 0.02) for y in ys], [100 * v[metric] for _, v in items], height=w,
                color=SERIES[k], label=metric.replace("-score", ""))
    ax.set_yticks(list(ys), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("%")
    ax.set_title(f"Test scores per entity type — {run}")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower left", ncol=3)
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
    tables = {s: study_table(summary, args.task, args.cell, s) for s in STUDIES}
    write_tables(tables, OUT / "tables.md")
    fig_studies(tables, OUT / "studies.png")
    base = run_name(args.task, {}, 1, args.cell)
    if base in summary:
        fig_learning_curves(base, OUT / "learning_curves.png")
        fig_per_type(base, OUT / "per_type_f1.png")
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
