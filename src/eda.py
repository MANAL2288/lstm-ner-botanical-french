"""Exploratory data analysis of FloraNER: statistics table + figures for the report.

Usage: python src/eda.py   (after `python src/data.py`)
Outputs: results/eda/eda_summary.json and results/eda/*.png
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ROOT, TASKS, load_raw, normalize, read_jsonl, tokenize

OUT = ROOT / "results" / "eda"
SPLITS = ("train", "val", "test")

# Reference palette (light mode, printed report)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3: validated all-pairs

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def entities(tags):
    """Yield (type, length) for each BIO entity of a tag sequence."""
    cur, n = None, 0
    for t in tags + ["O"]:
        if t.startswith("I-") and cur == t[2:]:
            n += 1
            continue
        if cur:
            yield cur, n
        cur, n = (t[2:], 1) if t != "O" else (None, 0)


def task_stats(task: str) -> dict:
    recs = {s: read_jsonl(task, s) for s in SPLITS}
    train_vocab = {normalize(t, True) for r in recs["train"] for t in r["tokens"]}
    out = {}
    for s in SPLITS:
        toks = [t for r in recs[s] for t in r["tokens"]]
        tags = [t for r in recs[s] for t in r["tags"]]
        ents = [e for r in recs[s] for e in entities(r["tags"])]
        ent_tok = [normalize(t, True) for r in recs[s] for t, g in zip(r["tokens"], r["tags"]) if g != "O"]
        out[s] = {
            "documents": len({r["doc_id"] for r in recs[s]}),
            "segments": len(recs[s]),
            "tokens": len(toks),
            "pct_O": round(100 * tags.count("O") / len(tags), 2),
            "entities": len(ents),
            "entities_per_type": dict(Counter(e for e, _ in ents).most_common()),
            "entity_length": {k: Counter(min(n, 6) for _, n in ents)[k] for k in range(1, 7)},
            "seg_len_mean": round(len(toks) / len(recs[s]), 1),
            "seg_len_max": max(len(r["tokens"]) for r in recs[s]),
        }
        if s != "train":
            out[s]["oov_token_pct"] = round(100 * sum(normalize(t, True) not in train_vocab for t in toks) / len(toks), 2)
            out[s]["oov_entity_token_pct"] = round(100 * sum(t not in train_vocab for t in ent_tok) / max(1, len(ent_tok)), 2)
    out["vocab_train"] = len(train_vocab)
    out["raw_doc_len"] = [len(tokenize(text)) for text, _ in load_raw(task, "train") + load_raw(task, "test")]
    return out


def fig_entity_types(stats):
    """Entity counts per type (train), one panel per task."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.2, 2.2]})
    for ax, task in zip(axes, TASKS):
        counts = stats[task]["train"]["entities_per_type"]
        labels, vals = list(counts)[::-1], list(counts.values())[::-1]
        ax.barh(labels, vals, color=SERIES[0], height=0.6)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:,}", va="center", fontsize=8, color=INK_2)
        ax.set_title(f"{task}  ({stats[task]['train']['pct_O']}% of tokens are O)")
        ax.set_xlabel("entities in train")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.22)
    fig.suptitle("Entity type distribution (train split)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "entity_types.png", dpi=200)
    plt.close(fig)


def fig_lengths(stats):
    """Raw document length (tokens) and entity length, one row per task."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, task in enumerate(TASKS):
        lens = stats[task]["raw_doc_len"]
        ax = axes[0, j]
        ax.hist(lens, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(128, color=INK_2, linestyle="--", linewidth=1)
        ax.text(128, ax.get_ylim()[1] * 0.92, " max_len=128", fontsize=8, color=INK_2)
        ax.set_title(f"{task}: document length (median {sorted(lens)[len(lens)//2]} tokens)")
        ax.set_xlabel("tokens per document")
        ax.set_ylabel("documents")

        ax = axes[1, j]
        el = stats[task]["train"]["entity_length"]
        total = sum(el.values())
        xs = [str(k) if k < 6 else "6+" for k in el]
        ax.bar(xs, [100 * v / total for v in el.values()], color=SERIES[0], width=0.6)
        ax.set_title(f"{task}: entity length")
        ax.set_xlabel("tokens per entity")
        ax.set_ylabel("% of entities")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "lengths.png", dpi=200)
    plt.close(fig)


def fig_split_balance(stats, task="fine"):
    """Share of each entity type per split: checks that val/test mirror train."""
    types = list(stats[task]["train"]["entities_per_type"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.26
    for k, s in enumerate(SPLITS):
        c = stats[task][s]["entities_per_type"]
        tot = sum(c.values())
        xs = [i + (k - 1) * (w + 0.02) for i in range(len(types))]
        ax.bar(xs, [100 * c.get(t, 0) / tot for t in types], width=w, color=SERIES[k], label=s)
    ax.set_xticks(range(len(types)), types, rotation=30, ha="right")
    ax.set_ylabel("% of entities in split")
    ax.set_title(f"{task}: entity type share per split")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"split_balance_{task}.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {t: task_stats(t) for t in TASKS}
    fig_entity_types(stats)
    fig_lengths(stats)
    fig_split_balance(stats, "fine")
    fig_split_balance(stats, "coarse")

    summary = {t: {k: v for k, v in s.items() if k != "raw_doc_len"} for t, s in stats.items()}
    with open(OUT / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"{'task':8s} {'split':5s} {'docs':>5s} {'segs':>5s} {'tokens':>7s} {'%O':>6s} {'ents':>6s} "
          f"{'segLen':>6s} {'OOV%':>5s} {'OOVent%':>7s}")
    for t, s in summary.items():
        for sp in SPLITS:
            d = s[sp]
            print(f"{t:8s} {sp:5s} {d['documents']:5d} {d['segments']:5d} {d['tokens']:7d} {d['pct_O']:6.2f} "
                  f"{d['entities']:6d} {d['seg_len_mean']:6.1f} {d.get('oov_token_pct', ''):>5} "
                  f"{d.get('oov_entity_token_pct', ''):>7}")
        print(f"{t:8s} vocab(train, lowercased) = {s['vocab_train']}")
        print(f"{t:8s} entity types = {s['train']['entities_per_type']}")
    print(f"\nFigures written to {OUT}")


if __name__ == "__main__":
    main()
