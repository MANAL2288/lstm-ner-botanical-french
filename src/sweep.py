"""One-factor-at-a-time hyperparameter studies around the baseline, several seeds each.

Usage:
    python src/sweep.py --group 1                     # the share of Colab account 1 (see GROUPS)
    python src/sweep.py --study optimizer lr          # some studies
    python src/sweep.py --study all --seeds 1 2 3     # everything
    python src/sweep.py --group 2 --list              # print the plan

Every run trains for at most 60 epochs and stops after 10 epochs without validation-F1
improvement; the best validation epoch is kept. Runs are named deterministically from their
settings; a run already present in results/summary.jsonl is skipped, so an interrupted sweep
simply resumes. Hyperparameters are chosen on the validation F1; the test split is only reported.
"""
from __future__ import annotations

import argparse
import json

from train import RESULTS, Config, run

BASELINE = dict(cell="vanilla", emb_dim=100, case_dim=8, hidden_size=128, num_layers=1, bidirectional=True,
                dropout=0.3, init="default", forget_bias=0.0, optimizer="adam", lr=1e-3, batch_size=32)

STUDIES: dict[str, list[dict]] = {
    "optimizer": [dict(optimizer="adam", lr=1e-3), dict(optimizer="adamw", lr=1e-3),
                  dict(optimizer="rmsprop", lr=1e-3), dict(optimizer="sgd", lr=0.1)],
    "lr": [dict(lr=lr) for lr in (1e-2, 3e-3, 1e-3, 3e-4)],
    "batch_size": [dict(batch_size=b) for b in (16, 32, 64, 128)],
    "init": [dict(init="default"), dict(init="xavier"), dict(init="orthogonal"),
             dict(init="orthogonal", forget_bias=1.0)],
    "hidden_size": [dict(hidden_size=h) for h in (64, 128, 256)],
    "num_layers": [dict(num_layers=n) for n in (1, 2)],
    "direction": [dict(bidirectional=False), dict(bidirectional=True)],
    "dropout": [dict(dropout=d) for d in (0.0, 0.3, 0.5)],
    "case_features": [dict(case_dim=0), dict(case_dim=8)],
}

# Share of the sweep per Colab account (~20 runs each). The baseline is trained by account 1 only;
# the other accounts skip it and their results are merged afterwards (src/merge_results.py).
GROUPS: dict[str, tuple[list[str], bool]] = {  # name -> (studies, exclude baseline)
    "1": (["optimizer", "lr"], False),
    "2": (["batch_size", "init"], True),
    "3": (["hidden_size", "num_layers", "direction", "dropout", "case_features"], True),
    "all": (list(STUDIES), False),
}

SHORT = {"optimizer": "opt", "lr": "lr", "batch_size": "bs", "init": "init", "forget_bias": "fb",
         "hidden_size": "h", "num_layers": "l", "bidirectional": "bi", "dropout": "do", "case_dim": "case",
         "emb_dim": "emb", "cell": "cell"}


def is_baseline(override: dict) -> bool:
    return all(BASELINE.get(k) == v for k, v in override.items())


def run_name(task: str, override: dict, seed: int, cell: str = "vanilla") -> str:
    diff = {k: v for k, v in override.items() if BASELINE.get(k) != v}
    suffix = "_".join(f"{SHORT.get(k, k)}{v}" for k, v in sorted(diff.items())) or "baseline"
    return f"{task}_{cell}_{suffix}_s{seed}"


def make_config(task: str, override: dict, seed: int, cell: str, epochs: int, patience: int) -> Config:
    params = {**BASELINE, "cell": cell, **override}
    return Config(task=task, seed=seed, epochs=epochs, patience=patience,
                  run_name=run_name(task, override, seed, cell), **params)


def done_runs() -> set[str]:
    path = RESULTS / "summary.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["run_name"] for line in open(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=list(GROUPS), help="predefined share of the sweep (overrides --study)")
    p.add_argument("--study", nargs="+", default=["all"])
    p.add_argument("--exclude-baseline", action="store_true", help="skip runs equal to the baseline")
    p.add_argument("--task", default="fine")
    p.add_argument("--cell", default="vanilla")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10, help="0 = always train all epochs")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.group:
        studies, exclude_baseline = GROUPS[args.group]
    else:
        studies = list(STUDIES) if args.study == ["all"] else args.study
        exclude_baseline = args.exclude_baseline
    plan, seen = [], set()
    for study in studies:
        for override in STUDIES[study]:
            if exclude_baseline and is_baseline(override):
                continue
            for seed in args.seeds:
                cfg = make_config(args.task, override, seed, args.cell, args.epochs, args.patience)
                if cfg.run_name not in seen:
                    seen.add(cfg.run_name)
                    plan.append(cfg)
    done = done_runs()
    todo = [c for c in plan if c.run_name not in done]
    print(f"studies: {', '.join(studies)} | {len(plan)} runs planned, {len(plan) - len(todo)} already done, "
          f"{len(todo)} to go")
    if args.list:
        for c in plan:
            print(("  [done] " if c.run_name in done else "  [todo] ") + c.run_name)
        return
    for k, cfg in enumerate(todo, 1):
        print(f"\n===== run {k}/{len(todo)}: {cfg.run_name}")
        run(cfg)


if __name__ == "__main__":
    main()
