"""Merge results produced on several Colab accounts into one results folder.

Usage: python src/merge_results.py floraner_results_account2.zip floraner_results_e60-2026....zip [...]
Each argument is a zip (made by the notebook, or downloaded from Google Drive) or a folder containing
summary.jsonl and runs/, possibly inside a sub-folder. Runs already present in the target
(FLORANER_RESULTS, default ./results) are kept as is, so merging twice changes nothing.
"""
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from train import RESULTS


def find_root(path: Path) -> Path:
    """Folder holding summary.jsonl (a Drive download nests it in a floraner_results_* folder)."""
    if (path / "summary.jsonl").exists():
        return path
    found = sorted(path.rglob("summary.jsonl"), key=lambda p: len(p.parts))
    if not found:
        raise SystemExit(f"no summary.jsonl found in {path}")
    return found[0].parent


def merge(source: Path) -> tuple[int, int]:
    source = find_root(source)
    target_summary = RESULTS / "summary.jsonl"
    known = {json.loads(l)["run_name"] for l in open(target_summary)} if target_summary.exists() else set()
    added = skipped = 0
    lines = [l for l in (source / "summary.jsonl").read_text().splitlines() if l.strip()]
    with open(target_summary, "a") as out:
        for line in lines:
            name = json.loads(line)["run_name"]
            if name in known:
                skipped += 1
                continue
            run_dir = source / "runs" / name
            if run_dir.exists():
                shutil.copytree(run_dir, RESULTS / "runs" / name, dirs_exist_ok=True)
            out.write(line + "\n")
            known.add(name)
            added += 1
    return added, skipped


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        path = Path(arg)
        with tempfile.TemporaryDirectory() as tmp:
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as z:
                    z.extractall(tmp)
                path = Path(tmp)
            added, skipped = merge(path)
        print(f"{arg}: {added} runs added, {skipped} already present")
    print(f"merged into {RESULTS}")


if __name__ == "__main__":
    main()
