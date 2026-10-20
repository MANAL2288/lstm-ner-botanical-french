"""Build notebooks/FloraNER_vanilla_LSTM_colab.ipynb: a self-contained Colab notebook.

The project sources are embedded with %%writefile, so the notebook is the only file to upload.
Re-run this script after any change in src/ or tests/:  python notebooks/build_colab.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "FloraNER_vanilla_LSTM_colab.ipynb"
SOURCES = ["src/data.py", "src/download_data.py", "src/metrics.py", "src/cells.py", "src/model.py",
           "src/train.py",
           "src/sweep.py", "src/merge_results.py", "src/eda.py", "src/analyze.py", "tests/test_model.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# FloraNER — Vanilla LSTM (from scratch) on Colab GPU

**Setup:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`. Nothing else to change.

* Each model trains for **at most 60 epochs** and stops after **10 epochs without validation-F1 gain**;
  the weights of the best validation epoch are evaluated on the test split.
* The full sweep is 60 runs (~9 h of GPU). Results are saved on Google Drive
  (`MyDrive/floraner_results_e60`) after every run: if Colab disconnects, just **Run all** again,
  finished runs are skipped.
* When everything is done, the last cell downloads `floraner_results_account0.zip`.
* GPU quota exhausted? Continue on another Google account: download `floraner_results_e60` from the first
  account's Drive, then use section 4 of this notebook on the new account.
"""),
    md("## 0. Settings, helper and GPU check"),
    code("""
# Which share of the sweep does THIS Colab account run?
#   0 = everything on this account (default)              (60 runs)
#   only if several people share the work:
#   1 = baseline + optimizer + learning rate             (21 runs)
#   2 = batch size + weight initialisation                (18 runs)
#   3 = hidden size + layers + direction + dropout + case (21 runs)
ACCOUNT = 0  # @param [0, 1, 2, 3]

import os, subprocess, sys
import torch

PY = sys.executable
GROUP = "all" if ACCOUNT == 0 else str(ACCOUNT)


def sh(cmd):
    \"\"\"Run a shell command, stream its output, and STOP 'Run all' if it fails.\"\"\"
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"Command failed (exit code {p.returncode}): {cmd}")


print("ACCOUNT", ACCOUNT, "| torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all again.")
print("GPU:", torch.cuda.get_device_name(0))
"""),
    md("## 1. Results folder on Google Drive"),
    code("""
try:
    from google.colab import drive
    drive.mount('/content/drive')
    os.environ["FLORANER_RESULTS"] = "/content/drive/MyDrive/floraner_results_e60"
except Exception as e:  # not on Colab / Drive refused: results stay in the session (download them!)
    print("Drive not mounted:", e)
    os.environ["FLORANER_RESULTS"] = "/content/floraner/results"
os.makedirs(os.environ["FLORANER_RESULTS"], exist_ok=True)
print("results ->", os.environ["FLORANER_RESULTS"])
"""),
    md("## 2. Project code\n(generated from the repository by `notebooks/build_colab.py` — do not edit here)"),
    code("""
!mkdir -p /content/floraner/src /content/floraner/tests /content/floraner/data
%cd /content/floraner
"""),
]
for rel in SOURCES:
    src = (ROOT / rel).read_text()
    cells.append(code(f"%%writefile {rel}\n{src}"))

cells += [
    md("## 3. Download the data, run the sanity tests"),
    code("""
try:
    import pytest  # preinstalled on Colab; only the tests need it
except ImportError:
    sh(f"{PY} -m pip install -q pytest")
sh(f"{PY} src/download_data.py")
sh(f"{PY} -m pytest tests -q")
"""),
    md("## 4. (Optional) import results from another Google account\n"
       "Only if you continue on a new account: tick the box, *Run all*, and when the button "
       "*Choose files* appears in this cell, select the zip downloaded from the other account's Drive "
       "(right click on `floraner_results_e60` → Download). Finished runs are then skipped."),
    code("""
IMPORT_PREVIOUS_RESULTS = False  # @param {type:"boolean"}
if IMPORT_PREVIOUS_RESULTS:
    from google.colab import files
    uploaded = files.upload()
    for name in uploaded:
        sh(f'{PY} src/merge_results.py "{name}"')
sh(f"{PY} src/sweep.py --group {GROUP} --list | head -1")  # progress: how many runs are already done
"""),
    md("## 5. Exploratory data analysis (figures for the report)"),
    code("""
sh(f"{PY} src/eda.py")
from IPython.display import Image, display
for f in ["entity_types.png", "lengths.png", "split_balance_fine.png"]:
    display(Image(f"results/eda/{f}", width=900))
"""),
    md("## 6. Baseline: vanilla BiLSTM, fine-grained task\nSkipped if it is already in the results."),
    code("""
import json
summary_path = os.path.join(os.environ["FLORANER_RESULTS"], "summary.jsonl")
baseline_done = os.path.exists(summary_path) and any(
    json.loads(line)["run_name"] == "fine_vanilla_baseline_s1" for line in open(summary_path) if line.strip())
if ACCOUNT not in (0, 1):
    print("The baseline is trained by account 1: skipped on this account.")
elif baseline_done:
    print("Baseline already trained: skipped.")
else:
    sh(f"{PY} -u src/train.py --task fine --run-name fine_vanilla_baseline_s1 --save-model")
"""),
    md("## 7. Hyperparameter studies (one factor at a time, 3 seeds)\nResumes automatically: finished runs are skipped."),
    code('sh(f"{PY} -u src/sweep.py --group {GROUP}")'),
    md("## 8. Analysis: tables and figures"),
    code("""
sh(f"{PY} src/analyze.py --task fine")
from IPython.display import Image, display
res = os.environ["FLORANER_RESULTS"]
for f in ["studies.png", "learning_curves.png", "per_type_f1.png"]:
    p = f"{res}/analysis/{f}"
    if os.path.exists(p):
        display(Image(p, width=900))
"""),
    md("## 9. Download all results as a zip"),
    code("""
import shutil
from google.colab import files
archive = shutil.make_archive(f"/content/floraner_results_account{ACCOUNT}", "zip", os.environ["FLORANER_RESULTS"])
files.download(archive)
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")
