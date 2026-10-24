# FloraNER — Named Entity Recognition in botanical French with from-scratch LSTMs

Lab project, Advanced NLP 2026/2027. Task: NER on **FloraNER**
([Zenodo 12765869](https://zenodo.org/records/12765869), CC-BY-4.0) with recurrent cells written
from `nn.Linear` only (no `nn.LSTM` / `nn.LSTMCell`).

## Quick start

```bash
pip install -r requirements.txt          # Intel Mac: Python 3.11, torch==2.2.2, numpy<2
python src/download_data.py              # download + convert to BIO splits
python -m pytest tests -q                # sanity checks of the LSTM implementation
python src/eda.py                        # statistics + figures -> results/eda/
python src/train.py --task fine          # one run (vanilla BiLSTM baseline)
python src/sweep.py --study all          # 60 runs: one factor at a time, 3 seeds, resumable
python src/analyze.py --task fine        # tables + figures -> results/analysis/
```

On Colab (GPU): upload `notebooks/FloraNER_vanilla_LSTM_colab.ipynb` (self-contained, regenerate it with
`python notebooks/build_colab.py`), select a T4 GPU and *Run all*.

**Training protocol (same for every cell of the group):** at most 60 epochs, early stopping after 10
epochs without validation-F1 improvement, weights of the best validation epoch evaluated on test;
hyperparameters chosen on validation F1, reported as mean ± std over 3 seeds.

## Repository layout

| Path | Content |
|---|---|
| `src/data.py` | CSV loading, tokenisation, char spans → BIO, document-level val split, vocab, `Dataset`, `collate_fn` |
| `src/cells.py` | `RecurrentCell` interface + `VanillaLSTMCell` (eqs. 1–6) and `WMCCell` (eqs. 7–12) |
| `src/model.py` | `RecurrentLayer` (masked, optionally bidirectional unrolling) and `NERTagger` |
| `src/metrics.py` | entity-level precision / recall / F1 (conlleval semantics) |
| `src/train.py` | training loop (≤ 60 epochs, early stopping, best validation epoch kept), evaluation, run logging |
| `src/sweep.py` | one-factor-at-a-time hyperparameter studies |
| `src/analyze.py` | aggregation over seeds → report tables and figures |
| `src/merge_results.py` | merge results produced on several Colab accounts |
| `src/eda.py` | exploratory data analysis |
| `tests/` | equivalence with `nn.LSTMCell` (reference only), padding invariance, parameter count, BIO conversion, metrics = seqeval |

## Data pipeline

* **Splits.** The official `Train/` and `Test/` folders are used as is. A validation set is carved out of
  train **at document level** (10 %, seed 42), so that no text appears in two splits. Every model of the
  group uses exactly the same `data/processed/<task>/{train,val,test}.jsonl`.
* **Tokenisation.** `\w+|[^\w\s]` (hyphens and apostrophes are split). This keeps **100 %** of the
  coarse/fine entity boundaries aligned with token boundaries (a tokenizer keeping `obovées-elliptiques`
  or `l'axe` as one token loses 1.3 % of the entities).
* **Annotation noise (species).** 1.8 % of species spans stop in the middle of a word at an accented
  character (`O. mont|ícola`), an artefact of the distant annotation. Spans are snapped to whole tokens.
* **Overlaps.** The `filtered_annotations` column (longest span kept), as in the authors' code.
* **Segmentation.** Coarse/fine documents are long (median 260 tokens): they are cut into segments of
  ≤ 128 tokens at clause boundaries, never inside an entity, so entity-level scores are unchanged.

## Model

```
word embedding (100) ⊕ casing embedding (8)
  → dropout → [custom LSTM layer, forward ⊕ backward] × L → dropout → Linear → tag logits
```

* The input projections `W_x·x_t + b` of the four gates are computed for the whole sequence in one
  `nn.Linear` call; only `W_h·h_{t-1}` stays inside the time loop.
* Padding: on a padded step the state is carried over unchanged, so the backward direction starts
  on the last real token; the loss ignores padded labels (`ignore_index=-100`).
* Evaluation: entity-level precision / recall / F1 (micro and macro) and per-type report, conlleval
  semantics, implemented in `src/metrics.py` (no dependency) and tested to give exactly seqeval's numbers.

## Adding a cell variant (CIFG, LSTMP, WMC)

Subclass `RecurrentCell` in `src/cells.py`, implement `input_projection`, `init_state`, `forward`,
`reset_parameters`, set `output_size` (= P for LSTMP), and register it in `CELLS`. Then:

```bash
python src/train.py --task fine --cell cifg
python src/sweep.py --cell cifg --study hidden_size
```

Everything else (data, training, evaluation, logging, parameter counts) is shared.

## Results — vanilla LSTM, fine-grained task (10 entity types)

56 runs on a Colab T4 (all with the protocol above). Full tables: `results/analysis/tables.md`.

| Model | Test F1 (%) | Test macro F1 (%) | Params (recurrent) | s/epoch (T4) | Best epoch |
|---|---|---|---|---|---|
| Baseline (Adam, lr 1e-3, batch 32, BiLSTM 128) | 94.01 ± 1.10 | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58 |
| **Best on validation: lr = 0.01** | **94.35 ± 0.69** | **92.31 ± 0.40** | 800,035 (242,688) | 11.9 | 22 |
| spaCy, dataset authors (reference) | 94.5 | — | — | — | — |

Main findings (Δ = test F1 difference with the baseline trained with the same seed, `results/analysis/effects.png`):

* **Seed effect.** Seed 3 is ~2 points below seeds 1–2 for almost every configuration (for the baseline, mostly
  false positives on FORME: precision −16 points). Configurations are therefore compared **paired by seed**, which
  removes this shared noise.
* **Bidirectionality is by far the most important choice:** a forward-only LSTM loses **4.1 points** (Δ −4.12 ± 0.25).
* **Optimisation speed matters:** lr = 0.01 converges ~3× faster (best epoch 22 vs 58) and is the only setting that
  improves the baseline (Δ +0.35); SGD (Δ −2.70), lr = 3e-4 (Δ −2.10) and batch 128 (Δ −1.07) have not converged
  within 60 epochs (best epoch ≈ 60).
* **Regularisation is needed:** without dropout the model overfits early (best epoch ≈ 29, Δ −1.64).
* **No measurable effect** (|Δ| ≤ 0.35, within seed noise): Adam / AdamW / RMSprop, weight initialisation
  (default, Xavier, orthogonal, forget-gate bias), batch 16 vs 32, 2 layers (2.6× recurrent parameters, 1.9× slower).
* Capacity: hidden size 64 or 256 is slightly worse than 128 (Δ −0.44 / −0.53); 256 overfits earlier.
* Per type: ORGANE, POSITION, MESURE > 95 %; SURFACE (~82 %) and FORME/DESCRIPTEUR confusions are the main errors.
  Macro F1 is fragile: DEVELOPPEMENT has only 4 test entities.

Not run (Colab GPU quota): dropout 0.5 seed 3, and the case-feature ablation.

## WMC contribution — Said & MS

WMC is implemented in `src/cells.py` and registered as `wmc`. The input/forget gates use the previous memory, and the output gate uses the **updated** memory (guidelines equation 9). All transformations use `nn.Linear`; the three memory maps are full bias-free matrices followed by `tanh`.

Under the common baseline protocol (Adam lr=0.001, BiWMC H=128, dropout 0.3, batch 32, seeds 1/2/3, at most 60 epochs, patience 10), test entity F1 is **93.75 ± 1.49%**. Individual seeds: **94.66%, 94.56%, 92.03%**. Trainable parameters: **898,339 total**, including **340,992 recurrent**. Full metrics and histories are in `results/runs/fine_wmc_baseline_s*/`; the summary is [results/wmc_summary.md](results/wmc_summary.md).

Run only WMC: `python src/sweep.py --cell wmc --study baseline`, then `python src/summarize_wmc.py`. For Colab T4, use [notebooks/FloraNER_WMC_colab.ipynb](notebooks/FloraNER_WMC_colab.ipynb); regenerate it with `python notebooks/build_wmc_colab.py`. See [WMC_HANDOFF.md](WMC_HANDOFF.md) and [WMC_VALIDATION.md](WMC_VALIDATION.md) for the duo handoff and checks. The collective architecture comparison, final report and slides remain group work.

## Reference

Nainia et al., *FloraNER: a Named Entity Recognition Dataset for Botanical French Text* (2024).
The authors' spaCy models reach entity F1 = 0.963 (species), 0.945 (coarse), 0.945 (fine) on their test split.
