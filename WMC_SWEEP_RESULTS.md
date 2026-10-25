# WMC hyperparameter experiments — Said & MS

All 60 unique runs are complete: 20 configurations with seeds 1, 2 and 3. This delivers the WMC experiments requested for the shared comparison; the cross-model analysis, report and slides remain collective work.

## Files to use

- [results/summary_wmc.json](results/summary_wmc.json): complete WMC-only array, 60 records.
- [results/summary_wmc.jsonl](results/summary_wmc.jsonl): the same records, one per line.
- [results/summary.jsonl](results/summary.jsonl): shared results with the 57 new WMC records appended; pre-existing records are preserved.
- [results/wmc_sweep_manifest.json](results/wmc_sweep_manifest.json): run names by group and platform provenance.
- `results/runs/<run_name>/`: original config.json, history.json and metrics.json for each run.

These are individual run results, not averages. F1 values are fractions: 0.94 means 94%. The older results/wmc_summary.md and WMC_HANDOFF.md baseline figures describe only the three baseline runs.

## Groups and protocol

| Group | Experiments | Completed runs |
|---|---|---:|
| 1 | Optimizers and learning rates, including baseline | 21 |
| 2 | Batch sizes and initializations | 18 |
| 3 | Hidden size, layers, direction, dropout, casing | 21 |

The unchanged shared sweep configuration generator and training code were used: fine NER, seeds 1/2/3, maximum 60 epochs, patience 10, full training data. Each test score is evaluated from the best validation checkpoint. Choose hyperparameters on validation scores; test scores are for final reporting.

## Execution and reproducibility

18 group-1 runs (including baselines) completed on Colab. After its GPU quota was exhausted, the remaining three lr=0.0003 runs and all 21 group-3 runs completed in a second Kaggle job. Group 2 completed separately on Kaggle. Every run used CUDA; the baseline and initial runs used Tesla T4, and both Kaggle sessions were configured as T4 x2. The training code used one GPU per run. Platform/library differences should be considered when comparing runtime measurements.

The Colab notebook in this branch remains a sequential reproduction workflow, not the executed Kaggle output. To reproduce a selected group with the project installed and data downloaded:

```sh
python src/sweep.py --cell wmc --group 1 --seeds 1 2 3 --epochs 60 --patience 10
```

Use group 2 or 3 as appropriate. Completed names in the selected results directory are skipped. For an independent rerun, set FLORANER_RESULTS to a new empty directory. Interrupted runs restart from their seed; model weights were not saved.

## Validation and backups

Validated all 60 unique names against the shared sweep plan, all three seeds per configuration, the per-run maximum epochs/patience, full training batches, history length against epochs_run, and summary test F1 against metrics.json. Both Kaggle saved jobs finished successfully. Source archives downloaded from Drive were verified against local SHA256 hashes.

[Drive: complete summary and README](https://drive.google.com/drive/folders/1W4BOYR9gANfwIHPDgbbUn1itJL7nW40d). The parent Drive folder also holds separate complete group archives. No dataset or model-weight archives are included in this PR.
