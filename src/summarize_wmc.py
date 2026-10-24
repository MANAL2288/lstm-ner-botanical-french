"""Summarize only the WMC baseline; reject incomplete or non-common runs."""
import json
import statistics
from train import RESULTS
from sweep import BASELINE


def main():
    rows = []
    for seed in (1, 2, 3):
        folder = RESULTS / 'runs' / f'fine_wmc_baseline_s{seed}'
        cfg = json.loads((folder / 'config.json').read_text())
        expected = {**BASELINE, 'cell': 'wmc', 'task': 'fine', 'seed': seed, 'epochs': 60, 'patience': 10,
                    'max_train_batches': 0, 'scheduler': 'none', 'weight_decay': 0.0,
                    'clip': 5.0, 'lowercase': True, 'min_freq': 1}
        for key, value in expected.items():
            if cfg[key] != value:
                raise ValueError(f'{folder.name}: {key}={cfg[key]!r}; expected {value!r}')
        metrics = json.loads((folder / 'metrics.json').read_text())
        rows.append(metrics)
    lines = ['# WMC baseline — Said & MS', '',
             'Common protocol; seeds 1, 2, 3. Test metrics use the best validation epoch.', '',
             '| Seed | Test P (%) | Test R (%) | Test F1 (%) | Macro F1 (%) | Best epoch | s/epoch |',
             '|---|---|---|---|---|---|---|']
    for seed, row in zip((1, 2, 3), rows):
        t = row['test']
        lines.append(f"| {seed} | {100*t['precision']:.2f} | {100*t['recall']:.2f} | {100*t['f1']:.2f} | {100*t['macro_f1']:.2f} | {row['best_epoch']} | {row['mean_epoch_time_s']:.2f} |")
    lines += ['', 'Mean ± sample standard deviation across three seeds:']
    for key in ('precision', 'recall', 'f1', 'macro_f1'):
        vals = [100*r['test'][key] for r in rows]
        lines.append(f'- Test {key}: {statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}%')
    if any(r['params'] != rows[0]['params'] for r in rows):
        raise ValueError('Parameter counts differ across seeds')
    lines += ['', 'Trainable parameters: ' + json.dumps(rows[0]['params']), '',
              'Collective comparison and interpretation remain group work.']
    out = RESULTS / 'wmc_summary.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
