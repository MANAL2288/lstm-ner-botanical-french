"""Generate the WMC groups 1/2/3 notebook; preserve the baseline notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(True))

def code(text):
    return dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[])

base = json.loads((ROOT/'notebooks/FloraNER_WMC_colab.ipynb').read_text(encoding='utf-8'))
cells = base['cells'][:15]
cells[0] = md('''# FloraNER — WMC hyperparameter studies (Said & MS)
Runs the shared groups 1, 2 and 3 sequentially on one T4 GPU. 60 runs in total, including the three completed baseline runs: 57 additional runs. Seeds 1/2/3, at most 60 epochs, patience 10. Uses the unchanged shared sweep, training and data code.

Results remain in `MyDrive/floraner_wmc_results_e60`. Completed runs are skipped. An interrupted run starts again from its seed; there is no mid-run checkpoint. Keep only one execution writing to this directory. The final architecture comparison and report remain group work.
''')
cells += [md('## Check existing baseline and prepare exports'), code('''import json
from pathlib import Path
RESULT_DIR = Path(os.environ['FLORANER_RESULTS'])

def export_wmc():
    rows = [json.loads(line) for line in (RESULT_DIR/'summary.jsonl').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['cell'] == 'wmc' and r['task'] == 'fine']
    assert len({r['run_name'] for r in rows}) == len(rows), 'Duplicate run names: inspect before continuing'
    for row in rows:
        folder = RESULT_DIR/'runs'/row['run_name']
        cfg = json.loads((folder/'config.json').read_text())
        metrics = json.loads((folder/'metrics.json').read_text())
        history = json.loads((folder/'history.json').read_text())
        assert cfg['epochs'] == 60 and cfg['patience'] == 10 and cfg['max_train_batches'] == 0
        assert len(history) == metrics['epochs_run']
        assert metrics['test']['f1'] == row['test_f1']
    rows.sort(key=lambda row: row['run_name'])
    for name, text in [('summary_wmc.json', json.dumps(rows, indent=2)),
                       ('summary_wmc.jsonl', '\\n'.join(json.dumps(row) for row in rows))]:
        temporary = RESULT_DIR/(name + '.tmp')
        temporary.write_text(text + '\\n', encoding='utf-8')
        temporary.replace(RESULT_DIR/name)
    print(f'Exported {len(rows)}/60 completed WMC runs', flush=True)
    return rows

existing = export_wmc()
assert {f'fine_wmc_baseline_s{s}' for s in (1,2,3)} <= {r['run_name'] for r in existing}, 'Restore the existing baseline results before running the sweep'
for group in ('1','2','3'):
    run('src/sweep.py', '--cell', 'wmc', '--group', group, '--list')
''')]
for group, description in [('1','Optimizers and learning rates'), ('2','Batch sizes and initializations'), ('3','Hidden sizes, layers, direction, dropout and casing')]:
    cells += [md(f'## Group {group} — {description}\nCompleted results are saved after every run. The log is also written to Drive.'), code(f'''command = [sys.executable, '-u', 'src/sweep.py', '--cell', 'wmc', '--group', '{group}', '--seeds', '1', '2', '3', '--epochs', '60', '--patience', '10']
print('Running:', ' '.join(command), flush=True)
try:
    with (RESULT_DIR/'wmc_group_{group}.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
finally:
    export_wmc()
''')]
cells += [md('## Final WMC export'), code('''completed = export_wmc()
assert len(completed) == 60, f'Only {len(completed)}/60 runs complete; resume unfinished groups'
run('src/summarize_wmc.py')  # baseline summary remains explicitly baseline-only
import shutil
from google.colab import files
archive = shutil.make_archive('/content/floraner_wmc_sweep_results', 'zip', RESULT_DIR)
files.download(archive)
''')]
base['cells'] = cells
target = ROOT/'notebooks/FloraNER_WMC_sweep_colab.ipynb'
target.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding='utf-8')
print(target)
