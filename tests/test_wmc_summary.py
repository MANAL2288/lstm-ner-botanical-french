"""Check handoff summary rejects partial seeds and protocol drift."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import summarize_wmc
from train import Config
from dataclasses import asdict


def write_run(root, seed):
    folder = root / 'runs' / f'fine_wmc_baseline_s{seed}'
    folder.mkdir(parents=True)
    cfg = asdict(Config(cell='wmc', seed=seed))
    (folder/'config.json').write_text(json.dumps(cfg))
    metrics = {'test':dict(precision=.9, recall=.8, f1=.84+seed*.01, macro_f1=.7),
               'best_epoch':20,'mean_epoch_time_s':15,'params':{'total':898339,'recurrent':340992}}
    (folder/'metrics.json').write_text(json.dumps(metrics))
    return folder


def test_summary_requires_all_three_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize_wmc, 'RESULTS', tmp_path)
    write_run(tmp_path,1)
    with pytest.raises(FileNotFoundError):
        summarize_wmc.main()
    assert not (tmp_path/'wmc_summary.md').exists()


def test_summary_rejects_protocol_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize_wmc, 'RESULTS', tmp_path)
    for seed in (1,2,3):
        folder = write_run(tmp_path,seed)
    cfg=json.loads((folder/'config.json').read_text())
    cfg['max_train_batches']=2
    (folder/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='max_train_batches'):
        summarize_wmc.main()
    assert not (tmp_path/'wmc_summary.md').exists()


def test_summary_aggregates_complete_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize_wmc, 'RESULTS', tmp_path)
    for seed in (1,2,3):
        write_run(tmp_path,seed)
    summarize_wmc.main()
    text=(tmp_path/'wmc_summary.md').read_text(encoding='utf-8')
    assert '86.00 ± 1.00%' in text
    assert '898339' in text
