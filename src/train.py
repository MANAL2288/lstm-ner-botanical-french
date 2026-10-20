"""Train / evaluate a recurrent NER tagger on FloraNER.

Example:
    python src/train.py --task fine --cell vanilla --hidden-size 128 --optimizer adam --lr 1e-3

Each run writes results/runs/<run_name>/{config,history,metrics}.json and appends one line
to results/summary.jsonl so that all experiments of the group can be compared side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import PAD_LABEL_ID, ROOT, collate_fn, load_task
from metrics import entity_scores
from model import NERTagger

RESULTS = Path(os.environ.get("FLORANER_RESULTS", ROOT / "results"))  # e.g. a Google Drive folder on Colab


@dataclass
class Config:
    task: str = "fine"
    cell: str = "vanilla"
    # architecture
    emb_dim: int = 100
    case_dim: int = 8
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.3
    init: str = "default"          # default | xavier | orthogonal
    forget_bias: float = 0.0
    lowercase: bool = True
    min_freq: int = 1
    # optimisation
    optimizer: str = "adam"        # adam | adamw | sgd | rmsprop
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    batch_size: int = 32
    epochs: int = 60               # upper bound
    patience: int = 10             # stop after 10 epochs without val F1 gain (0 = never); best val epoch kept
    clip: float = 5.0
    scheduler: str = "none"        # none | plateau
    # bookkeeping
    seed: int = 1
    run_name: str = ""
    tensorboard: bool = False
    save_model: bool = False
    max_train_batches: int = 0     # >0: debug on a few batches only
    threads: int = 0
    tags: list = field(default_factory=list)  # free labels, e.g. ["study:optimizer"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(cfg: Config, params):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay or 1e-2)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


@torch.no_grad()
def evaluate(model, loader, tag_vocab, device, criterion) -> dict:
    model.eval()
    y_true, y_pred, total_loss, n_tok, t0 = [], [], 0.0, 0, time.perf_counter()
    for W, C, Y, mask in loader:
        W, C, Y, mask = W.to(device), C.to(device), Y.to(device), mask.to(device)
        logits = model(W, C, mask)
        total_loss += criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1)).item() * mask.sum().item()
        n_tok += mask.sum().item()
        pred = logits.argmax(-1)
        for p, y, m in zip(pred.tolist(), Y.tolist(), mask.tolist()):
            L = sum(m)
            y_true.append([tag_vocab.itos[i] for i in y[:L]])
            y_pred.append([tag_vocab.itos[i] for i in p[:L]])
    elapsed = time.perf_counter() - t0
    flat_t = [t for s in y_true for t in s]
    flat_p = [t for s in y_pred for t in s]
    scores = entity_scores(y_true, y_pred)
    return {
        "loss": total_loss / n_tok,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
        "macro_f1": scores["macro_f1"],
        "token_accuracy": sum(a == b for a, b in zip(flat_t, flat_p)) / len(flat_t),
        "per_type": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in scores["per_type"].items()},
        "tokens_per_sec": n_tok / elapsed,
    }


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.run_name = cfg.run_name or (f"{cfg.task}_{cfg.cell}_h{cfg.hidden_size}_l{cfg.num_layers}"
                                    f"{'_bi' if cfg.bidirectional else ''}_{cfg.optimizer}_lr{cfg.lr}"
                                    f"_bs{cfg.batch_size}_{cfg.init}_s{cfg.seed}")
    out_dir = RESULTS / "runs" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data, word_vocab, tag_vocab = load_task(cfg.task, cfg.lowercase, cfg.min_freq)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(data["train"], cfg.batch_size, shuffle=True, collate_fn=collate_fn, generator=g)
    val_loader = DataLoader(data["val"], 64, collate_fn=collate_fn)
    test_loader = DataLoader(data["test"], 64, collate_fn=collate_fn)

    model = NERTagger(len(word_vocab), len(tag_vocab), cell=cfg.cell, emb_dim=cfg.emb_dim, case_dim=cfg.case_dim,
                      hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, bidirectional=cfg.bidirectional,
                      dropout=cfg.dropout, init=cfg.init, forget_bias=cfg.forget_bias).to(device)
    n_params = model.parameter_counts()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)
    optimizer = make_optimizer(cfg, model.parameters())
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
                 if cfg.scheduler == "plateau" else None)
    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(RESULTS / "tensorboard" / cfg.run_name)

    print(f"[{cfg.run_name}] device={device} params={n_params} train_segments={len(data['train'])}")
    history, best_f1, best_state, best_epoch, bad_epochs = [], -1.0, None, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, train_loss, n_tok = time.perf_counter(), 0.0, 0
        for b, (W, C, Y, mask) in enumerate(train_loader):
            if cfg.max_train_batches and b >= cfg.max_train_batches:
                break
            W, C, Y, mask = W.to(device), C.to(device), Y.to(device), mask.to(device)
            logits = model(W, C, mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            if cfg.clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            train_loss += loss.item() * mask.sum().item()
            n_tok += mask.sum().item()
        epoch_time = time.perf_counter() - t0
        val = evaluate(model, val_loader, tag_vocab, device, criterion)
        row = {"epoch": epoch, "train_loss": train_loss / n_tok, "val_loss": val["loss"], "val_f1": val["f1"],
               "val_macro_f1": val["macro_f1"], "val_precision": val["precision"], "val_recall": val["recall"],
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": epoch_time}
        history.append(row)
        print(f"  epoch {epoch:2d} | train_loss {row['train_loss']:.4f} | val_loss {val['loss']:.4f} "
              f"| val F1 {val['f1']:.4f} (macro {val['macro_f1']:.4f}) | {epoch_time:.1f}s")
        if writer:
            for k, v in row.items():
                if k != "epoch":
                    writer.add_scalar(k, v, epoch)
        if scheduler:
            scheduler.step(val["f1"])
        if val["f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = val["f1"], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if cfg.patience and bad_epochs >= cfg.patience:
                print(f"  early stopping (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    val = evaluate(model, val_loader, tag_vocab, device, criterion)
    test = evaluate(model, test_loader, tag_vocab, device, criterion)
    print(f"  BEST epoch {best_epoch} | val F1 {val['f1']:.4f} | TEST F1 {test['f1']:.4f} "
          f"(P {test['precision']:.4f} R {test['recall']:.4f} macro {test['macro_f1']:.4f})")

    epoch_times = [h["epoch_time_s"] for h in history]
    metrics = {"best_epoch": best_epoch, "epochs_run": len(history), "params": n_params,
               "mean_epoch_time_s": sum(epoch_times) / len(epoch_times), "device": str(device),
               "val": val, "test": test}
    for name, obj in (("config", asdict(cfg)), ("history", history), ("metrics", metrics)):
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    if cfg.save_model:
        torch.save({"state_dict": best_state, "config": asdict(cfg), "word_vocab": word_vocab.itos,
                    "tag_vocab": tag_vocab.itos}, out_dir / "model.pt")
    summary = {**asdict(cfg), "params_total": n_params["total"], "params_recurrent": n_params["recurrent"],
               "best_epoch": best_epoch, "mean_epoch_time_s": metrics["mean_epoch_time_s"], "device": str(device),
               **{f"val_{k}": val[k] for k in ("precision", "recall", "f1", "macro_f1")},
               **{f"test_{k}": test[k] for k in ("precision", "recall", "f1", "macro_f1", "token_accuracy",
                                                  "tokens_per_sec")}}
    with open(RESULTS / "summary.jsonl", "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    if writer:
        writer.close()
    return summary


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in Config.__dataclass_fields__.values():
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.name == "tags":
            p.add_argument(flag, nargs="*", default=[])
        else:
            typ = {"int": int, "float": float, "str": str}[f.type if isinstance(f.type, str) else f.type.__name__]
            p.add_argument(flag, type=typ, default=f.default)
    return Config(**vars(p.parse_args()))


if __name__ == "__main__":
    run(parse_args())
