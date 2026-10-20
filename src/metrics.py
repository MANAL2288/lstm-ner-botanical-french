"""Entity-level NER metrics (conlleval / seqeval default semantics), no external dependency.

An entity is a maximal chunk (type, start, end). A predicted entity is correct only if its type and
both boundaries match a gold entity exactly. As in conlleval, an I-X that does not continue an X chunk
starts a new chunk. tests/test_model.py checks that the numbers equal seqeval's.
"""
from __future__ import annotations

from collections import Counter


def get_entities(tags: list[str]) -> list[tuple[str, int, int]]:
    """BIO tags -> [(type, start, end_inclusive), ...]."""
    entities, start, etype = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, t = ("O", None) if tag == "O" else (tag[0], tag[2:])
        if start is not None and (prefix in ("B", "O") or t != etype):
            entities.append((etype, start, i - 1))
            start = None
        if prefix in ("B", "I") and start is None:
            start, etype = i, t
    return entities


def _prf(tp: int, n_pred: int, n_true: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def entity_scores(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """Micro/macro precision, recall, F1 and a per-type report over a corpus of sentences."""
    true = {(k, *e) for k, seq in enumerate(y_true) for e in get_entities(seq)}
    pred = {(k, *e) for k, seq in enumerate(y_pred) for e in get_entities(seq)}
    tp_by, true_by, pred_by = Counter(e[1] for e in true & pred), Counter(e[1] for e in true), \
        Counter(e[1] for e in pred)
    per_type = {}
    for t in sorted(set(true_by) | set(pred_by)):
        p, r, f = _prf(tp_by[t], pred_by[t], true_by[t])
        per_type[t] = {"precision": p, "recall": r, "f1-score": f, "support": true_by[t]}
    p, r, f = _prf(len(true & pred), len(pred), len(true))
    macro = sum(v["f1-score"] for v in per_type.values()) / len(per_type) if per_type else 0.0
    return {"precision": p, "recall": r, "f1": f, "macro_f1": macro, "per_type": per_type}
