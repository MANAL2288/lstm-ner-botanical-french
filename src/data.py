"""FloraNER data pipeline: character spans -> BIO tokens -> padded PyTorch batches.

The raw FloraNER CSVs store entities as character offsets ``(start, end, label)``.
This module tokenizes each text, projects the spans onto tokens with the BIO
scheme, carves a document-level validation split out of the official train set
and exposes a ``Dataset`` / ``collate_fn`` pair shared by every model of the group.
"""
from __future__ import annotations

import ast
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# task name -> (file stem, annotation column)
TASKS = {
    "species": ("species", "Annotations"),
    "coarse": ("coarse_grained", "filtered_annotations"),
    "fine": ("fine_grained", "filtered_annotations"),
}

PAD, UNK = "<PAD>", "<UNK>"
PAD_LABEL_ID = -100  # ignored by nn.CrossEntropyLoss
VAL_RATIO = 0.10
SPLIT_SEED = 42

# Words and single punctuation marks. Splitting hyphens/apostrophes keeps 100% of
# the coarse/fine entity boundaries aligned with token boundaries.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

csv.field_size_limit(10**8)


# --------------------------------------------------------------------------- #
# Raw loading and BIO conversion
# --------------------------------------------------------------------------- #
def load_raw(task: str, split: str) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Return ``[(text, [(start, end, label), ...]), ...]`` for an official split."""
    stem, column = TASKS[task]
    folder = "Train" if split == "train" else "Test"
    path = RAW_DIR / folder / f"{split}_{stem}.csv"
    docs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cell = row[column].strip()
            spans = ast.literal_eval(cell) if cell else []
            docs.append((row["Text"], [tuple(s) for s in spans]))
    return docs


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


@dataclass
class ConversionStats:
    entities: int = 0
    snapped: int = 0  # span boundary fell inside a token -> extended to the whole token
    dropped_empty: int = 0  # span covers no token
    dropped_overlap: int = 0  # span overlaps an entity already placed


def spans_to_bio(text: str, spans, stats: ConversionStats | None = None):
    """Project character spans onto tokens. A token belongs to an entity if it overlaps it."""
    toks = tokenize(text)
    tags = ["O"] * len(toks)
    stats = stats if stats is not None else ConversionStats()
    for start, end, label in sorted(set(spans), key=lambda s: (s[0], -s[1])):
        stats.entities += 1
        idx = [i for i, (_, s, e) in enumerate(toks) if s < end and e > start]
        if not idx:
            stats.dropped_empty += 1
            continue
        if any(tags[i] != "O" for i in idx):
            stats.dropped_overlap += 1
            continue
        if toks[idx[0]][1] != start or toks[idx[-1]][2] != end:
            stats.snapped += 1
        tags[idx[0]] = f"B-{label}"
        for i in idx[1:]:
            tags[i] = f"I-{label}"
    return [t for t, _, _ in toks], tags, [(s, e) for _, s, e in toks]


def segment(tokens, tags, offsets, text, max_len: int):
    """Split a long document into chunks of at most ``max_len`` tokens.

    Cuts happen only between two tokens where no entity is interrupted, preferably
    after a clause boundary ('.', ';' or a run of 2+ spaces in the source text).
    Entity-level metrics are therefore identical on segments and on full documents.
    """
    if len(tokens) <= max_len:
        return [(tokens, tags)]

    def can_cut(i):  # cut between token i-1 and token i
        return not tags[i].startswith("I-")

    def is_boundary(i):
        gap = text[offsets[i - 1][1]:offsets[i][0]]
        return tokens[i - 1] in {".", ";"} or "  " in gap or "\n" in gap

    chunks, start = [], 0
    while len(tokens) - start > max_len:
        limit = start + max_len
        cut = next((i for i in range(limit, start, -1) if can_cut(i) and is_boundary(i)), None)
        if cut is None:
            cut = next(i for i in range(limit, start, -1) if can_cut(i))
        chunks.append((tokens[start:cut], tags[start:cut]))
        start = cut
    chunks.append((tokens[start:], tags[start:]))
    return chunks


def build_processed(task: str, max_len: int = 128) -> dict:
    """Write ``data/processed/<task>/{train,val,test}.jsonl`` and return conversion stats."""
    out_dir = PROCESSED_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)

    train_docs = load_raw(task, "train")
    test_docs = load_raw(task, "test")
    order = list(range(len(train_docs)))
    random.Random(SPLIT_SEED).shuffle(order)
    n_val = round(VAL_RATIO * len(order))
    val_ids, train_ids = set(order[:n_val]), set(order[n_val:])

    splits = {
        "train": [(i, train_docs[i]) for i in sorted(train_ids)],
        "val": [(i, train_docs[i]) for i in sorted(val_ids)],
        "test": list(enumerate(test_docs)),
    }
    report = {}
    for name, docs in splits.items():
        stats = ConversionStats()
        n_seg = 0
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for doc_id, (text, spans) in docs:
                tokens, tags, offsets = spans_to_bio(text, spans, stats)
                if not tokens:
                    continue
                for k, (seg_toks, seg_tags) in enumerate(segment(tokens, tags, offsets, text, max_len)):
                    f.write(json.dumps({"doc_id": doc_id, "seg": k, "tokens": seg_toks, "tags": seg_tags},
                                       ensure_ascii=False) + "\n")
                    n_seg += 1
        report[name] = {"documents": len(docs), "segments": n_seg, **stats.__dict__}
    with open(out_dir / "conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def read_jsonl(task: str, split: str) -> list[dict]:
    with open(PROCESSED_DIR / task / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------- #
# Vocabularies, casing feature, Dataset
# --------------------------------------------------------------------------- #
CASES = ["<pad>", "lower", "title", "upper", "mixed", "numeric", "has_digit", "punct"]


def casing(token: str) -> int:
    """Small orthographic feature: lowercased embeddings lose capitalisation (genus names)."""
    if token.isdigit():
        c = "numeric"
    elif any(ch.isdigit() for ch in token):
        c = "has_digit"
    elif not any(ch.isalnum() for ch in token):
        c = "punct"
    elif token.islower():
        c = "lower"
    elif token.isupper():
        c = "upper"
    elif token[0].isupper() and token[1:].islower():
        c = "title"
    else:
        c = "mixed"
    return CASES.index(c)


@dataclass
class Vocab:
    itos: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi.get(UNK, 0))

    @classmethod
    def build(cls, sequences, min_freq: int = 1, specials=(PAD, UNK)):
        counts = Counter(tok for seq in sequences for tok in seq)
        return cls(list(specials) + sorted(t for t, c in counts.items() if c >= min_freq and t not in specials))


def normalize(token: str, lowercase: bool) -> str:
    token = token.lower() if lowercase else token
    return re.sub(r"\d", "0", token)  # all digits share one shape: "12 mm" ~ "35 mm"


class NERDataset(Dataset):
    def __init__(self, records, word_vocab: Vocab, tag_vocab: Vocab, lowercase: bool = True):
        self.records = records
        self.word_vocab, self.tag_vocab, self.lowercase = word_vocab, tag_vocab, lowercase

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        words = torch.tensor([self.word_vocab[normalize(t, self.lowercase)] for t in r["tokens"]])
        cases = torch.tensor([casing(t) for t in r["tokens"]])
        tags = torch.tensor([self.tag_vocab.stoi[t] for t in r["tags"]])
        return words, cases, tags


def collate_fn(batch):
    """Right-pad to the longest sequence of the batch; labels are padded with -100."""
    words, cases, tags = zip(*batch)
    lengths = torch.tensor([len(w) for w in words])
    T = int(lengths.max())
    W = torch.zeros(len(batch), T, dtype=torch.long)
    C = torch.zeros(len(batch), T, dtype=torch.long)
    Y = torch.full((len(batch), T), PAD_LABEL_ID, dtype=torch.long)
    for i, (w, c, y) in enumerate(zip(words, cases, tags)):
        W[i, : len(w)], C[i, : len(c)], Y[i, : len(y)] = w, c, y
    mask = torch.arange(T)[None, :] < lengths[:, None]
    return W, C, Y, mask


def load_task(task: str, lowercase: bool = True, min_freq: int = 1):
    """Return train/val/test datasets plus vocabularies (built on the train split only)."""
    recs = {s: read_jsonl(task, s) for s in ("train", "val", "test")}
    word_vocab = Vocab.build([[normalize(t, lowercase) for t in r["tokens"]] for r in recs["train"]], min_freq)
    labels = sorted({t for r in recs["train"] for t in r["tags"]} - {"O"}, key=lambda t: (t[2:], t[0]))
    tag_vocab = Vocab(["O"] + labels)
    data = {s: NERDataset(r, word_vocab, tag_vocab, lowercase) for s, r in recs.items()}
    return data, word_vocab, tag_vocab


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert FloraNER CSVs into BIO jsonl splits.")
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--max-len", type=int, default=128)
    args = p.parse_args()
    for t in args.tasks:
        print(t, json.dumps(build_processed(t, args.max_len), indent=1))
