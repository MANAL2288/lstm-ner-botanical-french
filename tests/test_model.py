"""Sanity checks for the from-scratch LSTM. Run: python -m pytest tests -q

nn.LSTMCell is used here ONLY as a numerical reference, never inside the model.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cells import VanillaLSTMCell  # noqa: E402
from data import collate_fn, spans_to_bio  # noqa: E402
from model import NERTagger, RecurrentLayer  # noqa: E402


def test_vanilla_cell_matches_pytorch_lstmcell():
    torch.manual_seed(0)
    D, H, B, T = 7, 5, 3, 12
    ours, ref = VanillaLSTMCell(D, H), nn.LSTMCell(D, H)
    with torch.no_grad():  # same gate order (i, f, g, o) -> copy weights directly
        ref.weight_ih.copy_(ours.x2h.weight)
        ref.bias_ih.copy_(ours.x2h.bias)
        ref.weight_hh.copy_(ours.h2h.weight)
        ref.bias_hh.zero_()
    x = torch.randn(B, T, D)
    xp = ours.input_projection(x)
    state, (h_ref, c_ref) = ours.init_state(B, "cpu"), (torch.zeros(B, H), torch.zeros(B, H))
    for t in range(T):
        h, state = ours(xp[:, t], state)
        h_ref, c_ref = ref(x[:, t], (h_ref, c_ref))
        assert torch.allclose(h, h_ref, atol=1e-6) and torch.allclose(state[1], c_ref, atol=1e-6)


@pytest.mark.parametrize("bidirectional", [False, True])
def test_padding_does_not_change_outputs(bidirectional):
    torch.manual_seed(0)
    layer = RecurrentLayer("vanilla", 4, 6, bidirectional=bidirectional)
    short = torch.randn(1, 5, 4)
    alone = layer(short, torch.ones(1, 5, dtype=torch.bool))
    batch = torch.cat([torch.cat([short, torch.randn(1, 4, 4)], 1), torch.randn(1, 9, 4)])
    mask = torch.tensor([[True] * 5 + [False] * 4, [True] * 9])
    padded = layer(batch, mask)
    assert torch.allclose(alone[0], padded[0, :5], atol=1e-6)
    assert torch.all(padded[0, 5:] == 0)


def test_parameter_count_matches_formula():
    D, H = 108, 128
    cell = VanillaLSTMCell(D, H)
    assert cell.recurrent_parameter_count() == 4 * (D * H + H * H + H)


def test_forward_shapes_and_loss_ignores_padding():
    torch.manual_seed(0)
    model = NERTagger(vocab_size=50, num_tags=5, hidden_size=8, emb_dim=6, case_dim=3, num_layers=2)
    batch = [(torch.tensor([3, 4, 5]), torch.tensor([1, 1, 2]), torch.tensor([0, 1, 2])),
             (torch.tensor([6]), torch.tensor([1]), torch.tensor([3]))]
    W, C, Y, mask = collate_fn(batch)
    logits = model(W, C, mask)
    assert logits.shape == (2, 3, 5) and Y[1, 1:].eq(-100).all()
    loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, 5), Y.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)


def test_entity_scores_match_seqeval():
    seqeval = pytest.importorskip("seqeval.metrics")  # reference only; not needed to train
    import random

    from metrics import entity_scores, get_entities

    assert get_entities(["B-A", "I-A", "O", "I-B", "I-A", "B-A"]) == [("A", 0, 1), ("B", 3, 3), ("A", 4, 4),
                                                                       ("A", 5, 5)]
    rng = random.Random(0)
    tags = ["O", "O", "O", "B-A", "I-A", "B-B", "I-B", "B-C"]
    y_true = [[rng.choice(tags) for _ in range(rng.randint(1, 15))] for _ in range(2000)]
    y_pred = [[t if rng.random() < 0.7 else rng.choice(tags) for t in s] for s in y_true]
    ours = entity_scores(y_true, y_pred)
    assert ours["precision"] == pytest.approx(seqeval.precision_score(y_true, y_pred))
    assert ours["recall"] == pytest.approx(seqeval.recall_score(y_true, y_pred))
    assert ours["f1"] == pytest.approx(seqeval.f1_score(y_true, y_pred))
    assert ours["macro_f1"] == pytest.approx(seqeval.f1_score(y_true, y_pred, average="macro"))
    report = seqeval.classification_report(y_true, y_pred, output_dict=True)
    for t, d in ours["per_type"].items():
        assert d["f1-score"] == pytest.approx(report[t]["f1-score"]) and d["support"] == report[t]["support"]


def test_spans_to_bio():
    text = "feuilles obovées-elliptiques, 20 cm de longueur"
    spans = [(0, 8, "ORGANE"), (9, 28, "FORME"), (30, 47, "MESURE")]
    tokens, tags, _ = spans_to_bio(text, spans)
    assert tokens == ["feuilles", "obovées", "-", "elliptiques", ",", "20", "cm", "de", "longueur"]
    assert tags == ["B-ORGANE", "B-FORME", "I-FORME", "I-FORME", "O",
                    "B-MESURE", "I-MESURE", "I-MESURE", "I-MESURE"]
