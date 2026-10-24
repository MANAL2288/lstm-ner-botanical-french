"""Independent equation and integration checks for the duo's WMC cell."""
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cells import WMCCell, VanillaLSTMCell
from model import RecurrentLayer, NERTagger


def reference(cell, x, h, c):
    H = cell.hidden_size
    # Separate affine operations directly mirror equations 7-12.
    def affine(k):
        sl = slice(k * H, (k + 1) * H)
        return F.linear(x, cell.x2h.weight[sl], cell.x2h.bias[sl]) + F.linear(h, cell.h2h.weight[sl])
    i = torch.sigmoid(affine(0) + torch.tanh(F.linear(c, cell.c2if.weight[:H])))
    f = torch.sigmoid(affine(1) + torch.tanh(F.linear(c, cell.c2if.weight[H:])))
    g = torch.tanh(affine(2))
    new_c = f * c + i * g
    o = torch.sigmoid(affine(3) + torch.tanh(F.linear(new_c, cell.c2o.weight)))
    return o * torch.tanh(new_c), new_c


def test_equations_and_gradients():
    torch.manual_seed(13)
    cell = WMCCell(3, 4).double()
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    h = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    c = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    actual, state = cell(cell.input_projection(x), (h, c))
    expected, expected_c = reference(cell, x, h, c)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state[1], expected_c)
    inputs = (x, h, c, *cell.parameters())
    ga = torch.autograd.grad(actual.square().sum() + state[1].square().sum(), inputs, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum() + expected_c.square().sum(), inputs)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)
        assert torch.isfinite(a).all()


def test_output_gate_uses_updated_memory():
    cell = WMCCell(1, 1)
    with torch.no_grad():
        for p in cell.parameters():
            p.zero_()
        cell.x2h.bias[2] = 1.0
        cell.c2o.weight.fill_(2.0)
    h, (_, c) = cell(cell.input_projection(torch.zeros(1, 1)), cell.init_state(1, 'cpu'))
    expected = torch.sigmoid(torch.tanh(2 * c)) * torch.tanh(c)
    torch.testing.assert_close(h, expected)
    assert not torch.allclose(h, 0.5 * torch.tanh(c))


def test_zero_memory_connections_reduce_to_vanilla():
    torch.manual_seed(9)
    wmc, vanilla = WMCCell(3, 5), VanillaLSTMCell(3, 5)
    with torch.no_grad():
        vanilla.x2h.load_state_dict(wmc.x2h.state_dict())
        vanilla.h2h.load_state_dict(wmc.h2h.state_dict())
        wmc.c2if.weight.zero_()
        wmc.c2o.weight.zero_()
    x = torch.randn(2, 7, 3)
    sw, sv = wmc.init_state(2, 'cpu'), vanilla.init_state(2, 'cpu')
    for t in range(7):
        hw, sw = wmc(wmc.input_projection(x[:, t]), sw)
        hv, sv = vanilla(vanilla.input_projection(x[:, t]), sv)
        torch.testing.assert_close(hw, hv)
        torch.testing.assert_close(sw[1], sv[1])


@pytest.mark.parametrize('scheme', ['default', 'xavier', 'orthogonal'])
def test_initialization_and_parameter_count(scheme):
    cell = WMCCell(108, 128)
    cell.reset_parameters(scheme, forget_bias=1.0)
    assert cell.recurrent_parameter_count() == 4 * (108 * 128 + 128 * 128 + 128) + 3 * 128**2
    torch.testing.assert_close(cell.x2h.bias[128:256], torch.ones(128))
    assert all(torch.isfinite(p).all() for p in cell.parameters())


@pytest.mark.parametrize('bidirectional', [False, True])
def test_padding_invariance(bidirectional):
    torch.manual_seed(8)
    layer = RecurrentLayer('wmc', 3, 5, bidirectional)
    short = torch.randn(1, 4, 3)
    alone = layer(short, torch.ones(1, 4, dtype=torch.bool))
    batch = torch.cat([F.pad(short, (0, 0, 0, 3)), torch.randn(1, 7, 3)])
    mask = torch.tensor([[True]*4 + [False]*3, [True]*7])
    padded = layer(batch, mask)
    torch.testing.assert_close(alone[0], padded[0, :4])
    assert padded[0, 4:].eq(0).all()


def test_tagger_backward_and_counts():
    model = NERTagger(5524, 19, cell='wmc')
    counts = model.parameter_counts()
    assert counts['recurrent'] == 340992
    assert counts['total'] == sum(p.numel() for p in model.parameters())
    small = NERTagger(20, 5, cell='wmc', emb_dim=4, hidden_size=6, num_layers=2)
    logits = small(torch.tensor([[1, 2, 3], [4, 0, 0]]), torch.ones(2, 3, dtype=torch.long),
                   torch.tensor([[True, True, True], [True, False, False]]))
    loss = F.cross_entropy(logits.reshape(-1, 5), torch.tensor([0, 1, 2, 3, -100, -100]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
