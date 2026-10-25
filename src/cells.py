"""Recurrent cells implemented from scratch with nn.Linear (nn.LSTM / nn.LSTMCell are not used).

Every cell follows the same small interface so that the sequence layer, the tagger and
the training script are shared by all the variants studied by the group:

    xp    = cell.input_projection(x)        # (B, T, D) -> (B, T, G): input terms for all t at once
    state = cell.init_state(B, device)      # tuple of tensors
    h, state = cell(xp[:, t], state)        # one recurrent step
    cell.output_size                        # size of h (what the next layer / classifier sees)

To add a variant (CIFG, LSTMP, WMC), subclass ``RecurrentCell`` and register it in ``CELLS``.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class RecurrentCell(nn.Module):
    """Base class: one recurrent step, plus a hoisted input projection."""

    output_size: int

    def input_projection(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_state(self, batch_size: int, device) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def reset_parameters(self, scheme: str = "default", forget_bias: float = 0.0) -> None:
        raise NotImplementedError

    def recurrent_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VanillaLSTMCell(RecurrentCell):
    r"""Vanilla LSTM, equations (1)-(6) of the guidelines.

        i_t = σ(W_hi h_{t-1} + W_xi x_t + b_i)
        f_t = σ(W_hf h_{t-1} + W_xf x_t + b_f)
        o_t = σ(W_ho h_{t-1} + W_xo x_t + b_o)
        g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)

    The four W_x* (resp. W_h*) matrices are stacked in one nn.Linear of output size 4H,
    in the order (i, f, g, o). This is the same maths as eight separate matrices, but a
    single matrix product per step. Biases live in ``x2h`` only (one bias per gate).
    """

    GATES = ("i", "f", "g", "o")

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)               # W_x[i,f,g,o], b[i,f,g,o]
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)  # W_h[i,f,g,o]
        self.reset_parameters()

    def input_projection(self, x):
        return self.x2h(x)

    def init_state(self, batch_size, device):
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()

    def forward(self, xp_t, state):
        h_prev, c_prev = state
        gates = xp_t + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    @torch.no_grad()
    def reset_parameters(self, scheme: str = "default", forget_bias: float = 0.0):
        H = self.hidden_size
        if scheme == "default":  # same as torch.nn.LSTM: U(-1/sqrt(H), 1/sqrt(H)) everywhere
            bound = 1.0 / math.sqrt(H)
            for p in self.parameters():
                nn.init.uniform_(p, -bound, bound)
        elif scheme in ("xavier", "orthogonal"):
            for k in range(4):  # initialise each gate block as its own (H x D) matrix
                nn.init.xavier_uniform_(self.x2h.weight[k * H:(k + 1) * H])
                if scheme == "xavier":
                    nn.init.xavier_uniform_(self.h2h.weight[k * H:(k + 1) * H])
                else:
                    nn.init.orthogonal_(self.h2h.weight[k * H:(k + 1) * H])
            nn.init.zeros_(self.x2h.bias)
        else:
            raise ValueError(f"unknown init scheme: {scheme}")
        if forget_bias:
            self.x2h.bias[H:2 * H].fill_(forget_bias)  # remember by default at the start of training


class WMCCell(VanillaLSTMCell):
    r"""Working Memory Connections, guidelines equations (7)-(12).

    Input and recurrent projections use the vanilla (i, f, g, o) layout.
    Memory projections are full H x H matrices, without extra biases:
    i/f use tanh(W_c c_prev), whereas o uses tanh(W_co c_new).
    """

    def __init__(self, input_size: int, hidden_size: int):
        # Build explicitly: the parent constructor calls reset_parameters before
        # the additional memory projections would have been created.
        RecurrentCell.__init__(self)
        self.input_size = input_size
        self.hidden_size = self.output_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.c2if = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.c2o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_parameters()

    def forward(self, xp_t, state):
        h_prev, c_prev = state
        i, f, g, o = (xp_t + self.h2h(h_prev)).chunk(4, dim=-1)
        ci, cf = torch.tanh(self.c2if(c_prev)).chunk(2, dim=-1)
        i = torch.sigmoid(i + ci)
        f = torch.sigmoid(f + cf)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        o = torch.sigmoid(o + torch.tanh(self.c2o(c)))
        h = o * torch.tanh(c)
        return h, (h, c)

    @torch.no_grad()
    def reset_parameters(self, scheme: str = "default", forget_bias: float = 0.0):
        super().reset_parameters(scheme, forget_bias)
        # The parent's default scheme already initializes every parameter,
        # including memory weights, with the common uniform distribution.
        if scheme in ("xavier", "orthogonal"):
            init = nn.init.xavier_uniform_ if scheme == "xavier" else nn.init.orthogonal_
            for block in self.c2if.weight.chunk(2, dim=0):
                init(block)
            init(self.c2o.weight)


class CIFGCell(RecurrentCell):
    r"""Coupled Input-Forget Gate LSTM (Greff et al., 2017).

        f_t = σ(W_hf h_{t-1} + W_xf x_t + b_f)
        i_t = 1 - f_t                                  <- the only change w.r.t. the vanilla LSTM
        o_t = σ(W_ho h_{t-1} + W_xo x_t + b_o)
        g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)

    Three gate blocks (f, o, g) are stacked in one nn.Linear of output size 3H, in this
    order, instead of four: about 25% fewer recurrent parameters than the vanilla LSTM.
    """

    GATES = ("f", "o", "g")

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, hidden_size
        self.x2h = nn.Linear(input_size, 3 * hidden_size)               # W_x[f,o,g], b[f,o,g]
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)  # W_h[f,o,g]
        self.reset_parameters()

    def input_projection(self, x):
        return self.x2h(x)

    def init_state(self, batch_size, device):
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()

    def forward(self, xp_t, state):
        h_prev, c_prev = state
        gates = xp_t + self.h2h(h_prev)
        f, o, g = gates.chunk(3, dim=-1)
        f, o = torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + (1.0 - f) * g          # i_t = 1 - f_t
        h = o * torch.tanh(c)
        return h, (h, c)

    @torch.no_grad()
    def reset_parameters(self, scheme: str = "default", forget_bias: float = 0.0):
        H = self.hidden_size
        if scheme == "default":  # same convention as the vanilla cell: U(-1/sqrt(H), 1/sqrt(H))
            bound = 1.0 / math.sqrt(H)
            for p in self.parameters():
                nn.init.uniform_(p, -bound, bound)
        elif scheme in ("xavier", "orthogonal"):
            for k in range(3):  # one (H x D) block per gate
                nn.init.xavier_uniform_(self.x2h.weight[k * H:(k + 1) * H])
                if scheme == "xavier":
                    nn.init.xavier_uniform_(self.h2h.weight[k * H:(k + 1) * H])
                else:
                    nn.init.orthogonal_(self.h2h.weight[k * H:(k + 1) * H])
            nn.init.zeros_(self.x2h.bias)
        else:
            raise ValueError(f"unknown init scheme: {scheme}")
        if forget_bias:
            self.x2h.bias[0:H].fill_(forget_bias)  # f is the FIRST block here (second in the vanilla cell)


CELLS: dict[str, type[RecurrentCell]] = {
    "vanilla": VanillaLSTMCell,
    "cifg": CIFGCell,
    "wmc": WMCCell,
}


def make_cell(name: str, input_size: int, hidden_size: int, **kwargs) -> RecurrentCell:
    return CELLS[name](input_size, hidden_size, **kwargs)
