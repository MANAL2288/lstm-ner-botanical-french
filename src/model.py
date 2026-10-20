"""Sequence layer and NER tagger built on top of the custom cells."""
from __future__ import annotations

import torch
from torch import nn

from cells import make_cell
from data import CASES


class RecurrentLayer(nn.Module):
    """Unrolls one cell over time (optionally a second one right-to-left).

    Padding is handled with the mask: on a padded position the state is carried over
    unchanged and the output is zero. This matters for the backward direction, which
    would otherwise start reading the sentence from the <PAD> tokens.
    """

    def __init__(self, cell: str, input_size: int, hidden_size: int, bidirectional: bool = True, **cell_kwargs):
        super().__init__()
        self.fwd = make_cell(cell, input_size, hidden_size, **cell_kwargs)
        self.bwd = make_cell(cell, input_size, hidden_size, **cell_kwargs) if bidirectional else None
        self.output_size = self.fwd.output_size * (2 if bidirectional else 1)

    @staticmethod
    def _unroll(cell, x, mask, reverse: bool):
        B, T, _ = x.shape
        xp = cell.input_projection(x)  # input terms W_x x_t + b for every t in one matmul
        state = cell.init_state(B, x.device)
        outputs = [None] * T
        for t in (range(T - 1, -1, -1) if reverse else range(T)):
            h, new_state = cell(xp[:, t], state)
            m = mask[:, t].unsqueeze(-1)
            state = tuple(torch.where(m, n, s) for n, s in zip(new_state, state))
            outputs[t] = h * m
        return torch.stack(outputs, dim=1)  # (B, T, output_size)

    def forward(self, x, mask):
        out = self._unroll(self.fwd, x, mask, reverse=False)
        if self.bwd is not None:
            out = torch.cat([out, self._unroll(self.bwd, x, mask, reverse=True)], dim=-1)
        return out

    def reset_parameters(self, scheme: str, forget_bias: float):
        for cell in (self.fwd, self.bwd):
            if cell is not None:
                cell.reset_parameters(scheme, forget_bias)


class NERTagger(nn.Module):
    """word emb ⊕ casing emb -> dropout -> [RecurrentLayer -> dropout] x L -> Linear -> tag logits."""

    def __init__(self, vocab_size: int, num_tags: int, cell: str = "vanilla", emb_dim: int = 100,
                 case_dim: int = 8, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.3, init: str = "default", forget_bias: float = 0.0, **cell_kwargs):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.case_emb = nn.Embedding(len(CASES), case_dim, padding_idx=0) if case_dim > 0 else None
        self.dropout = nn.Dropout(dropout)
        layers, in_size = [], emb_dim + max(case_dim, 0)
        for _ in range(num_layers):
            layer = RecurrentLayer(cell, in_size, hidden_size, bidirectional, **cell_kwargs)
            layer.reset_parameters(init, forget_bias)
            layers.append(layer)
            in_size = layer.output_size
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(in_size, num_tags)

    def forward(self, words, cases, mask):
        x = self.word_emb(words)
        if self.case_emb is not None:
            x = torch.cat([x, self.case_emb(cases)], dim=-1)
        x = self.dropout(x)
        for layer in self.layers:
            x = self.dropout(layer(x, mask))
        return self.classifier(x)  # (B, T, num_tags)

    def parameter_counts(self) -> dict[str, int]:
        def n(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {"total": n(self), "recurrent": sum(n(l) for l in self.layers),
                "embeddings": n(self.word_emb) + (n(self.case_emb) if self.case_emb else 0),
                "classifier": n(self.classifier)}
