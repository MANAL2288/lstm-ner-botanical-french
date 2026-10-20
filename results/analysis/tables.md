# Hyperparameter studies — vanilla LSTM, FloraNER fine-grained

Entity-level F1 (%), mean ± std over seeds. **Δ** = difference with the baseline trained with the
same seed (paired), mean ± std over seeds. Choice on **validation**, **test** only reported.
† = incomplete (fewer than 3 seeds): not eligible as best configuration.

**Best configuration (highest mean Δ val among complete ones): lr=0.01** — test F1 94.35 ± 0.69 (Δ +0.35 ± 0.48), macro F1 92.31 ± 0.40.

## optimizer

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
| optimizer=adamw | 3 | 93.98 ± 1.22 | -0.08 ± 0.11 | 93.80 ± 1.32 | -0.21 ± 0.23 | 89.98 ± 0.92 | 800,035 (242,688) | 12.1 | 47.3 |
| optimizer=rmsprop | 3 | 94.05 ± 1.15 | -0.02 ± 0.29 | 93.88 ± 0.97 | -0.13 ± 0.13 | 89.99 ± 0.50 | 800,035 (242,688) | 12.0 | 53.3 |
| optimizer=sgd, lr=0.1 | 3 | 90.96 ± 0.03 | -3.11 ± 1.24 | 91.31 ± 0.16 | -2.70 ± 1.21 | 78.80 ± 0.13 | 800,035 (242,688) | 11.9 | 59.3 |

## lr

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| lr=0.01 **(best)** | 3 | 94.37 ± 0.86 | +0.30 ± 0.43 | 94.35 ± 0.69 | +0.35 ± 0.48 | 92.31 ± 0.40 | 800,035 (242,688) | 11.9 | 22.3 |
| lr=0.003 | 3 | 94.11 ± 1.11 | +0.05 ± 0.19 | 93.90 ± 1.34 | -0.11 ± 0.23 | 90.84 ± 0.32 | 800,035 (242,688) | 12.0 | 23.3 |
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
| lr=0.0003 | 3 | 91.93 ± 1.49 | -2.14 ± 0.26 | 91.91 ± 1.44 | -2.10 ± 0.33 | 81.98 ± 2.78 | 800,035 (242,688) | 11.9 | 59.7 |

## batch_size

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| batch_size=16 | 3 | 94.27 ± 1.10 | +0.20 ± 0.16 | 94.06 ± 1.06 | +0.05 ± 0.11 | 90.91 ± 0.26 | 800,035 (242,688) | 23.1 | 44.7 |
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
| batch_size=64 | 3 | 93.66 ± 1.47 | -0.41 ± 0.23 | 93.45 ± 1.40 | -0.55 ± 0.31 | 89.42 ± 0.88 | 800,035 (242,688) | 6.1 | 54.3 |
| batch_size=128 | 3 | 92.81 ± 1.48 | -1.26 ± 0.25 | 92.94 ± 1.47 | -1.07 ± 0.38 | 83.59 ± 2.76 | 800,035 (242,688) | 4.0 | 59.3 |

## init

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
| init=xavier | 3 | 94.16 ± 0.78 | +0.09 ± 0.48 | 93.90 ± 0.82 | -0.10 ± 0.29 | 90.63 ± 0.73 | 800,035 (242,688) | 11.5 | 52.0 |
| init=orthogonal | 3 | 94.14 ± 0.98 | +0.07 ± 0.32 | 93.98 ± 0.93 | -0.02 ± 0.18 | 90.11 ± 0.44 | 800,035 (242,688) | 11.5 | 51.3 |
| init=orthogonal, forget_bias=1.0 | 3 | 94.26 ± 0.92 | +0.19 ± 0.33 | 93.90 ± 1.05 | -0.11 ± 0.08 | 90.11 ± 0.70 | 800,035 (242,688) | 11.4 | 53.0 |

## hidden_size

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| hidden_size=64 | 3 | 93.58 ± 1.58 | -0.49 ± 0.37 | 93.57 ± 1.60 | -0.44 ± 0.50 | 89.45 ± 1.38 | 643,491 (88,576) | 11.6 | 58.7 |
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
| hidden_size=256 | 3 | 93.90 ± 1.21 | -0.16 ± 0.06 | 93.47 ± 1.04 | -0.53 ± 0.10 | 89.41 ± 0.63 | 1,309,731 (747,520) | 11.7 | 41.3 |

## num_layers

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
| num_layers=2 | 3 | 94.26 ± 1.11 | +0.20 ± 0.27 | 94.03 ± 1.02 | +0.03 ± 0.08 | 90.76 ± 0.17 | 1,194,275 (636,928) | 22.9 | 50.0 |

## direction

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| bidirectional=False | 3 | 90.34 ± 1.43 | -3.73 ± 0.30 | 89.88 ± 1.24 | -4.12 ± 0.25 | 82.95 ± 0.61 | 676,259 (121,344) | 6.2 | 57.3 |
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |

## dropout

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| dropout=0.0 | 3 | 92.70 ± 1.73 | -1.37 ± 0.50 | 92.36 ± 1.88 | -1.64 ± 0.78 | 88.14 ± 1.35 | 800,035 (242,688) | 11.6 | 29.3 |
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
| dropout=0.5 † | 2 | 94.54 ± 0.04 | -0.25 ± 0.03 | 94.35 ± 0.08 | -0.29 ± 0.04 | 90.22 ± 0.05 | 800,035 (242,688) | 11.4 | 58.0 |

## case_features

| setting | seeds | val F1 | Δ val | test F1 | Δ test | test macro F1 | params (recurrent) | s/epoch | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 94.07 ± 1.25 | — | 94.01 ± 1.10 | — | 90.73 ± 0.28 | 800,035 (242,688) | 12.0 | 58.3 |
