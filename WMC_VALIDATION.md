# WMC validation status

- Branch: wmc-lstm, based on main commit 921acef; PR #2 targets main.
- Implementation follows guidelines equations 7–12, including updated c_t for the output gate. No built-in LSTM is used inside the cell.
- 19 successful local checks in total: independent equations and gradients, output-gate memory regression, zero-memory reduction to Vanilla, initialization, parameter counts, padding in both directions, tagger backward pass, shared data/metrics checks, complete-seed summary and protocol validation.
- Colab: 18 passed, 1 optional seqeval check skipped; that optional check passed separately locally.
- Notebook syntax and embedded sources verified against the current files.
- Three official fine-grained baseline runs completed on Tesla T4, torch 2.11.0+cu130, using the unchanged shared training and data pipeline.
- Seeds 1/2/3, Adam lr=0.001, batch 32, hidden 128, bidirectional, dropout 0.3, maximum 60 epochs, patience 10. Full configs are committed with the results.
- Executed epochs: 60 / 60 / 55. Best validation epochs: 51 / 60 / 45. Seed 3 stopped after exactly 10 epochs without validation improvement.
- Every history length, best validation epoch/F1, harmonic test F1, summary row, CUDA device, and parameter count was checked against the saved artifacts.
- Confirmed parameters: total 898339, recurrent 340992, embeddings 552464, classifier 4883.
- Official test F1: 94.6630924820%, 94.5556299045%, 92.0302707816%; mean ± sample std = 93.75 ± 1.49%.
- Original 56 Vanilla summary rows preserved; exactly 3 WMC rows appended (59 total).
- An earlier two-batch CPU smoke check was stored outside official results and is excluded from every reported score.
- Collective comparison, final report, slides, merge into main and submission remain group work.
