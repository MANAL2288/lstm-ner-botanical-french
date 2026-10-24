# WMC baseline — Said & MS

Common protocol; seeds 1, 2, 3. Test metrics use the best validation epoch.

| Seed | Test P (%) | Test R (%) | Test F1 (%) | Macro F1 (%) | Best epoch | s/epoch |
|---|---|---|---|---|---|---|
| 1 | 93.78 | 95.56 | 94.66 | 90.55 | 51 | 17.12 |
| 2 | 93.54 | 95.60 | 94.56 | 90.44 | 60 | 18.08 |
| 3 | 88.60 | 95.73 | 92.03 | 88.38 | 45 | 17.67 |

Mean ± sample standard deviation across three seeds:
- Test precision: 91.97 ± 2.92%
- Test recall: 95.63 ± 0.09%
- Test f1: 93.75 ± 1.49%
- Test macro_f1: 89.79 ± 1.22%

Trainable parameters: {"total": 898339, "recurrent": 340992, "embeddings": 552464, "classifier": 4883}

Collective comparison and interpretation remain group work.