WMC LSTM: cell, three-seed baseline results and Colab workflow

Implements Working Memory Connections from guidelines equations 7–12 in the shared botanical French NER pipeline. The input and forget gates use c_(t-1); the output gate uses the newly computed c_t. The three memory maps are full bias-free nn.Linear projections followed by tanh.

The three common-baseline runs on Tesla T4 achieved test entity F1 **93.75 ± 1.49%** (seeds 1/2/3: 94.66%, 94.56%, 92.03%). The model has **898,339 trainable parameters**, including **340,992 recurrent**. Data splits, preprocessing, Adam lr=0.001, bidirectionality, hidden size, dropout, batch size and best-validation evaluation protocol are unchanged. Seeds 1/2 ran 60 epochs; seed 3 stopped at 55 with patience 10.

Includes the complete configs/histories/metrics, three WMC summary rows, a validated summary script, dedicated self-contained Colab notebook, and duo handoff notes. The 56 existing Vanilla rows and their analysis are preserved.

Validation: 19 successful local checks in total; 18 passed plus one optional seqeval check skipped on Colab (that check passed locally). Verified notebook/source consistency and cross-checked each saved run's best epoch, history, scores and parameter counts.

This completes the Said & MS WMC contribution. Collective comparison with CIFG/Vanilla, final report and slides remain group work; this PR does not merge into main or submit the project.
