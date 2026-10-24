# WMC validation status

- Branch: wmc-lstm, based on main commit 921acef.
- Draft PR: https://github.com/MANAL2288/lstm-ner-botanical-french/pull/2
- Numerical/integration suite: 15 passed, 1 optional seqeval check initially skipped. That remaining check passed separately after dependency installation (16 successful checks in total).
- Summary handoff checks: 3 passed (missing seeds, debug/protocol drift, sample standard deviation); 19 successful checks in total.
- Notebook syntax and embedded sources: verified against current source files.
- Common official dataset downloaded and converted without changing the pipeline.
- Full-size model smoke run: two training batches, one validation/test evaluation, finite loss and successful backward pass. Stored outside official results. This is not a trained-model performance result.
- Confirmed parameter counts on real vocabulary: total 898339, recurrent 340992, embeddings 552464, classifier 4883.
- Three full WMC seed runs: launched on a Colab Tesla T4 (torch 2.11.0+cu130), Drive mounted. Colab subsequently lost its service connection; no completed seed result has been verified yet. Do not treat the smoke check as final performance.
- Collective comparison, report and slides: intentionally outside this duo delivery.
