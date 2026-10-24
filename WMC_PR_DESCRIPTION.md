Implement WMC botanical NER cell under the common training protocol

Adds the WMC cell from guidelines equations 7–12 to the shared NER pipeline. The input/forget gates use the previous memory; the output gate uses the updated memory. Memory connections are full bias-free linear maps followed by tanh.

Includes independent equation/gradient checks, a regression test for the output gate's memory dependency, vanilla reduction, padding and model integration checks, plus a self-contained WMC-only Colab notebook and three-seed summary script. Shared data, baseline settings and Vanilla results are unchanged.

Training results: pending the three full GPU runs. This PR delivers implementation and execution tooling; collective architecture comparison, final report and slides remain group work.
