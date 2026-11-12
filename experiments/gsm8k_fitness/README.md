# experiments/gsm8k_fitness

The genetic search with GSM8K (math, answers written out) in its fitness, tried on Qwen2.5 7B at 5 GiB only, where the
main search lost on the held-out test through GSM8K alone (62.0 vs the preset's 68.7; equal or better on MMLU,
HellaSwag, ARC and WinoGrande).

`run.py` imports the main pipeline (`scripts/server/pipeline.py`, unchanged) and replaces only:

- the fitness: the mean of five tasks (the four multiple-choice selection subsets as before, plus the first 100 GSM8K
  selection questions) minus 10 x KL;
- the top-3 scoring and the guard: all selection questions of the five tasks (4,500 multiple choice + 500 GSM8K);
- the names: files `-gsm8k`, progress lines `7B-gsm8k`, so the main results are not touched.

Steps (`ga`, `gascore`, `compare`) run as a chain on the server: `data/rewrite/chains/x5-7b.steps`. If it fixes the
loss, the change moves into the main pipeline after the current runs.
