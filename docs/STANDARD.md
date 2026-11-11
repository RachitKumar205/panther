# The standard (D49, 2026-11-11)

One procedure for every model and RAM budget. Settings live in `configs/search.yaml`; code in
`scripts/server/rewrite.py`. Anything not listed here is not done.

## Rules

- **Solution space.** For a budget B, a genotype (one format per weight matrix) is a solution only if its predicted
  RAM is within **B −1% … B +0.5%** (7 GiB: 6.93–7.035 GiB). Nothing outside that window is ever a candidate.
  RAM = weights + KV cache (q8_0, 4,096 tokens) + overhead (0.40 GiB until the Pi measures it).
- **Quality tests run on the GPU** (CUDA build, all layers offloaded): KL, perplexity, benchmarks, GSM8K.
  **The CPU only prepares** (format files, offline error, file assembly). No CPU quality tests: the Pi measures
  RAM and speed only.
- **Each solution is assembled once**; all its tests run on that one file; then it is deleted.
- **Budgets one at a time in priority order** (`active_budgets`), each model in its own lane.

## Steps

| # | Command | Per | What | Output |
|---|---|---|---|---|
| 1 | `store`, `error`, `refs` | model | Format files for the model's menu, offline error per matrix × format, the BF16 model's predictions on the sensitivity (24k tokens) and selection texts | building blocks |
| 2 | `sens` | model | Part costs: KL of each block moved from the base to a probe format, then 2 rounds measuring the formats the solver uses (±1). Instruments, never candidates | `sens.json` |
| 3 | `list B` | budget | Solver: the top 6 genotypes per model inside the window, each differing from the others in ≥5% of matrices, plus one padded-uniform baseline per model | `list.json` = the whole solution space |
| 4 | `test B` | budget | Every listed genotype (+ the cross-model reference, uniform Q6_K) assembled once; KL (selection), MMLU 1000, HellaSwag 2000, ARC-C 1172, WinoGrande 1267, GSM8K 300, perplexity (test_web); 8 at once | results records |
| 5 | `pick B` | budget | Per model: best mean accuracy (ties: lower KL). Winner vs its baseline, models vs each other and the reference: paired bootstrap | `pick.json` |
| 6 | `bundle B`, then the Pi | budget | Winners as GGUF + manifest; on the Pi: load once, peak RAM vs B, one speed run (256/64), 4 threads | Pi report |

Before the first budget's Pi run, the Pi measures its real runtime overhead once (peak RSS − weights − KV), and that
value replaces the 0.40 GiB placeholder.
