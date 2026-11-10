# Panther — Implementation Plan

Implements the research specification *LLM-Adapted Multi-Bit Post-Training Quantization with
Hardware-in-the-Loop Genetic Search for Energy-Efficient Inference* (v2.0), with the deviations listed
in §15.

**Status:** plan, nothing built yet · **Revision:** 6 · **Date:** 2026-11-10

**Pi time budget: 10 hours total** (§9). All search and quality work runs on the server; the Pi only
confirms RAM and speed for a small, fixed set of configurations.

Units: memory in **GiB** (2³⁰ bytes) throughout; "token" = model token.

---

## 1. Goal

> Given a **RAM budget** on an 8 GB Raspberry Pi 5, find the **best-performing model**, and for
> that model the **best-performing quantization configuration**, that fits within the budget.

```
maximize   Quality(m, b)                         (primary endpoint, §2.1)
subject to PeakRAM(m, b, ctx)       ≤ R_max      (measured on the Pi, at the stated context length)
           DecodeTokensPerSec(m, b) ≥ S_min      (optional speed floor; off unless set)
```

- **Inner problem:** for each model *m*, the best configuration **b** within budget.
- **Outer problem:** the model whose best configuration wins on the primary endpoint (§2.1 decision rule).
- **Budget map:** the same answer over a grid of R_max values, so nearby budgets are covered.
- **RAM as the resource budget:** energy is not measured. RAM usage is the budgeted resource, on the
  basis that decode reads every weight once per token, so a smaller model in RAM generally means less
  memory traffic. This is an assumption, not a measured result (see §17).

---

## 2. Metrics

Every metric below is computed for every finalist and every baseline, with the same procedure.

### 2.1 Quality

| Metric | Definition | Data | Used for |
|---|---|---|---|
| **Task accuracy (primary endpoint)** | Unweighted mean accuracy over the suite below; each task also reported separately with 95% CI | Test split (§3) | Final ranking, cross-model choice |
| Per-task accuracy | HellaSwag (acc_norm), WinoGrande, ARC-Challenge (acc_norm), MMLU (5-shot), GSM8K (exact match, 5-shot, greedy) | Full official sets: HellaSwag val 10,042; WinoGrande dev 1,267; ARC-C test 1,172; MMLU test 14,042; GSM8K test 1,319 | Reported; per-task regressions flagged |
| **Perplexity (key secondary)** | exp(mean token NLL), context 2048, stride = context, shared family tokenizer | Test text split, ≥ 300k tokens | Tie-break; cross-model comparison |
| KL divergence | Mean and 99th-percentile per-token KL(FP16 ‖ quantized) over the full vocabulary | Sensitivity split (screening), selection split (shortlisting) | Search penalty and shortlisting |
| Top-1 agreement | % of positions where the quantized model's top token equals FP16's | Same as KL | Reported; sanity |
| Generation sanity | 50 fixed prompts, greedy, 128 tokens: exact-match length vs FP16; manual spot-check of 10 | Fixed prompt set | Catch degenerate outputs |

- Loglikelihood tasks run through `llama-perplexity` (`--hellaswag`, `--winogrande`, `--multiple-choice`)
  on the GGUF actually deployed; GSM8K through `llama-server` + lm-evaluation-harness
  (`local-completions`), pinned version. Few-shot exemplars fixed by seed.
- **Instruct** variants for all models.
- Server screening uses the llama.cpp **CUDA** build; **final test-set numbers use the CPU build**,
  with the Pi agreement check in §8.

**Decision rule (pre-registered).** Configuration A beats B if the paired-bootstrap 95% CI of
(mean task accuracy A − B) lies entirely above **+0.5 percentage points**. Otherwise, the tie is broken by
perplexity (A wins if PPL lower by ≥ 1% relative with CI excluding 0), then by lower peak RAM, then by
higher decode throughput. Holm correction for multiple comparisons within a table.

### 2.2 Speed

| Metric | Definition |
|---|---|
| Prefill throughput | Prompt tokens processed per second while the model reads the prompt, before generating anything (tokens/s) |
| **Decode throughput** | Generated tokens per second once generation has started (tokens/s) |
| Per-token latency | p50 and p95 of the time between consecutive generated tokens (ms) |
| End-to-end latency | Request start → last generated token, at the reference workload (s) |

### 2.3 Memory and size

| Metric | Definition |
|---|---|
| **Peak RAM (budgeted)** | Max over runs of process `VmHWM` + any page cache retained by the load mode; system `MemAvailable` minimum logged |
| Context assumed | RAM budget is evaluated at a stated context length (default 4096 tokens) |
| KV-cache size | Bytes at that context and KV precision (f16 default) |
| GGUF size | File bytes |
| Bits per weight | Total weight bits / total weights (overall and per gene) |

### 2.4 Validity

Repetitions kept / discarded and why; coefficient of variation of speed metrics; throttle events;
temperature and frequency traces; type-verification result; fallback count (must be 0); OOM events.

---

## 3. Data splits

Four **disjoint** splits, with stored sample IDs and hashes:

| Split | Content | Used by |
|---|---|---|
| Calibration | General web text (e.g. FineWeb-Edu shard A) | `llama-imatrix`, Stage-2 quantizer |
| Sensitivity | Web text shard B | Per-gene KL penalties |
| Selection | Web text shard C + task dev/val items not in the test sets | Refinement, shortlist, finalist choice |
| **Test (touched once)** | WikiText-2 test + web text shard D + full task sets (§2.1) | Final numbers for finalists and baselines only |

No configuration is chosen using test-split numbers.

---

## 4. Environment and runtime

| Machine | Role |
|---|---|
| Mac (dev) | Write code; reaches the server |
| Server (x86, ~80 GB RAM, 96 GB VRAM) | Quantization, GGUF building, quality evaluation, sensitivity, solver, orchestration |
| Raspberry Pi 5, 8 GB, NVMe (M.2 HAT), Ethernet, active cooler | Target device: RAM and speed measurement. Access arrives later |

- **Runtime:** llama.cpp CPU, pinned commit. Load mode set explicitly (`--load-mode`) and recorded.
  Context set explicitly per workload (`n_ctx = 0` would allocate the 32k trained context).
- **Threads:** swept on the Pi for 14B and 7B (§9 row 3; decode is memory-bound; published Pi 5 results show 2 threads can beat
  4). Decode (`-t`) and prefill (`-tb`) threads chosen per model by measured throughput, then fixed.
- **Kernel repacking on Cortex-A76 (dotprod, no i8mm), current master:** repacked = Q4_0, Q4_K, Q5_K,
  Q6_K, Q8_0, IQ4_NL; not repacked = Q2_K, Q3_K, IQ2_*, IQ3_*, IQ4_XS. Re-checked for the pinned commit;
  affects speed per format.

---

## 5. Models

Qwen2.5 Instruct family (shared tokenizer). Qwen3 (14B / 8B / 4B) is an alternative; one family
throughout.

| Model | Layers | Hidden / MLP | Embedding & head | Role |
|---|---|---|---|---|
| Qwen2.5-1.5B | 28 | 1536 / 8960 | **tied** | Pilot |
| Qwen2.5-3B | 36 | 2048 / 11008 | **tied** | Candidate (fits at 8-bit) |
| Qwen2.5-7B | 28 | 3584 / 18944 | untied (1.09B params, 14%) | Candidate (fits at 5–6 bits) |
| **Qwen2.5-14B** | 48 | 5120 / 13824 | untied (0.78B each) | Main candidate (Q4_K_M ≈ 9 GB does not fit) |

All matrix widths divide by 256 (verified from `config.json`): every K-quant and IQ type applies without
fallback.

**RAM fit (Phase A deliverable):** per (model, uniform format, context): weights + KV cache + compute
buffers. Indicative 14B file sizes: Q2_K 5.77 GB, IQ2_M 5.36, IQ3_XS 6.38, Q3_K_S 6.66, IQ3_M 6.92,
Q3_K_M 7.34. 14B KV cache at f16: 192 KiB/token (0.75 GiB at 4096 context). At R_max ≈ 7 GiB, 14B lands
around 3.2–3.4 bits per weight on average; at 6.5 GiB, nearer 3.0.

---

## 6. Search space

### Genes

| Gene | Untied models (7B, 14B) | Tied models (1.5B, 3B) |
|---|---|---|
| Attention part per layer (Q/K/V/O) | 1 per layer | same |
| MLP part per layer (gate/up/down) | 1 per layer | same |
| Token embedding | 1 | — |
| Output head | 1 | **one combined gene** |

Optional later split: `ffn_down` as its own gene.

### Format menu

`Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL`
(all quantized with the imatrix; IQ2_XS *requires* imatrix data, which the token embedding never has, so
IQ2_XS is excluded from the embedding gene only). Pruned per model: a format is kept only if no other format is at least as
good on bytes and quality (and speed, if S_min is set) at once.

### Quantization stages

- **Stage 1 (now):** `llama-imatrix` on the calibration split, then `llama-quantize` per format into a
  per-tensor store; mixed files assembled with **anchored, escaped** `--tensor-type` patterns, plus a
  read-back type check.
- **Stage 2 (parallel track):** PTMQ-inspired quantizer emitting the same formats.

---

## 7. Cost models ("price tags")

| Price tag | Method |
|---|---|
| **RAM** | Exact tensor bytes + measured per-model overhead (KV cache as a function of context, compute buffers, runtime). Overhead measured on the Pi |
| **Quality penalty** | KL increase when one gene is lowered, from a base config near the budget, on the sensitivity split |
| Speed (only if S_min set) | Decode time per token = overhead + Σ_format (time per byte of that format × bytes in that format), fitted from the 7B format runs (§9), checked against the finalists' measured speed |

RAM is predicted exactly on the server (tensor bytes + per-model overhead measured once on the Pi). The
prediction is checked against every Pi run; if any prediction is off by more than 2%, the overhead model is
re-measured before results are reported.

---

## 8. Search and selection

Per model, per budget point:

1. **Solver.** Integer program (multiple-choice knapsack; OR-Tools CP-SAT): minimize Σ KL penalty
   s.t. RAM ≤ R_max (and predicted decode speed ≥ S_min if set). Optimal for the price tags only.
2. **Refine.** Re-measure penalties around the solution and re-solve; stop when unchanged or
   selection-split KL improves < 1% relative, max 3 iterations. Budget-map points reuse the nearest base.
3. **Shortlist.** Top 20 distinct solutions via no-good cuts, evaluated on the selection split.
4. **GA polish (ablation).** NSGA-II seeded from solver solutions, same number of server quality
   evaluations as the solver pipeline.
5. **Final quality.** Top 2 per model evaluated once on the test split (server CPU build); best by the
   §2.1 decision rule is the model's result.
6. **Pi confirmation.** Each model's winner, plus the shortlisted runner-up for 14B and 7B, runs on the Pi.
   *Within budget* = max peak RAM over runs ≤ R_max, all validity rules (§9) pass, and (if set) mean decode
   throughput ≥ S_min. If a winner fails, the runner-up takes its place. Best confirmed model overall is
   the recommendation.

**Pi agreement check** (small): server CPU-build vs Pi KL/perplexity on a 4k-token text for 2 configs each
of 1.5B and 3B and 1 config of 7B; must agree within 0.5% relative perplexity. 14B is not checked on the
Pi (a 4k-token pass would take over 30 minutes); it relies on the server CPU build, noted as a limitation.

Budget-map points not confirmed on the Pi are labelled **predicted**.

---

## 9. Pi measurement

**Workloads**

| ID | Prompt / generated | Use |
|---|---|---|
| W_short | 128 / 32 | Thread sweep, format speed runs |
| **W_ref** | 256 / 64 | Confirmation runs, reported speed metrics |
| W_gen | 128/32 and 1024/128 | Generalization, recommended configuration only |

Batch 1, greedy, context set explicitly (default 4096; the KV cache is allocated for the full context
even when the prompt is short, so peak RAM is measured correctly by short runs). 1 warm-up + 3 reps
(W_short: 1 + 2). Peak RAM is nearly deterministic, so few reps suffice; speed is reported as mean with
range, not with confidence intervals.

**Validity rules (a run is discarded if any fails):** no throttling, temperature within a band at
start; no OOM and system `MemAvailable` above a floor; tensor types match the gene map; peak RAM ≤ R_max.

**Tooling:** small C++ timing driver on `libllama` with per-token timestamps; RAM from `/proc`; files
built on the server and copied to the Pi NVMe over Ethernet.

### Pi time budget: 10 hours

Based on published Pi 5 rates (7B Q4_K_M ≈ 4.8 t/s prefill, 2.2 t/s decode; 14B ≈ 2 t/s prefill,
1.6 t/s decode). One W_ref run is ≈ 3.3 min for 14B and ≈ 1.6 min for 7B, including load.

| # | Pi task | Configs | Time |
|---|---|---|---|
| 1 | Setup check, llama.cpp build, smoke test | — | 1.0 h |
| 2 | RAM overhead per model (one short run each) | 4 | 0.3 h |
| 3 | Thread sweep, 14B and 7B (2/3/4 threads, W_short) | 6 | 0.5 h |
| 4 | **Confirmation, W_ref:** winners of all 4 models, runners-up of 14B and 7B, best same-size llama.cpp baseline (14B), off-the-shelf 7B Q4_K_M | 8 | 1.2 h |
| 5 | Agreement check (§8) | 5 | 0.7 h |
| 6 | Format speed runs, 7B only, pruned menu (≤ 6 formats, W_short) | ≤ 6 | 0.4 h |
| 7 | Generalization, recommended config only (W_gen) | 1 | 0.7 h |
| 8 | File copies, cool-downs, loads between runs | — | 1.0 h |
| 9 | Contingency: discarded runs, re-measurement | — | 2.5 h |
| | **Total** | | **≈ 8.3 h (1.7 h spare)** |

Runs execute unattended as one queue in this priority order: 1 → 2 → 3 → 4 → 5 → 6 → 7. If time runs
out, the lowest-priority tasks are dropped and reported as not done.

Not on the Pi (to fit the budget): baselines other than the two in row 4 (their RAM is predicted, and
the prediction is validated by the row-4 measurements); 14B quality agreement; per-format speed for
models other than 7B; the full generalization grid.

---

## 10. Least privilege

**Pi**
- Dedicated `bench` user; no sudo; the harness never changes system settings.
- One-time admin setup (`pi/SETUP.md`): fixed CPU governor; Wi-Fi/BT off; Ethernet; swap off
  (dphys-swapfile) and zram off (rpi-swap / systemd-zram); unneeded services off; active cooler with a
  fixed fan policy; NVMe mounted with a `bench`-owned directory.
- llama.cpp built in `~bench`. All telemetry (`/proc`, `/sys/class/thermal`, cpufreq,
  `vcgencmd get_throttled`) read without root.

**Server ↔ Pi:** dedicated SSH key, `bench` only, `command=`-restricted, no agent forwarding.

**Server:** ordinary user, `uv` virtualenv, no `docker` group, no root.

---

## 11. Budgets

- **R_max:** from the deployment target (team value), in GiB at a stated context length; committed in
  `configs/budgets.yaml` **before Phase C**.
- **S_min:** optional, committed at the same time if used.
- Budget-map grid (pre-registered rule): R_max from 3 to 7.5 GiB in 0.5 GiB steps.

---

## 12. Baselines and ablations

All baselines get the **same treatment** on the server: same imatrix data where re-quantizable, same
test-split evaluation, same budget rule (predicted RAM ≤ R_max). Only the two strongest baselines are
also confirmed on the Pi (§9, row 4).

| ID | Baseline |
|---|---|
| B0 | FP16 reference (server only where it does not fit the Pi) |
| B1 | llama.cpp uniform/standard mixes at the same size: Q2_K, Q3_K_S, Q3_K_M, IQ3_XXS, IQ3_XS (our imatrix) |
| B2 | Third-party dynamic quants (e.g. Unsloth), as published and, where possible, re-quantized with our imatrix |
| B3 | Smaller models at higher precision within the same budget, incl. off-the-shelf 7B Q4_K_M |
| B4 | Greedy allocation by KL penalty per byte (HAWQ-style), no solver |
| B5 | Random search with the same number of server quality evaluations |
| **B6** | **Proposed:** solver + RAM price tags + sensitivity refinement + Pi confirmation |

**Ablations (all server-only):** solver vs GA vs solver+GA (equal evaluations); sensitivity base FP16 vs
near-budget; KL vs ΔPPL penalties; with/without IQ formats; attention/MLP genes vs per-projection;
Stage 1 vs Stage 2 on identical layouts. Pi-side ablations (repacking, thread counts beyond row 3) are out
of scope under the 10-hour budget.

---

## 13. Phases

### Phase A — Foundations (server, now)
- Repo, configs, schemas, manifests; pinned model revisions and llama.cpp commit.
- Four data splits; task sets prepared for `llama-perplexity` and lm-eval.
- llama.cpp CUDA and CPU builds; FP16 reference logits per model; imatrix per model; per-format tensor
  store; assembler with anchored patterns and type read-back.
- Metrics pipeline (§2) end to end on the 1.5B pilot.
- RAM fit table (§5).
- **Gate:** a hand-built mixed GGUF loads with exactly the intended types; all §2 quality metrics run.

### Phase B — Early model comparison (server)
- B1/B3 quality at the RAM budget on the selection split.
- Does 14B at ~3.0–3.4 bpw beat 7B at 5–6 bits? Decides effort per model.

### Phase C — Sensitivity and solver (server)
- Budgets committed (§11) before this phase.
- KL penalties per gene/format from near-budget bases; solver + refinement + shortlist; budget map.

### Phase D — Final quality and baselines (server)
- §8 step 5; all §12 baselines and ablations on the test split.

### Phase E — Pi session (one ~10-hour unattended queue)
- Rows 1–9 of the §9 budget, in priority order, after the admin setup is done.
- Winners confirmed (§8 step 6); recommendation chosen; RAM predictions validated.

### Phase F — Stage 2 quantizer (parallel from Phase A, server)
- PTMQ-inspired quantizer in the chosen formats; retained/modified/omitted report vs PTMQ; re-solve;
  must beat Stage 1 under the §2.1 decision rule. Uses the same layouts, so Pi results carry over; if it
  changes a winner, that config needs one extra Pi confirmation run (~15 min for 14B).

---

## 14. Success criteria

At each pre-registered R_max (and S_min if set):

1. The recommended configuration is **within budget** by the §8 step 6 rule, measured on the Pi.
2. It beats every within-budget baseline under the §2.1 decision rule (Holm-corrected).
3. Zero type fallbacks; no unexplained discarded-run patterns.

Negative results are reported (e.g. a smaller model winning at a given budget).

---

## 15. Deviations from the specification (v2.0)

| Spec | This plan | Reason |
|---|---|---|
| Minimize **measured energy** s.t. quality, memory, latency | Maximize quality s.t. a RAM budget; speed floor optional; **energy not measured** | Team decision: RAM usage is the budgeted resource |
| GA is the main search | Exact solver on price tags; GA as equal-budget ablation | Exact optimum for the price tags in under a second |
| One model | Search across models and precisions | "Best model within budget" includes the model choice |
| Bits {2, 4, 8}, PTMQ quantizer first | llama.cpp formats; llama.cpp quantization first, PTMQ-inspired as Stage 2 | Unblocks hardware work |
| Energy instrumentation (RAPL / NVML / meter) | None | Energy dropped |

The specification's title and central question (§0.1, §7.2, §19) are energy-based; they need updating to
match this plan.

---

## 16. Repository layout

```
panther/
  pyproject.toml          # uv; torch, transformers, gguf, ortools, pymoo, pydantic, pyarrow, lm-eval
  configs/                # models, formats, budgets, workloads, eval, splits
  src/panther/
    models/  data/  gguf/  quality/  sensitivity/  costs/  search/  orchestrator/  quant/  tracking/  stats/
  pi/
    driver/  agent/  SETUP.md
  tests/
  results/                # raw traces + parquet (gitignored)
```

---

## 17. Risks

| Risk | Mitigation |
|---|---|
| RAM is not an exact stand-in for energy: formats with the same size differ in compute per byte (repacked vs not) | Stated as an assumption; decode throughput per format reported as a partial indicator |
| Contribution overlaps prior work (HAWQ-V3 integer-program allocation; llama.cpp / Unsloth mixes) | Strong same-size baselines; cross-model selection; Pi-confirmed results |
| KL penalties not additive | Near-budget bases, refinement, shortlist on selection split |
| Winner's curse | Four disjoint splits; test split touched once |
| 14B too slow to be useful | Optional S_min; smaller models compete |
| 10-hour Pi budget: few reps, short workloads, most baselines not run on the Pi | Server-side RAM prediction validated on every Pi run; prioritized queue; skipped items reported |
| Context/KV silently large | Explicit `n_ctx`; RAM budget states context |
| llama.cpp API/kernel drift | Pinned commit; repack set and load-mode recorded |
| Server vs Pi numerics | CPU build for test numbers; Pi agreement threshold |
| Stage 2 fails to beat imatrix | Stage 1 results stand; reported honestly |
| Late Pi access | Only Phase E needs the Pi; everything else is server-only |
