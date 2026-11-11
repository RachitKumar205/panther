# Panther rewrite: plan and TODO

The first version (everything currently in this repo) was a vibecoded proof that the pipeline is possible. This
file is the plan for rewriting it by hand: smaller, per-matrix, and across several RAM budgets.

## What the proof of concept showed

- Works: mixed-precision GGUFs built by copying tensors from single-format files load in stock llama.cpp with
  exactly the intended types. The whole chain (penalties → solver → shortlist → test subset → Pi bundle) runs.
- 14B beats 7B at 7 GiB by about 4.8 accuracy points (CI [3.3, 6.2]).
- At 7B (about 6 bits per weight) the mix ties uniform Q6_K: there's nothing to gain at high bit widths.
- **Not shown:** that the mix beats a fair baseline. The 14B winner (6.999 GiB) was compared with uniform
  IQ3_XXS at 6.419 GiB, whose output head is at 3 bits. The +1.7 points may be the extra 0.58 GiB.
- **Not shown:** that anything fits on the Pi. The 0.40 GiB runtime overhead is a guess, and the 14B winner is
  0.001 GiB under budget.
- Penalties are noisy: in both lanes the winner was the solver's 4th pick of 5.

## Decisions

1. **Budgets 1, 3, 5, 7 GiB** total RAM on the Pi (weights + KV cache + overhead) at 4,096 context. Use the
   whole budget.
2. **At each budget, search the largest models that land at about 1.5–6 bits per weight.** Above that, a model
   is near lossless and is scored as the largest uniform format that fits. Below 1.5 it isn't run.
3. **Per-matrix groups:** q, k, v, o, gate, up, down per layer, plus embedding and output head (one combined
   group when tied).
4. **Output is a standard GGUF** (per-tensor types, loads in stock llama.cpp), shipped with its recipe: the
   `llama-quantize --tensor-type` overrides and the importance matrix, so stock tools rebuild the same file.
5. **Penalty model:** `penalty(group, format) ≈ sensitivity[group] × error(group, format)`. The error is computed
   offline from the weights; only the sensitivity needs model runs (1–2 per group).
6. **Format menu:** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0.
   Q3_K, Q4_0 and IQ4_NL are dropped (beaten on both size and quality in the PoC).
7. **Never measure an option that can't fit:** a (group, format) option is dropped if, even with every other
   group at its smallest format, it exceeds the budget. A model is dropped if all-smallest doesn't fit.
8. **8-bit KV cache** by default, later a choice inside the search. **Real overhead measured on the Pi.**
9. **Re-measure around the solution** and solve again, 2–3 rounds.
10. **Decode speed per format** measured on the Pi becomes a constraint.
11. **Every result is compared at the same RAM** against llama.cpp's stock presets (same importance matrix) and
    against "uniform with head and V raised", padded to the budget.

Settings live in `configs/search.yaml`; `scripts/budget_table.py` prints the table below from them.

## Budgets and contests

Exact byte sizes, per-matrix groups, 8-bit KV cache, overhead 0.40 GiB (**unmeasured**: step 0.2 can change
this table, especially at 1 GiB).

| Budget | Searched (bits/weight available) | Reference, no search |
|---|---|---|
| 1 GiB | 1.5B (3.0) | none: 3B doesn't fit (needs 1.07 GiB at minimum); 0.5B can't use most low-bit formats (rows not a multiple of 256) |
| 3 GiB | 7B (2.8) | 3B uniform Q6_K (7.0) |
| 5 GiB | 14B (2.4), 7B (5.1) | — |
| 7 GiB | 14B (3.6), 32B (1.6) | 7B uniform Q6_K (7.3) |

72B doesn't fit any budget (and its MLP width, 29,568, isn't a multiple of 256). Strict pruning (decision 7)
only removes options for 32B at 7 GiB (1,810 of 5,396); everywhere else there is enough room.

Groups per model: 1.5B 197, 3B 253, 7B 198, 14B 338, 32B 450.

---

## Phase 0: before writing new code (uses the existing code and the PoC data on the server)

- [ ] **0.1 Fair baseline for the 14B PoC winner at 7 GiB.** Build with stock `llama-quantize` and the PoC's
      importance matrix every preset that fits 7.0 GiB (IQ3_XXS, IQ3_S, IQ3_M, Q3_K_S; check sizes first), plus
      uniform IQ3_XXS with head and every `attn_v` at Q6_K, padded with IQ3_S layers up to the budget. Score
      them like the PoC winner (KL on selection text, test-subset accuracy, perplexity).
      **If the mix doesn't beat the best of these, stop and rethink before building anything.**
- [ ] **0.2 Measure runtime overhead on the Pi.** For 1.5B, 7B, 14B: `-c 4096`, KV q8_0, weights fully loaded;
      overhead = peak RSS − weight bytes − KV bytes. Put the value in `configs/search.yaml`, rerun
      `scripts/budget_table.py`, update the table above.
- [ ] **0.3 Check the penalty model on the 14B penalty table** (`poc/Qwen2.5-14B-Instruct/penalties.json`, 98
      groups × 7 formats, all measured). For every tensor and format, compute the importance-weighted
      dequantization error from the store files. Fit one sensitivity per group. Report how well the model
      predicts the measured table (rank correlation, relative error), and solve the 7 GiB problem with modelled
      vs measured penalties: if the two chosen configurations' measured KL is within a few percent, decision 5
      holds. If not, fall back to measuring each option directly, but only the options kept by decision 7.

## Phase 1: cut (after Phase 0; git history keeps everything)

Keep only three checks, because each caught a real silent failure: tensor-type read-back after every build;
llama.cpp run output checked (it exits 0 on failure); benchmark item count checked (it silently drops items).

- [ ] Delete one-time validation scripts: `scripts/server/gate_a1*`, `gate_a2*`, `gate_a3*`, `gate_a4*`,
      `gate_a5a6.py`, `d36.py`, `d37.py`, `check_d35.*`, `check_d39.py`, `time_tasks_cpu.sh`.
- [ ] Delete the CPU-vs-GPU correction (D35–D37): final candidates are scored on the CPU build directly.
- [ ] Delete chunk-count study, noise-floor measurements, determinism repeats, size-prediction and byte-identity
      checks.
- [ ] Replace `src/panther/stats/decision.py` (pre-registered rule, Holm, per-chunk NLL recovery) with one
      bootstrap CI on the accuracy difference.
- [ ] Delete `src/panther/search/baselines.py` (greedy/random), smoke mode in `poc.py`, then `poc.py` itself once
      `run.py` replaces it.
- [ ] Delete the custom GSM8K runner and the 50-prompt generation check (read 5 outputs by hand instead).
- [ ] Data: two texts instead of four splits plus selection questions: one for importance matrix and
      sensitivity, one held out for final scoring (plus benchmark test sets).
- [ ] Replace per-file SHA-256 manifests and append-only JSON records with one results CSV per model.
- [ ] Replace `pi/panther_pi.py` with a short script around `llama-bench` and `/usr/bin/time -v`.
- [ ] Server-only infrastructure (`deploy/server/`: container, SELinux, host ledger, guard, `wait-job.sh`):
      keep only if we stay on that shared server; otherwise run llama.cpp directly.
- [ ] Drop unused dependencies: `pydantic` (never imported); `lm-eval`, `torch` (only the deleted checks).
- [ ] Replace `PLAN.md`, `docs/PIPELINE.md`, `docs/DECISIONS.md` with a short README of what we chose and why.

## Phase 2: core (target about 800 lines in total)

```
src/panther/
  inventory.py   per-matrix groups, allowed formats, bytes per format                ~100
  error.py       dequantize + importance-weighted error per (tensor, format)         ~80
  assemble.py    tensor-copy GGUF builder, type read-back, recipe export             ~120
  evaluate.py    llama-perplexity: KL, perplexity, benchmarks; the three checks      ~150
  search.py      penalty model, pruning, CP-SAT with RAM (+ speed), top-k, re-solve  ~150
  run.py         one budget × model, end to end; results CSV                         ~150
scripts/budget_table.py                                                              (done)
pi/bench.sh      llama-bench per file/format + peak RAM                              ~50
```

- [ ] **2.1 inventory.py:** per-matrix groups from `config.json`; allowed formats per group (no
      importance-matrix formats for the embedding). Verify at the pinned llama.cpp commit which formats need an
      importance matrix (`ggml_quantize_requires_imatrix`, and whether IQ1_M asserts on it) and fix
      `requires_imatrix` in `configs/search.yaml`.
- [ ] **2.2 Tensor store per model:** BF16 GGUF → `llama-imatrix --process-output` (so the output head gets
      data) → one `llama-quantize --pure` file per menu format → type read-back. Only the formats a model can use
      at its budgets.
- [ ] **2.3 error.py:** for each tensor and format, dequantize from the store file (gguf-py) and compute
      Σ importance × (w − ŵ)². Check that gguf-py dequantizes every format in the menu (IQ1/IQ2 included).
- [ ] **2.4 assemble.py:** port the existing assembler to per-matrix groups; write `<name>.recipe.txt` with the
      `--tensor-type` overrides next to each GGUF; test once that stock `llama-quantize` with the recipe gives the
      same tensors.
- [ ] **2.5 evaluate.py:** KL against a full-precision reference (BF16; for models over 96 GB VRAM use Q8_0),
      perplexity, and HellaSwag / ARC-C / MMLU through llama.cpp's built-in modes. Only the three checks.
- [ ] **2.6 search.py:** penalties from the model in decision 5; drop infeasible options (decision 7); CP-SAT
      multiple-choice knapsack under the RAM budget; top-k; re-solve loop (5.1).
- [ ] **2.7 run.py:** `run.py --budget 7 --model Qwen2.5-14B-Instruct` does store → sensitivity → search →
      score, resumable, one CSV row per scored file.

## Phase 3: sensitivity

- [ ] **3.1** Per group, raise only that group from a near-budget base and measure the KL increase (1–2 runs per
      group: 197–450 runs per model; roughly 1 h for 14B at the PoC's rate, 2–3 h for 32B). Fit
      `sensitivity[group]` against the offline error.
- [ ] **3.2** Use one base per (model, budget): the uniform format closest to the budget's average bits per weight.

## Phase 4: search per contest

- [ ] For each budget in `configs/search.yaml` and each searched model: solve, take the top 5, score them on the
      CPU build, keep the best.
- [ ] Score the references: largest uniform format that fits, llama.cpp presets that fit (same importance
      matrix), and "uniform with head and V raised" padded to the budget.

## Phase 5: refinements

- [ ] **5.1 Re-measure loop:** re-measure sensitivities around the chosen configuration, solve again; stop when
      the choice stops changing or after 3 rounds.
- [ ] **5.2** Check the extra-low formats (IQ1_S, IQ1_M) actually get chosen and help at 32B/7 GiB and 14B/5 GiB;
      drop them from the menu if never used.
- [ ] **5.3 KV cache type in the search:** f16 vs q8_0 (and q4_0) as one more group; measure the KL cost of each
      once per model.
- [ ] **5.4 Speed constraint:** decode tokens/s per format from `llama-bench` on the Pi (one uniform file per
      format for 7B) → seconds per byte per format → optional minimum-speed constraint in the solver
      (`Problem.s_min` already sketches this).

## Phase 6: Pi

- [ ] **6.1** Setup: headless (7 GiB leaves about 0.45 GiB for the OS), swap and zram off, llama.cpp built at the
      pinned commit.
- [ ] **6.2** Per budget, the winner and the best reference: peak RAM ≤ budget (`/usr/bin/time -v`), decode and
      prefill tokens/s (`llama-bench`), no throttling.
- [ ] **6.3** Perplexity on a short text matches the server's CPU build within 0.5% for one file per model.

## Phase 7: results

- [ ] Per budget: winner vs every same-RAM reference, and across models; accuracy difference with a bootstrap CI,
      perplexity, peak RAM, tokens/s.
- [ ] Publish each winning GGUF with its recipe and importance matrix.

## Later

- [ ] Better rounding inside the same formats (GPTQ-style, error-compensating) so files still load in stock
      llama.cpp. Biggest expected gain at 2–3 bits.
- [ ] MoE models (experts as groups, rarely used experts streamed from NVMe).

## Open questions

- Real overhead (0.2) decides whether 1 GiB has a contest at all.
- Is 4,096 context the right fixed length, or should context be a per-budget setting?
- Tied embedding/head (1.5B, 3B): does `--process-output` give it importance data, so it can use IQ1/IQ2?
