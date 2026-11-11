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
5. **Penalty model:** `penalty(matrix, format) ≈ sensitivity[block] × error(matrix, format)`. The error is computed
   offline from the weights; the sensitivity is measured **per block** (attention / MLP per layer, embedding,
   head), one KL run each, one sweep per model shared by its budgets (D45). The solver still chooses per matrix.
6. **Format menu:** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0.
   Q3_K, Q4_0 and IQ4_NL are dropped (beaten on both size and quality in the PoC).
7. **Never measure an option that can't fit:** a (group, format) option is dropped if, even with every other
   group at its smallest format, it exceeds the budget. A model is dropped if all-smallest doesn't fit.
8. **8-bit KV cache** by default, later a choice inside the search. **Real overhead measured on the Pi.**
9. **No re-measure loop by default** (D45): measure the top 3; re-measure blocks only if the measured KL is far
   from the prediction.
10. **Decode speed per format** measured on the Pi becomes a constraint.
11. **Every result is compared at the same RAM** against llama.cpp's stock presets (same importance matrix) and
    against "uniform with head and V raised", padded to the budget.

Settings live in `configs/search.yaml`; `scripts/budget_table.py` prints the table below from them.

**KL budget (D45):** ~390 runs in total (Phase 0 ~7, sensitivity 343, scoring 36), about 1.5 h on the GPU.

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
- [ ] Replace `pi/panther_pi.py` and `pi/fetch_bundle.sh` with the job queue in Phase 6.
- [x] ~~Server-only infrastructure~~: **kept** (D45): we stay on anton, whose rules require it.
- [ ] Drop unused dependencies: `pydantic` (never imported); `lm-eval`, `torch` (only the deleted checks).
- [ ] `docs/DECISIONS.md` and `docs/PIPELINE.md` are **kept** (D45, document before doing); delete stale
      markdown (e.g. `PLAN.md`) as the rewrite replaces it.

## Phase 2: core (target about 1,000 lines in total, Pi queue included)

```
src/panther/
  inventory.py   per-matrix groups, allowed formats, bytes per format                ~100
  error.py       dequantize + importance-weighted error per (tensor, format)         ~80
  assemble.py    tensor-copy GGUF builder, type read-back, recipe export             ~120
  evaluate.py    llama-perplexity: KL, perplexity, benchmarks; the three checks      ~150
  search.py      penalty model, pruning, CP-SAT with RAM (+ speed), top-k, re-solve  ~150
  run.py         one budget × model, end to end; results CSV                         ~150
scripts/budget_table.py                                                              (done)
pi/job.py, pi/sync.sh, pi/pi_worker.py, pi_submit.py, pi_collect.py   (Phase 6)  ~300
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

- [ ] **3.1** Per block, move only that block from the base to a probe format and measure the KL increase (one
      run per block: 1.5B 57, 7B 58, 14B 98, 32B 130). `s_b = ΔKL_b / Σ_{m∈b}(e(m, probe) − e(m, base))` (D45).
- [ ] **3.2** One base per model (shared by its budgets): the uniform format closest to the lower budget's
      average bits per weight.

## Phase 4: search per contest

- [ ] For each budget in `configs/search.yaml` and each searched model: solve, take the top 3, score them on the
      CPU build, keep the best (D45).
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

## Phase 6: Pi (job queue, relayed by the Mac)

Neither machine can reach the other, so the Mac relays. The server drops self-contained jobs into an outbox, the
Mac copies them to the Pi and the results back, and a worker on the Pi runs them one at a time. Neither side
ever runs commands on the other; only folders are copied.

```
server                     Mac                          Pi (user "bench", no sudo)
outbox/files/<sha>.gguf ─► pi/sync.sh ─► inbox/files/<sha>.gguf
outbox/jobs/<id>.json   ─►            ─► inbox/jobs/<id>.json ──► pi_worker.py (one job at a time)
inbox-results/<id>.json ◄─            ◄─ results/<id>.json    ◄──┘
```

- [ ] **6.1 Job format** (`pi/job.py`, shared by both sides): JSON with id, GGUF sha256 and size, budget,
      context, KV type, predicted RAM and tensor-type counts, test names (`ram`, `speed`, `agreement`), thread
      counts. **Data only: no command strings.** The worker builds every llama.cpp command itself from a fixed
      list of tests, so writing to the queue never means running arbitrary code on the Pi. GGUFs are stored by
      sha256, so a file used by several jobs is copied once.
- [ ] **6.2 Server: `pi_submit.py --budget B --model M`.** Takes the winners and references for that budget from
      the results CSV, copies each new GGUF into `outbox/files/`, writes `jobs/<id>.json.part` then renames it
      (a half-written job is never visible). Called by `run.py` at the end of each budget.
- [ ] **6.3 Mac: `pi/sync.sh SERVER PI`** (replaces `pi/fetch_bundle.sh`). Server outbox → Mac cache → Pi inbox,
      then Pi results → Mac → server `inbox-results/`. rsync with `--partial` and checksums; files already on
      the Pi are skipped. Copies jobs only after their GGUFs have arrived. One SSH connection per host per sync,
      no retry loops (the server bans addresses after repeated failed logins, D30).
- [ ] **6.4 Pi: `pi_worker.py`** (standard library only, ~150 lines; replaces `pi/panther_pi.py`):
      - claim the oldest job by moving it to `running/` (survives a crash or reboot), check the GGUF's sha256;
      - wait until the CPU is below 70 °C and the system is idle;
      - `ram`: load with `-c 4096` and the job's KV type, short prompt, record peak RSS (`VmHWM`) and minimum
        `MemAvailable`, compare tensor-type counts from the loader log with the prediction;
      - `speed`: `llama-bench -p 256 -n 64 -r 3` at each thread count;
      - `agreement`: perplexity on the shared 4k-token text;
      - invalid if throttled, out of memory, or over budget; write `results/<id>.json` (`.part` then rename),
        move the job to `done/`, delete GGUFs no queued job needs.
      Runs as a user-level systemd service (`systemctl --user`, linger on) so it restarts after a reboot.
      **Never two jobs at once**: they would spoil each other's RAM and speed numbers.
- [ ] **6.5 Server: `pi_collect.py`.** Reads `inbox-results/*.json` into the results CSV; flags any run whose
      peak RAM is more than 2% off the prediction (then re-measure the overhead, step 0.2).
- [ ] **6.6 Pi setup, once:** headless (7 GiB leaves about 0.45 GiB for the OS), swap and zram off, `bench` user,
      llama.cpp built natively at the pinned commit, the worker service enabled.
- [ ] **6.7 What gets submitted per budget:** the winner and the best same-RAM reference for every searched model;
      for one file per model, perplexity must match the server's CPU build within 0.5%.

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
