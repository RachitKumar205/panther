# How Panther works

A step-by-step account of the method, what each stage produces, and where it stands. PLAN.md is the
contract (what we promised to measure and how decisions are made); this file explains the machinery.
Decisions and their reasons are in [DECISIONS.md](DECISIONS.md) (referenced as D-numbers).

Status per stage: ✅ done · 🟡 in progress · 📝 planned (documented before running) · ⬜ not started.

```
 HF weights ──► BF16 GGUF ──► importance matrix ──► 12 single-format files (tensor store)
                                   ▲                         │
                    calibration text (300k tok)              ▼
                                               assemble any per-gene mix in seconds
                                                             │
 sensitivity text ──► penalty table (KL per gene × format) ──► solver (RAM budget) ──► top 20
                                                                                        │
 selection text + selection questions ──────────────────────────► shortlist scoring ◄──┘
                                                                        │ top 2 per model
 test text + full benchmarks (touched once) ─────────────────────► final quality (server CPU)
                                                                        │ winner + runner-up
                                                                 Pi 5: RAM, speed, validity
```

---

## 1. Models and genes ✅

Four Qwen2.5 Instruct models (1.5B, 3B, 7B, 14B), one shared tokenizer, pinned revisions
(`configs/models.yaml`).

A **gene** is a group of weight matrices that must share one format (D6):

| Gene | Matrices | Count |
|---|---|---|
| `attn.<i>` | Q, K, V, O of layer i | one per layer |
| `mlp.<i>` | gate, up, down of layer i | one per layer |
| `embd_head` (1.5B, 3B: tied) | token embedding = output head | 1 |
| `embd`, `head` (7B, 14B: untied) | token embedding; output head | 1 each |

| Model | Layers | Genes |
|---|---|---|
| 1.5B | 28 | 57 |
| 3B | 36 | 73 |
| 7B | 28 | 58 |
| 14B | 48 | 98 |

Norm weights and biases stay F32 and are not searched; they are counted in RAM as fixed bytes.

**Formats** are llama.cpp block-quantized integer types, not floating-point precisions:

| Format | Bits/weight | Notes |
|---|---|---|
| IQ2_XS | 2.31 | needs imatrix data → not allowed for the embedding (D21) |
| Q2_K | 2.63 | |
| IQ3_XXS | 3.06 | |
| Q3_K | 3.44 | |
| IQ3_S | 3.44 | |
| IQ4_XS | 4.25 | |
| Q4_0, Q4_K, IQ4_NL | 4.50 | Q4_0, Q4_K, IQ4_NL have fast repacked kernels on the Pi's A76 |
| Q5_K | 5.50 | repacked on A76 |
| Q6_K | 6.56 | repacked on A76 |
| Q8_0 | 8.50 | repacked on A76 |

A configuration is a vector with one format per gene (14B: 98 entries).

## 2. Data splits ✅

Built by `scripts/server/build_splits.py` into `/s3/panther/work/data/splits`, with `manifest.json`
(source revisions, file sha256, counts, overlaps). Details: D16–D19.

| Split | Web text (FineWeb-Edu shard) | Questions | Used by |
|---|---|---|---|
| Calibration | A: 334 docs, 300,384 tokens | — | importance matrix (§3) |
| Sensitivity | B: 123 docs, 100,124 tokens | — | penalty table (§5) |
| Selection | C: 147 docs, 152,684 tokens | HellaSwag 1000, WinoGrande 1000, ARC-C 1000, MMLU 1502, GSM8K 500 | shortlist (§7) |
| Test (once) | D: 163 docs, 150,243 tokens + WikiText-2 test 298,938 = 449,181 | full official sets: 10,042 / 1,267 / 1,172 / 14,042 / 1,319 | final numbers (§8) |
| Few-shot | — | MMLU dev 285 (5 per subject), GSM8K 5 | prompts only |

Shards are disjoint by construction (seeded hash of the document id). Selection questions never overlap
test or few-shot items (checked). MMLU has 5 test items duplicating standard few-shot items upstream;
kept and recorded (D19).

## 3. Calibration: importance matrix ✅ (1.5B)

`llama-imatrix` runs the BF16 model over the calibration text in 512-token chunks (~585 chunks) on the
GPU and records, for every matrix input, how strongly each input channel is used. The quantizer then
spends its precision where those channels need it. 1.5B: 1 min 50 s, 4 MB file.

The token embedding is a lookup, not a matrix product, so it has no imatrix data (D21).

## 4. Tensor store ✅ (1.5B)

`scripts/server/build_tensor_store.sh` (GPU job, no network) converts the downloaded weights to one
BF16 GGUF, builds the imatrix, then produces one file per format with `--pure` and explicit embedding
and output types, so each file holds that single format everywhere. A read-back check verifies every
searchable tensor's type (catches silent fallbacks).

1.5B result, all verified:

| IQ2_XS | Q2_K | IQ3_XXS | Q3_K | IQ3_S | IQ4_XS | Q4_K | Q4_0 | IQ4_NL | Q5_K | Q6_K | Q8_0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.43 | 0.48 | 0.56 | 0.62 | 0.62 | 0.77 | 0.81 | 0.81 | 0.81 | 0.99 | 1.19 | 1.53 GiB |

Every build writes `store/<model>/manifest.json`: weights revision, llama.cpp commit, image id, input and
output hashes, verification and step timings (D22).

Any mixed configuration is then **assembled** by copying each tensor's bytes from the file for its
gene's format (`panther.gguftools.assemble`), in seconds, and verified the same way (D8).

Expected time per model (scales with size): 1.5B ~5 min, 3B ~15–20 min, 7B ~35–50 min, 14B ~1.5–2 h.
Disk: 14B ≈ 30 GB BF16 + ~100 GB of format files.

## 5. Penalty table (sensitivity) ⬜

For each gene and each format: start from a base configuration near the RAM budget, change only that
gene to that format, assemble, and measure the increase in **KL divergence** from the BF16 model's
output distribution over the sensitivity text. That number is the gene's price tag in "quality lost".

Penalties are measured on the CUDA build and corrected for the CPU build's activation rounding where a
change moves a gene between activation groups (D36); the CPU build is what the Pi runs.

Cost: genes × formats measurements (14B: 98 × 11 ≈ 1,000). Measured on 1.5B: ~0.4 s per 2048-token
chunk on the GPU; 14B ≈ 4 s per chunk, so ≈ 53 GPU-hours at all 48 sensitivity chunks. The A2
chunk-count study sets how many chunks are needed; pruning formats that are never best on both size and
quality cuts the number of evaluations.

## 6. Solver ⬜ (code ✅)

Choose one format per gene to minimise the sum of penalties subject to

- **RAM:** exact tensor bytes + KV cache at 4096 context + measured runtime overhead ≤ R_max;
- **speed (optional):** predicted decode time per token ≤ 1 / S_min.

This is a multiple-choice knapsack, solved exactly by OR-Tools CP-SAT (14B in < 0.1 s). No-good cuts
give the top 20 distinct solutions. Refinement re-measures penalties around the solution and re-solves
(≤ 3 rounds) because penalties are not exactly additive. Refinement and the shortlist are scored on the
**CPU build** (D36), so the comparisons that pick a configuration use the Pi's arithmetic.

**GA ablation** (not built): NSGA-II on the same per-gene vector, seeded from solver solutions, mutation
changes one gene's format, crossover swaps runs of genes, fitness measured on assembled files (CUDA,
with the same CPU correction as the solver's prices, D36), same evaluation budget as the solver pipeline;
its best candidates are re-measured on the CPU build like the solver's shortlist. It tests whether the additivity assumption costs quality.

**Baselines** (built): greedy KL-per-byte (HAWQ-style), random feasible search.

## 7. Shortlist ⬜

The 20 solver candidates per model are assembled and scored on the selection split: KL (mean and 99th
percentile) and top-1 agreement on the selection text, plus accuracy on the selection questions.
The top 2 per model go to final evaluation.

## 8. Final quality ⬜

Once, on the test split, with llama.cpp's **CPU** build (closest to the Pi's arithmetic):

| Metric | How |
|---|---|
| **Mean task accuracy (primary)** | HellaSwag, WinoGrande, ARC-C, MMLU 5-shot via `llama-perplexity` task modes; GSM8K 5-shot greedy exact match via `llama-server` + lm-evaluation-harness |
| Perplexity | context 2048, 449k test tokens |
| KL, top-1 agreement | vs BF16 reference |
| Generation sanity | 50 fixed prompts, 128 tokens greedy, compared to BF16 |

Winner per model by the pre-registered rule (D5): accuracy lead with 95% CI above +0.5 points, else
perplexity ≥ 1%, else RAM, else speed.

## 9. Raspberry Pi ⬜

The Pi confirms, it does not search (D4). Eight configurations: winners of all four models, runners-up
for 14B and 7B, and two baselines. Each must stay ≤ R_max peak RAM, pass validity checks (no throttling,
no OOM, tensor types as intended) and, if set, meet S_min. A failing winner is replaced by its runner-up.
A small agreement check confirms server-CPU and Pi perplexity match within 0.5%.

---

## Next: Phase A gate on 1.5B 📝 planned

Gate (PLAN.md §13): *a hand-built mixed GGUF loads with exactly the intended types; all §2 quality
metrics run.* Steps, in order, all on 1.5B:

### A1. Hand-built mixed file ✅ passed 2026-11-10
A deliberately scrambled configuration that exercises every format; it tests the plumbing, not quality.

- **Assignment** (fixed in `scripts/server/gate_a1.py`): `attn.i` = format[i mod 12],
  `mlp.i` = format[(5i + 3) mod 12] (a different order, so the two halves of a layer usually differ),
  `embd_head` = Q8_0. Format order is the menu order: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, IQ2_XS,
  IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL. All 12 formats are used. Examples: layer 0 = Q2_K / Q5_K,
  layer 1 = Q3_K / IQ3_XXS, layer 27 = Q5_K / Q4_0.
- **Predicted:** 0.90 GiB of tensor data (966,944,768 bytes, of which 579,584 are the F32 norms and biases), 5.01 bits/weight on average (inflated by the
  8.5-bit embedding, 15% of the parameters), RAM at 4096 context ≈ 1.41 GiB (+ KV 0.11, overhead 0.40).
- **Expected tensor counts by type**, which llama.cpp prints when it loads a model:

  | F32 | Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_0 | Q4_0 | IQ2_XS | IQ3_XXS | IQ3_S | IQ4_XS | IQ4_NL |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 141 | 18 | 21 | 18 | 21 | 14 | 15 | 17 | 14 | 17 | 14 | 14 | 14 |

- **Steps:** assemble from the 1.5B store (CPU container) → our read-back check → load with llama.cpp's
  **CPU** build (CPU container) and **CUDA** build (GPU container), capturing the load log → generate
  64 tokens greedily from 3 fixed prompts on each → perplexity on the first 16 chunks of the
  *sensitivity* text (never test) for the mixed file, uniform Q2_K, uniform Q8_0 and BF16.
- **Pass:**
  1. read-back: 0 mismatches, 0 missing tensors;
  2. llama.cpp's per-type counts equal the table above, on both builds;
  3. file size = predicted weight bytes + F32 bytes + metadata, and a sample of tensors is byte-identical
     to the source files;
  4. coherent text from both builds;
  5. perplexity ordering BF16 ≤ Q8_0 < mixed < Q2_K (sanity only; not a quality result).
- **Outputs:** `data/gate/a1/` with the assignment, the GGUF, load logs, generations and a report.json.
- **Result (all 5 checks pass):**
  1. read-back: 0 mismatches, 0 missing;
  2. llama.cpp per-type counts equal the table on both CPU and CUDA builds;
  3. tensor data 966,944,768 bytes = prediction exactly; file 972,896,416 bytes (5.95 MB header and
     alignment); first tensor of each of the 12 formats byte-identical to its source;
  4. both builds: "Three primary colors are red, yellow and blue." / a correct one-sentence CPU
     description / "42";
  5. perplexity, sensitivity text, 16 × 2048 tokens:

     | BF16 | Q8_0 | A1 mixed (CUDA) | A1 mixed (CPU) | Q2_K |
     |---|---|---|---|---|
     | 10.438 ± 0.212 | 10.449 ± 0.212 | 11.894 ± 0.240 | 11.923 ± 0.241 | 37.016 ± 0.884 |

     CPU and CUDA builds agree on the same file within 0.25%.
- **Problems found on the way** (all in our checking code, fixed; D27): llama.cpp at this commit only
  prints the loader's per-type counts at log verbosity ≥ 4 (`-lv 4`); the new `llama-cli` is a chat UI
  (used with `-st`); the coherence rule wrongly failed the correct one-word answer "42".

### A2. BF16 reference logits ✅ passed 2026-11-11 (chunk-count rule not met → N = 48)
Every KL and top-1 number compares a quantized model's output distribution with the unquantized model's.
`llama-perplexity --kl-divergence-base` saves the BF16 model's distributions once; later runs with
`--kl-divergence` read them back instead of re-running BF16.

**How llama.cpp stores it** (read from `tools/perplexity/perplexity.cpp` at the pinned commit): for every
scored token, the full log-probability distribution over the vocabulary, quantized to 16 bits, plus 4
header values: (151,936 + 4) × 2 = 303,880 bytes per token for Qwen2.5. At `-c 2048` only the second half
of each chunk is scored: 1,023 tokens per chunk ≈ 311 MB per chunk.

**Probe (1.5B, 4 chunks of the sensitivity text)** — sizes and noise floors measured before planning:

| Measurement | Result |
|---|---|
| File size | 1,243,509,748 bytes = 310.9 MB per chunk (prediction exact) |
| Storage floor: BF16 vs its own saved reference | mean KL 0.000000, max 0.00006, top-1 100% |
| Build floor: BF16 on the CPU build vs the CUDA-build reference | mean KL 0.00005, 99th pct 0.00027, top-1 99.56%, PPL ratio 1.0002 |
| Signal for scale: uniform Q4_K | mean KL 0.0627, 99th pct 0.285, top-1 85.7%, PPL ratio 1.061 |
| Signal for scale: uniform Q2_K | mean KL 1.262, 99th pct 4.68, top-1 45.4%, PPL ratio 3.35 |
| Speed (CUDA, 1.5B) | ~0.4 s per chunk for both the base run and a KL run |

Both floors are ~1,000× below the smallest effect of interest, so a CUDA-built reference is valid for
CPU-build evaluations too. One gotcha: the KL run must be given the same `-c 2048 -b 2048` as the base
run, or it fails ("failed to find a memory slot").

**Plan (1.5B):**

1. **Reference files** (GPU job, CUDA build, BF16, `-c 2048 -b 2048`):

   | Text | Tokens | Full chunks | Scored tokens | File |
   |---|---|---|---|---|
   | Sensitivity | 100,124 | 48 | 49,104 | 14.9 GB |
   | Selection | 152,684 | 74 | 75,702 | 23.0 GB |

   ≈ 38 GB per model; ~1 minute of GPU time for 1.5B. Each file gets a manifest entry: text sha256,
   chunks, context, image id, llama.cpp commit, file sha256, BF16 perplexity.
2. **Floors on the full sensitivity file:** BF16 against itself (storage) and BF16 on the CPU build
   (build). Expected ≈ the probe values; recorded as the noise floor for every later KL comparison.
3. **Chunk-count study** (feeds D24 and the Phase C cost): from a uniform Q6_K base, lower one gene at a
   time to Q2_K for 6 genes (`embd_head`, `attn.0`, `mlp.0`, `attn.14`, `mlp.14`, `mlp.27`), plus the
   base itself: 7 assembled files. Per-chunk KL is needed, but llama.cpp's per-chunk rows are *running
   means* printed to 5 decimals, so recovering single chunks from them adds up to ~2×10⁻⁴ of rounding
   error, as large as some single-gene effects. Instead each of the 48 chunks becomes its own text file,
   with its own BF16 reference, and each of the
   7 files gets one KL run per chunk; the run's summary prints mean KL to 6 decimals. Each chunk file
   holds the chunk's 2048 tokens followed by 2048 + 128 more (llama-perplexity refuses texts shorter
   than 2 × context even with `--chunks 1`; only the first chunk is scored), wrapping to the start of
   the text for the last chunks. llama-perplexity exits 0 even when it fails, so every run's output is
   checked (reference file non-empty, KL summary present) and the job stops at the first failure. Check: the mean of
   the 48 per-chunk values for the Q6_K base must match its single contiguous 48-chunk run within 2%.
   A bootstrap over chunks of the paired increase (gene file − base, chunk by chunk) then gives, for
   N = 6, 12, 24, 48 chunks:
   - the standard error of each gene's KL *increase* over the base;
   - whether the 6 genes keep the same ranking.

   **Decision rule (fixed now):** Phase C uses the smallest N whose relative standard error is ≤ 5%
   for every gene's increase and whose ranking matches N = 48 in ≥ 95% of bootstrap samples. If no N
   meets it (e.g. two genes are nearly tied), Phase C uses N = 48 and the near-tie is reported.
   Temporary disk: 48 per-chunk references (~15 GB) and 7 study files (~8 GB), deleted after the study.
4. **Write `data/reference/<model>/manifest.json`** and record the results here and in D24.

**Why it matters for cost:** the 1.5B speed scales roughly with parameter count, so 14B should be ≈ 4 s
per chunk. At 48 chunks that is ≈ 3 min per KL evaluation, and the 14B penalty table (≈ 1,000
evaluations) would take ≈ 53 GPU-hours. Step 3 tells us how far N can safely come down; format pruning
cuts the evaluation count.

**Pass:** both reference files written with manifests; floors within 2× of the probe values; the
chunk-count study produces a recommended N with its evidence.

**Result:**

| Item | Value |
|---|---|
| Sensitivity reference | 48 chunks, 14.92 GB, BF16 perplexity 9.588 |
| Selection reference | 74 chunks, 23.00 GB, BF16 perplexity 10.745 |
| Storage floor (full sensitivity) | mean KL 0.000000, top-1 99.996% |
| Build floor, CPU vs CUDA (full sensitivity) | mean KL 0.000048, top-1 99.57% |
| Study base, uniform Q6_K | mean KL 0.00581: per-chunk mean and contiguous run agree within 0.01% |

Chunk-count study, KL increase when one gene drops from Q6_K to Q2_K:

| Gene | Increase (48 chunks) | Relative SE at N = 6 / 12 / 24 / 48 |
|---|---|---|
| embd_head | 0.5573 | 6.6% / 4.6% / 3.3% / 2.3% |
| mlp.27 | 0.0293 | 4.9% / 3.6% / 2.5% / 1.8% |
| attn.0 | 0.0167 | 36% / 26% / 18% / 12.6% |
| mlp.0 | 0.0066 | 37% / 26% / 18% / 12.9% |
| mlp.14 | 0.0062 | 22% / 15% / 11% / 7.5% |
| attn.14 | 0.0029 | 15% / 11% / 7.5% / 5.3% |

Ranking matches N = 48 in 47% / 60% / 69% / 77% of bootstrap samples. **No N meets the rule** (≤ 5%
relative SE for every gene and ≥ 95% ranking match), so by the pre-registered fallback **Phase C uses
N = 48**. The reasons, reported as the rule requires:
- **Layer-0 genes are heavy-tailed:** mlp.0's per-chunk increase ranges 0.0022–0.0416 (median 0.0047); a
  few chunks hit layer 0 far harder than the rest. attn.0 behaves the same way.
- **Near-tie:** mlp.0 (0.0066) vs mlp.14 (0.0062) swap places in many resamples.
- Every gene's increase is positive on all 48 chunks, so the *direction* of each effect is never in doubt;
  the uncertainty is in the size of small effects.

What this means for Phase C is an open question for the team (D31).

**Problems found on the way** (fixed): llama-perplexity needs ≥ 2 × context tokens even with
`--chunks 1`, and it **exits 0 when it fails**, so the first study run silently produced nothing. Every
run's output is now checked. A polling loop also got this Mac blocked by the server's sshd (D30).

Kept: the two references and their manifest (`data/reference/<model>/`), all logs and `study.json`.
Deleted as planned: per-chunk references, study files and chunk texts (23 GB), and the probe (1.2 GB).

### A3. KL, perplexity, top-1 ⚠️ ran 2026-11-11: 4 of 5 checks pass; check 3 fails (D35)
A3 builds the **evaluator** that every later stage uses (penalty table, shortlist, finalists), then
validates it on 1.5B. It does not touch the test split.

**What llama.cpp gives us** (checked in `tools/perplexity` at the pinned commit):
- `--kl-divergence` against a saved reference prints mean KL (± SE), percentiles including the 99th,
  maximum, top-1 agreement ("same top p", ± SE) and PPL(Q)/PPL(base).
- Plain perplexity with `--ppl-output-type 1` prints, after each chunk, the running mean negative
  log-likelihood to 6 decimals. Because every chunk scores the same number of tokens (1,023 at
  `-c 2048`), each chunk's own mean is `i·m_i − (i−1)·m_(i−1)`, accurate to ≈ 2×10⁻⁴ at 219 chunks.
  That is what the pre-registered perplexity bootstrap (D5) needs; the default output (4-decimal running
  perplexity) is not precise enough (D32).
- It exits 0 on failure, so the evaluator checks every output itself (as in A2).

**A3.1 Evaluator** (`src/panther/quality/`):
- `kl.py`: run a KL evaluation of one GGUF against a reference, taking `-c`/`-b` from the reference
  manifest (D28); parse the summary into mean KL ± SE, 99th percentile, max, top-1 ± SE, PPL ratio.
- `ppl.py`: run perplexity with `--ppl-output-type 1`; parse the per-chunk running means and recover
  per-chunk NLL; check the recovered values reproduce llama.cpp's final perplexity within 0.01%.
- `record.py`: one JSON record per evaluation in `data/results/<model>/<eval_id>.json` (D33): what was
  evaluated (assignment sha256, GGUF sha256), how (build, image id, llama.cpp commit, text split,
  reference, `-c`/`-b`, threads), the metrics, per-chunk NLL and token counts, and wall time. The record
  carries everything `panther.stats.decision.Evaluation` needs.
- Unit tests on the parsers, using real llama.cpp output captured from the A1/A2 logs as fixtures.

**A3.2 Validation on 1.5B** (sensitivity and selection text only):

| # | Run | Build | Purpose |
|---|---|---|---|
| a | A1 file, KL vs sensitivity and vs selection | CUDA and CPU | End to end on a mixed file; CPU − CUDA difference should be ≈ the build floor (KL 0.00005) |
| b | All 12 uniform formats, KL + top-1 vs sensitivity | CUDA | Bytes-vs-KL table: first look at which formats are dominated (feeds format pruning, PLAN §6) |
| c | Perplexity, selection text: A1 file and BF16 | CUDA and CPU | Per-chunk NLL recovery; perplexity pipeline end to end without touching test |
| d | One KL run repeated | CUDA | Determinism: identical, or differences ≤ the build floor |

Time ≈ 10 min on the GPU and ≈ 12 min on the CPU; results are a few MB.

**Pass:**
1. every run produces a record that passes its own output checks;
2. recovered per-chunk NLL reproduces the final perplexity within 0.01% in every perplexity run;
3. A1's KL lies between uniform Q8_0 and uniform Q2_K; CPU − CUDA difference ≤ 3 × the build floor;
4. the repeated run is identical or within the build floor;
5. the uniform sweep's KL increases monotonically as bits per weight fall, *within each family*
   (K-quants; IQ-quants). Across families, any format that is both larger and worse than another is
   listed as dominated; that list is input to pruning, not a decision.

**Result.** Evaluator built as `src/panther/quality/parse.py` (parsers) and `run.py` (runner + record);
8 parser tests on real llama.cpp output, including a failed run that exited 0. Records in
`data/results/Qwen2.5-1.5B-Instruct/`, report in `data/gate/a3/report.json`.

| Check | Result |
|---|---|
| 1. every record ok | ✅ 21 records (17 CUDA, 4 CPU) |
| 2. per-chunk NLL recovery ≤ 0.01% | ✅ all four perplexity runs |
| 3. A1 between Q8_0 and Q2_K; CPU − CUDA ≤ 3 × build floor | ❌ first part passes (0.0019 < 0.154 < 1.254); CPU − CUDA = 0.0020 (sensitivity), 0.0017 (selection) vs limit 0.00015 |
| 4. repeated run | ✅ bit-identical (KL 0.154253 both times) |
| 5. monotone within families | ✅ K-quants and IQ-quants |

A1 on both builds:

| | CUDA | CPU | CPU − CUDA |
|---|---|---|---|
| KL, sensitivity | 0.15425 | 0.15629 | +1.3% |
| KL, selection | 0.14361 | 0.14532 | +1.2% |
| PPL, selection | 12.2656 | 12.2806 | +0.12% |
| BF16 PPL, selection | 10.7445 | 10.7446 | ≈ 0 |

Uniform formats, KL vs BF16 on the sensitivity text (CUDA; whole model in one format, embedding
included):

| Format | GiB | bits/weight | KL mean | KL p99 | top-1 |
|---|---|---|---|---|---|
| IQ2_XS | 0.430 | 2.31 | 1.599 | 6.03 | 38.4% (embedding is a Q2_K placeholder) |
| Q2_K | 0.478 | 2.62 | 1.254 | 5.03 | 44.0% |
| IQ3_XXS | 0.556 | 3.06 | 0.483 | 2.53 | 64.3% |
| Q3_K | 0.624 | 3.44 | 0.353 | 2.03 | 68.1% |
| IQ3_S | 0.624 | 3.44 | 0.300 | 1.46 | 69.8% |
| IQ4_XS | 0.770 | 4.25 | 0.0870 | 0.398 | 82.4% |
| Q4_K | 0.815 | 4.50 | 0.0638 | 0.317 | 84.8% |
| Q4_0 | 0.815 | 4.50 | 0.1149 | 0.617 | 80.9% |
| IQ4_NL | 0.815 | 4.50 | 0.0812 | 0.381 | 83.1% |
| Q5_K | 0.994 | 5.50 | 0.0184 | 0.086 | 91.4% |
| Q6_K | 1.185 | 6.56 | 0.0058 | 0.029 | 95.5% |
| Q8_0 | 1.533 | 8.50 | 0.0019 | 0.009 | 97.4% |

Dominated (another format is no larger and no worse): **Q3_K** (by IQ3_S), **Q4_0** and **IQ4_NL** (by
Q4_K). Input to pruning only: on the Pi, Q4_0 and IQ4_NL have fast repacked kernels and Q3_K and IQ3_S
do not, so speed may keep them (PLAN §6). The BF16 reference's own selection perplexity (10.7445, A2)
equals the evaluator's CUDA BF16 run, a cross-check of both tools.

**Why check 3 fails, and what it means.** The limit used the BF16 build floor, which was the wrong
yardstick: BF16 does the same arithmetic on both builds, but quantized formats do not. The CPU build
quantizes activations to 8 bits before each quantized matrix product, and the CUDA kernels work
differently, so a quantized model scores ~1.3% worse KL on the CPU build. The CPU build is also the
one that matches the Pi. Whether this matters depends on whether the gap is *consistent* across
formats, so that CUDA screening still ranks configurations the way the CPU build would. That is the
follow-up proposed in D35; the check stays failed.

### A3b. CPU activation correction (D35, D36, D37) ✅ D37 passed 2026-11-11
- **D35 result:** KL on the CPU build vs CUDA, uniform formats (sensitivity text): table in DECISIONS.md D35
  (outcome). Ranking identical on both builds. Rule (b) fails: Q6_K +123%, Q5_K +41%, Q4_K +12%, while Q8_0, Q4_0 ≈ 0%.
  In absolute terms every A256 format gains a similar +0.007 to +0.010.
- **D36 plan:** measure the activation-rounding cost a_g of 17 sampled genes on 1.5B (`embd_head` and
  layers 0, 4, …, 24, 27; base uniform Q8_0, one gene at a time in Q6_K, both builds), interpolate the rest, scale each gene's cost by its format's ratio from D35 (coarser formats cost more), then check additivity (C1) and prediction on 5 mixed configurations (C2).
  Full design and pass rules in DECISIONS.md D36.
- **D36 result:** C1 passes (+7.2%); C2 fails with the scaled correction on 2 of 5 configurations, while
  the unscaled correction would pass all 5. Almost all of the extra CPU error sits in two genes: the
  output head (`embd_head`, 0.0044) and the last layer's MLP (`mlp.27`, 0.0008). Details and the
  next step (D37) in DECISIONS.md.
- **D37 result: PASS.** The unscaled correction, confirmed on 4 fresh configurations, removes 84–92% of the
  CPU − CUDA gap on each. It is what Phase C's penalty table uses.

### A4. Benchmark tasks through llama-perplexity ✅ passed 2026-11-11
Turns the four log-likelihood benchmarks (HellaSwag, WinoGrande, ARC-Challenge, MMLU 5-shot) into the
files llama.cpp reads, scores them, and recovers a right/wrong result per question, which the
pre-registered decision rule (D5) needs for its paired bootstrap. GSM8K is A5.

**What llama.cpp does** (read in `tools/perplexity/perplexity.cpp` at the pinned commit):

| Mode | Input | Order | Score per answer | Per-question result |
|---|---|---|---|---|
| `--hellaswag` | text, 6 lines per item | always shuffled (seed 1) | mean log-prob per token of the ending | by position only |
| `--winogrande` | CSV: index, sentence (with `_`), option 1, option 2, answer | file order if all items are used | mean log-prob per token after the blank, last token excluded | printed per row |
| `--multiple-choice` | binary: count, offsets, then per item a question, answers and 0/1 labels | file order if all items are used | mean log-prob per token of the answer | from the running count (8 decimals) |

Defaults evaluate subsets (e.g. 400 HellaSwag items), so every run sets the task count to the full file.
A failed or skipped row is only visible as a missing item, so every run's item count is checked.

**Design (D38):**
- HellaSwag, ARC-C and MMLU all go through `--multiple-choice` (same scoring rule as `--hellaswag`, but in
  file order, so every result maps to a question id). WinoGrande goes through `--winogrande`.
- Prompts follow lm-evaluation-harness conventions:
  - **HellaSwag:** `activity_label + ": " + ctx_a + " " + ctx_b.capitalize()`, with lm-eval's text clean-up
    (strip, " [title]" → ". ", drop bracketed tags, collapse double spaces); answers = the 4 endings.
  - **ARC-C:** `"Question: {question}\nAnswer:"`; answers = the choice texts.
  - **MMLU 5-shot:** `"The following are multiple choice questions (with answers) about {subject}.\n\n"`,
    then the subject's 5 dev examples as `"{question}\nA. …\nB. …\nC. …\nD. …\nAnswer: {letter}\n\n"`,
    then the question in the same layout ending `"Answer:"`; answers = `A`, `B`, `C`, `D`.
  - **WinoGrande:** the sentence with its `_`, both options, answer 1/2, as llama.cpp's CSV.
- **Scoring differs from lm-eval in places (D25):** HellaSwag and ARC-C normalise by token count (lm-eval
  `acc_norm` uses byte length); WinoGrande averages the tokens after the blank and drops the last one
  (lm-eval sums them all); MMLU with single-letter answers is identical. All configurations are scored the
  same way, so comparisons are fair; the size of each difference is measured in A4.4.
- Context `-c 4096` for task runs (MMLU 5-shot prompts reach ~1,500 tokens; llama.cpp packs several items
  per context).

**A4.1 Converters** (`src/panther/quality/tasks.py`): split JSONL → `tasks/{selection,test}/*.bin|csv`
plus a manifest (item ids in file order, counts, sha256). Both splits are converted; only selection is
*evaluated* in Phase A (D34). Checks, in Python:
- the binary files are read back with a reader that mirrors llama.cpp's `deserialize` exactly;
- the WinoGrande CSV is read back with a port of llama.cpp's CSV parser (which only understands a quoted
  sentence and unquoted options); any row it would drop or mis-split is a converter failure.

**A4.2 Runner and parsers:** `panther.quality.run task …` writes a record with per-question results
(question id, correct 0/1) recovered from the output, and checks the count and that the recovered results
reproduce llama.cpp's final accuracy exactly.

**A4.3 Validation runs (1.5B, selection items only):**

| Model file | Build | Purpose |
|---|---|---|
| BF16 | CUDA | baseline accuracies |
| Q8_0, Q4_K, A1 | CUDA | behaviour across quality levels |
| ~~BF16, A1~~ | ~~CPU~~ | ~~full CPU runs~~ cut for time (user, 2026-11-11): replaced by a timing sample, A1 only, 100 random selection questions per task (D40) |

**A4.4 Independent check (D39):** reimplement llama.cpp's scoring rule in transformers (BF16 on the GPU)
for 200 seeded selection items per task, with the same prompt strings, and compare per-question choices
with llama.cpp's BF16 CUDA run. The same items are also scored the lm-eval way (byte-normalised; full
WinoGrande continuation) to report how much the definitions differ.

**Pass:**
1. converters round-trip exactly; the simulated llama.cpp CSV parser keeps every WinoGrande row;
2. every run evaluates exactly the file's item count, and per-question results reproduce the final
   accuracy exactly;
3. BF16 is at least 10 points above chance on every task (HellaSwag, ARC-C, MMLU: 25%; WinoGrande: 50%),
   and Q8_0 is within 2 points of BF16 on every task;
4. the independent check agrees with llama.cpp on ≥ 95% of items for every task;
5. CPU timings recorded (from the 100-question sample) and a Phase D cost estimate written (D40).
   CPU-vs-CUDA agreement on tasks is not checked in A4; the CPU/CUDA difference is handled by D36/D37.

**Result (all 5 checks pass).** Selection questions, 1.5B, CUDA build (accuracy %):

| File | HellaSwag | ARC-C | MMLU 5-shot | WinoGrande |
|---|---|---|---|---|
| BF16 | 67.10 | 46.10 | 61.25 | 76.30 |
| Q8_0 | 67.50 | 45.40 | 60.85 | 77.20 |
| Q4_K | 67.20 | 44.40 | 59.39 | 76.30 |
| A1 (mixed) | 64.90 | 44.00 | 55.46 | 73.80 |

Independent check (D39), 200 questions per task: agreement with llama.cpp 99.5 / 99.0 / 99.0 / 97.5%.
The lm-eval scoring rules on the same questions give HellaSwag +3.0 and ARC-C +3.0 points (byte-length
normalisation); MMLU and WinoGrande are identical.

CPU timing sample (A1, 100 questions per task; full CPU stage cut for time): HellaSwag 30 s, ARC-C 14 s,
MMLU 111 s, WinoGrande 9 s. Extrapolated to the full test split, **one 1.5B configuration takes ≈ 5.3 h on
the CPU build** (MMLU ≈ 4.3 h); on CUDA the same is minutes. See D40.

Problems found on the way (all silent in llama.cpp, all caught by the item-count check): 5-choice ARC
questions need `-np 5`; `-f` plus the WinoGrande loader drops the last row unless the file ends with two
newlines; question text containing U+2028 broke a `splitlines()` JSONL reader.

**Known cost risk (D40):** final task accuracy is planned on the CPU build (PLAN §2.1). The full MMLU test
is 14,042 five-shot prompts (~11M tokens). At the 1.5B CPU rate seen so far that is hours per
configuration, and ~10× more for 14B. A4's CPU timings give the real figure; the Phase D plan is decided
from it, before Phase D.

### A5. GSM8K 📝 planned
GSM8K is generated, not scored by log-likelihood, so it runs through `llama-server` (greedy decoding) in the
same container as the client, over localhost only (GPU containers have no outside network).

**Definition (D41):** a copy of lm-evaluation-harness 0.4.13's `gsm8k` task (read from the pinned package):
- prompt: the 5 fixed few-shot exemplars from the splits as `"Question: {q}\nAnswer: {answer}"`, joined by
  `"\n\n"`, then `"\n\nQuestion: {q}\nAnswer:"`; the full reference answers, calculator annotations included;
- generation: greedy (temperature 0), stop at `Question:`, `</s>`, `<|im_end|>`, at most 256 tokens;
- extraction: **strict-match** (primary, fixed now): the first `#### (-?[0-9.,]+)`; **flexible-extract**
  (reported): the last match of `(-?[$0-9.,]{2,})|(-?[0-9]+)`;
- scoring: exact match after removing `,`, `$`, everything up to `#### `, and a trailing `.`; case-insensitive.

lm-eval itself is not used because it fetches the dataset from the Hub, and GPU containers have no network;
the definition is copied, with unit tests of the extraction rules on lm-eval's own patterns.

**Runs (selection, 500 questions):** BF16, Q8_0 and A1 on CUDA (8 parallel server slots); BF16 again
for determinism; CPU build on the first 20 questions of A1 for timing (D40).

**Pass:**
1. every question gets a completion (no request errors or empty outputs) in every run;
2. BF16 strict-match accuracy ≥ 40% (a sanity floor; Qwen2.5-1.5B-Instruct is reported far above it);
3. Q8_0 within 5 points of BF16 (500 questions; ≈ 2 SE);
4. the repeated BF16 run's accuracy is within 1 point of the first (batched GPU decoding is not
   guaranteed bit-identical; the number of differing completions is reported);
5. CPU timing recorded.

### A6. Generation sanity 📝 planned
**Prompts (D26, decided now):** 50 prompts, 10 per task, drawn with a fixed seed from the selection items of
HellaSwag ("Continue this text: …"), ARC-C and MMLU (question and lettered options, "Answer with the letter
and one sentence of reasoning."), WinoGrande ("Fill in the blank: …") and GSM8K (the question). Sent as a
single user message through the model's chat template (`/v1/chat/completions`), greedy, 128 tokens. Stored
with the splits (`splits/gen_prompts.json`).

**Measures, per configuration against BF16:** matched prefix (tokens until the first difference), share of
outputs identical to BF16, and a degeneration flag (empty output, or more than half of its 4-grams repeated).

**Runs:** BF16 twice, Q8_0, Q4_K, A1 (CUDA).

**Pass:**
1. every prompt produces non-empty output in every run;
2. the BF16 repeat matches the first BF16 run exactly on ≥ 48 of 50 prompts;
3. no degenerate outputs for BF16 or Q8_0;
4. mean matched prefix: Q8_0 ≥ A1;
5. a spot-check of 10 A1 outputs, read and judged coherent or not, recorded here.

### A7. End to end on 1.5B 📝 planned
Assemble, from the A3–A6 records only (no new runs), one evaluation each for **BF16** and **A1** with every
PLAN §2.1 quality metric on selection data: task accuracy (HellaSwag, ARC-C, MMLU, WinoGrande, GSM8K),
perplexity (selection text), KL and top-1 (selection reference), generation sanity. Then run the
pre-registered decision rule (`panther.stats.decision.compare`, paired bootstrap, 10,000 resamples) for
**BF16 vs A1** and for **Q8_0 vs Q4_K** (task accuracy only for the latter if perplexity was not measured).

**Pass (the Phase A gate's second half):** every metric is present for both configurations; the decision
rule runs and returns a winner with its deciding criterion; a results manifest lists every record used.

---

## Proof of concept: 7B vs 14B at 7.0 GiB 🟡 started 2026-11-11 (D42)

| # | Stage | 7B lane | 14B lane |
|---|---|---|---|
| P1 | Download (CPU job, network) | ~10 min | ~20 min |
| P2 | Tensor store for the model's formats (GPU job) | ~30 min | ~1.5 h |
| P3 | BF16 references: sensitivity (12 chunks), selection | ~10 min | ~20 min |
| P4 | CPU correction (head, last MLP) | ~15 min | not needed |
| P5 | Penalty table, 12 chunks, 3 KL runs at a time | ~1 h | ~4 h |
| P6 | Solver at 7.0 GiB, top 5 + uniform baseline scored on selection | ~30 min | ~1 h |
| P7 | Verdict on a fixed test subset, decision rule (after the user's OK) | ~30 min | ~1 h |

7B result ≈ 2.5–3 h after start; 14B ≈ 7–8 h.

### P8. Pi bundle (server, after P7) 📝 planned
`poc.py bundle` writes `data/pi_bundle/` on the server: the 7B and 14B winners and their uniform baselines
(7B Q6_K, 14B IQ3_XXS) as GGUF files, a 4,096-token agreement text, and `manifest.json` with each file's
sha256, assignment, predicted RAM at 4,096 context, expected per-type tensor counts, and the server CPU-build
perplexity on the agreement text. The final chain on the server runs P7 then P8 once both lanes finish.

### Phase E for the proof of concept: Raspberry Pi 5 📝 planned
Everything on the Pi runs as an ordinary user; only installing build tools needs `sudo`, which the user runs.
Tools in `pi/` (Python standard library only, plus llama.cpp built on the Pi at the pinned commit).

| # | Step | What | Time (est.) |
|---|---|---|---|
| E1 | Setup | apt build tools (user, sudo); clone + build llama.cpp at the pinned commit, CPU, native for the A76 | ~30 min |
| E2 | Bundle | copy `pi_bundle/` from the server; verify every sha256 | ~10 min |
| E3 | RAM + smoke | each file once (W_short): loads, generates, peak RAM vs prediction (±2%), measured runtime overhead | ~20 min |
| E4 | Threads | 2, 3, 4 threads on both winners (W_short, 1 warm-up + 2) | ~30 min |
| E5 | Confirmation | all 4 files at the best thread count, W_ref 256/64, 1 warm-up + 3 | ~1.5 h |
| E6 | Agreement | perplexity on the 4,096-token agreement text for both winners; must match the server CPU build within 0.5% | ~30 min |
| E7 | Generalisation | the recommended file at W_gen 128/32 and 1024/128 | ~30 min |

Measured per run (llama-server, `--load-mode none` so weights sit in ordinary memory, `-c 4096`, greedy,
`ignore_eos`, prompts given as exact token counts): prefill and decode tokens/s from the server's timings,
inter-token latency p50/p95 from streamed token arrival times, end-to-end latency, peak RAM (`VmHWM` of the
server process), minimum system `MemAvailable`, CPU temperature at start and end, and throttling flags when
`vcgencmd` is available. A run is invalid if peak RAM exceeds 7.0 GiB, `MemAvailable` drops below 256 MiB, the
SoC throttles, or the start temperature is above 70 °C (the queue waits up to 5 minutes to cool first).
`pi/analyze.py` writes `results/report.md` and `report.json`, including the RAM-prediction check and the
recommendation: the verdict winner, if it is within budget and valid on the Pi; otherwise the other model's winner.

### PoC run log (2026-11-11, unattended overnight)
- 00:44 UTC both lanes started. Smoke mode (`PANTHER_POC_SMOKE=1`, 1.5B, tiny settings, selection data)
  ran every new step end to end before the real lanes reached them: **passed**.
- Fixed on the way (all pushed):
  - the Pi recommendation now follows the decision rule (found by the smoke run: the rule preferred 1.5B's
    uniform baseline, the bundle had recommended the solver's winner);
  - 14B's IQ2_XS source aborted: llama-imatrix skips the untied output tensor, so IQ2_XS is excluded from the
    `head` gene too (D21 extended);
  - the final launcher misread a restarted lane's old failure line; it now judges only each lane's latest run;
  - penalty measurements ran ~1/min (three Python threads sharing ~2 cores while the GPU idled); now six
    processes, ~7/min.
- 7B CPU correction (D37 method, 12 chunks): output head a = 0.00180, last MLP a = 0.00049, the same pattern as
  1.5B.
- Expected: 7B shortlist ~03:00 UTC, 14B shortlist ~05:30, Pi bundle ~07:00 UTC.
