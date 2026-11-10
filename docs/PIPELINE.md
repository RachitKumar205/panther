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

Cost: genes × formats measurements (14B: 98 × 11 ≈ 1,000). Pruning formats that are never best on
both size and quality, and timing the 1.5B pilot, will give the real figure before 14B starts.

## 6. Solver ⬜ (code ✅)

Choose one format per gene to minimise the sum of penalties subject to

- **RAM:** exact tensor bytes + KV cache at 4096 context + measured runtime overhead ≤ R_max;
- **speed (optional):** predicted decode time per token ≤ 1 / S_min.

This is a multiple-choice knapsack, solved exactly by OR-Tools CP-SAT (14B in < 0.1 s). No-good cuts
give the top 20 distinct solutions. Refinement re-measures penalties around the solution and re-solves
(≤ 3 rounds) because penalties are not exactly additive.

**GA ablation** (not built): NSGA-II on the same per-gene vector, seeded from solver solutions, mutation
changes one gene's format, crossover swaps runs of genes, fitness measured on assembled files, same
evaluation budget as the solver pipeline. It tests whether the additivity assumption costs quality.

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

### A1. Hand-built mixed file 📝
- Assignment: every one of the 12 formats used by at least one gene, attention and MLP genes cycling
  through formats in different orders, `embd_head` at Q8_0. Fixed in a script, so it's reproducible.
- Assemble from the tensor store (CPU job), verify read-back, then load it in llama.cpp and compare
  llama.cpp's own per-type tensor counts in its load log with the expected counts.
- **Pass:** zero type mismatches, counts equal, model generates text.

### A2. BF16 reference logits 📝
- `llama-perplexity --kl-divergence-base` on the BF16 model saves its output distributions, which every
  KL and top-1 measurement compares against. Proposed: the sensitivity text and the selection text.
- **Size concern:** llama.cpp stores ~2 bytes per vocabulary entry per scored token. Qwen's vocabulary
  is 151,936, and at context 2048 llama-perplexity scores the second half of each chunk, so 100k tokens
  ≈ 50k scored ≈ 15 GB; sensitivity + selection ≈ 38 GB per model, ~150 GB for four models. Fits on
  `/s3` (~590 GB free) but is large. Alternative to decide in A2: fewer sensitivity tokens for KL, or
  generate the reference per model and delete it after that model's search.
- The reference is BF16 (the models' native precision), not FP16 as PLAN.md §2.1 says (D23).
- Storage choice: D24.

### A3. KL, perplexity, top-1 📝
- `llama-perplexity --kl-divergence` on the A1 file against A2: mean KL, 99th percentile, top-1
  agreement ("same top"), perplexity. Check the numbers are sane (mixed ≈ between its formats' extremes).

### A4. Task formats for llama-perplexity 📝
- Convert the split JSONL files into llama.cpp's input formats: HellaSwag text format, WinoGrande CSV,
  and the binary multiple-choice format for ARC-C and MMLU (with 5-shot MMLU prompts).
- **Verify first** against `tools/perplexity` source at the pinned commit: exact file formats and how
  each mode scores (e.g. whether HellaSwag is length-normalised like `acc_norm`). Differences from
  lm-eval's definitions get a decision entry.
- Sanity check: BF16 1.5B accuracies near Qwen's published numbers.

### A5. GSM8K 📝
- `llama-server` and lm-evaluation-harness (`local-completions`) in the same GPU container, talking over
  localhost (container still has no outside network). 5-shot with the fixed exemplars, greedy, exact match.

### A6. Generation sanity 📝
- 50 fixed prompts, 128 tokens greedy, compare with BF16. **Open:** the prompt set doesn't exist yet;
  proposal: 50 items drawn by seed from the selection questions of all five tasks, stored in the splits.

### A7. Run everything on the A1 file and BF16; write a results manifest 📝
- **Pass:** every metric produces a number; outputs and timings saved with a manifest.
