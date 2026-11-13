# Panther

For each RAM budget on an 8 GB Raspberry Pi 5 (1, 3, 5 and 7 GiB), find a quantization of an open model that answers
**better than the best existing quantization that fits**, by giving every weight matrix its own type
(heterogeneous quantization). Quality is measured on a server GPU; the Pi measures only RAM and speed of our winners.

Live progress: <https://huggingface.co/therachitkumar/panther-status>

- [Results so far](#results-so-far)
- [How it works](#how-it-works): every term, from what a weight is to the full pipeline
- [Run it](#run-it) · [Layout](#layout) · [Related work](#related-work)

## Results so far

First method (genetic search over llama.cpp's own types, fitness = accuracy − 10 × KL), held-out test, 2026-11-13.
"Points lost" is how far below the unquantized original.

| Contest | Verdict (ours − preset, 95% interval) | Original | Preset | Ours | Preset loses | We lose |
|---|---|---|---|---|---|---|
| Qwen2.5 1.5B @ 1 GiB | **WIN** +3.59 [+2.23, +4.97] | 55.77 | 39.98 | 43.56 | 15.79 | 12.21 |
| Qwen2.5 14B @ 5 GiB | **WIN** +1.51 [+0.24, +2.80] | 76.46 | 64.17 | 65.67 | 12.29 | 10.79 |
| Qwen2.5 7B @ 5 GiB, GSM8K in the fitness | **WIN** +1.39 [+0.35, +2.45] | 70.42 | 69.25 | | | |
| Qwen2.5 14B @ 7 GiB | TIE +0.55 [−0.15, +1.23] | 76.46 | 74.65 | 75.20 | 1.81 | 1.26 |
| Qwen2.5 7B @ 3 GiB | TIE +0.03 [−1.24, +1.28] | 70.42 | 64.69 | 64.72 | 5.73 | 5.70 |
| InternLM2.5 20B @ 7 GiB | TIE +0.76 [−0.22, +1.74] | 74.22 | 69.08 | 69.83 | 5.14 | 4.39 |
| Mistral Small 24B @ 7 GiB | TIE −0.11 [−1.10, +0.89] | 78.79 | 68.41 | 68.30 | 10.38 | 10.49 |
| Qwen2.5 7B @ 5 GiB | LOSE −1.13 [−2.08, −0.24] | 70.42 | 69.25 | 68.12 | 1.17 | 2.30 |

What this showed: choosing types per matrix wins a few points at most, because llama.cpp's presets are already
heterogeneous, and the big losses (10–16 points at 2–3 bits per weight) come from how each weight is rounded. Hence
the second method below. On the Pi: 1.5B @ 1: 0.71 GiB peak, 9.0 / 14.3 tokens/s (writing / reading); 7B @ 3: 2.73 GiB,
2.2 / 2.9; 7B @ 5: 4.69 GiB, 2.4 / 10.3; 14B @ 5: 4.74 GiB, 1.2 / 1.5; 14B @ 7: 6.72 GiB, 0.9 / 1.1.

## How it works

### 1. The basics

**A model** is a long chain of **layers**. Each layer holds a few large tables of numbers, **matrices**: four for
*attention* (which earlier words to look at: `attn_q`, `attn_k`, `attn_v`, `attn_output`) and three for the
*feed-forward* part (the processing: `ffn_gate`, `ffn_up`, `ffn_down`). The numbers in them are **weights**. Qwen2.5
14B has about 14 billion weights in 338 matrices (48 layers × 7, plus the word table and the output layer).

**What it computes.** Given text, the model gives a **probability for every possible next word** ("The capital of
France is" → Paris 92 %, Lyon 3 %, …). Answering a question is choosing likely next words, one after another.

**Quantizing** stores each weight in fewer bits. The original uses 16 bits per weight: 28 GB for the 14B, far beyond
the Pi. A **RAM budget** (1, 3, 5 or 7 GiB) must hold the weights plus the **KV cache** (the model's memory of the
text so far, 4,096 tokens at 8 bits) plus llama.cpp's own overhead. What is left, divided by the number of weights,
is the average **bits per weight** we can afford: about 2.6 for the 14B at 5 GiB. Fewer bits lose information, so the
model gets worse; the whole job is to lose as little as possible.

### 2. Types: how weights are stored

A **type** is one storage recipe. All of them cut a matrix into **blocks** (e.g. 32 or 256 weights of one row) and
store, per block, one **scale** (an ordinary decimal number) and each weight as a **small whole number**. A weight's
value is scale × whole number.

Example, 2 bits (whole numbers −2, −1, 0, 1), scale 0.3:

| Original | ÷ 0.3 | Rounded | Stored value |
|---|---|---|---|
| 0.31 | 1.03 | 1 | 0.30 |
| −0.58 | −1.93 | −2 | −0.60 |
| 0.02 | 0.07 | 0 | 0.00 |

Every weight ends up a little off: the **rounding error**.

- **K types** (Q2_K … Q6_K): exactly that, with 2 to 6 bits per whole number (4 to 64 levels); the scales are stored
  compactly too ("K"), so Q4_K costs about 4.5 bits per weight. Fast on the Pi's processor.
- **IQ types** (IQ1_S, IQ2_XS, IQ3_S, IQ4_XS…): below about 3 bits, rounding each weight on its own is poor, so these
  store **groups of 8 weights as one pattern** from a fixed list of a few hundred allowed patterns (plus signs). Real
  weights cluster in typical shapes, so the list fits them better than independent rounding. Better quality at very
  few bits, slower on the Pi (a lookup per group).
- **Legacy types** (Q4_0, Q5_0, Q8_0): the simplest block-and-scale recipes. **TQ** types: only for models trained
  to have three values per weight. **MXFP4**: a 4-bit floating-point type for mixture-of-experts models.

**Heterogeneous** means each matrix gets its own type: important matrices more bits, unimportant ones fewer, same
total. llama.cpp cannot mix types *inside* a matrix, so the choice is per matrix. **Presets** are llama.cpp's
ready-made heterogeneous recipes (e.g. "IQ2_XXS for most matrices, more for a few known-important ones"). The largest
preset that fits the budget is our **competitor**; if none fits, the smallest one competes (with more RAM than we
have). The **importance matrix** records, on ordinary text (the **calibration text**), which inputs to each matrix are
usually large: the weights that multiply them matter more and are rounded more carefully.

### 3. Rounding: the lever for big gains

- **Plain rounding** (llama.cpp): each block is rounded on its own, guided by the importance matrix. It never corrects
  for errors already made.
- **Error-compensated rounding** (GPTQ): a matrix's job is its *outputs*, each a sum of weight × input. When one weight
  is rounded up, the next ones are nudged down to cancel it, so the sums stay right. Columns are rounded in order and
  each one's error is spread over the columns not yet rounded, weighted by how the inputs move together (measured on
  calibration text). Individual weights may move further from their originals; the matrix's outputs move less. Same
  type, same file size, same speed. **Used for every type copy** (`src/panther/rounding.py`); measured below.
- **Learned rounding** (GSQ): starts from GPTQ and *learns* for each weight whether to go up or down a level.
- **Saliency** (GuidedQuant): one backward pass through the original measures how much each layer's output error moves
  the final answer, so rounding spends its effort where it matters.
- **Rotation** (SpinQuant): a fixed reversible mixing of each layer's directions spreads out extreme values, so every
  type rounds more cleanly; baked into the weights, the architecture is unchanged. To be tested.

**Which types work, and what smarter rounding buys** (Qwen2.5 7B, every matrix in one type, 2026-11-13). Drift is
KL from the original on the selection text (lower is better):

| Type | Bits/weight | Drift, llama.cpp rounding | Drift, error-compensated | |
|---|---|---|---|---|
| Q8_0 / Q6_K / Q5_K | 8.5 / 6.6 / 5.5 | 0.0011 / 0.0021 / 0.0048 | | near-lossless |
| Q4_K | 4.5 | 0.0157 | 0.0145 | −8 % |
| IQ4_XS / IQ4_NL / Q4_0 | 4.25 / 4.5 / 4.5 | 0.020 / 0.020 / 0.030 | | worse than Q4_K |
| **Q3_K** | 3.44 | 0.134 | **0.064** | **−52 %**, now equal to IQ3_S |
| IQ3_S | 3.44 | 0.064 | 0.065 | no change |
| IQ3_XXS | 3.06 | 0.102 | | |
| Q2_K | 2.63 | 0.241 | 0.214 | −11 % |
| IQ2_XS | 2.31 | 0.333 | 0.302 | −9 % |
| IQ2_XXS / IQ1_M | 2.06 / 1.75 | 0.562 / 0.980 | | |
| MXFP4 / TQ2_0 | 4.25 / 2.06 | 0.235 / 13.2 | | do not work for normal models |

Decision: every matrix may take any of the 14 K and IQ types from IQ1_S (1.56 bits) to Q8_0 (8.5 bits), all built
with error-compensated rounding; nothing else is restricted, so the search is as heterogeneous as the budget allows
(the only exception is llama.cpp's own: types that need importance-matrix data cannot be used for the word table,
which has none). MXFP4, TQ and the legacy Q4_0/Q5_0 are out: they lost to K types at the same size or do not work. At about 2 bits the gain is small because errors are compensated between blocks of 256 weights,
not inside them; learned rounding (GSQ) is the known fix if needed.

### 4. Measuring quality

| Measure | Meaning | Better |
|---|---|---|
| **Accuracy** | % of questions answered right: MMLU (knowledge), ARC (science), HellaSwag (common sense), WinoGrande (pronouns), GSM8K (math word problems, the model writes its answer) | higher |
| **KL** | how differently the quantized model predicts the next word than the original, averaged over text; 0 = identical | lower |
| **Drift** | KL in plain words: how far the model moved from the original | lower |
| **Flips** | % of questions whose right/wrong differs from the original's | lower |
| **Perplexity** | how surprised the model is by real text | lower |

Accuracy on a few hundred questions is noisy: a genome can be lucky on exactly those (the 24B's lead fell from +3.4 on
the search questions to −0.1 on the held-out test). Flips and drift measure the damage directly and are far steadier
("Accuracy is Not All You Need", NeurIPS 2024, recommends exactly these). **Drift on the original's own answers**: the
original writes its answers once (GSM8K solutions included); every candidate's KL is measured on that text, which tests
whether it still reasons like the original, math included, without the slow step of making each candidate write.

The questions are split once and never mixed: **calibration** (importance matrix, rounding), **selection** (the search
scores candidates on it), **held-out test** (used once, at the end, for the reported result).

### 5. The pipeline, step by step (`scripts/server/pipeline.py`)

Once per model:

1. **fetch**: download the original at a pinned revision.
2. **store**: convert it to a 16-bit **GGUF** (llama.cpp's file format), compute the importance matrix, and quantize
   the *whole* model once into every one of the 14 types (**type copies**), each matrix with error-compensated
   rounding. Building any candidate later is picking each matrix from the right copy, which takes seconds. The copies
   of a large model take a lot of disk (32B: about 200 GB); the step waits for free disk and the page says so.
3. **error**: the **offline error** of every matrix in every type (importance-weighted squared error), a cheap guess
   the search uses to repair and to fit its model.
4. **refs**: the original's reference outputs: next-word probabilities on the selection text and on its own answers,
   and which selection and test questions it gets right.
5. **preset**: build llama.cpp's presets near the budget; of the two largest that fit, the one more accurate on the
   held-out test is the competitor.

The search (**genetic algorithm**, GA):

6. **ga**: a **genome** is one type for every matrix, always inside the RAM window (−1 % / +0.5 % of the budget; a
   genome that is too big is **repaired** by lowering the matrices that lose least per byte saved).
   - **Start population**: the preset's own layout, every uniform type, random mixes, mutants of the preset.
   - **Children**: **mutation** (change 1–3 matrices by one step, or by two or three, or one whole group = one matrix
     role in one quarter of the depth) and **crossover** (whole layers from two parents). Each round a **solver**
     (CP-SAT) also proposes 4 genomes from a simple model fitted to every score so far, which predicts where extra
     bits pay.
   - **Score** (lower is better): **flips** against the original on 900 selection questions + 10 × **drift** on the
     original's own answers. Accuracy is recorded as a check.
   - **Steady state**: up to 16 genomes scored at a time, as many as the GPU holds next to the other contests; a
     new child starts whenever one finishes.
   - **Stop**: after at least 30 genomes, once 15 in a row bring no gain of 0.2; at most 70.

The end:

7. **final**: **distillation** of the top 3 (`src/panther/distill.py`): the original is the teacher, each finalist
   the student. Every K and IQ type stores one scale per block of 256 weights, and a block's values are proportional
   to it; distillation learns one multiplier per block (and the norm weights) so that the student's next-word
   probabilities match the teacher's top 64 on the original's own answers and the calibration text (300 steps). The
   whole numbers stay frozen: same file size, same speed. The new scales are written back into the GGUF. Then the 3
   are scored on all 4,500 selection questions; the lowest score wins. **Guard**: the winner must not be less
   accurate than the real preset file on those questions. Then winner and preset on the held-out test; a **paired
   bootstrap** (resampling the same questions for both) gives the difference with a **95 % interval**: **WIN** if the
   whole interval is above 0, **LOSE** if below, **TIE** otherwise.
8. **bundle, upload**: the winner's file and a manifest to a private Hugging Face repo; the **Pi** downloads it and
   measures real peak RAM and speed (tokens per second, reading the prompt and writing the answer).

| | First method | Second method |
|---|---|---|
| Type copies | llama.cpp's plain rounding | error-compensated rounding |
| Search score | accuracy − 10 × KL on benchmark text | flips + 10 × drift on the original's own answers |
| After the search | nothing | distillation of the top 3 |
| Search itself | GA + solver | unchanged |

## Run it

One command, from your machine (it syncs the code to the server and runs there; see
[deploy/server](deploy/server/README.md) for the one-time setup):

```bash
./panther start 1.5b@1 7b@3 7b@5 14b@5   # start contests (model size @ RAM budget, GiB); all at once, packed
./panther pause                          # pause everything; ./panther resume continues
./panther stop 7b@5                      # stop one contest (or all: ./panther stop); start it again to continue
./panther status                         # what runs, GPU, disk; the live page has everything else
./panther test                           # the tests, on your machine
```

Every event (queued, started, each step, the search's progress, every result, pauses, failures) goes to the live
page. Every step skips work already done, so nothing is lost by stopping. GitLab, which shares the server, is
checked every minute; if it becomes unhealthy everything pauses until it recovers.

On the Pi ([pi](pi/README.md)): `nohup python3 pi/panther_pi.py <hf-user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &`

## Layout

| Module | What |
|---|---|
| [configs](configs/README.md) | Models (pinned revisions), budgets and quantization types |
| [src/panther](src/panther/README.md) | Model shape and RAM model, llama.cpp runs and their records, statistics, solver |
| [scripts/server](scripts/server/README.md) | The pipeline and the status page |
| [deploy/server](deploy/server/README.md) | Container image and jobs, the contest runner, server setup and removal |
| [pi](pi/README.md) | Pi setup and the RAM/speed check of a winner |


## Related work

- Choosing a type per layer by search: **EvoPress** (ICML 2025, evolutionary search with KL fitness), many
  sensitivity- and RL-based allocators; in GGUF, llama.cpp's `--target-bpw`, Unsloth Dynamic quants, Bartowski's
  per-tensor layouts. Their gains are modest too, as ours were.
- Better rounding: **GPTQ**; **GSQ** (2026, learned rounding written back into GGUF Q2_K: Qwen3-8B 50.0 → 56.3);
  **YAQA** (2025, rounding against the whole model's KL, ~30 % lower KL); **GuidedQuant** (ICML 2025, end-loss
  saliency). Strongest 2-bit methods (QTIP, PV-Tuning, AQLM) need their own formats and kernels.
- Measuring: **Accuracy is Not All You Need** (NeurIPS 2024): flips and KL over accuracy.
- What we add: allocation search over better-rounded type copies, K and IQ types, stock GGUF, RAM budgets that
  include the KV cache, verified on a real Pi, and a search score that does not reward luck.
