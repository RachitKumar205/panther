# Panther

A Raspberry Pi 5 has 8 GB of RAM. For each **RAM budget** (1, 3, 5 and 7 GiB) and several open language models
(Qwen2.5 1.5B, 7B, 14B, 32B; InternLM2.5 20B; Mistral Small 24B), Panther builds a compressed version that should
**answer better than the best compressed version that already exists** for that budget: llama.cpp's own preset.
Each model-and-budget pair is a **contest**, written like `7b@3`.

Live progress: <https://huggingface.co/therachitkumar/panther-status>

- [1. The basics](#1-the-basics) · [2. Types](#2-types-how-weights-are-stored) ·
  [3. The three quality levers](#3-the-three-quality-levers) · [4. Measuring quality](#4-measuring-quality)
- [5. The steps of one contest](#5-the-steps-of-one-contest) · [6. In depth](#6-in-depth-every-number-and-formula)
- [7. Running it](#7-running-it-one-command) · [8. The code](#8-the-code) · [9. Related work](#9-related-work)

## 1. The basics

- **A model** is a chain of **layers**. Each layer holds 7 big tables of numbers, **matrices**: four for *attention*
  (which earlier words to look at: `attn_q`, `attn_k`, `attn_v`, `attn_output`) and three for *feed-forward*
  (processing: `ffn_gate`, `ffn_up`, `ffn_down`), plus the **word table** (words in) and the **output layer** (words
  out). The numbers inside are **weights**: billions of them.
- **What it does:** given text, it gives a **probability for every possible next word** ("The capital of France
  is" → Paris 92 %, Lyon 3 %…). Answering is picking likely next words, one after another.
- **Quantizing** stores each weight in fewer bits. The original uses 16 bits per weight (the 14B: 28 GB). A budget
  must hold the weights, the **KV cache** (the model's memory of the conversation, 4,096 tokens at 8 bits) and
  llama.cpp's overhead; what is left decides the average **bits per weight** (about 2.6 for the 14B at 5 GiB).
  Fewer bits, more damage; the job is to lose as little as possible.

## 2. Types: how weights are stored

A **type** is one storage recipe. Every type cuts a matrix into **blocks** (32 or 256 weights of a row) and stores
per block one **scale** (a decimal number) and each weight as a **small whole number**; a weight's value is
scale × whole number. Rounding each weight to an allowed value leaves a **rounding error**. Example, 2 bits (whole
numbers −2, −1, 0, 1), scale 0.3:

| Original | ÷ 0.3 | Rounded | Stored value |
|---|---|---|---|
| 0.31 | 1.03 | 1 | 0.30 |
| −0.58 | −1.93 | −2 | −0.60 |
| 0.02 | 0.07 | 0 | 0.00 |

- **K types** (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K): 2 to 6 bits per whole number; fast on the Pi.
- **IQ types** (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS): groups of 8 weights stored as one
  pattern from a fixed list; better at very few bits, slower on the Pi.
- **Q8_0**: 8 bits, almost lossless.

**14 types**, 1.56 to 8.5 bits per weight. **Heterogeneous** means each matrix gets its own type: important ones
more bits, unimportant ones fewer, same total. **Any matrix may take any of the 14 types**; the only exception is
llama.cpp's own: the word table has no importance data, so it cannot use the 4 types that need it. The **importance
matrix** records, on ordinary text (the **calibration text**), which inputs to each matrix are usually large: the
weights that multiply them matter more and are rounded more carefully. **Presets** are llama.cpp's ready-made mixes.

Every llama.cpp type was tested on Qwen2.5 7B (every matrix in one type; **drift** = how differently it predicts the
next word than the original, lower is better):

| Type | Bits/weight | Drift, llama.cpp rounding | Drift, smarter rounding | |
|---|---|---|---|---|
| Q8_0 / Q6_K / Q5_K | 8.5 / 6.6 / 5.5 | 0.0011 / 0.0021 / 0.0048 | | near-lossless |
| Q4_K | 4.5 | 0.0157 | 0.0145 | −8 % |
| IQ4_XS / IQ4_NL / Q4_0 | 4.25 / 4.5 / 4.5 | 0.020 / 0.020 / 0.030 | | worse than Q4_K |
| **Q3_K** | 3.44 | 0.134 | **0.064** | **−52 %**, equals IQ3_S |
| IQ3_S | 3.44 | 0.064 | 0.065 | no change |
| IQ3_XXS | 3.06 | 0.102 | | |
| Q2_K | 2.63 | 0.241 | 0.214 | −11 % |
| IQ2_XS | 2.31 | 0.333 | 0.302 | −9 % |
| IQ2_XXS / IQ1_M | 2.06 / 1.75 | 0.562 / 0.980 | | |
| MXFP4 / TQ2_0 | 4.25 / 2.06 | 0.235 / 13.2 | | do not work for normal models |

MXFP4 and TQ2_0 do not work for normal models and Q4_0/Q5_0 lose to the K types at the same size, so they are out.

## 3. The three quality levers

1. **Smarter rounding** (error compensation, GPTQ). A matrix's outputs are sums of weight × input; when one block's
   rounding pushes a sum up, the next blocks are nudged to pull it back. Same type, size and speed; the outputs stay
   closer to the original's. Every type copy is built this way (gains in the table above).
2. **The search** (a genetic algorithm) chooses each matrix's type: the heterogeneous part.
3. **Distillation**: the original is the teacher, our file the student. Each block's scale gets a learned
   multiplier, and the norm weights are adjusted, so the student's next-word probabilities match the teacher's; the
   whole numbers stay frozen, so size and speed are unchanged. On the 1.5B's 2-bit preset: drift −29 %, agreement
   with the original's top word 64.5 % → 71.9 %.

## 4. Measuring quality

| Measure | Meaning | Better |
|---|---|---|
| **Accuracy** | % of questions answered right: MMLU (knowledge), ARC (science), HellaSwag (common sense), WinoGrande (pronouns), GSM8K (math word problems, answered in writing) | higher |
| **Drift (KL)** | how differently our model predicts the next word than the original; 0 = identical | lower |
| **Flips** | % of questions whose right/wrong differs from the original's | lower |
| **Perplexity** | how surprised the model is by real text | lower |

The questions are split once and never mixed: **calibration** (importance matrix, rounding, distillation),
**selection** (the search scores candidates on it), **held-out test** (used once, at the end, for the reported
result). Accuracy on a few hundred questions is noisy and rewards lucky candidates; flips and drift measure the damage
directly ("Accuracy is Not All You Need", NeurIPS 2024). So the search's **score** is **flips + 10 × drift on the
original's own answers** (lower is better): the original writes its answers once, math solutions included, and every
candidate's drift is measured on exactly that text.

## 5. The steps of one contest

1. **fetch** (container with internet): the original model, at a pinned version.
2. **store** (GPU): convert to a 16-bit **GGUF** (llama.cpp's file format), compute the importance matrix, build the
   14 **type copies** (the whole model in each type, every matrix with smarter rounding), delete the download. A
   candidate is later assembled in seconds by picking each matrix from the right copy. Contests of the same model
   share one store.
3. **error**: for every matrix and type, a cheap estimate of the damage (for repairs and the solver).
4. **refs**: the original writes its GSM8K answers; its next-word probabilities on them and on the selection text,
   and which questions it gets right, are recorded.
5. **preset**: llama.cpp's presets near the budget are built and tested on the held-out questions: of the two largest
   that fit, the better one is the **competitor** (if none fits, the smallest, with more RAM than we get).
6. **ga**: the genetic search (below); keeps the top 3.
7. **final**: distill the top 3, score them on all 4,500 selection questions (lowest score wins), **guard** (not less
   accurate than the preset), then winner vs preset vs original on the **held-out test**: **WIN / TIE / LOSE**.
8. **bundle**, **upload**: the winning file to a private Hugging Face repo.
9. **The Pi** downloads it, measures real peak RAM and speed, and posts them to the page. Only our winners run on
   the Pi.

## 6. In depth: every number and formula

### The RAM model (does a genome fit?)

**RAM = norms and biases (32-bit, never quantized) + Σ every matrix's bytes in its type + KV cache + overhead**

- Bytes per type are exact (Q4_K: 256 weights in 144 bytes = 4.5 bits).
- KV cache = 2 × layers × KV width × 4,096 tokens × 34/32 bytes (8-bit cache); overhead = 0.40 GiB.
- A genome is valid only if **0.99 × B ≤ RAM ≤ 1.005 × B**: it fills the budget and never exceeds it by more than 0.5 %.

### Smarter rounding (in every type copy, `src/panther/rounding.py`)

For each matrix **W** (rows × input columns):

1. **H** = average of **x xᵀ** over 65,536 tokens of calibration text, x being what the original feeds into that
   matrix (how its inputs move together). q/k/v share one H; gate/up share one.
2. Damping: + 1 % of H's average diagonal; inputs that are always 0 get weight 0.
3. **U** = Cholesky factor of H⁻¹ (how an error in one column can be offset by later ones).
4. 256 columns at a time (one llama.cpp block): **Q** = llama.cpp's own quantizer on those columns; error
   **E = (W_chunk − Q) · U_cc⁻¹**; then **W_rest −= E · U_c,rest**: the error is pushed onto the columns not yet
   quantized, so the outputs stay right.
5. The codes overwrite the type copy in place: same type, size, speed.

### Offline error (a cheap guess, never the score)

**error(m, t) = Σ over columns importance_c × Σ over rows (w − ŵ)²**, ŵ being matrix m in type t (as rounded) and
importance_c the mean squared input of column c. Used only to repair genomes and as the solver's features.

### The score (fitness), lower is better

**score = flips % + 10 × drift**

- **Flips %**: on the search's 900 selection questions (MMLU 300, ARC 200, HellaSwag 200, WinoGrande 200; the same
  questions every time), the share whose right/wrong differs from the original's, averaged over the 4 tasks.
  6 % flips → 6.
- **Drift**: average KL divergence per token between the genome's and the original's next-word probabilities, on
  the original's own answers: its 200 GSM8K solutions (5 worked examples in the prompt, greedy, up to 256 tokens)
  interleaved with multiple-choice questions and their answers, 16 chunks of 2,048 tokens. 0.15 → 1.5.
- Example total: **7.5**. Accuracy is recorded, never optimized directly.

### The genetic search

- **Genome**: one type for every matrix (7 per layer + word table + output layer), any of the 14 types (the word
  table excepting the 4 that need importance data).
- **Repair** (keeps every genome inside the RAM window): too big → among all "one matrix one step down" moves, rank
  by offline error added per byte saved, pick at random among the best 8, repeat until it fits; too small → the same
  upwards (error removed per byte added, never past the limit). Random among the best 8 keeps the population diverse.
- **Start population (16)**: the preset's layout (each matrix its preset type, or the nearest), every uniform type,
  random mixes (a random type, each matrix shifted 0–2 steps), mutants of the first; repaired, duplicates removed.
- **Mutation**: 50 %: 1–3 random matrices ±1 step; 30 %: 1–3 matrices ±2 or ±3 steps; 20 %: a whole **group** (one
  matrix role in one quarter of the depth, e.g. "ffn_down, layers 0–6") ±1 step. Then repair.
- **Crossover**: per layer, all of its matrices from parent A or B (50/50); word table + output layer count as one
  more layer. Then repair.
- **A child**: 30 %: a mutant of one of the 4 best (**elitism**); otherwise two parents, each the better of 2 random
  population members (**tournament**), then crossover (50 %) or crossover + mutation (50 %). A genome already scored
  is never scored again.
- **Solver proposals** (from 8 scored genomes on, whenever the queue is empty): fit **score ≈ Σ_g w_g × x_g + c** by
  least squares over every scored genome, x_g = the summed offline error of group g (about 30 groups: role × depth
  quarter), which learns what each group's damage really costs in score. CP-SAT then finds the 4 genomes inside the
  window that minimize **Σ w_g × error(m, t)**, each differing from the others in at least 3 matrices; those 4 and 2
  mutants of them are scored next.
- **Steady state**: up to min(16, 80 GiB ÷ (Q4_K copy + 4 GiB)) genomes scored at once (1.5B: 16, 7B: 9, 14B: 6),
  as the GPU allows next to the other contests. Each is built from the type copies in seconds, books GPU memory
  (file × 1.1 + 4 GiB), is scored, then deleted. The best 16 are kept.
- **Stop**: after at least 30 genomes, once 15 in a row improve the best by less than 0.2; at most 70. The 3 lowest
  scores go to the final step.

### Distillation (in the final step, `src/panther/distill.py`)

- **Teacher**: the 16-bit original. **Student**: our file, loaded into PyTorch with every weight at its quantized value.
- **Data**: the original's answers + the calibration text, in sequences of 512 tokens (up to 256). For every position
  the teacher's **64 most likely next words** and their probabilities are stored.
- **Trained**: for every matrix, one number **θ per block of 256 weights of a row**, effective weights
  **W × exp(θ)** (θ starts at 0, so the student starts as the file; exp keeps scales positive); plus every **norm
  weight**. The whole numbers never change.
- **Loss**: at every position, KL(teacher ‖ student) over the teacher's top 64 (teacher probabilities renormalized
  over them; the student's from its full output). Adam, 300 steps of 4 sequences; learning rate 0.002 for θ, 0.0001
  for norms; activations recomputed instead of stored, to save memory.
- **Written back**: every K and IQ type stores one 16-bit scale **d** per block and a block's values are proportional
  to it, so **d ← d × exp(θ)** at the block's byte position (Q2_K, Q4_K, Q5_K also scale their minimum, dmin; Q8_0's
  8 small blocks per 256 weights all take the factor). Norms are overwritten.
- **Not tuned**: IQ1_M blocks (no single scale); Mistral's attention q/k matrices (llama.cpp reorders their rows);
  InternLM cannot be loaded from GGUF by the transformers library, so its finalists are scored as built (the page
  says so).

### The final step and the verdict

1. Each of the top 3 distilled, then scored with the same formula on all 4,502 selection questions; lowest wins.
2. **Guard**: the winner vs the real preset file on those questions (paired bootstrap, below); if the whole 95 %
   interval is below 0, the winner is rejected and the contest ends without a result.
3. **Held-out test**, never seen before: MMLU 1,000, HellaSwag 2,000, ARC 1,172, WinoGrande 1,267, and GSM8K (written
   answers, 5 worked examples, the final number must match exactly). **Accuracy = mean of the 5 task accuracies.**
   Also drift on the selection text, perplexity on web text, flips against the original.
4. **Paired bootstrap**: 10,000 random resamples of the questions within each task, the same questions for both
   files; ours − preset each time; the middle 95 % is the **95 % interval**. **WIN** if all of it is above 0,
   **LOSE** if all below, **TIE** otherwise.

### The competitor (preset step)

llama.cpp's own mixed presets (IQ1_S … Q8_0) from the same original and importance matrix; real RAM = the file's
actual bytes + KV cache + overhead. Of the **two largest that fit** (≤ 1.005 × B), the one with the **higher held-out
accuracy** competes (choosing on the held-out test can only favor the competitor, never us). None fits → the
smallest competes, with more RAM than we are allowed.

### The Pi

llama-server with the full 4,096-token context and 8-bit KV cache, 4 threads; one 256-token prompt and 64 generated
tokens. **Reading speed** = prompt tokens per second, **writing speed** = generated tokens per second, **peak RAM** =
the process's maximum resident memory (VmHWM), and the Pi's throttle state before and after.

## 7. Running it: one command

```bash
./panther start 1.5b@1 7b@3 14b@5   # start contests (model size @ RAM budget, GiB)
./panther pause                     # pause everything; ./panther resume continues
./panther stop 7b@3                 # stop one (./panther stop = all); start it again to continue
./panther status                    # what runs, GPU, disk
./panther test                      # the tests, on your machine
```

- From your machine it syncs the code to the server (anton) and runs the same command there; every step runs in a
  locked-down container ([deploy/server](deploy/server/README.md) has the one-time setup). Every step skips work
  already done, so nothing is lost by stopping.
- **Packed**: all contests start at once. Only downloading and building type copies take turns (one model at a time:
  they fill the disk); everything else runs in parallel, sharing the GPU through a booking ledger (92 of its 96 GB).
  When a contest ends, its type copies are deleted.
- **Safety**: GitLab, which shares the server, is checked every minute; if it becomes unhealthy everything pauses
  until it recovers.
- **The live page** updates every 30 s: each contest's state and current step, what every step does, recent events,
  time spent and left per step, verdicts, the Pi's results.
- On the Pi ([pi](pi/README.md)): `nohup python3 pi/panther_pi.py <hf-user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &`

## 8. The code

About 1,430 lines (blank and comment lines excluded):

| Module | What |
|---|---|
| `panther` | the one command |
| [scripts/server](scripts/server/README.md) | `pipeline.py` (every step, the search), `status.py` (the page) |
| [src/panther](src/panther/README.md) | `rounding.py`, `distill.py`, `llama.py` (runs llama.cpp, checks its output), `model.py` (RAM model, types), `stats.py`, `solver.py` |
| [deploy/server](deploy/server/README.md) | container image, server setup and removal |
| [pi](pi/README.md) | the Pi's check |
| [configs](configs/README.md) | models (pinned versions), budgets, types |

## 9. Related work

- Choosing a type per layer by search: **EvoPress** (ICML 2025, evolutionary search with KL fitness); in GGUF,
  llama.cpp's `--target-bpw`, Unsloth Dynamic quants. Their gains are modest, as ours were with type choice alone.
- Better rounding: **GPTQ**; **GSQ** (2026, learned rounding written back into GGUF Q2_K); **YAQA** (2025, rounding
  against the whole model's KL); **GuidedQuant** (ICML 2025). The strongest 2-bit methods (QTIP, PV-Tuning, AQLM) need
  their own formats.
- Distillation of quantized models: **EfficientQAT** (training only the quantization scales end to end).
- Measuring: **Accuracy is Not All You Need** (NeurIPS 2024): flips and KL over accuracy.
- What Panther adds: allocation search over better-rounded type copies, K and IQ types, distillation into stock GGUF,
  RAM budgets that include the KV cache, verified on a real Pi, and a search score that does not reward luck.
