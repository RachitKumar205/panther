# Panther

A Raspberry Pi 5 has 8 GB of RAM. For each **RAM budget** (1, 3, 5 and 7 GiB) and several open language models
(Qwen2.5 1.5B, 7B, 14B; Mistral Small 22B and 24B), Panther builds a compressed version that should
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
| **Same top word** | % of words where our model's most likely next word is the original's; reported next to drift, not scored | higher |
| **Perplexity** | how surprised the model is by real text | lower |

The data is split once and never mixed ([the dataset](#the-dataset-decision-6)): text in **train** (importance
matrix, rounding, distillation), **select** (the search and the final ranking) and **test** (used once, at the end);
benchmark questions in **selection** (the search and the final ranking) and **held-out** (used once, for the reported
result). Accuracy on a few hundred questions is noisy and rewards lucky candidates; flips and drift measure the damage
directly ("Accuracy is Not All You Need", NeurIPS 2024). So the search's **score** is **flips on selection questions
+ 10 × drift on the select text** (lower is better), reported relative to the preset: 100 = the preset's damage,
0 = none. The **verdict** is the same relative score on held-out data (flips on the held-out questions, drift on the
test text): **WIN** if its whole 95 % interval is below 100, **LOSE** if all above, **TIE** otherwise. Accuracy,
perplexity and same top word % are reported next to it and decide nothing.

## 5. The steps of one contest

1. **prompts** (container with internet): the dataset's chat prompts (made once, shared by all models), then the
   split check (decision 3); if the check fails, the contest does not start.
2. **fetch** (container with internet): the original model, at a pinned version.
3. **store** (GPU): convert to a 16-bit **GGUF** (llama.cpp's file format), compute the importance matrix, build the
   14 **type copies** (the whole model in each type, every matrix with smarter rounding; each copy's file is written
   directly, its layer matrices quantized only once, by the rounding), delete the download. A
   candidate is later assembled in seconds by picking each matrix from the right copy. Contests of the same model
   share one store.
4. **error**: for every matrix and type, a cheap estimate of the damage (for repairs and the solver).
   Steps 5 and 6 start on the GPU as soon as the 16-bit original and its importance matrix exist, while the type
   copies are still being built on the CPU.
5. **refs**: the original answers every chat prompt (the dataset's chat part); its next-word probabilities on the
   select and test text, and which questions it gets right, are recorded.
6. **preset**: every llama.cpp preset that fits the budget is scored on all selection questions; the most accurate is
   the **competitor** (if none fits, the smallest, with more RAM than we get). Only it takes the held-out test.
7. **ga**: the genetic search (below); keeps the top 3.
8. **final**: the top 3, distilled and undistilled (6 files), scored on all ~4,500 selection questions and the select
   text (lowest score wins); **guard** (a warning if clearly worse than the preset there), then winner vs preset vs
   original on the **held-out test**: **WIN / TIE / LOSE** by the relative score, in every contest.
9. **bundle**, **upload**: the winning file to a public Hugging Face repo.
10. **The Pi** downloads it, measures real peak RAM and speed, and posts them to the page. Only our winners run on
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
  the select text: the original's greedy answers (up to 256 tokens, in its chat template) to the 150 select prompts,
  then the select web text; the first 48 chunks of 2,048 tokens (all of the chat part, then the start of the web
  part). 0.15 → 1.5.
- Example total: **7.5**. Accuracy is recorded, never optimized directly.
- **Same top word %**, from the same drift run, is reported next to drift and is not in the score.
- **Relative score** = score ÷ the preset's score × 100, the preset scored the same way on the same questions and
  text (in the search the 900, in the final step all selection questions): 100 = the preset's damage, 0 = none,
  above 100 = worse. Dividing by one number does not change which genome is best.

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
- **Steady state**: as many genomes at once as both GPU memory and the container's RAM allow (each llama.cpp run
  also holds about 2.7 × its file in RAM; 1.5B: ~11, 7B: 3, 14B: 2),
  as the GPU allows next to the other contests. Each is built from the type copies in seconds, books GPU memory
  (file × 1.1 + 4 GiB), is scored, then deleted. The best 16 are kept.
- **First look** (once those 16 exist): a new genome is first scored on a third of the questions (MMLU 100, the
  others 66) plus the full drift. If that score is already more than 3 above the worst of the 16 (about 2 standard
  errors of the flips on that many questions), it is **dropped** without finishing; it would almost surely not
  enter the 16. Otherwise it is scored on all 900 as usual. Dropped genomes count toward the stop rule.
- **Stop** (early stopping, decision 9): a new best counts as an **improvement** only if it is lower than the
  previous best by more than **1 standard error of the difference**: flips paired over the same 900 questions (per
  task the standard error of the mean difference of the two genomes' flips, so only questions where they disagree
  add to it; averaged over the 4 tasks), combined with 10 × both drift runs' own standard errors (llama.cpp's, over
  the chunks). A smaller gain still becomes the best, but does not count. Stop after 15 genomes in a row without an
  improvement; at least 30, at most 70. The 3 lowest scores go to the final step, where all selection questions
  score them again (the best of many noisy scores looks better than it is).

### Distillation (in the final step, `src/panther/distill.py`)

- **Teacher**: the 16-bit original. **Student**: our file, loaded into PyTorch with every weight at its quantized value.
- **Data**: the train split only: the original's chat answers to the 300 train prompts, then the train web text
  (the calibration text), in sequences of 512 tokens (the first 256). No selection or held-out question or text. For
  every position the teacher's **64 most likely next words** and their probabilities are stored.
- **Trained**: for every matrix, one number **θ per block of 256 weights of a row**, effective weights
  **W × exp(θ)** (θ starts at 0, so the student starts as the file; exp keeps scales positive); plus every **norm
  weight**. The whole numbers never change.
- **Loss**: at every position, KL(teacher ‖ student) over the teacher's top 64 (teacher probabilities renormalized
  over them; the student's from its full output). Adam, 600 steps of 2 sequences; learning rate 0.002 for θ, 0.0001
  for norms; activations recomputed instead of stored, to save memory.
- **Written back**: every K and IQ type stores one 16-bit scale **d** per block and a block's values are proportional
  to it, so **d ← d × exp(θ)** at the block's byte position (Q2_K, Q4_K, Q5_K also scale their minimum, dmin; Q8_0's
  8 small blocks per 256 weights all take the factor). Norms are overwritten.
- **Llama-style models** (Mistral): llama.cpp reorders the rows of the attention q/k matrices when converting, so
  their learned multipliers are put into the file's row order; a check first confirms the reordered weights match
  the file row for row (if not, those two matrices are left as they are).
- **Not tuned**: IQ1_M blocks (no single scale).
- **InternLM2.5 20B was dropped** (2026-11-14) for Mistral Small 22B: its model code, written for an older
  transformers library, does not run with the current one, so it could get neither smarter rounding nor
  distillation. Its two contests (search only, before the pipeline review) ended in TIEs, +0.46 at 5 GiB and +0.89
  at 7 GiB; their logs and results are kept.
- **Qwen2.5 32B was dropped** (2026-11-15): loading its 16-bit weights (65.5 GB) for the rounding needs more RAM
  than a container gets (64 GB). Its contest (32b@7) has no result (its competitor would have been IQ1_S, 54.43 %
  held-out, 7.70 GiB); the model and all its files were removed.

### The final step and the verdict

1. Each of the top 3 distilled; the distilled and the undistilled version of each (6 files) scored with the same
   formula on all 4,502 selection questions and the select text; lowest wins (distillation can hurt files that are
   already near-lossless). Each is also reported relative to the preset's score there, with its same top word %.
   Overlapped: one distillation at a time (it needs most of the GPU), while every file already built (the
   undistilled ones, finished distillations, the preset) is scored beside it.
2. **Guard**: the winner's relative score and its 95 % interval (as in 4) on all selection questions and the select
   text; if the whole interval is above 100, a warning is logged and stored with the result. The held-out test runs
   anyway, so every contest ends in a WIN, TIE or LOSE.
3. **Held-out test**, never seen before: MMLU 1,000, HellaSwag 2,000, ARC 1,172, WinoGrande 1,267, and GSM8K (written
   answers, 5 worked examples, the final number must match exactly). Flips against the original's answers to the same
   questions, the mean over the 5 tasks. Drift and same top word % on the test text (the original's answers to the
   150 test prompts, then the test web text, all of it), perplexity on the same text. Ours, the preset and the
   original are tested side by side.
4. **The verdict**: score = flips % (the 5 held-out tasks) + 10 × drift (test text), for ours and for the preset.
   **Relative score** = 100 × ours ÷ preset's (100 = the preset's damage, lower is better). Its **95 % interval** =
   100 × (ours ± 1.96 SE) ÷ preset's, where SE is the standard error of ours − preset, the same one the search uses:
   flips paired question by question (per task, the SE of the mean difference of the two files' flips; the mean over
   the 5 tasks), combined with the two drift runs' own SEs (× 10). **WIN** if the whole interval is below 100,
   **LOSE** if all above, **TIE** otherwise. At 95 %, about 1 contest in 20 where the two files are truly equal still
   comes out WIN or LOSE by chance.
5. **Reported, deciding nothing**: accuracy (the mean of the 5 task accuracies) and ours − preset in accuracy points,
   perplexity, same top word %. The result file stores `relative` and `relative_ci`, `diff` and `ci` (the same minus
   100: negative is better than the preset), `verdict`, and `accuracy_diff`. Also `accuracy_kept` (ours and the
   preset's held-out accuracy as % of the original's) and `weight_check` (the relative score, its interval and the
   verdict again with the KL weight at 5 and at 20, from the same measurements), both on the final log line.

### The competitor (preset step)

llama.cpp's own mixed presets (IQ1_S … Q8_0) from the same original and importance matrix; real RAM = the file's
actual bytes + KV cache + overhead. **Every preset that fits** (≤ 1.005 × B) is scored like a finalist, on all
selection questions and the select text; the **most accurate** there competes (accuracy to 0.1 point; ties: lower
drift). Only it takes the held-out test, which no longer helps choose it. None fits → the smallest competes, with
more RAM than we are allowed. Its scores on the search's 900 questions and on all selection questions are the bases
of the relative score.

### The Pi

llama-server with the full 4,096-token context and 8-bit KV cache, 4 threads, **weight repacking off** (on the Pi's
ARM processor llama.cpp otherwise keeps a second, repacked copy of K-type weights: the 7B @ 5 GiB winner peaked at
6.86 GiB with it, 4.69 GiB without; writing speed is unchanged, reading the prompt ~35 % slower); one 256-token prompt and 64 generated
tokens. **Reading speed** = prompt tokens per second, **writing speed** = generated tokens per second, **peak RAM** =
the process's maximum resident memory (VmHWM), and the Pi's throttle state before and after.

**Writing speed on the Pi** (measured 16 Nov 2026 on the 14B @ 5 QAT file; `pi/panther_pi.py runtime` and `spec`,
`pi/membw.c`). Writing is limited by the CPU decoding weights, not by memory: the Pi reads 12.7–14.1 GB/s,
enough for ~2.9 tokens/s, but every type family tops out near 20–22 billion weights a second (a 14B: ~1.5 tokens/s),
and four busy cores reach 80–85 °C within a minute and throttle (1.48 cold, 1.25 over five answers).
`ik_llama.cpp` (`fcf7b364`) reads prompts 6.6x faster than mainline (9.63 vs 1.47 tokens/s, no repacking) and writes
25 % faster with run-time repacking (`-rtr`, which in turn halves reading). Speculative decoding with a
Qwen2.5-0.5B draft (Q4_K_M, default settings) writes 1.79 tokens/s (+43 %, 68 % of guesses kept) but peaks at 5.33
GiB: the draft adds ~0.6 GiB that batch size and draft context do not change. Within 5 GiB it fits only with a
2,048-token context and a 4-bit KV cache (1.44 tokens/s, +15 %, answers changed). Forcing longer guesses, auto-tuning,
n-gram and suffix lookups give nothing or lose. Any change of arithmetic path (`-rtr`, batching, a draft) changes
some greedy answers by rounding; none was tested on the held-out questions.
The 22B @ 7 QAT file (18 Nov, `therachitkumar/panther-22b-7g-v2-qat`) with `ik_llama.cpp -rtr` writes 0.96 tokens/s
on the 64-token check and 0.81 over five answers (the contest winner on mainline: 0.67), reads 2.97 tokens/s (mainline
0.90) and peaks at 6.60 GiB. Without `-rtr` it reads 5.12 tokens/s but writes only 0.54, so `-rtr` matters more here
than on the 14B (+78 % writing against +25 %).

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
- **Before every contest** a CPU job (`pipeline.py prompts`) makes the dataset's chat prompts, once for all models,
  and runs the split check; if it fails, that contest stops.
- **Built once**: each model's type copies (with its 16-bit original, importance matrix and presets) are saved to
  a public Hugging Face repo, `panther-store-<model>`; a later contest downloads them instead of rebuilding.
- **Packed**: all contests start at once. Only downloading and building type copies take turns (one model at a time:
  they fill the disk); everything else runs in parallel, sharing the GPU through a booking ledger (92 of its 96 GB).
  When a contest ends, its type copies are deleted.
- **Safety**: GitLab, which shares the server, is checked every minute; if it becomes unhealthy everything pauses
  until it recovers.
- **The live page** updates every 30 s: each contest's state and current step, what every step does, recent events,
  time spent and left per step, verdicts, the Pi's results.
- On the Pi ([pi](pi/README.md)): `nohup python3 pi/panther_pi.py <hf-user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &`

## Pipeline review (from 20 Nov 2026)

The pipeline is being rethought stage by stage; each decision is recorded here when made.

**Found:** from 13 Nov (commit 4832b55) the final step distilled each finalist on `answers-<model>.txt`, which holds
the selection GSM8K questions with the original's answers and the selection multiple-choice questions with their
correct answers. The finalists were then ranked, and the guard run, on those same selection questions: a distilled
file was scored on questions it had trained on. The held-out test shares no question or text with any of it
(checked on the server's splits), so every verdict is a true measurement of the file that was tested; which file
was picked, and whether the guard was passed, may have been affected.

| # | Stage | Decision |
|---|---|---|
| 1 | Data | Each pool has one job. Calibration text is what things are built and trained on; selection questions are only scored on (search, picking the winner, guard); held-out questions and text are used once, for the verdict. Nothing trains on selection or held-out data. |
| 2 | Final step | Distillation trains on the original's own answers to general prompts (instructions, chat, everyday maths) from a pool in no other split, mixed with calibration text. The targets stay the original's probabilities; no dataset answers are trained on. Practice is for real use, not the benchmarks' format. |
| 3 | Data | The splits stay as they are (their builder was removed on 11 Nov, 87289e1). A short check, run before a contest, confirms the pools on the server share no question or text and match the hashes in their manifest. |
| 4 | Store | The importance matrix and our error-compensated rounding stay on the calibration web text alone (the preset uses the same importance matrix). |
| 5 | Error | The offline error estimate stays as it is: it reads only the type copies and the importance matrix, and only guides the search (repair, solver proposals); every genome is still measured. |
| 6 | Data | One dataset, split into train, select and test (below). KL and perplexity are measured only on its splits; flips and accuracy only on the benchmark questions. Nothing trains on anything but the train split. |
| 7 | Preset | The competitor is the most accurate on the selection questions of every llama.cpp preset that fits the budget, whatever its size (ties: lower KL on the select text). The held-out questions are no longer used to choose it, only for the verdict. If none fits, the smallest preset, as before. |
| 8 | Search | The score stays flips % + 10 × KL (flips on the 900 search questions, KL on the select text). It is reported relative to the preset, scored the same way: our score ÷ the preset's × 100 (100 = the preset's damage, 0 = none, above 100 = worse). The search itself is unchanged: dividing by one number does not change which genome is best. |
| 9 | Search | Early stopping as in ML training, with the minimum improvement set by the measurement's own noise: a genome counts as an improvement only if it beats the best by more than 1 standard error of the difference (paired over the same questions, from the questions where the two disagree; plus KL's uncertainty over its chunks). Stop after 15 genomes in a row without one (patience); at least 30, at most 70. The best 3 are re-scored on all selection questions in the final step, which corrects for the best of many noisy scores looking better than it is. Replaces the hand-set 0.2. |
| 10 | Search | **The fitness**, used by the search and to pick the winner: **score = flips % + 10 × KL** (as in 8), lower is better, reported relative to the preset. **Flips %** on benchmark selection questions (a flip needs a correct answer, which only benchmarks have): the search uses the same 900 (MMLU 300; ARC, HellaSwag, WinoGrande 200 each) with the first look (a third first; dropped if already ~2 standard errors worse than the 16th best), no GSM8K (measured in the held-out verdict). **KL** on our dataset's select split (chat and web). **Same top word %** (the share of words where the file's most likely next word is the original's), from the same KL run at no cost, is reported next to KL, in the search and the held-out test, and is not in the score. A "kept %" term (how far each answer stays identical) was considered and dropped: KL already measures it (a top word only changes where the probabilities move), it would add noise, and it needed an image rebuild or generation. |
| 11 | Final | The search's best 3 genomes are each built twice, distilled and undistilled (distillation helps most at tight budgets and can hurt near-lossless files, so the measurement decides). The 6 files are scored with the fitness (10) on all ~4,500 selection questions and the select split; the lowest score wins. Clean under 1 and 6: distillation trains only on the train split. |
| 12 | Final | The guard no longer stops a contest. The winner is compared with the preset on all selection questions as before; if ours is clearly worse (the 95 % interval entirely below 0) that is logged as a warning, and the held-out test runs anyway, so every contest ends in a reported WIN, TIE or LOSE. Before, a contest that would lose ended with no verdict, so the record showed only the contests that passed. |
| 13 | Final | **The verdict is the relative score on held-out data**: our fitness ÷ the preset's × 100, with flips on the held-out questions and KL on the test split (100 = the preset's damage, lower is better). Its 95 % interval comes from the same standard error the search uses (9): ±1.96 standard errors of the difference, from the per-question flips (paired) and the two KL runs. WIN if the whole interval is below 100, LOSE if it is all above, TIE otherwise. Accuracy, its difference from the preset, perplexity and same top word % are reported next to it and decide nothing. One rule throughout: the search, picking the winner, the guard and the verdict all use the same relative fitness (the guard on selection data, the verdict on held-out data). Replaces the accuracy difference with a bootstrap interval. At 95 %, about 1 contest in 20 where the two files are truly equal still comes out WIN or LOSE by chance. |
| 14 | Pi | The Pi's result line says FITS or OVER: whether the real peak RAM is within the bundle's budget. Reported on the page next to the verdict; it does not change the verdict. |
| 15 | Final | Reported next to the verdict, deciding nothing: **accuracy kept**, each file's held-out accuracy ÷ the original's × 100, for ours and the preset; and whether the verdict stays the same with the KL weight at 5 and 20 instead of 10 (the weight is a choice, so its effect is shown). The relative score already places the original at 0 and the preset at 100. |

### The dataset (decision 6)

**The rule:** KL and perplexity are measured only on splits of this dataset. Flips and accuracy are measured only on
the benchmark questions. Nothing trains on anything but the train split.

Two kinds of text, each cut three ways with no overlap:

| | Train | Select | Test |
|---|---|---|---|
| **Chat**: OpenAssistant prompts (pinned, Apache-2.0) answered greedily by the original model in its own chat template | 300 prompts | 150 prompts | 150 prompts |
| **Web**: FineWeb-Edu documents (pinned) | 300k tokens | 150k tokens | 150k tokens |

- **The web part already exists:** the former calibration, selection and held-out web text, already disjoint by
  document, under these three names.
- **The chat part is new.** Prompts are split by a fixed seed. A prompt sharing 8 or more consecutive words with any
  benchmark question is dropped before splitting.
- **The answers are per model,** since each model is its own teacher.

**Which split is used at which stage:**

| Stage | Train | Select | Test |
|---|---|---|---|
| store: importance matrix, rounding | web part (decision 4) | | |
| refs: the original's probabilities recorded | | both parts | both parts |
| ga: the score's KL | | both parts | |
| final: distillation | both parts | | |
| final: picking the winner (the score's KL) | | both parts | |
| final: held-out KL and perplexity | | | both parts |

**Benchmark questions** (flips and accuracy only):

| Stage | Selection questions | Held-out questions |
|---|---|---|
| ga: the score's flips | ✓ | |
| final: picking the winner (flips), the guard (accuracy) | ✓ | |
| final: the verdict (accuracy), held-out flips | | ✓ |

The GSM8K few-shot examples stay as part of how a GSM8K question is asked.

**Where it is built and stored:** on the server under `data/splits/text/`.

1. `prompts` (a CPU job with network, before every contest): downloads OpenAssistant at the pinned revision, filters, splits,
   and writes `prompts.json` (each prompt's split, how many were dropped). Made once, shared by all models.
2. `refs` (GPU, no network): the original answers every prompt, written as `chat-<model>-train.txt`, `-select.txt`
   and `-test.txt`; then the original's probabilities on the select and test text (chat and web) are recorded in
   `reference/<model>/select.kld` and `test.kld`.
3. The check (decision 3), run by `prompts`: the three web files and both question splits match their manifests'
   hashes; no web document, question or chat prompt is in two splits; no prompt shares 8 words with a benchmark
   question.

**Archived, not deleted:** `bench.txt`, `answers-<model>.txt`, the sensitivity and agreement text,
`gen_prompts.json`, WikiText, the MMLU few-shot examples.

**Not covered yet:** the QAT experiment trains and validates on its own FineWeb sample, outside these splits; it is
its own decision, still to be made.

## 8. The code

About 1,430 lines (blank and comment lines excluded):

| Module | What |
|---|---|
| `panther` | the one command |
| [scripts/server](scripts/server/README.md) | `pipeline.py` (every step, the search, the chat prompts and the split check), `status.py` (the page) |
| [src/panther](src/panther/README.md) | `rounding.py`, `distill.py`, `llama.py` (runs llama.cpp, checks its output), `model.py` (RAM model, types), `stats.py`, `solver.py`, `fakequant.py` (fake quantizers, for the EfficientQAT experiment) |
| [deploy/server](deploy/server/README.md) | container image, server setup and removal |
| [pi](pi/README.md) | the Pi's check |
| [configs](configs/README.md) | models (pinned versions), budgets, types |

## 9. Related work

- Choosing a type per layer by search: **EvoPress** (ICML 2025, evolutionary search with KL fitness); in GGUF,
  llama.cpp's `--target-bpw`, Unsloth Dynamic quants. Their gains are modest, as ours were with type choice alone.
- Better rounding: **GPTQ**; **GSQ** (2026, learned rounding written back into GGUF Q2_K); **YAQA** (2025, rounding
  against the whole model's KL); **GuidedQuant** (ICML 2025). The strongest 2-bit methods (QTIP, PV-Tuning, AQLM) need
  their own formats.
- Distillation of quantized models: **EfficientQAT** (training only the quantization scales end to end). Its first stage
  (training the codes block by block) is being added as an experiment, with fake quantizers for all 14 of
  llama.cpp's types used here, matching llama.cpp's decoding bit for bit (`src/panther/fakequant.py`). `./panther qat 14b@5` runs, on a finished
  contest's winner rebuilt undistilled: Stage 1 (`src/panther/blockap.py`, the paper's size: 4,096 samples x 2,048
  tokens, 2 epochs); Stage 2 (scales only, end to end, 4,096 x 4,096 tokens, batches of 32, KL to the original instead
  of the paper's cross-entropy); Stage 2 alone on the undistilled file as a control; then each file through the
  final step's tests, against the current winner and the preset (`rewrite/budget-B/qat-<model>.json`, the page).
  The paper's text (RedPajama-1T-Sample) has been taken down, so FineWeb-Edu stands in, without the documents our
  own splits use. On the 1.5B (all blocks, a small trial) Stage 1 took web-text perplexity 19.92 -> 17.38.
  The 14B @ 7 QAT run was stopped at block 17 on 20 Nov and removed: until fix 79d9754 a block's codes were picked
  again in fp32 when saved, while training and its checks picked them in bf16, so a block was saved worse than it
  scored (block 5: scored 0.01606, saved 0.01981), and blocks 0-4 of that run were saved that way. The 14B @ 5 and
  22B @ 7 QAT files were made before the fix: their results are true for the files as saved. Since 20 Nov codes are
  picked in fp32 everywhere (in bf16 the pattern search gives 4-8 % of IQ groups another code), a block's logged
  error is measured from its saved bytes, and a block that does not beat the starting file's keeps the file's.
- Measuring: **Accuracy is Not All You Need** (NeurIPS 2024): flips and KL over accuracy.
- What Panther adds: allocation search over better-rounded type copies, K and IQ types, distillation into stock GGUF,
  RAM budgets that include the KV cache, verified on a real Pi, and a search score that does not reward luck.
