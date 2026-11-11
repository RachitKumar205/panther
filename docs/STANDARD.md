# How Panther finds the best model for a RAM budget (D49, D50; 2026-11-11, updated 2026-11-12)

Results so far: [RESULTS.md](RESULTS.md).

## The question

Given a RAM budget on the Raspberry Pi 5 (8 GB), which Qwen2.5 model, quantized how, gives the best answers?

A model is a stack of **layers**; each layer has an **attention** part and a **feed-forward (MLP)** part, made of
weight matrices (attention: q, k, v, output; feed-forward: gate, up, down). There is also the **input embedding** and
the **output layer**. Each weight matrix can be stored at a different **quantization type** (fewer bits per weight =
less RAM, more rounding error). A **genotype** is one choice of quantization type for every matrix.

| Model | Layers | Parts (attention + feed-forward per layer, + embedding + output layer) | Weight matrices |
|---|---|---|---|
| 1.5B | 28 | 57 | 197 |
| 7B | 28 | 58 | 198 |
| 14B | 48 | 98 | 338 |
| 32B | 64 | 130 | 450 |

## Rules

- **Only genotypes that fill the budget count:** predicted RAM within **−1% … +0.5%** of the budget
  (7 GiB: 6.93–7.035 GiB). RAM = weights + KV cache (8-bit, 4,096 tokens) + runtime overhead (0.40 GiB until
  the Pi measures it).
- **Quality is measured on the GPU** (CUDA build of llama.cpp, all layers on the GPU): **KL divergence** from the
  original 16-bit model (0 = identical), perplexity, MMLU, HellaSwag, ARC-Challenge, WinoGrande, GSM8K.
- **The CPU only prepares:** quantizing (llama-quantize), offline error, copying tensors into files.
- **The Pi measures only RAM and speed.**
- **The original models are downloaded once:** 16-bit GGUF files in `data/store/<model>/<model>-BF16.gguf` on the
  server; the Hugging Face downloads are deleted after conversion.

## Which quantization types each model can use

From 1.5 bits/weight below to 2.5 above what the budget allows on average:

| Model @ budget | Average bits/weight the budget allows | Quantization types |
|---|---|---|
| 1.5B @ 1 GiB | 3.0 | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K |
| 7B @ 3 GiB | 2.8 | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K |
| 7B @ 5 GiB | 5.1 | IQ4_XS, Q4_K, Q5_K, Q6_K |
| 14B @ 5 GiB | 2.4 | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K |
| 14B @ 7 GiB | 3.6 | IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K |
| 32B @ 7 GiB | 1.6 | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S |

8-bit (Q8_0) and 16-bit do not fit these budgets in any useful amount; the 16-bit model is only the reference
everything is compared against.

## Why the search is narrowed by measurement

The number of genotypes that fill a budget is astronomically large, so they cannot be listed or tested:

| Budget | Model | Genotypes inside the budget window |
|---|---|---|
| 1 GiB | 1.5B | ~10^196 |
| 3 GiB | 7B | ~10^200 |
| 5 GiB | 14B | ~10^319 |
| 5 GiB | 7B | ~10^188 |
| 7 GiB | 14B | ~10^333 |
| 7 GiB | 32B | ~10^187 |

Coarser designs are enumerable (14B @ 7 GiB, one type per matrix kind for all layers, 3 types: 2,155 genotypes) but
cannot give layer 0 more bits than layer 20, which is where the gain over llama.cpp's own recipes came from. Choosing
from the weights alone (no measurement) reaches ~75% of the achievable improvement on 14B, about llama.cpp-recipe
level; measuring which parts matter reaches ~97%.

## The steps (one command per budget: `./panther run <budget>` on the Mac)

Each model runs in its own lane, straight through, without waiting for the other models. Every step takes the budget
as an argument and checks the disk first: work already there is skipped. Several budgets can run at once: steps that
write a model's shared files take a lock per model (the second run waits, then skips), and heavy steps wait while
the data disk has less than 30 GB free.

| # | Step | What happens | Where |
|---|---|---|---|
| 1 | Quantized copies | The 16-bit model quantized once to each type on its list (llama-quantize, importance matrix) | CPU, once per model |
| 2 | Offline error | For every matrix and type: how much quantization changes its numbers, weighted by importance | CPU, once per model |
| 3 | Reference | The 16-bit model's predictions on two fixed texts (24k and ~400k tokens) | GPU, once per model |
| 4 | Part costs | (a) whole model at a starting type: 1 KL test; (b) each part alone raised to a higher type: 1 KL test per part; (c) the solver picks a genotype, then the types it chose for each part (±1 type) are measured; repeated twice | GPU |
| 5 | Solver list | The solver's 6 best genotypes inside the window (each ≥5% different) + 1 baseline (one type everywhere, raised layer by layer to fill the budget). This is the model's whole solution space; stored in `rewrite/budget-<B>/list-<model>.json` | CPU, seconds |
| 6 | Tests | Each of the 7 genotypes built once and given every test: KL, perplexity, MMLU (1,000), HellaSwag (2,000), ARC-C (1,172), WinoGrande (1,267), GSM8K (300); all 7 at once | GPU |
| 7 | Winner | Best mean accuracy. If the paired bootstrap can't separate another genotype from it (the 95% interval of the lead reaches below 0.5 points), they are tied, and the tied one with the lowest KL wins: KL is measured over far more text than the benchmarks have questions | seconds |
| 7b | Best existing quantization | llama.cpp's own mixed presets (Q3_K_S, IQ3_S, IQ2_M, …) built from the same 16-bit file and importance matrix; the two largest that fit the budget are tested like the genotypes and the better one is compared with the winner (paired bootstrap). Presets that don't fit are deleted | CPU + GPU |
| 8 | Pi | The server uploads each winner to its private Hugging Face repo (`<user>/panther-<size>-<budget>g`, ~2 min) and the Pi downloads it from there (~16 MB/s, ~7 min; the VPN to the server managed 0.5 MB/s); the Pi loads it once, checks peak RAM against the budget and measures speed (256-token prompt, 64 tokens out, 4 threads) | Pi |
| 9 | Comparison | All models' winners and the reference model against each other (paired bootstrap on accuracy) | seconds |

Part-cost files are made with a **reflink copy** (XFS copy-on-write clone: the copy shares the base file's data,
so it is instant) of the model's starting-type file, plus the one changed part appended. Before a model's sweep, one
such file is compared tensor by tensor with a normally built file; if anything differs, normal building is used.

## What each budget involves

| Budget | Searched models | Reference | KL tests (part costs + final) | Genotypes tested with all benchmarks |
|---|---|---|---|---|
| **7 GiB** | 14B, 32B | 7B at Q6_K | 14B: 450 (done) + 7; 32B: ~600 + 7 | 15 |
| 1 GiB | 1.5B | — | ~260 + 7 | 7 |
| 3 GiB | 7B | 3B at Q6_K | ~270 + 7 | 8 |
| 5 GiB | 14B, 7B | — | 14B ~450 + 7; 7B ~270 + 7 (each budget has its own starting type, so its own sweep) | 14 |

## Timeline (IST, 11 Nov; estimates from measured rates)

**7 GiB** (started 21:05):

| Item | Expected |
|---|---|
| 14B: part costs done, list written (6 at 7.035 GiB, baseline at 6.946 GiB) | done 21:05 |
| 14B: tests of 7 genotypes | ~22:15 |
| 14B: winner → Pi (RAM + speed) | ~22:45 |
| 32B: last quantized copies (IQ3_XXS, IQ3_S) + offline error | ~22:00 |
| 32B: part costs (~600 KL tests; with reflink copies) | ~23:00 (if reflink copies fail their check: ~01:00) |
| 32B: tests of 7 genotypes | ~00:30 |
| 32B: winner → Pi; comparison 14B vs 32B vs 7B | **~01:00** |

**Then 1 GiB (~1 h), 3 GiB (~2.5 h), 5 GiB (~3 h)**, each with `./panther run <budget>`.
