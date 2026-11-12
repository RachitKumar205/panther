# Results (updated 2026-11-12, 19:50 IST)

How the numbers are made: [STANDARD.md](STANDARD.md). Quality is measured on the server's GPU; the Pi measures only
RAM and speed. Accuracy is the mean of five test sets (MMLU 1,000, HellaSwag 2,000, ARC-Challenge 1,172,
WinoGrande 1,267, GSM8K 300). KL is the mean KL divergence from the model's own 16-bit original (0 = identical).
"Tied" means the paired bootstrap (10,000 resamples of the same questions) cannot separate two accuracies.
"Preset" means llama.cpp's own mixed quantization (llama-quantize ftype), built from the same 16-bit file with the
same importance matrix: the best existing quantization that fits the budget.

## Summary: our search against llama.cpp's presets

| Budget | Model | Bits/weight | What won our contest | Our pick vs the best fitting preset | Verdict |
|---|---|---|---|---|---|
| 5 GiB | Qwen2.5 7B | ~5.2 | searched genotype | +1.26 [+0.21, +2.31] vs Q4_K_M (KL 0.0105 vs 0.0139); +0.64 vs Q4_K_S, not bootstrapped | **win** vs the preset the rule picks; likely tie with Q4_K_S |
| 7 GiB | Qwen2.5 14B | ~3.6 | searched genotype | +0.23 [−0.79, +1.24] vs IQ3_S (KL 0.076 vs 0.070) | tie |
| 5 GiB | Qwen2.5 14B | ~2.6 | searched genotype | +1.35 [−0.22, +2.95] vs IQ2_XXS (KL 0.359 vs 0.398) | tie, lower KL |
| 1 GiB | Qwen2.5 1.5B | ~3 | uniform baseline (search lost) | +5.01 [+3.64, +6.44] vs IQ2_S | win, but not by the search |
| 7 GiB | Mistral Small 24B | ~2.3 | uniform baseline (search lost by 12 pts) | +1.01 [−0.46, +2.52] vs IQ2_XXS (KL 0.745 vs 0.665) | tie, not by the search |
| 3 GiB | Qwen2.5 7B | ~2.8 | uniform baseline (search lost) | −2.24 [−3.82, −0.63] vs IQ2_S | **loss** |
| 7 GiB | InternLM2.5 20B | ~2.6 | searched genotype | −5.21 [−6.65, −3.76] vs IQ2_S (KL 0.281 vs 0.159) | **loss** |

**The search matches or beats llama.cpp's hand-made mixes from about 3 bits per weight up, and loses at about 2–2.5
bits.** Why, and the fix now running: [Below ~2.5 bits](#below-25-bits-per-weight-why-the-search-loses).

## 7 GiB (RAM between 6.93 and 7.035 GiB): 14B wins

| Model @ 7 GiB | How | Bits/weight | Accuracy | KL | Perplexity | MMLU | HellaSwag | ARC-C | WinoGrande | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|
| **14B, our winner (sol2)** | searched, one type per matrix | ~3.6 | **74.88** | **0.076** | **7.17** | 76.9 | 81.3 | 59.2 | 73.6 | 83.3 |
| 14B, llama.cpp **IQ3_S** preset (best existing that fits, 7.026 GiB) | llama.cpp's own mix, same importance matrix | ~3.4 | 74.65 | **0.070** | **7.15** | | | | | |
| 14B, llama.cpp Q3_K_S preset (6.995 GiB) | llama.cpp's own mix, same importance matrix | ~3.5 | 74.16 | 0.103 | 7.42 | | | | | |
| 14B, IQ3_S everywhere, padded to fill | uniform baseline | ~3.6 | 75.32 | 0.143 | 7.73 | 78.2 | 82.5 | 60.1 | 71.2 | 84.7 |
| 7B, Q6_K | standard download (reference) | ~6.6 | 70.23 | — | 8.15 | 73.8 | 80.1 | 53.2 | 71.1 | 73.0 |
| 32B, our winner (sol2) | searched | ~1.6 | 42.51 | 0.962 | 16.86 | 50.0 | 63.5 | 36.9 | 59.5 | 2.7 |
| 32B, IQ1_S everywhere, padded | uniform baseline | ~1.6 | 41.07 | 1.270 | 22.45 | 52.3 | 62.4 | 34.4 | 55.6 | 0.7 |
| 32B, any llama.cpp preset | — | — | **none fits**: the smallest (IQ1_S) needs 7.70 GiB | | | | | | | |
| Mistral 24B, our pick (IQ2_XXS everywhere, padded) | uniform baseline | ~2.3 | 69.42 | 0.745 | 14.64 | | | | | 67.7 |
| Mistral 24B, llama.cpp **IQ2_XXS** preset (6.820 GiB) | llama.cpp's own mix | ~2.2 | 68.41 | **0.665** | 13.60 | | | | | |
| Mistral 24B, best searched (sol1) | searched | ~2.3 | 57.38 | 0.987 | 19.15 | 60.8 | 71.3 | 52.1 | 66.7 | 36.0 |
| Mistral 24B, llama.cpp IQ1_M preset (6.080 GiB) | llama.cpp's own mix | ~1.9 | 53.63 | 1.743 | 40.75 | | | | | |
| InternLM2.5 20B, llama.cpp **IQ2_S** preset (6.826 GiB) | llama.cpp's own mix (mostly IQ2_XS, sensitive matrices IQ3_S / Q4_K) | ~2.6 | **69.08** | **0.159** | **9.06** | | | | | |
| InternLM2.5 20B, llama.cpp IQ2_XS preset (6.478 GiB) | llama.cpp's own mix | ~2.4 | 67.89 | 0.188 | 9.32 | | | | | |
| InternLM2.5 20B, our winner (sol6) | searched | ~2.6 | 63.87 | 0.281 | 10.18 | | | | | 57.3 |
| InternLM2.5 20B, IQ2_S everywhere, padded | uniform baseline | ~2.6 | 29.67 | 3.584 | 345.8 | 24.8 | 31.3 | 27.0 | 65.2 | 0.0 |

Comparisons (paired bootstrap, 95% interval of the accuracy difference):

| Comparison | Accuracy difference | Verdict |
|---|---|---|
| 14B ours vs 7B Q6_K | **+4.64** [+3.24, +6.07] | 14B better |
| 14B ours vs 14B llama.cpp IQ3_S | +0.23 [−0.79, +1.24] | **tied**: accuracy, perplexity (ratio 1.002) equal; the preset's KL is 8% lower (0.070 vs 0.076) |
| 14B ours vs 14B llama.cpp Q3_K_S | +0.72 [−0.33, +1.80] | tied on accuracy; ours has 26% lower KL and 3.5% lower perplexity |
| 14B ours vs 14B uniform IQ3_S | −0.44 [−1.48, +0.61] | tied on accuracy; ours has 47% lower KL and 7% lower perplexity |
| 32B ours vs 32B uniform IQ1_S | **+1.44** [+0.21, +2.64] | ours better; 25% lower perplexity |
| 14B ours vs 32B ours | +32.4 [+30.8, +33.9] | 14B far better |
| 14B ours vs Mistral 24B pick | **+5.46** [+3.96, +6.94] | 14B better |
| 14B ours vs InternLM 20B ours | **+11.01** [+9.42, +12.63] | 14B better |
| Mistral 24B pick vs 7B Q6_K | −0.82 [−2.46, +0.82] | tied on accuracy (perplexity not comparable: different tokenizers) |
| InternLM 20B ours vs 7B Q6_K | −6.37 [−8.03, −4.68] | 7B better |
| Mistral 24B pick vs llama.cpp IQ2_XXS | +1.01 [−0.46, +2.52] | tied; the preset's KL is 11% lower |
| InternLM 20B ours vs llama.cpp IQ2_S | **−5.21** [−6.65, −3.76] | **preset better**, with 44% lower KL and 0.2 GiB less RAM |

The cross-model pick across all five (Qwen 14B, 32B, InternLM 20B, Mistral 24B, and the 7B reference) keeps
**Qwen 14B** as the 7 GiB winner.

**What this shows**

- At 7 GiB the best model is a 14B with a searched mix of types (mostly IQ3_S, 4–5 bits in layers 22–43, 2 bits in
  layers 0–8, output layer Q5_K). It beats the best 7B download by 4.6 points.
- Against the best existing quantization that fits, llama.cpp's own **IQ3_S** mix with the same importance matrix,
  it is **equal**: accuracy and perplexity tie, and the preset is slightly closer to the original (KL 0.070 vs
  0.076). Against the other fitting preset (Q3_K_S) ours is equal on accuracy and clearly closer (KL 0.076 vs
  0.103). So at 14B / 7 GiB, where a good hand-made mix exists, the search matches it but does not beat it.
- For 32B no existing llama.cpp quantization fits 7 GiB at all; ours does, and beats a uniform IQ1_S. But at ~1.6
  bits per weight 32B has lost most of its ability: GSM8K 2.7% (the 16-bit 32B scores over 90), ARC-C and
  WinoGrande barely above guessing (25 and 50), KL about 12× that of 14B. A bigger model wins only while it keeps
  more than about 2–3 bits per weight.
- Accuracy alone cannot separate genotypes this close (a few hundred to 2,000 questions per test); KL, measured over
  far more tokens, can. The winner is the most accurate genotype unless the bootstrap ties it with others, in which
  case the lowest KL wins (DECISIONS: "Winner").

## 1 GiB (RAM between 0.99 and 1.005 GiB): 1.5B

| 1.5B @ 1 GiB | RAM GiB | Accuracy | KL | Perplexity | MMLU | HellaSwag | ARC-C | WinoGrande | GSM8K |
|---|---|---|---|---|---|---|---|---|---|
| **Uniform IQ2_S, padded to fill** (the pick) | 0.993 | **44.99** | 0.777 | 21.97 | 49.9 | 61.9 | 38.2 | 61.6 | 13.3 |
| Best searched genotype (sol2) | 1.005 | 39.94 | **0.485** | **16.54** | 44.1 | 55.2 | 35.8 | 57.0 | 7.7 |
| llama.cpp IQ2_S preset (best existing) | 0.978 | 39.98 | 0.506 | 16.95 | | | | | |
| llama.cpp IQ2_XS preset | 0.965 | 37.71 | 0.640 | 19.46 | | | | | |

- Pick vs llama.cpp IQ2_S: **+5.01 points [+3.64, +6.44]**, but 54% higher KL and 30% higher perplexity.
- The searched genotypes beat llama.cpp on what the search minimises (KL 0.485 vs 0.506, perplexity 16.54 vs 16.95)
  with equal accuracy (39.94 vs 39.98).
- **At this compression (1.5B at ~3 bits/weight) the lowest-KL genotype is not the most accurate one**: the uniform
  padded genotype scores 5 points higher on every test with clearly worse KL. At 14B / 7 GiB the same direction
  showed up within noise. The search minimises measured KL, so where KL stops tracking accuracy it optimises the
  wrong thing; the pick came from the padding step (fill the budget exactly with one type), not from the search.

## 3 GiB (RAM between 2.97 and 3.015 GiB): 7B, and llama.cpp's preset wins

| 3 GiB | RAM GiB | Accuracy | KL | Perplexity | MMLU | HellaSwag | ARC-C | WinoGrande | GSM8K |
|---|---|---|---|---|---|---|---|---|---|
| **7B, llama.cpp IQ2_S preset** (best existing) | 2.928 | **64.69** | **0.255** | **10.37** | | | | | |
| 7B, our pick (uniform IQ2_S, padded) | 2.979 | 62.45 | 0.401 | 12.03 | 65.6 | 73.4 | 50.6 | 67.0 | 55.7 |
| 7B, best searched (sol1) | 3.015 | 60.19 | 0.288 | 10.59 | 64.1 | 72.3 | 48.5 | 64.1 | 52.0 |
| 7B, llama.cpp IQ2_XS preset | 2.810 | 61.15 | 0.293 | 10.72 | | | | | |
| 3B, Q6_K (standard download) | 2.833 | 53.21 | — | 9.14 | 65.6 | 73.9 | 46.0 | 69.2 | 11.3 |

- llama.cpp's IQ2_S preset beats our pick by **2.24 points [+0.63, +3.82]** and beats our best searched genotype on
  KL (0.255 vs 0.288), the very quantity the search minimises: at ~2.8 bits/weight the part-cost model (each block's
  KL interpolated from offline error) misses what llama.cpp's hand rules for sensitive matrices get right.
- A squeezed 7B still beats the 3B download by far: our pick +9.25 points [+7.50, +10.92], the preset more.
- Pi: our 7B pick peaks at 2.72 GiB (predicted 2.98), writes 1.86 tok/s, reads 2.44 tok/s; 3B Q6_K writes 4.1–4.5 and
  reads 21–24 (IQ types have no fast ARM path).
- Now running (below): the search started from the preset's own layout.

## 5 GiB (RAM between 4.95 and 5.025 GiB): 7B wins

| 5 GiB | RAM GiB | Accuracy | KL | Perplexity | MMLU | HellaSwag | ARC-C | WinoGrande | GSM8K |
|---|---|---|---|---|---|---|---|---|---|
| **7B, our winner (sol6)** (searched) | 5.025 | 70.51 | **0.0105** | **8.19** | 74.6 | 79.8 | 53.5 | 70.6 | 74.0 |
| 7B, other searched genotypes (sol1–5) | 5.025 | 70.16–70.68 | 0.0105–0.0106 | 8.19 | | | | | |
| 7B, Q4_K everywhere, padded | 4.968 | 70.87 | 0.0324 | 8.36 | 74.2 | 80.0 | 54.1 | 71.3 | 74.7 |
| 7B, llama.cpp Q4_K_S preset | 4.662 | 69.87 | 0.0164 | 8.25 | | | | | |
| 7B, llama.cpp **Q4_K_M** preset (the rule's best existing) | 4.872 | 69.25 | 0.0139 | 8.23 | | | | | |
| 14B, our winner (sol3) (searched) | 5.025 | 65.52 | 0.359 | 9.62 | 63.7 | 75.2 | 50.8 | 69.5 | 68.3 |
| 14B, best searched by accuracy (sol6) | 5.025 | 66.56 | 0.359 | 9.62 | 65.1 | 74.9 | 52.8 | 69.0 | 71.0 |
| 14B, IQ2_XS everywhere, padded | 4.951 | 65.78 | 0.523 | 11.34 | 69.3 | 75.0 | 52.1 | 69.1 | 63.3 |
| 14B, llama.cpp **IQ2_XXS** preset (best existing) | 4.810 | 64.17 | 0.398 | 9.89 | | | | | |
| 14B, llama.cpp IQ1_M preset | 4.399 | 53.22 | 0.701 | 14.25 | | | | | |

- **7B beats 14B at 5 GiB** by 4.99 points [+3.35, +6.58]: 14B at ~2.6 bits per weight has lost more than the extra
  size gives. The crossover from "bigger model" to "more bits" lies between 5 and 7 GiB for these two.
- 7B: ours vs Q4_K_M **+1.26 [+0.21, +2.31]**, with 24% lower KL. The two presets are tied on accuracy, and the rule
  (lowest KL among the tied) picks Q4_K_M; the other, Q4_K_S, scores 69.87, so ours leads it by 0.64, which is likely
  a tie. Against the uniform Q4_K baseline: tied on accuracy, KL a third.
- 14B: ours vs IQ2_XXS +1.35 [−0.22, +2.95], tied, with 10% lower KL and 3% lower perplexity.
- InternLM2.5 20B at 5 GiB (~1.9 bits per weight) is being tested; its result and the 5 GiB pick across all three
  follow.

## Below ~2.5 bits per weight: why the search loses

Four causes, each visible in the numbers above.

1. **The part costs are assumed to add up, and at ~2 bits they don't.** Each block's KL cost is measured by changing
   that block alone; the solver sums the costs. Where errors compound, the sum is wrong. Mistral 24B at 7 GiB: the
   solver predicted its six genotypes would beat uniform IQ2_XXS; built and measured, their KL was ~1.0 against the
   uniform's 0.745, and they scored 12 points lower.
2. **The part costs are measured from a start that is already broken.** The start is one uniform type near the
   budget's lowest average: for InternLM 20B at 5 GiB that is IQ1_M, whose KL is 7.96 (close to garbage). Raising one
   block of a broken model barely moves its KL, so the costs measured there are mostly noise.
3. **KL is what the search minimises, and at ~2 bits KL stops tracking accuracy.** 7B at 3 GiB: the best searched
   genotype has lower KL than the uniform pick (0.288 vs 0.401) and lower accuracy (60.19 vs 62.45). Same at 1 GiB.
4. **llama.cpp's presets encode which few matrices must never be squeezed** (attention value, early feed-forward
   down, output). InternLM 20B shows how much that matters: IQ2_S everywhere collapses (perplexity 346, GSM8K 0),
   while the IQ2_S preset at nearly the same size scores 69.08. The search must find those matrices from its noisy
   costs, and at ~2 bits it doesn't find them all.

### The fix being tested: start from the preset (`PANTHER_ANCHOR=preset`)

The part costs are measured from the best fitting preset's own layout instead of one uniform type, one type up and
one type down per block (then two rounds around the solver's choices, as before), and the preset's layout, padded
into the RAM window, is listed as a candidate itself. The search then works where the preset already is, with
costs measured from a working model, and its list always holds the preset's layout. Results are kept separately
(`-anchored` files); the uniform-start results above stay as they are.

| Contest | Preset it starts from | Start KL (the preset's own: 0.159) | Status |
|---|---|---|---|
| InternLM 20B @ 7 GiB | IQ2_S (mostly IQ2_XS; IQ3_S and Q4_K on sensitive matrices) | 0.160 | part costs, 197 KL runs in round 0, started 19:33 |
| Mistral 24B @ 7 GiB | IQ2_XXS | | queued after it (disk) |
| InternLM 20B @ 5 GiB | its best preset, once tested | | after its uniform-start run |
| Qwen 7B @ 3 GiB | IQ2_S | | needs its type copies rebuilt (deleted for disk) |

## On the Pi 5 (8 GB): RAM and speed

256-token prompt, 64 tokens written, 4 threads, 4,096-token context, 8-bit KV cache, 1 warm-up + 2 timed runs.

| Budget | Model | Peak RAM (predicted) | Writing tok/s | Prompt tok/s | Clock | Notes |
|---|---|---|---|---|---|---|
| 7 GiB | 14B ours | **6.70 GiB** (7.04) | **1.00** | 1.46 | 2.4 GHz, heat-limited | good supply; a repeat run gave 1.01 / 1.46 / 6.69 GiB. Capped at 1.8 GHz on the weak supply: 0.76 / 1.09 |
| 7 GiB | 32B ours | **6.78 GiB** (7.04) | 0.49 | 0.61 | 2.4 GHz, heat-limited (83–84 °C) | supply 5.02–5.04 V; a repeat gave 0.49 / 0.60 / 6.69 GiB |
| 7 GiB | 7B Q6_K | **6.07 GiB** (6.34) | 1.75 | 7.83 | 2.4 GHz, heat-limited | good supply; same as on the weak one (1.74 / 7.98): memory-bound |
| 1 GiB | 1.5B pick (uniform IQ2_S, padded) | **0.70 GiB** (0.99) | **9.40** | 13.9 | 2.4 GHz | good supply; RAM 0.3 GiB under the prediction |
| 3 GiB | 7B pick (uniform IQ2_S, padded) | **2.72 GiB** (2.98) | 1.86 | 2.44 | 2.4 GHz | |
| 3 GiB | 3B Q6_K | **2.78 GiB** (2.83) | **4.11** | 21.1 | 2.4 GHz | good supply; 3.41 / 17.3 on the weak one |

- Every model fits its budget. The RAM prediction is always a little high (0.05–0.28 GiB): the real runtime overhead
  is ~0.13 GiB, not the 0.40 GiB assumed.
- Speed roughly halves as the model doubles, and the types matter: IQ types (most of the 14B and all of the 32B)
  have no fast ARM path in llama.cpp, while Q4_K/Q5_K/Q6_K are repacked into one. 14B has 1.78 GB of its 6.4 GB of
  weights on the fast path, 32B none, 7B Q6_K all of it. A speed-aware search would weigh this.
- All speeds are lower bounds: the Pi throttled in every run (supply and heat).

### Pi power and heat (what went wrong and what fixed it)

- The first supply did not negotiate USB-PD (Pi assumed 0.9 A) and sagged to 4.45–4.7 V under load; the Pi reset
  three times during the 14B run. A 65 W USB-C PD charger (5 V/3 A) still sagged; 14B finished only with the CPU
  capped at 1.8 GHz (`bash pi/lean.sh 1800`).
- On the current supply (since 02:34) the Pi holds 5.02–5.13 V with no under-voltage, so the cap is off. It now
  heat-limits at 83–85 °C under full load: a heatsink on the fan would help.
- `pi/lean.sh` stops the desktop and other extras (frees ~0.4 GiB); `pi/power-log.sh` logs heat, voltage and free
  RAM every 5 s to disk, so a power cut leaves a record.
- **The 06:40 "crash" was the network, not the Pi.** Tailscale lost it at 06:40, but its power log ran on to 06:50
  with no under-voltage (flags 0xe0008: heat only, 84 °C). Its Wi-Fi had dropped, and after the restart it could
  not rejoin: the saved university login was rejected (MSCHAPv2 error 691). The Student profile now has the new
  login (tested: it joined Student at 19:20); the phone hotspot `jia` has priority 20, Student 10, so the Pi uses the
  hotspot when it is on and Student otherwise, at boot as well. The HDMI ports report no monitor at all (no
  hot-plug signal), so there is no screen to fall back on.

Pi checks still to run, biggest first (`pi/await.sh` on the Pi downloads each from Hugging Face as it appears):
Mistral 24B @ 7 (running since 19:28), InternLM 20B @ 7, InternLM 20B @ 5 (when uploaded), Qwen 14B @ 5, Qwen 7B @ 5.

## Status

| Budget | Searched | Reference | Status |
|---|---|---|---|
| 1 GiB | 1.5B | — | done; Pi done |
| 3 GiB | 7B (~2.8 bits/weight) | 3B Q6_K | done; Pi done; anchored rerun pending |
| 5 GiB | 14B (~2.6), 7B (~5.2), InternLM2.5 20B (~1.9) | — | 14B and 7B done (7B wins, uploaded); InternLM 20B testing; pick to rerun with it |
| 7 GiB | 14B, 32B, InternLM2.5 20B, Mistral 24B | 7B Q6_K | done (14B wins); all uploaded; anchored reruns for 20B and 24B running |

| Model | 3 GiB | 5 GiB | 7 GiB |
|---|---|---|---|
| InternLM2.5 20B (19.9B weights) | does not fit (needs ≥ 4.6 GiB) | ~1.9 bits/weight, testing | ~2.6 bits/weight, done; anchored running |
| Mistral Small 24B (23.6B) | does not fit (≥ 5.2 GiB) | does not fit | ~2.3 bits/weight, done; anchored queued |
| InternLM2.5 7B (smaller, same family) | planned | | |

Reka Flash 3.1 (21B) was dropped: its feed-forward width (19,648) is not a multiple of 256, which every IQ and K type
needs.

## Faults found

On 2026-11-12, daytime:

- **Disk short (22 GB free) stalled the 24B bundle and killed the InternLM 5 GiB part costs** (at 25 of 98). Freed
  78 GB (Pi bundles already on Hugging Face, the finished 7B type copies); the lane resumed from its saved
  measurements.
- **The 5 GiB contest config still left InternLM out** ("after its 7 GiB part costs"), so its restart failed at
  once; added back.
- **The 7 GiB pick needs the 7B reference file**, deleted with the 7B type copies; restored from its Hugging Face
  repo (the same Q6_K file the Pi tested).
- **The pick compared perplexity across tokenizers** (crash where chunk counts differed; a silent wrong tiebreak,
  24B vs 7B, where they matched). Perplexity is now compared only within one model family.

Night of 2026-11-11:

- **GPU booking file emptied** by a job force-stopped mid-write (01:54): ~230 part-cost runs in the 1 and 3 GiB lanes
  failed with "Expecting value". Fixed: atomic writes, an unreadable file counts as empty; both lanes restarted
  (finished measurements kept).
- **Infinite part cost** when the start type itself was re-measured (two points at one x). Fixed and tested.
- **InternLM conversion** needs protobuf 7.36 for its tokenizer; the image has 4.25. A newer protobuf sits in
  `data/pydeps-convert`, used by the converter only.
