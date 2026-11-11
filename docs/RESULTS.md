# Results (updated 2026-11-12, 03:45 IST)

How the numbers are made: [STANDARD.md](STANDARD.md). Quality is measured on the server's GPU; the Pi measures only
RAM and speed. Accuracy is the mean of five test sets (MMLU 1,000, HellaSwag 2,000, ARC-Challenge 1,172,
WinoGrande 1,267, GSM8K 300). KL is the mean KL divergence from the model's own 16-bit original (0 = identical).
"Tied" means the paired bootstrap (10,000 resamples of the same questions) cannot separate two accuracies.

## 7 GiB (RAM between 6.93 and 7.035 GiB): 14B wins

| Model @ 7 GiB | How | Bits/weight | Accuracy | KL | Perplexity | MMLU | HellaSwag | ARC-C | WinoGrande | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|
| **14B, our winner (sol2)** | searched, one type per matrix | ~3.6 | **74.88** | **0.076** | **7.17** | 76.9 | 81.3 | 59.2 | 73.6 | 83.3 |
| 14B, llama.cpp **Q3_K_S** preset (best existing that fits) | llama.cpp's own mix, same importance matrix | ~3.5 | 74.16 | 0.103 | 7.42 | | | | | |
| 14B, IQ3_S everywhere, padded to fill | uniform baseline | ~3.6 | 75.32 | 0.143 | 7.73 | 78.2 | 82.5 | 60.1 | 71.2 | 84.7 |
| 7B, Q6_K | standard download (reference) | ~6.6 | 70.23 | — | 8.15 | 73.8 | 80.1 | 53.2 | 71.1 | 73.0 |
| 32B, our winner (sol2) | searched | ~1.6 | 42.51 | 0.962 | 16.86 | 50.0 | 63.5 | 36.9 | 59.5 | 2.7 |
| 32B, IQ1_S everywhere, padded | uniform baseline | ~1.6 | 41.07 | 1.270 | 22.45 | 52.3 | 62.4 | 34.4 | 55.6 | 0.7 |
| 32B, any llama.cpp preset | — | — | **none fits**: the smallest (IQ1_S) needs 7.70 GiB | | | | | | | |

Comparisons (paired bootstrap, 95% interval of the accuracy difference):

| Comparison | Accuracy difference | Verdict |
|---|---|---|
| 14B ours vs 7B Q6_K | **+4.64** [+3.24, +6.07] | 14B better |
| 14B ours vs 14B llama.cpp Q3_K_S | +0.72 [−0.33, +1.80] | tied on accuracy; ours has 26% lower KL and 3.5% lower perplexity |
| 14B ours vs 14B uniform IQ3_S | −0.44 [−1.48, +0.61] | tied on accuracy; ours has 47% lower KL and 7% lower perplexity |
| 32B ours vs 32B uniform IQ1_S | **+1.44** [+0.21, +2.64] | ours better; 25% lower perplexity |
| 14B ours vs 32B ours | +32.4 [+30.8, +33.9] | 14B far better |

**What this shows**

- At 7 GiB the best model is a 14B with a searched mix of types (mostly IQ3_S, 4–5 bits in layers 22–43, 2 bits in
  layers 0–8, output layer Q5_K). It beats the best 7B download by 4.6 points.
- Against the best existing quantization that fits (llama.cpp's Q3_K_S, same importance matrix) it is equal on
  accuracy and clearly closer to the original model (KL 0.076 vs 0.103).
- For 32B no existing llama.cpp quantization fits 7 GiB at all; ours does, and beats a uniform IQ1_S. But at ~1.6
  bits per weight 32B has lost most of its ability: GSM8K 2.7% (the 16-bit 32B scores over 90), ARC-C and
  WinoGrande barely above guessing (25 and 50), KL about 12× that of 14B. A bigger model wins only while it keeps
  more than about 2–3 bits per weight.
- Accuracy alone cannot separate genotypes this close (a few hundred to 2,000 questions per test); KL, measured over
  far more tokens, can. The winner is the most accurate genotype unless the bootstrap ties it with others, in which
  case the lowest KL wins (DECISIONS: "Winner").

## On the Pi 5 (8 GB): RAM and speed

256-token prompt, 64 tokens written, 4 threads, 4,096-token context, 8-bit KV cache, 1 warm-up + 2 timed runs.

| Budget | Model | Peak RAM (predicted) | Writing tok/s | Prompt tok/s | Clock | Notes |
|---|---|---|---|---|---|---|
| 7 GiB | 14B ours | **6.70 GiB** (7.04) | **1.00** | 1.46 | 2.4 GHz, heat-limited | good supply. Capped at 1.8 GHz on the weak supply: 0.76 / 1.09 |
| 7 GiB | 32B ours | **6.78 GiB** (7.04) | 0.49 | 0.61 | 2.4 GHz, heat-limited (83–84 °C) | supply 5.02–5.04 V, no under-voltage |
| 7 GiB | 7B Q6_K | **6.07 GiB** (6.34) | 1.75 | 7.83 | 2.4 GHz, heat-limited | good supply; same as on the weak one (1.74 / 7.98): memory-bound |
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

## Other budgets (running)

| Budget | Searched | Reference | Status |
|---|---|---|---|
| 1 GiB | 1.5B | — | part costs (restarted after the booking-file fault, below) |
| 3 GiB | 7B (~2.8 bits/weight) | 3B Q6_K | 7B: part costs (restarted); 3B Pi-checked above |
| 5 GiB | 14B (~2.4), 7B (~5.1), InternLM2.5 20B (~1.8) | — | queued after the 3 GiB 7B |

Each also gets the best-existing-preset baseline.

## ~20B models (running / queued)

| Model | 3 GiB | 5 GiB | 7 GiB |
|---|---|---|---|
| InternLM2.5 20B (19.9B weights) | does not fit (needs ≥ 4.6 GiB) | ~1.8 bits/weight, queued | ~2.7 bits/weight, **running** (store) |
| Mistral Small 24B (23.6B) | does not fit (≥ 5.2 GiB) | does not fit | ~2.3 bits/weight, after InternLM (disk) |
| InternLM2.5 7B (smaller, same family) | planned | | |

Reka Flash 3.1 (21B) was dropped: its feed-forward width (19,648) is not a multiple of 256, which every IQ and K type
needs.

## Faults found tonight

- **GPU booking file emptied** by a job force-stopped mid-write (01:54): ~230 part-cost runs in the 1 and 3 GiB lanes
  failed with "Expecting value". Fixed: atomic writes, an unreadable file counts as empty; both lanes restarted
  (finished measurements kept).
- **Infinite part cost** when the start type itself was re-measured (two points at one x). Fixed and tested.
- **InternLM conversion** needs protobuf 7.36 for its tokenizer; the image has 4.25. A newer protobuf sits in
  `data/pydeps-convert`, used by the converter only.
