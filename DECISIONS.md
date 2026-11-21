# Panther rebuild: decisions by phase (21 Nov 2026)

## Overview

We rebuilt Panther on the `jia` branch as one plain package, one module per pipeline phase, each with a README and
tests (207 passing). It keeps Rachit's science where it was sound, fixes the bugs the 21 Nov audit and our reviews
found, and drops machinery that did not earn its keep.

- **How we worked:** one phase at a time; each design discussed and signed off before code; existing tools wherever
  one fits (llama.cpp, transformers, gguf-py, OR-Tools); no caches beyond "a step's output exists → skip"; failures
  loud, never caught.
- **Where:** branch `jia` on GitHub. The root `README.md` lists the phases; `READ-IF-JUST-PULLED` says how to run it
  on anton.
- **Status:** every phase is written and tested on a tiny model with the real llama.cpp tools. Nothing has run on a
  real model yet: Rachit's first `1.5b@1` run on anton is the test.

## Phase 0 — Data

The data is now rebuilt from pinned sources by code (master had lost its split builders), in four sets that never
share a document, prompt or question.

| Set | Used for | Web text | Chat prompts | Questions |
| --- | --- | --- | --- | --- |
| GA | the search's fitness | 100k tokens | 150 | MMLU 300, HellaSwag 200, ARC 200, WinoGrande 200 |
| winner (new) | choosing the preset and the winner | 100k | 150 | 1,000 / 800 / 800 / 800 |
| distill | importance matrix, GPTQ, distillation | 300k | 300 | none |
| verdict | the verdict, once | 150k | 150 | MMLU 2,000, HellaSwag 2,000, ARC all, WinoGrande all, GSM8K 500 |

- **Why a winner set:** the finalists became finalists on the GA set, partly by luck; choosing among them (and among
  presets) on fresh questions removes that.
- **Web text:** FineWeb-Edu, each document assigned to a set by hashing its id, so dropping one never moves the
  others. Lalit's n-gram dedup is in: a document containing 12 consecutive words of any benchmark question is
  dropped (question n-grams built once; stems under 8 words skipped so short stems don't wipe out documents).
- **Chat:** OpenAssistant prompts overlapping benchmarks dropped; each model answers them itself.
- **Verdict subsets** (not the full test sets) keep the verdict fast.
- **Checked at build:** no overlap between sets, every llama.cpp file reads back as written, a manifest with every
  source's revision and hash.

## Phase 1 — The model

Kept Rachit's core (type copies + GPTQ); changed what the copies hold and how they are made.

| Decision | Why | Before |
| --- | --- | --- |
| 13 types: IQ3_S dropped | same size as Q3_K, measured the same with GPTQ; a step between them saved 0 bytes | 14 types |
| GPTQ also on the output layer | it has an input like any layer matrix; its error goes straight into next-word probabilities | plain rounding |
| Importance data for the output layer (`--process-output`) | it can now take all 13 types | not collected (llama-imatrix default) |
| GPTQ records each matrix's output error per type (`errors.json`) | the search's repair steers by it | separate offline estimate, wrong scale for word table/output |
| `llama-quantize --pure` writes each copy, GPTQ overwrites the matrices | llama.cpp writes the file; our code only replaces matrices | Rachit wrote files by hand |
| The download is kept; no Hugging Face backup | never download twice (time); only type copies deleted for disk | download deleted; whole store uploaded publicly |

- **Precision:** the original stays 16-bit (bf16, as published); the maths run in 32-bit because llama.cpp's
  quantizer takes float32 and a 16-bit Cholesky is too imprecise.
- **Disk (500 GB server):** ~13 GB built for the 1.5B, ~190 GB for the 24B; all downloads ~140 GB.

## Phase 2 — Presets

The competitor is now the fitting preset with the lowest fitness on the winner set; every preset is sized exactly
before any is built.

- **Exact sizes from `llama-quantize --dry-run`** (instant, nothing quantized) replace the hand-written bits table
  that could skip a preset that fits (audit bug B15).
- **Chosen by fitness, not accuracy:** the strongest preset by the very measure we compete on. One rule throughout.
- **RAM model:** file size + KV cache (from the GGUF's own metadata, 8-bit, 4,096 tokens) + 0.40 GiB; fits if
  ≤ 1.005 × budget. File size, not tensor bytes: the Pi loads the whole file.
- **No preset fits:** the smallest competes anyway and the result is tagged **unfair**, so very small budgets can
  still be tried.
- Same 22 presets as before; unchosen preset files can be deleted.

## Phase 3 — Fitness

Kept the score, **flips % + 10 × KL** against the original, and made it the only fitness in the pipeline (presets
call it too).

- **No first look:** every candidate gets the full measurement (Rachit's hand-set early-drop margin is gone).
- **No error bars in the search:** plain scores; error bars live only in the verdict.
- **No caches:** `score(file)` measures and returns; only the original's reference answers are recorded, named by the
  content of the data they came from, so new data never reuses old records.
- **Every question in a file is run;** the search's 900 are their own files (llama.cpp's own subset choice could not
  be mapped to questions).
- **KL scores the second half of each 2,048-token chunk** (llama-perplexity's way): accepted and documented.
- **Fast:** a file's five measurements run at the same time, and several files can be scored at once; a shared
  GPU/RAM budget keeps parallel runs from running the machine out of memory (it reads the container's memory limit).

## Phase 4 — Search

A straightforward genetic search over layouts (one type per matrix), kept inside the RAM window (0.99–1.005 ×
budget), made faster and reproducible.

| Decision | Why | Before |
| --- | --- | --- |
| Starts: all 22 preset layouts + 13 one-type layouts, repaired, + 4 CP-SAT layouts | llama.cpp's own recipes are good starting points | competitor layout, uniform, random mixes |
| Repair by measured output error per byte (from `errors.json`) | steps the matrices that matter least | estimate on the wrong scale (audit finding 3) |
| Crossover: two-point slice of whole layers ("gene slicing") | neighbouring layers stay together | coin flip per layer |
| Children: 30 % mutate one of the 4 best, 70 % slice two parents | polish the leaders and explore (Rachit's split) | same |
| Mutation: 1–3 matrices, ±1–2 types | one simple rule | three kinds |
| In-search solver: 9 role weights, ridge fit pulled toward 1; 1 candidate per round | Rachit's ~30-weight unregularised fit was unstable (audit finding 10) | ~30 weights, clipped |
| Rounds of 8 scored at once, seeded | ~8× faster, still the same result for the same seed | parallel but not reproducible |
| Stop: 15 without a new best (plain score), 30–70 candidates | no ratchet (audit finding 9) | 1-SE rule |

- **Dropped:** screening children by summed error (we tried it; it steered the whole search by an estimate), the
  first look.
- **Resumable:** every round recorded; a stopped search continues exactly where it stopped.

## Phase 5 — Distillation

Kept Rachit's method (one multiplier per block of 256 weights plus every norm, trained against the original, written
back into the block scales), with one important fix to the loss.

- **The loss was biased.** The old loss renormalised the teacher's top 64 words but compared them with the student's
  whole vocabulary, which rewards piling probability onto those 64. On the test model, training lowered that loss
  while llama.cpp's KL got *worse* (0.0288 → 0.0329). The new loss is the exact KL over the top 64 plus "everything
  else": KL now falls ~70 % (Q2_K 0.0288 → 0.0079, IQ2_XS 0.0330 → 0.0105, Q4_K 0.0015 → 0.0004).
- **No bit errors:** the teacher is the 16-bit original as published; the student holds the file's exact values in
  32-bit (bf16 only for the 22B/24B, which don't fit on the GPU in 32-bit, during training only); multipliers and
  norm shifts are 32-bit; the write-back changes only scales and norms, byte-checked across 8 types.
- **The output layer is trained too;** the word table is not (a lookup).
- **Mistral's q/k row order** handled both ways, tested against llama.cpp's own formula.
- **What is scored is always the written file,** never the in-memory model.

## Phase 6 — QAT (optional)

Rachit's newer Block-AP (from his `rachit` branch) ported leaner: 590 lines against 660, an optional 7th finalist
that is then distilled.

- **What it does:** block by block, the weights are trained (rounded to llama.cpp's own grids in the forward pass,
  gradient straight through) so each quantized block reproduces the original's output; scales stay fixed,
  distillation tunes them after.
- **Kept, all four of his guarantees:** one 32-bit code picker everywhere; back to the best point exactly; each block
  scored from its saved bytes; never saved worse than the file's block.
- **Simpler:** loads the original from the download (not transformers' GGUF loader); Mistral row order from the
  architecture; plainer disk caches; two resume bugs fixed.
- **Measured on the test model:** QAT alone took Q2_K's KL from 0.0294 to 0.0165 (−44 %), before distillation.
- **Open:** sample count per model. Its disk cache at the paper's 4,096 samples is 172 GB for the 14B; ~1,024 is
  practical on 500 GB. Off by default (`--qat N`).

## Phase 7 — The winner and the verdict

The winner is chosen on the winner set; the verdict is the winner against the preset, once, on the verdict set.

- **Winner:** the 3 finalists, each as built and distilled (+ QAT's file), all scored at once on the winner set;
  lowest score wins; only its file is kept. The old "guard" step is dropped: its number is simply reported.
- **Verdict:** score = flips % + 10 × KL (the fitness's formula, 5 tasks incl. GSM8K via `llama-server`);
  relative = ours ÷ preset's × 100.
- **WIN / TIE / LOSE from a 95 % interval:** relative ± 1.96 standard errors of the *difference* (ours vs the preset,
  same questions and text). WIN if the whole interval is below 100, LOSE if above, TIE if 100 is inside. Example:
  90 ± 6 → WIN; 97 ± 6 → TIE.
- **The error is paired:** flips per question (only questions where the two files differ add uncertainty); KL per
  text chunk, recovered from llama.cpp's running mean (every chunk checked; its rounding can only widen the
  interval).
- **Reported beside it, deciding nothing:** per-task accuracy, accuracy kept vs the original, same top word %, and
  the verdict at KL weights 5 and 20.
- **Fast:** winner and preset measured at the same time; the original's answers recorded once per model.

## Phase 8 — Running it

One entrypoint calls the modules in order; one command runs a contest on anton.

- `python -m panther fetch 1.5b` (network, no GPU): dataset, model download. `python -m panther run 1.5b@1 [--qat N]`
  (GPU, no network): model → preset → search → (QAT) → winner → verdict → one summary line.
- `./panther build | run 1.5b@1 | logs | stop | status` from your machine: rsyncs the code to anton and runs it in
  containers, using Rachit's existing `local.env`.
- **Kept from Rachit's setup:** CUDA 12.9 image with llama.cpp at the pinned commit, torch pinned to the CUDA 12.9
  build (2.13), rootless Podman with no container having both GPU and network, the shared Hugging Face cache (so
  anton's existing downloads are reused).
- **Dropped:** cross-run GPU/RAM claims, the GitLab guard, the dashboard, the upload/publish step, pause/resume.
- **Rerunning continues** where it stopped; the costliest stop is while building the type copies.

## Bugs fixed, and what is still unproven

The biggest fixes were in the loss, the statistics and the cache keys; the biggest open risk is that nothing has run
on a real model.

**Fixed (from the audit and our reviews):**
- Distillation's biased loss (could make files worse).
- KL per chunk: llama.cpp prints running means, and a negative or `-nan` column silently dropped chunks; now
  recovered exactly and every chunk checked.
- Cache keys that missed inputs (B1): caches removed; records named by the content they came from.
- The split check checked other files than the ones measured (B2): the data is rebuilt and checked at build time.
- Repair's error on the wrong scale (finding 3); the solver's unstable fit (10); the stop rule's ratchet (9); the
  non-reproducible search (11); the preset bits table that could skip a fitting preset (B15).
- Memory planning read the whole server's RAM, not the container's limit.

**Unproven until Rachit's first run on anton:** the container build, the real dataset downloads,
`convert_hf_to_gguf.py`, the chat answers in each model's template, and time and memory on the large models. Results
are not directly comparable with the old pipeline's numbers (different sets and verdict); the fair comparison is both
winners through the same verdict.

**Open decisions:** the QAT sample count (~1,024 suggested), the KL weight (10, reported at 5 and 20), `RIDGE = 10`
in the in-search solver.
