# scripts/server

The pipeline, run inside the server's containers (see [deploy/server](../../deploy/server/README.md)). Every step is
`pipeline.py STEP BUDGET MODEL`, skips work already on disk, and writes one progress line per event to
`data/logs/progress-b<BUDGET>.log`. `run-budget.sh` runs the steps below in order, one lane per model.

| Step | What it does | Output |
|---|---|---|
| `fetch_model.sh` | Downloads the model at its pinned revision (CPU job: the only network access) | `data/hf/<model>` |
| `store` | Converts to a 16-bit GGUF, computes the importance matrix, quantizes the whole model once to every type the budget can use (`build_tensor_store.sh`), checks every copy (`store_manifest.py`) | `data/store/<model>/` |
| `error` | Offline error of every matrix at every type, weighted by the importance matrix | `rewrite/<model>/error.json` |
| `refs` | The original's next-token distributions on three texts: sensitivity, selection, and benchmark questions with their answers (selection split) | `data/reference/<model>/` |
| `preset` | llama.cpp's mixed presets built from the same original and importance matrix; the two largest that fit are tested; the better one is the competitor. If none fits, the smallest preset competes (with more RAM than we have). Run again after the search: our winner vs that preset | `rewrite/budget-B/preset-<model>.json` |
| `ga` | The genetic search (below) | `rewrite/budget-B/list-<model>-anchored.json` (top 3), `ga-<model>.json` (every genome), `ga-<model>-live.json` (best so far) |
| `gascore` | The top 3 on all 4,500 selection questions; the winner (most accurate, ties to the lower KL); the guard against the real preset file; the winner's full test | `rewrite/budget-B/winner-<model>-anchored.json` |
| `bundle`, `upload` | The winner as a GGUF + manifest for the Pi, to a private Hugging Face repo | `data/pi_bundle/`, `<user>/panther-<size>-<B>g-anchored` |

## The genetic search (`ga`)

- **Genome:** one type per matrix, inside the RAM window (−1 % / +0.5 % of the budget).
- **Fitness:** accuracy on 900 fixed selection questions (MMLU 300, ARC 200, HellaSwag 200, WinoGrande 200) minus
  10 × KL from the original on benchmark-question text. One model file per genome, built from the type copies.
- **Start:** the best preset's layout (its types read from the preset file, mapped to the nearest type available),
  every uniform type padded into the window, random mixes, mutants of the preset layout.
- **Children:** mutation (1–3 matrices one step; 1–3 matrices two or three steps; or every matrix of one role in
  one quarter of the depth one step), crossover keeping whole layers, then repair into the window (a raise is paid
  for by lowering the matrices that lose least per byte). Each round also adds 4 genomes the solver proposes from a
  least-squares fit of every score so far to the offline error per matrix role and depth.
- **Steady state:** 4 genomes scored at a time; a new child starts whenever one finishes.
- **Stop:** after at least 30 genomes, when 15 in a row bring no 0.2-point gain; at most 70.

## status.py

Rewrites `data/status.md` every 30 s (every lane, every step: progress, time spent, time left, end time, the GA's best
and the preset to beat) and publishes it to `<user>/panther-status` on Hugging Face every 60 s.

## Refining the fitness (to try next)

Held-out results so far (2026-11-13): 1.5B @ 1 WIN (+3.59), 7B @ 3 TIE (+0.03), 14B @ 7 TIE (+0.55), 24B @ 7 TIE
(-0.11), 7B @ 5 LOSS (-1.13). The GA's winners are equal or ahead on the four multiple-choice benchmarks almost
everywhere and closer to the original (lower KL and perplexity), but behind on GSM8K in every 7B and 24B contest
(7B @ 5: 62.0 vs 68.7; 7B @ 3: 68.3 vs 71.0; 24B @ 7: 57.7 vs 59.3). And the 24B's lead shrank from +3.4 (900 search
questions) to +1.5 (4,500 selection questions) to -0.1 (held-out test): the fitness also rewards luck.

1. **GSM8K in the fitness.** The fitness has no generation task, so the GA cannot see math. Being tried on 7B @ 5:
   [experiments/gsm8k_fitness](../../experiments/gsm8k_fitness/README.md) (100 GSM8K selection questions in the
   fitness, all 500 in the top-3 scoring and the guard).
2. **A floor per task.** Reject a genome that is worse than the preset on any one task (selection questions) by more
   than ~1 point, so the GA cannot buy multiple-choice points with math or any other single skill.
3. **Rotating question samples.** Score the elite genomes again on fresh selection questions each round. 900 fixed
   questions reward a genome that is lucky on exactly those; the drop from search to held-out test is that luck.
4. **Continuous scores instead of right/wrong.** Use the probability the model gives the correct answer: far less
   noise per question, so 900 questions carry the information of a few thousand and real gains separate from luck.
5. **KL on answer tokens.** KL is averaged over whole question texts; weight the answer and reasoning tokens (GSM8K
   solutions above all) so closeness to the original is measured where the benchmarks score.
6. **Speed on the Pi as a secondary objective.** IQ types have no fast path on the Pi's ARM CPU, Q4_K-Q6_K do;
   between two equally accurate genomes, prefer the faster.

Order to try: 1 + 2 together if the experiment closes the math gap, then 3 + 4 (noise is the next limit).
