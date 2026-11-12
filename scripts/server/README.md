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
| `preset` | llama.cpp's mixed presets built from the same original and importance matrix; the two largest that fit are tested; the better one is the competitor. Run again after the search: our winner vs that preset | `rewrite/budget-B/preset-<model>.json` |
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
