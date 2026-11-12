# Panther

For each RAM budget on an 8 GB Raspberry Pi 5 (1, 3, 5 and 7 GiB), find a quantization of an open model that answers
**better than the best existing quantization that fits**, by giving every weight matrix its own type
(heterogeneous quantization). Quality is measured on a server GPU; the Pi measures only RAM and speed of our winner.

Live progress: <https://huggingface.co/therachitkumar/panther-status>

## The method

A **genome** is one quantization type per weight matrix (e.g. Mistral Small 24B: 282 matrices, each one of 9 types from
IQ1_S to Q4_K). The **competitor** is llama.cpp's best preset that fits the budget, built from the same original model
and importance matrix.

1. **Start population:** the preset's own layout, every uniform type padded to fill the budget, random mixes, mutants.
2. **Fitness of a genome** (one model file, built once): accuracy on 900 fixed *selection* questions (MMLU 300, ARC 200,
   HellaSwag 200, WinoGrande 200) minus 10 × its KL from the original model on benchmark-question text. The selection
   questions are never the ones the results are reported on.
3. **Search:** steady state, 4 genomes scored at a time. Children are mutations (one step, two or three steps, or a
   whole matrix group), crossovers that keep whole layers, and genomes the solver proposes from a model of every
   score so far (sensitivity per matrix role and depth). Every genome stays within the RAM budget (−1 % / +0.5 %).
   The best are always kept. It stops after 15 genomes in a row without a 0.2-point gain (at least 30, at most 70).
4. **Pick:** the top 3 on all 4,500 selection questions; the most accurate wins (ties by paired bootstrap go to the
   lower KL). **Guard:** if it is worse than the real preset file on those 4,500 questions, it is not our result.
5. **Report:** the winner's full test (test-split MMLU, HellaSwag, ARC, WinoGrande, GSM8K, perplexity, KL) against the
   preset's, by paired bootstrap. Then the winner, and only the winner, goes to the Pi for peak RAM and speed.

## Run

On the server (see [deploy/server](deploy/server/README.md)):

```bash
deploy/server/run-budget.sh 7                                          # every searched model of the 7 GiB budget
deploy/server/run-budget.sh 7 Qwen2.5-14B-Instruct Mistral-24B-Instruct  # only these
```

On the Pi (see [pi](pi/README.md)): `nohup bash pi/await.sh <hf-user> 7:24b 7:14b > ~/panther-pi/await.log &`

Tests: `uv sync && uv run pytest`

## Layout

| Module | What |
|---|---|
| [configs](configs/README.md) | Models (pinned revisions), budgets and quantization types |
| [src/panther](src/panther/README.md) | Model inventory and RAM model, GGUF assembly, llama.cpp evaluation, statistics, solver |
| [scripts/server](scripts/server/README.md) | The pipeline: type copies, presets, the genetic search, scoring, bundles, the status page |
| [deploy/server](deploy/server/README.md) | Container image, job runner, one command per budget, server setup and removal |
| [pi](pi/README.md) | Pi setup and the RAM/speed check of a winner |

## Results

In progress (2026-11-13); the live page has the numbers. On the 900 search questions, the genetic search's best is
ahead of llama.cpp's preset for Mistral Small 24B at 7 GiB (accuracy 75.5 vs 72.1) and Qwen2.5 14B at 7 GiB
(77.7 vs 77.1); the guard and the held-out test decide what is reported.
