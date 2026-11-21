# Fitness (`panther.fitness`)

Part 3 of the pipeline. One number per file that says how much quantizing changed it, lower is better. It is the
only fitness in the pipeline: choosing the preset (part 2), ranking candidates in the search (part 4) and picking the
winner (part 7) all call it, on the **GA set**. The verdict (part 7) uses the same formula on the **verdict set**.

## The score

**score = flips % + 10 × KL**

- **Flips %**: the share of benchmark questions whose right/wrong differs from the original's, per task, then the
  mean over the tasks (MMLU, HellaSwag, ARC-Challenge, WinoGrande): each task counts equally. A flip counts both ways
  (the original right and the file wrong, or the reverse): either way the file behaves differently from the
  original.
- **KL**: how far the file's next-word probabilities drift from the original's, the mean over the text's scored
  tokens.
- **10**: the weight of KL. With typical numbers (flips 5–20 %, KL 0.05–0.5) flips dominate and KL breaks near-ties.
  The verdict also reports the result with the weight at 5 and 20.
- **Relative score** (`ours.relative(preset)`) = our score ÷ the preset's × 100, both on the same questions and text:
  100 = the preset's damage, 0 = none, above 100 = worse than the preset. The search ranks by the plain score
  (dividing every score by the same number does not change the order).
- **Same top word %**: the share of tokens where the file's most likely next word is the original's. It comes free
  with the KL run and is reported, never scored.

Fitness returns plain numbers. Error bars appear only in the verdict (part 7).

## How it is measured

All by llama.cpp's `llama-perplexity`:

1. **The original's answers, recorded once per model** (in `data/models/<model>/reference/`, each file named by the
   content of the text or question file it was made from, so new data gets new records and old ones are never reused):
   - its next-word probabilities on the text: `--kl-divergence-base` (a `.kld` file);
   - its right/wrong on each task's questions (a `.json` file per question file).
2. **For any file**:
   - one KL run against the original's probabilities: `--kl-divergence`, read in chunks of 2,048 tokens. It prints
     the mean KL and the same top word %.
   - one run per task: `--multiple-choice` (MMLU, HellaSwag, ARC; a binary question file) or `--winogrande` (a CSV),
     with a 4,096-token context and 5 sequences at once. Every question in the file is run, in file order.
     Multiple choice prints the running accuracy after each question (8 decimals): each question's right/wrong is
     the change in the running count, which is exact. WinoGrande prints the option chosen and the correct one on each
     question's line.
   - flips = where its right/wrong differs from the original's; then the score.

**KL covers half the text.** llama-perplexity scores only the second half of each 2,048-token chunk; the first half
is the context. So each scored token has at least 1,024 tokens before it, and the text needs twice the tokens one
wants scored.

**Disk.** The original's probabilities for every scored token: about 15 GB per 100k tokens of text for the Qwen
models (a 152k-word vocabulary), 13 GB for Mistral Small 24B, 3 GB for Mistral 22B. They are kept with the model and
can be deleted with it.

## What it needs from the data part (on hold)

Files, so the data can be decided separately:

- the **GA set's text**: one text file (chat answers and web text);
- the **GA set's questions**, per task, in llama.cpp's formats: one set of files with all of them (presets, the
  winner) and one with the search's 900 (MMLU 300; HellaSwag, ARC, WinoGrande 200 each). The 900 are their own files:
  told to run fewer than all questions of a file, llama-perplexity picks them itself, and the results could not be
  matched to questions.
- later, the **verdict set's** text and questions, for part 7.

## Decisions

| | Decision | Why | Before (Rachit's master) |
|---|---|---|---|
| A | score = flips % + 10 × KL, flips averaged per task | | same |
| B | Two question sets: the search's 900, and all of the GA set's questions for presets and the winner | each search candidate stays fast; the choices that matter use more questions (less noise). One `Fitness` class, made with either set | same |
| C | No "first look" | every candidate gets the full measurement; no hand-set margin | a candidate was scored on a third of the questions first and dropped if it looked clearly worse |
| D | Every question in a file is run; the 900 are their own files | every result maps to known questions | `--multiple-choice-tasks 300` on the full files |
| E | No error bars in fitness | the search compares plain scores; error bars only in the verdict | a standard error decided the search's improvements |
| F | No cache: `score(file)` measures and returns | only the original's records are kept; callers keep what they need (the preset's scores in `preset.json`, the search its own record) | results cached by label and inputs |
| G | KL scores the second half of each chunk, as llama-perplexity does | an existing tool, documented | same, undocumented |
| | One fitness for the whole pipeline, here; presets call it | one definition | presets scored by accuracy |

## Using it

```python
from panther.fitness import Fitness

fitness = Fitness(m, ga_text, ga_questions)      # ga_questions: {"mmlu": Path, ..., "winogrande": Path}
s = fitness.score(file)                          # s.score, s.flips, s.kl, s.same_top, s.flips_per_task
s.relative(preset_score)                         # 100 = the preset's damage
```

## Files

| File | What |
|---|---|
| `fitness.py` | `Score` (the numbers, `relative`), `Fitness` (`score`, `kl`, `answers`, the original's records), `perplexity` (runs llama-perplexity) |

## Tests (`tests/fitness/`)

- **Arithmetic and parsing** (no llama.cpp): the relative score; each question's right/wrong from the running
  accuracy; WinoGrande's chosen against correct; answers for fewer questions than the file holds fail; KL and same
  top word read from a real llama-perplexity log.
- **Real llama-perplexity on part 1's tiny model** (it has a byte-level vocabulary, so any text tokenizes), with
  small question files in llama.cpp's formats: the original scores 0; a quantized file drifts, its score adds up, and
  Q2_K drifts more than Q8_0; every question is answered; the original's records are made once and reused; another
  text or question file gets its own record.
