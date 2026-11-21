# The winner and the verdict (`panther.verdict`)

Part 7 of the pipeline. Choose our one file, then judge it against the preset once, honestly, on data nothing else
has seen.

## The sets it uses

| Set | Used here for |
|---|---|
| **winner set** | choosing the winner among the finalists (and, in part 2, choosing the preset) |
| **verdict set** | the verdict, once per contest (it is the "held-out test": the same thing) |

Neither is the **GA set** (the search's 900 questions and its text). The finalists became finalists because they did
well on the GA set, partly by luck; choosing among them on the same questions would reward the luckiest. Fresh
questions remove that. The same holds for choosing the best of the 22 presets.

## The winner (`Winner`)

1. Each of the search's 3 finalists is built twice: **as the search built it**, and **distilled** (part 5). With QAT
   (part 6), its file, distilled, joins them: 6 or 7 files.
2. **Every file is scored at once** with the fitness on the **winner set**'s text and questions (each llama.cpp run
   waits for its share of GPU memory and RAM).
3. **The lowest score wins.** Its file becomes `data/contests/<contest>/winner.gguf`; every other file is deleted.
4. Saved once to `winner.json`: which file won, its score, its score relative to the preset's (also on the winner set,
   from part 2's record), and every file's score.

## The verdict (`Verdict`)

1. **The winner and the preset are measured at the same time** on the whole verdict set, each with all its runs at
   once: KL on its text, every MMLU, HellaSwag, ARC and WinoGrande question, and **GSM8K**. The original's answers
   are recorded once per model (like the fitness's).
2. **The score**, as the fitness: **flips % + 10 × KL**, flips against the original's answers, averaged over the 5
   tasks. **Relative score = ours ÷ the preset's × 100** (100 = the preset's damage, below 100 = less damage).
3. **The 95 % interval** and **WIN, TIE or LOSE** (below).
4. **Reported beside it, deciding nothing**: each task's accuracy (ours, the preset's, the original's), accuracy kept
   (ours and the preset's as % of the original's), KL and same top word % of both, the verdict again with the KL
   weight at 5 and at 20 (the 10 is a choice), and whether the preset was unfair (part 2: no preset fit the budget).
5. Saved once to `verdict.json`.

### The 95 % interval, in plain words

Any score has luck in it: on a different set of questions both files would score a little differently. The
**standard error** measures that luck: how much the *difference* between our score and the preset's would wobble if
the verdict were run again on other questions and text like these.

The **interval** is our relative score ± 1.96 standard errors: the true value lies inside it 19 times in 20.

- **WIN**: the whole interval is below 100. Even the pessimistic end beats the preset: better by more than luck.
- **LOSE**: the whole interval is above 100.
- **TIE**: 100 is inside. Luck could explain the difference either way.

Example: relative 90 with a standard error of 3 points → interval 84–96 → WIN. Relative 97, error 3 → 91–103 → TIE.
At 95 %, about 1 contest in 20 where the two files are truly equal still comes out WIN or LOSE by chance.

### Where the standard error comes from

Both files answer the same questions and read the same text, so the error is of the **difference**, paired:

- **Flips**: per question, (our flip − the preset's flip): 0 where both flipped or neither did, so only questions
  where the two files differ add uncertainty. The standard error of each task's mean, combined over the 5 tasks as
  their mean is.
- **KL**: per 2,048-token chunk of the text, (our KL − the preset's KL); the standard error of their mean.
  llama-perplexity prints the running mean KL after each chunk; each chunk's own KL is recovered from it
  (`Fitness.kl_chunks`, which also checks every chunk was read). Its 5 printed decimals add a little noise, which can
  only make the interval wider, never falsely narrow.
- The two together, as independent errors: √(flips² + (10 × KL)²).

### GSM8K

Grade-school maths word problems, answered by writing out the working: 5 worked examples first, then the question;
greedy, at most 256 tokens, stopping at the next "Question:". Right if the number after the first "####" equals the
answer's (commas, `$` and a final dot ignored). It tests step-by-step reasoning, which multiple choice does not, and
reasoning is where heavy quantization often breaks first.

Run by `llama-server` (one per file, 8 questions at once), which holds its share of GPU memory and RAM while it runs.

## Decisions

| | Decision | Why | Before (Rachit's master) |
|---|---|---|---|
| 1 | The winner chosen on a **winner set**, not the GA set | the finalists were chosen on the GA set, partly by luck | all selection questions, which included the GA's |
| | The preset also chosen on the winner set (part 2) | the same reason | the selection questions |
| 2 | No guard step: the winner's relative score on the winner set is reported | the verdict runs anyway | a guard that warned |
| 3 | Score = flips % + 10 × KL, relative to the preset, 95 % interval, WIN/TIE/LOSE | the fitness's formula: one rule throughout | the same |
| 4 | KL's error paired over chunks, from each chunk's KL | the honest statistic; can only widen | llama.cpp's ± (tokens as independent; too small), combined as if the two files were independent (too large) |
| 5 | GSM8K kept, via llama-server, 8 at once | reasoning; the verdict runs once | the same |
| 6 | Everything measured at once: the winner and the preset in parallel, each file's runs in parallel; the original's answers once | fast | the original's answers recorded once; the rest partly in turn |
| 7 | Only the winner's file kept | disk | the same |

## Using it

```python
from panther.verdict import GSM8K, Verdict, Winner

winner = Winner(contest, search.genes, Distill(m, distill_text), Fitness(m, winner_text, winner_questions))
chosen = winner.run(search.run(), preset_record, extra=(qat_file,))       # extra: optional
fitness = Fitness(m, verdict_text, verdict_questions)
verdict = Verdict(fitness, GSM8K(m, gsm8k_questions, gsm8k_examples, fitness.resources))
result = verdict.run(contest, Path(chosen["file"]), preset_record)        # result["verdict"]: WIN / TIE / LOSE
```

## Files

| File | What |
|---|---|
| `winner.py` | `Winner`: build and distill the finalists, score them all at once, keep the lowest |
| `verdict.py` | `Verdict` (`measure`, `run`), `judge` (the relative score, its interval, WIN/TIE/LOSE), `Measured` |
| `gsm8k.py` | `GSM8K` (answers, the original's recorded once), `LlamaServer` (also used by the data part, with `chat` on, to have the original answer the chat prompts in its own template: `--jinja` and `/apply-template`), `correct` |

## Tests (`tests/verdict/`)

- **Statistics**: a file against itself is 100 and a TIE with no uncertainty; clearly better wins, clearly worse
  loses; a small difference ties; the score is flips + weight × KL and the interval is exactly ±1.96 standard errors;
  questions both files flip add no uncertainty; more questions narrow the interval.
- **GSM8K**: the answer check (the first "####", commas and `$` ignored); llama-server answers every prompt, then
  stops and gives back its share; one answer per question; the original's recorded once.
- **The verdict on the tiny model** (real llama-perplexity and llama-server): a verdict with its interval, Q8_0 drifting
  less than Q2_K, every task reported, read back once made.
- **The winner**: the lowest of every finalist (built and distilled) and the extra file wins; its score relative to
  the preset's; only its file is kept; chosen once.
