# The search (`panther.search`)

Part 4 of the pipeline. A **genome** gives every matrix one of the 13 types (the 7 matrices of every layer, the word
table, the output layer: 198 genes for the 7B, 282 for the 24B). The search looks for genomes that fit the budget
and have the lowest fitness on the **GA set**, and hands the best 3 to distillation (part 5).

## What happens, step by step

1. **The preset is scored** with the search's fitness (the GA set's text and its 900 search questions), so every
   candidate's relative score compares like with like.
2. **Starting candidates**, every one scored:
   - the layouts of llama.cpp's 22 presets (read from `llama-quantize --dry-run`, nothing built) and the 13 one-type
     layouts (all Q2_K, all IQ3_XXS, …), each **repaired** into the window. A type we do not have maps to the
     nearest of ours by bits (IQ3_S → Q3_K, the same size);
   - **4 genomes from CP-SAT** (OR-Tools): one type per matrix, inside the window, with the lowest summed output
     error (errors.json), each at least 3 matrices different from the others. CP-SAT finds the best by that measure
     exactly. The summed error is only an estimate of the fitness, so these are candidates like the others, and
     every one is measured.
   Duplicates are dropped.
3. **Then, in rounds of 8 candidates scored at once**: 7 children and 1 from the solver. Each child:
   - **30 %**: a **mutation** (1 to 3 random matrices, each 1 or 2 types up or down) of one of the **4 best**: this
     polishes the leaders;
   - **70 %**: a two-point **slice** of two **parents** (each the better of two random members of the population, the
     best 16 scored so far), half of those also mutated: this combines and explores;
   - then **repaired** into the window. A child already scored is replaced by another.
   The **solver's candidate** (see "The solver during the search" below) is the genome with the lowest summed output
   error once each role's errors are weighted by what the scores so far say that role costs, at least 3 matrices away
   from everything scored and from the round's children. If it finds none, the round gets one more child.
   Each candidate's file is **built in RAM** (`/dev/shm`), **scored**, and deleted. The round's scores are recorded
   in the order the candidates were made; the population is again the best 16.
4. **Stop** after 15 candidates in a row with no new best (a plain lower score), with at least 30 and at most 70
   candidates in all (the starting candidates count; checked after each round, so a stop can come up to 7 candidates
   past the 15). Also stop if 1,000 children in a row were all scored already.
5. **The best 3** distinct genomes, with their scores and scores relative to the preset's (100 = the preset's
   damage), are the finalists.

Everything is recorded in `data/contests/<model>@<budget>/search.json` (the preset's score, every scored genome with
its score, the finalists) after every round (a stop loses at most the round in progress). **A stopped search continues where it stopped**: it reads the record,
skips what is scored, and makes the same choices it would have made. Every random choice is seeded (the n-th
candidate's choices from the seed and n), so **the same run gives the same result**. A finished search returns its
finalists without scoring anything.

### The window

A genome is valid if its RAM (part 2's `Budget`) is **at least 0.99 and at most 1.005 × the budget**. The lower bound
makes sure it uses the budget: without it the search would spend candidates on files that leave RAM unused.

### Size, before building

A genome's file size is known exactly before it is built: the original's file size, minus its matrices' 16-bit
bytes, plus each matrix's bytes in its chosen type (gguf's block sizes, padded to the file's 32-byte alignment).

### Building a candidate's file

Each matrix's bytes are copied from the type copy of its type; metadata, norms and biases from the original, with
gguf-py's writer. No quantizing; it takes seconds. The file lives in the contest's folder only while it is scored.

### Repair

Too big: step **down** one matrix by one type; too small: step **up** one matrix by one type (never past the budget);
repeat until inside the window. Which matrix: one of the 8 with the **least output error added per byte saved**
(going down) or the **most removed per byte added** (going up), chosen at random among the 8 so repairs vary.

The **output error** of a matrix in a type is recorded by part 1 (`errors.json`): how much the matrix's output
changes, E[‖(W − Q)x‖²] over the distill set's text, computed by GPTQ from the same W, Q and H it rounds with. The
word table has no input: its error is each row's change weighted by how often that token appears in the distill
set's text, which is in the same units (the squared change per token of what the matrix hands on). It only steers
repair; every genome is still scored.

If no matrix can move the genome into the window, or 10,000 steps do not reach it, repair fails loudly.

### The solver during the search

The summed output error is an estimate. The scores show where it is wrong: perhaps errors in `ffn_down` cost more
fitness than their size suggests, and errors in `attn_k` less. So every round the solver learns one **weight per
role** (9: the word table, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`, the output
layer) from every score so far, and asks CP-SAT for the genome with the lowest **weighted** summed error.

The fit: score ≈ α × Σ (role weight × that role's summed error) + c. First α and c from the plain sum (all weights
1); then the weights, by a **ridge fit pulled toward 1** with the strength of `RIDGE` = 10 scored genomes, equally for
every role in score units. With few scores the weights stay near 1 (the plain summed error); a role whose errors
barely move the score stays near 1; a weight moves only as far as the scores support. Weights stay above 0.01.

Rachit's solver fitted about 30 weights (a role in a quarter of the depth) with no pull toward anything, from as few
as 8 scored genomes, and clipped negative weights to almost 0: too many weights for too few scores, and a clipped
group was then pushed down to IQ1_S (audit finding 10). Nine weights, pulled toward 1, avoid both.

CP-SAT runs on one thread with a deterministic time limit, so the same question gets the same answer and the search
stays reproducible.

### Crossover: two slices

The genes are laid out in the order the model runs them: the word table, layer 0, …, the last layer, the output layer.
Each of these is a **unit**: a layer's 7 matrices always come from one parent. Two cut points are drawn; the child
takes the units between them from the second parent and the rest from the first.

The other common choice is a coin flip per unit (Rachit's). There is no measurement here that either is better.
Slicing keeps runs of neighbouring layers together, and neighbouring layers tend to want similar precision.

## Decisions

| | Decision | Why | Before (Rachit's master) |
|---|---|---|---|
| 1 | Window 0.99–1.005 × the budget | | same |
| 2 | Build a candidate by copying bytes from the type copies; size known exactly before building | seconds per file; no surprises | same |
| 3 | Start from the 22 presets' layouts and the 13 one-type layouts, all repaired and scored | llama.cpp's own recipes are good starting points | the competitor's layout, the one-type layouts, random mixes, mutants |
| 4 | Repair by output error per byte, from errors.json (part 1, recorded by GPTQ) | steps the matrices that matter least; the error measures the matrix's output change | an offline estimate from the importance matrix, on a different scale for the word table and output layer (audit finding 3) |
| 5 | Crossover: a two-point slice of whole units | "gene slicing"; neighbouring layers stay together | a coin flip per layer |
| 6 | Mutation: 1–3 matrices, ±1 or 2 types | one simple rule | three kinds, including whole groups |
| | Each child is measured (no screening by summed error) | the summed error is an estimate; screening every child by it would steer the whole search by it | the same |
| 7 | Seeded | the same run gives the same result (audit finding 11) | not reproducible |
| 8 | Stop after 15 without a new best (plain score), 30 to 70 candidates | | a new best counted only if better by 1 standard error |
| 9 | Record every score; continue a stopped search; the best 3 with relative scores | | similar |
| | No first look | | a first look on a third of the questions |
| A | Rounds of 8 candidates scored at once, recorded in the order made | many times faster on a GPU that holds several files; still the same result for the same seed | several at once, recorded in the order they finished (not reproducible) |
| B | A candidate's five measurements run at once (part 3) | faster again | one after another |
| C | Candidate files in RAM (`/dev/shm`) | building and the five loads read from memory, not disk. RAM for a round's files (8 × the budget) is set aside before anything else runs | on disk |
| | 30 % of children polish one of the 4 best; 70 % slice two parents (half also mutated) | the leaders are polished and the population explored | same |
| | CP-SAT gives 4 starting candidates, from the summed output error | the best layouts by that estimate, found exactly; measured like any other | the competitor's layout, one-type layouts, random mixes |
| | 1 candidate per round from CP-SAT, with 9 role weights fitted by a ridge pulled toward 1 | learns from the scores where the estimate is wrong, without the instability | 4 proposals whenever the queue was empty, from about 30 unregularised weights, clipped (unstable: audit finding 10) |

**Measured on the test model** (a stand-in fitness: how many matrices differ from a hidden target genome): the best
start was 10 away; evolution reached 7 by 50 candidates, 3 by 200 and 2 by 400. The stopping rule (15 without a new
best) can stop a slow climb early; whether that matters on real fitness will show in the first real runs.

## Using it

```python
from panther.search import Search

search = Search(m, budget, Fitness(m, ga_text, ga_questions_900), preset_file)
finalists = search.run()          # [{"genome": {...}, "score": {...}, "relative": 87.3}, ...]
search.genes.build(finalists[0]["genome"], path)
```

## Files

| File | What |
|---|---|
| `genes.py` | `Genes`: one model's genes at one budget: the order, the units, the allowed types, exact bytes, the errors; `size`, `fits`, `like`, `solve` (CP-SAT, optionally weighted, away from given genomes), `fit` (the role weights), `slice`, `mutate`, `repair`, `build` |
| `search.py` | `Search`: the starting candidates, the rounds (children and the solver's proposal), the stop, the record |

## Tests (`tests/search/`)

On the tiny model's real type copies (built once), with stand-ins for the fitness.

- **Genomes**: matrices in the order the model runs them, in units; the word table never takes a type needing
  importance data; a layout maps to the nearest type each matrix may take; the predicted size is the built file's
  size; a built file holds each matrix from its type copy and the rest from the original; llama.cpp runs a built file
  exactly like the type copy it equals.
- **The window**: exactly 0.99 to 1.005 × the budget.
- **Repair**: too big and too small genomes reach the window; the same seed gives the same repair; it steps by the
  error table; it fails loudly when the window cannot be reached.
- **Moves**: mutation moves 1 to 3 matrices by at most 2 types; slicing takes one run of whole units from the second
  parent.
- **Speed**: a round's candidates are scored at once.
- **Children and the solver**: about 30 % of children are a small change to one of the 4 best; the solver's genomes fit
  the window, are at least 3 matrices apart, and the best has a lower summed error than every repaired layout; with
  no genome inside the window it gives none.
- **The solver during the search**: the fit recovers known role weights from synthetic scores, and its weights steer
  the solver to a genome that is better by those scores; with no or few consistent scores the weights stay at 1, and
  they are always positive; the weighted solver keeps 3 matrices away from given genomes and gives the same answer
  twice; every round has one solver candidate, recorded after the round's children, 3 matrices from everything
  before it.
- **The search**: the best 3 distinct genomes, all in the window, nothing scored twice, at most 70, relative scores,
  candidate files deleted; evolution beats every starting candidate on a target no start is good at; the same seed
  gives the same search; a stopped search continues exactly where it stopped (losing only the round in progress)
  without scoring anything again; a
  finished search is not run again.
