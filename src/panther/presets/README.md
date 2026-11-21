# Presets (`panther.presets`)

Part 2 of the pipeline. The **preset** is the competitor: for a contest (a model at a RAM budget, like `14b@5`), it is
llama.cpp's best ready-made file that fits the budget. Our file is judged against it.

A preset is one of llama.cpp's named recipes (`Q4_K_M`, `IQ2_M`, …), made by `llama-quantize` from the 16-bit
original and the importance matrix. Most are **mixes**, not one type everywhere: `Q4_K_M` puts some matrices in Q6_K,
`Q2_K` uses Q3_K and Q4_K in places, so a preset's real size differs from its nominal bits.

This module also holds **the RAM model** (`Budget`): what a file needs on the Pi, and whether it fits a budget. The
search (part 4) uses the same `Budget` for our files.

## What happens, step by step

For one model and one budget, `Presets(model).best(budget, score)`:

1. **Size all 22 presets exactly**, without building them: `llama-quantize --dry-run` prints the size a preset would
   have (its tensor bytes) in under a millisecond.
2. **Build the ones whose size fits** (`llama-quantize --imatrix original.gguf <PRESET>`), then check the **built
   file's** size against the budget: the file is what the Pi loads.
3. **Score every preset whose file fits** with the fitness ([`panther.fitness`](../fitness/README.md): flips % +
   10 × KL on the winner set's text and questions; lower is better). The winner set, not the GA set: choosing the best of
   22 on the search's questions would reward the luckiest, as choosing the winner would (part 7).
4. **The lowest score competes.** In practice this is nearly always the largest preset that fits, but IQ and K
   presets of similar size can swap places, so it is measured, not assumed.
5. **If none fits**, the smallest preset competes anyway and the result is marked **unfair** (`"unfair": true`): it
   has more RAM than our file is allowed. The contest still runs, so very small budgets can still be tried.
6. **The choice is saved** to `data/contests/<model>@<budget>/preset.json` (the preset, its file, its RAM, unfair or
   not, and every scored preset's fitness: score, flips, KL, same top word, flips per task) and returned from there on later calls, so it is made once.

Presets live in `data/models/<model>/presets/<PRESET>.gguf` and are shared by all of a model's budgets (`14b@5` and
`14b@7` reuse each other's). Each is written under a `.part` name and renamed when complete, and built once.
`Presets(model).delete_unused()` deletes every preset file no contest of that model chose.

### The 22 presets

`IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, Q2_K_S, Q2_K, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, Q3_K_S, Q3_K_M, Q3_K_L,
IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0`

Left out: Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL (older formats that lose to the K types at the same size, measured on the
7B); TQ1_0, TQ2_0 and MXFP4 (do not work on normal models, measured); Q1_0 and Q2_0 (new in llama.cpp, never
tested here).

### The RAM model

**RAM = file size + KV cache + overhead**

- **File size**: the whole GGUF, as the Pi loads it (the vocabulary in its metadata included).
- **KV cache** = 2 (keys and values) × layers × KV heads × head width × 4,096 tokens, in 8 bits (34 bytes per 32
  values, from gguf's block sizes). The shape is read from the original GGUF's own metadata; the head width is the
  one the GGUF states (`key_length`) where it does: Mistral Small 24B has 32 heads of 128 on a width of 5,120.
- **Overhead**: 0.40 GiB (llama.cpp's own buffers, measured on the Pi earlier).
- **A file fits a budget of B GiB if RAM ≤ 1.005 × B GiB.**

Context, KV cache type and overhead are settings in `configs/search.yaml`.

## Decisions

| | Decision | Why | Before (Rachit's master) |
|---|---|---|---|
| 1 | Size every preset exactly with `--dry-run`, build only those that fit | exact and instant; no hand-written table to be wrong | a hand table of nominal bits with 0.3 bits of slack decided what to build, and could skip a preset that fits (bug B15) |
| 2 | The same 22 presets | | same |
| 3 | The competitor is the fitting preset with the lowest **fitness** | the strongest preset by the very measure we compete on; one rule throughout | the most **accurate** (ties: lower KL) |
| 4 | RAM counts the **file size** | the Pi loads the whole file, metadata included; simpler | the tensor bytes |
| 5 | None fits: the smallest competes, marked unfair | very small budgets can still be tried; the record says the comparison favours the preset | the same, flagged `over_budget` |
| 6 | Presets shared by a model's budgets; unchosen ones deletable | built once per model; disk | same |
| | Presets use the same importance matrix as our type copies, and no GPTQ | GPTQ is our method under test; the presets are llama.cpp's own | same |

## Using it

```python
from panther.model import Model
from panther.fitness import Fitness
from panther.presets import Budget, Presets

m = Model.load("14b")                     # part 1 has built its original and importance matrix
budget = Budget(m, 5)
competitor = Presets(m).best(budget, Fitness(m, winner_text, winner_questions))  # the winner set
budget.fits(path.stat().st_size), budget.ram(path.stat().st_size)
```

## Files

| File | What |
|---|---|
| `budget.py` | `Budget`: the RAM model for one model and budget (`ram`, `fits`) |
| `presets.py` | `Presets`: the 22 presets of one model (`dry_run`, `size`, `layout`, `build`, `best`, `delete_unused`); `layout` gives the search its starting candidates |

## Tests (`tests/presets/`)

On part 1's tiny model, with the real llama-quantize (skipped where llama.cpp is not installed) and a stand-in for
`Fitness` (its own tests are in `tests/fitness/`).

- **RAM model**: the KV cache is keys and values of every layer at the context in 8 bits; it uses the head width the
  GGUF states (the Mistral 24B case); RAM = file + KV cache + overhead; fits up to exactly 1.005 × the budget.
- **Presets**: 22, none of the left-out ones; every name is one llama-quantize knows; dry-run sizes match the built
  files' tensor bytes and grow with the bits.
- **The competitor**: the fitting preset with the lowest score, even when it is not the largest; every fitting preset
  is scored and no other; presets too big are not built; the choice is saved and reused without scoring again; if
  none fits, the smallest competes and is marked unfair; unchosen presets are deleted.
