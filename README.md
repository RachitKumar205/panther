# Panther

For each RAM budget of a Raspberry Pi 5 (1, 3, 5 and 7 GiB) and several open language models (Qwen2.5 1.5B, 7B,
14B; Mistral Small 22B and 24B), Panther builds a llama.cpp file that fits the budget and loses less of the original
model's quality than llama.cpp's best ready-made preset that fits. Each model-and-budget pair is a **contest**,
written like `14b@5`.

This branch (`jia`) rebuilds the pipeline simply, one part at a time. Each part is a module under `src/panther/`
with a README that explains, in plain words, what it does, every decision and why, and its tests. A part is written
only after its design is agreed.

## The pipeline

| # | Part | Module | Status |
|---|---|---|---|
| 0 | Data: the GA set, the winner set, the distill set and the verdict set, built from pinned sources | [`panther.data`](src/panther/data/README.md) | written, tested |
| 1 | The model: download, 16-bit original, importance matrix, 13 type copies (GPTQ) | [`panther.model`](src/panther/model/README.md) | written, tested |
| 2 | Presets: which are built, which one competes; the RAM model | [`panther.presets`](src/panther/presets/README.md) | written, tested |
| 3 | Fitness: flips % + 10 × KL against the original, relative to the preset | [`panther.fitness`](src/panther/fitness/README.md) | written, tested |
| 4 | Search: start candidates, slicing, mutation, repair, stop | [`panther.search`](src/panther/search/README.md) | written, tested |
| 5 | Distillation: one multiplier per block of 256 weights and every norm, trained against the original, written back into the file | [`panther.distill`](src/panther/distill/README.md) | written, tested |
| 6 | QAT (optional): Block-AP, the weights trained block by block on llama.cpp's own grids (Rachit's, ported leaner) | [`panther.qat`](src/panther/qat/README.md) | written, tested |
| 7 | The winner (on the winner set) and the verdict (on the verdict set): WIN / TIE / LOSE with a 95 % interval | [`panther.verdict`](src/panther/verdict/README.md) | written, tested |
| 8 | Running it: `python -m panther fetch / run`, and `./panther run 14b@5` on anton | [`deploy`](deploy/README.md), `src/panther/run.py` | written, tested |

## The data: four sets that never share a document, prompt or question

| Set | Used for | Holds |
|---|---|---|
| **GA set** | the search's fitness | its own text (chat answers and web text) and the search's 900 benchmark questions |
| **winner set** | choosing the preset and the winner | its own text and benchmark questions |
| **distill set** | everything that is built or trained: the importance matrix, GPTQ, distillation | text (chat answers and web text) |
| **verdict set** | the verdict only, once per contest | text and benchmark questions (plus GSM8K) |

Nothing is trained on the GA, winner or verdict sets, and nothing chooses anything with the verdict set. QAT's large
text is separate again.

## Running it

On anton: see [`READ-IF-JUST-PULLED`](READ-IF-JUST-PULLED) and [`deploy/README.md`](deploy/README.md). On any GPU machine
with llama.cpp installed: `uv run python -m panther fetch 1.5b`, then `uv run python -m panther run 1.5b@1`.

## Working on it

`uv sync` installs it; `uv run pytest` runs the tests; `uv run ruff check .` lints. Existing libraries and tools
are used wherever one fits; our own code is only what is specific to Panther.

The previous implementations are for reference only: Rachit's on `master`, Lalit's rewrite on `origin/lalit`.
