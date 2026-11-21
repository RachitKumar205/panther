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
| 0 | Data: the GA set, the distill set and the verdict set | | on hold |
| 1 | The model: download, 16-bit original, importance matrix, 13 type copies (GPTQ) | [`panther.model`](src/panther/model/README.md) | written, tested |
| 2 | Presets: which are built, which one competes; the RAM model | [`panther.presets`](src/panther/presets/README.md) | written, tested |
| 3 | Fitness: flips % + 10 × KL against the original, relative to the preset | [`panther.fitness`](src/panther/fitness/README.md) | written, tested |
| 4 | Search: start candidates, slicing, mutation, repair, stop | [`panther.search`](src/panther/search/README.md) | written, tested |
| 5 | Distillation: block-scale multipliers trained against the original | | next |
| 6 | QAT (optional) | | |
| 7 | The winner (on the GA set) and the verdict (on the verdict set) | | |
| 8 | Running it: `panther run 14b@5`, output folder, upload, the Pi | | |

## The data: three sets that never share a document, prompt or question

| Set | Used for | Holds |
|---|---|---|
| **GA set** | fitness during the search, choosing the preset, picking the winner | text (chat answers and web text) and benchmark questions (all, and a 900-question subset for the search) |
| **distill set** | everything that is built or trained: the importance matrix, GPTQ, distillation | text (chat answers and web text) |
| **verdict set** | the verdict only, once per contest | text and benchmark questions (plus GSM8K) |

Nothing is trained on the GA set or the verdict set, and nothing chooses anything with the verdict set. QAT's large
text is separate again.

## Working on it

`uv sync` installs it; `uv run pytest` runs the tests; `uv run ruff check .` lints. Existing libraries and tools
are used wherever one fits; our own code is only what is specific to Panther.

The previous implementations are for reference only: Rachit's on `master`, Lalit's rewrite on `origin/lalit`.
