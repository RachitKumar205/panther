# The model (`panther.model`)

Part 1 of the pipeline. Everything later is built from what this module makes:

- **the original**: the model as llama.cpp runs it at full (16-bit) precision. It is the reference that every
  file's drift (KL) and answer flips are measured against, and the teacher for distillation.
- **the type copies**: the whole model stored once in each of the 13 types. A candidate file (one type per matrix)
  is then built by copying each matrix's bytes from the right type copy, with no quantizing during the search.

## What happens, step by step

```
download (once, kept)  →  convert  →  importance matrix  →  type copies (llama-quantize, then GPTQ)
```

| Step | What | Tool | Writes (under `data/models/<model>/`) |
|---|---|---|---|
| download | the original's weights at the commit pinned in `configs/models.yaml` (about 2 bytes per weight). Done once and kept: everything after it runs from it without the network | `huggingface_hub.snapshot_download` | `hf/` (`hf/.downloaded` when complete) |
| convert | the 16-bit GGUF; from here on this file *is* "the original" | llama.cpp's `convert_hf_to_gguf.py --outtype bf16` | `original.gguf` |
| importance matrix | runs the original over the distill set's web text; for every matrix, the output layer included, records each input column's mean squared value (how much each column matters) | `llama-imatrix -c 512 --process-output` | `imatrix.gguf` |
| type copies | for each of the 13 types: llama-quantize writes the whole model in that type, then GPTQ overwrites every layer matrix and the output layer with error-compensated codes | `llama-quantize --pure`, then our GPTQ calling llama.cpp's quantizer | `types/<TYPE>.gguf` |
| | and records each matrix's **output error** in each type, which the search's repair steps by (see below) | | `errors.json` |

**Every step is skipped if its file exists.** Every file is written under a `.part` name and renamed when complete,
so a stopped build never leaves a file that looks finished. A stopped type-copy build redoes all the copies that were
not finished. Before each step the free disk is checked (with a 20 GiB margin) against what the step will write:
the download's size from the Hub, the conversion's from the downloaded files, the type copies' from the original's
size scaled by each type's bits. If there is not enough, the build stops and says how much is needed.

### The 13 types

`IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, Q2_K, IQ3_XXS, Q3_K, IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0`: 1.56 to 8.5 bits
per weight, no two the same size. Each type cuts a row into blocks of 256 weights (Q8_0: 32) and stores a scale per
block and a small code per weight; the IQ types store groups of weights as patterns from a fixed grid.

### The word table and the output layer

The **word table** (`token_embd`) turns each token into a vector. It is a lookup with no input, so the importance
matrix has no data for it and GPTQ cannot apply to it: it is rounded the plain way (llama.cpp's own). llama.cpp
refuses IQ1_S, IQ2_XXS and IQ2_XS for a matrix without importance data (checked: it aborts); IQ1_M is kept out too, as
before. In those four type copies the word table is stored as Q8_0 instead; the search never gives it those types.

The **output layer** (`output.weight`) turns the final hidden vector into a score for every word, so its rounding
error goes straight into the next-word probabilities. It reads an input (the final hidden state), so it is treated
like every layer matrix:

- **Its importance data is collected** (`llama-imatrix --process-output`), so it can take all 13 types. llama-imatrix
  leaves the output layer out by default, and the earlier pipeline kept that default (no model turned it on, and no
  reason was recorded); the output layer could then not take the four lowest types.
- **GPTQ compensates it**, like the layer matrices. It is 7 % of the 7B, 5 % of the 14B, 3 % of the 24B and 1 % of the
  22B. The earlier pipeline rounded it plainly. Its effect has not been measured yet (planned: a uniform Q3_K copy of
  the 7B with and without it).
- **A tied model has no output layer**: the 1.5B uses the word table both ways, so there is nothing to compensate.

### GPTQ (error-compensated rounding)

Plain rounding rounds every weight on its own. But what matters is the matrix's **output**, and the inputs it gets
are correlated, so an error in one column can be partly cancelled by adjusting others. GPTQ does that:

1. **H** = the mean of x xᵀ over 65,536 tokens (128 × 512) of the distill set's web text, x being what the original
   (loaded from the download by transformers) feeds that matrix. q/k/v read the same input and share one H; so do
   gate/up; the output layer reads the final hidden state. H is collected for a group of inputs at a time (as many as
   fit in half the free GPU memory).
2. A column whose input is always 0 is zeroed; 1 % of H's mean diagonal is added to its diagonal (damping, for a
   stable inverse); **U** = the upper Cholesky factor of H⁻¹. This is done once per matrix and shared by all 13 types.
3. 256 columns at a time: llama.cpp's own quantizer rounds them (given those columns' importance, H's diagonal);
   the error, divided through U's block, is subtracted from the columns not yet rounded.
4. The codes, exactly the bytes llama.cpp produced, overwrite that matrix in the type copy: same type, same size,
   same speed on the Pi.

Measured on Qwen2.5 7B (13 Nov, every matrix in one type, drift = KL from the original, lower is better):

| Type | Plain rounding | GPTQ | |
|---|---|---|---|
| Q4_K | 0.0157 | 0.0145 | −8 % |
| Q3_K | 0.134 | 0.064 | −52 % |
| Q2_K | 0.241 | 0.214 | −11 % |
| IQ2_XS | 0.333 | 0.302 | −9 % |

**The output error (`errors.json`).** For every matrix and type, GPTQ also records how much the matrix's output
changes: E[‖(W − Q)x‖²] = trace((W − Q) H (W − Q)ᵀ), from the same W, Q and H it rounds with. The word table has no
input: its error is each row's change weighted by how often that token appears in the distill set's text (the same
units: the squared change per token of what the matrix hands on). The search's repair steps by it (part 4).

GPTQ on the GPU and on the CPU can pick a few different codes (the error propagation's floating-point sums differ
slightly); a build uses the GPU when there is one.

## Decisions

| | Decision | Why | Before (Rachit's master) |
|---|---|---|---|
| A | Keep type copies | GPTQ runs once per matrix per type; building a candidate is a byte copy (seconds) | same |
| B | Keep GPTQ, writing llama.cpp's own code bytes | the largest measured gain; the file holds exactly what was computed | same (Lalit's rewrite returned decoded values instead, which re-encode to different codes) |
| C | 13 types: IQ3_S dropped | same size as Q3_K (3.44 bits), measured the same with GPTQ; a step between them saved 0 bytes | 14 types |
| F | Disk check before each step; `.part` then rename | a full disk or a crash never leaves a broken file that looks finished | same |
| | Type copies made by `llama-quantize --pure`, then GPTQ overwrites the layer matrices and the output layer | llama.cpp writes the file (metadata, word table, norms); our code only replaces matrices. Costs one extra quantize pass per type | Rachit wrote the file himself (gguf-py) to skip that pass |
| D | The download is kept; no Hugging Face backup | nothing is ever downloaded twice; only the type copies are deleted for disk, and remade from the download without the network | the download was deleted after conversion; the whole store was uploaded to a public repo and restored |
| E1 | GPTQ on the output layer too | it has an input like any layer matrix, and its error goes straight into the next-word probabilities | plain rounding |
| E2 | Importance data for the output layer (`--process-output`) | the output layer can take all 13 types; one special case fewer | not collected (llama-imatrix's default) |
| | H from the download (transformers) | transformers loads its own format directly and handles Qwen2 and Mistral; GPTQ needs only the inputs' columns, which llama.cpp never reorders | H from the 16-bit GGUF through transformers' GGUF loader |
| | U computed once per matrix, shared by all types | the same H for every type | computed again for every type |
| | Download skips Mistral's duplicate `consolidated*` weights | they double the download | downloaded everything |
| | Existing tools wherever one fits | less of our own code to trust | — |

No library does GPTQ into llama.cpp's types. The closest, IST-DASLab's gptq-gguf-toolkit, is research scripts for the
K types only (no IQ types, no Q8_0) and does not call llama.cpp's quantizer, so we keep our ~20 lines of GPTQ and
let llama.cpp do the formats.

## Disk

The server has 500 GB. Per model:

| Model | Download (kept) | 16-bit original + 13 type copies (deletable) |
|---|---|---|
| 1.5B | 3 GB | ~13 GB |
| 7B | 15 GB | ~60 GB |
| 14B | 30 GB | ~120 GB |
| 22B | 44 GB | ~180 GB |
| 24B | 47 GB | ~190 GB |

All five downloads take ~140 GB. A model's original and type copies are built when its contests run;
`m.delete_type_copies()` frees them afterwards, and the next `build` remakes them from the download, with no
network: run a model's contests back to back. With every download kept and the 24B built, the models use ~330 GB,
leaving ~170 GB for the later parts. Rebuilding the type copies costs GPTQ's time (not yet measured; the 1.5B build
will give the first number).

## Using it

```python
from panther.model import Model

m = Model.load("14b")                # or the full name from configs/models.yaml
m.build(Path("distill-web.txt"))     # the distill set's web text (the data part makes it)
m.original, m.imatrix, m.copy("Q4_K")
m.delete_type_copies()
```

It needs llama.cpp: `llama-imatrix` and `llama-quantize` on `PATH`, `libggml-base.so` on the library path, and
`PANTHER_CONVERT` = its `convert_hf_to_gguf.py` (default `/opt/llama.cpp/convert_hf_to_gguf.py`). Files go under
`PANTHER_DATA` (default `data/`). Nothing is checked up front: a missing program, library or file fails with its own
error and traceback.

### Precision

The original is stored in **16 bits** (bf16, the format the models are published in): that is "full precision" here,
and nothing is ever stored above it. The arithmetic runs in **32 bits** (float32): bf16 → float32 is exact (it only
appends zero bits), llama.cpp's quantizer takes float32, and the Cholesky factorisation in GPTQ would be too imprecise
in 16 bits. The result is stored in the low-bit types.

## Files

| File | What |
|---|---|
| `model.py` | `Model`: the config entry, its files, and the steps as methods (`build`, `download`, `convert`, `make_imatrix`, `make_type_copies`, `hessians`, `delete_type_copies`); the 13 types |
| `gptq.py` | `GPTQ`: built from one input's H, rounds every matrix reading it; `input_of` (which transformers module's input a GGUF matrix reads); `collect_hessians` |
| `quantizer.py` | `Quantizer`: llama.cpp's quantizer called directly; the codes are decoded by gguf-py |

## Tests (`tests/model/`)

Run with `uv run pytest tests/model`. Tests that need llama.cpp are skipped where it is not installed.

- **Model**: 13 types, fewest bits first, no two the same size; every model loads by name and size and is pinned;
  file locations; too little disk fails.
- **GPTQ maths** (stand-in quantizer): with uncorrelated inputs GPTQ equals plain rounding; with correlated inputs it
  lowers the output error; each block gets its own columns' importance; the returned codes are the quantizer's; an
  unused column is zeroed.
- **Output error**: equals the mean squared change of the matrix's output; recorded for every matrix in every type it
  may take, growing as bits fall; the word table's counts only the tokens that occur.
- **GPTQ with llama.cpp**, every type: the codes, decoded by gguf-py (an independent decoder), are exactly the values
  GPTQ compensated for; GPTQ beats plain rounding.
- **H**: equals the mean outer product of what each input sees, for a layer input and for the output layer; every
  GGUF matrix maps to the transformers module whose input it reads; symmetric; the same whether inputs are collected
  together or one at a time; hooks removed.
- **Steps**: the commands each step runs; a done step is skipped; convert and the importance matrix write then
  rename; the importance matrix includes the output layer; the download is pinned, skips Mistral's duplicates and
  happens once; `build` runs the steps in order and keeps the download.
- **Type copies end to end** (a tiny Qwen2-shaped model, real llama-quantize and quantizer): every copy has every
  tensor; every layer matrix and the output layer in the copy's type (the lowest types included) and holding GPTQ's
  codes byte for byte; the word table in Q8_0 where the type needs importance data; norms and biases unchanged; a tied
  model (no output layer) builds too; a stopped build leaves no copy and is redone; existing copies are kept; a matrix
  GPTQ did not reach stops the build.
