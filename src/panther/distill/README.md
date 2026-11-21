# Distillation (`panther.distill`)

Part 5 of the pipeline. The search's 3 finalists are good layouts, but their rounding still loses some of the
original's behaviour. Distillation is a light, cheap kind of training: it nudges each block of weights up or down so
the finalist's next-word predictions match the original's again, and writes that into the finalist's file without
changing its size or speed. Part 7 then scores each finalist with and without distillation on the GA set, and the
lower score wins (distillation helps most at tight budgets and can do nothing for near-lossless files, so the
measurement decides).

## The idea in plain words

A llama.cpp file does not store each weight as a number. For each **block** of 256 weights in a row it stores:

- one **scale** `d`, a 16-bit number: how big this block's weights are;
- a small **code** per weight: which step of that scale it is.

A weight is roughly `d × code`. Rounding to a few bits leaves each block slightly too big or too small on average,
and the errors add up through the layers.

Distillation learns **one multiplier per block**, `exp(θ)` (θ starts at 0, so the multiplier starts at 1): "this
block 3 % bigger", "that one 2 % smaller". It also learns a small correction to every **norm** (the per-channel
weights the model uses to rescale its activations between layers). The **codes are never touched**.

Multiplying a block's weights by a factor is the same as multiplying its scale `d` by that factor. So writing it
back is just: **each block's `d` becomes `d × exp(θ)`**, and each norm gets its correction. Only those few bytes
change; the file keeps its size and runs exactly as fast.

## What happens, step by step

1. **The teacher's answers, recorded once per model** (`teacher()`): the original, the 16-bit model exactly as
   published, reads the **distill set**'s text (its chat answers and web text) in sequences of 512 tokens. At every
   position its **64 most likely next words** and their log-probabilities (32-bit) are stored in
   `data/models/<model>/distill/<content>.pt`, named by the text's content. The teacher is then freed from the GPU;
   all 3 finalists reuse the record.
2. **The student** (`student(file)`): the finalist's file, loaded as a transformers model with **every matrix and
   norm holding exactly the file's decoded values**, in 32 bits. (The 22B and 24B do not fit on the GPU in 32 bits:
   they are held in bf16, each weight within about 0.2 % of the file's value. That is far below the rounding error
   being corrected, 10–30 % at 2–4 bits, and only affects training: what is written and scored is exact.) Then
   attached, as the only trainable parts:
   - a **block scale** θ (32-bit, one per 256 weights of a row) on every layer matrix and on the output layer, where
     the type has a single scale per block (every type but IQ1_M);
   - a **norm shift** (32-bit) on every norm, so small updates are never lost in a 16-bit weight.
   The word table is not trained (it is a lookup: only the rows of tokens that appear would get any training).
3. **Training** (`train`): 600 updates of 2 sequences; every sequence is used once, in a fixed random order, before
   any is used again (about 1.6 passes over the distill set). Adam: block scales at 0.002, norm shifts at 0.0001.
   Activations are recomputed in the backward pass instead of stored (so the 24B fits).
   **The loss** (`kl_top`) is the KL divergence from the teacher to the student over 65 outcomes: the teacher's 64
   most likely next words, and "everything else" as one. Both use their true probabilities (nothing renormalized).
4. **Writing back** (`write`): a copy of the finalist's file in which each trained block's `d` (and `dmin`, for the
   types that have one) is `d × exp(θ)` rounded to its 16 bits, and each norm is the file's own 32-bit value plus its
   shift. Nothing else changes, byte for byte. Written under a `.part` name and renamed when complete.

`Distill(model, distill_text).run(finalist, out)` does 1 to 4 for one finalist and returns the loss of every update.

### Where the scales are

Each type keeps its block scale at a fixed byte offset inside its block (ggml's block structs, checked against llama.cpp
in the 21 Nov audit):

| Type | Block | Scale `d` at byte | Also |
|---|---|---|---|
| Q2_K | 256 weights | 80 | `dmin` at 82 |
| Q3_K | 256 | 108 | |
| Q4_K, Q5_K | 256 | 0 | `dmin` at 2 |
| Q6_K | 256 | 208 | |
| Q8_0 | 32 | 0 | 8 blocks per multiplier |
| IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ4_XS | 256 | 0 | |
| IQ1_M | 256 | none (spread over the block) | not trained |

`dmin` scales the block's offsets; scaling `d` and `dmin` by the same factor scales every weight of the block by it.

### Mistral: llama.cpp's row order

When llama.cpp converts a llama-style model (Mistral Small: `MistralForCausalLM`), it reorders the rows of each
layer's attention q and k (`convert_hf_to_gguf`'s `LlamaModel.permute`, by the head count for q and the KV head count
for k). Qwen2 is not reordered. So for Mistral the file's q/k rows are put back in transformers' order when the student
is loaded (`unpermute`), and the trained multipliers are put into the file's order when written (`permute`).

### Precision ("rounding")

Nothing is ever held below what it is:

- the teacher is the 16-bit original as published, and its answers are stored in 32 bits;
- the student holds the file's values exactly (32 bits; bf16 within 0.2 % only for the 22B and 24B, during training
  only);
- θ, the norm shifts and the products `W × exp(θ)` are 32-bit;
- **the only rounding is the file's own**: a new scale is stored in 16 bits. A 16-bit `d` keeps about 3 decimal
  digits, except when it is tiny: in Q6_K and IQ4_XS, `d` is about 1e-5 for weights of 0.05, below 16-bit's normal
  range, where it has fewer bits. Measured: the written weights are within 0.27 % of a row's largest weight of what
  was trained. llama.cpp's own quantizer writes `d` with the same limit.
- **What is scored is always the written file** (part 7), never the model in memory, so any rounding is in the result,
  never hidden.

## Measured on the test model

A tiny random Qwen2-shaped model (2 layers), 80 updates on 12k tokens of text, KL measured by llama.cpp on other text:

| Type | KL before | KL after | |
|---|---|---|---|
| Q2_K | 0.0288 | 0.0079 | −72 % |
| IQ2_XS | 0.0330 | 0.0105 | −68 % |
| Q4_K | 0.0015 | 0.0004 | −73 % |

Earlier, on Qwen2.5 1.5B's 2-bit preset (13 Nov, Rachit's method): drift −29 %, same top word 64.5 % → 71.9 %.

## Decisions

| | Decision | Why | Before (Rachit's master) |
|---|---|---|---|
| A | Train every layer matrix **and the output layer**; not the word table | the output layer's error goes straight into the next-word probabilities (and is GPTQ-rounded since part 1); the word table is a lookup | layer matrices only |
| B | Exact 32-bit student where it fits (1.5B, 7B, 14B), bf16 for 22B and 24B; 32-bit θ and norm shifts; teacher's answers in 32 bits; the written file is what is scored | no bit errors anywhere a choice exists | the student loaded in bf16 through transformers' GGUF loader; norms trained in bf16; teacher's answers in fp16 |
| | **The loss: exact KL over the teacher's top 64 + "everything else"** | the old loss renormalized the teacher's top 64 but not the student's, which rewards piling all probability onto those 64 words. On the test model it made llama.cpp's KL worse (0.0288 → 0.0329) while its own loss fell; the exact KL lowers it (→ 0.0079). On a real model the top 64 hold most of the probability, so the bias is smaller, but it was there | renormalized teacher top 64 |
| C | transformers for the model, PyTorch's `parametrize` for `W × exp(θ)` and the norm shifts, gguf-py for the bytes | existing tools; our code is only the write-back | the same, plus transformers' GGUF loader |
| | The student loaded from the download, then every matrix overwritten with the file's decoded values | exact values, and no second copy of the model in RAM (audit item 7) | transformers' GGUF loader |
| D | Mistral's q/k row order handled both ways, tested against llama.cpp's own formula | | the same, checked at load |
| E | The teacher recorded once per model, reused by all finalists | | the same |
| F | The distill set only | nothing trains on the GA or verdict set | the same (the train split) |
| | 600 updates × 2 sequences × 512 tokens, rates 0.002 / 0.0001, top 64 | | the same |

## Using it

```python
from panther.distill import Distill

distill = Distill(m, distill_text)                   # the distill set's text (the data part makes it)
losses = distill.run(finalist_file, distilled_file)  # same size and speed; score distilled_file in part 7
```

## Files

| File | What |
|---|---|
| `distill.py` | `Distill` (`run`, `teacher`, `student`, `train`, `write`), `kl_top` (the loss), `BlockScale` and `NormShift` (the trainable parts), `permute`/`unpermute` (Mistral's row order), `SCALE_AT` (where each type's scale is) |

## Tests (`tests/distill/`)

On a tiny transformers model saved as the download, with the same weights written as the 16-bit original GGUF (a
Qwen2 one, and a llama-style one with llama.cpp's q/k row order), and the real llama-quantize and llama-perplexity.

- **Row order**: `permute` equals llama.cpp's own formula (copied from the pinned commit), for q and k; `unpermute`
  undoes it.
- **The student**: holds exactly the file's values (32-bit); a llama-style file's q/k rows are put back in order
  (every weight equal to the transformers model's); only block scales and norm shifts train, the output layer included,
  the word table not; IQ1_M gets no block scale.
- **The teacher**: its top 64 (32-bit, most likely first), recorded once and reused.
- **Writing back**, for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XS, IQ4_XS: the file's size is unchanged; untrained
  tensors are unchanged byte for byte; in trained matrices the codes are unchanged byte for byte and every scale is
  exactly `d × exp(θ)` rounded to 16 bits; each norm is exactly the file's value plus its shift; the written weights
  are the trained ones up to the 16-bit scales. For a llama-style file the scales land in the file's row order.
- **Training**: the loss is the exact KL of the top 64 and the rest (zero for the teacher itself); each sequence is
  used once before any repeats; training lowers the drift llama.cpp measures.
