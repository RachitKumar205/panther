# QAT (`panther.qat`, optional)

Part 6 of the pipeline, and optional. Distillation (part 5) turns one knob per block of 256 weights, its scale; it
never changes a weight's code. QAT ("quantization-aware training") goes further: it **retrains the weights
themselves while keeping them on the type's grid**, so the model learns to work well despite the rounding. It is the
first stage of EfficientQAT, **Block-AP**, ported from Rachit's branch (`origin/rachit`), leaner.

In part 7 it makes a **7th finalist**: the search's best genome's file, trained by Block-AP, then distilled like every
finalist, and scored with the 6 others on the GA set. It is optional because it needs a lot of separate text and a
lot of disk, and its gain on our files is not measured yet.

## The idea in plain words

- **Block by block.** A model is a stack of transformer blocks (layers). Block-AP trains one block at a time: for each
  sample of text it records what the **original** block outputs, and trains the **quantized** block so its output
  matches (mean squared error). The quantized block's input is what the blocks already trained hand it, so each block
  also makes up for the error left by the ones before it.
- **Latent weights.** Each layer matrix gets a 32-bit copy of its weights, W, starting at the original's. In the
  forward pass W is rounded onto the file's grid: the **nearest codes under the file's scales**, decoded exactly as
  llama.cpp decodes them (`grids.py`). So training always runs the block the file would hold.
- **The straight-through trick.** Rounding has no useful gradient (it is flat between grid points). So the backward
  pass treats it as if it were not there: the gradient of the rounded weight goes straight to W. W moves a little each
  step; every 16 steps the codes nearest to it are picked again.
- **The scales stay as the file has them.** In the paper the scales train too, but llama.cpp's scale covers 256
  weights over fixed integer sub-scales, and on Rachit's 14B training them drifted (block 0's error went 0.0098 →
  0.0181 with scales, → 0.0040 without). Distillation trains the scales afterwards, end to end.
- **Written back**: the new codes go into a copy of the file. Same types, same scales, same size, same speed.

## What happens, step by step

`BlockAP(model, tokens, validation).run(start, out)`:

1. **The original** is loaded from the download (bf16, as published). The **start file** is the finalist's file (our
   13 types). Its work folder: `data/models/<model>/qat/<start file>-<samples>/`, named by their content.
2. **Disk check**: the two hidden-state caches need 2 × samples × 2,048 × hidden width × 2 bytes (the 14B at 4,096
   samples: 172 GB) plus part 1's 20 GiB margin; too little, and it stops before starting.
3. **The caches**: every sample's input to block 0, on the original's path (the original's word table) and on the
   quantized path (the start file's word table, decoded), on disk in 16 bits (bf16; Qwen2.5's activations overflow
   fp16). The 64 validation samples are kept in memory the same way.
4. **For each block**:
   - every layer matrix becomes a `QLinear` (latent weights W, the file's scales and grid); norms and biases frozen;
   - AdamW on W, each matrix at 1e-3 × the size of its weights (root mean square), cosine down to 1/20; 2 samples a
     step, 2 passes over the samples; the target is the original block's output on the original's path;
   - every 512 steps, and at the end, a **check** on the validation samples with codes picked afresh; the best point
     is kept; after 2 checks without a better one, training stops early;
   - back to the best point; the matrices are **packed into the file's bytes, rebuilt from those bytes, and scored
     again**: that is the block's score;
   - if that is **not better than the block the start file already has**, the file's block stays;
   - the block's bytes are saved (`block-<b>.npz`); the caches move on a block (through the original block, and
     through the block as saved).
5. **The output**: a copy of the start file with every block's saved bytes written in (`.part`, then renamed).

**A stopped run continues** after its last saved block: the caches are rebuilt, passing through the blocks already
saved. Returns each block's validation error: before, after, the file's, and whether the file's block was kept.

### Mistral: llama.cpp's row order

llama.cpp stores attention q and k of llama-style models (Mistral) with their rows reordered (part 5). `QLinear`
keeps W in the **file's** row order, so codes map to the file's bytes directly, and hands transformers its own order
in the forward pass (`permute`/`unpermute` from part 5, chosen by the file's architecture).

## Precision

| What | Precision | Why |
|---|---|---|
| The file's scales (`d`, `dmin`) | 16 bits | the format; every use snaps them to 16 bits first, so training sees what the file stores |
| Picking codes | 32 bits, autocast off, TF32 off | as llama.cpp picks them; near-ties between grid patterns decide the code, and 16 bits would round them together |
| The forward pass | bf16 | Qwen2.5's activations overflow fp16; bf16 has fp32's range |
| The latent weights W | 32 bits | updates are ~0.1 % of a weight a step, below bf16's resolution (~0.4 %) |

Training runs the decoded weights in bf16, which differs from llama.cpp by 0.14 % on average (Rachit's measurement).
That costs training a little accuracy, not honesty: **every finalist is measured on the real file.**

### The code-picking bug (Rachit's, 20 Nov) and the four guarantees

Codes were picked in bf16 during training and at checks, but in fp32 when saving. For 4–8 % of IQ groups the two
disagreed on a near-tie, so the file held a different block than the one scored (on the 14B @ 7, block 5 scored
0.01606 and was saved at 0.01981, worse than its start). Here, as in Rachit's fixed version:

1. **One picker everywhere**: 32 bits, autocast off; the same weights give the same codes in training, at a check
   and at the save (tested for all 13 types).
2. **Back to the best point exactly**: a 32-bit copy of the weights at the best check.
3. **Scored from the saved bytes**: packed, rebuilt from them, scored again; that is the score logged and used.
4. **Never worse than the start**: a block not better than the file's keeps the file's.

## Data and disk

The samples are windows of 2,048 tokens of QAT text (FineWeb-Edu), from documents in no other set: 4,096 samples by
default (the paper's) plus 64 for validation. The data part (on hold) draws them; `BlockAP` takes them as arrays.

| Model | Caches at 4,096 samples |
|---|---|
| 1.5B | 2 × 4,096 × 2,048 × 1,536 × 2 B = 52 GB |
| 7B | 120 GB |
| 14B | 172 GB |
| 22B, 24B | 206 GB, 172 GB |

With 500 GB of disk, fewer samples (1,024: a quarter) are the practical setting for the larger models.

## Measured

- **On the test model** (a tiny random Qwen2-shaped model, 16 samples of 256 tokens, its Q2_K file): llama.cpp's KL
  0.0294 → 0.0165 (−44 %), Block-AP alone (before distillation).
- **Rachit's earlier results** (old pipeline): on the 14B @ 5 and 22B @ 7, Block-AP + his Stage 2 beat the old winners
  by +2.84 and +2.18 accuracy points on held-out questions; on the 1.5B's IQ2_XS preset, web perplexity 19.92 → 17.38
  (original 10.72). Those files were made **before the code-picking fix**; the fixed version has not run on a real
  model yet.

## Decisions

| | Here | Why | Rachit's |
|---|---|---|---|
| | Block-AP as he has it: target, inputs, MSE, latent weights, rates, schedule, 2 passes, codes re-picked every 16 steps, checks every 512 with patience 2, scales fixed, his four guarantees | the algorithm and its honesty are his | — |
| | Our 13 types (IQ3_S dropped) | | 14 |
| | The original loaded from the download | exact values; no second copy in RAM (audit item 7) | transformers' GGUF loader |
| | Mistral's row order from the file's architecture (part 5's `permute`) | it is known, not guessed | detected per matrix by which order is nearer |
| | A final check when training ends | the last stretch of training (up to 511 steps, or all of a short run) is not thrown away | best point only from checks every 512 steps |
| | The caches move on a block in a pass of their own, after the block is saved; on resume, rebuilt through the saved blocks one at a time | plainer; one block on the GPU at a time | the next block's first pass did it, with bookkeeping to resume; on resume every saved block on the GPU at once |
| | Reads and writes of the caches in line | plainer | in background threads |
| | The work folder named by the start file's and the samples' content | another start file or other samples never reuse saved blocks | one folder per model |
| | `grids.py`: the type table as a dataclass, one `Grid` class | the same bit-exact code, tidier | functions keyed by type |

## Using it

```python
from panther.qat import BlockAP

log = BlockAP(m, tokens, validation).run(finalist_file, qat_file)  # then distill qat_file (part 5)
```

## Files

| File | What |
|---|---|
| `grids.py` | `Grid`: one of the 13 types in PyTorch: `unpack`/`pack` (bytes ↔ scales and codes, llama.cpp's layouts), `decode` (bit for bit), `encode` (nearest codes, 32-bit), `fake` (straight-through) |
| `blockap.py` | `QLinear` (a matrix on its grid, trained in the file's row order), `BlockAP` (`run`, `fill`, `train`, `advance`, `restored`, `validate`) |

## Tests (`tests/qat/`)

- **Grids**, for every one of the 13 types: the bytes come back byte for byte; decoding equals gguf-py's bit for bit;
  values from a llama.cpp file encode back to the same values; codes are picked the same under bf16 autocast; the
  forward pass is exact, the gradient reaches every weight, and the result packs to a valid file.
- **Block-AP**, on a tiny transformers model and its GGUF, with the real llama-quantize and llama-perplexity: a matrix
  trains in the file's row order and runs in transformers' order; too little disk stops it before it starts; every
  block ends no worse than the file's, and training beats the file somewhere; only layer matrices change (word
  table, output layer, norms, biases byte for byte the same), and the size is the same; llama.cpp runs the result and
  its KL is not worse; a stopped run continues after its last saved block without training it again; another start
  file gets its own work folder; a llama-style model trains with its q/k rows in order.
