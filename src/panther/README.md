# src/panther

| File | What |
|---|---|
| `model.py` | A model's matrices (from its vendored `config.json`), bytes per type, the RAM model, the budget's type menu |
| `llama.py` | llama.cpp runs (KL, perplexity, multiple choice, WinoGrande, GSM8K and chat answers via llama-server), output parsing and checks, one JSON record per run in `data/results/<model>/` |
| `stats.py` | Accuracy, flips against the original, the score's standard error, the relative score with its 95 % interval, WIN/TIE/LOSE |
| `rounding.py` | Error-compensated rounding (GPTQ) into llama.cpp's types, written into the type copies in place |
| `distill.py` | Distillation of a finished file: one multiplier per block scale and the norm weights, trained to match the original's next-word probabilities (on the train text: its own chat answers and web text), written back into the GGUF |
| `solver.py` | CP-SAT: one type per matrix inside a byte window, least predicted loss |
| `fakequant.py` | Fake quantizers for all 14 types (quantization-aware training, EfficientQAT-style; every type as step x value - offset): unpack/pack a matrix's bytes, decode exactly as llama.cpp, encode to the nearest codes, and a forward pass whose gradients reach the weights (straight through) and the scales (LSQ) |
| `blockap.py` | EfficientQAT's Stage 1 (Block-AP) for a llama.cpp file: block by block, each layer matrix's weights and block scales trained (mean squared error against the original block, inputs from the blocks already quantized), written back into a copy of the file. On the 1.5B (IQ2_XS preset, 256 x 512 tokens): web-text perplexity 19.92 -> 17.38 (original 10.72) The weights' learning rate is relative (1e-3 of each matrix's weight size: the paper's 2e-5 on Qwen2.5-14B, 2e-6 on Mistral-22B, whose weights are 10x smaller and blew up at 2e-5). Qwen and llama-style models (Mistral): the file's row order of attention q/k is found from the values. |

## Testing `fakequant.py`

`tests/test_fakequant.py` needs torch, so it runs on the server (it is skipped by `./panther test` on a Mac). It checks
decoding against llama.cpp's own C code and the gguf package bit for bit, packing byte for byte, encoding against
llama.cpp's quantizer, and the gradients; `PANTHER_FQ_FILE` adds a round trip of every matrix of a real GGUF. The
server image has no pytest; a copy lives in `data/tools/pylib`:

```bash
./panther exec bash -c 'cd /work/repo && PANTHER_FQ_FILE=/work/data/rewrite/Qwen2.5-14B-Instruct/tuned/5-7b24bfecb2c6.gguf PYTHONPATH=/work/repo/src:/work/data/tools/pylib /opt/venv/bin/python -m pytest -q -p no:cacheprovider tests/test_fakequant.py'
```
