# src/panther

The library the pipeline is built on. No server or Pi specifics here.

| File | What |
|---|---|
| `model.py` | A model's weight matrices (genes) from its config: layers, matrix roles, shapes. Quantization types, their exact byte sizes (from gguf-py), which types each matrix may use (IQ1/IQ2 types only where there is importance-matrix data). The RAM model: weights + KV cache (8-bit, 4,096 tokens) + runtime overhead. |
| `gguf_files.py` | Builds a mixed-type GGUF for a genome by copying each tensor from the model's per-type quantized copy (no requantizing), and verifies every tensor's type after writing. |
| `evaluate.py` | One llama.cpp run on the GPU and its parsed record: KL from the original (`--kl-divergence`), perplexity, multiple-choice benchmarks (MMLU, ARC, HellaSwag) and WinoGrande, per question; a fixed subset of a split with `n_tasks` (llama.cpp seeds its choice with 1). GSM8K prompts and answer checking. |
| `stats.py` | Paired bootstrap comparison of two configurations on the same questions (10,000 resamples): accuracy difference with its 95 % interval; perplexity ratio only within one model family (one tokenizer). |
| `solver.py` | CP-SAT: one type per matrix under a RAM budget minimising a per-matrix cost; the top k distinct solutions. The genetic search uses it to propose genomes from its fitted sensitivities. |

Records of every run are JSON files in `data/results/<model>/` (label, how it was run, metrics, per-question results).
