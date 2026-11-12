# configs

| File | What |
|---|---|
| `models.yaml` | Every model: Hugging Face repo and pinned revision, and the llama.cpp commit everything is built with |
| `model_configs/` | Each model's `config.json` (layers, widths, heads), vendored so shapes are known without a download |
| `search.yaml` | Budgets and the models searched at each (`contests`); the RAM model (4,096-token context, 8-bit KV cache, runtime overhead); the quantization types a matrix may take and which need importance-matrix data |

A model's types for a budget are those from 1.5 bits/weight below to 2.5 above the average the budget allows.
