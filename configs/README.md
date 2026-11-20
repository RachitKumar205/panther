# configs

| File | What |
|---|---|
| `models.yaml` | Every model: Hugging Face repo and pinned revision (optional `imatrix_process_output: true` lets the output layer take the 4 types that need importance data); the llama.cpp commit the server image is built at (`./panther build`) |
| `model_configs/` | Each model's `config.json` (layers, widths, heads), vendored so the RAM model and the matrices are known without a download |
| `search.yaml` | The RAM model's settings (4,096-token context, 8-bit KV cache, 0.40 GiB overhead); the 14 types a matrix may take and the 4 that need importance-matrix data; `contests`: the budgets and the models planned at each, which the dashboard lists (`./panther run` does not check it) |

A model is named on the command line by its size (`14b`, the second part of its name), so two models of one size
cannot both be listed.
