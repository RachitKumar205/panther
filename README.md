# Panther

For a RAM budget on an 8 GB Raspberry Pi 5, find the Qwen2.5 model and the quantization type for each of its weight
matrices that gives the best answers. How it works, with every number: [docs/STANDARD.md](docs/STANDARD.md).
Decisions that stand: [docs/DECISIONS.md](docs/DECISIONS.md). Server setup and rules:
[deploy/server/README.md](deploy/server/README.md).

## Layout

| Path | What |
|---|---|
| `configs/` | Pinned model revisions and llama.cpp commit, model shapes, budgets and quantization types (`search.yaml`) |
| `src/panther/model.py` | Layers, parts and matrices of a model; quantization types and their byte sizes; the RAM model |
| `src/panther/gguf_files.py` | Build a mixed GGUF by copying tensors from per-type quantized copies; check every tensor's type |
| `src/panther/solver.py` | One type per matrix under a RAM budget (CP-SAT), top-k distinct solutions that fill the budget |
| `src/panther/evaluate.py` | KL, perplexity, benchmarks and GSM8K with llama.cpp on the GPU; output parsing; run records |
| `src/panther/stats.py` | Paired bootstrap comparison of two configurations |
| `scripts/server/pipeline.py` | The pipeline steps (store, refs, error, sens, list, test, pick, bundle) |
| `scripts/server/` | Model download, quantized copies (llama-quantize) and their check |
| `panther` | The one entry point (Mac): `run` and `status` per budget |
| `deploy/server/` | Container image, job runner, `run-budget.sh` (what `./panther run` starts on the server), host setup/uninstall |
| `pi/` | Pi setup, the Pi check (`panther_pi.py`), copying a winner to the Pi (`fetch_bundle.sh`, `relay.sh`) |

## Run

One command, on the Mac:

```bash
./panther run 7       # sync the code, run the 7 GiB budget on the server, check each model's winner on the Pi
./panther status 7    # the latest progress lines and what every lane is doing
./panther watch 7     # follow the progress live
uv sync && uv run pytest
```
