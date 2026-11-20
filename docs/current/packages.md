# Packages

> Status: target layout. Each package is created in M0 and fills in during its phase.

| Package | Phase | Responsibility | Heavy deps |
|---|---|---|---|
| `panther-core` | M1 | types, geometry, RAM model, genome, score/statistics, cache keys | `numpy`, `gguf`, `pydantic` |
| `panther-data` | M2 | config schemas, split manifests, prompts, integrity checks | `pydantic`, `pyyaml`, `pyarrow` |
| `panther-measure` | M3 | `LlamaBackend`, parsers, result cache, references | `numpy`, `gguf` |
| `panther-quant` | M5 | store, imatrix, GPTQ rounding, distillation | `torch`, `transformers`, `huggingface_hub`, `gguf` |
| `panther-search` | M4 | preset selection, GA, CP-SAT | `numpy`, `ortools` |
| `panther-pipeline` | M4 | step graph, final/guard/verdict, bundle | — |
| `panther-ops` | M6 | `ContainerRuntime`, claims, locks, guard | — |
| `panther-dashboard` | M7 | state assembly, masking, HTTP/SSE, frontend | `pyyaml`, `huggingface_hub` |
| `panther-cli` | M6 | Typer entry point | `typer`, `rich` |

Public API rules:

- every package exports its types from `panther_<name>/__init__.py` or a documented module;
- no package reaches into another's private modules;
- `panther-core` has no optional/heavy imports at import time.
