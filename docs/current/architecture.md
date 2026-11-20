# Architecture

> Target architecture. Implementation starts in M0/M1; this file is updated as packages land.

## Package graph

```
panther-core        pure domain; no torch/transformers/HF/podman/ortools/HTTP
panther-data        → core    configs, splits, prompts, integrity
panther-measure     → core, data    llama.cpp backend, parsers, result cache
panther-quant       → core, data    store, imatrix, GPTQ rounding, distillation
panther-search      → core    preset, genetic search, CP-SAT
panther-pipeline    → core, data, measure, quant, search    step graph, verdict, bundle
panther-ops         → core    container runtime, claims, locks, guard
panther-dashboard   → core, data    state assembly, masking, HTTP/SSE
panther-cli         → ops, pipeline    the `panther` entry point
```

Dependency arrows point one way only and are enforced by `import-linter` in CI.

## Protocols (seams)

| Protocol | Real | Fake |
|---|---|---|
| `LlamaBackend` | subprocess `llama-perplexity`/`llama-server` | fixture backend |
| `StoreProvider` | gguf/torch store | tmp fixture store |
| `Distiller` | torch distillation | no-op |
| `HuggingFaceRepo` | `huggingface_hub` | local directory |
| `ContainerRuntime` | rootless podman | recording fake |
| `Clock` | system UTC/IST | frozen clock |

All boundaries exchange typed pydantic models, never bare dicts.

## Data flow

```
configs ──▶ data ──▶ measure ──▶ search ──▶ pipeline ──▶ bundle ──▶ HF
                │                                   ▲
                └──────────── quant ─────────────────┘
```

See [pipeline.md](pipeline.md) for the steps and the split each reads.
