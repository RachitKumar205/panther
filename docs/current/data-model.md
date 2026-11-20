# Data model and artifacts

> Target model. Exact schemas are defined with pydantic in M1–M4 and recorded here.

## Configuration

| Model | Source | Notes |
|---|---|---|
| `ModelsConfig` | `configs/models.yaml` | pinned HF revisions; optional `imatrix_process_output` |
| `SearchConfig` | `configs/search.yaml` | context, KV type, overhead, contests, formats, `requires_imatrix` |
| `ModelConfig` | `configs/model_configs/*.json` | vendored HF geometry |

## Persisted records (all carry `schema_version` + `inputs` digest)

| Model | Written to | Purpose |
|---|---|---|
| `ErrorRecord` | `rewrite/<model>/error.json` | offline error per matrix/type |
| `PresetRecord` | `rewrite/budget-<B>/preset-<model>.json` | competitor and its scores |
| `FinalistList` | `rewrite/budget-<B>/list-<model>.json` | best genomes from the GA |
| `WinnerRecord` | `rewrite/budget-<B>/winner-<model>.json` | verdict and every held-out number |
| `BundleManifest` | `pi_bundle/.../manifest.json` | file hash, budget, predicted RAM, context, KV type |
| `Claim` | `data/locks/claims.json` | GPU/RAM reservation per active run |
| `MeasurementRecord` | `data/results/<model>/*.json` | one llama.cpp result, keyed by label + inputs |

## On-disk artifacts

```
data/
├── hf/<model>/                 original HF weights (removed after GGUF conversion)
├── splits/                     manifests, tasks/, llama_tasks/, text/
├── store/<model>/              BF16, 14 type copies, imatrix, presets, ident
├── reference/<model>/          chat answers, .kld bases
├── results/<model>/            measurement records and logs
├── rewrite/                    error/preset/list/winner records, tuned files, tmp genomes
├── pi_bundle/                  per-contest bundle
├── locks/                      claims.json, run-*.lock/pid, PAUSED
└── logs/                       panther.log, progress-b<B>.log, step logs
```

## Compatibility

Public contracts (HF names, bundle manifest fields, dashboard `/api/state`, `claims.json`) are
frozen by ADR 0007. Internal records may evolve with a schema version.
