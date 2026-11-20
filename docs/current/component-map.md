# Component map

Authoritative inventory of every part of the system. Each component lists its responsibility,
interface, inputs/outputs, dependencies and the bug(s) it fixes. Components are refined one at a
time, in the dependency order below; a component is "done" only when it is typed, field-commented
and tested.

## System overview

Panther finds the best mixed-precision llama.cpp GGUF for a (model, RAM budget) contest. One run is:
config → data → store → measurement → search → final verdict → bundle → publish, with the Raspberry
Pi verifying the winner afterwards.

```
configs ─▶ data ─▶ measure ─▶ search ─▶ pipeline ─▶ bundle ─▶ HF
                │                                ▲
                └──────────── quant ──────────────┘
```

Every external dependency (llama.cpp, Hugging Face, podman, the clock) sits behind a typed
protocol, so the pipeline runs deterministically on fakes.

## Implementation order

| # | Component | Package | Depends on |
|---|---|---|---|
| 1 | quant | core | — |
| 2 | cache (`key`) | core | — |
| 3 | geometry | core | quant |
| 4 | ram | core | geometry, quant |
| 5 | stats | core | — |
| 6 | error | core | — |
| 7 | genome | core | geometry, ram, quant |
| 8 | config | data | core |
| 9 | integrity | data | — |
| 10 | parse | measure | — |
| 11 | backend | measure | parse |
| 12 | measure cache | measure | core.cache |
| 13 | preset | search | — |
| 14 | ga | search | core.genome, core.stats |
| 15 | steps | pipeline | — |
| 16 | contest | pipeline | core, search, steps |
| 17 | cli | cli | pipeline, ops |
| 18 | store | quant | core, data |
| 19 | rounding | quant | core, measure |
| 20 | distillation | quant | core |
| 21 | scoring | pipeline | measure, core.stats |
| 22 | refs | pipeline | measure |
| 23 | final | pipeline | search, measure, core.stats |
| 24 | bundle | pipeline | — |
| 25 | claims | ops | core |
| 26 | containers | ops | core |
| 27 | guard | ops | — |
| 28 | dashboard | dashboard | core, data |
| 29 | pi | pi | — |
| 30 | deploy | deploy | — |

## Cross-cutting rules

- **pydantic everywhere** (ADR 0012): every data structure is a frozen `BaseModel`; every field has
  a short comment.
- **Reuse existing libraries** (ADR 0010): stdlib and maintained packages over hand-rolled code.
- **Content-addressed cache** (B1): a result is reusable only if its digest hashes every input.
- **Atomic writes**: records are written via a part-file + `os.replace`.
- **Determinism** (B8): all randomness flows through an injected `random.Random`.

## Components

### 1. quant — `panther_core/quant.py`

- **Responsibility:** the 14 quantization types and exact byte accounting.
- **Interface:** `QuantType` (StrEnum with `block`, `type_size`, `bits_per_weight`, `nbytes`),
  `SEARCH_FORMATS`, `IMATRIX_FORMATS`.
- **Inputs:** type name, rows, cols. **Outputs:** bytes, bits per weight.
- **Dependencies:** `gguf`.
- **Bugs:** B15 (Q2_K's real bits per weight, not the old 3.2).

### 2. cache — `panther_core/cache.py`

- **Responsibility:** a deterministic content hash over arbitrary inputs.
- **Interface:** `key(*parts) -> str`.
- **Inputs:** any mix of scalars, mappings, sequences, pydantic models, enums, paths, bytes.
  **Outputs:** a 16-hex-char digest.
- **Dependencies:** `pydantic`, stdlib `hashlib`/`json`.
- **Bugs:** B1 (the hash that makes every other cache key complete).

### 3. geometry — `panther_core/geometry.py`

- **Responsibility:** the model's weight matrices, parameter count, and fixed (non-matrix) bytes.
- **Interface:** `Matrix`, `GeometrySpec`, `ModelGeometry.from_spec`, `ModelGeometry.allowed`.
- **Inputs:** a `GeometrySpec`. **Outputs:** matrices, params, fixed bytes, per-matrix allowed types.
- **Dependencies:** quant.
- **Bugs:** — (pure).

### 4. ram — `panther_core/ram.py`

- **Responsibility:** predicted peak RAM = fixed bytes + matrices + KV cache + overhead.
- **Interface:** `RuntimeConfig`, `RamModel` (`kv_bytes`, `runtime_bytes`, `matrix_bytes`,
  `total_bytes`), `KV_BYTES_PER_ELEMENT`, `GIB`.
- **Inputs:** geometry, runtime config, an assignment. **Outputs:** bytes.
- **Dependencies:** geometry, quant.
- **Bugs:** — (single source of truth for the RAM model).

### 5. stats — `panther_core/stats.py`

- **Responsibility:** the score, its standard error, and the verdict.
- **Interface:** `Verdict`, `accuracy`, `flips`, `paired_flip_std_error`, `kl_mean`,
  `kl_std_error`, `kl_diff_std_error`, `Score`, `relative_score`, `relative_interval`, `verdict`.
- **Inputs:** per-task correctness, per-chunk KL, the original's correctness. **Outputs:** percent
  scores, standard errors, WIN/TIE/LOSE.
- **Dependencies:** `numpy`.
- **Bugs:** B4 (chunk-level, paired SE instead of token-independent / independent-file).

### 6. error — `panther_core/error.py`

- **Responsibility:** offline quantization error, weighted per input column.
- **Interface:** `weighted_sqerr(w, wq, importance) -> float`.
- **Inputs:** original and quantized matrices `(rows, cols)`, importance `(cols,)`. **Outputs:** error.
- **Dependencies:** `numpy`.
- **Bugs:** B3 (correct axis and per-input weighting, not a truncated importance vector).

### 7. genome — `panther_core/genome.py`

- **Responsibility:** one type per matrix inside a RAM window, with the search operators.
- **Interface:** `Assignment`, `RamWindow`, `Genome` (`ram`, `fits`, `uniform`, `repair`,
  `mutate`, `crossover`).
- **Inputs:** geometry, ram model, allowed types, window, offline error, an RNG. **Outputs:**
  assignments.
- **Dependencies:** geometry, ram, quant.
- **Bugs:** B8 (all randomness via an injected `random.Random`).

### 8. config — `panther_data/config.py`

- **Responsibility:** validate models/search YAML and map them onto the core types.
- **Interface:** `ModelConfig`, `ModelEntry`, `LlamaCpp`, `ModelsConfig`, `ContestSpec`,
  `SearchConfig`, `load_search_config`, `load_models`, `load_model_config`.
- **Inputs:** `configs/*.yaml`, `configs/model_configs/*.json`. **Outputs:** `GeometrySpec`,
  `RuntimeConfig`, formats.
- **Dependencies:** core.
- **Bugs:** — (typed, validated config instead of loose dicts).

### 9. integrity — `panther_data/integrity.py`

- **Responsibility:** prove the measured files are the manifests' files and share nothing.
- **Interface:** `sha256_file`, `manifest_problems`, `question_overlap`.
- **Inputs:** a manifest, a directory, the measured question sets. **Outputs:** problem strings.
- **Dependencies:** stdlib.
- **Bugs:** B2 (check the measured files), B19 (manifest must cover exactly the files present).

### 10. parse — `panther_measure/parse.py`

- **Responsibility:** parse llama-perplexity output.
- **Interface:** `KlMetrics`, `ParseError`, `parse_kl`, `parse_ppl`, `parse_choice`,
  `parse_winogrande`.
- **Inputs:** raw output text. **Outputs:** typed metrics / correctness arrays.
- **Dependencies:** `numpy`, `pydantic`.
- **Bugs:** B4 (recover per-chunk KL from the running average via `_unchunk`).

### 11. backend — `panther_measure/backend.py`

- **Responsibility:** the llama.cpp seam, real and fake.
- **Interface:** `LlamaBackend` (protocol), `LlamaSettings`, `FakeLlama`, `SubprocessLlama`.
- **Inputs:** a GGUF file, reference/text/task files. **Outputs:** KL, perplexity, correctness,
  completions.
- **Dependencies:** parse.
- **Bugs:** B12 (port-race retry), B13 (one `threads` source, not the container CPU cap).

### 12. measure cache — `panther_measure/cache.py`

- **Responsibility:** cache a computation by its complete input digest.
- **Interface:** `ResultCache.get_or_run(*inputs, compute) -> (value, hit)`.
- **Dependencies:** core.cache.
- **Bugs:** B1 (never reuse on a partial match).

### 13. preset — `panther_search/preset.py`

- **Responsibility:** llama.cpp's presets and their nominal bits per weight.
- **Interface:** `PRESETS`, `presets_that_fit`, `choose_preset`.
- **Dependencies:** —.
- **Bugs:** B15 (corrected Q2_K).

### 14. ga — `panther_search/ga.py`

- **Responsibility:** the steady-state genetic search.
- **Interface:** `Evaluation`, `SearchParams`, `improvement`, `search`.
- **Inputs:** a `Genome`, a scorer, the original's correctness, the KL weight, params.
  **Outputs:** the top `Evaluation`s.
- **Dependencies:** core.genome, core.stats.
- **Bugs:** B6 (non-ratcheting stop), B8 (deterministic order). B5 (first look) and B7 (solver)
  land here as optimisations.

### 15. steps — `panther_pipeline/steps.py`

- **Responsibility:** a resumable step: run only if the output's digest differs.
- **Interface:** `write_atomic`, `run_step`.
- **Dependencies:** stdlib.
- **Bugs:** B1 (end-to-end resumability).

### 16. contest — `panther_pipeline/contest.py`

- **Responsibility:** the ordered, resumable steps that produce a winner.
- **Interface:** `Contest`, `Winner`, `genome_key`, `run_contest`.
- **Dependencies:** core, search, steps.
- **Bugs:** B1 (complete `genome_key`).

### 17. cli — `panther_cli/main.py`

- **Responsibility:** the `panther` entry point.
- **Interface:** `app` (Typer), `version`, later `run`/`stop`/`status`.
- **Dependencies:** pipeline, ops (for real runs).
- **Bugs:** —.

### 18. store — `panther_quant` (M5)

- **Responsibility:** the 16-bit original, importance matrix, 14 type copies; backup/restore.
- **Bugs:** B10 (disk preflight), B18 (typed byte/GPU planner).

### 19. rounding — `panther_quant` (M5)

- **Responsibility:** error-compensated (GPTQ) codes via `ggml_quantize_chunk`, in place.
- **Bugs:** B3 (correct importance), B11 (24B peak).

### 20. distillation — `panther_quant` (M5)

- **Responsibility:** train per-block scales and norms, write them back into a copy.
- **Bugs:** —.

### 21. scoring — `panther_pipeline` (M6)

- **Responsibility:** turn backend results into an `Evaluation` (flips + KL + SEs).
- **Bugs:** B4 (paired SE), B17 (GSM8K cache key).

### 22. refs — `panther_pipeline` (M6)

- **Responsibility:** the original's chat answers, reference probabilities, correctness.
- **Bugs:** B1 (reference hashes in keys).

### 23. final — `panther_pipeline` (M6)

- **Responsibility:** finalists distilled/not, winner, guard, held-out verdict.
- **Bugs:** B4 (verdict interval).

### 24. bundle — `panther_pipeline` (M6)

- **Responsibility:** the winner's file + manifest for the Pi.
- **Bugs:** — (versioned manifest).

### 25. claims — `panther_ops` (M6)

- **Responsibility:** reserve GPU/RAM for a run; refuse when they do not fit.
- **Bugs:** B9 (configurable caps), B16 (single inflight planner).

### 26. containers — `panther_ops` (M6)

- **Responsibility:** the container-per-stage topology behind a `ContainerRuntime` protocol.
- **Bugs:** B13 (thread policy), B14 (single source of numbers).

### 27. guard — `panther_ops` (M6)

- **Responsibility:** pause/resume Panther while the shared service is unhealthy.
- **Bugs:** —.

### 28. dashboard — `panther_dashboard` (M7)

- **Responsibility:** read-only state, masking, `/api/state`, `/api/logs`, SSE.
- **Bugs:** B14 (render caps from one source).

### 29. pi — `pi/` (M8)

- **Responsibility:** verify each winner (RAM, speed) and report FITS/OVER.
- **Bugs:** —.

### 30. deploy — `deploy/` (M9)

- **Responsibility:** image, host/proxy setup, uninstall, nginx.
- **Bugs:** B20 (ignore rules).
