# Status

**Last updated:** 2026-11-20
**Current phase:** M0 — Bootstrap (code complete; waiting on the first CI run)

| Phase | Status | Notes |
|---|---|---|
| M0 Bootstrap | Done | workspace, tooling, CI, docs and fixtures in place |
| M1 `panther-core` | Done | quant/geometry/RAM/genome/stats/cache/error, all pydantic |
| M2 `panther-data` | Done | config schemas + split-integrity checker |
| M3 `panther-measure` | Done | parsers (per-chunk KL), backends, result cache |
| M4 Vertical slice | In progress | search (GA + preset), resumable contest; CLI commands next |
| M3 `panther-measure` | Planned | — |
| M4 Vertical slice | Planned | — |
| M5 `panther-quant` | Planned | — |
| M6 `panther-ops` + CLI | Planned | — |
| M7 Dashboard | Planned | — |
| M8 Pi verifier | Planned | — |
| M9 Deploy | Planned | — |
| M10 Parity + E2E | Planned | — |
| M11 Cutover | Planned | — |

## Open work in the current phase (M4)

- CLI `run`/`stop`/`status` commands and their wiring to the pipeline (with real backends in M6).
- Full vertical slice with a real (GPU) backend remains for M5/M10.

## Completed

- `lalit` branch created from `master`; old tree removed from the branch (kept on `master` and in
  `panther-old/`).
- `.gitignore` and `.containerignore` rewritten; `panther-old/` ignored (B20).
- Rewrite plan written; ADRs 0001–0011 recorded.
- uv workspace with eight packages plus `apps/panther-cli`; `uv.lock` generated.
- Tooling: ruff, pyright `strict`, pytest, hypothesis, pytest-cov, import-linter, pydantic.
- CLI bootstrap: `panther --help` and `panther version`.
- CI workflow `.github/workflows/ci.yml`.
- Reference `llama-perplexity` fixtures copied to `tests/parity/fixtures/llama/`.
- `panther-core`: `QuantType`/`SEARCH_FORMATS` (B15), `GeometrySpec`/`ModelGeometry`, `RamModel`/`RuntimeConfig`,
  `weighted_sqerr` (B3), chunk-based statistics (B4), and content-addressed `key` (B1).
- All dataclasses replaced with pydantic models (ADR 0012); every field carries a short comment.
- `panther-data`: typed config schemas (`ModelConfig`, `SearchConfig`, `ModelsConfig`) and the split-integrity
  checker (`manifest_problems`, `question_overlap`) for B2 and B19.
- `panther-measure`: `LlamaBackend` + `SubprocessLlama`/`FakeLlama` (B12, B13), parsers, `ResultCache` (B1).
- `panther-search`: genetic search (deterministic B8, non-ratcheting stop B6) and preset selection (B15).
- `panther-pipeline`: resumable step graph (`run_step`) and a resumable `run_contest` (B1 end-to-end).
- KL parser recovers per-chunk values from llama-perplexity's running average (B4).

## Verification

```
uv sync
uv run ruff check .        # passes
uv run pyright             # 0 errors
uv run lint-imports        # 1 contract kept
uv run pytest -q           # 11 passed
```
