# Status

**Last updated:** 2026-11-20
**Current phase:** M0 — Bootstrap (code complete; waiting on the first CI run)

| Phase | Status | Notes |
|---|---|---|
| M0 Bootstrap | Done | workspace, tooling, CI, docs and fixtures in place |
| M1 `panther-core` | Done | quant/geometry/RAM/genome/stats/cache/error, all pydantic |
| M2 `panther-data` | In progress | config schemas + integrity checker landed |
| M3 `panther-measure` | Planned | — |
| M4 Vertical slice | Planned | — |
| M5 `panther-quant` | Planned | — |
| M6 `panther-ops` + CLI | Planned | — |
| M7 Dashboard | Planned | — |
| M8 Pi verifier | Planned | — |
| M9 Deploy | Planned | — |
| M10 Parity + E2E | Planned | — |
| M11 Cutover | Planned | — |

## Open work in the current phase (M2)

- Prompt builder (OpenAssistant) and split assignment.
- Wire the integrity checker to the measured `llama_tasks` files (done in the `check` step, M4).

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

## Verification

```
uv sync
uv run ruff check .        # passes
uv run pyright             # 0 errors
uv run lint-imports        # 1 contract kept
uv run pytest -q           # 11 passed
```
