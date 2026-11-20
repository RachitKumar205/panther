# 0003 — uv workspace with layered packages

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M0

## Context

The old code is one `src/panther` package with a module-level dependency soup: `model.py` imports
`gguf`, `search.py` imports `numpy` and `ortools`, `rounding.py` imports `torch` and `ctypes`, and
the orchestration imports everything. Domain logic cannot be tested without the heavy dependencies.

## Decision

A `uv` workspace of layered packages:

```
panther-core → panther-data → panther-measure → panther-search
             ↘ panther-quant ↗        ↘ panther-pipeline → panther-cli
panther-ops ─────────────────────────↗
panther-dashboard ───────────────────↗
```

Dependencies point one way only; `import-linter` enforces the contracts in CI. `panther-core`
depends on no heavy runtime (no torch, transformers, HF, podman, ortools).

## Alternatives considered

- **Single package with submodules.** Rejected: no enforceable boundary, and the heavy deps leak.
- **Two packages (`panther` + `panther-cli`).** Rejected: too coarse; the domain still drags in
  torch.

## Consequences

More `pyproject.toml` files and a slightly slower first `uv sync`, in exchange for reusable,
fast-testable core logic and a clear dependency graph.

## Links

- `docs/rewrite-plan.md` §5, §6
- ADR 0010
