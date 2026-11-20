# 0007 — Keep public contracts stable

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** all

## Context

External consumers exist: the Pi downloads bundles and posts results, the dashboard reads `data/`,
and released model repos use a fixed naming scheme. Breaking these would break running systems.

## Decision

Preserve:

- Hugging Face repo names: `<user>/panther-<size>-<budget>g-v2`,
  `<user>/panther-store-<model>`, `<user>/panther-status`.
- The Pi bundle manifest fields (`file`, `sha256`, `budget_gib`, `predicted_ram_gib`, `context`,
  `kv_cache_type`).
- Dashboard endpoints `/api/state`, `/api/logs`, `/api/logs/stream` and their JSON shape.
- `data/locks/claims.json` shape.

Internal records gain a `schema_version` and a complete `inputs` digest; they may evolve as long as
old records are migrated or ignored explicitly.

## Alternatives considered

- **Redesign everything.** Rejected: breaks the Pi and dashboard.
- **Keep HF + Pi only.** Rejected: the dashboard is also a public contract.

## Consequences

Contract tests (schemas and fixtures) are required for each boundary. New fields are additive.

## Links

- `docs/rewrite-plan.md` §5.3, §9
