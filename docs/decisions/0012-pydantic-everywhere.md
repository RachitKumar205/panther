# 0012 — Pydantic everywhere, no dataclasses

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M1/M2 (retroactive to M0)
- **Supersedes:** the dataclass clause of ADR 0004

## Context

The domain started with `@dataclass(frozen=True, slots=True)` for geometry, RAM, genomes and
scores, and pydantic only at config boundaries. That left two ways to do one thing.

## Decision

Use pydantic v2 (`BaseModel` with `ConfigDict(frozen=True)`) for **every** data structure in the
codebase. No `dataclasses`. `panther-core` therefore depends on pydantic (a small, pure library,
unlike the heavy runtime deps it still excludes).

## Alternatives considered

- Dataclasses for pure internals, pydantic at the edges (rejected: two idioms, no validation in the
  domain).
- A single hand-rolled model base (rejected: violates ADR 0010, reuse existing libraries).

## Consequences

One way to declare fields, validation, JSON serialisation and frozen value semantics everywhere.
Every field carries a short inline comment for review. The content-addressed cache serialises models
via `model_dump(mode="json")`.

## Links

- `docs/rewrite-plan.md` §10
- ADR 0004, ADR 0010
