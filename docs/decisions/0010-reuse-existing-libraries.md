# 0010 — Reuse existing libraries; new code only for Panther-specific logic

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** all

## Context

A rewrite invites re-inventing wheels: CLI parsing, config validation, HTTP, locking, hashing,
GGUF parsing, optimization. Every re-implementation is a new source of bugs and maintenance.

## Decision

Use the standard library or a maintained package for any functionality that is not specific to
Panther's science. New code is reserved for:

- the domain model (geometry, RAM model, genome, score, statistics),
- the pipeline (steps, cache keys, verdict),
- Panther-specific adapters (llama.cpp output parsing, GPTQ orchestration, bundle manifests).

Approved dependencies are listed in `docs/rewrite-plan.md` §10. A new non-trivial dependency
requires an ADR.

## Alternatives considered

- **Hand-roll a minimal stack.** Rejected: duplicates well-tested code and slows delivery.

## Consequences

Fewer lines of bespoke code, easier onboarding, upstream bug fixes for free. We must track
dependency versions (uv lock) and keep adapters thin.

## Links

- `docs/rewrite-plan.md` §3, §10
