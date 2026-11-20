# 0015 — Derivation before implementation

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** all

## Context

"Implement from first principles" is unverifiable without a written derivation to review against.

## Decision

Every algorithm gets a `docs/algorithms/NN-name.md` **before** its code: objective, the math,
assumptions, complexity, and failure modes. The derivation is part of the definition of done for
that component.

## Alternatives considered

- Derive and code together. Rejected: the math is not reviewable independently.
- Code-only with comments. Rejected: too easy to skip the reasoning.

## Consequences

Slower, but each line of code can be traced to a formula.

## Links

- ADR 0013
