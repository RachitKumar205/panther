# 0014 — Keep ortools for the solver; derive the model ourselves

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M4/M6

## Context

The genetic search proposes genomes by solving a small integer program: choose one type per matrix,
total bytes inside the window, minimizing the predicted error. Reimplementing a branch-and-bound
solver would be reinventing a wheel (ADR 0010).

## Decision

Derive the ILP ourselves — variables (one per matrix/type), the `AddExactlyOne` constraint, the
byte window as a linear constraint, and the least-error objective — and hand it to ortools' CP-SAT.

## Alternatives considered

- Hand-roll the solver. Rejected: duplicated, worse than a proven library.

## Consequences

The search's proposal *model* is ours and reviewable; the *solve* is delegated.

## Links

- ADR 0010, ADR 0013
