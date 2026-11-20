# 0009 — First milestone is a vertical slice on fixtures

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** M4

## Context

Building every library to completion before wiring anything delays feedback and risks discovering
integration problems late.

## Decision

After `core`, `data` and `measure` (with a fake backend), build a **vertical slice**: one contest
(`1.5b@1`) runs fetch → publish end-to-end on fixtures in a temp `data/` directory, with fake
store/llama/HF implementations. Real quantization and containers plug in behind the same protocols.

## Alternatives considered

- **Bottom-up completion.** Rejected: late integration risk.
- **Ops/CLI first.** Rejected: nothing to operate yet.

## Consequences

The step graph, cache invalidation and scoring are proven early. Real backends can be swapped in
without changing pipeline code.

## Links

- `docs/rewrite-plan.md` §7 (M4)
- ADR 0003, ADR 0006
