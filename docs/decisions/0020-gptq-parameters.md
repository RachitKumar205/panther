# 0020 — GPTQ rounding parameters

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** GPTQ (algorithm 04)

## Context

Error-compensated rounding needs a calibration size, a chunk size, a damping factor, and a rule for
precision across storage (the "rounding bs" — training in one precision, storing in another loses
gains).

## Decision

- **Calibration:** 65,536 tokens (128 × 512) of the train web split.
- **Chunk:** 256 columns at a time.
- **Damping:** 1% of the mean Hessian diagonal.
- **Precision:** compute the compensation in fp32, write the codes in the type's native precision,
  and verify the saved file equals what was scored.

## Links

- `docs/algorithms/04-gptq.md`
- ADR 0013, ADR 0019
