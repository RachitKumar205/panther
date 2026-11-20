# 0013 — Implement Panther's algorithms from first principles; llama.cpp is the oracle

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** all

## Context

The previous v2 pass wrapped llama.cpp's quantizer and used gguf's dequantize directly, so the
algorithms were not understood or owned. A rewrite that delegates the science cannot be reviewed
line by line.

## Decision

Implement Panther's algorithms ourselves: quantize/dequantize for the types we touch, the
transformer forward pass (KL/perplexity from our own logits), GPTQ error-compensated rounding,
the statistics (flips/KL/paired SE/CI/verdict), the genetic search, distillation, the RAM model and
the split construction. `numpy`/`torch` keep BLAS and autograd (primitives, not algorithms);
**llama.cpp is used only as a parity oracle** to cross-check our numbers. ortools keeps the solver
(ADR 0014).

## Alternatives considered

- Call llama.cpp for KL/quantization and implement only the search/statistics. Rejected: too much
  of the science stays hidden.
- Reimplement llama.cpp itself. Rejected: it would change what we measure.

## Consequences

Each algorithm needs a derivation (ADR 0015) and its own verification, which is slower but makes
the code reviewable line by line.

## Links

- `docs/rewrite-plan.md` §10
- ADR 0014, ADR 0015
