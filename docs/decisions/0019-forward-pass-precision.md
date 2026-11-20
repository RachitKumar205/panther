# 0019 — Forward-pass precision: bf16 with fp32 logits

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** forward pass (algorithm 05)

## Context

The old pipeline ran the transformer in `bfloat16` (weights are a bf16 GGUF), but upcast the logits
to `float32` before the softmax. The distillation scale multipliers were also computed in fp32, since
a bf16 multiplier near 1 moves in ~0.4–0.8% steps.

## Decision

- Weights and activations: `bfloat16`.
- Logits → softmax → probabilities: `float32` (`.float()` before `log_softmax`/`softmax`).
- Distillation block-scale multipliers: `float32`.

## Consequences

Matches the old numerics, so parity with llama.cpp is measured on the same precision path.

## Links

- ADR 0013 (algorithms ours, llama.cpp oracle)
- `docs/algorithms/05-forward-pass.md` (to be written)
