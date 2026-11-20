# 0029 — Measurements use llama.cpp (revises ADR 0013)

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** measurement
- **Revises:** ADR 0013 (the "implement KL/PPL ourselves" clause)

## Context

llama.cpp already implements KL, perplexity, multiple-choice, WinoGrande and the server completions.
Reimplementing a forward pass to measure them duplicates it and needs a parity run to trust. Rachit's
branch uses llama.cpp for measurement.

## Decision

- **Measurement** (KL, perplexity, benchmark answers) uses llama.cpp via `panther-measure`
  (`SubprocessLlama` + parsers). No own-forward-pass measurement, no parity run.
- Our `Qwen2Model` forward pass stays **only** where llama.cpp cannot help: GPTQ's per-layer
  Hessian (activations) and distillation's teacher top-k logits.

## Consequences

The measurement layer is already built; the forward pass is scoped to GPTQ + distillation.

## Links

- ADR 0013, ADR 0023, ADR 0024
