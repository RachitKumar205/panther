# 0025 — Distillation parameters

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** distillation (algorithm 09)

## Context

The distillation loop needs its sequence length, batch, top-k, step count, learning rates and seed.

## Decision

Sequence 512 tokens; batch 2 sequences; teacher top-64; 600 updates (≈1.6 passes over the train
split, seeded order); learning rate 2e-3 for the block scales, 1e-4 for the norms; seed 0.

## Links

- `docs/algorithms/09-distill.md`
- ADR 0019 (fp32 scales)
