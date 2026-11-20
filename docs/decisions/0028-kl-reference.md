# 0028 — KL reference: top-256 log-probs

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** refs (score's KL)

## Context

The score's KL needs the original's next-token distribution. Storing the full distribution is tens
of GB per split; storing too few tokens loses the tail.

## Decision

Store the original's **top-256 log-probs per token** (fp16). The top 256 tokens carry ~all
probability mass, so the KL is accurate to a tiny tail error. The score's KL is the (unrenormalized)
top-k KL; distillation still uses top-64 renormalized (its own objective).

## Links

- `panther_quant/reference.py`
- ADR 0022 (fitness), ADR 0025 (distillation)
