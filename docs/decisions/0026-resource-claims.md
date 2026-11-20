# 0026 — Resource claims: configurable caps, one planner

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** ops (claims)

## Context

The old code hardcoded `CAP_GPU = 94 GiB` and `CAP_RAM = 120 GiB` and computed the parallel-genome
count in two places (`run.sizes` and `search.ga`), which drifted (bugs B9, B16).

## Decision

- Caps are **configurable**, not literals; the default is 94 GiB GPU and 96 GiB RAM (leaving room
  for the shared services, fixing B9).
- One planner computes the claim (GPU/RAM) and the parallel-genome count; nothing re-derives it
  (B16).

## Links

- `panther_ops/claims.py`
