# 0022 — GA fitness function

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** genetic search (algorithm 07)

## Context

The search's fitness had to be a "nice function involving KL and answer flips, scaled relative to
the preset", and was left open to adjustment.

## Decision

`fitness = flips% + 10 × KL`, lower is better:

- **flips%** = share of selection benchmark questions whose right/wrong differs from the original.
- **KL** = mean KL divergence on the select text.
- Reported relative to the preset: `relative = 100 × ours / preset` (100 = the preset's damage).

The KL weight is fixed at **10** (reported again at 5 and 20 for the verdict, deciding nothing).

## Consequences

Already encoded in `panther_core.stats.Score` (kl_weight 10) and `relative_score`; the scorer glue
wires it to the measurement layer.

## Links

- `panther_core/stats.py`
- ADR 0021
