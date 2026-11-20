# 0021 — Preset selection rule

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** preset (algorithm 08)

## Context

The competitor is the homogeneous llama.cpp preset we must beat. The exact selection rule had
changed several times and was left open.

## Decision

Of every llama.cpp preset (IQ1_S…Q8_0) whose predicted RAM fits the budget, score each on all
selection questions and the select text; choose the **most accurate**, ties broken by **lower KL**.
If none fits, choose the **smallest** preset and record it as over budget.

## Consequences

`presets_that_fit` (bpw pre-filter) and `choose_preset` (accuracy-then-KL) already encode this; the
remaining work is wiring them to the measurement layer.

## Links

- `panther_search/preset.py`
- ADR 0022
