# 0023 — QAT is deferred

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** after distillation (algorithm 09+)

## Context

EfficientQAT (block-wise weight adjustment trained on a large text set) was a past experiment in the
old repo, removed on 20 Nov 2026. It is the "extra mile" after distillation, not a dependency of the
core chain.

## Decision

Defer QAT until the core chain — GPTQ → GA → distillation → verdict — works end to end. Its training
data is a separate, large text set (not the train/select/test splits). We derive the block-wise
algorithm then, from EfficientQAT's paper, if it is still wanted.

## Consequences

No QAT code now; the roadmap is GPTQ → GA → distill → verdict first.

## Links

- ADR 0013, ADR 0020
