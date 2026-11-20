# 0016 — Dataset: sources, split sizes and question partition

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** dataset (first component)

## Context

The old split builder was deleted and its data lived on the server. The rebuild must be testable
without the network, so the split algorithm is built and tested on synthetic fixtures and the real
sources plug in behind a `DatasetSource` protocol later.

## Decision

- **Sources (pinned):** OpenAssistant `oasst1` (chat prompts), FineWeb-Edu (web), and the
  benchmarks MMLU / ARC-Challenge / HellaSwag / WinoGrande / GSM8K.
- **Text split sizes:** 300 train / 150 select / 150 test chat prompts; web token budgets
  300k train / 100,352 select / 150k test.
- **Question partition:** per benchmark, selection first — MMLU 300, ARC-Challenge 200,
  HellaSwag 200, WinoGrande 200 — the rest held out; GSM8K held out only.
- **Overlap threshold:** 8 consecutive words (prompt vs any benchmark question).
- **Seed:** 0 everywhere for reproducibility.

## Alternatives considered

- Redefine sizes from scratch. Rejected: the old sizes are the baseline the results are compared to.

## Consequences

The split algorithm is fully testable offline; the real downloaders are thin adapters.

## Links

- `docs/algorithms/01-dataset-splits.md`
