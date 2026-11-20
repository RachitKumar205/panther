# 01 — Dataset and splits

**Status:** derivation (implementation follows)

## Objective

Partition the raw sources into splits so that a measurement on any split is never contaminated by
data from another. Two dimensions are split, each pairwise disjoint:

| dimension | splits |
|---|---|
| text (chat prompts + web documents) | train, select, test |
| benchmark questions | selection, held-out |

Every downstream number depends on these invariants:

- `train` → the store (importance matrix), rounding, distillation
- `select` → the original's references, the search's KL, the preset choice, the guard
- `test` → the held-out verdict, read exactly once

## Sources (pinned revisions)

- **Chat prompts:** OpenAssistant `oasst1` — English, first messages (`role=prompter`, no parent),
  not deleted, review-approved, 20–1000 chars.
- **Web:** FineWeb-Edu documents, disjoint by document id.
- **Benchmarks:** MMLU, ARC-Challenge, HellaSwag, WinoGrande (multiple choice) and GSM8K
  (generative, few-shot).

## Invariants (the checker proves these)

1. No web document is in two text splits (disjointness is by document id).
2. No benchmark question is in both selection and held-out (by canonical question key).
3. No chat prompt shares 8 consecutive words with any benchmark question (dropped before splitting).
4. Every measured file is in the manifest with its SHA-256.
5. The manifest covers exactly the files present.

## The overlap check (n-gram intersection)

A prompt leaks a benchmark question if any run of `n` words of the prompt appears in the question.
Tokenize to lowercase alphanumeric runs; then:

- For every prompt, insert each of its n-grams into a map `gram → prompt indices`.
- For every question text, look up its n-grams and collect the prompt indices.
- A prompt shorter than `n` words can only leak if its whole token sequence appears verbatim inside
  a question; handle it as a substring match on the padded sequence.

Complexity: `O((P + T) · ℓ)` where `P`, `T` are the numbers of prompts and texts and `ℓ` the mean
token length; lookups are hash-map `O(1)`.

## Web-vs-question leakage (V5)

A web document can contain a benchmark question (these datasets are scraped from the web). Such a
document in the train text would leak the question into the importance matrix, rounding and
distillation, contaminating the verdict even though the document sits in a disjoint split.

A document is **dropped** if it contains `>= 12` consecutive tokens of any question stem, or the
whole stem when the stem is shorter. 12 is high enough to ignore boilerplate ("which of the
following …") but specific enough to catch genuine copies. Detection is one n-gram index over all
stems, then one pass over the documents — `O(total tokens)`. Drops are counted and returned, and the
integrity checker re-proves that no kept document matches any question.

## Deterministic partition

- Chat prompts: shuffle with a seeded PRNG (Fisher–Yates via `random.Random(seed).shuffle`), then
  cut into `train`/`select`/`test` in order by the fixed counts.
- Questions: keep each benchmark in a canonical order (sort by id), then `selection = first k`,
  `held-out = rest`, for the fixed per-task counts.
- Web: from one pool, assign whole documents disjointly by id — filling train (300k) then test
  (150k) then select (<= 100,352) in a seeded order; the select pool is reconciled with the chat
  answers later (`select_text`).

## Token budgeting (select text)

The select split is read by every genome, so it is kept small: whole web documents are added in a
seeded random order until the next would exceed the budget, so no document is ever truncated.

## Deduplication

Chat prompts deduplicate by normalized text; questions deduplicate by their canonical key
(`stem + choices + answer`, tokenized). Both happen before splitting so a duplicate can never
straddle two splits.

## Artifacts

`prompts.json`, `splits/text/*` + `*.ids.json`, canonical `splits/tasks/{selection,test}/*.jsonl`,
generated `splits/llama_tasks/{selection,poc_test}/*` + manifest, and one root `manifest.json`.
The llama.cpp task files are generated from the canonical records, so the integrity check can never
validate the wrong files.

## Verification

- Golden counts: 600 prompts (300/150/150); select text ≈ 100,352 tokens; per-benchmark
  selection counts (MMLU 300; ARC/HellaSwag/WinoGrande 200 each).
- Adversarial fixtures: a document, question or prompt planted in two splits must fail.
- Reproducibility: two builds produce byte-identical manifests.
