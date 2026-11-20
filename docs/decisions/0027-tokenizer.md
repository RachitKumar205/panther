# 0027 — Tokenizer: reuse transformers

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** refs / benchmark evaluation

## Context

Text must be tokenized for the forward pass and decoded for chat answers. The old pipeline used
`transformers.AutoTokenizer` loading the tokenizer from the GGUF. Reimplementing a BPE tokenizer
would violate ADR 0010.

## Decision

Use `transformers.AutoTokenizer.from_pretrained(dir, gguf_file=name)`, exactly as the old pipeline
did. The `transformers` import is lazy (only when a tokenizer is actually loaded).

## Links

- ADR 0010, ADR 0013
