# 0018 — Quant codecs: reuse gguf's dequantize

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** quant formats (algorithm 03)

## Context

Dequantize is a file-format codec, not a Panther algorithm; `gguf` (an existing dependency) ships a
correct, tested implementation. Reimplementing it is transcription with no scientific payoff and
directly violates ADR 0010 ("reuse libraries") and ADR 0017 ("read the byte layout from gguf").

## Decision

- **Dequantize:** use `gguf.quants.dequantize`.
- **Quantize (when GPTQ needs it):** implement the block-scale types (`Q8_0`, `Q4_0`, `Q5_0`,
  `Q2_K`…`Q6_K`) ourselves; the IQ codebook types use llama.cpp's quantizer (parity oracle, ADR 0013).

The science built *on top of* dequantized weights — offline error, GPTQ, statistics, distillation —
remains ours.

## Consequences

No codec transcription; the "from first principles" effort goes to the algorithms.

## Links

- ADR 0010, ADR 0013, ADR 0017
