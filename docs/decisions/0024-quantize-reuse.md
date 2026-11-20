# 0024 — Quantize step reuses llama.cpp's quantizer (revises ADR 0018)

- **Status:** accepted
- **Date:** 2026-11-21
- **Phase:** GPTQ (algorithm 04)
- **Revises:** ADR 0018 (the "quantize block-scale ourselves" clause)

## Context

The GPTQ loop needs a quantizer per 256-column chunk. gguf provides *dequantize* but not *quantize*;
llama.cpp's `ggml_quantize_chunk` (called via ctypes, exactly as the old pipeline did) is the
existing, tested quantizer — including the K-type scale/min search and the IQ codebooks.

## Decision

Use llama.cpp's `ggml_quantize_chunk` via a thin ctypes adapter for the quantize step (all 14
types), and implement only the GPTQ compensation ourselves. This mirrors the old code and ADR 0010
("reuse libraries"); llama.cpp remains the parity oracle.

## Consequences

No hand-written K-type/IQ quantizers. The ctypes adapter is added when the real store/rounding is
wired (needs the llama.cpp `.so`, unavailable in CI).

## Links

- ADR 0010, ADR 0013, ADR 0018, ADR 0020
