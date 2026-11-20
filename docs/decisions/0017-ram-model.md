# 0017 — RAM model: KV cache, repacking, byte-accounting source

- **Status:** accepted
- **Date:** 2026-11-20
- **Phase:** geometry & RAM (algorithm 02)

## Context

The RAM model is the single source of truth for the search window, the claim and the dashboard.
Three parameters had to be fixed: the KV-cache configuration, whether to account for llama.cpp's
ARM weight repacking, and where the per-type byte layout comes from.

## Decision

- **KV cache:** Q8_0, 4096-token context, 0.40 GiB fixed overhead (the values in `configs/search.yaml`).
- **Repacking:** predict a single copy of the weights; the Pi runs the winner with `--no-repack`, so
  the ARM repacked second copy is not part of the prediction.
- **Byte accounting:** read block/type sizes from `gguf`'s static `GGML_QUANT_SIZES` table
  (authoritative spec data) and derive bits/bytes from it. The quantize/dequantize *math* is
  implemented ourselves in algorithm 03.

## Alternatives considered

- Model both weight copies. Rejected: the measurement target is `--no-repack`.
- Re-specify the 14 type layouts by hand. Rejected: duplicating spec constants invites drift.

## Links

- `docs/algorithms/02-geometry-ram.md`
- ADR 0013 (algorithms from first principles), ADR 0010 (reuse libraries)
