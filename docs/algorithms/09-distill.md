# 09 — Distillation (block-scale multipliers)

**Status:** derivation (core parametrization + loss implemented)

## Objective

Improve a quantized file without changing its codes: train a small set of coefficients that scale
whole blocks of weights, against the original as teacher.

## Parametrization

For each layer matrix, one multiplier `θ` per **256 weights of a row**; the effective weight is
`W_eff = W · exp(θ)` (computed in fp32, ADR 0019). The norm weights are also trained directly. The
codes never change.

## Loss

`L = mean_t Σ_v∈top64 p_t(v) · (log p_t(v) − log p_s(v))`, where the teacher's top-64 next words and
their probabilities are stored, `p_t` renormalized over the 64, and `p_s` the student's full-vocab
softmax gathered at those indices.

## Data

The train split only (chat answers + web text), in 512-token sequences, each used once in a seeded
order.

## Write-back

Every K/IQ type stores one 16-bit scale `d` per block and the block's values are proportional to it,
so `d ← d · exp(θ)` at the block's byte offset (Q2_K/Q4_K/Q5_K also scale `dmin`; Q8_0's 8 sub-blocks
all take the factor; IQ1_M has no single scale and is left alone). Norms are overwritten.

## Verification

- The parametrization scales each 256-block correctly.
- `kl_topk_loss` is 0 when the student matches the teacher, positive otherwise.
- Full-loop parity with the old pipeline.
