# 04 — Error-compensated (GPTQ) rounding

**Status:** derivation (core implemented; block-scale quantizers next)

## Objective

Quantize each layer matrix while minimizing the *output* error the model actually sees, not the
weight error. Naive rounding minimizes `‖W − Ŵ‖²`; GPTQ minimizes `E‖xW − xŴ‖²` over calibration
inputs `x`.

## The objective

`E‖x(W − Ŵ)‖² = trace((W − Ŵ) H (W − Ŵ)ᵀ)`, where `H = E[xᵀx]` is the input covariance estimated from
65,536 calibration tokens. `H` weights each column by its activation and couples the columns.

## The algorithm

1. **Damp** `H` (add 1% of the mean diagonal) so `H⁻¹` is stable; columns with zero activation are
   zeroed.
2. **`U = cholesky(H⁻¹)`**, upper triangular.
3. For each 256-column chunk, in order:
   - quantize the (already-compensated) chunk → `ŵ`,
   - compute the error `e = w_chunk − ŵ`,
   - solve `U_cc · e' = e` for `e'` (the error in the Cholesky basis), then
   - `w_rest ← w_rest − e' · U_c,rest`: push the error onto the columns not yet quantized.

The last columns absorb the earlier columns' mistakes, so the total output error stays small.

## Precision

Computed in fp32 (ADR 0019); the codes are written in the type's native precision, and the saved
file is re-checked to equal what was scored (ADR 0020).

## Verification

- On a synthetic layer with a random `H`: the compensated reconstruction error must be **≤** the
  naive per-chunk error.
- On a real layer: parity with the old pipeline.
