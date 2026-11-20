# 02 — Model geometry and the RAM model

**Status:** derivation (implementation verified against the vendored configs)

## Objective

From a model config, derive every weight matrix, the parameter count, and the predicted peak RAM
of a quantized file. This is the number the search window, the claim and the dashboard all share.

## Geometry

Given `L` layers, hidden `h`, `heads`, `kv_heads`, head-dim `d`, FFN width `i`, vocab `V`:

- `q = heads · d`, `kv = kv_heads · d`.
- Matrices: `token_embd` (`V × h`), `output` (`V × h`, only when untied), and per layer:
  `attn_q` (`q × h`), `attn_k` (`kv × h`), `attn_v` (`kv × h`), `attn_output` (`h × q`),
  `ffn_gate` (`i × h`), `ffn_up` (`i × h`), `ffn_down` (`h × i`).
- `params = Σ rows·cols`.
- `fixed = 4 · (L·(2h + bias) + h)`, in bytes, where `bias = q + 2·kv` for Qwen2 (q/k/v biases)
  and `0` otherwise. The `2h` is the two RMS-norm weights per layer; the `+h` is the final norm;
  each weight is F32 = 4 bytes.

## Quantization byte accounting

For type `t` with block size `B` and block bytes `s` (from `gguf.GGML_QUANT_SIZES`):

- `bits/weight = 8·s / B`.
- `bytes(rows, cols) = rows · (cols / B) · s`, valid only when `cols` is a multiple of `B`.

The dequantization *math* (recovering weights from the codes) is algorithm 03; here we only need
the byte sizes.

## KV cache

`KV = 2 · L · kv · T · b_elem`, where `b_elem = 34/32` for Q8_0 (32 weights → 32 bytes + a 2-byte
scale) and `2` for F16. A single copy is counted: the Pi runs `--no-repack`.

## RAM

`RAM = fixed + Σ matrix bytes + KV + overhead`, with overhead `0.40 GiB`.

## Verification

- Geometry: recompute all five vendored configs — matrix count `(1 or 2) + 7L`, hidden/layer
  counts, and the Qwen2 vs Mistral bias and head-dim differences.
- RAM: the Qwen2.5-7B at uniform Q4_K must land in a plausible band (≈4–6 GiB), consistent with
  the Pi's measured 4.69 GiB for the mixed-precision winner.
