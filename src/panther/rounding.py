"""Error-compensated rounding (GPTQ) into llama.cpp's own types. A matrix is quantized 256 input columns at a time
with llama.cpp's quantizer (ggml_quantize_chunk); each chunk's error is pushed onto the columns not yet quantized,
weighted by the inverse Hessian (X^T X) of the matrix's inputs on calibration text. The codes overwrite a type copy
in place, so the format, size and speed are unchanged. Measured on Qwen2.5 7B: drift -52% Q3_K, -11% Q2_K, -9%
IQ2_XS, -8% Q4_K, IQ3_S unchanged."""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CHUNK, SEQS, SEQ_LEN = 256, 128, 512
INPUTS = {"attn_in": ("self_attn.q_proj", "attention.wqkv"), "o_in": ("self_attn.o_proj", "attention.wo"),
          "mlp_in": ("mlp.gate_proj", "feed_forward.w1"), "down_in": ("mlp.down_proj", "feed_forward.w2")}
ROLE_INPUT = {"attn_q": "attn_in", "attn_k": "attn_in", "attn_v": "attn_in", "attn_output": "o_in",
              "ffn_gate": "mlp_in", "ffn_up": "mlp_in", "ffn_down": "down_in"}
_LIB = None


def quant(w: np.ndarray, imp: np.ndarray, qtype) -> tuple[np.ndarray, np.ndarray]:
    """(codes, dequantized values) of the rows of w in qtype, by llama.cpp's quantizer on 16 threads."""
    global _LIB
    from gguf.quants import dequantize
    if _LIB is None:
        _LIB = ctypes.CDLL("/opt/llama.cpp/build-cuda/bin/libggml-base.so")
        _LIB.ggml_quantize_chunk.restype, _LIB.ggml_row_size.restype = ctypes.c_size_t, ctypes.c_size_t
        _LIB.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int64] * 3 \
            + [ctypes.c_void_p]
        _LIB.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    t, (rows, n) = int(qtype.value), w.shape
    _LIB.ggml_quantize_init(t)
    w, imp = np.ascontiguousarray(w, np.float32), np.ascontiguousarray(imp, np.float32)
    dst = np.empty((rows, _LIB.ggml_row_size(t, n)), np.uint8)

    def part(idx):
        a, b = int(idx[0]), int(idx[-1]) + 1
        _LIB.ggml_quantize_chunk(t, w[a:b].ctypes.data, dst[a:b].ctypes.data, 0, b - a, n, imp.ctypes.data)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(part, [i for i in np.array_split(np.arange(rows), 16) if len(i)]))
    return dst, dequantize(dst, qtype).reshape(rows, n)


def gptq(w, h, qtype) -> np.ndarray:
    """Codes of w (rows x input columns, on the GPU) in qtype, each 256-column chunk's error compensated."""
    import torch
    w, h, imp = w.clone(), h.clone(), torch.diag(h).clone()
    dead = imp == 0
    h[dead, dead], w[:, dead] = 1, 0
    h += 0.01 * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], device=h.device)
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = []
    for a in range(0, w.shape[1], CHUNK):
        b = a + CHUNK
        c, q = quant(w[:, a:b].cpu().numpy(), imp[a:b].cpu().numpy(), qtype)
        codes.append(c)
        err = torch.linalg.solve_triangular(u[a:b, a:b], w[:, a:b] - torch.from_numpy(q).to(w.device), upper=True,
                                            left=False)
        w[:, b:] -= err @ u[a:b, b:]
    return np.concatenate(codes, axis=1)


def round_copies(hf_dir: Path, bf16: Path, copies: dict[str, Path], calib: Path, layers_per_pass: int) -> None:
    """Rewrites every matrix of every copy {type: path} with error-compensated codes. Inputs come from the HF
    model on calibration text; weights from the 16-bit GGUF (llama.cpp may reorder rows, never input columns)."""
    import gguf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    ids = tok(calib.read_text(), return_tensors="pt").input_ids[0][: SEQS * SEQ_LEN].view(SEQS, SEQ_LEN)
    m = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    src = {t.name: t for t in gguf.GGUFReader(bf16).tensors}
    out = {t: gguf.GGUFReader(p, "r+") for t, p in copies.items()}
    dst = {t: {x.name: x for x in r.tensors} for t, r in out.items()}
    n_layers = len(m.model.layers)
    for start in range(0, n_layers, layers_per_pass):
        H, hooks = {}, []
        for i in range(start, min(start + layers_per_pass, n_layers)):
            mods = dict(m.model.layers[i].named_modules())
            for key, names in INPUTS.items():
                def hook(_, inp, k=(i, key)):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    H[k] = H.get(k, 0) + x.T @ x / x.shape[0]
                hooks.append(mods[next(n for n in names if n in mods)].register_forward_pre_hook(hook))
        with torch.no_grad():
            for b in range(0, SEQS, 8):
                m(ids[b:b + 8].cuda())
        for hk in hooks:
            hk.remove()
        for (i, key), h in H.items():
            for role, k in ROLE_INPUT.items():
                name = f"blk.{i}.{role}.weight"
                if k != key or name not in src:
                    continue
                w = torch.from_numpy((np.asarray(src[name].data).view(np.uint16).astype(np.uint32) << 16)
                                     .view(np.float32).reshape(src[name].shape[::-1].tolist())).cuda()
                for t, ts in dst.items():
                    ts[name].data.reshape(-1)[:] = gptq(w, h / (SEQS // 8), ts[name].tensor_type).reshape(-1)
        del H
        torch.cuda.empty_cache()
    for r in out.values():
        r.data.flush()
