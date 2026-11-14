"""Error-compensated rounding (GPTQ) into llama.cpp's own types. A matrix is quantized 256 input columns at a time
with llama.cpp's quantizer (ggml_quantize_chunk); each chunk's error is pushed onto the columns not yet quantized,
weighted by the inverse Hessian (X^T X) of the matrix's inputs on calibration text. The codes overwrite a type copy
in place, so the format, size and speed are unchanged. Measured on Qwen2.5 7B: drift -52% Q3_K, -11% Q2_K, -9%
IQ2_XS, -8% Q4_K, IQ3_S unchanged."""

from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CHUNK, SEQS, SEQ_LEN = 256, 128, 512
THREADS = int(os.environ.get("PANTHER_THREADS", "16"))
INPUTS = {"attn_in": ("self_attn.q_proj", "attention.wqkv"), "o_in": ("self_attn.o_proj", "attention.wo"),
          "mlp_in": ("mlp.gate_proj", "feed_forward.w1"), "down_in": ("mlp.down_proj", "feed_forward.w2")}
ROLE_INPUT = {"attn_q": "attn_in", "attn_k": "attn_in", "attn_v": "attn_in", "attn_output": "o_in",
              "ffn_gate": "mlp_in", "ffn_up": "mlp_in", "ffn_down": "down_in"}
_LIB = None


def quant(w: np.ndarray, imp: np.ndarray | None, qtype, threads: int = THREADS, deq: bool = True):
    """(codes, dequantized values or None) of the rows of w in qtype, by llama.cpp's quantizer on every thread.
    imp None: llama.cpp's own weighting, as llama-quantize uses for a tensor the importance matrix lacks."""
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
    w = np.ascontiguousarray(w, np.float32)
    imp = None if imp is None else np.ascontiguousarray(imp, np.float32)
    dst = np.empty((rows, _LIB.ggml_row_size(t, n)), np.uint8)

    def part(idx):
        a, b = int(idx[0]), int(idx[-1]) + 1
        _LIB.ggml_quantize_chunk(t, w[a:b].ctypes.data, dst[a:b].ctypes.data, 0, b - a, n,
                                 None if imp is None else imp.ctypes.data)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(part, [i for i in np.array_split(np.arange(rows), threads) if len(i)]))
    return dst, dequantize(dst, qtype).reshape(rows, n) if deq else None


def f32(t) -> np.ndarray:
    """A tensor of the 16-bit GGUF as float32, rows x columns."""
    d = np.asarray(t.data)
    if t.tensor_type.name == "BF16":
        d = (d.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return d.astype(np.float32).reshape(t.shape[::-1].tolist())


def blank(bf16: Path, imat: Path, qtype: str, ftype: str, emb: str, head: str, dst: Path) -> None:
    """A type copy written directly, instead of by llama-quantize: the word table and the output layer quantized as
    llama-quantize does (importance from the importance matrix where it has them), norms and biases copied, and
    every layer matrix left as zeros of its final size for round_copies to fill. llama-quantize quantized every
    layer matrix only for round_copies to overwrite it: this skips that whole pass."""
    import gguf
    r, imx = gguf.GGUFReader(bf16), {t.name: t for t in gguf.GGUFReader(imat).tensors}
    w = gguf.GGUFWriter(f"{dst}.part", r.fields["general.architecture"].contents())
    for f in r.fields.values():
        if f.name not in ("general.architecture", "general.file_type") and not f.name.startswith("GGUF."):
            w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    w.add_file_type(int(gguf.LlamaFileType[f"MOSTLY_{ftype}"]))
    plan = []
    for t in r.tensors:
        typ = {"token_embd.weight": emb, "output.weight": head}.get(t.name)
        if t.name.startswith("blk.") and len(t.shape) == 2:
            assert t.name.split(".")[2] in ROLE_INPUT, f"{t.name}: not a matrix round_copies fills"
            typ = qtype
        if typ is None:  # norms, biases: as they are
            plan.append((t, None))
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            continue
        qt, (n, rows) = gguf.GGMLQuantizationType[typ.upper()], (int(x) for x in t.shape[:2])
        assert n % gguf.GGML_QUANT_SIZES[qt][0] == 0, f"{t.name}: {n} columns do not fit {typ}"
        shape = (rows, gguf.quant_shape_to_byte_shape((n,), qt)[0])
        plan.append((t, (qt, shape)))
        w.add_tensor_info(t.name, shape, np.dtype(np.uint8), shape[0] * shape[1], qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t, q in plan:
        if q is None:
            w.write_tensor_data(np.asarray(t.data))
        elif t.name.startswith("blk."):
            w.write_tensor_data(np.zeros(q[1], np.uint8))
        else:
            im = imx.get(f"{t.name}.in_sum2")
            imp = None if im is None else np.asarray(im.data, np.float32).reshape(-1) / float(
                np.asarray(imx[f"{t.name}.counts"].data).reshape(-1)[0])
            w.write_tensor_data(quant(f32(t), imp, q[0], deq=False)[0])
    w.close()
    os.rename(f"{dst}.part", dst)


def gptq(w, h, qtype, threads: int = THREADS) -> np.ndarray:
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
        c, q = quant(w[:, a:b].cpu().numpy(), imp[a:b].cpu().numpy(), qtype, threads)
        codes.append(c)
        err = torch.linalg.solve_triangular(u[a:b, a:b], w[:, a:b] - torch.from_numpy(q).to(w.device), upper=True,
                                            left=False)
        w[:, b:] -= err @ u[a:b, b:]
    return np.concatenate(codes, axis=1)


def round_copies(hf_dir: Path | None, bf16: Path, copies: dict[str, Path], calib: Path, layers_per_pass: int) -> None:
    """Rewrites every matrix of every copy {type: path} with error-compensated codes. Inputs come from the HF
    model on calibration text; weights from the 16-bit GGUF (llama.cpp may reorder rows, never input columns)."""
    import gguf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src_args = ({"pretrained_model_name_or_path": str(hf_dir), "trust_remote_code": True} if hf_dir else
                {"pretrained_model_name_or_path": str(bf16.parent), "gguf_file": bf16.name})  # no HF files: the GGUF
    tok = AutoTokenizer.from_pretrained(**src_args)
    ids = tok(calib.read_text(), return_tensors="pt").input_ids[0][: SEQS * SEQ_LEN].view(SEQS, SEQ_LEN)
    m = AutoModelForCausalLM.from_pretrained(**src_args, dtype=torch.bfloat16).cuda().eval()
    e = torch.eye(2, device="cuda")  # GPU linear algebra set up once, before threads race to do it
    torch.linalg.solve_triangular(torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(e))), e, upper=False)
    src = {t.name: t for t in gguf.GGUFReader(bf16).tensors}
    out = {t: gguf.GGUFReader(p, "r+") for t, p in copies.items()}
    dst = {t: {x.name: x for x in r.tensors} for t, r in out.items()}
    n_layers = len(m.model.layers)
    widest = max(t.shape[0] for t in src.values() if len(t.shape) == 2)  # input width of the widest matrix
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
                hm = h / (SEQS // 8)
                torch.cuda.empty_cache()  # types rounded at once: as many as the free GPU memory holds (each ~24 n^2)
                par = int(max(1, min(6, (torch.cuda.mem_get_info()[0] - 4 * 2**30) // (24 * widest**2))))

                def one(ts, w=w, hm=hm, name=name):  # several types at once: most of each call runs on one thread
                    ts[name].data.reshape(-1)[:] = gptq(w, hm, ts[name].tensor_type, max(2, THREADS // par)).reshape(-1)
                with ThreadPoolExecutor(par) as ex:
                    list(ex.map(one, dst.values()))
        del H
        torch.cuda.empty_cache()
    for r in out.values():
        r.data.flush()
