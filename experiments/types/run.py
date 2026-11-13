"""Experiment: which storage types work, and does smarter rounding beat llama.cpp's? One model (Qwen2.5 7B).

plain: every matrix of the model in one type (llama-quantize --pure with the model's importance matrix; embedding
       and output head Q8_0 in every file, so only the matrices differ), for every type llama.cpp has.
gptq:  the same file for a few types, but each matrix rounded with error compensation (GPTQ): the matrix is
       quantized 256 columns at a time with llama.cpp's own quantizer (ggml_quantize_chunk), and each chunk's error
       is pushed onto the columns not yet quantized, weighted by the inverse Hessian of the matrix's inputs on
       calibration text. The codes are written into a copy of the plain file, so the format is untouched.
Both are scored the same way: KL from the original on the selection text and accuracy on the GA's 900 selection
questions. Results: data/rewrite/types-<model>.json, one progress line per file.

  python experiments/types/run.py Qwen2.5-7B-Instruct /work/data/hf-exp/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "server"))
import pipeline as p  # noqa: E402

PLAIN = ["Q8_0", "Q6_K", "Q5_K", "Q5_0", "Q4_K", "Q4_0", "IQ4_XS", "IQ4_NL", "MXFP4_MOE", "Q3_K", "IQ3_S",
         "IQ3_XXS", "Q2_K", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "TQ2_0"]
GPTQ = ["Q4_K", "Q3_K", "IQ3_S", "Q2_K", "IQ2_XS"]
FTYPE = {"Q4_K": "Q4_K_S", "Q3_K": "Q3_K_S", "Q5_K": "Q5_K_S", "Q2_K": "Q2_K"}  # --pure makes every matrix the base type
CHUNK, SEQS, SEQ_LEN, BATCH, THREADS = 256, 128, 512, 8, 16
LIB = ctypes.CDLL("/opt/llama.cpp/build-cuda/bin/libggml-base.so")
LIB.ggml_quantize_chunk.restype = ctypes.c_size_t
LIB.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_void_p]
LIB.ggml_row_size.restype = ctypes.c_size_t
LIB.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
LIB.ggml_quantize_init.argtypes = [ctypes.c_int]


def note(msg: str) -> None:
    p.BUDGET = 0
    p.note(f"TYPES {msg}")


def quant(w: np.ndarray, imat: np.ndarray, qtype) -> tuple[np.ndarray, np.ndarray]:
    """(codes, dequantized) for rows of w with llama.cpp's quantizer, 16 threads over the rows."""
    from gguf.quants import dequantize
    t = int(qtype.value)
    LIB.ggml_quantize_init(t)
    rows, n = w.shape
    rs = LIB.ggml_row_size(t, n)
    w = np.ascontiguousarray(w, dtype=np.float32)
    imat = np.ascontiguousarray(imat, dtype=np.float32)
    dst = np.empty((rows, rs), dtype=np.uint8)
    parts = np.array_split(np.arange(rows), THREADS)

    def run(idx):
        if len(idx):
            a, b = int(idx[0]), int(idx[-1]) + 1
            LIB.ggml_quantize_chunk(t, w[a:b].ctypes.data, dst[a:b].ctypes.data, 0, b - a, n, imat.ctypes.data)
    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(run, parts))
    return dst, dequantize(dst, qtype).reshape(rows, n)


def quantize_file(model: str, ftype: str, path: Path) -> None:
    store = p.DATA / "store" / model
    subprocess.run([str(p.evaluate.LLAMA / "build-cuda/bin/llama-quantize"), "--pure", "--imatrix",
                    str(store / f"{model}.imatrix.gguf"), "--token-embedding-type", "q8_0", "--output-tensor-type",
                    "q8_0", str(store / f"{model}-BF16.gguf"), str(path), FTYPE.get(ftype, ftype), "16"],
                   check=True, capture_output=True)


def score(model: str, path: Path, label: str) -> dict:
    import gguf
    rd = gguf.GGUFReader(path)
    mats = [t for t in rd.tensors if t.name.startswith("blk.") and len(t.shape) == 2]
    types = sorted({t.tensor_type.name for t in mats})
    bpw = 8 * sum(int(t.n_bytes) for t in mats) / sum(int(t.n_elements) for t in mats)
    recs = p.records(model)
    if not p.find(recs, label, reference="selection"):
        p.kl(model, path, "selection", label)
    for t, n in p.GA_FIT_TASKS.items():
        if not p.find(p.records(model), label, task=t, split="selection", n_tasks=n):
            p.evaluate.run("task", model, path, label, p.THREADS, task=t, split="selection", n_tasks=n)
    recs = p.records(model)
    kl = p.find(recs, label, reference="selection")["metrics"]["kl_mean"]
    accs = [p.find(recs, label, task=t, split="selection", n_tasks=n) for t, n in p.GA_FIT_TASKS.items()]
    acc = float(np.mean([a["metrics"]["accuracy_pct"] for a in accs])) if all(accs) else None
    return {"matrix_types": types, "bpw": round(bpw, 3), "kl": kl, "accuracy_900": acc}


def hessians(hf_dir: Path, layers: list[int]):
    """Per matrix input of the given layers: X^T X / tokens on calibration text, original weights (HF)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    ids = tok(p.CALIB.read_text(), return_tensors="pt").input_ids[0][: SEQS * SEQ_LEN].view(SEQS, SEQ_LEN)
    m = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16).cuda().eval()
    H, n, hooks = {}, {}, []
    for i in layers:
        L = m.model.layers[i]
        for key, mod in (("attn_in", L.self_attn.q_proj), ("o_in", L.self_attn.o_proj), ("mlp_in", L.mlp.gate_proj),
                         ("down_in", L.mlp.down_proj)):
            k = (i, key)

            def hook(_, inp, k=k):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                H[k] = H.get(k, 0) + x.T @ x
                n[k] = n.get(k, 0) + x.shape[0]
            hooks.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for b in range(0, SEQS, BATCH):
            m(ids[b:b + BATCH].cuda())
    for h in hooks:
        h.remove()
    W = {}
    for i in layers:
        L = m.model.layers[i]
        for g, mod, key in (("attn_q", L.self_attn.q_proj, "attn_in"), ("attn_k", L.self_attn.k_proj, "attn_in"),
                            ("attn_v", L.self_attn.v_proj, "attn_in"), ("attn_output", L.self_attn.o_proj, "o_in"),
                            ("ffn_gate", L.mlp.gate_proj, "mlp_in"), ("ffn_up", L.mlp.up_proj, "mlp_in"),
                            ("ffn_down", L.mlp.down_proj, "down_in")):
            W[f"blk.{i}.{g}.weight"] = (mod.weight.detach().float(), H[(i, key)] / n[(i, key)])
    del m
    torch.cuda.empty_cache()
    return W


def gptq(w, h, qtype) -> np.ndarray:
    """Codes for w (out x in) in qtype, 256 input columns at a time, each chunk's error compensated on the rest."""
    import torch
    w = w.clone()
    imp = torch.diag(h).clone()
    h = h.clone()
    dead = imp == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += 0.01 * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], device=h.device)
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    codes = []
    for a in range(0, w.shape[1], CHUNK):
        b = a + CHUNK
        c, q = quant(w[:, a:b].cpu().numpy(), imp[a:b].cpu().numpy(), qtype)
        codes.append(c)
        err = torch.linalg.solve_triangular(u[a:b, a:b], (w[:, a:b] - torch.from_numpy(q).to(w.device)),
                                            upper=True, left=False)
        w[:, b:] -= err @ u[a:b, b:]
    return np.concatenate(codes, axis=1)


def smoke(hf_dir: Path) -> None:
    """Layer 0: output error (on calibration inputs) of plain vs GPTQ rounding for each GPTQ type."""
    import gguf
    import torch
    for name, (w, h) in hessians(hf_dir, [0]).items():
        for t in GPTQ:
            qt = gguf.GGMLQuantizationType[t]
            plain = np.concatenate([quant(w[:, a:a + CHUNK].cpu().numpy(), torch.diag(h)[a:a + CHUNK].cpu().numpy(),
                                          qt)[1] for a in range(0, w.shape[1], CHUNK)], axis=1)
            from gguf.quants import dequantize
            g = torch.from_numpy(dequantize(gptq(w, h, qt), qt).reshape(w.shape)).cuda()
            e = lambda d: float(torch.trace(d @ h @ d.T) / torch.trace(w @ h @ w.T))  # noqa: E731
            note(f"smoke {name} {t}: relative output error plain {e(w - torch.from_numpy(plain).cuda()):.5f}, "
                 f"gptq {e(w - g):.5f}")


def main() -> int:
    import gguf
    import torch
    model, hf_dir = sys.argv[1], Path(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] == "smoke":
        smoke(hf_dir)
        return 0
    tmp = p.DATA / "tmp" / "types"
    tmp.mkdir(parents=True, exist_ok=True)
    res_f = p.out(f"types-{model}.json")
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    for t in PLAIN:  # every type, plain llama.cpp rounding
        key = f"plain {t}"
        path = tmp / f"plain-{t}.gguf"
        if key not in res:
            try:
                if not path.exists():
                    quantize_file(model, t, path)
                res[key] = score(model, path, f"types:plain:{t}")
            except Exception as e:  # noqa: BLE001  (a type that does not work is a result too)
                res[key] = {"error": f"{type(e).__name__}: {str(e)[-300:]}"}
            res_f.write_text(json.dumps(res, indent=1))
            note(f"{key}: {res[key]}")
        if t not in GPTQ:
            path.unlink(missing_ok=True)
    todo = [t for t in GPTQ if f"gptq {t}" not in res]
    if todo:
        rd0 = gguf.GGUFReader(p.DATA / "store" / model / f"{model}-BF16.gguf")
        n_layers = 1 + max(int(t.name.split(".")[1]) for t in rd0.tensors if t.name.startswith("blk."))
        for t in todo:
            shutil.copy(tmp / f"plain-{t}.gguf", tmp / f"gptq-{t}.gguf")
        readers = {t: gguf.GGUFReader(tmp / f"gptq-{t}.gguf", "r+") for t in todo}
        checked = False
        for start in range(0, n_layers, 7):
            for name, (w, h) in hessians(hf_dir, list(range(start, min(start + 7, n_layers)))).items():
                if not checked:  # the HF weights are the GGUF's (no permutation)
                    g = next(x for x in rd0.tensors if x.name == name)
                    ref = np.asarray(g.data).view(np.uint16).astype(np.uint32) << 16
                    assert np.allclose(ref.view(np.float32).reshape(w.shape), w.cpu().numpy()), name
                    checked = True
                for t in todo:
                    tt = next(x for x in readers[t].tensors if x.name == name)
                    codes = gptq(w, h, tt.tensor_type)
                    flat = tt.data.reshape(-1)
                    flat[:] = codes.reshape(-1)
                del w, h
            torch.cuda.empty_cache()
            note(f"gptq layers {start}-{min(start + 7, n_layers) - 1} done")
        for r in readers.values():
            r.data.flush()
        del readers
        for t in todo:
            res[f"gptq {t}"] = score(model, tmp / f"gptq-{t}.gguf", f"types:gptq:{t}")
            res_f.write_text(json.dumps(res, indent=1))
            note(f"gptq {t}: {res[f'gptq {t}']}")
    for f in tmp.glob("*.gguf"):
        f.unlink()
    note("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
