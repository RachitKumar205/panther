"""Distillation of a finished file (the original as teacher, KL on its top 64): only the block scales (one multiplier
per 256 weights) and norms are trained, and written into a copy: same file size and speed."""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

import numpy as np

from panther.model import RULES, digest, made, sha, stamp

# byte offsets of the fp16 block scale d (and dmin, for types with a minimum) inside one block
D_OFF = {"Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,), "Q8_0": (0,),
         "IQ1_S": (0,), "IQ2_XXS": (0,), "IQ2_XS": (0,), "IQ2_S": (0,), "IQ3_XXS": (0,), "IQ3_S": (0,), "IQ4_XS": (0,)}
LINEAR = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
          "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
          "mlp.down_proj": "ffn_down"}
NORM = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}
SEQ, BATCH, TOPK, STEPS = 512, 2, 64, 600
LR_SCALE, LR_NORM = 2e-3, 1e-4  # Adam's learning rates: block-scale multipliers, norms


def _load(path: Path):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(path.parent), gguf_file=path.name, dtype=torch.bfloat16).cuda()


def reorder(x, n_head: int):
    """llama.cpp's row order for attention q/k of llama-style models (convert_hf_to_gguf permute)."""
    return x.reshape(n_head, 2, x.shape[0] // n_head // 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)


def same_rows(w, t, n_head: int) -> bool:
    """True if the PyTorch weight, reordered, matches the file's dequantized tensor row for row."""
    from gguf.quants import dequantize
    a = reorder(w.detach().float().cpu().numpy(), n_head)
    return bool(np.allclose(a, dequantize(t.data, t.tensor_type).reshape(a.shape), atol=1e-2, rtol=1e-2))


def teacher_tops(teacher: Path, ids):
    """The teacher's top-TOPK next-word log-probabilities at every position: (values fp16, indices int32), on the CPU."""
    import torch
    m = _load(teacher).eval()
    tv = torch.empty(*ids.shape, TOPK, dtype=torch.float16)
    ti = torch.empty(*ids.shape, TOPK, dtype=torch.int32)
    with torch.no_grad():
        for b in range(0, len(ids), BATCH):
            t = torch.topk(torch.log_softmax(m(ids[b:b + BATCH].cuda()).logits.float(), -1), TOPK, -1)
            tv[b:b + BATCH], ti[b:b + BATCH] = t.values.half().cpu(), t.indices.int().cpu()
    del m
    torch.cuda.empty_cache()
    return tv, ti


def passes(n: int, batch: int, steps: int, rng) -> list:
    """steps batches of `batch` sequences of n: shuffled passes, every sequence used before any repeats."""
    order = np.concatenate([rng.permutation(n) for _ in range(-(-steps * batch // n))])
    return [order[k * batch:(k + 1) * batch] for k in range(steps)]


def distill(student: Path, teacher: Path, texts: list[Path], dst: Path, tops: Path, log=print) -> Path:
    """The final step's distillation: STEPS batches of BATCH sequences of the whole texts (every sequence once, in a
    random order, before any repeats); scales at a rate of LR_SCALE, norms at LR_NORM. The teacher's top TOPK are
    computed once for these texts and kept in tops."""
    import gguf
    import torch
    import torch.nn.utils.parametrize as P
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(teacher.parent), gguf_file=teacher.name)
    ids = torch.cat([tok(t.read_text(), return_tensors="pt").input_ids[0] for t in texts])
    ids = ids[: len(ids) // SEQ * SEQ].view(-1, SEQ)  # the whole train text
    if not made(tops, ins := digest([sha(teacher), [sha(t) for t in texts], SEQ, TOPK, RULES["final"]])):
        torch.save(dict(zip("vi", teacher_tops(teacher, ids))), f"{tops}.part")
        os.rename(f"{tops}.part", tops)
        stamp(tops, ins)
    tv, ti = torch.load(tops).values()
    m = _load(student).train()  # checkpointing only applies in training mode: without it every rescaled weight
    m.gradient_checkpointing_enable()  # is kept for the backward pass, a second copy of the model (24B: 95 GB)
    sr = gguf.GGUFReader(student)
    arch = sr.fields["general.architecture"].contents()
    heads = {g: sr.fields[f"{arch}.attention.{k}"].contents() for g, k in (("attn_q", "head_count"),
                                                                           ("attn_k", "head_count_kv"))
             if f"{arch}.attention.{k}" in sr.fields}
    rd, scales, norms = {t.name: t for t in sr.tensors}, {}, {}
    m.requires_grad_(False)
    for i, layer in enumerate(m.model.layers):
        mods = dict(layer.named_modules())
        for hf, g in LINEAR.items():
            name, lin = f"blk.{i}.{g}.weight", mods[hf]
            if rd[name].tensor_type.name not in D_OFF or (arch == "llama" and g in heads and not same_rows(
                    lin.weight, rd[name], heads[g])):
                continue  # no block scale to tune (or rows that cannot be matched to the file's)
            th = torch.zeros(lin.weight.shape[0], lin.weight.shape[1] // 256, device="cuda", requires_grad=True)

            class Scale(torch.nn.Module):
                def forward(self, w, th=th):  # one multiplier per 256 weights; multiplied in fp32, as the save does
                    # (a bf16 multiplier moves in steps of 0.4-0.8 % near 1: not the scale that is written)
                    return (w.float().view(w.shape[0], -1, 256) * torch.exp(th)[:, :, None]).view(w.shape).to(w.dtype)
            P.register_parametrization(lin, "weight", Scale())
            scales[name] = th
        norms |= {f"blk.{i}.{g}.weight": mods[hf].weight for hf, g in NORM.items()}
    norms["output_norm.weight"] = m.model.norm.weight
    for w in norms.values():
        w.requires_grad_(True)
    opt = torch.optim.Adam([{"params": list(scales.values()), "lr": LR_SCALE},
                            {"params": list(norms.values()), "lr": LR_NORM}])
    for u, b in enumerate(passes(len(ids), BATCH, STEPS, np.random.default_rng(0)), 1):
        b = torch.from_numpy(b)
        logp = torch.log_softmax(m(ids[b].cuda()).logits.float(), -1)
        pt = torch.softmax(tv[b].cuda().float(), -1)  # the teacher's top 64, renormalized
        loss = (pt * (torch.log(pt + 1e-12) - torch.gather(logp, -1, ti[b].cuda().long()))).sum(-1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if u % (STEPS // 12) == 0 or u == STEPS:
            log(f"distill update {u}/{STEPS}: KL on the teacher's top {TOPK} {loss.item():.4f}")
    shutil.copy(student, f"{dst}.part")
    rw = gguf.GGUFReader(f"{dst}.part", "r+")
    out = {t.name: t for t in rw.tensors}
    for name, th in scales.items():  # the new scale of each block: d (and dmin) x exp(theta)
        t, s = out[name], torch.exp(th).detach().cpu().numpy()
        g = name.split(".")[2]
        if arch == "llama" and g in heads:  # into the file's row order (llama.cpp reorders q and k rows)
            s = reorder(s, heads[g])
        bw, bb = gguf.GGML_QUANT_SIZES[t.tensor_type]
        blocks = t.data.reshape(s.shape[0], -1, bb)
        s = np.repeat(s, 256 // bw, axis=1)
        for off in D_OFF[t.tensor_type.name]:
            d = blocks[:, :, off:off + 2].copy().view(np.float16)[..., 0]
            blocks[:, :, off:off + 2] = (d * s).astype(np.float16)[..., None].view(np.uint8)
    for name, w in norms.items():
        out[name].data[:] = w.detach().float().cpu().numpy().reshape(out[name].data.shape)
    rw.data.flush()
    del rw, out
    shutil.move(f"{dst}.part", dst)
    del m, opt, scales, norms
    gc.collect()  # the parametrized layers form reference cycles: collect them, or their GPU memory stays held
    torch.cuda.empty_cache()  # hand the GPU back: llama.cpp runs next, in another process
    return dst
