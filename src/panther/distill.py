"""Distillation of a finished quantized file: the original is the teacher, the quantized model the student. Only
the per-block scales (one multiplier per 256 weights of a row: every K and IQ type stores one fp16 scale `d` per
block, and a block's values are linear in it) and the norm weights are trained, on the original's own answers,
to match the teacher's next-word probabilities (KL on its top 64). Whole numbers stay frozen: same file, same size,
same speed. The new scales are written into a copy of the file."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

# byte offsets of the fp16 block scale d (and dmin, for types with a minimum) inside one block
D_OFF = {"Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,), "Q8_0": (0,),
         "IQ1_S": (0,), "IQ2_XXS": (0,), "IQ2_XS": (0,), "IQ2_S": (0,), "IQ3_XXS": (0,), "IQ3_S": (0,), "IQ4_XS": (0,)}
LINEAR = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
          "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
          "mlp.down_proj": "ffn_down"}
NORM = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm"}
SEQ, BATCH, TOPK, STEPS = 512, 2, 64, 600


def _load(path: Path):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(path.parent), gguf_file=path.name, dtype=torch.bfloat16).cuda()


def distill(student: Path, teacher: Path, texts: list[Path], dst: Path, log=print) -> Path:
    import gguf
    import torch
    import torch.nn.utils.parametrize as P
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(teacher.parent), gguf_file=teacher.name)
    ids = torch.cat([tok(t.read_text(), return_tensors="pt").input_ids[0] for t in texts])
    ids = ids[: len(ids) // SEQ * SEQ].view(-1, SEQ)[:256]
    m = _load(teacher).eval()
    with torch.no_grad():  # the teacher's top-64 next-word probabilities, kept on the CPU
        tops = [torch.topk(torch.log_softmax(m(ids[b:b + BATCH].cuda()).logits.float(), -1), TOPK, -1)
                for b in range(0, len(ids), BATCH)]
        tv, ti = torch.cat([t.values.cpu() for t in tops]), torch.cat([t.indices.cpu() for t in tops])
    del m, tops
    torch.cuda.empty_cache()
    m = _load(student).train()  # checkpointing only applies in training mode: without it every rescaled weight
    m.gradient_checkpointing_enable()  # is kept for the backward pass, a second copy of the model (24B: 95 GB)
    arch = gguf.GGUFReader(student).fields["general.architecture"].contents()
    rd = {t.name: t for t in gguf.GGUFReader(student).tensors}
    scales, norms = {}, {}
    for p in m.parameters():
        p.requires_grad_(False)
    for i, layer in enumerate(m.model.layers):
        mods = dict(layer.named_modules())
        for hf, g in LINEAR.items():
            name, lin = f"blk.{i}.{g}.weight", mods[hf]
            if rd[name].tensor_type.name not in D_OFF or (arch == "llama" and g in ("attn_q", "attn_k")):
                continue  # no block scale to tune (or llama.cpp reordered the rows)
            th = torch.zeros(lin.weight.shape[0], lin.weight.shape[1] // 256, device="cuda", requires_grad=True)

            class Scale(torch.nn.Module):
                def forward(self, w, th=th):  # one multiplier per 256 weights, broadcast (no full-size copy)
                    return (w.view(w.shape[0], -1, 256) * torch.exp(th).to(w.dtype)[:, :, None]).view(w.shape)
            P.register_parametrization(lin, "weight", Scale())
            scales[name] = th
        for hf, g in NORM.items():
            norms[f"blk.{i}.{g}.weight"] = mods[hf].weight
    norms["output_norm.weight"] = m.model.norm.weight
    for w in norms.values():
        w.requires_grad_(True)
    opt = torch.optim.Adam([{"params": list(scales.values()), "lr": 2e-3},
                            {"params": list(norms.values()), "lr": 1e-4}])
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        b = torch.from_numpy(rng.choice(len(ids), BATCH, replace=False))
        logp = torch.log_softmax(m(ids[b].cuda()).logits.float(), -1)
        t_lp, t_i = tv[b].cuda(), ti[b].cuda()
        pt = torch.softmax(t_lp, -1)  # the teacher's top 64, renormalized
        loss = (pt * (torch.log(pt + 1e-12) - torch.gather(logp, -1, t_i))).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            log(f"distill step {step}: KL on the teacher's top {TOPK} {loss.item():.4f}")
    shutil.copy(student, f"{dst}.part")
    rw = gguf.GGUFReader(f"{dst}.part", "r+")
    out = {t.name: t for t in rw.tensors}
    for name, th in scales.items():  # the new scale of each block: d (and dmin) x exp(theta)
        t, s = out[name], torch.exp(th).detach().cpu().numpy()
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
    import gc
    gc.collect()  # the parametrized layers form reference cycles: collect them, or their GPU memory stays held
    torch.cuda.empty_cache()  # hand the GPU back: llama.cpp runs next, in another process
    return dst
