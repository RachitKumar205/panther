"""Stage 1 of EfficientQAT (Block-AP) for a llama.cpp file: block by block, each layer matrix's weights and its
format's scales are trained so the quantized block reproduces the original block, then the codes and scales are
written into a copy of the file (same types, same size). As the paper (arXiv 2407.11062, official code):
  - target: the original block's output on the original model's own hidden states; input: the outputs of the blocks
    already quantized, so each block also corrects the error before it; loss: mean squared error
  - trained: the latent weights (starting at the original's) and the scales (starting at the file's); norms and
    biases frozen; AdamW, weights 2e-5 (2-bit), cosine to 1/20 per batch, batch 2, 2 epochs. Scales: relative
    multipliers at 1e-3, about the paper's 1e-4 on its ~0.05 step sizes (tested on the 1.5B: 1e-2 hurts)
Ours: llama.cpp's own types (panther.fakequant) instead of uniform quantization; bf16 instead of fp16 (Qwen2.5
overflows fp16); a matrix's codes are re-chosen every RECODE steps (the IQ pattern search is the slow part; the
weights move by ~1e-5 a step). The hidden states of every sample (two copies, original and quantized) do not fit in
memory at the paper's size (14B: 2 x 86 GB), so they are streamed from disk (bf16), as the paper's code does for large models (fp16 there).
"""

from __future__ import annotations

import copy
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from panther import fakequant as fq
from panther.distill import LINEAR

BATCH, EPOCHS, LR_W, LR_S, MIN_LR, RECODE = 2, 2, 2e-5, 1e-3, 1 / 20, 16  # LR_S: relative (per unit of log d)


class QLinear(torch.nn.Module):
    """A linear layer whose weight is the fake-quantized latent weight W. The file's block scales d (and dmin) are
    trained as relative multipliers, d = d0 * exp(theta): the paper's learning rate is for step sizes ~50x larger
    than llama.cpp's d, and in absolute units it wrecks them. The 4-bit sub-scales stay as in the file."""

    def __init__(self, lin: torch.nn.Linear, t: str, raw: np.ndarray):
        super().__init__()
        self.t, self.n, self.codes, self.frozen = t, 0, None, False
        self.W = torch.nn.Parameter(lin.weight.detach().float().clone())
        self.st = fq.unpack(t, raw, self.W.device)
        self.theta = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros_like(self.st[k]))
                                             for k in ("d", "dmin") if k in self.st})
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().float())

    def scales(self) -> dict:
        return {**self.st, **{k: self.st[k] * torch.exp(th) for k, th in self.theta.items()}}

    def forward(self, x):
        st = self.scales()
        if self.codes is None or (not self.frozen and self.n % RECODE == 0):
            self.codes = fq.encode(self.t, self.W, st)
        self.n += 1
        w = fq.fake(self.t, self.W, st, self.codes)
        return torch.nn.functional.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def final(self) -> np.ndarray:
        """The trained matrix as llama.cpp bytes."""
        st = self.scales()
        return fq.pack(self.t, {**st, **fq.encode(self.t, self.W, st)})


def _run(layer, x, rope):
    y = layer(x, attention_mask=None, position_embeddings=rope)
    return y[0] if isinstance(y, tuple) else y


def train(original: Path, start: Path, tokens: np.ndarray, val: np.ndarray, dst: Path, cache: Path, log=print,
          blocks: int | None = None) -> Path:
    """Block-AP on every layer matrix of `start` (a GGUF in fakequant's types) against `original` (the 16-bit
    GGUF), on token samples (n x length); the trained codes and scales go into dst, a copy of start."""
    import gguf
    from gguf.quants import dequantize
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(original.parent), gguf_file=original.name, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").eval()
    if m.config.model_type not in ("qwen2",):  # llama-style models reorder q/k rows in the file (distill.reorder)
        raise SystemExit(f"Block-AP: {m.config.model_type} not supported yet")
    src = {t.name: t for t in gguf.GGUFReader(start).tensors}
    n, L, H = *tokens.shape, m.config.hidden_size
    cache.mkdir(parents=True, exist_ok=True)
    assert n % BATCH == 0 and len(val) % BATCH == 0
    # bf16 bits, stored as int16 (numpy has no bf16): exact, and no fp16 overflow
    fp, qn = (np.lib.format.open_memmap(cache / f"{k}.npy", "w+", np.int16, (n, L, H)) for k in ("fp", "q"))
    pos = torch.arange(L, device="cuda")[None].expand(BATCH, L)
    emb = m.model.embed_tokens.cuda()
    qemb = torch.from_numpy(dequantize(src["token_embd.weight"].data, src["token_embd.weight"].tensor_type)
                            .reshape(-1, H)).to("cuda", torch.bfloat16)  # what the quantized model really reads
    io = ThreadPoolExecutor(4)

    def put(arr, i, x):  # hidden states -> the disk cache, in the background
        return io.submit(arr.__setitem__, slice(i, i + len(x)), x.to(torch.bfloat16).view(torch.int16).cpu().numpy())

    def get(arr, i):
        return io.submit(lambda: torch.from_numpy(np.array(arr[i:i + BATCH])).view(torch.bfloat16))

    pending = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            ids = torch.from_numpy(tokens[i:i + BATCH].astype(np.int64)).cuda()
            pending += [put(fp, i, emb(ids)), put(qn, i, qemb[ids])]
        ids = torch.from_numpy(val.astype(np.int64)).cuda()
        vfp, vq = emb(ids).cpu(), qemb[ids].cpu()  # validation samples stay in memory
    for p in pending:
        p.result()
    m.model.embed_tokens.cpu()
    del emb, qemb
    rope_layer = m.model.rotary_emb.cuda()
    packed = {}
    for b in range(blocks or len(m.model.layers)):
        orig = m.model.layers[b].cuda()
        rope = rope_layer(torch.zeros(1, device="cuda", dtype=torch.bfloat16), pos)
        ql = copy.deepcopy(orig).float()
        mods = dict(ql.named_modules())
        qlin = {}
        for hf, g in LINEAR.items():
            name, t = f"blk.{b}.{g}.weight", src[f"blk.{b}.{g}.weight"].tensor_type.name
            if t not in fq.TYPES:  # no fake quantizer for this type: frozen, at the values the file really holds
                w = dequantize(src[name].data, src[name].tensor_type).reshape(mods[hf].weight.shape)
                mods[hf].weight.data = torch.from_numpy(w).to(mods[hf].weight)
                continue
            parent, attr = hf.rsplit(".", 1)
            qlin[name] = QLinear(mods[hf], t, np.asarray(src[name].data))
            setattr(mods[parent], attr, qlin[name])
        for p in ql.parameters():
            p.requires_grad_(False)
        ws = [q.W for q in qlin.values()]
        ss = [p for q in qlin.values() for p in q.theta.values()]
        for p in ws + ss:
            p.requires_grad_(True)
        steps = EPOCHS * math.ceil(n / BATCH)
        opt = torch.optim.AdamW([{"params": ws, "lr": LR_W}, {"params": ss, "lr": LR_S}], weight_decay=0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: MIN_LR + (1 - MIN_LR) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

        def val_loss():
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                return sum(torch.nn.functional.mse_loss(_run(ql, vq[i:i + BATCH].cuda(), rope).float(),
                                                        _run(orig, vfp[i:i + BATCH].cuda(), rope).float()).item()
                           for i in range(0, len(vq), BATCH)) / math.ceil(len(vq) / BATCH)
        before, total = val_loss(), 0.0
        for ep in range(EPOCHS):
            nxt, writes = (get(fp, 0), get(qn, 0)), []
            for i in range(0, n, BATCH):
                xf, xq = (f.result().cuda() for f in nxt)
                if i + BATCH < n:
                    nxt = (get(fp, i + BATCH), get(qn, i + BATCH))
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    tgt = _run(orig, xf, rope) if ep == 0 else xf  # epoch 2 reads the targets epoch 1 stored
                if ep == 0:
                    writes.append(put(fp, i, tgt))  # the cache becomes the original's input to the next block
                with torch.autocast("cuda", torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(_run(ql, xq, rope).float(), tgt.float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total += loss.item()
            for w in writes:
                w.result()
        for q in qlin.values():
            q.frozen, q.codes = True, fq.encode(q.t, q.W, q.scales())  # the final codes, used from here on
        after = val_loss()
        writes = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):  # the quantized block's output: next input
            for i in range(0, n, BATCH):
                writes.append(put(qn, i, _run(ql, get(qn, i).result().cuda(), rope)))
            vq = torch.cat([_run(ql, vq[i:i + BATCH].cuda(), rope).cpu() for i in range(0, len(vq), BATCH)])
            vfp = torch.cat([_run(orig, vfp[i:i + BATCH].cuda(), rope).cpu() for i in range(0, len(vfp), BATCH)])
        for w in writes:
            w.result()
        packed.update({k: q.final() for k, q in qlin.items()})
        log(f"block-ap block {b}: validation MSE {before:.5f} -> {after:.5f} (train {total / steps:.5f})")
        m.model.layers[b] = orig.cpu()
        del ql, qlin, opt, ws, ss
        torch.cuda.empty_cache()
    shutil.copy(start, f"{dst}.part")
    rw = gguf.GGUFReader(f"{dst}.part", "r+")
    out = {t.name: t for t in rw.tensors}
    for name, raw in packed.items():
        out[name].data[:] = raw.reshape(out[name].data.shape)
    rw.data.flush()
    del rw, out
    shutil.move(f"{dst}.part", dst)
    return dst
