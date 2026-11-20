"""Stage 1 of EfficientQAT (Block-AP) for a llama.cpp file: block by block, each layer matrix's weights and its
format's scales are trained so the quantized block reproduces the original block, then the codes and scales are
written into a copy of the file (same types, same size). As the paper (arXiv 2407.11062, official code):
  - target: the original block's output on the original model's own hidden states; input: the outputs of the blocks
    already quantized, so each block also corrects the error before it; loss: mean squared error
  - trained: the latent weights (starting at the original's) and the scales (starting at the file's); norms and
    biases frozen; AdamW, weights at 1e-3 of their size (the paper's 2e-5 on Qwen2.5-14B; see LR_W), cosine to 1/20 per batch, batch 2, 2 epochs. Unlike the paper, the
    block scales stay frozen here (LR_S): llama.cpp's d covers 256 weights over fixed integer sub-scales, and
    training it drifted (see LR_S); they are trained in Stage 2
Ours: llama.cpp's own types (panther.fakequant) instead of uniform quantization; bf16 instead of fp16 (Qwen2.5
overflows fp16); a matrix's codes are re-chosen every RECODE steps (the IQ pattern search is the slow part; the
weights move by ~1e-5 a step). Every 512 steps the block's validation error is measured and the best point is kept
(training can drift late, or blow up: on the 14B, block 0's error halved by step 512, then doubled), and a
block stops training after PATIENCE checks without a better point. Codes are always picked in fp32
(fakequant.encode), and a finished block is scored, and run for the blocks after it, from the bytes that are saved;
if that is not better than the block the starting file already has, the file's block stays. The hidden
states of every sample (two copies, original and quantized) do not fit in memory at the paper's size (14B: 2 x 86
GB), so they are streamed from disk (bf16), as the paper's code does for large models (fp16 there). A finished
block's output is not written in a pass of its own: the next block's first epoch computes it from the cache (which
still holds the finished block's input) and writes it back, one read of the cache fewer per block.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from panther import fakequant as fq
from panther.distill import LINEAR, reorder

BATCH, EPOCHS, LR_W, LR_S, MIN_LR, RECODE, PATIENCE = 2, 2, 1e-3, 0, 1 / 20, 16, 2
# LR_W is relative: a matrix's learning rate is LR_W x the root mean square of its weights. Adam moves every weight
# by about the learning rate a step, whatever the loss, so an absolute rate suits one weight size only: the paper's
# 2e-5 is 1e-3 of Qwen2.5-14B's weights (~0.02, where it worked), but 1e-2 of Mistral-22B's (~0.002), whose block 0
# went 2.9e-6 -> 6.2e-2 in 192 steps at 2e-5 and 2.9e-6 -> 3.8e-7 at 2e-6 (the same 1e-3 of its weights).
CHECK = 512  # steps between validation checks (a trial on a few samples sets it lower)


def _f(v: float) -> str:
    """An error for the log: 5 decimals, more when it is small (Mistral's first blocks are ~1e-6)."""
    return f"{v:.5f}" if v >= 0.01 else f"{v:.8f}"
# LR_S (relative, per unit of log d) is 0: weights only. On the 14B's block 0 at 4,096 steps, weights + scales (1e-3)
# went 0.0098 -> 0.0181 (the error halved, then doubled: the scales drift); weights only 0.0098 -> 0.0040. Scales are
# trained in Stage 2 instead, end to end, with the codes frozen.


HEADS = {"self_attn.q_proj": "num_attention_heads", "self_attn.k_proj": "num_key_value_heads"}


def row_order(hf: str, w: torch.Tensor, deq: np.ndarray, cfg) -> tuple[torch.Tensor | None, float, float]:
    """Where the file keeps this matrix's rows. llama.cpp's converter reorders the rows of attention q and k for
    llama-style models (Mistral, Llama), not for Qwen: the order (as it is, or reordered) under which the file's
    values are nearer the PyTorch weight is the file's. Returns (the index of each file row in PyTorch's order, or
    None if the same; the mean squared difference as it is; reordered)."""
    d = torch.from_numpy(deq).to(w.device, torch.float32)
    same = (w.float() - d).pow(2).mean().item()
    if hf not in HEADS:
        return None, same, float("nan")
    p = torch.from_numpy(reorder(np.arange(w.shape[0]), getattr(cfg, HEADS[hf]))).to(w.device)
    moved = (w.float()[p] - d).pow(2).mean().item()
    return (p, same, moved) if moved < same else (None, same, moved)


class QLinear(torch.nn.Module):
    """A linear layer whose weight is the fake-quantized latent weight W. The file's block scales d (and dmin) are
    trained as relative multipliers, d = d0 * exp(theta): the paper's learning rate is for step sizes ~50x larger
    than llama.cpp's d, and in absolute units it wrecks them. The 4-bit sub-scales stay as in the file."""

    def __init__(self, lin: torch.nn.Linear, t: str, raw: np.ndarray, rows: torch.Tensor | None = None):
        super().__init__()
        self.t, self.n, self.codes = t, 0, None
        w = lin.weight.detach().float()
        self.W = torch.nn.Parameter((w if rows is None else w[rows]).clone())  # kept in the file's row order
        self.register_buffer("back", None if rows is None else torch.argsort(rows))  # to PyTorch's order
        self.st = fq.unpack(t, raw, self.W.device)
        self.theta = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros_like(self.st[k]))
                                             for k in ("d", "dmin") if k in self.st})
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().float())

    def scales(self) -> dict:
        return {**self.st, **{k: self.st[k] * torch.exp(th) for k, th in self.theta.items()}}

    def forward(self, x):
        st = self.scales()
        if self.codes is None or self.n % RECODE == 0:
            self.codes = fq.encode(self.t, self.W, st)
        self.n += 1
        w = fq.fake(self.t, self.W, st, self.codes)
        if self.back is not None:
            w = w[self.back]
        return torch.nn.functional.linear(x, w.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def final(self) -> np.ndarray:
        """The trained matrix as llama.cpp bytes (codes picked now, from the weights as they are)."""
        st = self.scales()
        return fq.pack(self.t, {**st, **fq.encode(self.t, self.W, st)})


def _restored(layer, b: int, src: dict, saved: dict, cfg):
    """A trained block, rebuilt from its saved bytes (other matrices at the file's values), on the GPU."""
    from gguf.quants import dequantize
    q = copy.deepcopy(layer).cuda()
    mods = dict(q.named_modules())
    for hf, g in LINEAR.items():
        name = f"blk.{b}.{g}.weight"
        w = mods[hf].weight
        rows = row_order(hf, w, dequantize(np.asarray(src[name].data), src[name].tensor_type).reshape(w.shape), cfg)[0]
        new = torch.from_numpy(dequantize(saved.get(name, np.asarray(src[name].data)), src[name].tensor_type)
                               .reshape(w.shape)).to(w)
        w.data = new if rows is None else new[torch.argsort(rows)]
    return q


def _run(layer, x, rope):
    y = layer(x, attention_mask=None, position_embeddings=rope)
    return y[0] if isinstance(y, tuple) else y


def train(original: Path, start: Path, tokens: np.ndarray, val: np.ndarray, dst: Path, cache: Path, log=print,
          blocks: int | None = None, rebuild_gpu=None) -> Path:
    """Block-AP on every layer matrix of `start` (a GGUF in fakequant's types) against `original` (the 16-bit
    GGUF), on token samples (n x length); the trained codes and scales go into dst, a copy of start."""
    import gguf
    from gguf.quants import dequantize
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(original.parent), gguf_file=original.name, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").eval()
    src = {t.name: t for t in gguf.GGUFReader(start).tensors}
    log(f"block-ap strategy: {'weights only (block scales frozen)' if LR_S == 0 else 'weights and block scales'}, "
        f"weights lr {LR_W:g} x each matrix's weight size, scales lr {LR_S:g} (relative), {EPOCHS} epochs x {len(tokens)} samples, "
        f"{blocks or len(m.model.layers)} blocks")
    n, L, H = *tokens.shape, m.config.hidden_size
    cache.mkdir(parents=True, exist_ok=True)
    assert n % BATCH == 0 and len(val) % BATCH == 0
    # bf16 bits, stored as int16 (numpy has no bf16): exact, and no fp16 overflow
    mode = "r+" if (cache / "fp.npy").exists() else "w+"
    fp, qn = (np.lib.format.open_memmap(cache / f"{k}.npy", mode, np.int16, (n, L, H)) for k in ("fp", "q"))
    pos = torch.arange(L, device="cuda")[None].expand(BATCH, L)
    rope_layer = m.model.rotary_emb.cuda()
    rope = rope_layer(torch.zeros(1, device="cuda", dtype=torch.bfloat16), pos)
    rd, wr, pending = ThreadPoolExecutor(2), ThreadPoolExecutor(2), deque()

    def put(arr, i, x):  # hidden states -> the disk cache, in the background; at most 8 batches wait in memory
        while len(pending) >= 8:
            pending.popleft().result()
        pending.append(wr.submit(arr.__setitem__, slice(i, i + len(x)),
                                 x.to(torch.bfloat16).view(torch.int16).cpu().numpy()))

    def flush():
        while pending:
            pending.popleft().result()

    def get(arr, i):
        return rd.submit(lambda: torch.from_numpy(np.array(arr[i:i + BATCH])).view(torch.bfloat16))

    def saved(b: int) -> Path:
        return cache / f"block-{b}.npz"
    # resume: blocks already trained are kept (their codes and scales); the caches are rebuilt unless they are
    # known to hold exactly the inputs of the next block
    done = 0
    while saved(done).exists() and (blocks is None or done < blocks):
        done += 1
    state = json.loads((cache / "state.json").read_text()) if (cache / "state.json").exists() else {}
    import contextlib
    held = (rebuild_gpu or contextlib.nullcontext)() if done else contextlib.nullcontext()
    held.__enter__()  # extra GPU memory while every trained block is on the GPU (a resume), released below
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        emb = m.model.embed_tokens.cuda()
        qemb = torch.from_numpy(dequantize(src["token_embd.weight"].data, src["token_embd.weight"].tensor_type)
                                .reshape(-1, H)).to("cuda", torch.bfloat16)  # what the quantized model really reads
        origs = [m.model.layers[b].cuda() for b in range(done)]
        quants = [_restored(m.model.layers[b], b, src, dict(np.load(saved(b))), m.config) for b in range(done)]

        def through(ids):
            x, xq = emb(ids), qemb[ids]
            for o, q in zip(origs, quants):
                x, xq = _run(o, x, rope), _run(q, xq, rope)
            return x, xq
        # the quantized cache is one block behind after a finished block ("lag"): the next block's first epoch
        # runs the block before it on what it reads and writes that back, instead of a pass of its own
        lag = state.get("lag", 0) if (state.get("done"), state.get("clean"), state.get("n")) == (done, True, n) else 0
        if (state.get("done"), state.get("clean"), state.get("n")) != (done, True, n):
            if done:
                log(f"block-ap: resuming after block {done - 1}; rebuilding the caches through the trained blocks")
            for i in range(0, n, BATCH):
                x, xq = through(torch.from_numpy(tokens[i:i + BATCH].astype(np.int64)).cuda())
                put(fp, i, x)
                put(qn, i, xq)
            flush()
        vs = [through(torch.from_numpy(val[i:i + BATCH].astype(np.int64)).cuda()) for i in range(0, len(val), BATCH)]
        vfp, vq = torch.cat([v[0] for v in vs]).cpu(), torch.cat([v[1] for v in vs]).cpu()  # stay in memory
    for o in origs:
        o.cpu()
    m.model.embed_tokens.cpu()
    prev = quants[-1] if lag and quants else None  # the last trained block, quantized: the cache is its input
    del emb, qemb, origs, quants, vs
    torch.cuda.empty_cache()
    held.__exit__(None, None, None)
    for b in range(done, blocks or len(m.model.layers)):
        (cache / "state.json").write_text(json.dumps({"done": b, "clean": False, "n": n}))  # caches change below
        orig = m.model.layers[b].cuda()
        ql = copy.deepcopy(orig).float()
        mods = dict(ql.named_modules())
        qlin = {}
        for hf, g in LINEAR.items():
            name, t = f"blk.{b}.{g}.weight", src[f"blk.{b}.{g}.weight"].tensor_type.name
            deq = dequantize(np.asarray(src[name].data), src[name].tensor_type).reshape(mods[hf].weight.shape)
            rows, same, moved = row_order(hf, mods[hf].weight, deq, m.config)
            if b == 0 and hf in HEADS:
                log(f"block-ap rows of {g}: {'reordered in the file (llama-style)' if rows is not None else 'as in PyTorch'}"
                    f" (difference from the original {min(same, moved):.2e}; the other order {max(same, moved):.2e})")
            if t not in fq.TYPES:  # no fake quantizer for this type: frozen, at the values the file really holds
                w = torch.from_numpy(deq).to(mods[hf].weight)
                mods[hf].weight.data = w if rows is None else w[torch.argsort(rows)]
                continue
            parent, attr = hf.rsplit(".", 1)
            qlin[name] = QLinear(mods[hf], t, np.asarray(src[name].data), rows)
            setattr(mods[parent], attr, qlin[name])
        for p in ql.parameters():
            p.requires_grad_(False)
        ws = [q.W for q in qlin.values()]
        ss = [p for q in qlin.values() for p in q.theta.values()]
        for p in ws + ss:
            p.requires_grad_(True)
        steps = EPOCHS * math.ceil(n / BATCH)
        opt = torch.optim.AdamW([{"params": [q.W], "lr": LR_W * q.W.detach().pow(2).mean().sqrt().item()}
                                 for q in qlin.values()] + [{"params": ss, "lr": LR_S}], weight_decay=0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: MIN_LR + (1 - MIN_LR) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

        def val_loss(layer=None):  # of the block in training, or of another version of it
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                return sum(torch.nn.functional.mse_loss(_run(layer or ql, vq[i:i + BATCH].cuda(), rope).float(),
                                                        _run(orig, vfp[i:i + BATCH].cuda(), rope).float()).item()
                           for i in range(0, len(vq), BATCH)) / math.ceil(len(vq) / BATCH)
        def snap():
            return [p.detach().cpu().clone() for p in ws + ss]
        file_err = val_loss(_restored(orig, b, src, {}, m.config))  # the block as the starting file has it
        before, total, mark = val_loss(), 0.0, 0.0
        best = (before, 0, snap())  # (validation error, step, weights and scales): never end worse than the start
        log(f"block-ap block {b}: step 0, validation {_f(before)}")  # the starting point, so a curve starts at 0
        stop, ran = False, 0  # no better point for PATIENCE checks: stop (the paper's early stopping)
        for ep in range(EPOCHS):
            nxt = (get(fp, 0), get(qn, 0))
            for i in range(0, n, BATCH):
                xf, xq = (f.result().cuda() for f in nxt)
                if i + BATCH < n:
                    nxt = (get(fp, i + BATCH), get(qn, i + BATCH))
                if stop and ep > 0:
                    break
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    tgt = _run(orig, xf, rope) if ep == 0 else xf  # epoch 2 reads the targets epoch 1 stored
                    if ep == 0 and prev is not None:
                        xq = _run(prev, xq, rope)  # the cache held the input of the block before: now this one's
                if ep == 0:
                    put(fp, i, tgt)  # the cache becomes the original's input to the next block
                    if prev is not None:
                        put(qn, i, xq)  # and epoch 2 reads this block's input
                if stop:
                    continue  # stopped early: the original's outputs are still written for the next block
                with torch.autocast("cuda", torch.bfloat16):
                    loss = torch.nn.functional.mse_loss(_run(ql, xq, rope).float(), tgt.float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                total, ran = total + loss.item(), ran + 1
                k = ep * (n // BATCH) + i // BATCH + 1
                if k % CHECK == 0:  # every 512 steps: the training error of the stretch, and the best point so far
                    v = val_loss()
                    if v < best[0]:
                        best = (v, k, snap())
                    stop = k - best[1] >= PATIENCE * CHECK
                    if stop:
                        log(f"block-ap block {b}: stopping early at step {k} (best at step {best[1]})")
                    log(f"block-ap block {b}: step {k}/{steps}, train MSE {_f((total - mark) / CHECK)}, "
                        f"validation {_f(v)}")
                    mark = total
            flush()  # epoch 2 reads what epoch 1 wrote
        with torch.no_grad():  # back to the best point (the paper's code stops early on the validation error)
            for p, v in zip(ws + ss, best[2]):
                p.copy_(v.to(p.device))
        # the block is scored, and run for the blocks after it, from the bytes that are saved: what is logged is
        # what the file holds
        packed = {k: q.final() for k, q in qlin.items()}
        ql = _restored(orig, b, src, packed, m.config)
        after = val_loss()
        if after >= file_err:  # training did not beat the block the file already has: that one stays
            log(f"block-ap block {b}: trained {_f(after)}, not better than the file's {_f(file_err)}: keeping the file's")
            packed = {k: np.asarray(src[k].data) for k in qlin}
            ql, after = _restored(orig, b, src, {}, m.config), file_err
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):  # the validation samples move on a block
            vq = torch.cat([_run(ql, vq[i:i + BATCH].cuda(), rope).cpu() for i in range(0, len(vq), BATCH)])
            vfp = torch.cat([_run(orig, vfp[i:i + BATCH].cuda(), rope).cpu() for i in range(0, len(vfp), BATCH)])
        flush()
        np.savez(f"{saved(b)}.part.npz", **packed)
        os.rename(f"{saved(b)}.part.npz", saved(b))
        (cache / "state.json").write_text(json.dumps({"done": b + 1, "clean": True, "n": n, "lag": 1}))
        log(f"block-ap block {b}: validation MSE {_f(before)} -> {_f(after)} (best at step {best[1]}; "
            f"train {_f(total / max(1, ran))}; the file's {_f(file_err)})")
        m.model.layers[b] = orig.cpu()
        prev = ql  # the saved block: the next block's first epoch runs it on the quantized cache
        del qlin, opt, ws, ss, packed
        torch.cuda.empty_cache()
    shutil.copy(start, f"{dst}.part")
    rw = gguf.GGUFReader(f"{dst}.part", "r+")
    out = {t.name: t for t in rw.tensors}
    for b in range(blocks or len(m.model.layers)):
        for name, raw in np.load(saved(b)).items():
            out[name].data[:] = raw.reshape(out[name].data.shape)
    rw.data.flush()
    del rw, out
    shutil.move(f"{dst}.part", dst)
    return dst
