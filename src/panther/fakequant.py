"""Fake quantizers for llama.cpp's Q2_K, IQ2_XS and IQ2_S, for quantization-aware training (EfficientQAT-style).

A matrix in one of these types is held as its codes and its scales (a "state"):
  unpack(t, raw)   file bytes -> state           pack(t, st)      state -> file bytes (exactly llama.cpp's layout)
  decode(t, st)    state -> weights, by llama.cpp's own formula, bit for bit
  encode(t, W, st) the codes nearest to W under st's scales (signs, then the nearest level or grid pattern)
  fake(t, W, st)   decode(encode(W)) in the forward pass, exactly; gradients: straight through to W (dW_hat/dW = 1),
                   and LSQ's to the scales (dW_hat/dstep = code - W/step), so W and the scales can be trained.
Scales are trained as floats; every use snaps them to what the file stores (16-bit d, 4-bit sub-scales).

Per block of 256 weights (llama.cpp, ggml-common.h):
  Q2_K    16 x (4-bit scale | 4-bit min << 4), 64 bytes of 2-bit codes, fp16 d, fp16 dmin; w = d*sc*q - dmin*m
  IQ2_XS  fp16 d, 32 x uint16 (9-bit index into 512 patterns of 8 magnitudes | 7-bit sign index << 9: an even number
          of minus signs), 8 bytes of 4-bit scales (one per 16 weights); w = d*(0.5+s)*0.25 * pattern * sign
  IQ2_S   fp16 d, 32 bytes of index low bits, 32 bytes of signs (one per weight), 8 bytes of index high bits
          (2 per group of 8: 1024 patterns), 8 bytes of 4-bit scales; w as IQ2_XS
"""

from __future__ import annotations

import numpy as np
import torch
from gguf.quants import IQ2_S as _IQ2S
from gguf.quants import IQ2_XS as _IQ2XS
from gguf.quants import IQ2_XXS as _IQ2XXS

TYPES = ("Q2_K", "IQ2_XS", "IQ2_S")
SIZE = {"Q2_K": 84, "IQ2_XS": 74, "IQ2_S": 82}  # bytes per block of 256 weights
CHUNK = 1 << 18  # groups of 8 searched at once (256k groups x 1024 patterns x 4 bytes = 1 GiB)
_GRID: dict[tuple[str, str], torch.Tensor] = {}
_KSIGNS = np.frombuffer(_IQ2XXS.ksigns, np.uint8)  # 7-bit sign index -> 8 sign bits (bit 7 = parity)


def grid(t: str, dev) -> torch.Tensor:
    """The type's patterns (n x 8 magnitudes: 8, 25 or 43), from the tables the gguf package ships."""
    if (t, str(dev)) not in _GRID:
        q = {"IQ2_XS": _IQ2XS, "IQ2_S": _IQ2S}[t]
        q.init_grid()
        _GRID[t, str(dev)] = torch.from_numpy(np.ascontiguousarray(q.grid.reshape(q.grid_shape))).to(dev)
    return _GRID[t, str(dev)]


def _blocks(raw: np.ndarray, t: str) -> np.ndarray:
    return np.ascontiguousarray(raw).reshape(raw.shape[0], -1, SIZE[t])  # rows x blocks x bytes


def _f16(b: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(b).view(np.float16)[..., 0].astype(np.float32))


def _bits(x: np.ndarray, n: int, width: int) -> np.ndarray:
    """The n fields of `width` bits in each element of x, lowest first, as a new last axis."""
    return (x[..., None] >> (np.arange(n, dtype=x.dtype) * width)) & ((1 << width) - 1)


def unpack(t: str, raw: np.ndarray, dev="cpu") -> dict[str, torch.Tensor]:
    """A matrix's bytes (rows x row bytes) -> state: float scales and integer codes, rows x blocks x ..."""
    b = _blocks(raw, t)
    if t == "Q2_K":
        q = _bits(b[..., 16:80].reshape(*b.shape[:2], 2, 32), 4, 2)  # half, byte, shift -> weight half*128+shift*32+byte
        st = {"d": _f16(b[..., 80:82]), "dmin": _f16(b[..., 82:84]),
              "sc": torch.from_numpy((b[..., :16] & 15).astype(np.float32)),
              "m": torch.from_numpy((b[..., :16] >> 4).astype(np.float32)),
              "q": torch.from_numpy(q.transpose(0, 1, 2, 4, 3).reshape(*b.shape[:2], 256).astype(np.int64))}
    elif t == "IQ2_XS":
        qs = b[..., 2:66].copy().view(np.uint16)
        st = {"d": _f16(b[..., 0:2]), "s": torch.from_numpy(_bits(b[..., 66:74], 2, 4).reshape(*b.shape[:2], 16)
                                                             .astype(np.float32)),
              "idx": torch.from_numpy((qs & 511).astype(np.int64)),
              "neg": torch.from_numpy(_bits(_KSIGNS[qs >> 9], 8, 1).astype(bool))}
    else:
        hi = _bits(b[..., 66:74], 4, 2).reshape(*b.shape[:2], 32)
        st = {"d": _f16(b[..., 0:2]), "s": torch.from_numpy(_bits(b[..., 74:82], 2, 4).reshape(*b.shape[:2], 16)
                                                             .astype(np.float32)),
              "idx": torch.from_numpy(b[..., 2:34].astype(np.int64) | (hi.astype(np.int64) << 8)),
              "neg": torch.from_numpy(_bits(b[..., 34:66], 8, 1).astype(bool))}
    return {k: v.to(dev) for k, v in st.items()}


def snap(t: str, st: dict) -> dict:
    """The scales as the file stores them: d (and dmin) to 16 bits, sub-scales (and mins) to 0-15."""
    out = {k: v for k, v in st.items() if k in ("q", "idx", "neg")}
    for k in ("d", "dmin"):
        if k in st:
            out[k] = st[k].detach().half().float()
    for k in ("sc", "m", "s"):
        if k in st:
            out[k] = st[k].detach().round().clamp(0, 15)
    return out


def pack(t: str, st: dict) -> np.ndarray:
    """State -> llama.cpp's bytes (rows x row bytes)."""
    s = {k: v.detach().cpu().numpy() for k, v in snap(t, st).items()}
    f16 = lambda x: x.astype(np.float16)[..., None].view(np.uint8)  # noqa: E731
    nib = lambda x: (x[..., 0::2] + 16 * x[..., 1::2]).astype(np.uint8)  # noqa: E731
    if t == "Q2_K":
        q = s["q"].reshape(*s["q"].shape[:2], 2, 4, 32).astype(np.uint8)  # half, shift, byte
        qs = (q << (2 * np.arange(4, dtype=np.uint8))[:, None]).sum(-2, dtype=np.uint8).reshape(*q.shape[:2], 64)
        b = np.concatenate([(s["sc"] + 16 * s["m"]).astype(np.uint8), qs, f16(s["d"]), f16(s["dmin"])], -1)
    else:
        sign = (s["neg"].astype(np.uint8) << np.arange(8, dtype=np.uint8)).sum(-1, dtype=np.uint8)
        if t == "IQ2_XS":
            si = sign & 127
            if (_KSIGNS[si] != sign).any():
                raise ValueError("IQ2_XS: a group of 8 has an odd number of minus signs")
            qs = (s["idx"].astype(np.uint16) | (si.astype(np.uint16) << 9)).view(np.uint8)
            b = np.concatenate([f16(s["d"]), qs, nib(s["s"])], -1)
        else:
            hi = (s["idx"] >> 8).astype(np.uint8).reshape(*s["idx"].shape[:2], 8, 4)
            qh = (hi << (2 * np.arange(4, dtype=np.uint8))).sum(-1, dtype=np.uint8)
            b = np.concatenate([f16(s["d"]), (s["idx"] & 255).astype(np.uint8), sign, qh, nib(s["s"])], -1)
    return b.reshape(b.shape[0], -1)


def _step(t: str, st: dict) -> torch.Tensor:
    """IQ: the step of every group of 8, rows x blocks x 32 x 1, in llama.cpp's order of operations."""
    return (st["d"][..., None] * (0.5 + st["s"]) * 0.25).repeat_interleave(2, -1)[..., None]


def decode(t: str, st: dict) -> torch.Tensor:
    """State -> weights (rows x columns), exactly as llama.cpp dequantizes them."""
    s = snap(t, st)
    if t == "Q2_K":
        dl, ml = (s["d"][..., None] * s["sc"])[..., None], (s["dmin"][..., None] * s["m"])[..., None]
        w = dl * s["q"].float().view(*s["q"].shape[:2], 16, 16) - ml
    else:
        g = grid(t, s["idx"].device)[s["idx"]]
        w = _step(t, s) * g * torch.where(s["neg"], -1.0, 1.0)
    return w.reshape(w.shape[0], -1)


def _signs(t: str, W: torch.Tensor) -> torch.Tensor:
    """Minus signs where W < 0; IQ2_XS stores only an even number per group, so an odd group flips its smallest."""
    neg = W < 0
    if t == "IQ2_XS":
        odd = neg.sum(-1, keepdim=True) % 2 == 1
        small = torch.nn.functional.one_hot(W.abs().argmin(-1), 8).bool()
        neg = neg ^ (odd & small)
    return neg


def encode(t: str, W: torch.Tensor, st: dict) -> dict:
    """The codes nearest to W (rows x columns) under st's scales, plain squared error. Returns the new codes."""
    s = snap(t, st)
    with torch.no_grad():
        if t == "Q2_K":
            dl = (s["d"][..., None] * s["sc"])[..., None]
            u = (W.float().view(*dl.shape[:3], 16) + (s["dmin"][..., None] * s["m"])[..., None]) / dl
            return {"q": torch.nan_to_num(u.round(), 0).clamp(0, 3).long().view(*dl.shape[:2], 256)}
        w = W.float().view(*s["d"].shape, 32, 8)
        neg = _signs(t, w)
        a = (w * torch.where(neg, -1.0, 1.0) / _step(t, s).clamp_min(1e-30)).view(-1, 8)  # a negative a: a flipped sign
        g = grid(t, w.device)
        g2, idx = (g * g).sum(-1), torch.empty(a.shape[0], dtype=torch.long, device=w.device)
        for i in range(0, a.shape[0], CHUNK):
            idx[i:i + CHUNK] = (g2 - 2 * a[i:i + CHUNK] @ g.T).argmin(-1)  # |a - g|^2 without the constant |a|^2
        return {"idx": idx.view(*w.shape[:3]), "neg": neg}


def fake(t: str, W: torch.Tensor, st: dict, codes: dict | None = None) -> torch.Tensor:
    """W's quantized value (forward: exactly decode(encode(W)), or decode(codes) for codes chosen earlier), with
    gradients to W and to st's float scales."""
    c = codes or encode(t, W, st)
    wq = decode(t, {**st, **c})
    snapped = lambda x, lo=None: x + ((x.round().clamp(0, 15) if lo else x.half().float()) - x).detach()  # noqa: E731
    w = W.view(*st["d"].shape, -1)
    if t == "Q2_K":
        dl = (snapped(st["d"])[..., None] * snapped(st["sc"], 1))[..., None]
        ml = (snapped(st["dmin"])[..., None] * snapped(st["m"], 1))[..., None]
        w, q = w.view(*dl.shape[:3], 16), c["q"].view(*dl.shape[:3], 16).float()
        u = ((w + ml) / dl.clamp_min(1e-30)).detach()
        out = (q != u.round()).float()  # clamped to 0..3: the min moves those weights
        lsq = dl * (q - u).detach() - ml * out
    else:
        step = (snapped(st["d"])[..., None] * (0.5 + snapped(st["s"], 1)) * 0.25).repeat_interleave(2, -1)[..., None]
        sg = torch.where(c["neg"], -1.0, 1.0)
        a = (w.view(*sg.shape) * sg / step.clamp_min(1e-30)).detach()
        lsq = step * ((grid(t, w.device)[c["idx"]] - a) * sg).detach()
    lsq = lsq.reshape(wq.shape)
    return wq + (W - W.detach()) + (lsq - lsq.detach())  # both added terms are exactly 0 in the forward pass
