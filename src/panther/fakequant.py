"""Fake quantizers for all 14 of llama.cpp's types used here, for quantization-aware training (EfficientQAT-style).

Every type decodes as  w = step * value - offset,  per sub-block of a block (256 weights; Q8_0: 32), where
  step   comes from the block scale d and an integer sub-scale (each type its own formula, in llama.cpp's order),
  offset is dmin * m for Q2_K, Q4_K and Q5_K (else 0),
  value  is the code: an integer level (K types, Q8_0), a level of a 16-entry table (IQ4_XS), or a pattern of 8 or 4
         magnitudes from a fixed grid times signs (IQ2/IQ3 types), or a grid pattern of -1/0/1 plus a +-1/8 shift
         (IQ1 types).
A matrix is held as its scales and codes (a "state"):
  unpack(t, raw) / pack(t, st)   file bytes <-> state, exactly llama.cpp's layout (ggml-common.h)
  decode(t, st)                  state -> weights, bit for bit as llama.cpp
  encode(t, W, st)               the codes nearest to W under st's scales (plain squared error)
  fake(t, W, st)                 decode(encode(W)) exactly in the forward pass; gradients straight through to W and
                                 LSQ's (value - W/step) to d, dmin and the sub-scales.
Scales are trained as floats; every use snaps them to what the file stores (16-bit d, integer sub-scales)."""

from __future__ import annotations

import numpy as np
import torch
import gguf.quants as gq

QK = 256
CHUNK = 1 << 18  # groups searched at once against a grid (256k groups x 2048 patterns x 4 bytes = 2 GiB)
KSIGNS = np.frombuffer(gq.IQ2_XXS.ksigns, np.uint8)  # 7-bit sign index -> 8 sign bits (bit 7 = parity)
KVALUES = torch.tensor(gq.IQ4_NL.kvalues, dtype=torch.float32)
# type: (block bytes, block weights, sub-blocks, scale range (lo, hi), kind, extra)
#   kind "level": integer codes in extra=(lo, hi); "table": IQ4_XS; "grid": extra=(grid width, signs: parity|free);
#   "iq1": extra=(weights per shift)
SPEC = {
    "Q2_K": (84, QK, 16, (0, 15), "level", (0, 3)), "Q3_K": (110, QK, 16, (-32, 31), "level", (-4, 3)),
    "Q4_K": (144, QK, 8, (0, 63), "level", (0, 15)), "Q5_K": (176, QK, 8, (0, 63), "level", (0, 31)),
    "Q6_K": (210, QK, 16, (-128, 127), "level", (-32, 31)), "Q8_0": (34, 32, 1, None, "level", (-128, 127)),
    "IQ4_XS": (136, QK, 8, (-32, 31), "table", None),
    "IQ2_XXS": (66, QK, 8, (0, 15), "grid", (8, "parity")), "IQ2_XS": (74, QK, 16, (0, 15), "grid", (8, "parity")),
    "IQ2_S": (82, QK, 16, (0, 15), "grid", (8, "free")), "IQ3_XXS": (98, QK, 8, (0, 15), "grid", (4, "parity")),
    "IQ3_S": (110, QK, 8, (0, 15), "grid", (4, "free")), "IQ1_S": (50, QK, 8, (0, 7), "iq1", 32),
    "IQ1_M": (56, QK, 16, (0, 7), "iq1", 8)}
TYPES = tuple(SPEC)
SIZE = {t: s[0] for t, s in SPEC.items()}
OFFSET = ("Q2_K", "Q4_K", "Q5_K")
_GRID: dict[tuple[str, str], torch.Tensor] = {}


def grid(t: str, dev) -> torch.Tensor:
    """The type's patterns (n x width), from the tables the gguf package ships (IQ1_M shares IQ1_S's)."""
    t = "IQ1_S" if t == "IQ1_M" else t
    if (t, str(dev)) not in _GRID:
        q = getattr(gq, t)
        q.init_grid()
        _GRID[t, str(dev)] = torch.from_numpy(np.ascontiguousarray(q.grid.reshape(q.grid_shape))).to(dev)
    return _GRID[t, str(dev)]


# --- bytes <-> state --------------------------------------------------------------------------------------------

def _bits(x: np.ndarray, n: int, width: int) -> np.ndarray:
    """The n fields of `width` bits in each element of x, lowest first, as a new last axis."""
    return (x[..., None] >> (np.arange(n, dtype=x.dtype) * width)) & ((1 << width) - 1)


def _join(f: np.ndarray, width: int, dtype=np.uint8) -> np.ndarray:
    """Inverse of _bits: the last axis' fields packed, lowest first."""
    return (f.astype(dtype) << (np.arange(f.shape[-1], dtype=dtype) * width)).sum(-1, dtype=dtype)


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16)[..., 0].astype(np.float32)


def _h(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)[..., None].view(np.uint8)


def _k4(s: np.ndarray):
    """Q4_K/Q5_K: 12 bytes -> 8 six-bit scales and 8 six-bit mins."""
    a, m, md = s[..., 0:4], s[..., 4:8], s[..., 8:12]
    return (np.concatenate([a & 63, (md & 15) | ((a >> 2) & 48)], -1),
            np.concatenate([m & 63, (md >> 4) | ((m >> 2) & 48)], -1))


def _k4pack(sc: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.concatenate([(sc[..., :4] & 63) | ((sc[..., 4:] >> 4) << 6), (m[..., :4] & 63) | ((m[..., 4:] >> 4) << 6),
                           (sc[..., 4:] & 15) | ((m[..., 4:] & 15) << 4)], -1).astype(np.uint8)


def _unpack(t: str, b: np.ndarray) -> dict[str, np.ndarray]:
    """b: rows x blocks x block bytes -> numpy state, codes in the block's weight order."""
    sh = b.shape[:2]
    if t == "Q2_K":
        q = _bits(b[..., 16:80].reshape(*sh, 2, 32), 4, 2).swapaxes(-1, -2)  # half, shift, byte
        return {"d": _f16(b[..., 80:82]), "dmin": _f16(b[..., 82:84]), "sc": b[..., :16] & 15, "m": b[..., :16] >> 4,
                "q": q.reshape(*sh, QK)}
    if t == "Q3_K":
        s = b[..., 96:108]
        sc = (_bits(s[..., :8], 2, 4).swapaxes(-1, -2).reshape(*sh, 16)
              | (_bits(s[..., 8:12], 4, 2).swapaxes(-1, -2).reshape(*sh, 16) << 4)).astype(np.int16) - 32
        ql = _bits(b[..., 32:96].reshape(*sh, 2, 32), 4, 2).swapaxes(-1, -2).reshape(*sh, QK)
        hb = _bits(b[..., :32], 8, 1).swapaxes(-1, -2).reshape(*sh, QK)  # bit, byte
        return {"d": _f16(b[..., 108:110]), "sc": sc, "q": ql.astype(np.int16) - 4 * (1 - hb.astype(np.int16))}
    if t in ("Q4_K", "Q5_K"):
        sc, m = _k4(b[..., 4:16])
        qs = b[..., 16:144] if t == "Q4_K" else b[..., 48:176]
        q = _bits(qs.reshape(*sh, 4, 32), 2, 4).swapaxes(-1, -2).reshape(*sh, QK)  # chunk, nibble, byte
        if t == "Q5_K":
            q = q | (_bits(b[..., 16:48], 8, 1).swapaxes(-1, -2).reshape(*sh, QK) << 4)
        return {"d": _f16(b[..., 0:2]), "dmin": _f16(b[..., 2:4]), "sc": sc, "m": m, "q": q}
    if t == "Q6_K":
        ql = _bits(b[..., :128].reshape(*sh, 2, 64), 2, 4).swapaxes(-1, -2).reshape(*sh, QK)  # half, nibble, j
        qh = _bits(b[..., 128:192].reshape(*sh, 2, 32), 4, 2).swapaxes(-1, -2).reshape(*sh, QK)  # half, shift, byte
        return {"d": _f16(b[..., 208:210]), "sc": b[..., 192:208].view(np.int8).astype(np.int16),
                "q": (ql | (qh << 4)).astype(np.int16) - 32}
    if t == "Q8_0":
        return {"d": _f16(b[..., 0:2]), "q": b[..., 2:34].view(np.int8).astype(np.int16)}
    if t == "IQ4_XS":
        sc = (_bits(b[..., 4:8], 2, 4).reshape(*sh, 8)
              | (_bits(b[..., 2:4].copy().view(np.uint16)[..., 0], 8, 2).astype(np.uint8) << 4)).astype(np.int16) - 32
        q = _bits(b[..., 8:136].reshape(*sh, 8, 16), 2, 4).swapaxes(-1, -2).reshape(*sh, QK)  # sub, nibble, j
        return {"d": _f16(b[..., 0:2]), "sc": sc, "q": q}
    if t in ("IQ2_XXS", "IQ3_XXS"):
        aux = (b[..., 2:66] if t == "IQ2_XXS" else b[..., 66:98]).copy().view(np.uint32)
        if t == "IQ2_XXS":
            idx, aux = aux[..., 0::2].copy().view(np.uint8).reshape(*sh, 32), aux[..., 1::2]
        else:
            idx = b[..., 2:66]
        si = _bits(aux, 4, 7).reshape(*sh, 32)
        return {"d": _f16(b[..., 0:2]), "sc": (aux >> 28).astype(np.uint8), "idx": idx,
                "neg": _bits(KSIGNS[si], 8, 1).astype(bool)}
    if t == "IQ2_XS":
        qs = b[..., 2:66].copy().view(np.uint16)
        return {"d": _f16(b[..., 0:2]), "sc": _bits(b[..., 66:74], 2, 4).reshape(*sh, 16), "idx": qs & 511,
                "neg": _bits(KSIGNS[qs >> 9], 8, 1).astype(bool)}
    if t == "IQ2_S":
        hi = _bits(b[..., 66:74], 4, 2).reshape(*sh, 32).astype(np.uint16)
        return {"d": _f16(b[..., 0:2]), "sc": _bits(b[..., 74:82], 2, 4).reshape(*sh, 16),
                "idx": b[..., 2:34].astype(np.uint16) | (hi << 8), "neg": _bits(b[..., 34:66], 8, 1).astype(bool)}
    if t == "IQ3_S":
        hi = _bits(b[..., 66:74], 8, 1).reshape(*sh, 64).astype(np.uint16)
        return {"d": _f16(b[..., 0:2]), "sc": _bits(b[..., 106:110], 2, 4).reshape(*sh, 8),
                "idx": b[..., 2:66].astype(np.uint16) | (hi << 8), "neg": _bits(b[..., 74:106], 8, 1).astype(bool)}
    if t == "IQ1_S":
        qh = b[..., 34:50].copy().view(np.uint16)
        hi = _bits(qh & 4095, 4, 3).reshape(*sh, 32)
        return {"d": _f16(b[..., 0:2]), "sc": ((qh >> 12) & 7).astype(np.uint8),
                "idx": b[..., 2:34].astype(np.uint16) | (hi << 8), "dneg": (qh >> 15).astype(bool)}
    if t == "IQ1_M":
        w = b[..., 48:56].copy().view(np.uint16)
        d = (w >> 12).astype(np.uint16) << (np.arange(4, dtype=np.uint16) * 4)
        nib = _bits(b[..., 32:48], 2, 4).reshape(*sh, 32)
        return {"d": (d.sum(-1, dtype=np.uint16)).view(np.float16).astype(np.float32),
                "sc": _bits(w & 4095, 4, 3).reshape(*sh, 16).astype(np.uint8),
                "idx": b[..., 0:32].astype(np.uint16) | ((nib & 7).astype(np.uint16) << 8), "dneg": (nib >> 3).astype(bool)}
    raise KeyError(t)


def _pack(t: str, s: dict[str, np.ndarray]) -> np.ndarray:
    """numpy state (snapped) -> rows x blocks x block bytes."""
    sh = s["d"].shape
    if t == "Q2_K":
        q = s["q"].reshape(*sh, 2, 4, 32).swapaxes(-1, -2).astype(np.uint8)
        return np.concatenate([(s["sc"] + 16 * s["m"]).astype(np.uint8), _join(q, 2).reshape(*sh, 64),
                               _h(s["d"]), _h(s["dmin"])], -1)
    if t == "Q3_K":
        sc, q = (s["sc"] + 32).astype(np.uint8), s["q"].astype(np.int16)
        lo = _join((sc & 15).reshape(*sh, 2, 8).swapaxes(-1, -2), 4)
        hi = _join((sc >> 4).reshape(*sh, 4, 4).swapaxes(-1, -2), 2)
        ql = _join((q & 3).astype(np.uint8).reshape(*sh, 2, 4, 32).swapaxes(-1, -2), 2).reshape(*sh, 64)
        hm = _join((q >= 0).astype(np.uint8).reshape(*sh, 8, 32).swapaxes(-1, -2), 1)
        return np.concatenate([hm, ql, lo, hi, _h(s["d"])], -1)
    if t in ("Q4_K", "Q5_K"):
        q = s["q"].astype(np.uint8)
        ql = _join((q & 15).reshape(*sh, 4, 2, 32).swapaxes(-1, -2), 4).reshape(*sh, 128)
        parts = [_h(s["d"]), _h(s["dmin"]), _k4pack(s["sc"].astype(np.uint8), s["m"].astype(np.uint8))]
        if t == "Q5_K":
            parts.append(_join((q >> 4).reshape(*sh, 8, 32).swapaxes(-1, -2), 1))
        return np.concatenate(parts + [ql], -1)
    if t == "Q6_K":
        q = (s["q"] + 32).astype(np.uint8)
        ql = _join((q & 15).reshape(*sh, 2, 2, 64).swapaxes(-1, -2), 4).reshape(*sh, 128)
        qh = _join((q >> 4).reshape(*sh, 2, 4, 32).swapaxes(-1, -2), 2).reshape(*sh, 64)
        return np.concatenate([ql, qh, s["sc"].astype(np.int8).view(np.uint8), _h(s["d"])], -1)
    if t == "Q8_0":
        return np.concatenate([_h(s["d"]), s["q"].astype(np.int8).view(np.uint8)], -1)
    if t == "IQ4_XS":
        sc = (s["sc"] + 32).astype(np.uint8)
        q = _join(s["q"].astype(np.uint8).reshape(*sh, 8, 2, 16).swapaxes(-1, -2), 4).reshape(*sh, 128)
        sh_ = _join((sc >> 4).astype(np.uint16), 2, np.uint16)[..., None].view(np.uint8)
        return np.concatenate([_h(s["d"]), sh_, _join((sc & 15).reshape(*sh, 4, 2), 4), q], -1)
    if t in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"):
        sign = _join(s["neg"].astype(np.uint8), 1)
        sc = s["sc"].astype(np.uint8)
        if t in ("IQ2_XXS", "IQ2_XS", "IQ3_XXS"):
            si = sign & 127
            if (KSIGNS[si] != sign).any():
                raise ValueError(f"{t}: a group of 8 has an odd number of minus signs")
        if t == "IQ2_XS":
            qs = (s["idx"].astype(np.uint16) | (si.astype(np.uint16) << 9)).view(np.uint8)
            return np.concatenate([_h(s["d"]), qs, _join(sc.reshape(*sh, 8, 2), 4)], -1)
        if t in ("IQ2_XXS", "IQ3_XXS"):
            aux = _join(si.reshape(*sh, 8, 4).astype(np.uint32), 7, np.uint32) | (sc.astype(np.uint32) << 28)
            if t == "IQ3_XXS":
                return np.concatenate([_h(s["d"]), s["idx"].astype(np.uint8), aux.view(np.uint8)], -1)
            g = s["idx"].astype(np.uint8).reshape(*sh, 8, 4).view(np.uint32)[..., 0]
            return np.concatenate([_h(s["d"]), np.stack([g, aux], -1).view(np.uint8).reshape(*sh, 64)], -1)
        if t == "IQ2_S":
            qh = _join((s["idx"] >> 8).astype(np.uint8).reshape(*sh, 8, 4), 2)
            return np.concatenate([_h(s["d"]), (s["idx"] & 255).astype(np.uint8), sign, qh,
                                   _join(sc.reshape(*sh, 8, 2), 4)], -1)
        qh = _join((s["idx"] >> 8).astype(np.uint8).reshape(*sh, 8, 8), 1)  # IQ3_S
        return np.concatenate([_h(s["d"]), (s["idx"] & 255).astype(np.uint8), qh, sign,
                               _join(sc.reshape(*sh, 4, 2), 4)], -1)
    if t == "IQ1_S":
        hi = _join((s["idx"] >> 8).astype(np.uint16).reshape(*sh, 8, 4), 3, np.uint16)
        qh = hi | (s["sc"].astype(np.uint16) << 12) | (s["dneg"].astype(np.uint16) << 15)
        return np.concatenate([_h(s["d"]), (s["idx"] & 255).astype(np.uint8), qh.view(np.uint8)], -1)
    if t == "IQ1_M":
        nib = ((s["idx"] >> 8) & 7).astype(np.uint8) | (s["dneg"].astype(np.uint8) << 3)
        dbits = s["d"].astype(np.float16).view(np.uint16)
        w = _join(s["sc"].astype(np.uint16).reshape(*sh, 4, 4), 3, np.uint16) | (_bits(dbits, 4, 4) << 12)
        return np.concatenate([(s["idx"] & 255).astype(np.uint8), _join(nib.reshape(*sh, 16, 2), 4),
                               w.view(np.uint8)], -1)
    raise KeyError(t)


def unpack(t: str, raw: np.ndarray, dev="cpu") -> dict[str, torch.Tensor]:
    """A matrix's bytes (rows x row bytes) -> state: float d (dmin) and sub-scales, integer codes."""
    b = np.ascontiguousarray(raw).reshape(raw.shape[0], -1, SIZE[t])
    st = {}
    for k, v in _unpack(t, b).items():
        v = torch.from_numpy(np.ascontiguousarray(v).astype(np.float32 if k in ("d", "dmin", "sc", "m") else
                                                             (bool if v.dtype == bool else np.int64)))
        st[k] = v.to(dev)
    return st


def snap(t: str, st: dict) -> dict:
    """The scales as the file stores them: d (and dmin) to 16 bits, sub-scales (and mins) rounded into range."""
    lo, hi = SPEC[t][3] or (0, 0)
    out = {k: v for k, v in st.items() if k in ("q", "idx", "neg", "dneg")}
    for k in ("d", "dmin"):
        if k in st:
            out[k] = st[k].detach().half().float()
    for k in ("sc", "m"):
        if k in st:
            out[k] = st[k].detach().round().clamp(lo, hi)
    return out


def pack(t: str, st: dict) -> np.ndarray:
    """State -> llama.cpp's bytes (rows x row bytes)."""
    s = {k: v.detach().cpu().numpy() for k, v in snap(t, st).items()}
    b = _pack(t, s)
    return b.reshape(b.shape[0], -1)


# --- the arithmetic --------------------------------------------------------------------------------------------

def _scales(t: str, st: dict):
    """(step, offset or None), shaped rows x blocks x sub-blocks x 1, in llama.cpp's order of operations."""
    d, sc = st["d"][..., None], st.get("sc")
    step = {"Q2_K": lambda: d * sc, "Q3_K": lambda: d * sc, "Q4_K": lambda: d * sc, "Q5_K": lambda: d * sc,
            "Q6_K": lambda: d * sc, "Q8_0": lambda: d, "IQ4_XS": lambda: d * sc,
            "IQ2_XXS": lambda: d * (0.5 + sc) * 0.25, "IQ2_XS": lambda: d * (0.5 + sc) * 0.25,
            "IQ2_S": lambda: d * (0.5 + sc) * 0.25, "IQ3_XXS": lambda: d * (0.5 + sc) * 0.5,
            "IQ3_S": lambda: d * (1 + 2 * sc), "IQ1_S": lambda: d * (2 * sc + 1), "IQ1_M": lambda: d * (2 * sc + 1)}[t]()
    off = (st["dmin"][..., None] * st["m"])[..., None] if t in OFFSET else None
    return step[..., None], off


def _value(t: str, c: dict, dev) -> torch.Tensor:
    """Codes -> the value of every weight, rows x blocks x block weights."""
    kind = SPEC[t][4]
    if kind == "level":
        return c["q"].float()
    if kind == "table":
        return KVALUES.to(dev)[c["q"]]
    g = grid(t, dev)[c["idx"]]  # rows x blocks x patterns x width
    sh = g.shape[:2]
    if kind == "grid":
        return g.reshape(*sh, -1) * torch.where(c["neg"], -1.0, 1.0).reshape(*sh, -1)
    per = SPEC[t][5] // 8  # IQ1: patterns sharing one shift
    delta = torch.where(c["dneg"], -0.125, 0.125).repeat_interleave(per, -1)[..., None]
    return (g + delta).reshape(*sh, -1)


def _sub(t: str, x: torch.Tensor) -> torch.Tensor:
    """rows x (blocks * block weights) -> rows x blocks x sub-blocks x sub-block weights."""
    _, n, s, _, _, _ = SPEC[t]
    return x.reshape(x.shape[0], -1, s, n // s)


def decode(t: str, st: dict) -> torch.Tensor:
    """State -> weights (rows x columns), exactly as llama.cpp dequantizes them."""
    s = snap(t, st)
    step, off = _scales(t, s)
    w = step * _sub(t, _value(t, s, s["d"].device).reshape(s["d"].shape[0], -1))
    if off is not None:
        w = w - off
    return w.reshape(w.shape[0], -1)


def _nearest(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Index of the nearest pattern of g (n x width) to every row of a (m x width)."""
    g2, out = (g * g).sum(-1), torch.empty(a.shape[0], dtype=torch.long, device=a.device)
    for i in range(0, a.shape[0], CHUNK):
        out[i:i + CHUNK] = (g2 - 2 * a[i:i + CHUNK] @ g.T).argmin(-1)
    return out


def encode(t: str, W: torch.Tensor, st: dict) -> dict:
    """The codes nearest to W (rows x columns) under st's scales, plain squared error. Always in fp32, whatever
    autocast the caller is under: in bf16 the pattern search's distances are too coarse and 4-8 % of IQ groups get
    another code, so codes picked in training, at a check and at the save would not be the same codes."""
    s = snap(t, st)
    kind, extra = SPEC[t][4], SPEC[t][5]
    with torch.no_grad(), torch.autocast(W.device.type, enabled=False):
        step, off = _scales(t, s)
        u = _sub(t, W.float()) + (off if off is not None else 0)
        u = (u / torch.where(step == 0, torch.ones_like(step), step)).reshape(*s["d"].shape, -1)  # value units
        if kind == "level":
            return {"q": torch.nan_to_num(u.round(), 0).clamp(*extra).long()}
        if kind == "table":
            return {"q": (u[..., None] - KVALUES.to(u.device)).abs().argmin(-1)}
        g = grid(t, u.device)
        if kind == "grid":
            width, signs = extra
            w8 = u.reshape(*u.shape[:2], -1, 8)
            neg = w8 < 0
            if signs == "parity":  # only an even number of minus signs per 8: an odd group flips its smallest
                odd = neg.sum(-1, keepdim=True) % 2 == 1
                neg = neg ^ (odd & torch.nn.functional.one_hot(w8.abs().argmin(-1), 8).bool())
            a = (w8 * torch.where(neg, -1.0, 1.0)).reshape(-1, width)  # negative where a sign was flipped
            return {"idx": _nearest(a, g).view(*u.shape[:2], -1), "neg": neg}
        per = extra // 8  # IQ1: a +-1/8 shift shared by `per` patterns; the better of the two
        a, best = u.reshape(-1, 8), None
        for sign in (1.0, -1.0):
            idx = _nearest(a - 0.125 * sign, g)
            err = ((a - 0.125 * sign - g[idx]) ** 2).sum(-1).view(*u.shape[:2], -1, per).sum(-1)
            best = (idx, err) if best is None else best
            take = (err < best[1]).repeat_interleave(per, -1).view(-1)
            best = (torch.where(take, idx, best[0]), torch.minimum(err, best[1]))
            if sign == 1.0:
                pos_err = err
        return {"idx": best[0].view(*u.shape[:2], -1), "dneg": best[1] < pos_err}


def fake(t: str, W: torch.Tensor, st: dict, codes: dict | None = None) -> torch.Tensor:
    """W's quantized value (forward: exactly decode(encode(W)), or decode(codes) for codes chosen earlier), with
    gradients to W and to st's float scales."""
    c = codes or encode(t, W, st)
    wq = decode(t, {**st, **c})
    lo, hi = SPEC[t][3] or (0, 0)
    sn = {k: v + ((v.round().clamp(lo, hi) if k in ("sc", "m") else v.half().float()) - v).detach()
          for k, v in st.items() if k in ("d", "dmin", "sc", "m")}
    step, off = _scales(t, sn)
    val = _sub(t, _value(t, c, W.device).reshape(W.shape[0], -1))
    u = ((_sub(t, W) + (off if off is not None else 0)) / step.clamp_min(1e-30)).detach()
    lsq = step * (val - u).detach()
    if off is not None:  # weights clamped at the ends of 0..max: the min moves them
        lsq = lsq - off * (val != u.round()).float()
    lsq = lsq.reshape(wq.shape)
    return wq + (W - W.detach()) + (lsq - lsq.detach())  # both added terms are exactly 0 in the forward pass
