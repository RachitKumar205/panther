"""llama.cpp's 13 types in PyTorch, exactly: a matrix's bytes as scales and codes, decoded bit for bit as llama.cpp
decodes them, and the codes nearest to any weights under those scales. QAT trains through them.

Every type decodes as  w = step × value − offset,  per sub-block of a block (256 weights; Q8_0: 32):
  step    from the block's 16-bit scale d and an integer sub-scale (each type its own formula, in llama.cpp's order);
  offset  dmin × m for Q2_K, Q4_K and Q5_K (else 0);
  value   the code: an integer level (K types, Q8_0), one of 16 table levels (IQ4_XS), a pattern of 8 or 4 magnitudes
          from a fixed grid times signs (IQ2 and IQ3 types), or a grid pattern of −1/0/1 plus a ±1/8 shift (IQ1 types).
A matrix is held as its state: d (and dmin), sub-scales (and mins), codes. Every use of the scales first snaps them to
what the file stores (16-bit d, integer sub-scales in range).

Ported from Rachit's fakequant.py (origin/rachit), IQ3_S dropped (not one of our types)."""

from __future__ import annotations

from dataclasses import dataclass

import gguf.quants as gq
import numpy as np
import torch

torch.set_float32_matmul_precision("highest")  # the code search's distances are fp32 matrix products: never TF32

QK = 256
CHUNK = 1 << 18  # groups searched against a grid at once (256k groups × 2,048 patterns × 4 bytes = 2 GiB)
KSIGNS = np.frombuffer(gq.IQ2_XXS.ksigns, np.uint8)  # a 7-bit sign index → 8 sign bits (bit 7: the parity)
KVALUES = torch.tensor(gq.IQ4_NL.kvalues, dtype=torch.float32)  # IQ4_XS's 16 levels
OFFSET = ("Q2_K", "Q4_K", "Q5_K")  # types with a per-sub-block minimum


def _pattern_table(name: str) -> torch.Tensor:
    """A type's patterns (n × width), from the tables gguf-py ships (IQ1_M uses IQ1_S's)."""
    q = getattr(gq, name)
    q.init_grid()
    return torch.from_numpy(np.ascontiguousarray(q.grid.reshape(q.grid_shape)))


PATTERNS = {t: _pattern_table("IQ1_S" if t == "IQ1_M" else t)
            for t in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ1_S", "IQ1_M")}


@dataclass(frozen=True)
class Spec:
    block_bytes: int
    block_weights: int
    sub_blocks: int
    sub_scale: tuple[int, int] | None  # the integer sub-scale's range
    kind: str  # "level" (integer codes), "table" (IQ4_XS), "grid" (IQ2/IQ3), "iq1"
    # level: the code range; grid: (pattern width, signs "parity" | "free"); iq1: weights per shift
    extra: tuple | int | None


SPECS = {
    "Q2_K": Spec(84, QK, 16, (0, 15), "level", (0, 3)), "Q3_K": Spec(110, QK, 16, (-32, 31), "level", (-4, 3)),
    "Q4_K": Spec(144, QK, 8, (0, 63), "level", (0, 15)), "Q5_K": Spec(176, QK, 8, (0, 63), "level", (0, 31)),
    "Q6_K": Spec(210, QK, 16, (-128, 127), "level", (-32, 31)), "Q8_0": Spec(34, 32, 1, None, "level", (-128, 127)),
    "IQ4_XS": Spec(136, QK, 8, (-32, 31), "table", None),
    "IQ2_XXS": Spec(66, QK, 8, (0, 15), "grid", (8, "parity")),
    "IQ2_XS": Spec(74, QK, 16, (0, 15), "grid", (8, "parity")),
    "IQ2_S": Spec(82, QK, 16, (0, 15), "grid", (8, "free")),
    "IQ3_XXS": Spec(98, QK, 8, (0, 15), "grid", (4, "parity")),
    "IQ1_S": Spec(50, QK, 8, (0, 7), "iq1", 32), "IQ1_M": Spec(56, QK, 16, (0, 7), "iq1", 8)}


# --- bytes ↔ state (llama.cpp's block layouts, ggml-common.h) -------------------------------------------------------

def _bits(x: np.ndarray, n: int, width: int) -> np.ndarray:
    """The n fields of `width` bits in each element of x, lowest first, as a new last axis."""
    return (x[..., None] >> (np.arange(n, dtype=x.dtype) * width)) & ((1 << width) - 1)


def _join(f: np.ndarray, width: int, dtype=np.uint8) -> np.ndarray:
    """The inverse of _bits: the last axis' fields packed, lowest first."""
    return (f.astype(dtype) << (np.arange(f.shape[-1], dtype=dtype) * width)).sum(-1, dtype=dtype)


def _f16(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.float16)[..., 0].astype(np.float32)


def _h(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)[..., None].view(np.uint8)


def _unpack(t: str, b: np.ndarray) -> dict[str, np.ndarray]:
    """Blocks (rows × blocks × block bytes) → numpy state, codes in the block's weight order."""
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
        s = b[..., 4:16]
        a, m, md = s[..., 0:4], s[..., 4:8], s[..., 8:12]  # 8 six-bit scales and 8 six-bit mins in 12 bytes
        sc = np.concatenate([a & 63, (md & 15) | ((a >> 2) & 48)], -1)
        mins = np.concatenate([m & 63, (md >> 4) | ((m >> 2) & 48)], -1)
        qs = b[..., 16:144] if t == "Q4_K" else b[..., 48:176]
        q = _bits(qs.reshape(*sh, 4, 32), 2, 4).swapaxes(-1, -2).reshape(*sh, QK)  # chunk, nibble, byte
        if t == "Q5_K":
            q = q | (_bits(b[..., 16:48], 8, 1).swapaxes(-1, -2).reshape(*sh, QK) << 4)
        return {"d": _f16(b[..., 0:2]), "dmin": _f16(b[..., 2:4]), "sc": sc, "m": mins, "q": q}
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
    if t == "IQ1_S":
        qh = b[..., 34:50].copy().view(np.uint16)
        hi = _bits(qh & 4095, 4, 3).reshape(*sh, 32)
        return {"d": _f16(b[..., 0:2]), "sc": ((qh >> 12) & 7).astype(np.uint8),
                "idx": b[..., 2:34].astype(np.uint16) | (hi << 8), "dneg": (qh >> 15).astype(bool)}
    if t == "IQ1_M":
        w = b[..., 48:56].copy().view(np.uint16)
        d = (w >> 12).astype(np.uint16) << (np.arange(4, dtype=np.uint16) * 4)  # d is spread over the 4 words' tops
        nib = _bits(b[..., 32:48], 2, 4).reshape(*sh, 32)
        return {"d": d.sum(-1, dtype=np.uint16).view(np.float16).astype(np.float32),
                "sc": _bits(w & 4095, 4, 3).reshape(*sh, 16).astype(np.uint8),
                "idx": b[..., 0:32].astype(np.uint16) | ((nib & 7).astype(np.uint16) << 8),
                "dneg": (nib >> 3).astype(bool)}
    raise KeyError(t)


def _pack(t: str, s: dict[str, np.ndarray]) -> np.ndarray:
    """Numpy state (snapped) → blocks (rows × blocks × block bytes)."""
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
        q, sc, m = s["q"].astype(np.uint8), s["sc"].astype(np.uint8), s["m"].astype(np.uint8)
        scales = np.concatenate([(sc[..., :4] & 63) | ((sc[..., 4:] >> 4) << 6),
                                 (m[..., :4] & 63) | ((m[..., 4:] >> 4) << 6),
                                 (sc[..., 4:] & 15) | ((m[..., 4:] & 15) << 4)], -1).astype(np.uint8)
        parts = [_h(s["d"]), _h(s["dmin"]), scales]
        if t == "Q5_K":
            parts.append(_join((q >> 4).reshape(*sh, 8, 32).swapaxes(-1, -2), 1))
        low = _join((q & 15).reshape(*sh, 4, 2, 32).swapaxes(-1, -2), 4).reshape(*sh, 128)
        return np.concatenate([*parts, low], -1)
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
        high = _join((sc >> 4).astype(np.uint16), 2, np.uint16)[..., None].view(np.uint8)
        return np.concatenate([_h(s["d"]), high, _join((sc & 15).reshape(*sh, 4, 2), 4), q], -1)
    if t in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS"):
        sign, sc = _join(s["neg"].astype(np.uint8), 1), s["sc"].astype(np.uint8)
        if t == "IQ2_S":
            qh = _join((s["idx"] >> 8).astype(np.uint8).reshape(*sh, 8, 4), 2)
            return np.concatenate([_h(s["d"]), (s["idx"] & 255).astype(np.uint8), sign, qh,
                                   _join(sc.reshape(*sh, 8, 2), 4)], -1)
        si = sign & 127  # the other three store 7 sign bits; the 8th is the parity
        assert (KSIGNS[si] == sign).all(), f"{t}: a group of 8 with an odd number of minus signs"
        if t == "IQ2_XS":
            qs = (s["idx"].astype(np.uint16) | (si.astype(np.uint16) << 9)).view(np.uint8)
            return np.concatenate([_h(s["d"]), qs, _join(sc.reshape(*sh, 8, 2), 4)], -1)
        aux = _join(si.reshape(*sh, 8, 4).astype(np.uint32), 7, np.uint32) | (sc.astype(np.uint32) << 28)
        if t == "IQ3_XXS":
            return np.concatenate([_h(s["d"]), s["idx"].astype(np.uint8), aux.view(np.uint8)], -1)
        g = s["idx"].astype(np.uint8).reshape(*sh, 8, 4).view(np.uint32)[..., 0]
        return np.concatenate([_h(s["d"]), np.stack([g, aux], -1).view(np.uint8).reshape(*sh, 64)], -1)
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


# --- the arithmetic ---------------------------------------------------------------------------------------------------

class Grid:
    """One of llama.cpp's types, in PyTorch. A state is a dict of tensors: d (and dmin), sc (and m), and the codes
    (q; or idx with neg or dneg)."""

    def __init__(self, type_: str) -> None:
        self.type, self.spec = type_, SPECS[type_]

    def unpack(self, raw: np.ndarray, device="cpu") -> dict[str, torch.Tensor]:
        """A matrix's bytes (rows × row bytes) → its state."""
        blocks = np.ascontiguousarray(raw).reshape(raw.shape[0], -1, self.spec.block_bytes)
        state = {}
        for k, v in _unpack(self.type, blocks).items():
            kind = np.float32 if k in ("d", "dmin", "sc", "m") else bool if v.dtype == bool else np.int64
            state[k] = torch.from_numpy(np.ascontiguousarray(v).astype(kind)).to(device)
        return state

    def snap(self, state: dict) -> dict:
        """The scales as the file stores them: d (and dmin) in 16 bits, sub-scales (and mins) integers in range."""
        lo, hi = self.spec.sub_scale or (0, 0)
        out = {k: v for k, v in state.items() if k in ("q", "idx", "neg", "dneg")}
        out |= {k: state[k].detach().half().float() for k in ("d", "dmin") if k in state}
        out |= {k: state[k].detach().round().clamp(lo, hi) for k in ("sc", "m") if k in state}
        return out

    def pack(self, state: dict) -> np.ndarray:
        """The state → llama.cpp's bytes (rows × row bytes)."""
        blocks = _pack(self.type, {k: v.detach().cpu().numpy() for k, v in self.snap(state).items()})
        return blocks.reshape(blocks.shape[0], -1)

    def scales(self, s: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(step, offset or None), shaped rows × blocks × sub-blocks × 1, in llama.cpp's order of operations."""
        d, sc, t = s["d"][..., None], s.get("sc"), self.type
        if t == "Q8_0":
            step = d
        elif t in ("IQ2_XXS", "IQ2_XS", "IQ2_S"):
            step = d * (0.5 + sc) * 0.25
        elif t == "IQ3_XXS":
            step = d * (0.5 + sc) * 0.5
        elif t in ("IQ1_S", "IQ1_M"):
            step = d * (2 * sc + 1)
        else:  # K types, IQ4_XS
            step = d * sc
        offset = (s["dmin"][..., None] * s["m"])[..., None] if t in OFFSET else None
        return step[..., None], offset

    def values(self, codes: dict, device) -> torch.Tensor:
        """Codes → every weight's value, rows × blocks × block weights."""
        kind = self.spec.kind
        if kind == "level":
            return codes["q"].float()
        if kind == "table":
            return KVALUES.to(device)[codes["q"]]
        g = PATTERNS[self.type].to(device)[codes["idx"]]  # rows × blocks × patterns × width
        sh = g.shape[:2]
        if kind == "grid":
            return g.reshape(*sh, -1) * torch.where(codes["neg"], -1.0, 1.0).reshape(*sh, -1)
        shift = torch.where(codes["dneg"], -0.125, 0.125).repeat_interleave(self.spec.extra // 8, -1)[..., None]
        return (g + shift).reshape(*sh, -1)

    def sub(self, x: torch.Tensor) -> torch.Tensor:
        """rows × columns → rows × blocks × sub-blocks × sub-block weights."""
        return x.reshape(x.shape[0], -1, self.spec.sub_blocks, self.spec.block_weights // self.spec.sub_blocks)

    def decode(self, state: dict) -> torch.Tensor:
        """The state → weights (rows × columns), exactly as llama.cpp dequantizes them."""
        s = self.snap(state)
        step, offset = self.scales(s)
        w = step * self.sub(self.values(s, s["d"].device).reshape(s["d"].shape[0], -1))
        return (w if offset is None else w - offset).reshape(w.shape[0], -1)

    def encode(self, W: torch.Tensor, state: dict) -> dict:
        """The codes nearest to W (rows × columns) under the state's scales, plain squared error. Always in fp32,
        whatever autocast the caller is under: in bf16 the pattern search's distances are too coarse and 4–8 % of IQ
        groups get another code, so training, a check and the save would not pick the same codes."""
        s = self.snap(state)
        kind, extra = self.spec.kind, self.spec.extra
        with torch.no_grad(), torch.autocast(W.device.type, enabled=False):
            step, offset = self.scales(s)
            u = self.sub(W.float()) + (offset if offset is not None else 0)
            u = (u / torch.where(step == 0, torch.ones_like(step), step)).reshape(*s["d"].shape, -1)  # in value units
            if kind == "level":
                return {"q": torch.nan_to_num(u.round(), 0).clamp(*extra).long()}
            if kind == "table":
                return {"q": (u[..., None] - KVALUES.to(u.device)).abs().argmin(-1)}
            g = PATTERNS[self.type].to(u.device)
            if kind == "grid":
                width, signs = extra
                w8 = u.reshape(*u.shape[:2], -1, 8)
                neg = w8 < 0
                if signs == "parity":  # an even number of minus signs per 8: an odd group flips its smallest
                    odd = neg.sum(-1, keepdim=True) % 2 == 1
                    neg = neg ^ (odd & torch.nn.functional.one_hot(w8.abs().argmin(-1), 8).bool())
                a = (w8 * torch.where(neg, -1.0, 1.0)).reshape(-1, width)
                return {"idx": nearest(a, g).view(*u.shape[:2], -1), "neg": neg}
            per, a = extra // 8, u.reshape(-1, 8)  # IQ1: a ±1/8 shift shared by `per` patterns; the better of the two
            found = {}
            for sign in (1.0, -1.0):
                idx = nearest(a - 0.125 * sign, g)
                err = ((a - 0.125 * sign - g[idx]) ** 2).sum(-1).view(*u.shape[:2], -1, per).sum(-1)
                found[sign] = (idx, err)
            minus = found[-1.0][1] < found[1.0][1]
            idx = torch.where(minus.repeat_interleave(per, -1).view(-1), found[-1.0][0], found[1.0][0])
            return {"idx": idx.view(*u.shape[:2], -1), "dneg": minus}

    def fake(self, W: torch.Tensor, state: dict, codes: dict | None = None) -> torch.Tensor:
        """W's quantized value in the forward pass (exactly decode(encode(W)), or decode(codes) for codes picked
        earlier), with its gradient straight through to W (the added term is exactly 0 in the forward pass)."""
        return self.decode({**state, **(codes or self.encode(W, state))}) + (W - W.detach())


def nearest(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """The index of the nearest pattern of g (n × width) to every row of a (m × width)."""
    g2, out = (g * g).sum(-1), torch.empty(a.shape[0], dtype=torch.long, device=a.device)
    for i in range(0, a.shape[0], CHUNK):
        out[i:i + CHUNK] = (g2 - 2 * a[i:i + CHUNK] @ g.T).argmin(-1)
    return out
