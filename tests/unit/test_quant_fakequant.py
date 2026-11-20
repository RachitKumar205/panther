"""The fake quantizers (all 14 types) against llama.cpp: decoding bit for bit, packing byte for byte, sane encoding,
gradients to the weights.
Needs torch (the server image). PANTHER_FQ_FILE=<gguf> also round-trips every matrix of a
real file (any of the 14 types in it); llama.cpp's C library, where present, is a second reference
besides the gguf package."""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from panther_quant import fakequant as fq  # noqa: E402

LIB = Path("/opt/llama.cpp/build-cuda/bin/libggml-base.so")
QT = {t: gguf.GGMLQuantizationType[t] for t in fq.TYPES}
rng = np.random.default_rng(0)


def random_raw(t: str, rows: int = 8, blocks: int = 4) -> np.ndarray:
    """Random valid bytes: any codes, scales and signs; a finite 16-bit d (and dmin)."""
    from panther_quant.distill import D_OFF
    b = rng.integers(0, 256, (rows, blocks, fq.SPEC[t][0]), dtype=np.uint8)
    d = (rng.standard_normal((rows, blocks, 2)) * 1e-2).astype(np.float16)
    if t == "IQ1_M":  # d is spread over the top 4 bits of the 4 scale words
        w = b[..., 48:56].copy().view(np.uint16)
        nib = (d[..., 0].view(np.uint16)[..., None] >> (np.arange(4, dtype=np.uint16) * 4)) & 15
        b[..., 48:56] = ((w & 4095) | (nib << 12)).astype(np.uint16).view(np.uint8)
    else:
        for k, off in enumerate(D_OFF[t]):
            b[..., off:off + 2] = d[..., k:k + 1].view(np.uint8)
    return b.reshape(rows, -1)


def c_decode(t: str, raw: np.ndarray) -> np.ndarray:
    """llama.cpp's own C dequantize (ggml type traits' to_float)."""
    from panther_quant.rounding import _TO_FLOAT, _Traits
    lib = ctypes.CDLL(str(LIB))
    lib.ggml_get_type_traits.restype, lib.ggml_get_type_traits.argtypes = ctypes.POINTER(_Traits), [ctypes.c_int]
    raw = np.ascontiguousarray(raw)
    out = np.empty((raw.shape[0], raw.shape[1] // fq.SPEC[t][0] * fq.SPEC[t][1]), np.float32)
    _TO_FLOAT(lib.ggml_get_type_traits(int(QT[t].value)).contents.to_float)(raw.ctypes.data, out.ctypes.data, out.size)
    return out


@pytest.mark.parametrize("t", fq.TYPES)
def test_decode_and_pack_exact(t):
    raw = random_raw(t)
    st = fq.unpack(t, raw)
    ours = fq.decode(t, st).numpy()
    ref = dequantize(raw, QT[t]).reshape(ours.shape)
    assert np.array_equal(ours.view(np.uint32), ref.view(np.uint32))  # bit for bit, -0.0 included
    if LIB.exists():
        assert np.array_equal(ours.view(np.uint32), c_decode(t, raw).view(np.uint32))
    assert np.array_equal(fq.pack(t, st), raw)  # every byte comes back


@pytest.mark.skipif(not os.environ.get("PANTHER_FQ_FILE"), reason="PANTHER_FQ_FILE not set")
def test_real_file_round_trip():
    seen = set()
    for x in gguf.GGUFReader(os.environ["PANTHER_FQ_FILE"]).tensors:
        t = x.tensor_type.name
        if t in fq.TYPES:
            raw = np.asarray(x.data)
            st = fq.unpack(t, raw)
            assert np.array_equal(fq.pack(t, st), raw), x.name
            assert np.array_equal(fq.decode(t, st).numpy().ravel(), dequantize(raw, x.tensor_type).ravel()), x.name
            seen.add(t)
    assert seen


@pytest.mark.skipif(not LIB.exists(), reason="needs llama.cpp's quantizer")
@pytest.mark.parametrize("t", fq.TYPES)
def test_encode_beats_or_matches_llama(t):
    """With llama.cpp's own scales, our codes are at least about as close to W (plain squared error)."""
    from panther_quant.rounding import quant
    W = rng.standard_normal((64, 1024)).astype(np.float32) * 0.02
    codes, deq = quant(W, np.ones(1024, np.float32), QT[t], 4)  # the IQ2 types need importance data
    st = fq.unpack(t, codes)
    ours = fq.decode(t, {**st, **fq.encode(t, torch.from_numpy(W), st)}).numpy()
    assert ((ours - W) ** 2).sum() <= 1.02 * ((deq - W) ** 2).sum()


@pytest.mark.parametrize("t", fq.TYPES)
def test_encode_ignores_autocast(t):
    """Codes are picked in fp32 whatever autocast the caller is under: training and its checks run under bf16, the
    save does not, and all three must pick the same codes."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = fq.unpack(t, random_raw(t, 32, 8), dev)
    W = fq.decode(t, st)
    W = W + torch.randn_like(W) * W.std() * 0.05  # off the grid, as trained weights are
    plain = fq.encode(t, W, st)
    with torch.autocast(dev, torch.bfloat16):
        cast = fq.encode(t, W, st)
    assert all(torch.equal(plain[k], cast[k]) for k in plain)


@pytest.mark.parametrize("t", fq.TYPES)
def test_fake_forward_exact_and_gradients(t):
    st = fq.unpack(t, random_raw(t))
    W = fq.decode(t, st)
    W = (W + torch.randn_like(W) * 1e-3).requires_grad_(True)
    out = fq.fake(t, W, st)
    ref = fq.decode(t, {**st, **fq.encode(t, W, st)})
    assert torch.equal(out.detach(), ref)  # the forward pass is exactly what llama.cpp would run
    (out * torch.randn_like(out)).sum().backward()
    assert torch.equal(W.grad != 0, torch.ones_like(W, dtype=torch.bool))  # straight through
    packed = fq.pack(t, {**st, **fq.encode(t, W, st)})  # what training ends with is a valid file
    assert np.array_equal(fq.decode(t, fq.unpack(t, packed)).numpy(), ref.numpy())
