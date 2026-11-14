"""The fake quantizers against llama.cpp: decoding bit for bit, packing byte for byte, sane encoding, gradients.
Needs torch (the server image). PANTHER_FQ_FILE=<gguf> also round-trips every Q2_K / IQ2_XS / IQ2_S matrix of a
real file; llama.cpp's C library, where present, is a second reference besides the gguf package."""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from panther import fakequant as fq  # noqa: E402

LIB = Path("/opt/llama.cpp/build-cuda/bin/libggml-base.so")
QT = {t: gguf.GGMLQuantizationType[t] for t in fq.TYPES}
rng = np.random.default_rng(0)


def random_raw(t: str, rows: int = 8, blocks: int = 4) -> np.ndarray:
    """Random valid bytes: any codes, scales and signs; finite 16-bit d (and dmin)."""
    b = rng.integers(0, 256, (rows, blocks, fq.SIZE[t]), dtype=np.uint8)
    for off in ((80, 82) if t == "Q2_K" else (0,)):
        b[..., off:off + 2] = (rng.standard_normal((rows, blocks)) * 1e-2).astype(np.float16)[..., None].view(np.uint8)
    return b.reshape(rows, -1)


def c_decode(t: str, raw: np.ndarray) -> np.ndarray:
    """llama.cpp's own C dequantize (ggml type traits' to_float)."""
    from panther.rounding import _TO_FLOAT, _Traits
    lib = ctypes.CDLL(str(LIB))
    lib.ggml_get_type_traits.restype, lib.ggml_get_type_traits.argtypes = ctypes.POINTER(_Traits), [ctypes.c_int]
    raw = np.ascontiguousarray(raw)
    out = np.empty((raw.shape[0], raw.shape[1] // fq.SIZE[t] * 256), np.float32)
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
    from panther.rounding import quant
    W = rng.standard_normal((64, 1024)).astype(np.float32) * 0.02
    codes, deq = quant(W, np.ones(1024, np.float32), QT[t], 4)  # the IQ2 types need importance data
    st = fq.unpack(t, codes)
    ours = fq.decode(t, {**st, **fq.encode(t, torch.from_numpy(W), st)}).numpy()
    assert ((ours - W) ** 2).sum() <= 1.02 * ((deq - W) ** 2).sum()


@pytest.mark.parametrize("t", fq.TYPES)
def test_fake_forward_exact_and_gradients(t):
    st = {k: (v.clone().requires_grad_(True) if v.is_floating_point() else v) for k, v in fq.unpack(t, random_raw(t))
          .items()}
    W = (fq.decode(t, st) + torch.randn(8, 4 * 256) * 1e-3).requires_grad_(True)
    out = fq.fake(t, W, st)
    ref = fq.decode(t, {**st, **fq.encode(t, W, st)})
    assert torch.equal(out.detach(), ref)  # the forward pass is exactly what llama.cpp would run
    (out * torch.randn_like(out)).sum().backward()
    assert torch.equal(W.grad != 0, torch.ones_like(W, dtype=torch.bool))  # straight through
    for k in ("d", "sc", "dmin", "m", "s"):
        if k in st:
            assert torch.isfinite(st[k].grad).all() and st[k].grad.abs().sum() > 0, k
    packed = fq.pack(t, {**st, **fq.encode(t, W, st)})  # what training ends with is a valid file
    assert np.array_equal(fq.decode(t, fq.unpack(t, packed)).numpy(), ref.numpy())
