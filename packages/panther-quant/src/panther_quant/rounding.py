"""llama.cpp's quantizer via ctypes (the quantize step of GPTQ; ADR 0024).

Ported from the rachit branch. This is ctypes/numpy glue over ``libggml-base.so``, not an algorithm we
own, so its internals are not held to strict typing.
"""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_lib: Any = None
_TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)  # releases the GIL


class _Traits(ctypes.Structure):  # struct ggml_type_traits (ggml.h)
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


def quant(w, imp, qtype, bin_dir: Path, threads: int = 16, deq: bool = True):
    """``(codes, dequantized)`` of the rows of ``w`` in ``qtype``, by llama.cpp's quantizer."""
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(bin_dir / "libggml-base.so"))
        _lib.ggml_quantize_chunk.restype, _lib.ggml_row_size.restype = ctypes.c_size_t, ctypes.c_size_t
        _lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int64] * 3 + [ctypes.c_void_p]
        _lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        _lib.ggml_get_type_traits.restype, _lib.ggml_get_type_traits.argtypes = ctypes.POINTER(_Traits), [ctypes.c_int]
    t, (rows, n) = int(qtype.value), w.shape
    _lib.ggml_quantize_init(t)
    w = np.ascontiguousarray(w, np.float32)
    imp = None if imp is None else np.ascontiguousarray(imp, np.float32)
    dst = np.empty((rows, _lib.ggml_row_size(t, n)), np.uint8)
    back = np.empty((rows, n), np.float32) if deq else None
    to_float = _TO_FLOAT(_lib.ggml_get_type_traits(t).contents.to_float)

    def part(idx):
        a, b = int(idx[0]), int(idx[-1]) + 1
        _lib.ggml_quantize_chunk(t, w[a:b].ctypes.data, dst[a:b].ctypes.data, 0, b - a, n, None if imp is None else imp.ctypes.data)
        if deq:
            to_float(dst[a:b].ctypes.data, back[a:b].ctypes.data, (b - a) * n)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(part, [i for i in np.array_split(np.arange(rows), threads) if len(i)]))
    return dst, back
