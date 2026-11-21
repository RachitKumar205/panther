"""llama.cpp's own quantizer, called directly, so our codes are exactly the bytes llama.cpp would write."""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor

import gguf
import numpy as np


class Quantizer:
    """ggml_quantize_chunk from llama.cpp's libggml-base (found on the library path)."""

    def __init__(self, threads: int = 8) -> None:
        self.threads = threads
        self.lib = ctypes.CDLL("libggml-base.so")
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                                 ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]

    def quantize(self, w: np.ndarray, importance: np.ndarray, type_: str) -> tuple[np.ndarray, np.ndarray]:
        """w (rows x columns) in type_: (codes, values). codes are the bytes llama.cpp stores, values what they decode
        to (decoded by gguf-py). importance: one weight per column. llama.cpp's quantizer takes float32."""
        t = gguf.GGMLQuantizationType[type_]
        w = np.ascontiguousarray(w, np.float32)
        importance = np.ascontiguousarray(importance, np.float32)
        codes = np.empty(gguf.quant_shape_to_byte_shape(w.shape, t), np.uint8)
        self.lib.ggml_quantize_init(t)

        def rows(span: np.ndarray) -> None:  # C without the GIL: the row spans run in parallel
            a, b = span[0], span[-1] + 1
            self.lib.ggml_quantize_chunk(t, w[a:b].ctypes.data, codes[a:b].ctypes.data, 0, b - a, w.shape[1],
                                         importance.ctypes.data)

        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(rows, np.array_split(np.arange(len(w)), min(self.threads, len(w)))))
        return codes, gguf.quants.dequantize(codes, t)
