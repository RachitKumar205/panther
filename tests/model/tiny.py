"""A tiny Qwen2-shaped 16-bit GGUF and importance matrix, small enough to build every type copy in seconds."""

from __future__ import annotations

import shutil
import zlib
from ctypes.util import find_library
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from tokenizers.pre_tokenizers import ByteLevel

# A byte-level vocabulary (one token per byte, plus one merge, which llama.cpp requires): any text tokenizes, so
# llama-perplexity can run on this model.
TOKENS = [*sorted(ByteLevel.alphabet()), "th"]
HIDDEN, INTER, LAYERS, VOCAB, HEADS, KV_HEADS = 256, 512, 2, len(TOKENS), 4, 2
KV = KV_HEADS * HIDDEN // HEADS
SHAPES = {"attn_q": (HIDDEN, HIDDEN), "attn_k": (KV, HIDDEN), "attn_v": (KV, HIDDEN), "attn_output": (HIDDEN, HIDDEN),
          "ffn_gate": (INTER, HIDDEN), "ffn_up": (INTER, HIDDEN), "ffn_down": (HIDDEN, INTER)}

needs_llama = pytest.mark.skipif(not (shutil.which("llama-quantize") and find_library("ggml-base")),
                                 reason="needs llama.cpp (llama-quantize and libggml-base)")
needs_ggml = pytest.mark.skipif(not find_library("ggml-base"), reason="needs llama.cpp's libggml-base")
needs_perplexity = pytest.mark.skipif(not (shutil.which("llama-perplexity") and shutil.which("llama-quantize")),
                                      reason="needs llama.cpp (llama-perplexity and llama-quantize)")


def bf16(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_original(path: Path, tied: bool = False, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(path, "qwen2")
    w.add_context_length(512)
    w.add_embedding_length(HIDDEN)
    w.add_block_count(LAYERS)
    w.add_feed_forward_length(INTER)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1e6)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(TOKENS)
    w.add_token_types([1] * VOCAB)
    w.add_token_merges(["t h"])

    def matrix(name: str, rows: int, cols: int) -> None:
        w.add_tensor(name, bf16(rng.normal(0, 0.05, (rows, cols))), raw_dtype=gguf.GGMLQuantizationType.BF16)

    matrix("token_embd.weight", VOCAB, HIDDEN)
    for layer in range(LAYERS):
        for role, (rows, cols) in SHAPES.items():
            matrix(f"blk.{layer}.{role}.weight", rows, cols)
        for role, n in (("attn_q", HIDDEN), ("attn_k", KV), ("attn_v", KV)):
            w.add_tensor(f"blk.{layer}.{role}.bias", rng.normal(0, 0.01, n).astype(np.float32))
        w.add_tensor(f"blk.{layer}.attn_norm.weight", rng.uniform(0.5, 1.5, HIDDEN).astype(np.float32))
        w.add_tensor(f"blk.{layer}.ffn_norm.weight", rng.uniform(0.5, 1.5, HIDDEN).astype(np.float32))
    w.add_tensor("output_norm.weight", np.ones(HIDDEN, np.float32))
    if not tied:
        matrix("output.weight", VOCAB, HIDDEN)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_imatrix(original: Path, path: Path, seed: int = 1) -> None:
    """An importance matrix in llama-imatrix's GGUF format (with --process-output): every matrix but the word table."""
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(path, arch="")
    w.add_type("imatrix")
    w.add_array("imatrix.datasets", ["tiny"])
    w.add_uint32("imatrix.chunk_count", 1)
    w.add_uint32("imatrix.chunk_size", 512)
    for t in gguf.GGUFReader(original).tensors:
        if len(t.shape) == 2 and t.name != "token_embd.weight":
            w.add_tensor(f"{t.name}.in_sum2", rng.uniform(0.5, 2, (1, int(t.shape[0]))).astype(np.float32))
            w.add_tensor(f"{t.name}.counts", np.ones((1, 1), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def fake_hessians(model, text, modules):
    """Stands in for Model.hessians: a fixed random H per module (the same on every call), one group per module."""
    for name in modules:
        n = INTER if name.endswith("down_proj") else HIDDEN
        yield {name: torch.from_numpy(random_hessian(n, zlib.crc32(name.encode())))}


def random_hessian(n: int, seed: int) -> np.ndarray:
    """A realistic H: inputs with correlated columns and very different scales."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4 * n, n // 4)) @ rng.normal(size=(n // 4, n)) + 0.1 * rng.normal(size=(4 * n, n))
    x *= rng.uniform(0.2, 3, n)
    return (x.T @ x / len(x)).astype(np.float32)
