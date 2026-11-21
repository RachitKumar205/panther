"""The RAM model: what a file needs on the Pi, and whether it fits a budget."""

from __future__ import annotations

import gguf
import yaml

from panther.model import Model
from panther.model.model import REPO

GIB = 1024**3
SETTINGS = yaml.safe_load((REPO / "configs/search.yaml").read_text())
TOP = 1.005  # a file fits a budget of B GiB if it needs at most 1.005 × B GiB


class Budget:
    """A RAM budget for one model: RAM = file size + KV cache + overhead.

    The KV cache holds keys and values for every layer at the configured context, in the configured type (8-bit:
    34 bytes per 32 values). The model's shape comes from its original GGUF's own metadata."""

    def __init__(self, model: Model, gib: float) -> None:
        self.model, self.gib = model, gib
        fields = gguf.GGUFReader(model.original).fields
        arch = fields["general.architecture"].contents()

        def value(key: str):
            return fields[f"{arch}.{key}"].contents()

        head = value("attention.key_length") if f"{arch}.attention.key_length" in fields \
            else value("embedding_length") // value("attention.head_count")
        block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[SETTINGS["kv_cache_type"]]]
        values = 2 * value("block_count") * value("attention.head_count_kv") * head * SETTINGS["context_tokens"]
        self.kv_cache = int(values * size / block)
        self.overhead = int(SETTINGS["overhead_gib"] * GIB)

    def ram(self, file_bytes: int) -> int:
        return file_bytes + self.kv_cache + self.overhead

    def fits(self, file_bytes: int) -> bool:
        return self.ram(file_bytes) <= TOP * self.gib * GIB
