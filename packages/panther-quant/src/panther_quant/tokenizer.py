"""The tokenizer, reused from transformers (ADR 0027)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_tokenizer(gguf_dir: Path, gguf_name: str) -> Any:
    """Load the model's tokenizer from its GGUF (transformers extracts the vocab and BPE merges)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(gguf_dir), gguf_file=gguf_name)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
