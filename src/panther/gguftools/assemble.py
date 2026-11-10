"""Assemble a mixed-precision GGUF from per-format source files (the "tensor store").

Each source is the same model quantized uniformly to one format with llama-quantize
(`--pure --imatrix ... --token-embedding-type F --output-tensor-type F`). With explicit per-tensor
types, llama.cpp quantizes every tensor independently, so copying a tensor's bytes from the source of
its assigned format gives the same result as quantizing the mixed file directly, in seconds instead of
minutes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import gguf

from panther.gguftools.patterns import tensor_assignment
from panther.models.inventory import Inventory

ASSIGNMENT_KEY = "panther.assignment_sha256"


def assignment_digest(assignment: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()


@dataclass
class TensorStore:
    """Maps format name -> path of the uniform GGUF for that format."""

    sources: dict[str, Path]

    def readers(self, formats: set[str]) -> dict[str, gguf.GGUFReader]:
        missing = formats - self.sources.keys()
        if missing:
            raise FileNotFoundError(f"no source GGUF for formats {sorted(missing)}")
        return {f: gguf.GGUFReader(self.sources[f]) for f in formats}


def _copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def assemble(
    inv: Inventory,
    assignment: dict[str, str],
    store: TensorStore,
    out_path: Path,
    base_format: str | None = None,
) -> Path:
    """Write a GGUF where each gene's tensors come from the source of its assigned format.

    Tensors outside the search space (F32 norms, biases) are copied from `base_format`'s source.
    """
    per_tensor = tensor_assignment(inv, assignment)
    formats = set(per_tensor.values())
    base_format = base_format or sorted(formats)[0]
    readers = store.readers(formats | {base_format})
    base = readers[base_format]

    by_format = {f: {t.name: t for t in r.tensors} for f, r in readers.items()}
    arch = base.fields[gguf.Keys.General.ARCHITECTURE].contents()

    writer = gguf.GGUFWriter(out_path, arch, endianess=base.endianess)
    _copy_metadata(base, writer)
    writer.add_string(ASSIGNMENT_KEY, assignment_digest(assignment))

    chosen = []
    for t in base.tensors:  # keep the base file's tensor order
        fmt = per_tensor.get(t.name, base_format)
        src = by_format[fmt].get(t.name)
        if src is None:
            raise KeyError(f"{t.name} missing from {fmt} source")
        expected = gguf.GGMLQuantizationType[fmt] if t.name in per_tensor else src.tensor_type
        if src.tensor_type != expected:
            raise ValueError(
                f"{t.name}: {fmt} source holds {src.tensor_type.name}, expected {expected.name} "
                "(source was not quantized with --pure and explicit embedding/output types?)"
            )
        chosen.append(src)
        writer.add_tensor_info(src.name, src.data.shape, src.data.dtype, src.data.nbytes, src.tensor_type)

    extra = set(per_tensor) - {t.name for t in base.tensors}
    if extra:
        raise KeyError(f"inventory tensors not in GGUF: {sorted(extra)[:5]}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for src in chosen:
        writer.write_tensor_data(src.data, tensor_endianess=base.endianess)
    writer.close()
    return out_path
