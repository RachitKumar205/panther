"""Build a mixed GGUF by copying tensors from per-type quantized copies, and check it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re

import gguf

from panther.model import Inventory


def tensor_assignment(inv: Inventory, assignment: dict[str, str]) -> dict[str, str]:
    """Expand gene_id -> format into tensor name -> format."""
    out: dict[str, str] = {}
    for g in inv.genes:
        fmt = assignment[g.gene_id]
        for t in g.tensors:
            out[t.name] = fmt
    return out


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


@dataclass
class VerifyReport:
    path: Path
    mismatches: list[tuple[str, str, str]] = field(default_factory=list)  # name, expected, actual
    missing: list[str] = field(default_factory=list)
    file_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.missing


def verify(inv: Inventory, assignment: dict[str, str], path: Path) -> VerifyReport:
    reader = gguf.GGUFReader(path)
    actual = {t.name: t.tensor_type.name for t in reader.tensors}
    report = VerifyReport(path=Path(path), file_bytes=Path(path).stat().st_size)
    for name, fmt in tensor_assignment(inv, assignment).items():
        if name not in actual:
            report.missing.append(name)
        elif actual[name] != fmt:
            report.mismatches.append((name, fmt, actual[name]))
    return report
