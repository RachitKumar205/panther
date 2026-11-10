"""Read a GGUF back and check every searched tensor has exactly its assigned type (no fallback)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gguf

from panther.gguftools.patterns import tensor_assignment
from panther.models.inventory import Inventory


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
