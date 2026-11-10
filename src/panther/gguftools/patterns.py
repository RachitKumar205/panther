"""Generate `--tensor-type-file` contents for llama-quantize.

llama-quantize matches each pattern with std::regex_search against the tensor name and the first
match wins, so an unanchored `blk.1.` would also hit `blk.10`–`blk.19`. Every pattern here is anchored
and escaped, one per tensor, which makes the file order-independent.
"""

from __future__ import annotations

import re

from panther.models.inventory import Inventory


def tensor_pattern(tensor_name: str) -> str:
    return "^" + re.escape(tensor_name) + "$"


def tensor_assignment(inv: Inventory, assignment: dict[str, str]) -> dict[str, str]:
    """Expand gene_id -> format into tensor name -> format."""
    out: dict[str, str] = {}
    for g in inv.genes:
        fmt = assignment[g.gene_id]
        for t in g.tensors:
            out[t.name] = fmt
    return out


def tensor_type_file(inv: Inventory, assignment: dict[str, str]) -> str:
    """One `pattern=type` line per tensor, lower-case type names as llama-quantize expects."""
    lines = [
        f"{tensor_pattern(name)}={fmt.lower()}"
        for name, fmt in tensor_assignment(inv, assignment).items()
    ]
    return "\n".join(lines) + "\n"
