"""Which models fit each RAM budget, and how many bits per weight each gets (configs/search.yaml).

Per-matrix groups, exact llama.cpp byte sizes from gguf-py. Also counts the (group, format) options that can
appear in at least one within-budget configuration: an option is dropped if, even with every other group at
its smallest format, it would exceed the budget. A model is dropped if all-smallest does not fit.

    uv run python scripts/budget_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GIB = 2**30


def nbytes(rows: int, cols: int, fmt: str) -> int | None:
    block, block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType[fmt]]
    return None if cols % block else rows * (cols // block) * block_bytes


def model_geometry(name: str) -> dict:
    c = json.loads((CONFIGS / "model_configs" / f"{name}.json").read_text())
    return {"L": c["num_hidden_layers"], "h": c["hidden_size"], "m": c["intermediate_size"],
            "kv": c["num_key_value_heads"] * c["hidden_size"] // c["num_attention_heads"],
            "V": c["vocab_size"], "tied": c["tie_word_embeddings"]}


def groups(g: dict, cfg: dict) -> dict[str, dict[str, int]]:
    """group id -> {format: bytes}, for the formats that apply to that matrix."""
    L, h, m, kv, V = g["L"], g["h"], g["m"], g["kv"], g["V"]
    all_f = cfg["formats"]
    no_im = [f for f in all_f if f not in cfg["requires_imatrix"]]
    shapes = []
    for i in range(L):
        shapes += [(f"q.{i}", h, h, all_f), (f"k.{i}", kv, h, all_f), (f"v.{i}", kv, h, all_f),
                   (f"o.{i}", h, h, all_f), (f"gate.{i}", m, h, all_f), (f"up.{i}", m, h, all_f),
                   (f"down.{i}", h, m, all_f)]
    if g["tied"]:
        shapes.append(("embd_head", V, h, no_im))
    else:
        shapes += [("embd", V, h, no_im), ("head", V, h, all_f)]
    return {gid: {f: b for f in fs if (b := nbytes(r, c, f)) is not None} for gid, r, c, fs in shapes}


def main() -> None:
    cfg = yaml.safe_load((CONFIGS / "search.yaml").read_text())
    models = [m["name"] for m in yaml.safe_load((CONFIGS / "models.yaml").read_text())["models"]]
    kv_block, kv_block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType[cfg["kv_cache_type"]]]
    print(f"context {cfg['context_tokens']}, KV cache {cfg['kv_cache_type']}, "
          f"overhead {cfg['overhead_gib']} GiB\n")
    print("| Budget | Model | Smallest RAM | Bits/weight | Options kept / total | Role |")
    print("|---|---|---|---|---|---|")
    for budget_gib in cfg["budgets_gib"]:
        budget = budget_gib * GIB
        for name in models:
            g = model_geometry(name)
            opts = groups(g, cfg)
            params = sum(_n_weights(gid, g) for gid in opts)
            kv_bytes = 2 * g["L"] * g["kv"] * cfg["context_tokens"] * kv_block_bytes // kv_block
            norms = 4 * (g["L"] * (3 * g["h"] + 2 * g["kv"]) + g["h"])  # F32 norms + q/k/v biases
            other = norms + kv_bytes + int(cfg["overhead_gib"] * GIB)
            mins = {gid: min(o.values()) for gid, o in opts.items()}
            min_total = other + sum(mins.values())
            total = sum(len(o) for o in opts.values())
            short = name.split("-")[1]
            if min_total > budget:
                print(f"| {budget_gib} GiB | {short} | {min_total / GIB:.2f} GiB | — | 0 / {total} | doesn't fit |")
                continue
            slack = budget - min_total
            kept = sum(1 for gid, o in opts.items() for b in o.values() if b - mins[gid] <= slack)
            bpw = 8 * (budget - other) / params
            if bpw >= cfg["search_max_bpw"]:
                role = f"reference: uniform {largest_uniform(opts, budget - other)}"
            else:
                role = "search" if bpw >= cfg["min_bpw"] else "too few bits"
            print(f"| {budget_gib} GiB | {short} | {min_total / GIB:.2f} GiB | {bpw:.2f} | {kept} / {total} | {role} |")


def largest_uniform(opts: dict[str, dict[str, int]], weight_budget: int) -> str:
    """Largest single format that fits; groups that can't take it (embedding) use their largest allowed one."""
    best = None
    for f in sorted({f for o in opts.values() for f in o}, key=lambda f: sum(o.get(f, 0) for o in opts.values())):
        if sum(o[f] if f in o else max(o.values()) for o in opts.values()) <= weight_budget:
            best = f
    return best or "none"


def _n_weights(gid: str, g: dict) -> int:
    kind = gid.split(".")[0]
    h, m, kv, V = g["h"], g["m"], g["kv"], g["V"]
    return {"q": h * h, "k": kv * h, "v": kv * h, "o": h * h, "gate": m * h, "up": m * h, "down": h * m,
            "embd": V * h, "head": V * h, "embd_head": V * h}[kind]


if __name__ == "__main__":
    main()
