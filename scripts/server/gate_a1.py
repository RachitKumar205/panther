"""Phase A gate, step A1 (docs/PIPELINE.md): a hand-built mixed 1.5B GGUF that uses every format.

  python scripts/server/gate_a1.py assemble     # assignment, assemble, read-back, size, byte identity
  python scripts/server/gate_a1.py report       # parse load logs, generations, perplexities; pass/fail

Outputs in $PANTHER_DATA/gate/a1/. scripts/server/gate_a1.sh runs the llama.cpp steps in between.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import gguf

from panther.costs.formats import candidate_formats, gene_formats
from panther.costs.ram import average_bits_per_weight, estimate_ram, fixed_bytes, gene_bytes
from panther.gguftools.assemble import TensorStore, assemble, assignment_digest
from panther.gguftools.verify import verify
from panther.models.inventory import REPO_ROOT, build_inventory, load_geometry

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
STORE = DATA / "store" / MODEL
OUT = DATA / "gate" / "a1"
GGUF_OUT = OUT / "a1.gguf"


def a1_assignment(inv) -> dict[str, str]:
    f = candidate_formats()
    a = {}
    for g in inv.genes:
        if g.kind == "attn":
            a[g.gene_id] = f[g.layer % 12]
        elif g.kind == "mlp":
            a[g.gene_id] = f[(5 * g.layer + 3) % 12]
        else:
            a[g.gene_id] = "Q8_0"
    allowed = gene_formats(inv)
    bad = [g for g, fmt in a.items() if fmt not in allowed[g]]
    if bad:
        raise ValueError(f"assignment uses disallowed formats for {bad}")
    return a


def expected_type_counts(inv, a: dict[str, str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for g in inv.genes:
        c[a[g.gene_id].lower()] += len(g.tensors)
    c["f32"] = len(inv.fixed_tensors)
    return dict(sorted(c.items()))


def tensor_sha(t) -> str:
    return hashlib.sha256(memoryview(t.data).cast("B")).hexdigest()


def cmd_assemble() -> int:
    inv = build_inventory(load_geometry(MODEL))
    a = a1_assignment(inv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assignment.json").write_text(json.dumps(a, indent=2))

    store = TensorStore({f: STORE / f"{MODEL}-{f}.gguf" for f in candidate_formats()})
    tmp = GGUF_OUT.with_suffix(".gguf.part")
    assemble(inv, a, store, tmp, base_format="Q8_0")
    tmp.rename(GGUF_OUT)

    r = verify(inv, a, GGUF_OUT)
    reader = gguf.GGUFReader(GGUF_OUT)
    data_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    predicted_weights = sum(gene_bytes(g, a[g.gene_id]) for g in inv.genes)
    predicted_data = predicted_weights + fixed_bytes(inv)

    # Byte identity: first tensor of each format vs the same tensor in that format's source file.
    by_name = {t.name: t for t in reader.tensors}
    identity = {}
    for g in inv.genes:
        fmt = a[g.gene_id]
        if fmt in identity:
            continue
        name = g.tensors[0].name
        src = {t.name: t for t in gguf.GGUFReader(store.sources[fmt]).tensors}[name]
        identity[fmt] = {"tensor": name, "identical": tensor_sha(by_name[name]) == tensor_sha(src)}

    ram = estimate_ram(inv, a, 4096, 0.40)
    part = {
        "model": MODEL,
        "assignment_sha256": assignment_digest(a),
        "formats_used": sorted(set(a.values())),
        "predicted": {"weight_bytes": predicted_weights, "fixed_bytes": fixed_bytes(inv),
                      "avg_bits_per_weight": round(average_bits_per_weight(inv, a), 4),
                      "ram_gib_at_4096": round(ram.total_gib, 4)},
        "expected_type_counts": expected_type_counts(inv, a),
        "readback": {"mismatches": r.mismatches, "missing": r.missing},
        "size": {"file_bytes": r.file_bytes, "tensor_data_bytes": data_bytes,
                 "predicted_tensor_data_bytes": predicted_data,
                 "data_matches_prediction": data_bytes == predicted_data,
                 "header_and_padding_bytes": r.file_bytes - data_bytes},
        "byte_identity": identity,
    }
    (OUT / "assemble.json").write_text(json.dumps(part, indent=2))
    ok = r.ok and data_bytes == predicted_data and all(v["identical"] for v in identity.values())
    print(json.dumps({k: part[k] for k in ("readback", "size")}, indent=1))
    print("byte identity:", all(v["identical"] for v in identity.values()))
    print("assemble:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


TYPE_LINE = re.compile(r"- type\s+(\S+):\s+(\d+) tensors")
PPL_LINE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


def parse_types(log: Path) -> dict[str, int]:
    return {m.group(1).lower(): int(m.group(2)) for m in TYPE_LINE.finditer(log.read_text(errors="replace"))}


def parse_ppl(log: Path) -> tuple[float, float] | None:
    m = PPL_LINE.search(log.read_text(errors="replace"))
    return (float(m.group(1)), float(m.group(2))) if m else None


# Prompt 3 ("What is 17 + 25? Answer with the number only.") is checked by its answer; the others by form.
EXPECTED_ANSWER = {3: "42"}


def coherent(text: str, prompt_no: int) -> bool:
    if prompt_no in EXPECTED_ANSWER:
        return EXPECTED_ANSWER[prompt_no] in text.split() or text.strip() == EXPECTED_ANSWER[prompt_no]
    words = text.split()
    if len(words) < 3:
        return False
    printable = sum(ch.isprintable() or ch.isspace() for ch in text) / max(len(text), 1)
    distinct = len(set(words)) / len(words)
    return printable > 0.98 and distinct > 0.3


def cmd_report() -> int:
    part = json.loads((OUT / "assemble.json").read_text())
    expected = part["expected_type_counts"]
    checks: dict[str, bool] = {}

    checks["1_readback"] = not part["readback"]["mismatches"] and not part["readback"]["missing"]

    loads = {}
    for build in ("cpu", "cuda"):
        got = parse_types(OUT / f"ppl-{build}-mixed.log")
        loads[build] = {"counts": got, "matches": got == expected}
    checks["2_llama_type_counts"] = all(v["matches"] for v in loads.values())

    checks["3_size_and_bytes"] = (part["size"]["data_matches_prediction"]
                                  and all(v["identical"] for v in part["byte_identity"].values()))

    gens = {}
    for build in ("cpu", "cuda"):
        for p in sorted(OUT.glob(f"gen-{build}-*.txt")):
            text = p.read_text(errors="replace").strip()
            gens[p.stem] = {"text": text, "coherent": coherent(text, int(p.stem.rsplit("-", 1)[1]))}
    checks["4_coherent_text"] = len(gens) == 6 and all(v["coherent"] for v in gens.values())

    ppl = {name: parse_ppl(OUT / f"ppl-cuda-{name}.log") for name in ("bf16", "q8_0", "mixed", "q2_k")}
    ppl["mixed_cpu_build"] = parse_ppl(OUT / "ppl-cpu-mixed.log")
    v = {k: x[0] for k, x in ppl.items() if x}
    checks["5_ppl_order"] = (len(v) == 5 and v["bf16"] <= v["q8_0"] * 1.001
                             and v["q8_0"] < v["mixed"] < v["q2_k"])

    report = {**part, "llama_load": loads, "generations": gens,
              "perplexity_sensitivity_16x2048": {k: list(x) if x else None for k, x in ppl.items()},
              "checks": checks, "pass": all(checks.values())}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    for k, ok in checks.items():
        print(f"{k:24s} {'PASS' if ok else 'FAIL'}")
    print("perplexity:", {k: round(x, 4) for k, x in v.items()})
    print("A1:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    cmds = {"assemble": cmd_assemble, "report": cmd_report}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]]())
