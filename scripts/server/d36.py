"""D36 (docs/DECISIONS.md): per-gene CPU activation-rounding cost a_g on 1.5B, and its checks.

  python scripts/server/d36.py measure    # GPU container: base + 17 sampled one-gene configs, both builds
  python scripts/server/d36.py validate   # GPU container: 4 seeded random mixes, both builds
  python scripts/server/d36.py analyze    # C1 additivity, C2 prediction, C3 structure -> report.json

Each configuration is assembled, evaluated on the CUDA and CPU builds (KL vs the sensitivity reference),
then deleted. Resumable: configurations whose records already exist are skipped.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from panther.costs.formats import candidate_formats, gene_formats
from panther.costs.ram import uniform
from panther.gguftools.assemble import TensorStore, assemble, assignment_digest
from panther.gguftools.verify import verify
from panther.models.inventory import REPO_ROOT, build_inventory, load_geometry
from panther.quality import run as evaluator

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
STORE = DATA / "store" / MODEL
OUT = DATA / "gate" / "d36"
TMP = OUT / "tmp"
A32 = {"Q8_0", "Q4_0", "IQ4_NL"}
A256 = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_XS"}
BASE_FMT, PROBE_FMT = "Q8_0", "Q6_K"
UNIFORM_Q6K_GAP = 0.012955 - 0.005810  # D35: CPU - CUDA, uniform Q6_K
N_VAL, VAL_SEED = 4, 20261111
SAMPLE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]  # D36 sampled variant (1.5B has 28 layers)


def sampled_genes(inv) -> list[str]:
    """Genes measured directly: the embedding gene(s) and attention + MLP of the sampled layers."""
    out = [g.gene_id for g in inv.genes if g.kind in ("embd", "embd_head", "head")]
    for L in SAMPLE_LAYERS:
        out += [f"attn.{L}", f"mlp.{L}"]
    return out


def interpolate(inv, measured: dict[str, float]) -> dict[str, float]:
    """a_g for every gene: measured where available, else linear in layer index within its kind."""
    out = dict(measured)
    n = inv.geometry.n_layers
    for kind in ("attn", "mlp"):
        xs = [L for L in SAMPLE_LAYERS if f"{kind}.{L}" in measured]
        ys = [measured[f"{kind}.{L}"] for L in xs]
        for L, v in zip(range(n), np.interp(np.arange(n), xs, ys)):
            out.setdefault(f"{kind}.{L}", float(v))
    return out


def records() -> list[dict]:
    return [json.loads(p.read_text()) for p in (DATA / "results" / MODEL).glob("*.json")]


def have(recs: list[dict], label: str, build: str) -> bool:
    return any(r["label"] == label and r["how"]["build"] == build and r["ok"] for r in recs)


def kl_of(recs: list[dict], label: str, build: str) -> float:
    hits = sorted((r for r in recs if r["label"] == label and r["how"]["build"] == build and r["ok"]
                   and r["kind"] == "kl" and r["how"].get("reference") == "sensitivity"),
                  key=lambda r: r["eval_id"])
    if not hits:
        raise SystemExit(f"missing record {label} {build}")
    return hits[-1]["metrics"]["kl_mean"]


def evaluate(inv, store, assignment: dict[str, str], label: str) -> None:
    recs = records()
    todo = [b for b in ("cuda", "cpu") if not have(recs, label, b)]
    if not todo:
        return
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / f"{assignment_digest(assignment)[:16]}.gguf"
    if not path.exists():
        assemble(inv, assignment, store, path.with_suffix(".part"), base_format=BASE_FMT)
        path.with_suffix(".part").rename(path)
    if not verify(inv, assignment, path).ok:
        raise SystemExit(f"{label}: assembled file failed read-back")
    for build in todo:
        rc = evaluator.main(["kl", "--model", MODEL, "--gguf", str(path), "--build", build,
                             "--reference", "sensitivity", "--label", label])
        if rc != 0:
            raise SystemExit(f"{label} {build}: evaluation failed")
    path.unlink()


def setup():
    inv = build_inventory(load_geometry(MODEL))
    store = TensorStore({f: STORE / f"{MODEL}-{f}.gguf" for f in candidate_formats()})
    return inv, store


def cmd_measure() -> int:
    inv, store = setup()
    base = uniform(inv, BASE_FMT)
    evaluate(inv, store, base, "d36:base")
    for g in sampled_genes(inv):
        evaluate(inv, store, {**base, g: PROBE_FMT}, f"d36:gene:{g}")
    return 0


def validation_assignments(inv) -> list[dict[str, str]]:
    rng = np.random.default_rng(VAL_SEED)
    allowed = gene_formats(inv)
    return [{g: str(rng.choice(allowed[g])) for g in sorted(allowed)} for _ in range(N_VAL)]


def cmd_validate() -> int:
    inv, store = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    vals = validation_assignments(inv)
    (OUT / "validation_assignments.json").write_text(json.dumps(vals, indent=2))
    for i, a in enumerate(vals):
        evaluate(inv, store, a, f"d36:val:{i}")
    return 0


def cmd_analyze() -> int:
    inv, _ = setup()
    recs = records()
    base = {b: kl_of(recs, "d36:base", b) for b in ("cuda", "cpu")}
    measured = {}
    for g in sampled_genes(inv):
        d_cpu = kl_of(recs, f"d36:gene:{g}", "cpu") - base["cpu"]
        d_cuda = kl_of(recs, f"d36:gene:{g}", "cuda") - base["cuda"]
        measured[g] = d_cpu - d_cuda
    a = interpolate(inv, measured)
    assert set(a) == {g.gene_id for g in inv.genes}
    total = sum(a.values())
    c1 = abs(total - UNIFORM_Q6K_GAP) <= 0.25 * UNIFORM_Q6K_GAP

    # C2: prediction on mixed configurations. A1 was measured on both builds in A3.
    vals = json.loads((OUT / "validation_assignments.json").read_text())
    a1 = json.loads((DATA / "gate" / "a1" / "assignment.json").read_text())
    configs = [("A1", a1, "a3:a1")] + [(f"val{i}", v, f"d36:val:{i}") for i, v in enumerate(vals)]
    c2_rows = []
    for name, assignment, label in configs:
        cuda, cpu = kl_of(recs, label, "cuda"), kl_of(recs, label, "cpu")
        pred = cuda + sum(a[g] for g, f in assignment.items() if f in A256)
        err, gap = abs(pred - cpu), abs(cuda - cpu)
        ok = err <= 0.25 * gap or err <= 0.0005
        c2_rows.append({"config": name, "cuda": cuda, "cpu": cpu, "predicted_cpu": pred,
                        "abs_error": err, "uncorrected_error": gap,
                        "gap_removed": 1 - err / gap if gap else None, "ok": ok,
                        "genes_in_A256": sum(f in A256 for f in assignment.values())})
    c2 = all(r["ok"] for r in c2_rows)

    # C3 (sampled variant): measured values by layer and kind, and each kind's share of the total.
    c3 = {}
    for kind in ("attn", "mlp"):
        c3[kind] = {"measured_by_layer": {L: round(measured[f"{kind}.{L}"], 7) for L in SAMPLE_LAYERS},
                    "sum_interpolated": float(sum(v for g, v in a.items() if g.startswith(kind + ".")))}
    embd = [g.gene_id for g in inv.genes if g.kind in ("embd", "embd_head", "head")]

    report = {"model": MODEL, "base": f"uniform {BASE_FMT}", "probe": PROBE_FMT, "base_kl": base,
              "variant": "sampled", "sample_layers": SAMPLE_LAYERS,
              "a_g_measured": {g: round(v, 7) for g, v in measured.items()},
              "a_g": {g: round(v, 7) for g, v in a.items()},
              "C1": {"sum_a_g": total, "uniform_q6k_gap": UNIFORM_Q6K_GAP,
                     "rel_diff": (total - UNIFORM_Q6K_GAP) / UNIFORM_Q6K_GAP, "pass": c1},
              "C2": {"rows": c2_rows, "pass": c2},
              "C3": {"by_kind": c3, "embedding_genes": {g: a[g] for g in embd}},
              "pass": c1 and c2}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    print(f"base KL: CUDA {base['cuda']:.6f} CPU {base['cpu']:.6f}")
    print(f"C1: sum a_g = {total:.6f} vs uniform-Q6_K gap {UNIFORM_Q6K_GAP:.6f} "
          f"({report['C1']['rel_diff']:+.1%}) {'PASS' if c1 else 'FAIL'}")
    for r in c2_rows:
        print(f"C2 {r['config']:5s} A256 genes {r['genes_in_A256']:2d}: CUDA {r['cuda']:.5f} CPU {r['cpu']:.5f} "
              f"pred {r['predicted_cpu']:.5f} err {r['abs_error']:.5f} (uncorrected {r['uncorrected_error']:.5f}) "
              f"{'ok' if r['ok'] else 'FAIL'}")
    print("C2:", "PASS" if c2 else "FAIL")
    for kind, v in c3.items():
        print(f"C3 {kind}: measured", {L: round(x, 5) for L, x in v["measured_by_layer"].items()},
              f"| sum incl. interpolated {v['sum_interpolated']:.6f}")
    print("C3 embedding:", {g: round(a[g], 6) for g in embd})
    print("D36:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    cmds = {"measure": cmd_measure, "validate": cmd_validate, "analyze": cmd_analyze}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]]())
