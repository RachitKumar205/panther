"""D37 (docs/DECISIONS.md): confirm the unscaled CPU correction on 4 fresh configurations.

  python scripts/server/d37.py validate   # GPU container: assemble + evaluate on both builds
  python scripts/server/d37.py analyze    # apply the rule, write $PANTHER_DATA/gate/d37/report.json

Rule (unchanged from D36's original C2): for every configuration,
|predicted CPU - measured CPU| <= 0.25 x |CUDA - CPU| or <= 0.0005, where predicted CPU = CUDA KL + the
sum of a_g over genes in an A256 format (no per-format scaling). a_g comes from D36's 17 measured genes
with linear interpolation, unchanged.
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import d36  # noqa: E402  (same directory; reuse D36's measured costs and helpers)

from panther.costs.formats import gene_formats  # noqa: E402

OUT = d36.DATA / "gate" / "d37"
SEED, N = 20261112, 4


def assignments(inv) -> list[dict[str, str]]:
    rng = np.random.default_rng(SEED)
    allowed = gene_formats(inv)
    return [{g: str(rng.choice(allowed[g])) for g in sorted(allowed)} for _ in range(N)]


def cmd_validate() -> int:
    inv, store = d36.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    vals = assignments(inv)
    old = json.loads((d36.OUT / "validation_assignments.json").read_text())
    if any(v in old for v in vals):
        raise SystemExit("a fresh configuration repeats a D36 one")
    (OUT / "validation_assignments.json").write_text(json.dumps(vals, indent=2))
    for i, a in enumerate(vals):
        d36.evaluate(inv, store, a, f"d37:val:{i}")
    return 0


def cmd_analyze() -> int:
    inv, _ = d36.setup()
    recs = d36.records()
    base = {b: d36.kl_of(recs, "d36:base", b) for b in ("cuda", "cpu")}
    measured = {g: (d36.kl_of(recs, f"d36:gene:{g}", "cpu") - base["cpu"])
                - (d36.kl_of(recs, f"d36:gene:{g}", "cuda") - base["cuda"]) for g in d36.sampled_genes(inv)}
    a = d36.interpolate(inv, measured)
    vals = json.loads((OUT / "validation_assignments.json").read_text())
    rows = []
    for i, assignment in enumerate(vals):
        label = f"d37:val:{i}"
        cuda, cpu = d36.kl_of(recs, label, "cuda"), d36.kl_of(recs, label, "cpu")
        pred = d36.predict_cpu(cuda, assignment, a, None)
        err, gap = abs(pred - cpu), abs(cuda - cpu)
        rows.append({"config": label, "genes_in_A256": sum(f in d36.A256 for f in assignment.values()),
                     "embd_head": assignment["embd_head"], "mlp.27": assignment["mlp.27"],
                     "cuda": cuda, "cpu": cpu, "predicted_cpu": pred, "abs_error": err,
                     "uncorrected_error": gap, "gap_removed": 1 - err / gap if gap else None,
                     "ok": err <= 0.25 * gap or err <= 0.0005})
    report = {"rule": "unscaled; |pred-cpu| <= 0.25|cuda-cpu| or <= 0.0005 for all 4", "seed": SEED,
              "rows": rows, "pass": all(r["ok"] for r in rows)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    for r in rows:
        print(f"{r['config']} A256 genes {r['genes_in_A256']:2d} (head {r['embd_head']}, mlp.27 {r['mlp.27']}): "
              f"CUDA {r['cuda']:.5f} CPU {r['cpu']:.5f} pred {r['predicted_cpu']:.5f} err {r['abs_error']:.5f} "
              f"uncorrected {r['uncorrected_error']:.5f} removed {r['gap_removed']:.0%} {'ok' if r['ok'] else 'FAIL'}")
    print("D37:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    cmds = {"validate": cmd_validate, "analyze": cmd_analyze}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]]())
