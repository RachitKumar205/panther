"""Phase A gate, step A3: check the evaluator's validation runs against the pass conditions in
docs/PIPELINE.md and write $PANTHER_DATA/gate/a3/report.json. Reads only the A3 records."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from panther.costs.formats import bits_per_weight
from panther.models.inventory import REPO_ROOT

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
BUILD_FLOOR = 0.000048  # A2: BF16 CPU build vs CUDA reference, mean KL on the sensitivity text
FAMILIES = {"K": ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"], "IQ": ["IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_XS"]}


def latest(records: list[dict], kind: str, label: str, build: str, where: str | None = None) -> dict:
    hits = [r for r in records if r["kind"] == kind and r["label"] == label and r["how"]["build"] == build
            and (where is None or r["how"].get("reference", r["how"].get("text")) == where)]
    if not hits:
        raise SystemExit(f"missing record: {kind} {label} {build} {where}")
    return sorted(hits, key=lambda r: r["eval_id"])[-1]


def main() -> int:
    res = DATA / "results" / MODEL
    records = [json.loads(p.read_text()) for p in sorted(res.glob("*.json"))]
    records = [r for r in records if r["label"].startswith("a3:")]
    checks: dict[str, bool] = {}
    report: dict = {"model": MODEL, "records": [r["eval_id"] for r in records]}

    checks["1_all_records_ok"] = bool(records) and all(r["ok"] for r in records)
    failed = [r["eval_id"] for r in records if not r["ok"]]

    ppl = {(lab, b): latest(records, "ppl", lab, b, "selection") for lab in ("a3:a1", "a3:bf16") for b in ("cuda", "cpu")}
    checks["2_ppl_recovery"] = all(r["metrics"]["recovery_rel_error"] <= 1e-4 for r in ppl.values())
    report["perplexity_selection"] = {f"{lab}/{b}": {k: r["metrics"][k] for k in ("ppl", "ppl_se", "recovery_rel_error")}
                                      for (lab, b), r in ppl.items()}

    kl = {f: latest(records, "kl", f"a3:uniform:{f}", "cuda", "sensitivity")["metrics"]
          for f in FAMILIES["K"] + FAMILIES["IQ"] + ["Q8_0", "Q4_0", "IQ4_NL"]}
    a1 = {(b, ref): latest(records, "kl", "a3:a1", b, ref)["metrics"] for b in ("cuda", "cpu")
          for ref in ("sensitivity", "selection")}
    a1_sens = a1[("cuda", "sensitivity")]["kl_mean"]
    build_diff = {ref: abs(a1[("cpu", ref)]["kl_mean"] - a1[("cuda", ref)]["kl_mean"]) for ref in ("sensitivity", "selection")}
    checks["3_a1_between_and_builds_agree"] = (kl["Q8_0"]["kl_mean"] < a1_sens < kl["Q2_K"]["kl_mean"]
                                               and all(d <= 3 * BUILD_FLOOR for d in build_diff.values()))
    report["a1_kl"] = {f"{b}/{ref}": {k: m[k] for k in ("kl_mean", "kl_mean_se", "kl_p99", "top1_pct", "ppl_ratio")}
                       for (b, ref), m in a1.items()}
    report["a1_build_diff"] = build_diff

    rep = latest(records, "kl", "a3:a1:repeat", "cuda", "sensitivity")["metrics"]
    repeat_diff = abs(rep["kl_mean"] - a1_sens)
    checks["4_repeat"] = repeat_diff <= BUILD_FLOOR
    report["repeat"] = {"first": a1_sens, "repeat": rep["kl_mean"], "abs_diff": repeat_diff,
                        "identical": rep == a1[("cuda", "sensitivity")]}

    store = json.loads((DATA / "store" / MODEL / "manifest.json").read_text())["formats"]
    sweep = {f: {"bytes": store[f]["bytes"], "bpw": bits_per_weight(f), "kl_mean": m["kl_mean"],
                 "kl_p99": m["kl_p99"], "top1_pct": m["top1_pct"],
                 "embedding_placeholder": store[f]["embedding_placeholder"]} for f, m in kl.items()}
    mono = {fam: all(sweep[a]["kl_mean"] > sweep[b]["kl_mean"] for a, b in zip(fs, fs[1:]))
            for fam, fs in FAMILIES.items()}
    checks["5_monotone_within_family"] = all(mono.values())
    dominated = sorted(f for f in sweep if any(
        g != f and sweep[g]["bytes"] <= sweep[f]["bytes"] and sweep[g]["kl_mean"] <= sweep[f]["kl_mean"]
        and (sweep[g]["bytes"] < sweep[f]["bytes"] or sweep[g]["kl_mean"] < sweep[f]["kl_mean"]) for g in sweep))
    report["uniform_sweep_sensitivity"] = dict(sorted(sweep.items(), key=lambda kv: kv[1]["bytes"]))
    report["monotone"] = mono
    report["dominated_formats"] = dominated

    report["checks"] = checks
    report["failed_records"] = failed
    report["pass"] = all(checks.values())
    out = DATA / "gate" / "a3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"{'format':8s} {'GiB':>6s} {'bpw':>5s} {'KL mean':>9s} {'KL p99':>8s} {'top-1':>7s}")
    for f, v in report["uniform_sweep_sensitivity"].items():
        print(f"{f:8s} {v['bytes'] / 2**30:6.3f} {v['bpw']:5.2f} {v['kl_mean']:9.5f} {v['kl_p99']:8.4f} {v['top1_pct']:6.2f}%"
              + (f"  (embedding {v['embedding_placeholder']})" if v["embedding_placeholder"] else ""))
    print("dominated:", dominated, "| monotone:", mono)
    print("A1 KL:", {k: round(v["kl_mean"], 6) for k, v in report["a1_kl"].items()}, "| build diff:", build_diff)
    print("repeat:", report["repeat"])
    print("PPL selection:", {k: round(v["ppl"], 4) for k, v in report["perplexity_selection"].items()})
    for k, ok in checks.items():
        print(f"{k:32s} {'PASS' if ok else 'FAIL'}")
    print("A3:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
