"""D35 pass rule (docs/DECISIONS.md, fixed before running): compare CPU-build KL for the 12 uniform 1.5B
formats with the A3 CUDA sweep.

Pass if (a) both builds rank the formats identically, a swapped pair being allowed only where the two
formats are within 2% of each other on CUDA, and (b) (CPU - CUDA) / CUDA is within +/-5% for every format.
Writes $PANTHER_DATA/gate/d35/report.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from panther.models.inventory import REPO_ROOT

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
FORMATS = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL"]
TIE = 0.02
REL_LIMIT = 0.05


def latest(records: list[dict], label: str, build: str) -> dict:
    hits = [r for r in records if r["kind"] == "kl" and r["label"] == label and r["how"]["build"] == build
            and r["how"].get("reference") == "sensitivity" and r["ok"]]
    if not hits:
        raise SystemExit(f"missing record {label} {build}")
    return sorted(hits, key=lambda r: r["eval_id"])[-1]["metrics"]


def main() -> int:
    records = [json.loads(p.read_text()) for p in (DATA / "results" / MODEL).glob("*.json")]
    cuda = {f: latest(records, f"a3:uniform:{f}", "cuda") for f in FORMATS}
    cpu = {f: latest(records, f"d35:uniform:{f}", "cpu") for f in FORMATS}

    rel = {f: (cpu[f]["kl_mean"] - cuda[f]["kl_mean"]) / cuda[f]["kl_mean"] for f in FORMATS}
    top1_diff = {f: cpu[f]["top1_pct"] - cuda[f]["top1_pct"] for f in FORMATS}
    order_cuda = sorted(FORMATS, key=lambda f: cuda[f]["kl_mean"])
    order_cpu = sorted(FORMATS, key=lambda f: cpu[f]["kl_mean"])

    # Every pair the builds order differently must be a near-tie on CUDA.
    swaps = []
    for i, f in enumerate(FORMATS):
        for g in FORMATS[i + 1:]:
            if (cuda[f]["kl_mean"] < cuda[g]["kl_mean"]) != (cpu[f]["kl_mean"] < cpu[g]["kl_mean"]):
                gap = abs(cuda[f]["kl_mean"] - cuda[g]["kl_mean"]) / min(cuda[f]["kl_mean"], cuda[g]["kl_mean"])
                swaps.append({"pair": [f, g], "cuda_gap": gap, "allowed": gap <= TIE})
    rule_a = all(s["allowed"] for s in swaps)
    rule_b = all(abs(v) <= REL_LIMIT for v in rel.values())

    report = {
        "rule": "ranking identical except near-ties (<= 2% on CUDA); |CPU-CUDA|/CUDA <= 5% for every format",
        "kl": {f: {"cuda": cuda[f]["kl_mean"], "cpu": cpu[f]["kl_mean"], "rel_diff": rel[f],
                   "top1_cuda": cuda[f]["top1_pct"], "top1_cpu": cpu[f]["top1_pct"]} for f in order_cuda},
        "order_cuda": order_cuda, "order_cpu": order_cpu, "swaps": swaps,
        "a_ranking": rule_a, "b_relative_diff": rule_b, "pass": rule_a and rule_b,
    }
    out = DATA / "gate" / "d35"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"{'format':8s} {'CUDA KL':>9s} {'CPU KL':>9s} {'CPU-CUDA':>9s} {'top-1 CUDA':>10s} {'top-1 CPU':>9s}")
    for f in order_cuda:
        print(f"{f:8s} {cuda[f]['kl_mean']:9.5f} {cpu[f]['kl_mean']:9.5f} {rel[f]:+8.2%} "
              f"{cuda[f]['top1_pct']:9.2f}% {cpu[f]['top1_pct']:8.2f}%")
    print("order CUDA:", order_cuda)
    print("order CPU: ", order_cpu)
    print("swaps:", swaps or "none")
    print(f"(a) ranking {'PASS' if rule_a else 'FAIL'} | (b) relative diff {'PASS' if rule_b else 'FAIL'}"
          f" (max |diff| {max(abs(v) for v in rel.values()):.2%})")
    print("D35:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
