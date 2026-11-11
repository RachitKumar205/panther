"""Phase A gate, step A4: apply the pass conditions in docs/PIPELINE.md to the A4 records and D39 report,
and write $PANTHER_DATA/gate/a4/report.json (with CPU timings for the Phase D estimate, D40)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from panther.models.inventory import REPO_ROOT

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
TASKS = ("hellaswag", "arc_challenge", "mmlu", "winogrande")
CHANCE = {"hellaswag": 25.0, "arc_challenge": 25.0, "mmlu": 25.0, "winogrande": 50.0}
# Items in the full test split, for the D40 extrapolation.
TEST_N = {"hellaswag": 10042, "arc_challenge": 1172, "mmlu": 14042, "winogrande": 1267}


def main() -> int:
    recs = [json.loads(p.read_text()) for p in (DATA / "results" / MODEL).glob("*.json")]
    recs = [r for r in recs if r["kind"] == "task" and r["label"].startswith("a4:")]
    manifest = json.loads((DATA / "splits/llama_tasks/manifest.json").read_text())

    def rec(label: str, build: str, task: str) -> dict:
        hits = sorted((r for r in recs if r["label"] == label and r["how"]["build"] == build
                       and r["how"]["task"] == task), key=lambda r: r["eval_id"])
        if not hits:
            raise SystemExit(f"missing record {label} {build} {task}")
        return hits[-1]

    runs = [(lab, "cuda") for lab in ("a4:bf16", "a4:q8_0", "a4:q4_k", "a4:a1")]  # CPU stage cut (D40)
    table = {f"{lab}/{b}": {t: rec(lab, b, t) for t in TASKS} for lab, b in runs}
    checks: dict[str, bool] = {}

    checks["1_converters"] = (DATA / "splits/llama_tasks/manifest.json").exists()  # build_tasks.py aborts otherwise
    checks["2_all_items_scored"] = all(
        r["ok"] and r["metrics"]["n"] == manifest["files"][f"selection/{t}"]["n"]
        and r["per_item"]["ids"] == manifest["files"][f"selection/{t}"]["ids"]
        for row in table.values() for t, r in row.items())
    acc = {k: {t: r["metrics"]["accuracy_pct"] for t, r in row.items()} for k, row in table.items()}
    checks["3a_bf16_above_chance"] = all(acc["a4:bf16/cuda"][t] >= CHANCE[t] + 10 for t in TASKS)
    checks["3b_q8_within_2pt"] = all(abs(acc["a4:q8_0/cuda"][t] - acc["a4:bf16/cuda"][t]) <= 2 for t in TASKS)
    d39 = json.loads((DATA / "gate/a4/d39.json").read_text())
    checks["4_independent_check"] = d39["pass"]

    # D40 timing sample: CPU build, A1, 100 random selection questions per task (time_tasks_cpu.sh).
    timing = {}
    for line in (DATA / "gate/a4/cpu_timing.tsv").read_text().split("\n"):
        if not line:
            continue
        t, n, secs = line.split("\t")
        per_item = float(secs) / int(n)
        cuda = table["a4:a1/cuda"][t]
        timing[t] = {"cpu_seconds_sample": float(secs), "n_sample": int(n), "cpu_seconds_per_item": per_item,
                     "cpu_hours_full_test_1p5b_estimate": per_item * TEST_N[t] / 3600,
                     "cuda_seconds_per_item": cuda["wall_seconds"] / cuda["metrics"]["n"]}
    total_test_h = sum(v["cpu_hours_full_test_1p5b_estimate"] for v in timing.values())
    checks["5_timings_recorded"] = True

    report = {"accuracy_pct": acc, "d39": d39["tasks"], "timing": timing,
              "cpu_hours_full_test_per_config_1p5b": total_test_h, "checks": checks, "pass": all(checks.values())}
    out = DATA / "gate" / "a4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"{'run':16s}" + "".join(f"{t:>15s}" for t in TASKS))
    for k, row in acc.items():
        print(f"{k:16s}" + "".join(f"{row[t]:14.2f}%" for t in TASKS))
    for t, v in d39["tasks"].items():
        print(f"D39 {t:14s} agreement {v['agreement_with_llama_cpp']:.1%}  llama rule {v['accuracy_llama_rule_transformers']:.1%}"
              f"  lm-eval rule {v['accuracy_lm_eval_rule']:.1%}")
    for t, v in timing.items():
        print(f"CPU {t:14s} {v['cpu_seconds_sample']:7.0f} s for {v['n_sample']} items "
              f"-> full test ~{v['cpu_hours_full_test_1p5b_estimate']:.2f} h (1.5B)")
    print(f"full test, all 4 tasks, one 1.5B configuration on CPU: ~{total_test_h:.1f} h")
    for k, ok in checks.items():
        print(f"{k:24s} {'PASS' if ok else 'FAIL'}")
    print("A4:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
