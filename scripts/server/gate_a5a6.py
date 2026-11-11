"""Phase A gate, steps A5 (GSM8K) and A6 (generation sanity) on 1.5B (docs/PIPELINE.md, D26, D41).

  python scripts/server/gate_a5a6.py gpu      # GPU container: CUDA runs for both steps
  python scripts/server/gate_a5a6.py cpu      # CPU timing: A1, first 20 GSM8K questions (D40)
  python scripts/server/gate_a5a6.py report   # pass conditions -> $PANTHER_DATA/gate/a5a6/report.json

Selection items only (D34). llama-server listens on 127.0.0.1 inside the container.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from panther.models.inventory import REPO_ROOT
from panther.quality import gencheck, gsm8k
from panther.quality.server import LlamaServer

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
LLAMA = Path(os.environ.get("PANTHER_LLAMA", "/opt/llama.cpp"))
STORE = DATA / "store" / MODEL
RES = DATA / "results" / MODEL
OUT = DATA / "gate" / "a5a6"
THREADS = int(os.environ.get("PANTHER_THREADS", "16"))
FILES = {"bf16": STORE / f"{MODEL}-BF16.gguf", "q8_0": STORE / f"{MODEL}-Q8_0.gguf",
         "q4_k": STORE / f"{MODEL}-Q4_K.gguf", "a1": DATA / "gate/a1/a1.gguf"}
# (label, file key, run GSM8K?, run generation check?)
GPU_RUNS = [("bf16", "bf16", True, True), ("bf16:repeat", "bf16", True, True), ("q8_0", "q8_0", True, True),
            ("q4_k", "q4_k", False, True), ("a1", "a1", True, True)]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x]


def gen_prompts() -> list[dict]:
    path = DATA / "splits" / "gen_prompts.json"
    if not path.exists():
        sel = {t: jsonl(DATA / "splits/tasks/selection" / f"{t}.jsonl")
               for t in ("hellaswag", "arc_challenge", "mmlu", "winogrande", "gsm8k")}
        path.write_text(json.dumps(gencheck.make_prompts(sel), indent=1))
    return json.loads(path.read_text())


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def write_record(kind: str, label: str, gguf: Path, build: str, server_cmd: list[str], seconds: float,
                 outputs: list[dict], extra: dict) -> None:
    started = dt.datetime.now(dt.timezone.utc)
    sha = file_sha(gguf)
    eval_id = f"{started:%Y%m%dT%H%M%SZ}-{kind}-{build}-{sha[:10]}"
    (RES / "outputs").mkdir(parents=True, exist_ok=True)
    (RES / "outputs" / f"{eval_id}.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outputs) + "\n")
    rec = {"eval_id": eval_id, "kind": kind, "model": MODEL, "label": label,
           "evaluated": {"gguf": gguf.name, "gguf_sha256": sha},
           "how": {"build": build, "server_command": server_cmd, "image_id": os.environ.get("PANTHER_IMAGE_ID", "unknown")},
           "wall_seconds": round(seconds, 1), "outputs_file": f"outputs/{eval_id}.jsonl", "ok": True, **extra}
    (RES / f"{eval_id}.json").write_text(json.dumps(rec, indent=2))
    print(f"{eval_id} {label}: {extra.get('summary', '')} [{seconds:.0f}s]", flush=True)


def run_gsm8k(srv: LlamaServer, label: str, gguf: Path, build: str, items: list[dict], fewshot: list[dict]) -> None:
    t0 = time.monotonic()
    comps = srv.map(lambda it: srv.complete(gsm8k.prompt(it["question"], fewshot), gsm8k.MAX_TOKENS, gsm8k.STOP), items)
    secs = time.monotonic() - t0
    outs, strict, flex = [], [], []
    for it, c in zip(items, comps):
        s_ok = gsm8k.exact_match(gsm8k.strict_match(c), it["answer"])
        f_ok = gsm8k.exact_match(gsm8k.flexible_extract(c), it["answer"])
        strict.append(int(s_ok))
        flex.append(int(f_ok))
        outs.append({"id": it["id"], "completion": c, "strict": int(s_ok), "flexible": int(f_ok)})
    m = {"strict_pct": 100 * float(np.mean(strict)), "flexible_pct": 100 * float(np.mean(flex)), "n": len(items),
         "empty": sum(not c.strip() for c in comps)}
    write_record("gsm8k", f"a5:{label}", gguf, build, srv.cmd, secs, outs,
                 {"metrics": m, "per_item": {"ids": [it["id"] for it in items], "correct": strict, "correct_flexible": flex},
                  "summary": f"strict {m['strict_pct']:.1f}% flexible {m['flexible_pct']:.1f}% (n={m['n']}, empty {m['empty']})"})


def run_gen(srv: LlamaServer, label: str, gguf: Path, build: str, prompts: list[dict]) -> None:
    t0 = time.monotonic()
    texts = srv.map(lambda p: srv.chat(p["prompt"], gencheck.MAX_TOKENS), prompts)
    secs = time.monotonic() - t0
    outs = [{"id": p["id"], "task": p["task"], "output": t} for p, t in zip(prompts, texts)]
    write_record("gen", f"a6:{label}", gguf, build, srv.cmd, secs, outs,
                 {"metrics": {"n": len(texts), "empty": sum(not t.strip() for t in texts)},
                  "summary": f"{len(texts)} outputs, empty {sum(not t.strip() for t in texts)}"})


def cmd_gpu() -> int:
    items = jsonl(DATA / "splits/tasks/selection/gsm8k.jsonl")
    fewshot = jsonl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    prompts = gen_prompts()
    OUT.mkdir(parents=True, exist_ok=True)
    for label, key, do_gsm, do_gen in GPU_RUNS:
        with LlamaServer(LLAMA, FILES[key], "cuda", parallel=8, ctx_per_slot=2048, threads=THREADS,
                         log=OUT / f"server-{label.replace(':', '-')}.log") as srv:
            if do_gsm:
                run_gsm8k(srv, label, FILES[key], "cuda", items, fewshot)
            if do_gen:
                run_gen(srv, label, FILES[key], "cuda", prompts)
    return 0


def cmd_cpu() -> int:
    items = jsonl(DATA / "splits/tasks/selection/gsm8k.jsonl")[:20]
    fewshot = jsonl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    with LlamaServer(LLAMA, FILES["a1"], "cpu", parallel=4, ctx_per_slot=2048, threads=THREADS,
                     log=OUT / "server-a1-cpu.log") as srv:
        run_gsm8k(srv, "a1:cpu-timing", FILES["a1"], "cpu", items, fewshot)
    return 0


def latest(kind: str, label: str) -> dict:
    hits = sorted((json.loads(p.read_text()) for p in RES.glob("*.json")), key=lambda r: r["eval_id"])
    hits = [r for r in hits if r["kind"] == kind and r["label"] == label]
    if not hits:
        raise SystemExit(f"missing {kind} {label}")
    return hits[-1]


def outputs(rec: dict) -> list[dict]:
    return jsonl(RES / rec["outputs_file"])


def cmd_report() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DATA / "hf" / MODEL)
    rep: dict = {"A5": {}, "A6": {}}

    # --- A5 ---
    g = {k: latest("gsm8k", f"a5:{k}") for k in ("bf16", "bf16:repeat", "q8_0", "a1")}
    acc = {k: r["metrics"]["strict_pct"] for k, r in g.items()}
    c1, c2 = outputs(g["bf16"]), outputs(g["bf16:repeat"])
    differing = sum(a["completion"] != b["completion"] for a, b in zip(c1, c2))
    cpu = latest("gsm8k", "a5:a1:cpu-timing")
    a5 = {"1_all_answered": all(r["metrics"]["n"] == 500 and r["metrics"]["empty"] == 0 for r in g.values()),
          "2_bf16_floor": acc["bf16"] >= 40,
          "3_q8_within_5": abs(acc["q8_0"] - acc["bf16"]) <= 5,
          "4_repeat_within_1": abs(acc["bf16:repeat"] - acc["bf16"]) <= 1,
          "5_cpu_timing": cpu["metrics"]["n"] == 20}
    rep["A5"] = {"strict_pct": acc, "flexible_pct": {k: r["metrics"]["flexible_pct"] for k, r in g.items()},
                 "repeat_differing_completions": differing,
                 "cpu_seconds_per_question_a1": cpu["wall_seconds"] / 20,
                 "cpu_hours_full_test_1p5b_estimate": cpu["wall_seconds"] / 20 * 1319 / 3600,
                 "cuda_seconds_per_question_a1": g["a1"]["wall_seconds"] / 500,
                 "checks": a5, "pass": all(a5.values())}

    # --- A6 ---
    gen = {k: outputs(latest("gen", f"a6:{k}")) for k in ("bf16", "bf16:repeat", "q8_0", "q4_k", "a1")}
    ref = [tok(o["output"], add_special_tokens=False)["input_ids"] for o in gen["bf16"]]
    stats = {}
    for k, outs in gen.items():
        ids = [tok(o["output"], add_special_tokens=False)["input_ids"] for o in outs]
        pref = [gencheck.matched_prefix(a, b) for a, b in zip(ids, ref)]
        stats[k] = {"mean_matched_prefix_tokens": float(np.mean(pref)),
                    "identical_to_bf16": sum(o["output"] == r["output"] for o, r in zip(outs, gen["bf16"])),
                    "empty": sum(not o["output"].strip() for o in outs),
                    "degenerate": sum(gencheck.degenerate(o["output"]) for o in outs)}
    a6 = {"1_non_empty": all(s["empty"] == 0 for s in stats.values()),
          "2_bf16_repeat_identical_48": stats["bf16:repeat"]["identical_to_bf16"] >= 48,
          "3_no_degenerate_bf16_q8": stats["bf16"]["degenerate"] == 0 and stats["q8_0"]["degenerate"] == 0,
          "4_q8_prefix_ge_a1": stats["q8_0"]["mean_matched_prefix_tokens"] >= stats["a1"]["mean_matched_prefix_tokens"]}
    rep["A6"] = {"stats": stats, "checks": a6, "pass": all(a6.values()),
                 "spot_check_a1": [{"id": o["id"], "task": o["task"], "output": o["output"]} for o in gen["a1"][::5]]}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print("A5 strict %:", {k: round(v, 1) for k, v in acc.items()},
          "| flexible %:", {k: round(v, 1) for k, v in rep["A5"]["flexible_pct"].items()})
    print(f"A5 repeat: {differing} of 500 completions differ | CPU {rep['A5']['cpu_seconds_per_question_a1']:.1f} s/question "
          f"-> full test ~{rep['A5']['cpu_hours_full_test_1p5b_estimate']:.2f} h (1.5B) | CUDA "
          f"{rep['A5']['cuda_seconds_per_question_a1']:.2f} s/question")
    for k, ok in a5.items():
        print(f"  A5 {k:22s} {'PASS' if ok else 'FAIL'}")
    for k, s in stats.items():
        print(f"A6 {k:12s} prefix {s['mean_matched_prefix_tokens']:6.1f} identical {s['identical_to_bf16']:2d}/50 "
              f"empty {s['empty']} degenerate {s['degenerate']}")
    for k, ok in a6.items():
        print(f"  A6 {k:28s} {'PASS' if ok else 'FAIL'}")
    print("A5:", "PASS" if rep["A5"]["pass"] else "FAIL", "| A6:", "PASS" if rep["A6"]["pass"] else "FAIL")
    print("\n--- A6 spot check: 10 A1 outputs ---")
    for o in rep["A6"]["spot_check_a1"]:
        print(f"[{o['task']}] {o['output'][:300]!r}")
    return 0 if rep["A5"]["pass"] and rep["A6"]["pass"] else 1


if __name__ == "__main__":
    cmds = {"gpu": cmd_gpu, "cpu": cmd_cpu, "report": cmd_report}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]]())
