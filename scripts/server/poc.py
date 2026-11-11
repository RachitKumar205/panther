"""Proof of concept: 7B vs 14B at R_max = 7.0 GiB (docs/DECISIONS.md D42). Runs in the GPU container.

  python scripts/server/poc.py refs       MODEL   # P3: BF16 references (sensitivity 12 chunks, selection)
  python scripts/server/poc.py correct    MODEL   # P4: CPU correction for the output head and last MLP (7B)
  python scripts/server/poc.py penalties  MODEL   # P5: penalty table, 3 KL runs at a time
  python scripts/server/poc.py shortlist  MODEL   # P6: solver top 5 + uniform baseline, scored on selection
  python scripts/server/poc.py verdict    all     # P7: test subset, decision rule (needs the user's approval)
  python scripts/server/poc.py bundle     all     # P8: files + manifest for the Pi session

Every step is resumable: finished measurements (records labelled poc:...) are not repeated.
Outputs: $PANTHER_DATA/poc/<model>/ and records in $PANTHER_DATA/results/<model>/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from panther.costs.formats import gene_formats
from panther.costs.ram import estimate_ram, uniform
from panther.gguftools.assemble import TensorStore, assemble, assignment_digest
from panther.gguftools.verify import verify
from panther.models.inventory import REPO_ROOT, build_inventory, load_geometry
from panther.quality import run as evaluator
from panther.search.problem import Problem
from panther.search.solver import solve_top_k

DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
LLAMA = Path(os.environ.get("PANTHER_LLAMA", "/opt/llama.cpp"))
THREADS = int(os.environ.get("PANTHER_THREADS", "16"))
R_MAX, CTX, OVERHEAD = 7.0, 4096, 0.40
SENS_CHUNKS = 12
CONCURRENT = 3
A256 = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_XS"}

# Smoke mode (PANTHER_POC_SMOKE=1): the same steps on 1.5B with tiny settings, separate outputs and labels,
# and selection data in place of the test subset, to catch bugs before the real lanes reach a step.
SMOKE = os.environ.get("PANTHER_POC_SMOKE") == "1"
L = "pocsmoke" if SMOKE else "poc"
POC = DATA / ("poc_smoke" if SMOKE else "poc")
TOPK = 2 if SMOKE else 5
MAX_PEN_GENES = 3 if SMOKE else None
V_SPLIT, V_TEXT = ("selection", "selection") if SMOKE else ("poc_test", "test_web")
BUNDLE_DIR = DATA / ("pi_bundle_smoke" if SMOKE else "pi_bundle")

MENU = {
    "Qwen2.5-7B-Instruct": {"formats": ["Q8_0", "Q6_K", "Q5_K", "Q4_K", "IQ4_XS", "IQ3_S"], "base": "Q6_K"},
    "Qwen2.5-14B-Instruct": {"formats": ["Q6_K", "Q4_K", "IQ4_XS", "IQ3_S", "IQ3_XXS", "Q2_K", "IQ2_XS"],
                             "base": "IQ3_XXS"},
}
if SMOKE:
    MENU = {"Qwen2.5-1.5B-Instruct": {"formats": ["Q8_0", "Q6_K", "Q4_K"], "base": "Q6_K"}}


def setup(model: str):
    inv = build_inventory(load_geometry(model))
    menu = MENU[model]
    store = TensorStore({f: DATA / "store" / model / f"{model}-{f}.gguf" for f in menu["formats"]})
    allowed = {g: [f for f in fs if f in menu["formats"]] for g, fs in gene_formats(inv).items()}
    return inv, menu, store, allowed


def out_dir(model: str) -> Path:
    p = POC / model
    p.mkdir(parents=True, exist_ok=True)
    return p


def records(model: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in (DATA / "results" / model).glob("*.json")]


def find(recs: list[dict], label: str, build: str = "cuda", **how) -> dict | None:
    hits = [r for r in recs if r.get("label") == label and r["how"]["build"] == build and r.get("ok")
            and all(r["how"].get(k) == v for k, v in how.items())]
    return sorted(hits, key=lambda r: r["eval_id"])[-1] if hits else None



def build_file(inv, store, assignment: dict[str, str], base_fmt: str, model: str) -> Path:
    tmp = out_dir(model) / "tmp"
    tmp.mkdir(exist_ok=True)
    path = tmp / f"{assignment_digest(assignment)[:16]}.gguf"
    if not path.exists():
        part = path.with_suffix(".part")
        assemble(inv, assignment, store, part, base_format=base_fmt)
        part.rename(path)
    if not verify(inv, assignment, path).ok:
        raise SystemExit(f"read-back failed for {path.name}")
    return path


def kl(model: str, gguf: Path, ref: str, label: str, build: str = "cuda") -> None:
    rc = evaluator.main(["kl", "--model", model, "--gguf", str(gguf), "--build", build, "--reference", ref,
                         "--label", label, "--threads", str(THREADS)])
    if rc != 0:
        raise SystemExit(f"{label} {build}: KL run failed")


# --- P3 references ------------------------------------------------------------------------------------------

def cmd_refs(model: str) -> int:
    ref_dir = DATA / "reference" / model
    ref_dir.mkdir(parents=True, exist_ok=True)
    bf16 = DATA / "store" / model / f"{model}-BF16.gguf"
    manifest_path = ref_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "model": model, "built_from": bf16.name, "build": "cuda", "kl_runs_must_use": {"-c": 2048, "-b": 2048},
        "references": {}}
    store_manifest = json.loads((DATA / "store" / model / "manifest.json").read_text())
    manifest["bf16_sha256"] = store_manifest["bf16"]["sha256"]
    manifest["llama_cpp_commit"] = store_manifest["llama_cpp_commit"]
    for name, chunks in (("sensitivity", SENS_CHUNKS), ("selection", None)):
        kld = ref_dir / f"{name}.kld"
        if name in manifest["references"] and kld.exists():
            continue
        text = DATA / "splits" / "text" / f"{name}.txt"
        cmd = [str(LLAMA / "build-cuda/bin/llama-perplexity"), "-m", str(bf16), "-f", str(text), "-c", "2048",
               "-b", "2048", "-ngl", "99", "-t", str(THREADS), "--kl-divergence-base", str(kld) + ".part"]
        if chunks:
            cmd += ["--chunks", str(chunks)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        log = out.stdout + out.stderr
        (ref_dir / f"ref-{name}.log").write_text(log)
        part = Path(str(kld) + ".part")
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", log)
        n = re.search(r"over (\d+) chunks", log)
        if not part.exists() or part.stat().st_size == 0 or not m or not n:
            raise SystemExit(f"reference {name} failed; see {ref_dir}/ref-{name}.log")
        part.rename(kld)
        h = hashlib.sha256()
        with open(kld, "rb") as f:
            for c in iter(lambda: f.read(1 << 24), b""):
                h.update(c)
        manifest["references"][name] = {"file": kld.name, "bytes": kld.stat().st_size, "sha256": h.hexdigest(),
                                        "context": 2048, "chunks": int(n.group(1)), "bf16_ppl": float(m.group(1))}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"reference {name}: {n.group(1)} chunks, {kld.stat().st_size / 1e9:.1f} GB, BF16 PPL {m.group(1)}",
              flush=True)
    return 0


# --- P4 CPU correction (D37 method, two genes) ---------------------------------------------------------------

def correction_genes(inv) -> list[str]:
    head = next(g.gene_id for g in inv.genes if g.kind in ("head", "embd_head"))
    return [head, f"mlp.{inv.geometry.n_layers - 1}"]


def cmd_correct(model: str) -> int:
    inv, menu, store, _ = setup(model)
    if not set(menu["formats"]) - A256:
        (out_dir(model) / "correction.json").write_text(json.dumps(
            {"a_g": {}, "note": "every menu format rounds activations the same way; no correction needed (D42)"}))
        print("no correction needed for this menu")
        return 0
    base = uniform(inv, "Q8_0")
    configs = {"base": base} | {g: {**base, g: "Q6_K"} for g in correction_genes(inv)}
    for name, a in configs.items():
        recs = records(model)
        todo = [b for b in ("cuda", "cpu") if not find(recs, f"{L}:corr:{name}", b, reference="sensitivity")]
        if not todo:
            continue
        path = build_file(inv, store, a, "Q8_0", model)
        for b in todo:
            kl(model, path, "sensitivity", f"{L}:corr:{name}", b)
        path.unlink()
    recs = records(model)
    v = {(n, b): find(recs, f"{L}:corr:{n}", b, reference="sensitivity")["metrics"]["kl_mean"]
         for n in configs for b in ("cuda", "cpu")}
    a_g = {g: (v[(g, "cpu")] - v[("base", "cpu")]) - (v[(g, "cuda")] - v[("base", "cuda")])
           for g in correction_genes(inv)}
    (out_dir(model) / "correction.json").write_text(json.dumps({"a_g": a_g, "kl": {f"{k[0]}/{k[1]}": x for k, x in v.items()}}, indent=2))
    print("a_g:", {g: round(x, 6) for g, x in a_g.items()})
    return 0


# --- P5 penalty table --------------------------------------------------------------------------------------

def cmd_penalties(model: str) -> int:
    inv, menu, store, allowed = setup(model)
    base_fmt = menu["base"]
    base = uniform(inv, base_fmt)
    recs = records(model)
    if not find(recs, f"{L}:pen:base", reference="sensitivity"):
        path = build_file(inv, store, base, base_fmt, model)
        kl(model, path, "sensitivity", f"{L}:pen:base")
        path.unlink()
    measured = list(allowed)[:MAX_PEN_GENES] if MAX_PEN_GENES else list(allowed)
    jobs = [(g, f) for g in measured for f in allowed[g] if f != base_fmt
            and not find(recs, f"{L}:pen:{g}:{f}", reference="sensitivity")]
    print(f"{len(jobs)} penalty measurements to run", flush=True)

    def one(job):
        g, f = job
        path = build_file(inv, store, {**base, g: f}, base_fmt, model)
        try:
            kl(model, path, "sensitivity", f"{L}:pen:{g}:{f}")
        finally:
            path.unlink(missing_ok=True)

    with ThreadPoolExecutor(CONCURRENT) as ex:
        list(ex.map(one, jobs))

    recs = records(model)
    base_kl = find(recs, f"{L}:pen:base", reference="sensitivity")["metrics"]["kl_mean"]
    def value(g: str, f: str) -> float:
        if f == base_fmt:
            return 0.0
        if g not in measured:  # smoke mode only: placeholder for unmeasured genes
            return 0.001 * (len(menu["formats"]) - menu["formats"].index(f))
        return find(recs, f"{L}:pen:{g}:{f}", reference="sensitivity")["metrics"]["kl_mean"] - base_kl
    pen = {g: {f: value(g, f) for f in fs} for g, fs in allowed.items()}
    (out_dir(model) / "penalties.json").write_text(json.dumps({"base": base_fmt, "base_kl": base_kl,
                                                               "penalty": pen}, indent=1))
    print(f"penalty table written: {sum(len(v) for v in pen.values())} entries, base KL {base_kl:.5f}")
    return 0


# --- P6 solver + shortlist ---------------------------------------------------------------------------------

TASKS = ("hellaswag", "arc_challenge", "mmlu", "winogrande")


def corrected_penalties(model: str, base_fmt: str) -> dict:
    p = json.loads((out_dir(model) / "penalties.json").read_text())["penalty"]
    a_g = json.loads((out_dir(model) / "correction.json").read_text())["a_g"]
    for g, a in a_g.items():
        for f in p[g]:
            p[g][f] += a * ((f in A256) - (base_fmt in A256))  # D36 penalty correction
    return p


def cmd_shortlist(model: str) -> int:
    inv, menu, store, allowed = setup(model)
    base_fmt = menu["base"]
    pen = corrected_penalties(model, base_fmt)
    prob = Problem(inv, allowed, pen, R_MAX, CTX, OVERHEAD)
    sols = solve_top_k(prob, k=TOPK)
    a_g = json.loads((out_dir(model) / "correction.json").read_text())["a_g"]
    cands = {f"top{i + 1}": s.assignment for i, s in enumerate(sols)}
    baseline = uniform(inv, base_fmt)
    if estimate_ram(inv, baseline, CTX, OVERHEAD).total_gib <= R_MAX:
        cands["uniform"] = baseline
    od = out_dir(model)
    rows = {}
    for name, a in cands.items():
        label = f"{L}:short:{name}:{assignment_digest(a)[:12]}"
        recs = records(model)
        need_kl = not find(recs, label, reference="selection")
        need_tasks = [t for t in TASKS if not find(recs, label, task=t, split="selection")]
        if need_kl or need_tasks:
            path = build_file(inv, store, a, base_fmt, model)
            if need_kl:
                kl(model, path, "selection", label)
            for t in need_tasks:
                rc = evaluator.main(["task", "--model", model, "--gguf", str(path), "--build", "cuda", "--task", t,
                                     "--split", "selection", "--label", label, "--threads", str(THREADS)])
                if rc != 0:
                    raise SystemExit(f"{label} {t}: task run failed")
            path.unlink()
        recs = records(model)
        k = find(recs, label, reference="selection")["metrics"]
        acc = {t: find(recs, label, task=t, split="selection")["metrics"]["accuracy_pct"] for t in TASKS}
        corr = sum(v for g, v in a_g.items() if a[g] in A256)
        ram = estimate_ram(inv, a, CTX, OVERHEAD)
        rows[name] = {"label": label, "assignment": a, "ram_gib": ram.total_gib,
                      "kl_selection_cuda": k["kl_mean"], "kl_selection_cpu_est": k["kl_mean"] + corr,
                      "top1_pct": k["top1_pct"], "ppl_ratio": k["ppl_ratio"], "accuracy_pct": acc,
                      "mean_accuracy_pct": sum(acc.values()) / len(acc),
                      "predicted_penalty": prob.total_penalty(a)}
        print(f"{name:8s} RAM {ram.total_gib:.3f} GiB | KL sel {k['kl_mean']:.5f} (CPU est {rows[name]['kl_selection_cpu_est']:.5f}) "
              f"| top-1 {k['top1_pct']:.1f}% | tasks " + " ".join(f"{t[:4]} {v:.1f}" for t, v in acc.items()), flush=True)
    winner = min((n for n in rows if n != "uniform"), key=lambda n: rows[n]["kl_selection_cpu_est"])
    (od / "shortlist.json").write_text(json.dumps({"r_max_gib": R_MAX, "winner": winner, "rows": rows}, indent=1))
    print(f"winner: {winner} (lowest estimated CPU KL on selection)")
    return 0


# --- P7 verdict on the fixed test subset (only after the user's approval) -----------------------------------

SUBSET = {"mmlu": 1000, "hellaswag": 2000, "arc_challenge": None, "winogrande": None}  # None: all items
GSM8K_N = 300
SUBSET_SEED = 20261111


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x]


def build_test_subset() -> None:
    import numpy as np

    from panther.quality.tasks import (mc_tasks, mmlu_fewshot_by_subject, read_mc_binary, write_mc_binary,
                                       write_winogrande_csv, parse_winogrande_csv)
    lt = DATA / "splits" / "llama_tasks"
    manifest = json.loads((lt / "manifest.json").read_text())
    if "poc_test/mmlu" in manifest["files"]:
        return
    rng = np.random.default_rng(SUBSET_SEED)
    fewshot = mmlu_fewshot_by_subject(jsonl(DATA / "splits/tasks/fewshot/mmlu.jsonl"))
    (lt / "poc_test").mkdir(exist_ok=True)
    for task, n in SUBSET.items():
        items = jsonl(DATA / "splits/tasks/test" / f"{task}.jsonl")
        if n is not None:
            items = [items[i] for i in sorted(rng.choice(len(items), size=n, replace=False).tolist())]
        if task == "winogrande":
            data = write_winogrande_csv(items).encode()
            assert len(parse_winogrande_csv(data.decode())[0]) == len(items)
            path = lt / "poc_test" / "winogrande.csv"
        else:
            tasks = mc_tasks(task, items, fewshot if task == "mmlu" else None)
            data = write_mc_binary(tasks)
            assert read_mc_binary(data) == tasks
            path = lt / "poc_test" / f"{task}.bin"
        path.write_bytes(data)
        manifest["files"][f"poc_test/{task}"] = {"file": str(path.relative_to(DATA / "splits")), "n": len(items),
                                                 "sha256": hashlib.sha256(data).hexdigest(),
                                                 "ids": [it["id"] for it in items]}
    g = jsonl(DATA / "splits/tasks/test/gsm8k.jsonl")
    g = [g[i] for i in sorted(rng.choice(len(g), size=GSM8K_N, replace=False).tolist())]
    (lt / "poc_test" / "gsm8k.jsonl").write_text("\n".join(json.dumps(x) for x in g) + "\n")
    (lt / "manifest.json").write_text(json.dumps(manifest, indent=1))


def run_gsm8k(model: str, path: Path, label: str) -> dict:
    from panther.quality import gsm8k
    from panther.quality.server import LlamaServer

    rec_path = out_dir(model) / f"gsm8k-{label.replace(':', '_')}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    items = (jsonl(DATA / "splits/tasks/selection/gsm8k.jsonl")[:20] if SMOKE
             else jsonl(DATA / "splits/llama_tasks/poc_test/gsm8k.jsonl"))
    fewshot = jsonl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    with LlamaServer(LLAMA, path, "cuda", parallel=8, ctx_per_slot=2048, threads=THREADS,
                     log=out_dir(model) / "server.log") as srv:
        comps = srv.map(lambda it: srv.complete(gsm8k.prompt(it["question"], fewshot), gsm8k.MAX_TOKENS,
                                                gsm8k.STOP), items)
    correct = [int(gsm8k.exact_match(gsm8k.strict_match(c), it["answer"])) for it, c in zip(items, comps)]
    rec = {"ids": [it["id"] for it in items], "correct": correct, "completions": comps}
    rec_path.write_text(json.dumps(rec))
    return rec


def evaluate_config(model: str, name: str, a: dict, store, inv, base_fmt: str):
    import numpy as np

    from panther.stats.decision import Evaluation
    label = f"{L}:test:{name}:{assignment_digest(a)[:12]}"
    recs = records(model)
    todo_tasks = [t for t in SUBSET if not find(recs, label, task=t, split=V_SPLIT)]
    todo_ppl = not find(recs, label, text=V_TEXT)
    gsm_path = out_dir(model) / f"gsm8k-{label.replace(':', '_')}.json"
    path = None
    if todo_tasks or todo_ppl or not gsm_path.exists():
        path = build_file(inv, store, a, base_fmt, model)
        for t in todo_tasks:
            if evaluator.main(["task", "--model", model, "--gguf", str(path), "--build", "cuda", "--task", t,
                               "--split", V_SPLIT, "--label", label, "--threads", str(THREADS)]) != 0:
                raise SystemExit(f"{label} {t} failed")
        if todo_ppl and evaluator.main(["ppl", "--model", model, "--gguf", str(path), "--build", "cuda",
                                        "--text", V_TEXT, "--label", label, "--threads", str(THREADS)]) != 0:
            raise SystemExit(f"{label} ppl failed")
        g = run_gsm8k(model, path, label)
        path.unlink()
    else:
        g = json.loads(gsm_path.read_text())
    recs = records(model)
    correct = {t: np.array(find(recs, label, task=t, split=V_SPLIT)["per_item"]["correct"], dtype=np.int8)
               for t in SUBSET}
    correct["gsm8k"] = np.array(g["correct"], dtype=np.int8)
    p = find(recs, label, text=V_TEXT)["per_chunk"]
    nll = np.array(p["mean_nll"]) * p["tokens"]
    toks = np.full(len(nll), p["tokens"])
    ram = int(estimate_ram(inv, a, CTX, OVERHEAD).total)
    return Evaluation(f"{model.split('-')[1]}:{name}", correct, nll, toks, ram, 0.0)


def cmd_verdict(_: str) -> int:
    from panther.stats.decision import compare
    if not SMOKE and not (DATA / "poc" / "APPROVED_TEST_SUBSET").exists():
        raise SystemExit("P7 uses the test subset and needs the user's explicit approval (D42); "
                         "create $PANTHER_DATA/poc/APPROVED_TEST_SUBSET once approved")
    build_test_subset()
    evs = {}
    for model in MENU:
        inv, menu, store, _ = setup(model)
        sl = json.loads((out_dir(model) / "shortlist.json").read_text())
        for name in (sl["winner"], "uniform"):
            if name in sl["rows"]:
                evs[(model, name)] = evaluate_config(model, name, sl["rows"][name]["assignment"], store, inv,
                                                     menu["base"])
    models = list(MENU)
    winners = {m: json.loads((out_dir(m) / "shortlist.json").read_text())["winner"] for m in models}
    pairs = {}
    if len(models) >= 2:
        m7, m14 = models[:2]
        pairs["7B winner vs 14B winner"] = (evs[(m7, winners[m7])], evs[(m14, winners[m14])])
    for m, w in winners.items():
        if (m, "uniform") in evs:
            pairs[f"{m.split('-')[1]} winner vs uniform"] = (evs[(m, w)], evs[(m, "uniform")])
    out = {"configs": {e.name: {"mean_accuracy_pct": 100 * e.mean_accuracy(), "ppl_test_web": e.perplexity(),
                                "per_task_pct": {t: 100 * float(v.mean()) for t, v in e.task_correct.items()},
                                "ram_gib": e.peak_ram_bytes / 2**30} for e in evs.values()},
           "comparisons": {}}
    for k, (a, b) in pairs.items():
        c = compare(a, b)
        out["comparisons"][k] = {"winner": c.winner, "decided_by": c.decided_by, "acc_diff_pts": 100 * c.acc_diff,
                                 "acc_ci_pts": [100 * c.acc_ci[0], 100 * c.acc_ci[1]], "ppl_ratio": c.ppl_ratio,
                                 "ppl_ratio_ci": list(c.ppl_ratio_ci)}
    (POC / "verdict.json").write_text(json.dumps(out, indent=1))
    for n, v in out["configs"].items():
        print(f"{n:22s} RAM {v['ram_gib']:.2f} GiB  mean acc {v['mean_accuracy_pct']:.2f}%  PPL {v['ppl_test_web']:.3f}  "
              + " ".join(f"{t[:4]} {x:.1f}" for t, x in v["per_task_pct"].items()))
    for k, c in out["comparisons"].items():
        print(f"{k}: winner {c['winner']} by {c['decided_by']} | acc diff {c['acc_diff_pts']:+.2f} pts "
              f"CI [{c['acc_ci_pts'][0]:+.2f}, {c['acc_ci_pts'][1]:+.2f}] | PPL ratio {c['ppl_ratio']:.4f}")
    return 0


# --- P8 Pi bundle -----------------------------------------------------------------------------------------

AGREEMENT_TOKENS = 4096 + 64  # two 2048-token chunks for llama-perplexity, plus margin


def make_agreement_text(model: str) -> None:
    """First ~4k tokens of the sensitivity text (never test), decoded, as splits/text/agreement.txt."""
    path = DATA / "splits" / "text" / "agreement.txt"
    if path.exists():
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DATA / "hf" / model)
    ids = tok((DATA / "splits/text/sensitivity.txt").read_text(), add_special_tokens=False)["input_ids"]
    path.write_text(tok.decode(ids[:AGREEMENT_TOKENS]))


def type_counts(inv, a: dict[str, str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for g in inv.genes:
        k = a[g.gene_id].lower()
        c[k] = c.get(k, 0) + len(g.tensors)
    c["f32"] = len(inv.fixed_tensors)
    return dict(sorted(c.items()))


def cmd_bundle(_: str) -> int:
    import shutil
    bundle = BUNDLE_DIR
    bundle.mkdir(exist_ok=True)
    verdict = json.loads((POC / "verdict.json").read_text())
    make_agreement_text(next(iter(MENU)))
    shutil.copy(DATA / "splits/text/agreement.txt", bundle / "agreement.txt")
    files = {}
    for model in MENU:
        inv, menu, store, _ = setup(model)
        sl = json.loads((out_dir(model) / "shortlist.json").read_text())
        short = model.split("-")[1].lower()  # "7b" / "14b"
        for role in ("winner", "uniform"):
            name = sl["winner"] if role == "winner" else "uniform"
            if name not in sl["rows"]:
                continue
            a = sl["rows"][name]["assignment"]
            dst = bundle / f"qwen2.5-{short}-poc-{role}.gguf"
            if not dst.exists():
                src = build_file(inv, store, a, menu["base"], model)
                src.rename(dst)
            label = f"{L}:bundle:{dst.stem}"
            if not find(records(model), label, build="cpu", text="agreement"):
                if evaluator.main(["ppl", "--model", model, "--gguf", str(dst), "--build", "cpu", "--text",
                                   "agreement", "--label", label, "--threads", str(THREADS)]) != 0:
                    raise SystemExit(f"{label}: agreement perplexity failed")
            ppl = find(records(model), label, build="cpu", text="agreement")["metrics"]["ppl"]
            ram = estimate_ram(inv, a, CTX, OVERHEAD)
            h = hashlib.sha256()
            with open(dst, "rb") as f:
                for c in iter(lambda: f.read(1 << 24), b""):
                    h.update(c)
            files[dst.name] = {"model": model, "role": role, "shortlist_name": name, "sha256": h.hexdigest(),
                               "bytes": dst.stat().st_size, "assignment_sha256": assignment_digest(a),
                               "assignment": a, "predicted_ram_gib": ram.total_gib,
                               "predicted": {"weights": ram.weights, "kv_cache": ram.kv_cache, "overhead": ram.overhead},
                               "expected_type_counts": type_counts(inv, a), "server_cpu_ppl_agreement": ppl}
            print(f"{dst.name}: {dst.stat().st_size / 2**30:.2f} GiB, predicted RAM {ram.total_gib:.3f} GiB, "
                  f"server CPU PPL {ppl:.4f}", flush=True)
    m7 = list(MENU)[0]
    cmp = verdict["comparisons"].get("7B winner vs 14B winner", {"winner": None, "decided_by": "n/a"})
    first = next(iter(files))
    recommended = ("qwen2.5-7b-poc-winner.gguf" if cmp["winner"] and cmp["winner"].startswith("7B")
                   else "qwen2.5-14b-poc-winner.gguf" if cmp["winner"] else first)
    store_manifest = json.loads((DATA / "store" / m7 / "manifest.json").read_text())
    manifest = {"r_max_gib": R_MAX, "context_tokens": CTX, "overhead_gib_assumed": OVERHEAD,
                "llama_cpp_commit": store_manifest["llama_cpp_commit"], "files": files,
                "agreement_text": "agreement.txt", "verdict": verdict,
                "recommended_file": recommended,
                "recommended_reason": (f"7B vs 14B: winner {cmp['winner']} by {cmp['decided_by']}" if cmp["winner"]
                                       else "7B vs 14B indistinguishable; the smaller model is recommended")}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"bundle ready: {bundle} (recommended: {recommended})")
    return 0


if __name__ == "__main__":
    cmds = {"refs": cmd_refs, "correct": cmd_correct, "penalties": cmd_penalties, "shortlist": cmd_shortlist,
            "verdict": cmd_verdict, "bundle": cmd_bundle}
    if len(sys.argv) != 3 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]](sys.argv[2]))
