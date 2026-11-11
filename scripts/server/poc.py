"""Proof of concept: 7B vs 14B at R_max = 7.0 GiB (docs/DECISIONS.md D42). Runs in the GPU container.

  python scripts/server/poc.py refs       MODEL   # P3: BF16 references (sensitivity 12 chunks, selection)
  python scripts/server/poc.py correct    MODEL   # P4: CPU correction for the output head and last MLP (7B)
  python scripts/server/poc.py penalties  MODEL   # P5: penalty table, 3 KL runs at a time
  python scripts/server/poc.py shortlist  MODEL   # P6: solver top 5 + uniform baseline, scored on selection

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

MENU = {
    "Qwen2.5-7B-Instruct": {"formats": ["Q8_0", "Q6_K", "Q5_K", "Q4_K", "IQ4_XS", "IQ3_S"], "base": "Q6_K"},
    "Qwen2.5-14B-Instruct": {"formats": ["Q6_K", "Q4_K", "IQ4_XS", "IQ3_S", "IQ3_XXS", "Q2_K", "IQ2_XS"],
                             "base": "IQ3_XXS"},
}


def setup(model: str):
    inv = build_inventory(load_geometry(model))
    menu = MENU[model]
    store = TensorStore({f: DATA / "store" / model / f"{model}-{f}.gguf" for f in menu["formats"]})
    allowed = {g: [f for f in fs if f in menu["formats"]] for g, fs in gene_formats(inv).items()}
    return inv, menu, store, allowed


def out_dir(model: str) -> Path:
    p = DATA / "poc" / model
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
        todo = [b for b in ("cuda", "cpu") if not find(recs, f"poc:corr:{name}", b, reference="sensitivity")]
        if not todo:
            continue
        path = build_file(inv, store, a, "Q8_0", model)
        for b in todo:
            kl(model, path, "sensitivity", f"poc:corr:{name}", b)
        path.unlink()
    recs = records(model)
    v = {(n, b): find(recs, f"poc:corr:{n}", b, reference="sensitivity")["metrics"]["kl_mean"]
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
    if not find(recs, "poc:pen:base", reference="sensitivity"):
        path = build_file(inv, store, base, base_fmt, model)
        kl(model, path, "sensitivity", "poc:pen:base")
        path.unlink()
    jobs = [(g, f) for g, fs in allowed.items() for f in fs if f != base_fmt
            and not find(recs, f"poc:pen:{g}:{f}", reference="sensitivity")]
    print(f"{len(jobs)} penalty measurements to run", flush=True)

    def one(job):
        g, f = job
        path = build_file(inv, store, {**base, g: f}, base_fmt, model)
        try:
            kl(model, path, "sensitivity", f"poc:pen:{g}:{f}")
        finally:
            path.unlink(missing_ok=True)

    with ThreadPoolExecutor(CONCURRENT) as ex:
        list(ex.map(one, jobs))

    recs = records(model)
    base_kl = find(recs, "poc:pen:base", reference="sensitivity")["metrics"]["kl_mean"]
    pen = {g: {f: (0.0 if f == base_fmt else
                   find(recs, f"poc:pen:{g}:{f}", reference="sensitivity")["metrics"]["kl_mean"] - base_kl)
               for f in fs} for g, fs in allowed.items()}
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
    sols = solve_top_k(prob, k=5)
    a_g = json.loads((out_dir(model) / "correction.json").read_text())["a_g"]
    cands = {f"top{i + 1}": s.assignment for i, s in enumerate(sols)}
    baseline = uniform(inv, base_fmt)
    if estimate_ram(inv, baseline, CTX, OVERHEAD).total_gib <= R_MAX:
        cands["uniform"] = baseline
    od = out_dir(model)
    rows = {}
    for name, a in cands.items():
        label = f"poc:short:{name}:{assignment_digest(a)[:12]}"
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


if __name__ == "__main__":
    cmds = {"refs": cmd_refs, "correct": cmd_correct, "penalties": cmd_penalties, "shortlist": cmd_shortlist}
    if len(sys.argv) != 3 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]](sys.argv[2]))
