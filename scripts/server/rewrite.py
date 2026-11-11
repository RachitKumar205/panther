"""Rewrite of the search (TODO.md; D45, D46): per-matrix formats over the configs/search.yaml table, with
one sensitivity KL run per block. Runs in the GPU container; every step is resumable.

  rewrite.py plan                 # print menus, bases, probes and contests (no work)
  rewrite.py store   MODEL        # BF16 + importance matrix (new models) and the missing menu formats
  rewrite.py refs    MODEL        # BF16 KL references: sensitivity (12 chunks) and selection
  rewrite.py error   MODEL        # importance-weighted dequantization error per matrix and format
  rewrite.py sens    MODEL        # one KL run per block: the block moved from base to probe format
  rewrite.py contest BUDGET MODEL # solve top 3; references; selection KL for all; winner = lowest KL of top 3
  rewrite.py eval    BUDGET       # test-subset evaluation of winners and references, paired comparisons
  rewrite.py bundle  BUDGET       # winners + best references as GGUF files and a manifest for the Pi

Outputs: $PANTHER_DATA/rewrite/<model>/ and $PANTHER_DATA/rewrite/budget-<B>/; KL/task records in
$PANTHER_DATA/results/<model>/ (labels rw:...).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poc  # noqa: E402  (records, find, KL runner, test subset, GSM8K, task list)
from panther.costs.formats import bits_per_weight, gene_formats  # noqa: E402
from panther.costs.ram import GIB, estimate_ram, fixed_bytes, gene_bytes  # noqa: E402
from panther.gguftools.assemble import TensorStore, assemble, assignment_digest  # noqa: E402
from panther.gguftools.verify import verify  # noqa: E402
from panther.models.inventory import REPO_ROOT, block_of, build_matrix_inventory, load_geometry  # noqa: E402

DATA, LLAMA, THREADS = poc.DATA, poc.LLAMA, poc.THREADS
RW = DATA / "rewrite"
CFG = yaml.safe_load((REPO_ROOT / "configs" / "search.yaml").read_text())
CTX = CFG["context_tokens"]
OVERHEAD = CFG["overhead_gib"]
KV_BPE = {"Q8_0": 34 / 32, "F16": 2.0}[CFG["kv_cache_type"]]
MENU_ALL = CFG["formats"]
CONCURRENT = int(os.environ.get("PANTHER_KL_CONCURRENT", "10"))  # D48
GPU_HEADROOM_MIB = 4096  # per KL run on top of the file size: compute buffers and logits


def wait_for_gpu_memory(need_bytes: int, timeout_s: int = 3600) -> None:
    """Several lanes share the GPU: start a KL run only when the file plus headroom fits in free memory."""
    import random
    import shutil
    import time
    if not shutil.which("nvidia-smi"):
        return
    need = need_bytes / 2**20 + GPU_HEADROOM_MIB
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        out_ = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()
        if not out_ or float(out_[0]) >= need:
            return
        time.sleep(random.uniform(5, 15))
CALIB = DATA / "splits" / "calibration.txt"
REF_MODEL_FORMAT = "Q6_K"  # cross-model references: uniform Q6_K, accuracy only (D45)
STOCK = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
         "Q3_K_S", "Q3_K_M", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"]
STOCK_BPW = {"IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7,
             "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66, "Q3_K_S": 3.5, "Q3_K_M": 3.9,
             "IQ4_XS": 4.25, "Q4_K_S": 4.6, "Q4_K_M": 4.9, "Q5_K_S": 5.5, "Q5_K_M": 5.7, "Q6_K": 6.6}


def contests() -> dict[int, dict]:
    return {int(b): c for b, c in CFG["contests"].items()}


def out(*parts: str) -> Path:
    p = RW.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@lru_cache(maxsize=None)
def inv_of(model: str):
    return build_matrix_inventory(load_geometry(model))


def ram_gib(model: str, a: dict[str, str]) -> float:
    return estimate_ram(inv_of(model), a, CTX, OVERHEAD, KV_BPE).total_gib


def avg_bpw(model: str, budget: float) -> float:
    inv = inv_of(model)
    room = budget * GIB - inv.geometry.kv_cache_bytes(CTX, KV_BPE) - OVERHEAD * GIB - fixed_bytes(inv)
    return 8 * room / sum(g.n_weights for g in inv.genes)


def searched_budgets(model: str) -> list[int]:
    active = set(CFG.get("active_budgets", contests()))  # D48: only active budgets are searched for now
    return sorted(b for b, c in contests().items() if model in c["searched"] and b in active)


@lru_cache(maxsize=None)
def menu(model: str) -> list[str]:
    """D46: formats between the lowest budget average - 1.5 and the highest + 2.5 bits per weight."""
    bs = [avg_bpw(model, b) for b in searched_budgets(model)]
    lo, hi = min(bs) - 1.5, max(bs) + 2.5
    return [f for f in MENU_ALL if lo <= bits_per_weight(f) <= hi]


@lru_cache(maxsize=None)
def allowed(model: str) -> dict[str, list[str]]:
    m = menu(model)
    return {g: [f for f in fs if f in m] for g, fs in gene_formats(inv_of(model)).items()}


def fill(model: str, fmt: str) -> dict[str, str]:
    """Uniform `fmt`; genes that cannot use it take the nearest allowed format above (else below)."""
    out_ = {}
    for g, fs in allowed(model).items():
        if fmt in fs:
            out_[g] = fmt
        else:
            up = [f for f in fs if bits_per_weight(f) >= bits_per_weight(fmt)]
            out_[g] = min(up, key=bits_per_weight) if up else max(fs, key=bits_per_weight)
    return out_


def base_format(model: str) -> str:
    """D46: the menu format closest to the lowest searched budget's average bits per weight."""
    target = avg_bpw(model, searched_budgets(model)[0])
    return min(menu(model), key=lambda f: abs(bits_per_weight(f) - target))


def probe_format(model: str) -> str:
    """D47: the highest menu format at most `sensitivity_probe_up_bpw` above the base (else the one below)."""
    m = sorted(menu(model), key=bits_per_weight)
    b = bits_per_weight(base_format(model))
    up = [f for f in m if b < bits_per_weight(f) <= b + CFG["sensitivity_probe_up_bpw"]]
    return up[-1] if up else m[m.index(base_format(model)) - 1]


def store(model: str) -> TensorStore:
    return TensorStore({f: DATA / "store" / model / f"{model}-{f}.gguf" for f in menu(model)})


def build_file(model: str, a: dict[str, str]) -> Path:
    tmp = out(model, "tmp", "x").parent
    path = tmp / f"{assignment_digest(a)[:16]}.gguf"
    if not path.exists():
        part = path.with_suffix(".part")
        assemble(inv_of(model), a, store(model), part, base_format=base_format(model))
        part.rename(path)
    if not verify(inv_of(model), a, path).ok:
        raise SystemExit(f"read-back failed for {path.name}")
    return path


def blocks(model: str) -> dict[str, list[str]]:
    b: dict[str, list[str]] = {}
    for g in inv_of(model).genes:
        b.setdefault(block_of(g), []).append(g.gene_id)
    return b


# --- plan ----------------------------------------------------------------------------------------------------

def cmd_plan() -> int:
    for b, c in contests().items():
        print(f"{b} GiB: searched {c['searched']}, reference {c['reference']}")
    models = sorted({m for c in contests().values() for m in c["searched"] if searched_budgets(m)},
                    key=lambda m: inv_of(m).geometry.hidden)
    for m in models:
        print(f"{m}: budgets {searched_budgets(m)} avg bpw {[round(avg_bpw(m, b), 2) for b in searched_budgets(m)]}"
              f" | menu {menu(m)} | base {base_format(m)} -> probe {probe_format(m)} | blocks {len(blocks(m))}"
              f" | matrices {len(inv_of(m).genes)}")
        lo = {g: min(fs, key=bits_per_weight) for g, fs in allowed(m).items()}
        print(f"   all-smallest RAM {ram_gib(m, lo):.3f} GiB")
    return 0


# --- store ---------------------------------------------------------------------------------------------------

def cmd_store(model: str) -> int:
    fmts = menu(model) if searched_budgets(model) else [REF_MODEL_FORMAT]
    missing = [f for f in fmts if not (DATA / "store" / model / f"{model}-{f}.gguf").exists()]
    print(f"{model}: menu {fmts}; to build {missing}", flush=True)
    if not missing and (DATA / "store" / model / f"{model}-BF16.gguf").exists():
        return 0
    env = dict(os.environ, PANTHER_FORMATS=" ".join(missing))
    return subprocess.run(["bash", str(REPO_ROOT / "scripts/server/build_tensor_store.sh"), model, str(CALIB)],
                          env=env).returncode


# --- error ---------------------------------------------------------------------------------------------------

_READERS: dict[Path, dict] = {}
_IMP: dict[str, np.ndarray] = {}


def _tensor(path: Path, name: str) -> np.ndarray:
    import gguf
    from gguf.quants import dequantize
    if path not in _READERS:
        _READERS[path] = {t.name: t for t in gguf.GGUFReader(path).tensors}
    t = _READERS[path][name]
    return dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1, int(t.shape[0]))


def _init_error(model: str) -> None:
    import gguf
    global _IMP
    r = gguf.GGUFReader(DATA / "store" / model / f"{model}.imatrix.gguf")
    t = {x.name: np.asarray(x.data, dtype=np.float64) for x in r.tensors}
    _IMP = {n[: -len(".in_sum2")]: v.reshape(-1) / max(float(np.asarray(t[n[: -len(".in_sum2")] + ".counts"]).reshape(-1)[0]), 1.0)
            for n, v in t.items() if n.endswith(".in_sum2")}


def _error_one(job: tuple[str, str, list[str]]) -> tuple[str, dict[str, float]]:
    model, name, formats = job
    s = DATA / "store" / model
    w = _tensor(s / f"{model}-BF16.gguf", name)
    imp = _IMP.get(name, np.ones(w.shape[1]))[: w.shape[1]]
    return name, {f: float(((w - _tensor(s / f"{model}-{f}.gguf", name)) ** 2).sum(axis=0) @ imp) for f in formats}


def cmd_error(model: str) -> int:
    dst = out(model, "error.json")
    done = json.loads(dst.read_text()) if dst.exists() else {}
    jobs = []
    for g in inv_of(model).genes:
        fs = [f for f in allowed(model)[g.gene_id] if f not in done.get(g.tensors[0].name, {})]
        if fs:
            jobs.append((model, g.tensors[0].name, fs))
    print(f"{len(jobs)} matrices to do", flush=True)
    with ProcessPoolExecutor(8, initializer=_init_error, initargs=(model,)) as ex:
        for i, (name, res) in enumerate(ex.map(_error_one, jobs), 1):
            done.setdefault(name, {}).update(res)
            if i % 25 == 0 or i == len(jobs):
                dst.write_text(json.dumps(done))
                print(f"  {i}/{len(jobs)}", flush=True)
    return 0


# --- sensitivity (D47: adaptive) ------------------------------------------------------------------------------

def block_assignment(model: str, block: str | None, fmt: str | None = None) -> dict[str, str]:
    """The base, with every matrix of `block` moved to `fmt` where allowed."""
    a = fill(model, base_format(model))
    if block is not None:
        for g in blocks(model)[block]:
            if fmt in allowed(model)[g]:
                a[g] = fmt
    return a


def sens_label(model: str, block: str, fmt: str) -> str:
    return f"rw:sens:{base_format(model)}:{block}:{fmt}"


def _sens_one(job: tuple[str, str, str]) -> None:
    model, block, fmt = job
    path = build_file(model, block_assignment(model, None if block == "base" else block, fmt))
    wait_for_gpu_memory(path.stat().st_size)
    try:
        poc.kl(model, path, "sensitivity", sens_label(model, block, fmt))
    finally:
        path.unlink(missing_ok=True)


def measured_table(model: str) -> tuple[float | None, dict[str, dict[str, float]]]:
    """Base KL and measured dKL per (block, format), from the KL records."""
    base = base_format(model)
    pre = f"rw:sens:{base}:"
    kls = {}
    for r in poc.records(model):
        lab = r.get("label") or ""
        if lab.startswith(pre) and r.get("ok") and r["how"].get("reference") == "sensitivity":
            kls[lab[len(pre):]] = r["metrics"]["kl_mean"]
    base_kl = kls.get(f"base:{base}")
    table: dict[str, dict[str, float]] = {}
    if base_kl is not None:
        for k, v in kls.items():
            b, f = k.rsplit(":", 1)
            if b != "base":
                table.setdefault(b, {})[f] = v - base_kl
    return base_kl, table


def block_formats(model: str, block: str) -> list[str]:
    al = allowed(model)
    return sorted({f for g in blocks(model)[block] for f in al[g]}, key=bits_per_weight)


def penalties(model: str, table: dict[str, dict[str, float]] | None = None,
              err: dict | None = None) -> dict[str, dict[str, float]]:
    """D47: per block, dKL as a piecewise-linear function of the block's offline error E through (base, 0) and
    the measured points; per matrix, penalty(m, f) = r_b(f) * (e(m, f) - e(m, base)), r_b(f) = dKL_b(f) / dE_b(f)."""
    if table is None:
        table = measured_table(model)[1]
    if err is None:
        err = json.loads(out(model, "error.json").read_text())
    inv = inv_of(model)
    name = {g.gene_id: g.tensors[0].name for g in inv.genes}
    base_a = block_assignment(model, None)
    pen = {}
    for b, gs in blocks(model).items():
        def E(f: str) -> float:
            a = block_assignment(model, b, f)
            return sum(err[name[g]][a[g]] - err[name[g]][base_a[g]] for g in gs)
        pts = sorted({(0.0, 0.0)} | {(E(f), v) for f, v in table.get(b, {}).items()})
        xs, ys = np.array([x for x, _ in pts]), np.array([y for _, y in pts])
        def dkl(x: float) -> float:
            if len(xs) == 1:
                return 0.0
            if x < xs[0]:
                return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
            if x > xs[-1]:
                return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1])
            return float(np.interp(x, xs, ys))
        ratio = {}
        for f in block_formats(model, b):
            x = E(f)
            ratio[f] = dkl(x) / x if abs(x) > 1e-12 else 0.0
        for g in gs:
            e0 = err[name[g]][base_a[g]]
            pen[g] = {f: ratio[f] * (err[name[g]][f] - e0) for f in allowed(model)[g]}
    return pen


def next_measurements(model: str, pen: dict[str, dict[str, float]], measured: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Solve every budget of the model; for each block, the formats its matrices use and their menu neighbours."""
    from panther.search.problem import Problem
    from panther.search.solver import solve_top_k
    k = CFG["remeasure_neighbours"]
    base = base_format(model)
    want = set()
    for budget in searched_budgets(model):
        prob = Problem(inv_of(model), allowed(model), pen, budget, CTX, OVERHEAD, kv_bytes_per_element=KV_BPE)
        a = solve_top_k(prob, k=1, time_limit_s=60)[0].assignment
        for b, gs in blocks(model).items():
            fs = block_formats(model, b)
            for f in {a[g] for g in gs}:
                i = fs.index(f)
                want |= {(b, fs[j]) for j in range(max(0, i - k), min(len(fs), i + k + 1)) if fs[j] != base}
    return want - measured


def cmd_sens(model: str) -> int:
    base, probe = base_format(model), probe_format(model)
    stats = []
    todo = {("base", base)} | {(b, probe) for b in blocks(model)}
    for rnd in range(CFG["remeasure_rounds"] + 1):
        base_kl, table = measured_table(model)
        done = {(b, f) for b, row in table.items() for f in row} | ({("base", base)} if base_kl is not None else set())
        run = sorted(todo - done)
        print(f"{model} round {rnd}: {len(run)} KL runs (base {base}, probe {probe}, measured so far {len(done)})",
              flush=True)
        if ("base", base) in run:
            _sens_one((model, "base", base))
            run.remove(("base", base))
        with ProcessPoolExecutor(CONCURRENT) as ex:
            for _ in ex.map(_sens_one, [(model, b, f) for b, f in run]):
                pass
        base_kl, table = measured_table(model)
        measured = {(b, f) for b, row in table.items() for f in row}
        stats.append({"round": rnd, "runs": len(run), "measured_total": len(measured)})
        if rnd == CFG["remeasure_rounds"]:
            break
        todo = next_measurements(model, penalties(model, table), measured)
        if not todo:
            print(f"{model}: converged after round {rnd}", flush=True)
            break
    out(model, "sens.json").write_text(json.dumps({"base": base, "probe": probe, "base_kl": base_kl,
                                                   "table": table, "rounds": stats}, indent=1))
    print(f"{model}: sensitivity done, {sum(s['runs'] for s in stats)} KL runs, base KL {base_kl:.5f}", flush=True)
    return 0


# --- contest -------------------------------------------------------------------------------------------------

def fits(model: str, a: dict[str, str], budget: float) -> bool:
    return ram_gib(model, a) <= budget


def largest_uniform(model: str, budget: float) -> tuple[str, dict[str, str]] | None:
    ok = [(f, fill(model, f)) for f in sorted(menu(model), key=bits_per_weight) if fits(model, fill(model, f), budget)]
    return ok[-1] if ok else None


def headv_padded(model: str, budget: float) -> dict[str, str] | None:
    """Uniform with head and every attn_v raised (Q6_K, or the top menu format), whole layers raised one
    format, alternating from both ends, up to the budget (D46)."""
    m = sorted(menu(model), key=bits_per_weight)
    hi = "Q6_K" if "Q6_K" in m else m[-1]
    inv = inv_of(model)
    for u in reversed(m):
        a = fill(model, u)
        for g in inv.genes:
            if (g.kind in ("head", "embd_head") or g.gene_id.startswith("attn_v.")) and hi in allowed(model)[g.gene_id]:
                a[g.gene_id] = hi
        if not fits(model, a, budget):
            continue
        nxt = m[min(m.index(u) + 1, len(m) - 1)]
        order, lo_, hi_ = [], 0, inv.geometry.n_layers - 1
        while lo_ <= hi_:
            order += [lo_] + ([hi_] if hi_ != lo_ else [])
            lo_, hi_ = lo_ + 1, hi_ - 1
        for L in order:
            trial = dict(a)
            for g in inv.genes:
                if g.layer == L and not g.gene_id.startswith("attn_v.") and nxt in allowed(model)[g.gene_id]:
                    trial[g.gene_id] = nxt
            if fits(model, trial, budget):
                a = trial
        return a
    return None


def quantize_preset(model: str, preset: str) -> Path:
    dst = DATA / "rewrite" / model / "presets" / f"{model}-{preset}.gguf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        s = DATA / "store" / model
        cmd = [str(LLAMA / "build-cuda/bin/llama-quantize"), "--imatrix", str(s / f"{model}.imatrix.gguf"),
               str(s / f"{model}-BF16.gguf"), str(dst) + ".part", preset, str(THREADS)]
        with open(dst.with_suffix(".log"), "w") as f:
            if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
                raise SystemExit(f"llama-quantize {preset} failed; see {dst.with_suffix('.log')}")
        Path(str(dst) + ".part").rename(dst)
    return dst


def file_ram_gib(model: str, path: Path) -> float:
    import gguf
    w = sum(int(t.n_bytes) for t in gguf.GGUFReader(path).tensors)
    return (w + inv_of(model).geometry.kv_cache_bytes(CTX, KV_BPE) + OVERHEAD * GIB) / GIB


def best_preset(model: str, budget: float) -> tuple[str, Path] | None:
    """Largest stock preset that fits: try the two nearest below the budget's average bits per weight, then go
    down until one fits (real sizes read back, since stock mixes raise some tensors)."""
    target = avg_bpw(model, budget)
    cands = sorted([p for p in STOCK if STOCK_BPW[p] <= target + 0.3], key=lambda p: -STOCK_BPW[p])
    found = []
    for p in cands:
        path = quantize_preset(model, p)
        if file_ram_gib(model, path) <= budget:
            found.append((p, path))
            if len(found) == 2 or STOCK_BPW[p] < target - 0.6:
                break
        else:
            path.unlink()
    return max(found, key=lambda x: file_ram_gib(model, x[1])) if found else None


def kl_sel(model: str, label: str, path: Path) -> dict:
    if not poc.find(poc.records(model), label, reference="selection"):
        poc.kl(model, path, "selection", label)
    return poc.find(poc.records(model), label, reference="selection")["metrics"]


def cmd_contest(budget: int, model: str) -> int:
    from panther.search.problem import Problem
    from panther.search.solver import solve_top_k
    od = out(f"budget-{budget}", model, "x").parent
    res_path = od / "contest.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"rows": {}}
    inv = inv_of(model)
    if "candidates" not in res:
        prob = Problem(inv, allowed(model), penalties(model), budget, CTX, OVERHEAD, kv_bytes_per_element=KV_BPE)
        sols = solve_top_k(prob, k=CFG["top_k"], time_limit_s=120)
        cands = {f"top{i + 1}": {"assignment": s.assignment, "predicted_penalty": s.penalty} for i, s in enumerate(sols)}
        lu = largest_uniform(model, budget)
        if lu:
            cands[f"uniform-{lu[0]}"] = {"assignment": lu[1]}
        hv = headv_padded(model, budget)
        if hv:
            cands["headv"] = {"assignment": hv}
        res["candidates"] = cands
        res_path.write_text(json.dumps(res, indent=1))
    if "preset" not in res:
        bp = best_preset(model, budget)
        res["preset"] = {"name": bp[0], "path": str(bp[1])} if bp else None
        res_path.write_text(json.dumps(res, indent=1))
    for name, c in res["candidates"].items():
        if name in res["rows"]:
            continue
        a = c["assignment"]
        label = f"rw:b{budget}:{name}:{assignment_digest(a)[:12]}"
        path = build_file(model, a)
        k = kl_sel(model, label, path)
        path.unlink()
        res["rows"][name] = {"label": label, "ram_gib": ram_gib(model, a), "kl": k["kl_mean"], "top1_pct": k["top1_pct"],
                             "bpw": 8 * sum(gene_bytes(g, a[g.gene_id]) for g in inv.genes) / sum(g.n_weights for g in inv.genes),
                             "predicted_penalty": c.get("predicted_penalty")}
        res_path.write_text(json.dumps(res, indent=1))
        print(f"{model} @ {budget} GiB {name:16s} RAM {res['rows'][name]['ram_gib']:.3f} | KL sel {k['kl_mean']:.5f} "
              f"| top-1 {k['top1_pct']:.1f}%", flush=True)
    if res["preset"] and "preset" not in res["rows"]:
        p = res["preset"]
        k = kl_sel(model, f"rw:b{budget}:preset:{p['name']}", Path(p["path"]))
        res["rows"]["preset"] = {"label": f"rw:b{budget}:preset:{p['name']}", "name": p["name"],
                                 "ram_gib": file_ram_gib(model, Path(p["path"])), "kl": k["kl_mean"], "top1_pct": k["top1_pct"]}
        print(f"{model} @ {budget} GiB preset {p['name']:9s} RAM {res['rows']['preset']['ram_gib']:.3f} | "
              f"KL sel {k['kl_mean']:.5f} | top-1 {k['top1_pct']:.1f}%", flush=True)
    tops = [n for n in res["rows"] if n.startswith("top")]
    res["winner"] = min(tops, key=lambda n: res["rows"][n]["kl"])
    res_path.write_text(json.dumps(res, indent=1))
    print(f"{model} @ {budget} GiB winner: {res['winner']}", flush=True)
    return 0


# --- eval ----------------------------------------------------------------------------------------------------

def gsm8k_run(model: str, path: Path, label: str) -> dict:
    from panther.quality import gsm8k
    from panther.quality.server import LlamaServer
    rec_path = out(model, "gsm8k", f"{label.replace(':', '_')}.json")
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    items = poc.jsonl(DATA / "splits/llama_tasks/poc_test/gsm8k.jsonl")
    fewshot = poc.jsonl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    with LlamaServer(LLAMA, path, "cuda", parallel=8, ctx_per_slot=2048, threads=THREADS,
                     log=out(model, "server.log")) as srv:
        comps = srv.map(lambda it: srv.complete(gsm8k.prompt(it["question"], fewshot), gsm8k.MAX_TOKENS,
                                                gsm8k.STOP), items)
    correct = [int(gsm8k.exact_match(gsm8k.strict_match(c), it["answer"])) for it, c in zip(items, comps)]
    rec = {"ids": [it["id"] for it in items], "correct": correct}
    rec_path.write_text(json.dumps(rec))
    return rec


def evaluate(model: str, name: str, path_fn, ram: float):
    """The PoC verdict's evaluation (test subset: 4 tasks, GSM8K 300, perplexity on test_web)."""
    from panther.stats.decision import Evaluation
    label = f"rw:test:{name}"
    recs = poc.records(model)
    todo = [t for t in poc.SUBSET if not poc.find(recs, label, task=t, split="poc_test")]
    need_ppl = not poc.find(recs, label, text="test_web")
    gpath = out(model, "gsm8k", f"{label.replace(':', '_')}.json")
    if todo or need_ppl or not gpath.exists():
        path, temp = path_fn()
        for t in todo:
            if poc.evaluator.main(["task", "--model", model, "--gguf", str(path), "--build", "cuda", "--task", t,
                                   "--split", "poc_test", "--label", label, "--threads", str(THREADS)]) != 0:
                raise SystemExit(f"{label} {t} failed")
        if need_ppl and poc.evaluator.main(["ppl", "--model", model, "--gguf", str(path), "--build", "cuda",
                                            "--text", "test_web", "--label", label, "--threads", str(THREADS)]) != 0:
            raise SystemExit(f"{label} ppl failed")
        gsm8k_run(model, path, label)
        if temp:
            path.unlink()
    recs = poc.records(model)
    correct = {t: np.array(poc.find(recs, label, task=t, split="poc_test")["per_item"]["correct"], dtype=np.int8)
               for t in poc.SUBSET}
    correct["gsm8k"] = np.array(json.loads(gpath.read_text())["correct"], dtype=np.int8)
    p = poc.find(recs, label, text="test_web")["per_chunk"]
    nll = np.array(p["mean_nll"]) * p["tokens"]
    return Evaluation(name, correct, nll, np.full(len(nll), p["tokens"]), int(ram * GIB), 0.0)


def cmd_eval(budget: int) -> int:
    from panther.stats.decision import compare
    c = contests()[budget]
    evs = {}
    for model in c["searched"]:
        res = json.loads(out(f"budget-{budget}", model, "contest.json").read_text())
        short = model.split("-")[1]
        refs = [n for n in res["rows"] if not n.startswith("top")]
        best_ref = min(refs, key=lambda n: res["rows"][n]["kl"]) if refs else None  # D48: one reference
        for name in [res["winner"]] + ([best_ref] if best_ref else []):
            row = res["rows"][name]
            if name == "preset":
                pp = Path(res["preset"]["path"])
                fn = lambda pp=pp: (pp, False)  # noqa: E731
            else:
                a = res["candidates"][name]["assignment"]
                fn = lambda model=model, a=a: (build_file(model, a), True)  # noqa: E731
            tag = "winner" if name == res["winner"] else name
            evs[f"{short}:{tag}"] = evaluate(model, f"b{budget}:{short}:{tag}:{row['label'].split(':')[-1]}", fn,
                                             row["ram_gib"])
    for model in c["reference"]:
        short = model.split("-")[1]
        path = DATA / "store" / model / f"{model}-{REF_MODEL_FORMAT}.gguf"
        ram = file_ram_gib(model, path)
        evs[f"{short}:uniform-{REF_MODEL_FORMAT}"] = evaluate(model, f"ref:{short}:{REF_MODEL_FORMAT}",
                                                              lambda path=path: (path, False), ram)
    for k, e in evs.items():
        e.name = k
    pairs = {}
    winners = [k for k in evs if k.endswith(":winner")]
    for w in winners:
        short = w.split(":")[0]
        for k in evs:
            if k.startswith(short + ":") and k != w:
                pairs[f"{w} vs {k}"] = (evs[w], evs[k])
    for i, w in enumerate(winners):
        for k in winners[i + 1:] + [k for k in evs if k.endswith(f":uniform-{REF_MODEL_FORMAT}") and
                                    k.split(":")[0] not in {x.split(":")[0] for x in winners}]:
            pairs[f"{w} vs {k}"] = (evs[w], evs[k])
    result = {"budget_gib": budget, "configs": {}, "comparisons": {}}
    for k, e in evs.items():
        result["configs"][k] = {"mean_accuracy_pct": 100 * e.mean_accuracy(), "ppl_test_web": e.perplexity(),
                                "per_task_pct": {t: 100 * float(v.mean()) for t, v in e.task_correct.items()},
                                "ram_gib": e.peak_ram_bytes / GIB}
        v = result["configs"][k]
        print(f"{budget} GiB {k:24s} RAM {v['ram_gib']:.3f}  acc {v['mean_accuracy_pct']:.2f}  PPL {v['ppl_test_web']:.3f}  "
              + " ".join(f"{t[:4]} {x:.1f}" for t, x in v["per_task_pct"].items()), flush=True)
    for k, (a, b) in pairs.items():
        cmp_ = compare(a, b)
        result["comparisons"][k] = {"winner": cmp_.winner, "decided_by": cmp_.decided_by,
                                    "acc_diff_pts": 100 * cmp_.acc_diff,
                                    "acc_ci_pts": [100 * cmp_.acc_ci[0], 100 * cmp_.acc_ci[1]],
                                    "ppl_ratio": cmp_.ppl_ratio}
        r = result["comparisons"][k]
        print(f"  {k}: {r['winner']} by {r['decided_by']} | acc {r['acc_diff_pts']:+.2f} "
              f"[{r['acc_ci_pts'][0]:+.2f}, {r['acc_ci_pts'][1]:+.2f}] | PPL ratio {r['ppl_ratio']:.4f}", flush=True)
    out(f"budget-{budget}", "eval.json").write_text(json.dumps(result, indent=1))
    return 0


# --- bundle for the Pi ---------------------------------------------------------------------------------------

def cmd_bundle(budget: int) -> int:
    """Winners of a budget (+ each model's best reference) as GGUF files and a manifest for pi/panther_pi.py."""
    import hashlib
    import shutil
    ev = json.loads(out(f"budget-{budget}", "eval.json").read_text())
    bundle = DATA / "pi_bundle" / f"budget-{budget}"
    bundle.mkdir(parents=True, exist_ok=True)
    poc.make_agreement_text("Qwen2.5-1.5B-Instruct")
    shutil.copy(DATA / "splits/text/agreement.txt", bundle / "agreement.txt")
    files = {}
    for model in contests()[budget]["searched"]:
        res = json.loads(out(f"budget-{budget}", model, "contest.json").read_text())
        short = model.split("-")[1].lower()
        refs = [n for n in res["rows"] if not n.startswith("top")]
        best_ref = min(refs, key=lambda n: res["rows"][n]["kl"]) if refs else None
        for role, name in (("winner", res["winner"]), ("reference", best_ref)):
            if name is None:
                continue
            dst = bundle / f"qwen2.5-{short}-b{budget}-{role}.gguf"
            if name == "preset":
                src, a = Path(res["preset"]["path"]), None
                if not dst.exists():
                    shutil.copy(src, dst)
                import gguf
                tc: dict[str, int] = {}
                for t in gguf.GGUFReader(dst).tensors:
                    k = t.tensor_type.name.lower()
                    tc[k] = tc.get(k, 0) + 1
                ram = file_ram_gib(model, dst)
                weights = dst.stat().st_size
            else:
                a = res["candidates"][name]["assignment"]
                if not dst.exists():
                    build_file(model, a).rename(dst)
                tc = poc.type_counts(inv_of(model), a)
                ram = ram_gib(model, a)
                weights = estimate_ram(inv_of(model), a, CTX, OVERHEAD, KV_BPE).weights
            h = hashlib.sha256()
            with open(dst, "rb") as f:
                for c in iter(lambda: f.read(1 << 24), b""):
                    h.update(c)
            kv = inv_of(model).geometry.kv_cache_bytes(CTX, KV_BPE)
            files[dst.name] = {"model": model, "role": role, "contest_name": name, "sha256": h.hexdigest(),
                               "bytes": dst.stat().st_size, "predicted_ram_gib": ram,
                               "predicted": {"weights": weights, "kv_cache": kv, "overhead": int(OVERHEAD * GIB)},
                               "expected_type_counts": dict(sorted(tc.items())), "server_cpu_ppl_agreement": None,
                               "kl_selection": res["rows"][name]["kl"]}
            print(f"{dst.name}: predicted RAM {ram:.3f} GiB", flush=True)
    # Recommended: the winner with the best test-subset accuracy among models (smaller model if tied).
    wins = [k for k in ev["configs"] if k.endswith(":winner")]
    rec = max(wins, key=lambda k: ev["configs"][k]["mean_accuracy_pct"])
    for k in wins:
        c = ev["comparisons"].get(f"{rec} vs {k}") or ev["comparisons"].get(f"{k} vs {rec}")
        if c and c["winner"] is None and inv_of(f"Qwen2.5-{k.split(':')[0]}-Instruct").geometry.hidden < \
                inv_of(f"Qwen2.5-{rec.split(':')[0]}-Instruct").geometry.hidden:
            rec = k
    recommended = f"qwen2.5-{rec.split(':')[0].lower()}-b{budget}-winner.gguf"
    manifest = {"r_max_gib": float(budget), "context_tokens": CTX, "kv_cache_type": CFG["kv_cache_type"],
                "overhead_gib_assumed": OVERHEAD, "files": files, "agreement_text": "agreement.txt",
                "verdict": {"comparisons": ev["comparisons"], "configs": ev["configs"]},
                "recommended_file": recommended, "recommended_reason": f"best test-subset accuracy at {budget} GiB"}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"bundle ready: {bundle} (recommended {recommended})", flush=True)
    return 0


def main() -> int:
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "plan":
        return cmd_plan()
    if cmd in ("store", "refs", "error", "sens"):
        return {"store": cmd_store, "refs": poc.cmd_refs, "error": cmd_error, "sens": cmd_sens}[cmd](args[0])
    if cmd == "contest":
        return cmd_contest(int(args[0]), args[1])
    if cmd == "eval":
        return cmd_eval(int(args[0]))
    if cmd == "bundle":
        return cmd_bundle(int(args[0]))
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
