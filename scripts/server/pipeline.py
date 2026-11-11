"""The Panther pipeline (docs/STANDARD.md). Every command works for one budget and one model; run-budget.sh chains
them. Runs in the GPU container; every step checks the disk and skips work already done.

  pipeline.py store  BUDGET MODEL  # the quantized copies the model needs for this budget (missing ones only)
  pipeline.py refs   MODEL         # the 16-bit model's predictions on the sensitivity and selection texts
  pipeline.py error  BUDGET MODEL  # importance-weighted quantization error per matrix and type
  pipeline.py sens   BUDGET MODEL  # part costs: KL per part at the budget's starting type, then 2 rounds
  pipeline.py list   BUDGET MODEL  # the model's solution space: top genotypes inside the RAM window
  pipeline.py test   BUDGET MODEL  # each listed genotype built once; KL, benchmarks, GSM8K, perplexity on it
  pipeline.py pick   BUDGET MODEL  # the model's winner; `pick BUDGET` compares all models' winners
  pipeline.py bundle BUDGET MODEL  # the winner as a GGUF + manifest for the Pi
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from panther import evaluate
from panther.gguf_files import TensorStore, assemble, assignment_digest, verify
from panther.model import (GIB, REPO_ROOT, bits_per_weight, block_of, build_matrix_inventory, estimate_ram,
                           fixed_bytes, gene_bytes, gene_formats, load_geometry)

DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
LLAMA = Path(os.environ.get("PANTHER_LLAMA", "/opt/llama.cpp"))
THREADS = int(os.environ.get("PANTHER_THREADS", "16"))
SENS_CHUNKS = 12
SUBSET = {"mmlu": 1000, "hellaswag": 2000, "arc_challenge": None, "winogrande": None}  # test subset (D42)
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


# --- records of quality runs (written by panther.evaluate) ---------------------------------------------------

def records(model: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in (DATA / "results" / model).glob("*.json")]


def find(recs: list[dict], label: str, build: str = "cuda", **how) -> dict | None:
    hits = [r for r in recs if r.get("label") == label and r["how"]["build"] == build and r.get("ok")
            and all(r["how"].get(k) == v for k, v in how.items())]
    return sorted(hits, key=lambda r: r["eval_id"])[-1] if hits else None


def kl(model: str, gguf: Path, ref: str, label: str) -> None:
    if evaluate.main(["kl", "--model", model, "--gguf", str(gguf), "--build", "cuda", "--reference", ref,
                      "--label", label, "--threads", str(THREADS)]) != 0:
        raise SystemExit(f"{label}: KL run failed")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x]


def type_counts(inv, a: dict[str, str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for g in inv.genes:
        c[a[g.gene_id].lower()] = c.get(a[g.gene_id].lower(), 0) + len(g.tensors)
    c["f32"] = len(inv.fixed_tensors)
    return dict(sorted(c.items()))


def cmd_refs(model: str) -> int:
    """The 16-bit model's next-token distributions on the sensitivity (12 chunks) and selection texts."""
    ref_dir = DATA / "reference" / model
    ref_dir.mkdir(parents=True, exist_ok=True)
    bf16 = DATA / "store" / model / f"{model}-BF16.gguf"
    mpath = ref_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"model": model, "references": {}}
    for name, chunks in (("sensitivity", SENS_CHUNKS), ("selection", None)):
        kld = ref_dir / f"{name}.kld"
        if name in manifest["references"] and kld.exists():
            continue
        cmd = [str(LLAMA / "build-cuda/bin/llama-perplexity"), "-m", str(bf16), "-f",
               str(DATA / "splits" / "text" / f"{name}.txt"), "-c", "2048", "-b", "2048", "-ngl", "99",
               "-t", str(THREADS), "--kl-divergence-base", f"{kld}.part"] + (["--chunks", str(chunks)] if chunks else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        log = r.stdout + r.stderr
        (ref_dir / f"ref-{name}.log").write_text(log)
        part = Path(f"{kld}.part")
        m, n = re.search(r"Final estimate: PPL = ([0-9.]+)", log), re.search(r"over (\d+) chunks", log)
        if not part.exists() or part.stat().st_size == 0 or not m or not n:
            raise SystemExit(f"reference {name} failed; see {ref_dir}/ref-{name}.log")
        part.rename(kld)
        manifest["references"][name] = {"file": kld.name, "bytes": kld.stat().st_size, "context": 2048,
                                        "chunks": int(n.group(1)), "bf16_ppl": float(m.group(1))}
        mpath.write_text(json.dumps(manifest, indent=2))
        print(f"reference {name}: {n.group(1)} chunks, BF16 PPL {m.group(1)}", flush=True)
    return 0


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


BUDGET: int | None = None  # D51: set from the command line; every step then works for exactly this budget


def searched_budgets(model: str) -> list[int]:
    if BUDGET is not None:
        return [BUDGET] if model in contests()[BUDGET]["searched"] else []
    active = set(CFG.get("active_budgets", contests()))  # without a budget argument: the active budgets (D48)
    return sorted(b for b, c in contests().items() if model in c["searched"] and b in active)


class locked:
    """D51: an exclusive lock per (model, step), so runs for different budgets never write the same files at once;
    the second one waits, then finds the work done and skips it."""

    def __init__(self, name: str) -> None:
        self.path = DATA / "locks" / f"{name}.lock"

    def __enter__(self) -> "locked":
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


MIN_FREE_GB = 30


def need_disk(timeout_s: int = 3600) -> None:
    """D51: heavy steps wait while the data disk has less than MIN_FREE_GB free, then fail clearly."""
    import shutil
    import time
    t0 = time.monotonic()
    while shutil.disk_usage(DATA).free < MIN_FREE_GB * 1e9:
        if time.monotonic() - t0 > timeout_s:
            raise SystemExit(f"less than {MIN_FREE_GB} GB free on {DATA} for an hour; free space and rerun")
        print(f"waiting: less than {MIN_FREE_GB} GB free on {DATA}", flush=True)
        time.sleep(60)


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


_SRC_READERS: dict[Path, dict] = {}


def base_file(model: str) -> Path:
    """The sensitivity base, assembled once per model and kept for the whole sweep."""
    a = fill(model, base_format(model))
    path = out(model, f"base-{base_format(model)}.gguf")
    if not path.exists():
        part = path.with_suffix(".part")
        assemble(inv_of(model), a, store(model), part, base_format=base_format(model))
        part.rename(path)
    return path


def fast_probe(model: str, a: dict[str, str]) -> Path:
    """A probe file without rewriting the model: reflink-copy the base file (instant on XFS), point the changed
    tensors' table entries at their new data, and append only that data. Every tensor's bytes are identical to a
    normally assembled file; only the order inside the file differs (D50)."""
    import gguf
    base = base_file(model)
    base_a = fill(model, base_format(model))
    inv = inv_of(model)
    changed = {t.name: a[g.gene_id] for g in inv.genes for t in g.tensors if a[g.gene_id] != base_a[g.gene_id]}
    dst = out(model, "tmp", f"{assignment_digest(a)[:16]}.gguf")
    if dst.exists():
        return dst
    part = dst.with_suffix(".part")
    subprocess.run(["cp", "--reflink=always", str(base), str(part)], check=True)
    r = gguf.GGUFReader(part, "r+")
    tens = {t.name: t for t in r.tensors}
    t0 = r.tensors[0]
    data_start = int(t0.data_offset) - int(t0.field.parts[-1][0])
    al = r.fields.get("general.alignment")
    alignment = int(al.parts[-1][0]) if al else 32
    pos = part.stat().st_size
    plan = []
    for name, fmt in changed.items():
        src_path = DATA / "store" / model / f"{model}-{fmt}.gguf"
        if src_path not in _SRC_READERS:
            _SRC_READERS[src_path] = {t.name: t for t in gguf.GGUFReader(src_path).tensors}
        src = _SRC_READERS[src_path][name]
        pos += (-(pos - data_start)) % alignment
        plan.append((name, src, pos))
        tens[name].field.parts[-2][0] = int(src.tensor_type)
        tens[name].field.parts[-1][0] = pos - data_start
        pos += int(src.n_bytes)
    r.data.flush()
    del r
    with open(part, "ab") as f:
        for name, src, at in plan:
            f.write(b"\0" * (at - f.tell()))
            f.write(src.data.tobytes())
    part.rename(dst)
    if not verify(inv, a, dst).ok:
        raise SystemExit(f"fast probe read-back failed for {dst.name}")
    return dst


def reflink_ok(model: str) -> bool:
    """D50: reflink-copied probe files are used only if, for a test block, every tensor (type and bytes) matches a
    normally assembled file. Checked once per model; the answer is stored."""
    import gguf
    flag = out(model, f"reflink-check-{base_format(model)}.json")
    if flag.exists():
        return json.loads(flag.read_text())["identical"]
    b0 = sorted(blocks(model))[len(blocks(model)) // 2]
    a = block_assignment(model, b0, probe_format(model))
    try:
        fast = fast_probe(model, a)
        slow = out(model, "tmp", "reflink-check-slow.gguf")
        assemble(inv_of(model), a, store(model), slow, base_format=base_format(model))
        rf = {t.name: t for t in gguf.GGUFReader(fast).tensors}
        rs = {t.name: t for t in gguf.GGUFReader(slow).tensors}
        same = rf.keys() == rs.keys() and all(rf[n].tensor_type == rs[n].tensor_type and
                                              np.array_equal(rf[n].data, rs[n].data) for n in rs)
        fast.unlink()
        slow.unlink()
    except Exception as e:  # noqa: BLE001  (any failure means: build files the normal way)
        print(f"reflink check failed: {e}", flush=True)
        same = False
    flag.write_text(json.dumps({"block": b0, "identical": bool(same)}))
    print(f"{model}: reflink-copied probe files {'match' if same else 'DO NOT match; using normal assembly'}",
          flush=True)
    return bool(same)


def blocks(model: str) -> dict[str, list[str]]:
    b: dict[str, list[str]] = {}
    for g in inv_of(model).genes:
        b.setdefault(block_of(g), []).append(g.gene_id)
    return b


# --- store ---------------------------------------------------------------------------------------------------

def cmd_store(model: str) -> int:
    fmts = menu(model) if searched_budgets(model) else [REF_MODEL_FORMAT]
    with locked(f"{model}-store"):
        missing = [f for f in fmts if not (DATA / "store" / model / f"{model}-{f}.gguf").exists()]
        print(f"{model}: types {fmts}; to build {missing}", flush=True)
        if not missing and (DATA / "store" / model / f"{model}-BF16.gguf").exists():
            return 0
        need_disk()
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
    with locked(f"{model}-error"):
        return _cmd_error(model)


def _cmd_error(model: str) -> int:
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
    a = block_assignment(model, None if block == "base" else block, fmt)
    path = fast_probe(model, a) if reflink_ok(model) else build_file(model, a)
    wait_for_gpu_memory(path.stat().st_size)
    try:
        kl(model, path, "sensitivity", sens_label(model, block, fmt))
    finally:
        path.unlink(missing_ok=True)


def measured_table(model: str) -> tuple[float | None, dict[str, dict[str, float]]]:
    """Base KL and measured dKL per (block, format), from the KL records."""
    base = base_format(model)
    pre = f"rw:sens:{base}:"
    kls = {}
    for r in records(model):
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
    from panther.solver import Problem
    from panther.solver import solve_top_k
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
    with locked(f"{model}-sens-{base_format(model)}"):
        return _cmd_sens(model)


def _cmd_sens(model: str) -> int:
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
    out(model, f"sens-{base}.json").write_text(json.dumps({"base": base, "probe": probe, "base_kl": base_kl,
                                                   "table": table, "rounds": stats}, indent=1))
    print(f"{model}: sensitivity done, {sum(s['runs'] for s in stats)} KL runs, base KL {base_kl:.5f}", flush=True)
    return 0


# --- helpers -------------------------------------------------------------------------------------------------

def file_ram_gib(model: str, path: Path) -> float:
    import gguf
    w = sum(int(t.n_bytes) for t in gguf.GGUFReader(path).tensors)
    return (w + inv_of(model).geometry.kv_cache_bytes(CTX, KV_BPE) + OVERHEAD * GIB) / GIB


# --- solution list, test all, pick (D49) ---------------------------------------------------------------------

WINDOW = (-0.01, 0.005)   # D49: in the solution space iff RAM is within -1% / +0.5% of the budget
LIST_K = 6                # solver genotypes per model
LIST_DISTANCE = 0.05      # each differs from every other in at least 5% of its matrices
TEST_CONCURRENT = 8


def in_window(model: str, a: dict[str, str], budget: float) -> bool:
    r = ram_gib(model, a)
    return budget * (1 + WINDOW[0]) <= r <= budget * (1 + WINDOW[1])


def padded_uniform(model: str, budget: float) -> tuple[str, dict[str, str]] | None:
    """In-space baseline: the largest uniform format that fits, whole layers raised one format (alternating from
    both ends) until the RAM is inside the window."""
    m = sorted(menu(model), key=bits_per_weight)
    inv = inv_of(model)
    for u in reversed(m):
        a = fill(model, u)
        if ram_gib(model, a) > budget * (1 + WINDOW[1]):
            continue
        nxt = m[min(m.index(u) + 1, len(m) - 1)]
        order, lo_, hi_ = [], 0, inv.geometry.n_layers - 1
        while lo_ <= hi_:
            order += [lo_] + ([hi_] if hi_ != lo_ else [])
            lo_, hi_ = lo_ + 1, hi_ - 1
        for L in order:
            if in_window(model, a, budget):
                break
            trial = dict(a)
            for g in inv.genes:
                if g.layer == L and nxt in allowed(model)[g.gene_id]:
                    trial[g.gene_id] = nxt
            if ram_gib(model, trial) <= budget * (1 + WINDOW[1]):
                a = trial
        return (u, a) if in_window(model, a, budget) else None
    return None


def solution_list(budget: int, pen_fn=None, models: list[str] | None = None) -> dict[str, dict]:
    from panther.solver import Problem
    from panther.solver import solve_top_k
    items = {}
    for model in models or contests()[budget]["searched"]:
        inv = inv_of(model)
        short = model.split("-")[1]
        prob = Problem(inv, allowed(model), (pen_fn or penalties)(model), budget * (1 + WINDOW[1]), CTX, OVERHEAD,
                       kv_bytes_per_element=KV_BPE)
        lo_bytes = prob.weight_budget_bytes - int((WINDOW[1] - WINDOW[0]) * budget * GIB)
        sols = solve_top_k(prob, k=LIST_K, time_limit_s=90, min_bytes=lo_bytes,
                           min_distance=max(1, int(LIST_DISTANCE * len(inv.genes))))
        for i, sol in enumerate(sols):
            if not in_window(model, sol.assignment, budget):
                raise SystemExit(f"{short} sol{i + 1} outside the window ({ram_gib(model, sol.assignment):.3f} GiB)")
            items[f"{short}:sol{i + 1}"] = {"model": model, "assignment": sol.assignment,
                                            "predicted_penalty": sol.penalty}
        pu = padded_uniform(model, budget)
        if pu:
            items[f"{short}:uniform-{pu[0]}-padded"] = {"model": model, "assignment": pu[1]}
    for it in items.values():
        it["ram_gib"] = ram_gib(it["model"], it["assignment"])
    return items


def list_path(budget: int, model: str) -> Path:
    return out(f"budget-{budget}", f"list-{model}.json")


def load_list(budget: int) -> dict[str, dict]:
    """The budget's whole solution space: the union of its models' lists."""
    items = {}
    for model in contests()[budget]["searched"]:
        items.update(json.loads(list_path(budget, model).read_text()))
    return items


def cmd_list(budget: int, model: str) -> int:
    """One model's solution space for a budget, written once and never extended (D49)."""
    path = list_path(budget, model)
    if path.exists():
        print(f"list exists: {path}")
        return 0
    items = solution_list(budget, models=[model])
    for name, it in items.items():
        print(f"{budget} GiB {name:28s} RAM {it['ram_gib']:.3f} GiB", flush=True)
    path.write_text(json.dumps(items, indent=1))
    return 0


def test_label(budget: int, name: str, a: dict[str, str] | None) -> str:
    return f"rw:b{budget}:{name}" + (f":{assignment_digest(a)[:12]}" if a else "")


def run_all_tests(model: str, path: Path, label: str, kl: bool) -> None:
    """Every quality test on one file: KL (selection text), the four test-subset tasks, perplexity, GSM8K."""
    if kl and not find(records(model), label, reference="selection"):
        kl(model, path, "selection", label)
    for t in SUBSET:
        if not find(records(model), label, task=t, split="poc_test"):
            if evaluate.main(["task", "--model", model, "--gguf", str(path), "--build", "cuda", "--task", t,
                                   "--split", "poc_test", "--label", label, "--threads", str(THREADS)]) != 0:
                raise SystemExit(f"{label} {t} failed")
    if not find(records(model), label, text="test_web"):
        if evaluate.main(["ppl", "--model", model, "--gguf", str(path), "--build", "cuda", "--text",
                               "test_web", "--label", label, "--threads", str(THREADS)]) != 0:
            raise SystemExit(f"{label} ppl failed")
    gsm8k_run(model, path, label)


def tests_done(model: str, label: str, kl: bool) -> bool:
    recs = records(model)
    return ((not kl or find(recs, label, reference="selection") is not None)
            and all(find(recs, label, task=t, split="poc_test") for t in SUBSET)
            and find(recs, label, text="test_web") is not None
            and out(model, "gsm8k", f"{label.replace(':', '_')}.json").exists())


def _test_one(job: tuple[int, str, dict]) -> str:
    """One genotype: assembled once, all tests run on that file, then the file is deleted."""
    budget, name, it = job
    model, a = it["model"], it["assignment"]
    label = test_label(budget, name, a)
    if tests_done(model, label, kl=True):
        return name
    need_disk()
    path = build_file(model, a)
    wait_for_gpu_memory(2 * path.stat().st_size)
    try:
        run_all_tests(model, path, label, kl=True)
    finally:
        path.unlink(missing_ok=True)
    return name


def _test_ref(job: tuple[int, str]) -> str:
    budget, model = job
    name = f"{model.split('-')[1]}:uniform-{REF_MODEL_FORMAT}"
    path = DATA / "store" / model / f"{model}-{REF_MODEL_FORMAT}.gguf"
    label = test_label(budget, name, None)
    if not tests_done(model, label, kl=False):
        wait_for_gpu_memory(2 * path.stat().st_size)
        run_all_tests(model, path, label, kl=False)  # KL against a different model's original is meaningless
    return name


def cmd_test(budget: int, model: str) -> int:
    """One model's listed genotypes (or, for a reference model, its uniform Q6_K file), all at once on the GPU."""
    if model in contests()[budget]["reference"]:
        print(f"tested {_test_ref((budget, model))}", flush=True)
        return 0
    items = json.loads(list_path(budget, model).read_text())
    print(f"{model}: {len(items)} genotypes, all at once", flush=True)
    with ProcessPoolExecutor(len(items)) as ex:
        for n in ex.map(_test_one, [(budget, n, it) for n, it in items.items()]):
            print(f"  tested {n}", flush=True)
    return 0


def gsm8k_run(model: str, path: Path, label: str) -> dict:
    rec_path = out(model, "gsm8k", f"{label.replace(':', '_')}.json")
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    items = jsonl(DATA / "splits/llama_tasks/poc_test/gsm8k.jsonl")
    fewshot = jsonl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    with evaluate.LlamaServer(LLAMA, path, "cuda", parallel=8, ctx_per_slot=2048, threads=THREADS,
                     log=out(model, "servers", f"{path.stem}.log")) as srv:
        comps = srv.map(lambda it: srv.complete(evaluate.prompt(it["question"], fewshot), evaluate.MAX_TOKENS,
                                                evaluate.STOP), items)
    correct = [int(evaluate.exact_match(evaluate.strict_match(c), it["answer"])) for it, c in zip(items, comps)]
    rec = {"ids": [it["id"] for it in items], "correct": correct}
    rec_path.write_text(json.dumps(rec))
    return rec


def evaluation(model: str, label: str, ram: float, name: str):
    from panther.stats import Evaluation
    recs = records(model)
    correct = {t: np.array(find(recs, label, task=t, split="poc_test")["per_item"]["correct"], dtype=np.int8)
               for t in SUBSET}
    g = json.loads(out(model, "gsm8k", f"{label.replace(':', '_')}.json").read_text())
    correct["gsm8k"] = np.array(g["correct"], dtype=np.int8)
    p = find(recs, label, text="test_web")["per_chunk"]
    nll = np.array(p["mean_nll"]) * p["tokens"]
    return Evaluation(name, correct, nll, np.full(len(nll), p["tokens"]), int(ram * GIB), 0.0)


def cmd_pick(budget: int) -> int:
    """Per model: best mean test-subset accuracy (ties: lower KL). Then winner vs its padded-uniform baseline, the
    models' winners against each other, and against the cross-model reference."""
    from panther.stats import compare
    items = load_list(budget)
    rows, evs = {}, {}
    for name, it in items.items():
        label = test_label(budget, name, it["assignment"])
        evs[name] = evaluation(it["model"], label, it["ram_gib"], name)
        k = find(records(it["model"]), label, reference="selection")["metrics"]
        rows[name] = {"model": it["model"], "ram_gib": it["ram_gib"], "kl": k["kl_mean"], "top1_pct": k["top1_pct"]}
    for model in contests()[budget]["reference"]:
        name = f"{model.split('-')[1]}:uniform-{REF_MODEL_FORMAT}"
        ram = file_ram_gib(model, DATA / "store" / model / f"{model}-{REF_MODEL_FORMAT}.gguf")
        evs[name] = evaluation(model, test_label(budget, name, None), ram, name)
        rows[name] = {"model": model, "ram_gib": ram, "kl": None, "reference": True}
    for n, r in rows.items():
        r.update({"mean_accuracy_pct": 100 * evs[n].mean_accuracy(), "ppl": evs[n].perplexity(),
                  "per_task_pct": {t: 100 * float(v.mean()) for t, v in evs[n].task_correct.items()}})
    winners = {}
    for model in contests()[budget]["searched"]:
        cands = [n for n in items if items[n]["model"] == model]
        winners[model] = max(cands, key=lambda n: (round(rows[n]["mean_accuracy_pct"], 2), -rows[n]["kl"]))
    pairs = {}
    for model, w in winners.items():
        base = next((n for n in items if items[n]["model"] == model and n.endswith("-padded")), None)
        if base and base != w:
            pairs[f"{w} vs {base}"] = (w, base)
    ws = list(winners.values())
    for i, w in enumerate(ws):
        for v in ws[i + 1:] + [n for n in rows if rows[n].get("reference")]:
            pairs[f"{w} vs {v}"] = (w, v)
    result = {"budget_gib": budget, "window": WINDOW, "rows": rows, "winners": winners, "comparisons": {}}
    for k, (x, y) in pairs.items():
        c = compare(evs[x], evs[y])
        result["comparisons"][k] = {"winner": c.winner, "decided_by": c.decided_by, "acc_diff_pts": 100 * c.acc_diff,
                                    "acc_ci_pts": [100 * c.acc_ci[0], 100 * c.acc_ci[1]], "ppl_ratio": c.ppl_ratio}
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean_accuracy_pct"]):
        kl = f"{r['kl']:.4f}" if r["kl"] is not None else "   -  "
        print(f"{budget} GiB {n:28s} RAM {r['ram_gib']:.3f} | KL {kl} | acc {r['mean_accuracy_pct']:.2f} | "
              f"PPL {r['ppl']:.3f} | " + " ".join(f"{t[:4]} {v:.1f}" for t, v in r["per_task_pct"].items()), flush=True)
    for k, c in result["comparisons"].items():
        print(f"  {k}: {c['winner']} by {c['decided_by']} | acc {c['acc_diff_pts']:+.2f} "
              f"[{c['acc_ci_pts'][0]:+.2f}, {c['acc_ci_pts'][1]:+.2f}] | PPL ratio {c['ppl_ratio']:.4f}", flush=True)
    out(f"budget-{budget}", "pick.json").write_text(json.dumps(result, indent=1))
    return 0


def cmd_pick_model(budget: int, model: str) -> int:
    """One model's winner as soon as its own tests are done: best mean accuracy (ties: lower KL)."""
    items = json.loads(list_path(budget, model).read_text())
    rows = {}
    for name, it in items.items():
        label = test_label(budget, name, it["assignment"])
        ev = evaluation(model, label, it["ram_gib"], name)
        k = find(records(model), label, reference="selection")["metrics"]
        rows[name] = {"ram_gib": it["ram_gib"], "kl": k["kl_mean"], "mean_accuracy_pct": 100 * ev.mean_accuracy(),
                      "ppl": ev.perplexity(), "per_task_pct": {t: 100 * float(v.mean()) for t, v in ev.task_correct.items()}}
    win = max(rows, key=lambda n: (round(rows[n]["mean_accuracy_pct"], 2), -rows[n]["kl"]))
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean_accuracy_pct"]):
        print(f"{budget} GiB {n:28s} RAM {r['ram_gib']:.3f} | KL {r['kl']:.4f} | acc {r['mean_accuracy_pct']:.2f} | "
              f"PPL {r['ppl']:.3f} | " + " ".join(f"{t[:4]} {v:.1f}" for t, v in r["per_task_pct"].items())
              + ("   <- winner" if n == win else ""), flush=True)
    out(f"budget-{budget}", f"winner-{model}.json").write_text(json.dumps({"winner": win, "rows": rows}, indent=1))
    return 0


# --- bundle for the Pi ---------------------------------------------------------------------------------------

def cmd_bundle_model(budget: int, model: str) -> int:
    """One model's winner as a Pi bundle (pi_bundle/budget-B-<size>), ready as soon as that model is picked."""
    import hashlib
    import shutil
    name = json.loads(out(f"budget-{budget}", f"winner-{model}.json").read_text())["winner"]
    a = json.loads(list_path(budget, model).read_text())[name]["assignment"]
    short = model.split("-")[1].lower()
    bundle = DATA / "pi_bundle" / f"budget-{budget}-{short}"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "splits/text/agreement.txt", bundle / "agreement.txt")
    dst = bundle / f"qwen2.5-{short}-b{budget}-winner.gguf"
    if not dst.exists():
        build_file(model, a).rename(dst)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    r = estimate_ram(inv_of(model), a, CTX, OVERHEAD, KV_BPE)
    files = {dst.name: {"model": model, "role": "winner", "list_name": name, "sha256": h.hexdigest(),
                        "bytes": dst.stat().st_size, "predicted_ram_gib": r.total_gib,
                        "predicted": {"weights": r.weights, "kv_cache": r.kv_cache, "overhead": r.overhead},
                        "expected_type_counts": type_counts(inv_of(model), a), "server_cpu_ppl_agreement": None}}
    manifest = {"r_max_gib": float(budget) * (1 + WINDOW[1]), "context_tokens": CTX,
                "kv_cache_type": CFG["kv_cache_type"], "overhead_gib_assumed": OVERHEAD, "files": files,
                "agreement_text": "agreement.txt", "verdict": {"comparisons": {}}, "recommended_file": dst.name,
                "recommended_reason": f"{model} winner at {budget} GiB"}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"bundle ready: {bundle} ({dst.name}, predicted RAM {r.total_gib:.3f} GiB)", flush=True)
    return 0



def main() -> int:
    cmd, args = sys.argv[1], sys.argv[2:]
    global BUDGET
    if cmd == "refs":
        with locked(f"{args[-1]}-refs"):
            return cmd_refs(args[-1])
    if cmd in ("store", "error", "sens"):  # "store B MODEL" (D51), or "store MODEL" for the active budgets
        if len(args) == 2:
            BUDGET = int(args[0])
        return {"store": cmd_store, "error": cmd_error, "sens": cmd_sens}[cmd](args[-1])
    if cmd in ("list", "test", "pick", "bundle"):
        BUDGET = int(args[0])
    if cmd in ("list", "test"):
        return {"list": cmd_list, "test": cmd_test}[cmd](int(args[0]), args[1])
    if cmd in ("pick", "bundle") and len(args) == 2:
        return {"pick": cmd_pick_model, "bundle": cmd_bundle_model}[cmd](int(args[0]), args[1])
    if cmd == "pick":
        return cmd_pick(int(args[0]))
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
