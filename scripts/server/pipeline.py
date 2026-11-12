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
  pipeline.py preset BUDGET MODEL  # the best existing quantization (llama.cpp's largest fitting preset), tested
  pipeline.py upload BUDGET MODEL  # the bundle to its private Hugging Face repo, for the Pi to download
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
BENCH_CHUNKS = 16  # KL text for the GA: selection-split benchmark questions with their answers (bench_text)
SUBSET = {"mmlu": 1000, "hellaswag": 2000, "arc_challenge": None, "winogrande": None}  # test subset
RW = DATA / "rewrite"
CFG = yaml.safe_load((REPO_ROOT / "configs" / "search.yaml").read_text())
CTX = CFG["context_tokens"]
OVERHEAD = CFG["overhead_gib"]
KV_BPE = {"Q8_0": 34 / 32, "F16": 2.0}[CFG["kv_cache_type"]]
MENU_ALL = CFG["formats"]
CONCURRENT = int(os.environ.get("PANTHER_KL_CONCURRENT", "10"))
LANE_SCRATCH_GB = 25  # part-cost files one lane may hold on disk at once
# The search starts from the budget's best llama.cpp preset (its layout, mapped to the menu), probes each part one
# type up and one down from there, and lists the padded preset layout as a control. Where no preset fits, it starts
# from one uniform type. PANTHER_ANCHOR=none runs the first method (uniform start only), whose files have no suffix.
ANCHOR = os.environ.get("PANTHER_ANCHOR", "preset") != "none"
SUFFIX = "-anchored" if ANCHOR else ""


GPU_BOOKABLE = 88 * 2**30  # of the GPU's 96 GiB; the rest is margin


class gpu_booking:
    """Shared booking of GPU memory across every process and lane: a run books its file size + working space in
    one ledger (under a file lock) and waits until the total fits in GPU_BOOKABLE; it releases on exit. Checking
    free memory instead raced: ten runs saw room at the same moment and all loaded. Bookings expire after `hold_s`
    so a killed job cannot block the GPU for long."""

    def __init__(self, need_bytes: int, hold_s: int) -> None:
        self.need, self.hold = int(need_bytes), hold_s
        self.key = f"{os.uname().nodename}-{os.getpid()}-{id(self)}"
        self.ledger = DATA / "locks" / "gpu-bookings.json"

    def _edit(self, fn):
        import fcntl
        import time
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger.with_suffix(".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:  # a job killed mid-write once left this empty, and every run after it failed; never again
                book = json.loads(self.ledger.read_text())
            except (OSError, ValueError):
                book = {}
            now = time.time()
            book = {k: v for k, v in book.items() if v["until"] > now}
            ok = fn(book, now)
            tmp = self.ledger.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(book))
            os.replace(tmp, self.ledger)  # atomic: a reader sees the old file or the new one, never half of one
            return ok

    def __enter__(self) -> "gpu_booking":
        import random
        import time

        def take(book, now):
            if sum(v["bytes"] for v in book.values()) + self.need > GPU_BOOKABLE and book:
                return False
            book[self.key] = {"bytes": self.need, "until": now + self.hold}
            return True
        while not self._edit(take):
            time.sleep(random.uniform(3, 8))
        return self

    def __exit__(self, *_: object) -> None:
        self._edit(lambda book, now: book.pop(self.key, None) or True)


CALIB = DATA / "splits" / "calibration.txt"
REF_MODEL_FORMAT = "Q6_K"  # cross-model references: uniform Q6_K, accuracy only


# --- records of quality runs (written by panther.evaluate) ---------------------------------------------------

def records(model: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in (DATA / "results" / model).glob("*.json")]


def find(recs: list[dict], label: str, build: str = "cuda", **how) -> dict | None:
    hits = [r for r in recs if r.get("label") == label and r["how"]["build"] == build and r.get("ok")
            and all(r["how"].get(k) == v for k, v in how.items())]
    return sorted(hits, key=lambda r: r["eval_id"])[-1] if hits else None


def kl(model: str, gguf: Path, ref: str, label: str) -> None:
    if not evaluate.run("kl", model, gguf, label, THREADS, reference=ref):
        raise SystemExit(f"{label}: KL run failed")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().split("\n") if x]


def type_counts(inv, a: dict[str, str]) -> dict[str, int]:
    c: dict[str, int] = {}
    for g in inv.genes:
        c[a[g.gene_id].lower()] = c.get(a[g.gene_id].lower(), 0) + len(g.tensors)
    c["f32"] = len(inv.fixed_tensors)
    return dict(sorted(c.items()))


def bench_text() -> Path:
    """Benchmark questions with their correct answers, from the selection split (never the test questions), the
    five benchmarks interleaved: the GA's fitness is KL on this text, so closeness to the original is judged where
    the benchmarks probe. Deterministic; written once."""
    import random
    path = DATA / "splits" / "text" / "bench.txt"
    if path.exists():
        return path
    per: dict[str, list[str]] = {}
    for t in ("mmlu", "arc_challenge", "hellaswag", "winogrande", "gsm8k"):
        items = []
        for it in jsonl(DATA / "splits" / "tasks" / "selection" / f"{t}.jsonl"):
            if t in ("mmlu", "arc_challenge"):
                labels = it.get("choice_labels") or list("ABCDE")
                items.append(f"Question: {it['question']}\n" + "".join(f"{l}. {c}\n" for l, c in zip(labels, it["choices"]))
                             + f"Answer: {labels[it['label']]}. {it['choices'][it['label']]}")
            elif t == "hellaswag":
                items.append(f"{it['ctx']} {it['endings'][it['label']]}")
            elif t == "winogrande":
                items.append(it["sentence"].replace("_", it["option1"] if it["label"] == 0 else it["option2"]))
            else:
                items.append(f"Question: {it['question']}\nAnswer: {it['answer']}")
        random.Random(0).shuffle(items)
        per[t] = items
    out_, n, i = [], 0, 0
    while n < BENCH_CHUNKS * 2048 * 4 * 1.5 and i < max(len(v) for v in per.values()):  # ~4 characters per token
        for t, v in per.items():
            if i < len(v):
                out_.append(v[i])
                n += len(v[i]) + 2
        i += 1
    path.write_text("\n\n".join(out_) + "\n")
    return path


def cmd_refs(model: str) -> int:
    """The 16-bit model's next-token distributions on the sensitivity (12 chunks), selection and benchmark-question
    (16 chunks) texts."""
    bench_text()
    ref_dir = DATA / "reference" / model
    ref_dir.mkdir(parents=True, exist_ok=True)
    bf16 = DATA / "store" / model / f"{model}-BF16.gguf"
    mpath = ref_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"model": model, "references": {}}
    for name, chunks in (("sensitivity", SENS_CHUNKS), ("selection", None), ("bench", BENCH_CHUNKS)):
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


BUDGET: int | None = None  # set from the command line; every step then works for exactly this budget


def searched_budgets(model: str) -> list[int]:
    if BUDGET is not None:
        return [BUDGET] if model in contests()[BUDGET]["searched"] else []
    raise SystemExit("every step needs the budget: pipeline.py <step> BUDGET MODEL")


def note(msg: str) -> None:
    """One plain progress line per event, in data/logs/progress-b<BUDGET>.log (IST), and on stdout."""
    import datetime as dt
    ist = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime("%H:%M")
    line = f"{ist} {msg}"
    print(line, flush=True)
    path = DATA / "logs" / f"progress-b{BUDGET}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def short_name(model: str) -> str:
    return model.split("-")[1]


class locked:
    """an exclusive lock per (model, step), so runs for different budgets never write the same files at once;
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
    """heavy steps wait while the data disk has less than MIN_FREE_GB free, then fail clearly."""
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
    """formats between the lowest budget average - 1.5 and the highest + 2.5 bits per weight."""
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
    """the menu format closest to the lowest searched budget's average bits per weight."""
    target = avg_bpw(model, searched_budgets(model)[0])
    return min(menu(model), key=lambda f: abs(bits_per_weight(f) - target))


def probe_format(model: str) -> str:
    """the highest menu format at most `sensitivity_probe_up_bpw` above the base (else the one below)."""
    m = sorted(menu(model), key=bits_per_weight)
    b = bits_per_weight(base_format(model))
    up = [f for f in m if b < bits_per_weight(f) <= b + CFG["sensitivity_probe_up_bpw"]]
    return up[-1] if up else m[m.index(base_format(model)) - 1]


@lru_cache(maxsize=None)
def anchor(model: str) -> tuple[str, dict[str, str]] | None:
    """The budget's best fitting llama.cpp preset (from the preset step) as a genotype: each matrix takes the
    preset's type for it, or the allowed menu type nearest in bits per weight (the higher on a tie). None if no
    preset fits the budget (or the preset step has not run)."""
    import gguf
    pj = out(f"budget-{BUDGET}", f"preset-{model}.json")
    if not pj.exists():
        raise SystemExit(f"{model} at {BUDGET} GiB: run the preset step first (the search starts from its result)")
    ftype = json.loads(pj.read_text()).get("preset")
    if not ftype:
        return None
    path = DATA / "store" / model / f"preset-{ftype}.gguf"
    types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(path).tensors}
    a = {}
    for g in inv_of(model).genes:
        want, fs = types[g.tensors[0].name], allowed(model)[g.gene_id]
        a[g.gene_id] = want if want in fs else min(
            fs, key=lambda f: (abs(bits_per_weight(f) - bits_per_weight(want)), -bits_per_weight(f)))
    return ftype, a


def method_suffix(model: str) -> str:
    """Where no preset fits, the preset-start search is the uniform-start one, so its files are shared."""
    return SUFFIX if ANCHOR and anchor(model) else ""


def start_assignment(model: str) -> dict[str, str]:
    """Where the part costs are measured from: the preset layout (anchored) or one uniform type."""
    return dict(anchor(model)[1]) if ANCHOR and anchor(model) else fill(model, base_format(model))


def start_tag(model: str) -> str:
    return f"anchor{BUDGET}-{anchor(model)[0]}" if ANCHOR and anchor(model) else base_format(model)


def anchor_probes(model: str) -> tuple[str, str]:
    """One menu type above and one below the type holding most of the preset layout's weights."""
    a = anchor(model)[1]
    w: dict[str, int] = {}
    for g in inv_of(model).genes:
        w[a[g.gene_id]] = w.get(a[g.gene_id], 0) + g.n_weights
    m = sorted(menu(model), key=bits_per_weight)
    i = m.index(max(w, key=w.get))
    return m[min(i + 1, len(m) - 1)], m[max(i - 1, 0)]


def store(model: str) -> TensorStore:
    return TensorStore({f: DATA / "store" / model / f"{model}-{f}.gguf" for f in menu(model)})


def build_file(model: str, a: dict[str, str]) -> Path:
    tmp = out(model, "tmp", "x").parent
    path = tmp / f"{assignment_digest(a)[:16]}.gguf"
    if not path.exists():
        need_disk()  # every worker writes a whole-model file; several at once once filled the disk
        part = path.with_suffix(f".{os.getpid()}.part")
        assemble(inv_of(model), a, store(model), part, base_format=base_format(model))
        part.rename(path)
    if not verify(inv_of(model), a, path).ok:
        raise SystemExit(f"read-back failed for {path.name}")
    return path


_SRC_READERS: dict[Path, dict] = {}


def base_file(model: str) -> Path:
    """The sensitivity base, assembled once per model and kept for the whole sweep."""
    a = start_assignment(model)
    path = out(model, f"base-{start_tag(model)}.gguf")
    if not path.exists():
        part = path.with_suffix(".part")
        assemble(inv_of(model), a, store(model), part, base_format=base_format(model))
        part.rename(path)
    return path


def fast_probe(model: str, a: dict[str, str]) -> Path:
    """A probe file without rewriting the model in Python: the base file's header, then every tensor in the same
    order, copied by the kernel (copy_file_range: reflinked on XFS where block alignment allows). Unchanged runs
    come from the base file, changed tensors from their type's quantized copy; then the header's type and offset
    entries are updated. Tensors stay in order and back to back, as llama.cpp requires."""
    import gguf
    need_disk()  # every worker writes a whole-model file; several at once once filled the disk
    base = base_file(model)
    base_a = start_assignment(model)
    inv = inv_of(model)
    new_type = {t.name: a[g.gene_id] for g in inv.genes for t in g.tensors if a[g.gene_id] != base_a[g.gene_id]}
    dst = out(model, "tmp", f"{assignment_digest(a)[:16]}.gguf")
    if dst.exists():
        return dst
    rb = gguf.GGUFReader(base)
    t0 = rb.tensors[0]
    data_start = int(t0.data_offset) - int(t0.field.parts[-1][0])
    al = rb.fields.get("general.alignment")
    alignment = int(al.parts[-1][0]) if al else 32
    srcs = {}
    for fmt in set(new_type.values()):
        path = DATA / "store" / model / f"{model}-{fmt}.gguf"
        srcs[fmt] = (path, {t.name: t for t in gguf.GGUFReader(path).tensors})
    part = dst.with_suffix(f".{os.getpid()}.part")
    plan, pos = [], 0  # (name, type, new relative offset)
    with open(base, "rb") as fb, open(part, "wb") as fo:
        def copy(fin, src_off: int, n: int) -> None:
            if not hasattr(os, "copy_file_range"):  # not Linux (tests on the Mac): plain copy
                fin.seek(src_off)
                fo.write(fin.read(n))
                return
            while n > 0:
                k = os.copy_file_range(fin.fileno(), fo.fileno(), n, src_off)
                if k <= 0:
                    raise OSError("copy_file_range copied nothing")
                src_off, n = src_off + k, n - k
        copy(fb, 0, data_start)  # header (same length: only type and offset values change)
        run = None  # [base start, length] of the current run of unchanged tensors
        for t in rb.tensors:
            pad = (-pos) % alignment
            if t.name in new_type:
                if run:
                    copy(fb, *run)
                    run = None
                fo.write(b"\0" * pad)
                pos += pad
                fmt = new_type[t.name]
                s_path, s_t = srcs[fmt]
                src = s_t[t.name]
                with open(s_path, "rb") as fs:
                    copy(fs, int(src.data_offset), int(src.n_bytes))
                plan.append((t.name, int(src.tensor_type), pos))
                pos += int(src.n_bytes)
            else:
                old = int(t.field.parts[-1][0])
                if run is None or (data_start + old) != run[0] + run[1] + pad:
                    if run:
                        copy(fb, *run)
                    fo.write(b"\0" * pad)
                    pos += pad
                    run = [data_start + old, int(t.n_bytes)]
                else:
                    run[1] += pad + int(t.n_bytes)
                    pos += pad
                plan.append((t.name, int(t.tensor_type), pos))
                pos += int(t.n_bytes)
        if run:
            copy(fb, *run)
    # Patch each tensor's type and offset in the header, at the byte positions they have in the base file (the
    # header is copied verbatim, so the positions are the same).
    base_addr = rb.data.ctypes.data
    where = {t.name: (t.field.parts[-2].ctypes.data - base_addr, t.field.parts[-1].ctypes.data - base_addr)
             for t in rb.tensors}
    with open(part, "r+b") as f:
        for name, ttype, off in plan:
            f.seek(where[name][0])
            f.write(np.uint32(ttype).tobytes())
            f.seek(where[name][1])
            f.write(np.uint64(off).tobytes())
    expect = 0  # llama.cpp's rule: each tensor starts where the previous one ends, rounded up to the alignment
    for t in gguf.GGUFReader(part).tensors:
        expect += (-expect) % alignment
        if int(t.field.parts[-1][0]) != expect:
            raise SystemExit(f"probe file layout: {t.name} at {int(t.field.parts[-1][0])}, llama.cpp expects {expect}")
        expect += int(t.n_bytes)
    part.rename(dst)
    if not verify(inv, a, dst).ok:
        raise SystemExit(f"probe file read-back failed for {dst.name}")
    return dst


def reflink_ok(model: str) -> bool:
    """Probe files built by copy_file_range are used only if llama.cpp loads one and its KL equals that of the
    same genotype assembled normally (2 KL runs, once per model and starting type; the answer is stored)."""
    flag = out(model, f"probe-check-{start_tag(model)}.json")
    if flag.exists():
        return json.loads(flag.read_text())["identical"]
    b0 = sorted(blocks(model))[len(blocks(model)) // 2]
    a = block_assignment(model, b0, anchor_probes(model)[0] if ANCHOR and anchor(model) else probe_format(model))
    same, kls = False, {}
    try:
        fast = fast_probe(model, a)
        slow = out(model, "tmp", "probe-check-normal.gguf")
        assemble(inv_of(model), a, store(model), slow, base_format=base_format(model))
        for name, path in (("copied", fast), ("normal", slow)):
            label = f"rw:probecheck:{start_tag(model)}:{name}:{b0}"
            with gpu_booking(path.stat().st_size + 6 * 2**30, hold_s=1800):
                ok = evaluate.run("kl", model, path, label, THREADS, reference="sensitivity")
            if ok:
                kls[name] = find(records(model), label, reference="sensitivity")["metrics"]["kl_mean"]
        same = len(kls) == 2 and abs(kls["copied"] - kls["normal"]) <= 1e-6 + 1e-5 * kls["normal"]
        fast.unlink()
        slow.unlink()
    except Exception as e:  # noqa: BLE001  (any failure means: build probe files the normal way)
        print(f"probe check failed: {e}", flush=True)
    flag.write_text(json.dumps({"block": b0, "kl": kls, "identical": bool(same)}))
    note(f"{short_name(model)} part costs: fast probe files {'verified (same KL)' if same else 'FAILED the KL check; using normal assembly'}")
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


# --- sensitivity ------------------------------------------------------------------------------

def block_assignment(model: str, block: str | None, fmt: str | None = None) -> dict[str, str]:
    """The start, with every matrix of `block` moved to `fmt` where allowed."""
    a = start_assignment(model)
    if block is not None:
        for g in blocks(model)[block]:
            if fmt in allowed(model)[g]:
                a[g] = fmt
    return a


def sens_label(model: str, block: str, fmt: str) -> str:
    return f"rw:sens:{start_tag(model)}:{block}:{fmt}"


def _sens_one(job: tuple[str, str, str, bool]) -> None:
    model, block, fmt, fast = job
    a = block_assignment(model, None if block == "base" else block, fmt)
    path = fast_probe(model, a) if fast else build_file(model, a)
    try:
        with gpu_booking(path.stat().st_size + 6 * 2**30, hold_s=1800):
            kl(model, path, "sensitivity", sens_label(model, block, fmt))
    finally:
        path.unlink(missing_ok=True)


def measured_table(model: str) -> tuple[float | None, dict[str, dict[str, float]]]:
    """Base KL and measured dKL per (block, format), from the KL records."""
    base = start_tag(model)
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
    """per block, dKL as a piecewise-linear function of the block's offline error E through (base, 0) and
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
        # The start is (0, 0) by definition; a measured format with the same error as the start (the start type
        # itself, re-measured) is noise around it, and two points at one x would make the slope infinite.
        at: dict[float, list[float]] = {}
        for f, v in table.get(b, {}).items():
            x = E(f)
            if abs(x) > 1e-12:
                at.setdefault(x, []).append(v)
        pts = sorted([(0.0, 0.0)] + [(x, float(np.mean(vs))) for x, vs in at.items()])
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
    with locked(f"{model}-sens-{start_tag(model)}"):
        return _cmd_sens(model)


def _cmd_sens(model: str) -> int:
    base = start_tag(model)
    probes = anchor_probes(model) if ANCHOR and anchor(model) else (probe_format(model),)
    probe = "/".join(probes)
    stats = []
    todo = {("base", base)} | {(b, f) for b in blocks(model) for f in probes}
    for rnd in range(CFG["remeasure_rounds"] + 1):
        base_kl, table = measured_table(model)
        done = {(b, f) for b, row in table.items() for f in row} | ({("base", base)} if base_kl is not None else set())
        run = sorted(todo - done)
        print(f"{model} round {rnd}: {len(run)} KL runs (base {base}, probe {probe}, measured so far {len(done)})",
              flush=True)
        if ("base", base) in run:
            _sens_one((model, "base", base, False))
            run.remove(("base", base))
        fast = bool(run) and reflink_ok(model)  # checked once, here, before the parallel runs
        note(f"{short_name(model)} part costs round {rnd}: {len(run)} KL runs to do ({len(done)} measured so far)")
        # Each worker holds a whole-model file on disk; four lanes of ten once held 170 GB and filled the disk.
        workers = max(1, min(CONCURRENT, int(LANE_SCRATCH_GB * 1e9 // (base_file(model).stat().st_size * 1.2))))
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(_sens_one, (model, b, f, fast)): (b, f) for b, f in run}
            failed = []
            for i, fut in enumerate(as_completed(futs), 1):  # count runs as they finish
                try:
                    fut.result()
                except BaseException:  # noqa: BLE001  (one failed run never stops the lane)
                    failed.append(futs[fut])
                if i % 25 == 0 or i == len(run):
                    note(f"{short_name(model)} part costs round {rnd}: {i}/{len(run)}")
        for b, f in failed:  # retried one at a time
            try:
                _sens_one((model, b, f, fast))
            except BaseException as e:  # noqa: BLE001
                note(f"{short_name(model)} part costs: {b} {f} failed twice ({e}); continuing without it")
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
    note(f"{short_name(model)} part costs done: {sum(x['runs'] for x in stats)} KL runs")
    return 0


# --- helpers -------------------------------------------------------------------------------------------------

def file_ram_gib(model: str, path: Path) -> float:
    import gguf
    w = sum(int(t.n_bytes) for t in gguf.GGUFReader(path).tensors)
    return (w + inv_of(model).geometry.kv_cache_bytes(CTX, KV_BPE) + OVERHEAD * GIB) / GIB


# --- solution list, test all, pick ---------------------------------------------------------------------

WINDOW = (-0.01, 0.005)   # in the solution space iff RAM is within -1% / +0.5% of the budget
LIST_K = 6                # solver genotypes per model
LIST_DISTANCE = 0.05      # each differs from every other in at least 5% of its matrices
TEST_CONCURRENT = 4


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


def padded_from(model: str, a: dict[str, str], budget: float) -> dict[str, str] | None:
    """`a` with whole layers raised one menu type per matrix (alternating from both ends) until its RAM is inside
    the window, as padded_uniform does from a uniform start."""
    m = sorted(menu(model), key=bits_per_weight)
    inv = inv_of(model)
    a = dict(a)
    if ram_gib(model, a) > budget * (1 + WINDOW[1]):
        return None
    order, lo_, hi_ = [], 0, inv.geometry.n_layers - 1
    while lo_ <= hi_:
        order += [lo_] + ([hi_] if hi_ != lo_ else [])
        lo_, hi_ = lo_ + 1, hi_ - 1
    changed = True
    while changed and not in_window(model, a, budget):  # passes over the layers until inside, or nothing fits
        changed = False
        for L in order:
            if in_window(model, a, budget):
                break
            trial = dict(a)
            for g in inv.genes:
                if g.layer == L and a[g.gene_id] in m:
                    up = [f for f in m[m.index(a[g.gene_id]) + 1:] if f in allowed(model)[g.gene_id]]
                    if up:
                        trial[g.gene_id] = up[0]
            if trial != a and ram_gib(model, trial) <= budget * (1 + WINDOW[1]):
                a, changed = trial, True
    return a if in_window(model, a, budget) else None


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
            items[f"{short}:{'a-' if method_suffix(model) else ''}sol{i + 1}"] = {"model": model, "assignment": sol.assignment,
                                            "predicted_penalty": sol.penalty}
        pu = padded_uniform(model, budget)
        if pu:
            items[f"{short}:uniform-{pu[0]}-padded"] = {"model": model, "assignment": pu[1]}
        if ANCHOR and anchor(model):
            pa = padded_from(model, anchor(model)[1], budget)
            if pa:
                items[f"{short}:preset-{anchor(model)[0]}-layout-padded"] = {"model": model, "assignment": pa}
    for it in items.values():
        it["ram_gib"] = ram_gib(it["model"], it["assignment"])
    return items


def list_path(budget: int, model: str) -> Path:
    return out(f"budget-{budget}", f"list-{model}{method_suffix(model)}.json")


def load_list(budget: int) -> dict[str, dict]:
    """The budget's whole solution space: the union of its models' lists."""
    items = {}
    for model in contests()[budget]["searched"]:
        items.update(json.loads(list_path(budget, model).read_text()))
    return items


def cmd_list(budget: int, model: str) -> int:
    """One model's solution space for a budget, written once and never extended."""
    path = list_path(budget, model)
    if path.exists():
        print(f"list exists: {path}")
        return 0
    items = solution_list(budget, models=[model])
    for name, it in items.items():
        print(f"{budget} GiB {name:28s} RAM {it['ram_gib']:.3f} GiB", flush=True)
    path.write_text(json.dumps(items, indent=1))
    note(f"{short_name(model)} list: {len(items)} genotypes, RAM " + ", ".join(f"{it['ram_gib']:.3f}" for it in items.values()) + " GiB")
    return 0


def test_label(budget: int, name: str, a: dict[str, str] | None) -> str:
    return f"rw:b{budget}:{name}" + (f":{assignment_digest(a)[:12]}" if a else "")


def run_all_tests(model: str, path: Path, label: str, with_kl: bool) -> None:
    """Every quality test on one file: KL (selection text), the four test-subset tasks, perplexity, GSM8K."""
    if with_kl and not find(records(model), label, reference="selection"):
        kl(model, path, "selection", label)
    for t in SUBSET:
        if not find(records(model), label, task=t, split="poc_test"):
            if not evaluate.run("task", model, path, label, THREADS, task=t, split="poc_test"):
                raise SystemExit(f"{label} {t} failed")
    if not find(records(model), label, text="test_web"):
        if not evaluate.run("ppl", model, path, label, THREADS, text="test_web"):
            raise SystemExit(f"{label} ppl failed")
    gsm8k_run(model, path, label)


def tests_done(model: str, label: str, with_kl: bool) -> bool:
    recs = records(model)
    return ((not with_kl or find(recs, label, reference="selection") is not None)
            and all(find(recs, label, task=t, split="poc_test") for t in SUBSET)
            and find(recs, label, text="test_web") is not None
            and out(model, "gsm8k", f"{label.replace(':', '_')}.json").exists())


def _test_one(job: tuple[int, str, dict]) -> str:
    """One genotype: assembled once, all tests run on that file, then the file is deleted."""
    global BUDGET
    budget, name, it = job
    BUDGET = budget
    model, a = it["model"], it["assignment"]
    label = test_label(budget, name, a)
    if tests_done(model, label, with_kl=True):
        return name
    need_disk()
    path = build_file(model, a)
    try:
        with gpu_booking(path.stat().st_size + 10 * 2**30, hold_s=3 * 3600):
            run_all_tests(model, path, label, with_kl=True)
    finally:
        path.unlink(missing_ok=True)
    return name


def _test_ref(job: tuple[int, str]) -> str:
    budget, model = job
    name = f"{model.split('-')[1]}:uniform-{REF_MODEL_FORMAT}"
    path = DATA / "store" / model / f"{model}-{REF_MODEL_FORMAT}.gguf"
    label = test_label(budget, name, None)
    if not tests_done(model, label, with_kl=False):
        with gpu_booking(path.stat().st_size + 10 * 2**30, hold_s=3 * 3600):
            run_all_tests(model, path, label, with_kl=False)  # KL against a different model's original is meaningless
    return name


def cmd_test(budget: int, model: str) -> int:
    """One model's listed genotypes (or, for a reference model, its uniform Q6_K file), all at once on the GPU."""
    if model in contests()[budget]["reference"]:
        print(f"tested {_test_ref((budget, model))}", flush=True)
        return 0
    items = json.loads(list_path(budget, model).read_text())
    sm = short_name(model)

    def done() -> int:
        return sum(tests_done(model, test_label(budget, n, it["assignment"]), with_kl=True) for n, it in items.items())
    note(f"{sm} tests: {done()}/{len(items)} genotypes done, testing the rest {TEST_CONCURRENT} at a time")
    failed = []
    for attempt in (1, 2):  # a genotype that fails (e.g. a crashed run) is retried once
        todo = [(budget, n, it) for n, it in items.items() if not tests_done(model, test_label(budget, n, it["assignment"]), with_kl=True)]
        with ProcessPoolExecutor(TEST_CONCURRENT if attempt == 1 else 1) as ex:
            futs = {ex.submit(_test_one, j): j[1] for j in todo}
            failed = []
            for f in as_completed(futs):
                try:
                    f.result()
                    note(f"{sm} tests: {futs[f]} done ({done()}/{len(items)} genotypes)")
                except BaseException as e:  # noqa: BLE001
                    failed.append(futs[f])
                    note(f"{sm} tests: {futs[f]} failed ({e}); {'retrying' if attempt == 1 else 'giving up'}")
        if not failed:
            break
    return 1 if failed else 0


def gsm8k_run(model: str, path: Path, label: str) -> dict:
    rec_path = out(model, "gsm8k", f"{label.replace(':', '_')}.json")
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    items = jsonl(DATA / "splits/llama_tasks/poc_test/gsm8k.jsonl")
    fewshot = jsonl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    import socket
    with socket.socket() as sock:  # a free port: several genotypes run GSM8K at the same time
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with evaluate.LlamaServer(path, parallel=8, ctx_per_slot=2048, threads=THREADS, port=port,
                              log=out(model, "servers", f"{path.stem}.log")) as srv:
        comps = srv.map(lambda it: srv.complete(evaluate.gsm8k_prompt(it["question"], fewshot),
                                                evaluate.GSM8K_MAX_TOKENS, evaluate.GSM8K_STOP), items)
    correct = [int(evaluate.gsm8k_correct(c, it["answer"])) for it, c in zip(items, comps)]
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
    return Evaluation(name, correct, nll, np.full(len(nll), p["tokens"]), int(ram * GIB), 0.0,
                      tokenizer=model.split("-")[0])  # Qwen2.5 sizes share one tokenizer; other families don't


def is_searched(name: str) -> bool:
    """Only the solver's genotypes (sol1.., a-sol1..) can win; padded baselines and preset layouts are controls."""
    return ":sol" in name or ":a-sol" in name or ":ga" in name


def choose_winner(rows: dict, evs: dict) -> tuple[str, list[str]]:
    """Highest mean accuracy, unless the paired bootstrap cannot tell others apart from it (stats.compare's rule: the
    top one is better only if the 95% interval of its lead lies above 0.5 points). Among those tied, the lowest KL
    wins: KL is measured over far more tokens than the benchmarks have questions. Returns (winner, tied names)."""
    from panther.stats import ACC_MARGIN, compare
    top = max(rows, key=lambda n: rows[n]["mean_accuracy_pct"])
    tied = [n for n in rows if n == top or compare(evs[top], evs[n]).acc_ci[0] <= ACC_MARGIN]
    return min(tied, key=lambda n: rows[n]["kl"]), tied


def cmd_pick(budget: int) -> int:
    """Per model: choose_winner (best accuracy; bootstrap ties go to the lower KL). Then winner vs its padded-uniform baseline, the
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
        cands = [n for n in items if items[n]["model"] == model and is_searched(n)]
        winners[model] = choose_winner({n: rows[n] for n in cands}, evs)[0]
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
    """One model's winner as soon as its own tests are done (choose_winner)."""
    items = json.loads(list_path(budget, model).read_text())
    rows, evs = {}, {}
    for name, it in items.items():
        label = test_label(budget, name, it["assignment"])
        ev = evs[name] = evaluation(model, label, it["ram_gib"], name)
        k = find(records(model), label, reference="selection")["metrics"]
        rows[name] = {"ram_gib": it["ram_gib"], "kl": k["kl_mean"], "mean_accuracy_pct": 100 * ev.mean_accuracy(),
                      "ppl": ev.perplexity(), "per_task_pct": {t: 100 * float(v.mean()) for t, v in ev.task_correct.items()}}
    win, tied = choose_winner({n: r for n, r in rows.items() if is_searched(n)}, evs)
    for n, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean_accuracy_pct"]):
        print(f"{budget} GiB {n:28s} RAM {r['ram_gib']:.3f} | KL {r['kl']:.4f} | acc {r['mean_accuracy_pct']:.2f} | "
              f"PPL {r['ppl']:.3f} | " + " ".join(f"{t[:4]} {v:.1f}" for t, v in r["per_task_pct"].items())
              + ("   <- winner" if n == win else "   (tied on accuracy)" if n in tied else
                 "" if is_searched(n) else "   (control)"), flush=True)
    out(f"budget-{budget}", f"winner-{model}{method_suffix(model)}.json").write_text(
        json.dumps({"winner": win, "tied_on_accuracy": tied, "rows": rows}, indent=1))
    return 0


def bundle_reference(budget: int, model: str, short: str) -> int:
    """A reference model's uniform Q6_K file as a Pi bundle, so its RAM and speed sit next to the winners'."""
    import gguf
    import shutil
    src = DATA / "store" / model / f"{model}-{REF_MODEL_FORMAT}.gguf"
    bundle = DATA / "pi_bundle" / f"budget-{budget}-{short}"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "splits/text/agreement.txt", bundle / "agreement.txt")
    dst = bundle / f"qwen2.5-{short}-b{budget}-uniform-{REF_MODEL_FORMAT.lower()}.gguf"
    if not dst.exists():
        subprocess.run(["cp", "--reflink=auto", str(src), str(dst)], check=True)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    tc: dict[str, int] = {}
    for t in gguf.GGUFReader(dst).tensors:
        tc[t.tensor_type.name.lower()] = tc.get(t.tensor_type.name.lower(), 0) + 1
    weights = sum(int(t.n_bytes) for t in gguf.GGUFReader(dst).tensors)
    kv = inv_of(model).geometry.kv_cache_bytes(CTX, KV_BPE)
    files = {dst.name: {"model": model, "role": "reference", "sha256": h.hexdigest(), "bytes": dst.stat().st_size,
                        "predicted_ram_gib": (weights + kv + OVERHEAD * GIB) / GIB,
                        "predicted": {"weights": weights, "kv_cache": kv, "overhead": int(OVERHEAD * GIB)},
                        "expected_type_counts": dict(sorted(tc.items())), "server_cpu_ppl_agreement": None}}
    manifest = {"r_max_gib": float(budget) * (1 + WINDOW[1]), "context_tokens": CTX,
                "kv_cache_type": CFG["kv_cache_type"], "overhead_gib_assumed": OVERHEAD, "files": files,
                "agreement_text": "agreement.txt", "verdict": {"comparisons": {}}, "recommended_file": dst.name,
                "recommended_reason": f"{model} uniform {REF_MODEL_FORMAT}, reference at {budget} GiB"}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=1))
    note(f"{short.upper()} Pi bundle ready: {dst.name}")
    return 0


# --- bundle for the Pi ---------------------------------------------------------------------------------------

def cmd_bundle_model(budget: int, model: str) -> int:
    """One model's winner as a Pi bundle (pi_bundle/budget-B-<size>), ready as soon as that model is picked."""
    import hashlib
    import shutil
    short = model.split("-")[1].lower()
    if model in contests()[budget]["reference"]:
        return bundle_reference(budget, model, short)
    name = json.loads(out(f"budget-{budget}", f"winner-{model}{method_suffix(model)}.json").read_text())["winner"]
    a = json.loads(list_path(budget, model).read_text())[name]["assignment"]
    bundle = DATA / "pi_bundle" / f"budget-{budget}-{short}{method_suffix(model)}"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "splits/text/agreement.txt", bundle / "agreement.txt")
    dst = bundle / f"{model.lower()}-b{budget}-winner{method_suffix(model)}.gguf"
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

# llama.cpp's own mixed presets (llama-quantize ftypes, which raise sensitive tensors on their own), ordered by
# their nominal bits per weight. The best existing quantization at a budget is the largest that fits.
PRESETS = ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "Q2_K_S", "IQ3_XXS", "Q2_K", "IQ3_XS", "IQ3_S",
           "Q3_K_S", "IQ3_M", "Q3_K_M", "IQ4_XS", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
PRESET_BPW = [1.56, 1.75, 2.06, 2.31, 2.5, 2.7, 2.9, 3.06, 3.2, 3.3, 3.44, 3.5, 3.66, 3.9, 4.25, 4.27, 4.58, 4.89,
              5.54, 5.69, 6.56, 8.5]


def cmd_preset(budget: int, model: str) -> int:
    """The best existing quantization that fits the budget: of llama.cpp's mixed presets (same BF16 file and
    importance matrix) whose RAM is at most the budget's top, the two largest are tested like the genotypes and the
    better one (choose_winner) is compared with the model's winner (paired bootstrap)."""
    from panther.stats import compare
    store = DATA / "store" / model
    bf16, imat = store / f"{model}-BF16.gguf", store / f"{model}.imatrix.gguf"
    inv = inv_of(model)
    top = budget * (1 + WINDOW[1])
    room = (top * GIB - inv.geometry.kv_cache_bytes(CTX, KV_BPE) - OVERHEAD * GIB) * 8 / sum(g.n_weights for g in inv.genes)
    fitting = []  # the two largest presets that fit; the better of them is "the best existing quantization"
    for ftype, bpw in sorted(zip(PRESETS, PRESET_BPW), key=lambda x: -x[1]):
        if bpw > room + 0.3 or len(fitting) == 2:
            continue
        path = store / f"preset-{ftype}.gguf"
        if not path.exists():
            subprocess.run([str(evaluate.LLAMA / "build-cuda/bin/llama-quantize"), "--imatrix", str(imat), str(bf16),
                            str(path) + ".part", ftype, str(THREADS)], check=True, capture_output=True)
            os.rename(str(path) + ".part", path)
        ram = file_ram_gib(model, path)
        note(f"{model.split('-')[1]} preset {ftype}: {ram:.3f} GiB ({'fits' if ram <= top else 'too big'})")
        if ram <= top:
            fitting.append((ftype, path, ram))
        else:
            path.unlink()  # a preset that doesn't fit is never used; it only takes disk
    if not fitting:
        note(f"{model.split('-')[1]} preset: none of llama.cpp's presets fits {budget} GiB")
        pfile = out(f"budget-{budget}", f"preset-{model}.json")
        pfile.write_text(json.dumps({**(json.loads(pfile.read_text()) if pfile.exists() else {}), "preset": None}))
        return 0
    rows, evs, info = {}, {}, {}
    for ftype, path, ram in fitting:
        name = f"preset-{ftype}"
        label = test_label(budget, name, None)
        run_all_tests(model, path, label, with_kl=True)
        evs[name] = evaluation(model, label, ram, name)
        rows[name] = {"mean_accuracy_pct": 100 * evs[name].mean_accuracy(),
                      "kl": find(records(model), label, reference="selection")["metrics"]["kl_mean"]}
        info[name] = (ftype, ram)
    name = choose_winner(rows, evs)[0]
    ftype, ram = info[name]
    ev_p, kl_p = evs[name], rows[name]["kl"]
    tried = {n: {"ram_gib": info[n][1], **rows[n], "ppl": evs[n].perplexity()} for n in rows}
    result = {"preset": ftype, "ram_gib": ram, "kl": kl_p, "mean_accuracy_pct": 100 * ev_p.mean_accuracy(),
              "ppl": ev_p.perplexity(), "tried": tried}
    pfile = out(f"budget-{budget}", f"preset-{model}.json")  # the presets themselves don't depend on the method
    result = {**(json.loads(pfile.read_text()) if pfile.exists() else {}), **result}
    wfile = out(f"budget-{budget}", f"winner-{model}{method_suffix(model)}.json")
    if wfile.exists():
        w = json.loads(wfile.read_text())
        it = json.loads(list_path(budget, model).read_text())[w["winner"]]
        ev_w = evaluation(model, test_label(budget, w["winner"], it["assignment"]), it["ram_gib"], w["winner"])
        c = compare(ev_w, ev_p)
        result[f"vs_winner{method_suffix(model)}"] = {"winner": w["winner"], "acc_diff_pts": 100 * c.acc_diff,
                               "acc_ci_pts": [100 * c.acc_ci[0], 100 * c.acc_ci[1]], "ppl_ratio": c.ppl_ratio,
                               "kl_winner": w["rows"][w["winner"]]["kl"]}
    pfile.write_text(json.dumps(result, indent=1))
    v = result.get(f"vs_winner{method_suffix(model)}") if wfile.exists() else None
    note(f"{model.split('-')[1]} preset {ftype}: acc {result['mean_accuracy_pct']:.2f}, KL {kl_p:.4f}, PPL "
         f"{result['ppl']:.3f}" + (f" | our winner {v['acc_diff_pts']:+.2f} pts [{v['acc_ci_pts'][0]:+.2f}, "
                                   f"{v['acc_ci_pts'][1]:+.2f}], KL {v['kl_winner']:.4f}, PPL ratio "
                                   f"{v['ppl_ratio']:.3f}" if v else ""))
    return 0


def cmd_upload(budget: int, model: str) -> int:
    """The model's Pi bundle to its private Hugging Face repo (PANTHER_HF_REPO, e.g. user/panther-{size}-{budget}g).
    The server and the Pi cannot reach each other, and the VPN to the server is slow; the Pi downloads from there.
    Files already there with the same content are not sent again."""
    from huggingface_hub import HfApi
    short = model.split("-")[1].lower()
    ref = model in contests()[budget]["reference"]  # a reference's uniform file doesn't depend on the method
    sfx = "" if ref else method_suffix(model)
    repo = os.environ["PANTHER_HF_REPO"].format(size=short, budget=budget) + sfx
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=DATA / "pi_bundle" / f"budget-{budget}-{short}{sfx}")
    note(f"{short.upper()} Pi bundle uploaded to {repo}")
    return 0


# --- genetic search ------------------------------------------------------------------------------------------
# Genome: one type per matrix. Fitness: measured KL from the original on benchmark questions (bench_text), one KL
# run per individual. Steady state: GA_INFLIGHT individuals are scored at once; whenever one finishes, a child is
# bred from the current population and started, so the GPU never waits for a generation to end.

GA_POP, GA_ELITE, GA_INFLIGHT = 16, 4, 4
GA_EVALS = 70                     # at most this many individuals
GA_MIN_EVALS, GA_PATIENCE, GA_MIN_GAIN = 30, 15, 0.2  # plateau: after 30, stop when 15 in a row gain < 0.2 points
GA_TOP = 3                        # the best by fitness, then confirmed on all selection questions
# Fitness: accuracy on a fixed subset of the selection questions (never the test split), with KL on the benchmark
# questions as a mild tie-break: F = accuracy points - GA_KL_WEIGHT x KL (0.01 KL weighs 0.1 points).
GA_FIT_TASKS = {"mmlu": 300, "arc_challenge": 200, "hellaswag": 200, "winogrande": 200}
GA_KL_WEIGHT = 10.0
GA_SOLVED = 4                     # children per round proposed by the solver on the fitted sensitivities


class Genes:
    """Per-matrix steps, bytes and offline error for one model at one budget, and the operators on genomes."""

    def __init__(self, model: str, budget: int, seed: int = 0, err: dict | None = None) -> None:
        import random
        self.model, self.budget, self.rng = model, budget, random.Random(seed)
        inv = inv_of(model)
        self.genes = {g.gene_id: g for g in inv.genes}
        al = allowed(model)
        self.steps = {g: sorted(al[g], key=bits_per_weight) for g in self.genes}
        self.bytes = {g: {f: gene_bytes(self.genes[g], f) for f in self.steps[g]} for g in self.genes}
        err = err or json.loads(out(model, "error.json").read_text())
        self.err = {g: {f: err[gene.tensors[0].name][f] for f in self.steps[g]} for g, gene in self.genes.items()}
        a0 = start_assignment(model)
        self.const = ram_gib(model, a0) * GIB - self.weights(a0)  # RAM that doesn't depend on the genome
        self.lo, self.hi = budget * (1 + WINDOW[0]) * GIB, budget * (1 + WINDOW[1]) * GIB
        n = inv.geometry.n_layers
        self.group = {}
        for g, gene in self.genes.items():
            role = gene.tensors[0].name.split(".")[-2] if gene.layer is not None else gene.kind
            self.group[g] = (role, -1 if gene.layer is None else 4 * gene.layer // n)

    def weights(self, a: dict[str, str]) -> int:
        return sum(self.bytes[g][a[g]] for g in self.genes)

    def ram(self, a: dict[str, str]) -> float:
        return self.weights(a) + self.const

    def fits(self, a: dict[str, str]) -> bool:
        return self.lo <= self.ram(a) <= self.hi

    def move(self, a: dict[str, str], g: str, k: int) -> None:
        fs = self.steps[g]
        a[g] = fs[max(0, min(len(fs) - 1, fs.index(a[g]) + k))]

    def repair(self, a: dict[str, str]) -> dict[str, str]:
        """Into the RAM window: lower (raise) one step at a time a gene picked at random among the 8 that cost the
        least offline error per byte saved (remove the most per byte added). Random among the best, so repairs
        differ and the population stays diverse."""
        r = self.ram(a)
        for _ in range(20000):
            if r > self.hi:
                c = [((self.err[g][fs[i - 1]] - self.err[g][a[g]]) / max(1, self.bytes[g][a[g]] - self.bytes[g][fs[i - 1]]), g)
                     for g, fs in self.steps.items() for i in [fs.index(a[g])] if i > 0]
                g = self.rng.choice(sorted(c)[:8])[1]
                b0 = self.bytes[g][a[g]]
                self.move(a, g, -1)
                r += self.bytes[g][a[g]] - b0
            elif r < self.lo:
                c = [(-(self.err[g][a[g]] - self.err[g][fs[i + 1]]) / max(1, self.bytes[g][fs[i + 1]] - self.bytes[g][a[g]]), g)
                     for g, fs in self.steps.items() for i in [fs.index(a[g])]
                     if i + 1 < len(fs) and r + self.bytes[g][fs[i + 1]] - self.bytes[g][a[g]] <= self.hi]
                if not c:
                    break
                g = self.rng.choice(sorted(c)[:8])[1]
                b0 = self.bytes[g][a[g]]
                self.move(a, g, +1)
                r += self.bytes[g][a[g]] - b0
            else:
                break
        return a

    def mutate(self, a: dict[str, str]) -> dict[str, str]:
        """Small (1-3 genes one step), large (1-3 genes two or three steps) or group (every gene of one matrix role
        in one quarter of the depth, one step), then repaired into the window, which pays for raises with lowers."""
        a, gs, x = dict(a), list(self.genes), self.rng.random()
        if x < 0.5:
            for g in self.rng.sample(gs, self.rng.randint(1, 3)):
                self.move(a, g, self.rng.choice((-1, 1)))
        elif x < 0.8:
            for g in self.rng.sample(gs, self.rng.randint(1, 3)):
                self.move(a, g, self.rng.choice((-3, -2, 2, 3)))
        else:
            grp, k = self.rng.choice(sorted(set(self.group.values()))), self.rng.choice((-1, 1))
            for g in gs:
                if self.group[g] == grp:
                    self.move(a, g, k)
        return self.repair(a)

    def features(self, a: dict[str, str]) -> np.ndarray:
        """Offline error summed per group (matrix role x depth quarter): the surrogate's inputs."""
        groups = sorted(set(self.group.values()))
        gi = {x: i for i, x in enumerate(groups)}
        x = np.zeros(len(groups))
        for g in a:
            x[gi[self.group[g]]] += self.err[g][a[g]]
        return x

    def proposals(self, measured: list[tuple[float, dict[str, str]]], k: int) -> list[dict[str, str]]:
        """Fit KL ~ sum over groups of (sensitivity x the group's offline error) to every measured genome, then let
        the solver find the k best distinct genomes inside the window under that fit."""
        from panther.solver import Problem
        from panther.solver import solve_top_k
        groups = sorted(set(self.group.values()))
        gi = {x: i for i, x in enumerate(groups)}
        X = np.array([self.features(a) for _, a in measured])
        y = np.array([kl for kl, _ in measured])
        scale = X.mean(axis=0) + 1e-30
        coef, *_ = np.linalg.lstsq(np.c_[X / scale, np.ones(len(y))], y, rcond=None)
        sens = np.maximum(coef[:-1], 1e-9 * max(1e-12, float(np.abs(coef[:-1]).max()))) / scale
        pen = {g: {f: float(sens[gi[self.group[g]]] * self.err[g][f]) for f in self.steps[g]} for g in self.genes}
        prob = Problem(inv_of(self.model), allowed(self.model), pen, self.hi / GIB, CTX, OVERHEAD,
                       kv_bytes_per_element=KV_BPE)
        lo = prob.weight_budget_bytes - int((self.hi - self.lo))
        sols = solve_top_k(prob, k=k, time_limit_s=20, min_bytes=lo, min_distance=3)
        return [s.assignment for s in sols if self.fits(s.assignment)]

    def crossover(self, p: dict[str, str], q: dict[str, str]) -> dict[str, str]:
        """Whole layers from one parent or the other (layers keep their own balance of types)."""
        take = {}
        child = {}
        for g, gene in self.genes.items():
            L = gene.layer if gene.layer is not None else -1
            if L not in take:
                take[L] = self.rng.random() < 0.5
            child[g] = p[g] if take[L] else q[g]
        return self.repair(child)

    def initial(self) -> list[dict[str, str]]:
        """Diverse: the preset's layout, every uniform type padded into the window, random mixes, preset mutants."""
        pop = [self.repair(start_assignment(self.model))]
        for f in sorted(menu(self.model), key=bits_per_weight):
            a = padded_from(self.model, fill(self.model, f), self.budget)
            if a:
                pop.append(a)
        while len(pop) < GA_POP // 2 + 2:
            a = fill(self.model, self.rng.choice(sorted(menu(self.model), key=bits_per_weight)))
            for g in self.genes:
                self.move(a, g, self.rng.choice((-2, -1, 0, 0, 1, 2)))
            pop.append(self.repair(a))
        uniq: dict[str, dict[str, str]] = {}
        for a in pop:  # padded uniforms of neighbouring types can end up the same genome
            if self.fits(a):
                uniq.setdefault(assignment_digest(a), a)
        for _ in range(500):
            if len(uniq) >= GA_POP:
                break
            a = pop[0]
            for _ in range(self.rng.randint(2, 6)):
                a = self.mutate(a)
            if self.fits(a):
                uniq.setdefault(assignment_digest(a), a)
        return list(uniq.values())


def ga_label(budget: int, a: dict[str, str]) -> str:
    return f"rw:ga{budget}:{assignment_digest(a)[:12]}"


def ga_scores(model: str, label: str) -> tuple[float, float] | None:
    """(subset accuracy %, bench KL) if both are measured."""
    recs = records(model)
    r = find(recs, label, reference="bench")
    accs = [find(recs, label, task=t, split="selection", n_tasks=n) for t, n in GA_FIT_TASKS.items()]
    if r is None or not all(accs):
        return None
    return float(np.mean([x["metrics"]["accuracy_pct"] for x in accs])), r["metrics"]["kl_mean"]


def ga_fitness(acc: float, kl_: float) -> float:
    return acc - GA_KL_WEIGHT * kl_


def _ga_eval(job: tuple[int, str, dict, bool]) -> tuple[float, float]:
    """One individual: built once; KL on the benchmark questions and the subset accuracy on that one file."""
    global BUDGET
    budget, model, a, fast = job
    BUDGET = budget
    label = ga_label(budget, a)
    if ga_scores(model, label) is None:
        path = fast_probe(model, a) if fast else build_file(model, a)
        try:
            with gpu_booking(path.stat().st_size + 8 * 2**30, hold_s=1800):
                if not find(records(model), label, reference="bench"):
                    kl(model, path, "bench", label)
                for t, n in GA_FIT_TASKS.items():
                    if not find(records(model), label, task=t, split="selection", n_tasks=n):
                        if not evaluate.run("task", model, path, label, THREADS, task=t, split="selection", n_tasks=n):
                            raise SystemExit(f"{label} {t} failed")
        finally:
            path.unlink(missing_ok=True)
    return ga_scores(model, label)


def cmd_ga(budget: int, model: str) -> int:
    """The genetic search: steady state, fitness = KL on benchmark questions. Writes the GA_TOP best as the
    model's list (names <size>:ga1..) and the whole history to budget-B/ga-<model>.json."""
    sm = short_name(model)
    G = Genes(model, budget)
    fast = reflink_ok(model)
    pop: list[tuple[float, str, dict]] = []  # (-fitness, digest, genome), the best GA_POP kept
    seen: dict[str, float] = {}
    history = []
    measured: list[tuple[float, dict[str, str]]] = []
    queue = G.initial()
    lb = out(f"budget-{budget}", f"ga-{model}-live.json")  # a best genome from an earlier run joins the start
    if lb.exists():
        queue.insert(0, json.loads(lb.read_text())["assignment"])
    best, last_gain, n = float("inf"), 0, 0
    note(f"{sm} GA: {len(queue)} starting genomes (preset layout, padded uniforms, random mixes, preset mutants)")

    def child() -> dict[str, str]:
        for _ in range(50):
            if len(pop) < 2 or G.rng.random() < 0.3:
                a = G.mutate(G.rng.choice(pop[:GA_ELITE] or pop)[2])
            else:
                p, q = (min(G.rng.sample(pop, 2), key=lambda t: t[0])[2] for _ in range(2))
                a = G.mutate(G.crossover(p, q)) if G.rng.random() < 0.5 else G.crossover(p, q)
            if assignment_digest(a) not in seen and G.fits(a):
                return a
        return G.mutate(pop[0][2])

    with ProcessPoolExecutor(GA_INFLIGHT) as ex:
        running = {}

        def submit(a: dict[str, str]) -> None:
            d = assignment_digest(a)
            seen[d] = float("nan")
            running[ex.submit(_ga_eval, (budget, model, a, fast))] = (d, a)
        while queue and len(running) < GA_INFLIGHT:
            submit(queue.pop(0))
        while running:
            fut = next(as_completed(running))
            d, a = running.pop(fut)
            try:
                acc, kl_ = fut.result()
                k = -ga_fitness(acc, kl_)
            except BaseException as e:  # noqa: BLE001  (one failed individual never stops the search)
                note(f"{sm} GA: an individual failed ({e})")
                acc, kl_, k = float("nan"), float("nan"), float("inf")
            n += 1
            seen[d] = k
            history.append({"n": n, "digest": d, "fitness": -k, "accuracy": acc, "kl": kl_, "ram_gib": G.ram(a) / GIB})
            if k != float("inf"):
                measured.append((k, a))
            pop.append((k, d, a))
            pop.sort(key=lambda t: t[0])
            del pop[GA_POP:]
            if k < best - 1e-9:
                if k < best - GA_MIN_GAIN:  # smaller gains are within the noise of the subset: not progress
                    last_gain = n
                best = k
                out(f"budget-{budget}", f"ga-{model}-live.json").write_text(json.dumps(  # for an early check
                    {"n": n, "fitness": -k, "accuracy": acc, "kl": kl_, "assignment": a, "ram_gib": G.ram(a) / GIB}))
                note(f"{sm} GA: individual {n}: new best fitness {-k:.2f} (subset accuracy {acc:.2f}, bench KL "
                     f"{kl_:.4f}, RAM {G.ram(a) / GIB:.3f} GiB)")
            elif n % 10 == 0:
                note(f"{sm} GA: {n} individuals, best fitness {-best:.2f}")
            if n + len(running) < GA_EVALS and (n < GA_MIN_EVALS or n - last_gain < GA_PATIENCE):
                if not queue and len(measured) >= 8:
                    try:  # a round: the solver's picks under the fit, then GA children around them
                        picks = [a for a in G.proposals(measured, GA_SOLVED) if assignment_digest(a) not in seen]
                    except Exception as e:  # noqa: BLE001  (a failed fit only means GA children this round)
                        note(f"{sm} GA: surrogate fit failed ({e})")
                        picks = []
                    queue.extend(picks)
                    queue.extend(G.mutate(a) for a in picks[:2])
                submit(queue.pop(0) if queue else child())
    top = pop[:GA_TOP]
    items = {}
    for i, (k, d, a) in enumerate(top, 1):
        acc, kl_ = ga_scores(model, ga_label(budget, a))
        items[f"{model.split('-')[1]}:ga{i}"] = {"model": model, "assignment": a, "fitness": -k, "subset_accuracy": acc,
                                                 "bench_kl": kl_, "ram_gib": ram_gib(model, a)}
    list_path(budget, model).write_text(json.dumps(items, indent=1))
    out(f"budget-{budget}", f"ga-{model}.json").write_text(json.dumps({"evaluated": n, "best_fitness": -best,
                                                                       "history": history}, indent=1))
    note(f"{sm} GA done: {n} individuals; top {len(items)}: "
         + ", ".join(f"fitness {v['fitness']:.2f} (accuracy {v['subset_accuracy']:.2f}, KL {v['bench_kl']:.4f})"
                     for v in items.values()))
    return 0


def cmd_gacheck(budget: int, model: str) -> int:
    """Early check while the GA runs: its current best and the budget's preset, both on the selection benchmarks
    (multiple choice, never the test split), compared by the paired bootstrap."""
    from panther.stats import Evaluation, compare
    live = json.loads(out(f"budget-{budget}", f"ga-{model}-live.json").read_text())
    ftype = json.loads(out(f"budget-{budget}", f"preset-{model}.json").read_text())["preset"]
    evs = {}
    for name, a, path in (("preset", None, DATA / "store" / model / f"preset-{ftype}.gguf"),
                          (f"ga-n{live['n']}", live["assignment"], None)):
        label = f"rw:gacheck{budget}:{name}" + (f":{assignment_digest(a)[:12]}" if a else f":{ftype}")
        if not all(find(records(model), label, task=t, split="selection") for t in SUBSET):
            built = path is None
            path = path or build_file(model, a)
            try:
                with gpu_booking(path.stat().st_size + 10 * 2**30, hold_s=3600):
                    for t in SUBSET:
                        if not find(records(model), label, task=t, split="selection"):
                            evaluate.run("task", model, path, label, THREADS, task=t, split="selection")
            finally:
                if built:
                    path.unlink(missing_ok=True)
        c = {t: np.array(find(records(model), label, task=t, split="selection")["per_item"]["correct"], dtype=np.int8)
             for t in SUBSET}
        evs[name] = Evaluation(name, c, np.ones(1), np.ones(1), 0, 0.0, tokenizer=model)
    g, pr = evs[f"ga-n{live['n']}"], evs["preset"]
    cmp = compare(g, pr)
    note(f"{short_name(model)} early check, GA best so far (individual {live['n']}, bench KL {live['kl']:.4f}) vs "
         f"preset {ftype}: selection accuracy {100 * g.mean_accuracy():.2f} vs {100 * pr.mean_accuracy():.2f}, "
         f"{100 * cmp.acc_diff:+.2f} pts [{100 * cmp.acc_ci[0]:+.2f}, {100 * cmp.acc_ci[1]:+.2f}] | "
         + " ".join(f"{t[:4]} {100 * g.task_correct[t].mean():.1f}/{100 * pr.task_correct[t].mean():.1f}" for t in SUBSET))
    return 0


def cmd_gascore(budget: int, model: str) -> int:
    """The GA's top genomes scored on the selection benchmarks (multiple choice, all 4,500 questions; never the test
    split). The winner: most accurate, ties by bootstrap to the lowest bench KL. Then the winner's full test (test
    split, perplexity, GSM8K) and the winner file the later steps use."""
    from panther.stats import Evaluation
    items = json.loads(list_path(budget, model).read_text())
    sm = short_name(model)
    rows, evs = {}, {}

    def score(name: str) -> None:
        it = items[name]
        label = test_label(budget, name, it["assignment"])
        if not all(find(records(model), label, task=t, split="selection") for t in SUBSET):
            path = build_file(model, it["assignment"])
            try:
                with gpu_booking(path.stat().st_size + 10 * 2**30, hold_s=3600):
                    for t in SUBSET:
                        if not find(records(model), label, task=t, split="selection"):
                            if not evaluate.run("task", model, path, label, THREADS, task=t, split="selection"):
                                raise SystemExit(f"{label} {t} failed")
            finally:
                path.unlink(missing_ok=True)
        correct = {t: np.array(find(records(model), label, task=t, split="selection")["per_item"]["correct"], dtype=np.int8)
                   for t in SUBSET}
        evs[name] = Evaluation(name, correct, np.ones(1), np.ones(1), int(it["ram_gib"] * GIB), 0.0, tokenizer=model)
        rows[name] = {"mean_accuracy_pct": 100 * evs[name].mean_accuracy(), "kl": it["bench_kl"]}
        note(f"{sm} GA {name}: selection accuracy {rows[name]['mean_accuracy_pct']:.2f}, bench KL {it['bench_kl']:.4f}")
    for name in items:
        score(name)
    win, tied = choose_winner(rows, evs)
    # Guard: the winner against llama.cpp's real preset file on the same 4,500 questions (paired bootstrap).
    from panther.stats import compare
    ftype = json.loads(out(f"budget-{budget}", f"preset-{model}.json").read_text())["preset"]
    plabel = f"rw:gacheck{budget}:preset:{ftype}"
    ppath = DATA / "store" / model / f"preset-{ftype}.gguf"
    for t in SUBSET:
        if not find(records(model), plabel, task=t, split="selection"):
            with gpu_booking(ppath.stat().st_size + 10 * 2**30, hold_s=3600):
                evaluate.run("task", model, ppath, plabel, THREADS, task=t, split="selection")
    pev = Evaluation("preset", {t: np.array(find(records(model), plabel, task=t, split="selection")["per_item"]["correct"],
                                            dtype=np.int8) for t in SUBSET}, np.ones(1), np.ones(1), 0, 0.0, tokenizer=model)
    g = compare(evs[win], pev)
    note(f"{sm} guard: {win} vs preset {ftype} on all selection questions: {100 * evs[win].mean_accuracy():.2f} vs "
         f"{100 * pev.mean_accuracy():.2f}, {100 * g.acc_diff:+.2f} pts [{100 * g.acc_ci[0]:+.2f}, {100 * g.acc_ci[1]:+.2f}]")
    if g.acc_ci[1] < 0:
        note(f"{sm} guard: WORSE than the preset; not taken as our result (the GA must search further)")
        return 2
    it = items[win]
    label = test_label(budget, win, it["assignment"])
    if not tests_done(model, label, with_kl=True):
        path = build_file(model, it["assignment"])
        try:
            with gpu_booking(path.stat().st_size + 10 * 2**30, hold_s=3 * 3600):
                run_all_tests(model, path, label, with_kl=True)
        finally:
            path.unlink(missing_ok=True)
    ev = evaluation(model, label, it["ram_gib"], win)
    k = find(records(model), label, reference="selection")["metrics"]
    final = {"ram_gib": it["ram_gib"], "kl": k["kl_mean"], "bench_kl": it["bench_kl"],
             "selection_accuracy_pct": rows[win]["mean_accuracy_pct"], "mean_accuracy_pct": 100 * ev.mean_accuracy(),
             "ppl": ev.perplexity(), "per_task_pct": {t: 100 * float(v.mean()) for t, v in ev.task_correct.items()}}
    out(f"budget-{budget}", f"winner-{model}{method_suffix(model)}.json").write_text(json.dumps(
        {"winner": win, "tied_on_accuracy": tied, "selection": rows, "rows": {win: final}}, indent=1))
    note(f"{sm} GA winner {win}: test accuracy {final['mean_accuracy_pct']:.2f}, KL {final['kl']:.4f}, "
         f"PPL {final['ppl']:.3f}, GSM8K {final['per_task_pct']['gsm8k']:.1f}")
    return 0


def main() -> int:
    cmd, args = sys.argv[1], sys.argv[2:]
    global BUDGET
    if cmd == "refs":
        with locked(f"{args[-1]}-refs"):
            return cmd_refs(args[-1])
    BUDGET = int(args[0])
    if cmd == "pick" and len(args) == 1:
        return cmd_pick(BUDGET)
    steps = {"store": cmd_store, "error": cmd_error, "sens": cmd_sens}
    if cmd in steps:
        return steps[cmd](args[1])
    steps2 = {"list": cmd_list, "test": cmd_test, "pick": cmd_pick_model, "bundle": cmd_bundle_model,
              "upload": cmd_upload, "preset": cmd_preset, "ga": cmd_ga, "gascore": cmd_gascore,
              "gacheck": cmd_gacheck}
    if cmd in steps2:
        return steps2[cmd](BUDGET, args[1])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001  (a failure is never silent: it goes into the progress stream)
        if BUDGET is not None:
            note(f"FAILED: {' '.join(sys.argv[1:])}: {type(e).__name__}: {e}")
        raise
