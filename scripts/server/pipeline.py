"""The Panther pipeline (README: "How it works"). A contest is one model at one RAM budget (GiB).

  pipeline.py fetch BUDGET MODEL     # CPU container (has network): the weights at the pinned revision
  pipeline.py run BUDGET MODEL       # GPU container: store, error, refs, preset, ga, final, bundle in order
  pipeline.py STEP BUDGET MODEL      # one of those steps; every step skips work already on disk
  pipeline.py upload BUDGET MODEL    # CPU container: the Pi bundle to Hugging Face
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path

import numpy as np

from panther import llama, stats
from panther.llama import DATA, THREADS
from panther.model import CFG, GIB, MODELS, Model, bpw
from panther.solver import solve

RW, STORE, BIN = DATA / "rewrite", DATA / "store", llama.BIN
TASKS = ["mmlu", "hellaswag", "arc_challenge", "winogrande"]
FIT = {"mmlu": 300, "arc_challenge": 200, "hellaswag": 200, "winogrande": 200}  # the GA's selection questions
WINDOW = (-0.01, 0.005)  # a genome's RAM must be within -1% / +0.5% of the budget
POP, ELITE, INFLIGHT, MAX_EVALS, MIN_EVALS, PATIENCE, MIN_GAIN, TOP, SOLVED = 16, 4, 4, 70, 30, 15, 0.2, 3, 4
KL_WEIGHT = 10  # score = flips % + 10 x KL on the original's own answers (lower is better)
SUFFIX = "-v2"
FTYPE = {"Q3_K": "Q3_K_S", "Q4_K": "Q4_K_S", "Q5_K": "Q5_K_S", "IQ2_S": "IQ2_M"}  # --pure: one type everywhere
PRESETS = {"IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7, "Q2_K_S": 2.9,
           "IQ3_XXS": 3.06, "Q2_K": 3.2, "IQ3_XS": 3.3, "IQ3_S": 3.44, "Q3_K_S": 3.5, "IQ3_M": 3.66, "Q3_K_M": 3.9,
           "IQ4_XS": 4.25, "Q3_K_L": 4.27, "Q4_K_S": 4.58, "Q4_K_M": 4.89, "Q5_K_S": 5.54, "Q5_K_M": 5.69,
           "Q6_K": 6.56, "Q8_0": 8.5}
BUDGET = 0


def note(msg: str) -> None:
    """One progress line (IST clock) in data/logs/progress-b<BUDGET>.log, which the status page reads."""
    line = f"{time.strftime('%H:%M', time.gmtime(time.time() + 19800))} {msg}"
    print(line, flush=True)
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    with open(DATA / "logs" / f"progress-b{BUDGET}.log", "a") as f:
        f.write(line + "\n")


def out(*parts: str) -> Path:
    p = RW.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def short(model: str) -> str:
    return model.split("-")[1]


def copy_of(model: str, t: str) -> Path:
    return STORE / model / f"{model}-{t}.gguf"


def digest(a: dict[str, str]) -> str:
    return sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()[:12]


@contextmanager
def lock(name: str):
    """One writer per (model, step) across lanes; the second waits, then finds the work done."""
    (DATA / "locks").mkdir(parents=True, exist_ok=True)
    with open(DATA / "locks" / f"{name}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


@contextmanager
def gpu(nbytes: int, hold_s: int = 3600):
    """Books GPU memory in a shared ledger and waits until it fits in 92 of the GPU's 96 GiB."""
    ledger, key = DATA / "locks/gpu.json", f"{os.uname().nodename}-{os.getpid()}-{time.time()}"

    def edit(fn):
        with lock("gpu-ledger"):
            try:
                book = {k: v for k, v in json.loads(ledger.read_text()).items() if v[1] > time.time()}
            except (OSError, ValueError):
                book = {}
            ok = fn(book)
            ledger.write_text(json.dumps(book))
            return ok
    take = lambda b: (sum(v[0] for v in b.values()) + nbytes <= 92 * GIB or not b) and not b.update(  # noqa: E731
        {key: (nbytes, time.time() + 600)})
    while not edit(take):
        time.sleep(random.uniform(3, 8))
    import threading
    done = threading.Event()

    def renew():  # a killed job's booking lapses within 10 minutes
        while not done.wait(120):
            edit(lambda b: key in b and not b.update({key: (nbytes, time.time() + 600)}) or True)
    threading.Thread(target=renew, daemon=True).start()
    try:
        yield
    finally:
        done.set()
        edit(lambda b: b.pop(key, None) or True)


# --- fetch, store, error, references ----------------------------------------------------------------------------

def fetch(model: str) -> None:
    """The weights, unless the store already has everything this budget needs (rounding reads them)."""
    hf, done = DATA / "hf" / model, STORE / model / "rounded.json"
    if (hf / "config.json").exists() or (done.exists() and set(CFG["formats"]) <= set(
            json.loads(done.read_text()))):
        return
    m = MODELS[model]
    subprocess.run([os.environ["PANTHER_HF"], "download", m["repo"], "--revision", m["revision"], "--local-dir",
                    str(hf)], check=True)


def store(model: str) -> None:
    """The 16-bit original, its importance matrix, and the whole model once in every type of the budget's menu."""
    d, bf16, imat = STORE / model, copy_of(model, "BF16"), STORE / model / f"{model}.imatrix.gguf"
    d.mkdir(parents=True, exist_ok=True)
    if not bf16.exists():  # InternLM's converter needs a newer protobuf, kept in data/pydeps-convert
        env = dict(os.environ, PYTHONPATH=f"{DATA}/pydeps-convert:{os.environ.get('PYTHONPATH', '')}")
        subprocess.run([os.environ.get("PANTHER_CONVERT_PY", sys.executable), "/opt/llama.cpp/convert_hf_to_gguf.py",
                        str(DATA / "hf" / model), "--outtype", "bf16", "--outfile", f"{bf16}.part"], env=env, check=True)
        os.rename(f"{bf16}.part", bf16)
    if not imat.exists():
        po = ["--process-output"] if MODELS[model].get("imatrix_process_output") else []
        subprocess.run([str(BIN / "llama-imatrix"), "-m", str(bf16), "-f", str(DATA / "splits/calibration.txt"),
                        "-o", f"{imat}.part.gguf", "-c", "512", "-ngl", "99", "-t", str(THREADS)] + po, check=True)
        os.rename(f"{imat}.part.gguf", imat)
    M = Model(model)
    al, n = M.allowed(), sum(m.rows * m.cols for m in M.matrices)
    for t in CFG["formats"]:
        while not copy_of(model, t).exists() and shutil.disk_usage(DATA).free < n * bpw(t) / 8 + 30 * GIB:
            note(f"{short(model)} store: waiting for {n * bpw(t) / 8 / GIB + 30:.0f} GB of free disk for the {t} copy")
            time.sleep(600)
        if not copy_of(model, t).exists():  # embedding / head take Q8_0 where the type needs imatrix data they lack
            emb, head = ((t if t in al.get(n, al["token_embd.weight"]) else "Q8_0").lower()
                         for n in ("token_embd.weight", "output.weight"))
            part = f"{copy_of(model, t)}.part"
            subprocess.run([str(BIN / "llama-quantize"), "--pure", "--imatrix", str(imat), "--token-embedding-type",
                            emb, "--output-tensor-type", head, str(bf16), part, FTYPE.get(t, t), str(THREADS)],
                           check=True, capture_output=True)
            os.rename(part, copy_of(model, t))
    done = d / "rounded.json"  # types whose copy has error-compensated codes (panther.rounding)
    have = json.loads(done.read_text()) if done.exists() else []
    if todo := [t for t in CFG["formats"] if t not in have]:
        from panther.rounding import round_copies
        h, i = M.matrices[2].cols, max(x.cols for x in M.matrices)
        per_layer = 4 * (3 * h * h + i * i)  # bytes of the four input Hessians of one layer
        with gpu(bf16.stat().st_size + 24 * GIB + 2 * per_layer, 6 * 3600):
            round_copies(DATA / "hf" / model, bf16, {t: copy_of(model, t) for t in todo},
                         DATA / "splits/calibration.txt", max(1, int(20 * GIB // per_layer)))
        done.write_text(json.dumps(have + todo))
    if (DATA / "hf" / model).exists():
        shutil.rmtree(DATA / "hf" / model)  # the 16-bit GGUF is the original from here on


_READ: dict[Path, dict] = {}


def tensors(path: Path) -> dict:
    import gguf
    if path not in _READ:
        _READ[path] = {t.name: t for t in gguf.GGUFReader(path).tensors}
    return _READ[path]


def _error_one(job: tuple[str, str, list[str], np.ndarray]) -> tuple[str, dict[str, float]]:
    from gguf.quants import dequantize
    model, name, types, imp = job
    deq = lambda t: dequantize(tensors(copy_of(model, t))[name].data, tensors(copy_of(model, t))[name].tensor_type)  # noqa: E731
    w = deq("BF16").astype(np.float32)
    return name, {t: float(((w - deq(t).reshape(w.shape)) ** 2).sum(axis=0) @ imp[: w.shape[1]]) for t in types}


def error(model: str) -> None:
    """Offline error of every matrix in every type: squared error weighted by the importance matrix."""
    dst = out(model, "error.json")
    done = json.loads(dst.read_text()) if dst.exists() else {}
    imat = {k: np.asarray(v.data, dtype=np.float64).reshape(-1) for k, v in
            tensors(STORE / model / f"{model}.imatrix.gguf").items()}
    jobs = []
    for name, types in Model(model).allowed().items():
        if todo := [t for t in types if t not in done.get(name, {})]:
            cnt = max(float(imat.get(f"{name}.counts", np.ones(1))[0]), 1.0)
            jobs.append((model, name, todo, imat.get(f"{name}.in_sum2", np.ones(1 << 16)) / cnt))
    with ProcessPoolExecutor(8) as ex:
        for name, res in ex.map(_error_one, jobs):
            done.setdefault(name, {}).update(res)
    dst.write_text(json.dumps(done))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def refs(model: str) -> None:
    """The original's reference outputs: its next-word probabilities on the selection text and on its own answers
    (its GSM8K solutions and the multiple-choice questions with their answers), and which selection and test
    questions it gets right (to count flips)."""
    ref, bf16 = DATA / "reference" / model, copy_of(model, "BF16")
    ref.mkdir(parents=True, exist_ok=True)
    ans = DATA / "splits/text" / f"answers-{model}.txt"
    with gpu(bf16.stat().st_size + 12 * GIB, 4 * 3600):
        if not ans.exists():
            items = jsonl(DATA / "splits/tasks/selection/gsm8k.jsonl")[:200]
            shots = jsonl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
            sols = llama.complete(bf16, [llama.gsm8k_prompt(it["question"], shots) for it in items])
            text = [f"Question: {it['question']}\nAnswer:{s}" for it, s in zip(items, sols)]
            bench = DATA / "splits/text/bench.txt"  # selection multiple-choice questions with their answers
            mc = bench.read_text().split("\n\n") if bench.exists() else []
            ans.write_text("\n\n".join(x for pair in zip_longest(text, mc, fillvalue="") for x in pair if x) + "\n")
        for name, text, chunks in (("selection", DATA / "splits/text/selection.txt", None), ("answers", ans, 16)):
            if not (ref / f"{name}.kld").exists():
                subprocess.run([str(BIN / "llama-perplexity"), "-m", str(bf16), "-f", str(text), "-c", "2048", "-b",
                                "2048", "-ngl", "99", "-t", str(THREADS), "--kl-divergence-base", f"{ref}/{name}.part"]
                               + (["--chunks", str(chunks)] if chunks else []), check=True, capture_output=True)
                os.rename(f"{ref}/{name}.part", ref / f"{name}.kld")
        for t in TASKS:
            llama.run("task", model, bf16, "original", task=t, split="selection", n=FIT[t])
            llama.run("task", model, bf16, "original", task=t, split="selection", n=0)
        test(model, bf16, "original")


# --- building and scoring a genome ------------------------------------------------------------------------------

def build(model: str, a: dict[str, str], dst: Path | None = None) -> Path:
    """A GGUF with each matrix copied from the type copy of its type; norms and biases from the first copy."""
    import gguf
    dst = dst or out(model, "tmp", f"{digest(a)}.gguf")
    if dst.exists():
        return dst
    srcs = {t: gguf.GGUFReader(copy_of(model, t)) for t in set(a.values())}
    by = {t: {x.name: x for x in r.tensors} for t, r in srcs.items()}
    base = srcs[sorted(srcs)[0]]
    w = gguf.GGUFWriter(f"{dst}.{os.getpid()}.part", base.fields["general.architecture"].contents())
    for f in base.fields.values():
        if f.name != "general.architecture" and not f.name.startswith("GGUF."):
            w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    chosen = [by[a[t.name]][t.name] if t.name in a else t for t in base.tensors]
    for t in chosen:
        if t.name in a and t.tensor_type.name != a[t.name]:
            raise SystemExit(f"{t.name}: the {a[t.name]} copy holds {t.tensor_type.name}")
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in chosen:
        w.write_tensor_data(t.data)
    w.close()
    os.rename(f"{dst}.{os.getpid()}.part", dst)
    return dst


def correct(model: str, label: str, n: dict | None = None) -> dict[str, np.ndarray]:
    return {t: np.array(llama.find(model, label, task=t, split="selection", n=n[t] if n else 0)["correct"])
            for t in TASKS}


def test(model: str, path: Path, label: str) -> dict:
    """The held-out test: 4 multiple-choice tasks, GSM8K (test split), KL on the selection text, perplexity."""
    c = {t: np.array(llama.run("task", model, path, label, task=t, split="poc_test", n=0)["correct"]) for t in TASKS}
    c["gsm8k"] = llama.gsm8k(model, path, label, DATA / "splits/llama_tasks/poc_test/gsm8k.jsonl")
    kl = None if label == "original" else llama.run("kl", model, path, label, reference="selection")["metrics"]
    return {"correct": c, "kl": kl and kl["kl_mean"], "ppl": llama.run("ppl", model, path, label, text="test_web")[
        "metrics"]["ppl"]}


def score(model: str, a: dict[str, str] | None, label: str, path: Path | None = None, full: bool = False) -> dict:
    """Flips against the original and KL on its own answers (GA questions, or all selection questions if full)."""
    n = None if full else FIT
    built = path is None and not (llama.find(model, label, reference="answers") and all(
        llama.find(model, label, task=t, split="selection", n=n[t] if n else 0) for t in TASKS))
    try:
        path = path or (build(model, a) if built else Path("/dev/null"))
        with gpu(int(1.1 * path.stat().st_size) + 4 * GIB, 1800):
            kl = llama.run("kl", model, path, label, reference="answers")["metrics"]["kl_mean"]
            for t in TASKS:
                llama.run("task", model, path, label, task=t, split="selection", n=n[t] if n else 0)
    finally:
        if built and path:
            path.unlink(missing_ok=True)
    c, o = correct(model, label, n), correct(model, "original", n)
    f = stats.flips(c, o)
    return {"score": f + KL_WEIGHT * kl, "flips": f, "kl": kl, "accuracy": stats.acc(c)}


def _score_job(job):
    global BUDGET
    BUDGET, model, a = job
    return score(model, a, f"v2:{digest(a)}")


# --- the genetic search ----------------------------------------------------------------------------------------

class Genes:
    """Genome = one type per matrix inside the RAM window; mutation, crossover, repair and the solver's proposals."""

    def __init__(self, model: str, budget: int, seed: int = 0) -> None:
        M, self.rng, self.model = Model(model), random.Random(seed), model
        self.steps = M.allowed()
        self.mats = {m.name: m for m in M.matrices}
        self.bytes = {g: {t: self.mats[g].nbytes(t) for t in ts} for g, ts in self.steps.items()}
        err = json.loads(out(model, "error.json").read_text())
        self.err = {g: {t: err[g][t] for t in ts} for g, ts in self.steps.items()}
        a0 = {g: ts[0] for g, ts in self.steps.items()}
        self.const = M.ram(a0) - sum(self.bytes[g][a0[g]] for g in a0)
        self.lo, self.hi = budget * (1 + WINDOW[0]) * GIB, budget * (1 + WINDOW[1]) * GIB
        self.group = {g: (m.role, m.layer * 4 // M.layers) for g, m in self.mats.items()}

    def ram(self, a: dict[str, str]) -> int:
        return self.const + sum(self.bytes[g][a[g]] for g in a)

    def fits(self, a: dict[str, str]) -> bool:
        return self.lo <= self.ram(a) <= self.hi

    def move(self, a: dict[str, str], g: str, k: int) -> None:
        ts = self.steps[g]
        a[g] = ts[max(0, min(len(ts) - 1, ts.index(a[g]) + k))]

    def repair(self, a: dict[str, str]) -> dict[str, str]:
        """Into the window, one step at a time: lower one of the 8 matrices that lose least error per byte saved,
        or raise one of the 8 that gain most per byte added (random among them, so repairs differ)."""
        a = dict(a)
        for _ in range(20000):
            r, cand = self.ram(a), []
            if self.lo <= r <= self.hi:
                break
            k = -1 if r > self.hi else 1
            for g, ts in self.steps.items():
                i = ts.index(a[g]) + k
                if 0 <= i < len(ts):
                    db = self.bytes[g][ts[i]] - self.bytes[g][a[g]]
                    if k < 0 or r + db <= self.hi:
                        cand.append(((self.err[g][ts[i]] - self.err[g][a[g]]) / max(1, abs(db)), g))
            if not cand:
                break
            self.move(a, self.rng.choice(sorted(cand)[:8])[1], k)
        return a

    def mutate(self, a: dict[str, str]) -> dict[str, str]:
        """1-3 matrices one step, 1-3 matrices two or three steps, or a whole group (role x depth quarter)."""
        a, gs, x = dict(a), list(self.steps), self.rng.random()
        if x < 0.8:
            for g in self.rng.sample(gs, self.rng.randint(1, 3)):
                self.move(a, g, self.rng.choice((-1, 1) if x < 0.5 else (-3, -2, 2, 3)))
        else:
            grp, k = self.rng.choice(sorted(set(self.group.values()))), self.rng.choice((-1, 1))
            for g in gs:
                if self.group[g] == grp:
                    self.move(a, g, k)
        return self.repair(a)

    def crossover(self, p: dict[str, str], q: dict[str, str]) -> dict[str, str]:
        take = {L: self.rng.random() < 0.5 for L in {m.layer for m in self.mats.values()}}
        return self.repair({g: p[g] if take[self.mats[g].layer] else q[g] for g in p})

    def proposals(self, scored: list[tuple[float, dict[str, str]]], k: int) -> list[dict[str, str]]:
        """Fit score ~ sum over groups of (weight x the group's offline error), then the solver's k best genomes."""
        groups = sorted(set(self.group.values()))
        X = np.array([[sum(self.err[g][a[g]] for g in a if self.group[g] == grp) for grp in groups] for _, a in scored])
        s = X.mean(axis=0) + 1e-30
        coef = np.linalg.lstsq(np.c_[X / s, np.ones(len(X))], [v for v, _ in scored], rcond=None)[0][:-1]
        w = dict(zip(groups, np.maximum(coef, 1e-9 * max(1e-12, np.abs(coef).max())) / s))
        pen = {g: {t: float(w[self.group[g]] * e) for t, e in es.items()} for g, es in self.err.items()}
        sols = solve(self.bytes, pen, int(self.lo - self.const), int(self.hi - self.const), k)
        return [a for a in sols if self.fits(a)]

    def initial(self, preset: dict[str, str] | None) -> list[dict[str, str]]:
        """The preset's layout, every uniform type, random mixes, and mutants of the first, all repaired."""
        types = sorted({t for ts in self.steps.values() for t in ts}, key=bpw)
        pop = ([preset] if preset else []) + [{g: min(ts, key=lambda x: abs(bpw(x) - bpw(t))) for g, ts in
                                                self.steps.items()} for t in types]
        while len(pop) < POP // 2 + 2:
            t = self.rng.choice(types)
            pop.append({g: ts[max(0, min(len(ts) - 1, ts.index(min(ts, key=lambda x: abs(bpw(x) - bpw(t))))
                                          + self.rng.choice((-2, -1, 0, 0, 1, 2))))] for g, ts in self.steps.items()})
        uniq = {digest(a): a for a in map(self.repair, pop) if self.fits(a)}
        for _ in range(500):
            if len(uniq) >= POP:
                break
            a = self.mutate(self.mutate(pop[0]))
            if self.fits(a):
                uniq.setdefault(digest(a), a)
        return list(uniq.values())


def preset_layout(model: str, budget: int) -> dict[str, str] | None:
    """The competing preset's types, each mapped to the nearest type the genome may use."""
    p = json.loads(out(f"budget-{budget}", f"preset-{model}.json").read_text())["preset"]
    steps = Genes(model, budget).steps
    have = {n: t.tensor_type.name for n, t in tensors(STORE / model / f"preset-{p}.gguf").items()}
    return {g: have[g] if have[g] in ts else min(ts, key=lambda x: (abs(bpw(x) - bpw(have[g])), -bpw(x)))
            for g, ts in steps.items()}


def ga(model: str) -> None:
    """Steady state: as many genomes scored at once as the GPU holds (up to 16, as GPU memory and the container's RAM allow), a new child whenever one finishes; each round the solver
    adds SOLVED proposals from a fit of every score so far. Stops on a plateau. Writes the TOP best."""
    if (lst := out(f"budget-{BUDGET}", f"list-{model}{SUFFIX}.json")).exists():
        return
    for f in out(model, "tmp", "x").parent.glob("*.gguf*"):  # left by a killed run
        if time.time() - f.stat().st_mtime > 3600:
            f.unlink()
    G, sm = Genes(model, BUDGET), short(model)
    queue, pop, seen, scored, hist = G.initial(preset_layout(model, BUDGET)), [], set(), [], []
    best, last_gain, n = float("inf"), 0, 0
    note(f"{sm} GA: {len(queue)} starting genomes")

    def child() -> dict[str, str]:
        for _ in range(50):
            if len(pop) < 2 or G.rng.random() < 0.3:
                a = G.mutate(G.rng.choice(pop[:ELITE] or pop)[2])
            else:
                p, q = (min(G.rng.sample(pop, 2))[2] for _ in range(2))
                a = G.crossover(p, q) if G.rng.random() < 0.5 else G.mutate(G.crossover(p, q))
            if digest(a) not in seen and G.fits(a):
                return a
        return G.mutate(pop[0][2])
    size, mx = copy_of(model, "Q4_K").stat().st_size, Path("/sys/fs/cgroup/memory.max")
    ram = int(mx.read_text()) if mx.exists() and mx.read_text().strip().isdigit() else 64 * GIB
    # each llama.cpp run holds ~2.7 x the file in RAM besides the GPU copy: bounded by GPU memory and container RAM
    inflight = max(1, min(16, int(80 * GIB // (size + 4 * GIB)), int(0.8 * ram // (3 * size + 2 * GIB))))
    with ProcessPoolExecutor(inflight) as ex:
        running = {}

        def submit(a: dict[str, str]) -> None:
            seen.add(digest(a))
            running[ex.submit(_score_job, (BUDGET, model, a))] = a
        while queue and len(running) < inflight:
            submit(queue.pop(0))
        while running:
            fut = next(as_completed(running))
            a = running.pop(fut)
            n += 1
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001  (one failed genome never stops the search)
                note(f"{sm} GA: genome {n} failed ({e})")
                r = {"score": float("inf")}
            hist.append({"n": n, "digest": digest(a), **r})
            if r["score"] < float("inf"):
                scored.append((r["score"], a))
                pop = sorted(pop + [(r["score"], digest(a), a)])[:POP]
            if r["score"] < best:
                last_gain, best = (n if r["score"] < best - MIN_GAIN else last_gain), r["score"]
                out(f"budget-{BUDGET}", f"ga-{model}-live.json").write_text(json.dumps({"n": n, **r, "assignment": a}))
                note(f"{sm} GA: genome {n}: new best score {best:.2f} (flips {r['flips']:.2f}%, KL {r['kl']:.4f}, "
                     f"accuracy {r['accuracy']:.2f}, RAM {G.ram(a) / GIB:.3f} GiB)")
            elif n % 10 == 0:
                note(f"{sm} GA: {n} genomes, best score {best:.2f}")
            if n + len(running) < MAX_EVALS and (n < MIN_EVALS or n - last_gain < PATIENCE):
                if not queue and len(scored) >= 8:
                    try:
                        picks = [p for p in G.proposals(scored, SOLVED) if digest(p) not in seen]
                    except Exception as e:  # noqa: BLE001  (a failed fit only means GA children this round)
                        note(f"{sm} GA: solver proposals failed ({e})")
                        picks = []
                    queue += picks + [G.mutate(p) for p in picks[:2]]
                submit(queue.pop(0) if queue else child())
    lst.write_text(json.dumps({f"{sm}:ga{i}": {"assignment": a, "score": s, "ram_gib": G.ram(a) / GIB}
                               for i, (s, _, a) in enumerate(pop[:TOP], 1)}, indent=1))
    out(f"budget-{BUDGET}", f"ga-{model}{SUFFIX}.json").write_text(json.dumps(hist, indent=1))
    note(f"{sm} GA done: {n} genomes; best score {best:.2f}")


# --- the competitor and the final comparison -------------------------------------------------------------------

def preset(model: str) -> None:
    """llama.cpp's best preset that fits: of the two largest whose RAM fits the budget, the more accurate on the
    held-out test. If none fits, the smallest preset (it then has more RAM than we do)."""
    pj = out(f"budget-{BUDGET}", f"preset-{model}.json")
    if pj.exists() and json.loads(pj.read_text()).get("test"):
        return
    M, top = Model(model), BUDGET * (1 + WINDOW[1]) * GIB
    room = M.avg_bpw(BUDGET * (1 + WINDOW[1]))
    fit, over = [], []
    for p, b in sorted(PRESETS.items(), key=lambda x: -x[1]):
        if len(fit) == 2 or (b > room + 0.3 and p != "IQ1_S"):
            continue
        path = STORE / model / f"preset-{p}.gguf"
        if not path.exists():
            subprocess.run([str(BIN / "llama-quantize"), "--imatrix", str(STORE / model / f"{model}.imatrix.gguf"),
                            str(copy_of(model, "BF16")), f"{path}.part", p, str(THREADS)], check=True,
                           capture_output=True)
            os.rename(f"{path}.part", path)
        ram = sum(int(t.n_bytes) for t in tensors(path).values()) + M.runtime()
        (fit if ram <= top else over).append((p, path, ram))
        note(f"{short(model)} preset {p}: {ram / GIB:.3f} GiB ({'fits' if ram <= top else 'too big'})")
    fit = fit or [min(over, key=lambda x: x[2])]
    res = {}
    for p, path, ram in fit:
        with gpu(path.stat().st_size + 12 * GIB, 4 * 3600):
            r = test(model, path, f"preset:{p}")
        res[p] = {"ram_gib": ram / GIB, "accuracy": stats.acc(r["correct"]), "kl": r["kl"], "ppl": r["ppl"]}
    p = max(res, key=lambda x: (round(res[x]["accuracy"], 1), -res[x]["kl"]))
    pj.write_text(json.dumps({"preset": p, "over_budget": res[p]["ram_gib"] * GIB > top, "test": res[p],
                              "tried": res}, indent=1))
    for q, path, _ in over:
        if q != p:
            path.unlink(missing_ok=True)
    note(f"{short(model)} preset {p}: held-out accuracy {res[p]['accuracy']:.2f}, KL {res[p]['kl']:.4f}")


def final(model: str) -> int:
    """The GA's top 3, each distilled (the original as teacher), scored on all selection questions: the lowest
    score wins. Guard: not less accurate than the preset there. Then both on the held-out test, question by question."""
    from panther.distill import distill
    sm, bf16 = short(model), copy_of(model, "BF16")
    items = json.loads(out(f"budget-{BUDGET}", f"list-{model}{SUFFIX}.json").read_text())
    files, rows = {}, {}
    for k, it in items.items():
        files[k] = out(model, "tuned", f"{BUDGET}-{digest(it['assignment'])}.gguf")
        if not files[k].exists():
            src = build(model, it["assignment"])
            try:
                with gpu(2 * bf16.stat().st_size + 12 * GIB, 3 * 3600):  # generous: weights, logits, gradients
                    distill(src, bf16, [DATA / "splits/text" / f"answers-{model}.txt", DATA / "splits/calibration.txt"],
                            files[k], lambda m, k=k: note(f"{sm} {k} {m}"))
            except Exception as e:  # noqa: BLE001  (an architecture transformers cannot load from GGUF; never memory)
                if "out of memory" in str(e).lower():
                    raise
                note(f"{sm} {k}: distillation not possible ({type(e).__name__}: {str(e)[:120]}); scored as built")
                shutil.copy(src, files[k])
            finally:
                src.unlink(missing_ok=True)
        rows[k] = score(model, None, f"v2t:{digest(it['assignment'])}", path=files[k], full=True)
        plain = f"{k} undistilled"  # distillation can hurt near-lossless files: the undistilled version competes too
        files[plain] = build(model, it["assignment"], out(model, "tuned", f"{BUDGET}-{digest(it['assignment'])}-plain.gguf"))
        rows[plain] = score(model, None, f"v2:{digest(it['assignment'])}", path=files[plain], full=True)
        for name in (k, plain):
            note(f"{sm} final {name}: score {rows[name]['score']:.2f} (flips {rows[name]['flips']:.2f}%, KL "
                 f"{rows[name]['kl']:.4f}, accuracy {rows[name]['accuracy']:.2f}) on all selection questions")
    win = min(rows, key=lambda k: rows[k]["score"])
    item = items[win.split()[0]]
    label = f"{'v2' if win.endswith('undistilled') else 'v2t'}:{digest(item['assignment'])}"
    p = json.loads(out(f"budget-{BUDGET}", f"preset-{model}.json").read_text())["preset"]
    score(model, None, f"preset:{p}", path=STORE / model / f"preset-{p}.gguf", full=True)
    d, lo, hi = stats.compare(correct(model, label), correct(model, f"preset:{p}"))
    note(f"{sm} guard: {win} vs preset {p} on all selection questions: {d:+.2f} pts [{lo:+.2f}, {hi:+.2f}]")
    if hi < 0:
        note(f"{sm} guard: WORSE than the preset; not taken as our result")
        return 2
    with gpu(files[win].stat().st_size + 12 * GIB, 4 * 3600):
        ours = test(model, files[win], label)
        theirs, orig = test(model, STORE / model / f"preset-{p}.gguf", f"preset:{p}"), test(model, bf16, "original")
    d, lo, hi = stats.compare(ours["correct"], theirs["correct"])
    res = {"winner": win, "file": str(files[win]), "assignment": item["assignment"], "ram_gib": item["ram_gib"],
           "preset": p, "diff": d, "ci": [lo, hi], "verdict": stats.verdict(lo, hi)}
    for k, r in (("ours", ours), ("preset_test", theirs), ("original", orig)):
        res[k] = {"accuracy": stats.acc(r["correct"]), "kl": r["kl"], "ppl": r["ppl"],
                  "flips": stats.flips(r["correct"], orig["correct"]),
                  "per_task": {t: 100 * float(np.mean(v)) for t, v in r["correct"].items()}}
    out(f"budget-{BUDGET}", f"winner-{model}{SUFFIX}.json").write_text(json.dumps(res, indent=1))
    note(f"{sm} held-out test vs preset {p}: {res['ours']['accuracy']:.2f} vs {res['preset_test']['accuracy']:.2f} "
         f"(original {res['original']['accuracy']:.2f}) | our winner {d:+.2f} pts [{lo:+.2f}, {hi:+.2f}] "
         f"{res['verdict']}, flips {res['ours']['flips']:.2f} vs {res['preset_test']['flips']:.2f}%")
    return 0


# --- the Pi -----------------------------------------------------------------------------------------------------

def bundle(model: str) -> None:
    w = json.loads(out(f"budget-{BUDGET}", f"winner-{model}{SUFFIX}.json").read_text())
    d = DATA / "pi_bundle" / f"budget-{BUDGET}-{short(model).lower()}{SUFFIX}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{model.lower()}-b{BUDGET}{SUFFIX}.gguf"
    shutil.copy(w["file"], f)
    h = sha256()
    with open(f, "rb") as fh:
        while c := fh.read(1 << 24):
            h.update(c)
    (d / "manifest.json").write_text(json.dumps({"file": f.name, "sha256": h.hexdigest(), "budget_gib": BUDGET,
                                                 "predicted_ram_gib": w["ram_gib"], "context": CFG["context_tokens"],
                                                 "kv_cache_type": CFG["kv_cache_type"]}, indent=1))


def upload(model: str) -> None:
    from huggingface_hub import HfApi
    repo = os.environ["PANTHER_HF_REPO"].format(size=short(model).lower(), budget=BUDGET) + SUFFIX
    HfApi().create_repo(repo, private=True, exist_ok=True)
    HfApi().upload_folder(repo_id=repo, folder_path=DATA / "pi_bundle" / f"budget-{BUDGET}-{short(model).lower()}{SUFFIX}")
    note(f"{short(model)} Pi bundle uploaded")


STEPS = {"store": store, "error": error, "refs": refs, "preset": preset, "ga": ga, "final": final, "bundle": bundle}


def main() -> int:
    global BUDGET
    cmd, b, model = sys.argv[1:4]
    BUDGET = int(b)
    if cmd in ("fetch", "upload"):
        (fetch if cmd == "fetch" else upload)(model)
        return 0
    for s in STEPS if cmd == "run" else [cmd]:
        t0 = time.time()
        note(f"{short(model)} step {s}: start")
        with lock(f"{model}-{s}" if s in ("store", "error", "refs") else f"{model}-{BUDGET}-{s}"):
            if STEPS[s](model) == 2:
                return 2
        note(f"{short(model)} step {s}: done in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        note(f"FAILED: {' '.join(sys.argv[1:])}: {type(e).__name__}: {e}")
        raise
