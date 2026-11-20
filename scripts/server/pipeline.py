"""The Panther pipeline (README: "How it works"). A contest is one model at one RAM budget (GiB).

  pipeline.py fetch BUDGET MODEL     # CPU container (has network): the weights at the pinned revision
  pipeline.py prompts BUDGET MODEL   # CPU container: the dataset's chat prompts, then the check of every split
  pipeline.py run BUDGET MODEL       # GPU container: store, error, refs, preset, ga, final, bundle in order
  pipeline.py STEP BUDGET MODEL      # one of those steps; every step skips work already on disk
  pipeline.py upload BUDGET MODEL    # CPU container: the Pi bundle to Hugging Face
  pipeline.py clean BUDGET MODEL     # the end of a contest: the type copies and reference outputs removed
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np

from panther import llama, stats
from panther.llama import DATA, THREADS
from panther.model import CFG, GIB, MODELS, Model, bpw
from panther.solver import solve

RW, STORE, BIN = DATA / "rewrite", DATA / "store", llama.BIN
TASKS = ["mmlu", "hellaswag", "arc_challenge", "winogrande"]
FIT = {"mmlu": 300, "arc_challenge": 200, "hellaswag": 200, "winogrande": 200}  # the GA's selection questions
FIRST = {t: v // 3 for t, v in FIT.items()}  # a first look: a genome clearly worse here is not finished
ALL = dict.fromkeys(TASKS, 0)  # every selection question (llama.cpp's n = 0)
DROP_MARGIN = 3.0  # dropped only if its first-look score is this much worse than the population's worst (~2 SE)
WINDOW = (-0.01, 0.005)  # a genome's RAM must be within -1% / +0.5% of the budget
POP, ELITE, MAX_EVALS, MIN_EVALS, PATIENCE, TOP, SOLVED = 16, 4, 70, 30, 15, 3, 4  # an improvement: beyond 1 SE
KL_WEIGHT = 10  # score = flips % + 10 x KL on the select text (lower is better)
SUFFIX = "-v2"
FTYPE = {"Q3_K": "Q3_K_S", "Q4_K": "Q4_K_S", "Q5_K": "Q5_K_S", "IQ2_S": "IQ2_M"}  # --pure: one type everywhere
PRESETS = {"IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7, "Q2_K_S": 2.9,
           "IQ3_XXS": 3.06, "Q2_K": 3.2, "IQ3_XS": 3.3, "IQ3_S": 3.44, "Q3_K_S": 3.5, "IQ3_M": 3.66, "Q3_K_M": 3.9,
           "IQ4_XS": 4.25, "Q3_K_L": 4.27, "Q4_K_S": 4.58, "Q4_K_M": 4.89, "Q5_K_S": 5.54, "Q5_K_M": 5.69,
           "Q6_K": 6.56, "Q8_0": 8.5}
BUDGET = 0
# The dataset (README "The dataset"): train / select / test, each a web part (the existing FineWeb-Edu files) and a
# chat part (the original's own answers to general prompts, per model). KL and perplexity are measured only on it;
# nothing trains on anything but train.
SPLITS = ("train", "select", "test")
WEB = dict(zip(SPLITS, (DATA / "splits/calibration.txt", DATA / "splits/text/selection.txt",
                        DATA / "splits/text/test_web.txt")))
# Every stage reads the whole of its split. select is read by every genome, so it is kept small: the chat part and
# whole web documents in a fixed random order up to SELECT_TOKENS in all. Its name carries the definition, so a
# select text, reference and cached KL records from another definition are never reused
SELECT_TOKENS = 49 * 2048
REF = {"select": "select100k", "test": "test"}  # each scored split's joined text and .kld reference
# The chat prompts: English first messages of OpenAssistant oasst1 (Apache-2.0), a fixed draw, none overlapping a
# benchmark question
PROMPTS = {"repo": "OpenAssistant/oasst1", "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
           "file": "data/train-00000-of-00001-b42a775f407cee45.parquet",
           "n": {"train": 300, "select": 150, "test": 150}}
PROMPT_FILE = DATA / "splits/text/prompts.json"


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


def imatrix(model: str) -> Path:
    return STORE / model / f"{model}.imatrix.gguf"


def preset_file(model: str, p: str) -> Path:
    return STORE / model / f"preset-{p}.gguf"


def bundle_dir(model: str, tag: str) -> Path:
    return DATA / "pi_bundle" / f"budget-{BUDGET}-{short(model).lower()}{tag}"


def chat_text(model: str, split: str) -> Path:
    """A split's chat part: the original's own answers to that split's prompts, in its chat template (made by refs)."""
    return DATA / "splits/text" / f"chat-{model}-{split}.txt"


def joined_text(model: str, split: str) -> Path:
    """A split's chat and web parts in one file (llama-perplexity reads one); its stem is the ppl run's text name."""
    return DATA / "splits/text" / f"{REF[split]}-{model}.txt"


def record(kind: str, model: str, suffix: str = "") -> Path:
    """A contest's result file: rewrite/budget-<BUDGET>/<kind>-<model><suffix>.json."""
    return out(f"budget-{BUDGET}", f"{kind}-{model}{suffix}.json")


def load(kind: str, model: str, suffix: str = "") -> dict:
    return json.loads(record(kind, model, suffix).read_text())


def digest(a: dict[str, str]) -> str:
    return sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()[:12]


def digest_file(f: Path) -> str:
    h = sha256()
    with open(f, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def label_of(winner: str, a: dict[str, str]) -> str:
    """The results label of a contest's winner: v2 for an undistilled file, v2t for a distilled one."""
    return f"{'v2' if winner.endswith('undistilled') else 'v2t'}:{digest(a)}"


def step_lock(model: str, s: str) -> str:
    """A step's lock name: per model for the steps every budget shares, per contest for the rest."""
    return f"{model}-{s}" if s in ("store", "error", "refs") else f"{model}-{BUDGET}-{s}"


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

def margin() -> int:
    """Disk that downloads and type copies leave free: 100 GB while other jobs use the GPU (their genome files
    need room), 20 GB otherwise."""
    try:
        busy = bool(json.loads((DATA / "locks/gpu.json").read_text()))
    except (OSError, ValueError):
        busy = False
    return (100 if busy else 20) * GIB


def wait_disk(need: int, what: str) -> None:
    while shutil.disk_usage(DATA).free < need + margin():
        note(f"{what}: waiting for {(need + margin()) / GIB:.0f} GB of free disk")
        time.sleep(600)


def store_repo(model: str) -> str:
    return os.environ["PANTHER_HF_REPO"].split("/")[0] + f"/panther-store-{model}"


def stored(model: str) -> bool:
    done = STORE / model / "rounded.json"
    return (all(copy_of(model, t).exists() for t in ["BF16", *CFG["formats"]]) and done.exists()
            and set(CFG["formats"]) <= set(json.loads(done.read_text())))


def fetch(model: str) -> None:
    """The type copies saved by an earlier run (public Hugging Face repo), else the weights if the 16-bit GGUF
    is missing (to convert)."""
    if stored(model):
        return
    n = sum(x.rows * x.cols for x in Model(model).matrices)
    have = sum(f.stat().st_size for f in (STORE / model).glob("*.gguf")) if (STORE / model).exists() else 0
    need = max(0, int(n * 8.5) - have) + margin()  # 16-bit original + 14 type copies
    if shutil.disk_usage(DATA).free < need:  # not enough yet: give the turn to another model, try again later
        note(f"{short(model)} fetch: needs {need / GIB:.0f} GB of free disk; letting others go first")
        sys.exit(3)
    if subprocess.run([os.environ["PANTHER_HF"], "download", store_repo(model), "--local-dir", str(STORE / model)]
                      ).returncode == 0 and stored(model):
        note(f"{short(model)} fetch: type copies downloaded from {store_repo(model)}")
        return
    hf = DATA / "hf" / model
    if copy_of(model, "BF16").exists():
        return
    m = MODELS[model]
    subprocess.run([os.environ["PANTHER_HF"], "download", m["repo"], "--revision", m["revision"], "--local-dir",
                    str(hf)], check=True)  # resumes a cut-off download, skips files already there


def save(model: str) -> None:
    """The finished store (16-bit original, importance matrix, presets, type copies) to a public Hugging Face repo,
    so later runs download it instead of rebuilding. Public: the models are openly licensed."""
    from huggingface_hub import HfApi
    if not stored(model):
        return
    api, repo = HfApi(), store_repo(model)
    api.create_repo(repo, private=False, exist_ok=True)
    have = set(api.list_repo_files(repo))
    if all(f.name in have for f in (STORE / model).iterdir()):
        return
    api.upload_folder(repo_id=repo, folder_path=STORE / model, allow_patterns=["*.gguf", "rounded.json"])
    note(f"{short(model)} type copies saved to {repo}")


def store(model: str, types: list[str] | None = None) -> None:
    """The 16-bit original, its importance matrix, and the whole model once in every type (or only `types`)."""
    types = types or CFG["formats"]
    d, bf16, imat = STORE / model, copy_of(model, "BF16"), imatrix(model)
    hf = DATA / "hf" / model
    d.mkdir(parents=True, exist_ok=True)
    if not bf16.exists():
        subprocess.run([os.environ.get("PANTHER_CONVERT_PY", sys.executable), "/opt/llama.cpp/convert_hf_to_gguf.py",
                        str(hf), "--outtype", "bf16", "--outfile", f"{bf16}.part"], check=True)
        os.rename(f"{bf16}.part", bf16)
    if hf.exists():
        shutil.rmtree(hf)  # rounding reads the 16-bit GGUF: the original from here on
    if not imat.exists():
        po = ["--process-output"] if MODELS[model].get("imatrix_process_output") else []
        with gpu(bf16.stat().st_size + 4 * GIB):  # the whole 16-bit model on the GPU: booked like every other job
            subprocess.run([str(BIN / "llama-imatrix"), "-m", str(bf16), "-f", str(WEB["train"]),
                            "-o", f"{imat}.part.gguf", "-c", "512", "-ngl", "99", "-t", str(THREADS)] + po, check=True)
        os.rename(f"{imat}.part.gguf", imat)
    M = Model(model)
    al, n = M.allowed(), sum(m.rows * m.cols for m in M.matrices)
    done = d / "rounded.json"  # types whose copy has error-compensated codes (panther.rounding)
    for t in types:
        if not copy_of(model, t).exists():  # embedding / head take Q8_0 where the type needs imatrix data they lack
            wait_disk(int(n * bpw(t) / 8), f"{short(model)} store, {t} copy")
            emb, head = ((t if t in al.get(n, al["token_embd.weight"]) else "Q8_0").lower()
                         for n in ("token_embd.weight", "output.weight"))
            from panther.rounding import blank  # layer matrices come from the rounding below, quantized once
            blank(bf16, imat, t, FTYPE.get(t, t), emb, head, copy_of(model, t))
    have = json.loads(done.read_text()) if done.exists() else []
    if todo := [t for t in types if t not in have]:
        from panther.rounding import round_copies
        h, i = M.matrices[2].cols, max(x.cols for x in M.matrices)
        per_layer = 4 * (3 * h * h + i * i)  # bytes of the four input Hessians of one layer
        book = bf16.stat().st_size + 44 * GIB  # the model, 20 GiB of Hessians, several types rounded at once
        with gpu(book, 6 * 3600):
            round_copies(None, bf16, {t: copy_of(model, t) for t in todo},
                         WEB["train"], max(1, int(20 * GIB // per_layer)), book)
        done.write_text(json.dumps(have + todo))


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
            tensors(imatrix(model)).items()}
    jobs = []
    for name, types in Model(model).allowed().items():
        if todo := [t for t in types if t not in done.get(name, {})]:
            cnt = max(float(imat.get(f"{name}.counts", np.ones(1))[0]), 1.0)
            jobs.append((model, name, todo, imat.get(f"{name}.in_sum2", np.ones(1 << 16)) / cnt))
    with ProcessPoolExecutor(8) as ex:
        for name, res in ex.map(_error_one, jobs):
            done.setdefault(name, {}).update(res)
    dst.write_text(json.dumps(done))


def select_docs(docs: list[str], first: int, count, limit: int = SELECT_TOKENS) -> list[str]:
    """The select split's web part: whole documents in a fixed random order while the total (first + theirs) stays
    within limit; the rest are used by nothing."""
    keep = []
    for i in np.random.default_rng(0).permutation(len(docs)):
        if (first := first + count(docs[i])) > limit:
            break
        keep.append(docs[i])
    return keep


def select_text(model: str, bf16: Path) -> str:
    """The select split: the original's answers to the 150 select prompts and whole web documents of the web select
    text, up to SELECT_TOKENS tokens in all (counted with the model's tokenizer)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(bf16.parent), gguf_file=bf16.name)
    count = lambda x: len(tok(x + "\n\n", add_special_tokens=False).input_ids)  # noqa: E731
    docs, ids = WEB["select"].read_text().split("\n\n"), json.loads(WEB["select"].with_suffix(".ids.json").read_text())
    if len(docs) != len(ids):  # documents were joined by a blank line; one with a blank line inside splits in two
        note(f"{short(model)} select text: {len(docs)} pieces for {len(ids)} documents; pieces taken as documents")
    chat = chat_text(model, "select").read_text()
    web = select_docs(docs, count(chat), count)
    text = chat + "\n" + "\n\n".join(web)
    n = len(tok(text).input_ids)
    note(f"{short(model)} select text: {len(web)} of {len(docs)} web documents, {n} tokens, {n // 2048} chunks read "
         f"({n % 2048} tokens in the last partial chunk not read)")
    return text


def refs(model: str) -> None:
    """The original's reference outputs: its chat answers to the prompts of every split (the dataset's chat part),
    its next-word probabilities on the select and test text (chat + web), and which selection and held-out questions
    it gets right (to count flips)."""
    ref, bf16 = DATA / "reference" / model, copy_of(model, "BF16")
    ref.mkdir(parents=True, exist_ok=True)
    with gpu(bf16.stat().st_size + 12 * GIB, 4 * 3600):
        if not all(chat_text(model, sp).exists() for sp in SPLITS):  # each model in its own chat template
            ps = json.loads(PROMPT_FILE.read_text())["prompts"]
            outs = llama.complete(bf16, [x["text"] for x in ps], chat=True)
            for sp in SPLITS:
                chat_text(model, sp).write_text("\n\n".join(o for o, x in zip(outs, ps) if x["split"] == sp) + "\n")
        for sp, name in REF.items():
            if not (j := joined_text(model, sp)).exists():
                j.write_text(select_text(model, bf16) if sp == "select" else
                             chat_text(model, sp).read_text() + "\n" + WEB[sp].read_text())
            if not (ref / f"{name}.kld").exists():  # whole 2048-token chunks: a last partial one is not read
                subprocess.run([str(BIN / "llama-perplexity"), "-m", str(bf16), "-f", str(j), "-c", "2048", "-b",
                                "2048", "-ngl", "99", "-t", str(THREADS), "--kl-divergence-base",
                                f"{ref}/{name}.part"], check=True, capture_output=True)
                os.rename(f"{ref}/{name}.part", ref / f"{name}.kld")
        for t in TASKS:
            for n in (FIT, FIRST, ALL):
                llama.run("task", model, bf16, "original", task=t, split="selection", n=n[t])
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


def correct(model: str, label: str, n: dict = ALL) -> dict[str, np.ndarray]:
    return {t: np.array(llama.find(model, label, task=t, split="selection", n=n[t])["correct"]) for t in TASKS}


def test(model: str, path: Path, label: str) -> dict:
    """The held-out test: 4 multiple-choice tasks, GSM8K (test split), KL and perplexity on the test text."""
    c = {t: np.array(llama.run("task", model, path, label, task=t, split="poc_test", n=0)["correct"]) for t in TASKS}
    c["gsm8k"] = llama.gsm8k(model, path, label, DATA / "splits/llama_tasks/poc_test/gsm8k.jsonl")
    kl = None if label == "original" else llama.run("kl", model, path, label, reference=REF["test"])["metrics"]
    return {"correct": c, "kl": kl and kl["kl_mean"], "kl_se": kl and kl["kl_se"], "same_top": kl and kl["top1_pct"],
            "ppl": llama.run("ppl", model, path, label, text=joined_text(model, "test").stem)["metrics"]["ppl"]}


def score(model: str, a: dict[str, str] | None, label: str, path: Path | None = None, full: bool = False,
          cut: float | None = None) -> dict:
    """Flips against the original (GA questions, or all selection questions if full) and KL on the select text.
    With cut: a first look on a third of the questions; a genome already worse than cut there is dropped
    ("dropped": True, its score from that third) instead of finishing."""
    n = ALL if full else FIT
    built = path is None and not (llama.find(model, label, reference=REF["select"]) and all(
        llama.find(model, label, task=t, split="selection", n=n[t]) for t in TASKS))

    def result(n: dict) -> dict:
        c = correct(model, label, n)
        f = stats.flips(c, correct(model, "original", n))
        return {"score": f + KL_WEIGHT * kl, "flips": f, "kl": kl, "kl_se": se, "same_top": top1,
                "accuracy": stats.acc(c)}
    try:
        path = path or (build(model, a) if built else Path("/dev/null"))
        with gpu(int(1.1 * path.stat().st_size) + 4 * GIB, 1800):
            m = llama.run("kl", model, path, label, reference=REF["select"])["metrics"]
            kl, se, top1 = m["kl_mean"], m["kl_se"], m["top1_pct"]  # same top word %: reported, not scored
            if cut is not None:
                for t in TASKS:
                    llama.run("task", model, path, label, task=t, split="selection", n=FIRST[t])
                if (first := result(FIRST))["score"] > cut:
                    return {**first, "dropped": True}
            for t in TASKS:
                llama.run("task", model, path, label, task=t, split="selection", n=n[t])
    finally:
        if built and path:
            path.unlink(missing_ok=True)
    return result(n)


def relative(r: dict, base: float) -> dict:
    """With the score as a share of the preset's on the same questions and text (100: the preset's damage, 0: none)."""
    return {**r, "relative": 100 * r["score"] / base}


def versus(a: dict, b: dict, orig: dict[str, np.ndarray], w: float = KL_WEIGHT) -> dict:
    """a against b (each with "correct", "kl", "kl_se"; the same questions and text): a's relative score (100 = b's
    damage) and its 95% interval, from the same SE as the GA's. diff and ci are that minus 100 (negative = better
    than b), so the dashboard draws them around 0 as before. w: the KL weight."""
    rel, lo, hi = stats.relative(a["correct"], b["correct"], orig, a["kl"], a["kl_se"], b["kl"], b["kl_se"], w)
    return {"relative": rel, "relative_ci": [lo, hi], "diff": rel - 100, "ci": [lo - 100, hi - 100],
            "verdict": stats.verdict(lo, hi), "unit": "relative score - 100"}


def beside(a: dict, b: dict, orig: dict) -> dict:
    """Reported next to a's verdict against b, deciding nothing: each one's held-out accuracy as % of the original's
    (accuracy_kept), and the verdict again with the KL weight at 5 and 20 (weight_check), from the same numbers."""
    o = stats.acc(orig["correct"])
    kept = {"ours": 100 * stats.acc(a["correct"]) / o, "preset": 100 * stats.acc(b["correct"]) / o}
    wc = {str(w): {"relative": v["relative"], "ci": v["relative_ci"], "verdict": v["verdict"]}
          for w in (5, 20) for v in [versus(a, b, orig["correct"], w)]}
    return {"accuracy_kept": kept, "weight_check": wc}


def beside_text(c: dict, verdict: str) -> str:
    """The log line's tail for beside()'s numbers."""
    k, w = c["accuracy_kept"], c["weight_check"]
    same = all(x["verdict"] == verdict for x in w.values())
    return (f"; accuracy kept {k['ours']:.1f} vs {k['preset']:.1f}% of the original; "
            + ("same verdict at KL weight 5 and 20" if same
               else f"verdict at KL weight 5: {w['5']['verdict']}, 20: {w['20']['verdict']}"))


def _score_job(job):
    global BUDGET
    BUDGET, model, a, cut = job
    return score(model, a, f"v2:{digest(a)}", cut=cut)


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


def preset_layout(model: str) -> dict[str, str] | None:
    """The competing preset's types, each mapped to the nearest type the genome may use."""
    steps = Genes(model, BUDGET).steps
    have = {n: t.tensor_type.name for n, t in tensors(preset_file(model, load("preset", model)["preset"])).items()}
    return {g: have[g] if have[g] in ts else min(ts, key=lambda x: (abs(bpw(x) - bpw(have[g])), -bpw(x)))
            for g, ts in steps.items()}


def ga(model: str) -> None:
    """Steady state: as many genomes scored at once as the GPU holds (up to 16, as GPU memory and the container's RAM allow), a new child whenever one finishes; each round the solver
    adds SOLVED proposals from a fit of every score so far. Stops on a plateau. Writes the TOP best."""
    if (lst := record("list", model, SUFFIX)).exists():
        return
    for f in out(model, "tmp", "x").parent.glob("*.gguf*"):  # left by a killed run
        if time.time() - f.stat().st_mtime > 3600:
            f.unlink()
    G, sm = Genes(model, BUDGET), short(model)
    base = load("preset", model)["score_fit"]  # the preset's score on the GA questions: relative scores
    if not all(llama.find(model, "original", task=t, split="selection", n=FIRST[t]) for t in TASKS):
        with lock(f"{model}-first"), gpu(copy_of(model, "BF16").stat().st_size + 8 * GIB, 3600):  # first look
            for t in TASKS:
                llama.run("task", model, copy_of(model, "BF16"), "original", task=t, split="selection", n=FIRST[t])
    queue, pop, seen, scored, hist = G.initial(preset_layout(model)), [], set(), [], []
    best, last_gain, n, dropped, top, se = float("inf"), 0, 0, 0, None, 0.0
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
            cut = pop[-1][0] + DROP_MARGIN if len(pop) >= POP else None  # a full population: weak ones dropped early
            running[ex.submit(_score_job, (BUDGET, model, a, cut))] = a
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
            r = relative(r, base)
            hist.append({"n": n, "digest": digest(a), **r})
            if r.get("dropped"):
                dropped += 1
            elif r["score"] < float("inf"):
                scored.append((r["score"], a))
                pop = sorted(pop + [(r["score"], digest(a), a)])[:POP]
            if r["score"] < best and not r.get("dropped"):
                if top:  # an improvement only beyond 1 SE of the difference (flips paired on the GA questions)
                    se = stats.score_se(correct(model, f"v2:{digest(a)}", FIT), correct(model, top[0], FIT),
                                        correct(model, "original", FIT), r["kl_se"], top[1], KL_WEIGHT)
                last_gain = n if best - r["score"] > se else last_gain
                best, top = r["score"], (f"v2:{digest(a)}", r["kl_se"])
                record("ga", model, "-live").write_text(json.dumps({"n": n, **r, "assignment": a}))
                note(f"{sm} GA: genome {n}: new best score {best:.2f} (flips {r['flips']:.2f}%, KL {r['kl']:.4f}, "
                     f"accuracy {r['accuracy']:.2f}, RAM {G.ram(a) / GIB:.3f} GiB); relative {r['relative']:.1f}")
            elif n % 10 == 0:
                note(f"{sm} GA: {n} genomes, best score {best:.2f} ({dropped} dropped after a first look)")
            if n + len(running) < MAX_EVALS and (n < MIN_EVALS or n - last_gain < PATIENCE):
                if not queue and len(scored) >= 8:
                    try:
                        picks = [p for p in G.proposals(scored, SOLVED) if digest(p) not in seen]
                    except Exception as e:  # noqa: BLE001  (a failed fit only means GA children this round)
                        note(f"{sm} GA: solver proposals failed ({e})")
                        picks = []
                    queue += picks + [G.mutate(p) for p in picks[:2]]
                submit(queue.pop(0) if queue else child())
    lst.write_text(json.dumps({f"{sm}:ga{i}": {"assignment": a, "score": s, "relative": 100 * s / base,
                                              "ram_gib": G.ram(a) / GIB}
                               for i, (s, _, a) in enumerate(pop[:TOP], 1)}, indent=1))
    record("ga", model, SUFFIX).write_text(json.dumps(hist, indent=1))
    why = (f"{n - last_gain} genomes without an improvement beyond 1 SE (best {best:.2f}, SE {se:.2f})"
           if n - last_gain >= PATIENCE else f"{MAX_EVALS} genomes, best score {best:.2f}")
    note(f"{sm} GA done: {n} genomes ({dropped} dropped after a first look); {why}")


# --- the competitor and the final comparison -------------------------------------------------------------------

def preset(model: str) -> None:
    """llama.cpp's best preset that fits: of every preset whose RAM fits the budget, the most accurate on all
    selection questions (ties: lower KL on the select text); only it takes the held-out test. If none fits, the
    smallest preset (it then has more RAM than we do). Also its score on the GA's questions: the GA's relative base."""
    pj = record("preset", model)
    if pj.exists() and "score_fit" in json.loads(pj.read_text()):  # a record from before decision 7 is redone
        return
    M, top = Model(model), BUDGET * (1 + WINDOW[1]) * GIB
    room = M.avg_bpw(BUDGET * (1 + WINDOW[1]))
    fit, over = [], []
    for p, b in sorted(PRESETS.items(), key=lambda x: -x[1]):
        if b > room + 0.3 and p != "IQ1_S":  # clearly too big (nominal bits per weight); IQ1_S is the fallback
            continue
        path = preset_file(model, p)
        if not path.exists():
            subprocess.run([str(BIN / "llama-quantize"), "--imatrix", str(imatrix(model)),
                            str(copy_of(model, "BF16")), f"{path}.part", p, str(THREADS)], check=True,
                           capture_output=True)
            os.rename(f"{path}.part", path)
        ram = sum(int(t.n_bytes) for t in tensors(path).values()) + M.runtime()
        (fit if ram <= top else over).append((p, path, ram))
        note(f"{short(model)} preset {p}: {ram / GIB:.3f} GiB ({'fits' if ram <= top else 'too big'})")
    fit = fit or [min(over, key=lambda x: x[2])]
    res = {p: {"ram_gib": ram / GIB, **score(model, None, f"preset:{p}", path=path, full=True)} for p, path, ram in fit}
    p = max(res, key=lambda x: (round(res[x]["accuracy"], 1), -res[x]["kl"]))
    with gpu(preset_file(model, p).stat().st_size + 12 * GIB, 4 * 3600):
        r = test(model, preset_file(model, p), f"preset:{p}")
    t = {"ram_gib": res[p]["ram_gib"], "accuracy": stats.acc(r["correct"]), "kl": r["kl"], "ppl": r["ppl"]}
    fs = score(model, None, f"preset:{p}", path=preset_file(model, p))["score"]  # the base of the GA's relative score
    pj.write_text(json.dumps({"preset": p, "over_budget": t["ram_gib"] * GIB > top, "test": t, "score_fit": fs,
                              "tried": res}, indent=1))
    for q, path, _ in fit + over:
        if q != p:
            path.unlink(missing_ok=True)
    note(f"{short(model)} preset {p}: held-out accuracy {t['accuracy']:.2f}, KL {t['kl']:.4f} (selection accuracy "
         f"{res[p]['accuracy']:.2f}, best of {len(res)})")


def summary(r: dict, orig: dict) -> dict:
    """A held-out test's numbers as the result files store them (flips against the original's answers)."""
    return {"accuracy": stats.acc(r["correct"]), "kl": r["kl"], "same_top": r.get("same_top"), "ppl": r["ppl"],
            "flips": stats.flips(r["correct"], orig["correct"]),
            "per_task": {t: 100 * float(np.mean(v)) for t, v in r["correct"].items()}}


def final(model: str) -> int:
    """The GA's top 3, each distilled (the original as teacher, on the train text: nothing the finalists are ranked
    or tested on) and undistilled, scored on all selection questions and the select text: the lowest score wins.
    Guard: a warning if clearly worse than the preset there (relative score). Then both on the held-out test: the
    verdict is our relative score there (flips on the held-out questions, KL on the test text) and its interval.
    Overlapped: one distillation at a time (it needs most of the GPU) while everything already built is scored beside
    it, and the held-out tests run side by side."""
    from panther.distill import distill
    sm, bf16 = short(model), copy_of(model, "BF16")
    items, p = load("list", model, SUFFIX), load("preset", model)["preset"]
    pf, pl, files, rows, one = preset_file(model, p), f"preset:{p}", {}, {}, threading.Lock()
    base = load("preset", model)["tried"][p]["score"]  # the preset's score on all selection questions (preset step)

    def scored(name: str, label: str, path: Path) -> None:
        rows[name] = r = relative(score(model, None, label, path=path, full=True), base)
        note(f"{sm} final {name}: score {r['score']:.2f} (flips {r['flips']:.2f}%, KL {r['kl']:.4f}, accuracy "
             f"{r['accuracy']:.2f}) on all selection questions; relative {r['relative']:.1f}; same top word "
             f"{r['same_top']:.2f}%")

    def finalist(k: str, it: dict, ex: ThreadPoolExecutor) -> list:
        dg, plain = digest(it["assignment"]), f"{k} undistilled"  # distillation can hurt near-lossless files
        files[plain] = build(model, it["assignment"], out(model, "tuned", f"{BUDGET}-{dg}-plain.gguf"))
        jobs, files[k] = [ex.submit(scored, plain, f"v2:{dg}", files[plain])], out(model, "tuned", f"{BUDGET}-{dg}.gguf")
        with one:
            if not files[k].exists():
                try:
                    with gpu(bf16.stat().st_size + 20 * GIB, 3 * 3600):  # student weights + activations
                        distill(files[plain], bf16, [chat_text(model, "train"), WEB["train"]],
                                files[k], lambda m: note(f"{sm} {k} {m}"))
                except Exception as e:  # noqa: BLE001  (an architecture transformers cannot load from GGUF)
                    if "out of memory" in str(e).lower():
                        raise
                    note(f"{sm} {k}: distillation not possible ({type(e).__name__}: {str(e)[:120]}); scored as built")
                    shutil.copy(files[plain], files[k])
        return jobs + [ex.submit(scored, k, f"v2t:{dg}", files[k])]
    with ThreadPoolExecutor(8) as ex, ThreadPoolExecutor(len(items)) as fx:
        jobs = [ex.submit(score, model, None, pl, path=pf, full=True)]
        for f in [fx.submit(finalist, k, it, ex) for k, it in items.items()]:
            jobs += f.result()
        for f in jobs:
            f.result()
    win = min(rows, key=lambda k: rows[k]["score"])
    item = items[win.split()[0]]
    label = label_of(win, item["assignment"])
    g = versus({**rows[win], "correct": correct(model, label)}, {**jobs[0].result(), "correct": correct(model, pl)},
               correct(model, "original"))  # the guard: the same rule as the verdict, on selection data
    note(f"{sm} guard: {win} vs preset {p} on all selection questions: relative {g['relative']:.1f} "
         f"[{g['relative_ci'][0]:.1f}, {g['relative_ci'][1]:.1f}]")
    if warn := g["verdict"] == "LOSE":  # reported, not a stop: a contest that would lose still gets its verdict
        note(f"{sm} guard: WORSE than the preset on selection questions; held-out test run anyway")

    def held(path: Path, lab: str) -> dict:
        with gpu(path.stat().st_size + 12 * GIB, 4 * 3600):
            return test(model, path, lab)
    with ThreadPoolExecutor(2) as ex:  # side by side; the original's held-out results were recorded by refs
        a, b = ex.submit(held, files[win], label), ex.submit(held, pf, pl)
        orig = test(model, bf16, "original")
        ours, theirs = a.result(), b.result()
    # The verdict: our relative score on held-out data (relative, relative_ci; diff, ci = those - 100, negative =
    # better than the preset); the accuracy difference (accuracy_diff) is reported, not judged
    v = versus(ours, theirs, orig["correct"])
    res = {"winner": win, "file": str(files[win]), "assignment": item["assignment"], "ram_gib": item["ram_gib"],
           "preset": p, **v, **beside(ours, theirs, orig),
           "accuracy_diff": stats.acc(ours["correct"]) - stats.acc(theirs["correct"]),
           "selection": rows[win], "preset_score": base, "guard": {**g, "warning": warn}}
    for k, r in (("ours", ours), ("preset_test", theirs), ("original", orig)):
        res[k] = summary(r, orig)
    record("winner", model, SUFFIX).write_text(json.dumps(res, indent=1))
    for f in files.values():  # only the winner's file is kept
        if f != files[win]:
            f.unlink(missing_ok=True)
    note(f"{sm} held-out test vs preset {p}: {res['ours']['accuracy']:.2f} vs {res['preset_test']['accuracy']:.2f} "
         f"(original {res['original']['accuracy']:.2f}, ours - preset {res['accuracy_diff']:+.2f} pts) | our winner "
         f"{v['diff']:+.2f} rel [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] {v['verdict']} (relative score "
         f"{v['relative']:.1f}, 100 = the preset), flips {res['ours']['flips']:.2f} vs "
         f"{res['preset_test']['flips']:.2f}%, same top word "
         f"{res['ours']['same_top']:.2f} vs {res['preset_test']['same_top']:.2f}%{beside_text(res, v['verdict'])}")
    return 0


# --- the Pi -----------------------------------------------------------------------------------------------------

def _pack(model: str, src: Path, tag: str, extra: dict | None = None) -> Path:
    """A Pi bundle: the file and a manifest with its hash, budget and predicted RAM (the winner's: same types)."""
    d = bundle_dir(model, tag)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{model.lower()}-b{BUDGET}{tag}.gguf"
    shutil.copy(src, f)
    (d / "manifest.json").write_text(json.dumps({"file": f.name, "sha256": digest_file(f), "budget_gib": BUDGET,
                                                 "predicted_ram_gib": load("winner", model, SUFFIX)["ram_gib"],
                                                 "context": CFG["context_tokens"],
                                                 "kv_cache_type": CFG["kv_cache_type"], **(extra or {})}, indent=1))
    return d


def _push(model: str, tag: str, d: Path) -> str:
    from huggingface_hub import HfApi
    repo = os.environ["PANTHER_HF_REPO"].format(size=short(model).lower(), budget=BUDGET) + tag
    HfApi().create_repo(repo, private=False, exist_ok=True)
    HfApi().upload_folder(repo_id=repo, folder_path=d)
    return repo


def bundle(model: str) -> None:
    _pack(model, Path(load("winner", model, SUFFIX)["file"]), SUFFIX)


def upload(model: str) -> None:
    _push(model, SUFFIX, bundle_dir(model, SUFFIX))
    note(f"{short(model)} Pi bundle uploaded")


def qatupload(model: str) -> None:
    """The EfficientQAT file (Stage 1 + 2) as its own Pi bundle and repo (<winner's repo>-qat); the contest's winner,
    its bundle and its repo are left as they are."""
    f = _qatfiles(model)["stage12"]
    win = Path(load("winner", model, SUFFIX)["file"])
    if abs(f.stat().st_size - win.stat().st_size) > 64 * 2**20:
        raise SystemExit(f"{f.name} is {f.stat().st_size} bytes, the winner {win.stat().st_size}: not the same types")
    r = load("qat", model)["stage 1 + 2"]
    d = _pack(model, f, f"{SUFFIX}-qat", {"method": "EfficientQAT (Block-AP weights only, then scales end to end)",
                                          "held_out_accuracy": round(r["held_out"]["accuracy"], 2),
                                          "vs_winner": {k: r["vs_winner"].get(k, "accuracy points")
                                                        for k in ("diff", "ci", "verdict", "unit")}})
    repo = _push(model, f"{SUFFIX}-qat", d)
    note(f"{short(model)} QAT file uploaded: {repo} (held-out {r['held_out']['accuracy']:.2f})")


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def strings(x) -> list[str]:
    """Every string in a parsed JSON record."""
    vs = x.values() if isinstance(x, dict) else x if isinstance(x, list) else []
    return [x] if isinstance(x, str) else [y for v in vs for y in strings(v)]


def overlapping(prompts: list[str], texts: list[str], n: int = 8) -> set[int]:
    """The prompts that share a run of n words with any of the texts (one shorter than n words: found inside one),
    compared lowercased, on letters and digits only."""
    grams, shorts, hit = {}, {}, set()
    for k, p in enumerate(prompts):
        w = words(p)
        if len(w) < n:
            shorts[k] = f" {' '.join(w)} "
        for i in range(len(w) - n + 1):
            grams.setdefault(" ".join(w[i:i + n]), []).append(k)
    for t in texts:
        w = words(t)
        hit.update(k for i in range(len(w) - n + 1) for k in grams.get(" ".join(w[i:i + n]), ()))
        joined = f" {' '.join(w)} "
        hit.update(k for k, s in shorts.items() if s in joined)
    return hit


LABELS = ("id", "task", "source_split", "key", "subject")  # a record's bookkeeping, not its question


def questions(group: str) -> dict[str, list[list[str]]]:
    """A question split's records (tasks/<group>/*.jsonl), each as the strings of its question and answers, by file."""
    return {f.name: [strings({k: v for k, v in json.loads(x).items() if k not in LABELS})
                     for x in f.read_text().split("\n") if x]  # not splitlines: questions hold U+2028
            for f in sorted((DATA / "splits/tasks" / group).glob("*.jsonl"))}


def benchmark() -> list[str]:
    """Every selection and held-out question, one text per file (with the held-out llama.cpp task files)."""
    qs = ["\n".join(y for r in rs for y in r) for g in ("selection", "test") for rs in questions(g).values()]
    return qs + [g.read_bytes().decode("utf-8", "ignore")
                 for g in (DATA / "splits/llama_tasks/poc_test").iterdir() if g.is_file()]


def prompts(model: str) -> None:
    """CPU, with internet; ./panther runs it before a contest's GPU steps. The dataset's chat prompts (made once for
    every model): people's instructions, chat and everyday maths from a pool no other split uses, deduplicated, any
    sharing 8 words with a benchmark question dropped, then cut into train / select / test by a fixed seed. Then the
    check of every split."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    with lock("prompts"):
        if not PROMPT_FILE.exists():
            f = hf_hub_download(PROMPTS["repo"], PROMPTS["file"], repo_type="dataset", revision=PROMPTS["revision"])
            cols = ["text", "parent_id", "role", "lang", "deleted", "review_result"]
            t = pq.read_table(f, columns=cols).to_pydict()
            pool = {" ".join(words(x)): x for x, up, r, lang, gone, ok in zip(*(t[c] for c in cols))
                    if up is None and r == "prompter" and lang == "en" and not gone and ok and 20 <= len(x) <= 1000}
            pool = sorted(pool.values())
            random.Random(0).shuffle(pool)
            n = sum(PROMPTS["n"].values())
            pool = pool[:2 * n]  # checked in this order; the first n clean ones are kept, then cut in that order
            bad = overlapping(pool, benchmark())
            keep = [x for k, x in enumerate(pool) if k not in bad][:n]
            assert len(keep) == n, f"only {len(keep)} clean prompts"
            cut = [sp for sp, m in PROMPTS["n"].items() for _ in range(m)]
            PROMPT_FILE.write_text(json.dumps({**PROMPTS, "checked": len(pool), "dropped": len(bad),
                                               "prompts": [{"split": sp, "text": x} for sp, x in zip(cut, keep)]},
                                              indent=1))
            note(f"chat prompts: {n} from {PROMPTS['repo']} ({PROMPTS['n']}); {len(bad)} of {len(pool)} dropped as "
                 f"overlapping a benchmark question")
    check()


def check() -> None:
    """Decision 3: the splits share no question or text and are the files their manifests record; else the contest
    stops. Web: no document in two splits. Questions: no selection question among the held-out ones. Chat: no prompt
    in two splits, none sharing 8 words with a benchmark question."""
    s, bad = DATA / "splits", []
    want = json.loads((s / "manifest.json").read_text())["files"]
    tasks = json.loads((s / "llama_tasks/manifest.json").read_text())["files"].values()
    want |= {e["file"]: e["sha256"] for e in tasks}
    used = [*WEB.values(), *(s / "tasks/selection").glob("*.jsonl"), *(s / "tasks/test").glob("*.jsonl")]
    bad += [f"{f.relative_to(s)}: not in the manifest" for f in used if str(f.relative_to(s)) not in want]
    bad += [f"{e['file']}: missing" for e in tasks if not (s / e["file"]).exists()]
    bad += [f"{f}: hash differs" for f, h in want.items() if (s / f).exists() and digest_file(s / f) != h]
    ids = {sp: set(json.loads((s / "text" / f"{p.stem}.ids.json").read_text())) for sp, p in WEB.items()}
    bad += [f"web {a} and {b}: {n} documents in both" for a, b in combinations(SPLITS, 2)
            if (n := len(ids[a] & ids[b]))]
    qat = [p.with_suffix(".ids.json") for p in RW.glob("*/qat/tokens.npz")]  # decision 17: QAT text is train text
    bad += [f"{q}: {n} documents in select or test" for q in qat if q.exists()
            and (n := len(set(json.loads(q.read_text())) & (ids["select"] | ids["test"])))]
    old = sum(not q.exists() for q in qat)
    key = lambda r: " ".join(words(" ".join(r)))  # noqa: E731  (the whole question: stem, choices, answer)
    held = questions("test")
    for f, rs in questions("selection").items():
        if n := len({key(r) for r in rs} & {key(r) for r in held.get(f, [])}):
            bad.append(f"{f}: {n} selection questions also held out")
    ps = json.loads(PROMPT_FILE.read_text())["prompts"]
    if (n := len(ps) - len({" ".join(words(x["text"])) for x in ps})):
        bad.append(f"chat: {n} prompts twice")
    if hit := overlapping([x["text"] for x in ps], benchmark()):
        bad.append(f"chat: {len(hit)} prompts share 8 words with a benchmark question")
    if bad:
        note("split check FAILED: " + "; ".join(bad))
        raise SystemExit(1)
    note(f"split check: {sum((s / f).exists() for f in want)} files match their hashes; web documents, questions and "
         f"chat prompts share nothing; QAT samples share no document with select or test"
         + (f" ({old} QAT samples without recorded documents (made before 20 Nov), not checked)" if old else ""))


def prep(model: str) -> None:
    """refs and preset once this model has its build turn and the 16-bit original and the importance matrix exist:
    they run on the GPU while the type copies are built on the CPU. A queued model waits, so it never takes the GPU
    from the model being built."""
    while not ((STORE / model / ".turn").exists() and copy_of(model, "BF16").exists() and imatrix(model).exists()):
        time.sleep(60)
    for s in ("refs", "preset"):
        with lock(step_lock(model, s)):
            STEPS[s](model)


# --- EfficientQAT's Stage 1 (Block-AP) on a contest's winner: an experiment -------------------------------------

QAT_SIZES = {"block": (4096, 2048), "e2e": (4096, 4096), "val": (64, 2048)}  # EfficientQAT's samples x tokens


def corpus(model: str) -> None:
    """Token samples for EfficientQAT, drawn as its code draws them (documents shuffled with seed 0, the first 90 %
    for training and the rest for validation; each sample a random window in a random document longer than the
    window). The paper's text, RedPajama-1T-Sample, has been taken down; FineWeb-Edu stands in: the file pinned in
    the splits manifest, without the documents our calibration, selection and test texts were cut from."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    dst = out(model, "qat", "tokens.npz")
    if dst.exists():
        return
    src = json.loads((DATA / "splits/manifest.json").read_text())["sources"]["fineweb_edu"]
    (name, want), = src["files"].items()
    f = Path(hf_hub_download(src["repo"], name, repo_type="dataset", revision=src["revision"]))
    if digest_file(f) != want:
        raise SystemExit(f"{f}: not the pinned file")
    used = {i for g in (DATA / "splits/text").glob("*.ids.json") for i in json.loads(g.read_text())}
    t = pq.read_table(f, columns=["text", "id", "token_count"]).to_pydict()
    docs = [(x, n, i) for x, i, n in zip(t["text"], t["id"], t["token_count"]) if i not in used]
    hf, bf16 = DATA / "hf" / model, copy_of(model, "BF16")
    tok = (AutoTokenizer.from_pretrained(str(hf)) if (hf / "config.json").exists() else
           AutoTokenizer.from_pretrained(str(bf16.parent), gguf_file=bf16.name))
    rng = random.Random(0)
    order = list(range(len(docs)))
    rng.shuffle(order)
    cut = int(0.9 * len(order))

    def draw(pool: list[int], n: int, length: int) -> np.ndarray:
        pool = [i for i in pool if docs[i][1] > length * 0.8]  # the file's own token counts: skip clearly short ones
        a, k = np.empty((n, length), np.int32), 0
        while k < n:
            ids = tok(docs[d := rng.choice(pool)][0]).input_ids
            if len(ids) > length:
                i = rng.randint(0, len(ids) - length - 1)
                a[k], k = ids[i:i + length], k + 1
                drawn.add(docs[d][2])
        return a
    drawn: set[str] = set()  # the documents the samples come from, for the split check
    np.savez(f"{dst}.part.npz", **{k: draw(order[:cut] if k != "val" else order[cut:], *v) for k, v in QAT_SIZES.items()})
    dst.with_suffix(".ids.json").write_text(json.dumps(sorted(drawn)))
    os.rename(f"{dst}.part.npz", dst)
    note(f"{short(model)} QAT corpus: FineWeb-Edu, {len(docs)} documents ({len(used)} held out); samples drawn")


def qatdata(model: str) -> None:
    """CPU, with internet: the original (if its 16-bit file is gone) and the token samples."""
    if not copy_of(model, "BF16").exists():
        fetch(model)
    corpus(model)


E2E_LR = 2e-3  # Stage 2, relative: the paper's 2e-5 scaled like Stage 1's (2e-4) moved little in a 1.5B test
# (selection-text perplexity 16.70 -> 16.17); 2e-3, the final step's distillation rate, reached 14.66 (original 9.32)


def _qatfiles(model: str) -> dict[str, Path]:
    return {k: out(model, "qat", f"{BUDGET}-{k}.gguf") for k in ("start", "stage1", "stage12", "stage2")}


def _stage2(model: str, src: Path, dst: Path, what: str) -> None:
    """Stage 2 on src (scales only, end to end, KL to the original's top 64 on the 4,096 x 4,096 samples). The
    original's probabilities are computed once, by whichever job gets there first, and saved for the other."""
    import torch
    from panther.distill import distill, teacher_tops
    if dst.exists():
        return
    bf16, sm = copy_of(model, "BF16"), short(model)
    ids = torch.from_numpy(np.load(out(model, "qat", "tokens.npz"))["e2e"].astype(np.int64))
    tf = out(model, "qat", "teacher-tops.npz")
    with gpu(bf16.stat().st_size + 22 * GIB, 12 * 3600):  # model + logits and their gradients at 4,096 tokens
        with lock(f"{model}-teacher-tops"):
            if not tf.exists():
                tv, ti = teacher_tops(bf16, ids, 1)
                np.savez(f"{tf}.part.npz", v=tv.numpy(), i=ti.numpy())
                os.rename(f"{tf}.part.npz", tf)
                del tv, ti  # 6 GB on the CPU; read back below
        z = np.load(tf)
        distill(src, bf16, [], dst, lambda m: note(f"{sm} QAT {what}: {m}"), ids=ids,
                tops=(torch.from_numpy(z["v"]), torch.from_numpy(z["i"])), steps=None, batch=1, accum=32,
                lr=E2E_LR, train_norms=False)


def drop_copies(model: str, refs: bool = False) -> None:
    """The model's type copies removed (the 16-bit original, the importance matrix and the presets stay; the record
    of what was rounded is archived first), and with refs the original's reference outputs: the end of a contest
    (./panther, command `clean`). Not while another contest of this model is running."""
    d = STORE / model
    r = d / "rounded.json"
    if r.exists():
        a = DATA / "archive" / "rounded"
        a.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r, a / f"{model}-{time.strftime('%Y%m%dT%H%M', time.localtime(r.stat().st_mtime))}.json")
        r.unlink()
    for g in [*d.glob(f"{model}-*.gguf"), *((DATA / "reference" / model).glob("*.kld") if refs else [])]:
        if not g.name.endswith("-BF16.gguf"):
            g.unlink()
    note(f"{short(model)} type copies removed")


def qat(model: str, lean: bool = False) -> None:
    """EfficientQAT on this contest's winner: its types rebuilt with our rounding (the undistilled file the final
    step started from); Stage 1 (Block-AP); Stage 2 (scales only, end to end, KL to the original instead of the
    paper's cross-entropy) on that; both files tested like the winner. The control (qatcontrol) runs beside it."""
    from panther.blockap import train
    w = load("winner", model, SUFFIX)["assignment"]
    f, cache, bf16, sm = _qatfiles(model), DATA / "cache" / "blockap" / model, copy_of(model, "BF16"), short(model)
    if not f["start"].exists():  # the type copies are needed only to build the starting file
        store(model, sorted(set(w.values())))
        build(model, w, f["start"])
        drop_copies(model)  # Stage 1's cache needs the disk (22B: 206 GB beside 82 GB of copies)
    if not f["stage1"].exists():
        t = np.load(out(model, "qat", "tokens.npz"))
        with gpu(24 * GIB, 24 * 3600):  # a block in training; a resume books 40 GB more while it rebuilds the caches
            train(bf16, f["start"], t["block"], t["val"], f["stage1"], cache, lambda m: note(f"{sm} {m}"),
                  rebuild_gpu=lambda: gpu(40 * GIB, 3 * 3600))
        shutil.rmtree(cache, ignore_errors=True)
        note(f"{sm} QAT stage 1 done")
    _stage2(model, f["stage1"], f["stage12"], "stage 2 after stage 1")
    # lean: only the result that matters (what Stage 1 alone and the control show is known from the 14B)
    qateval(model, {"stage 1 + 2": (f["stage12"], f"qat12:{BUDGET}"),
                    **({} if lean else {"stage 1 only": (f["stage1"], f"qat1:{BUDGET}")})})


def qatcontrol(model: str) -> None:
    """The control, beside Stage 1 (it does not need it): Stage 2 alone on the undistilled file, then tested."""
    f = _qatfiles(model)
    while not f["start"].exists():  # the main job rebuilds it
        time.sleep(60)
    _stage2(model, f["start"], f["stage2"], "stage 2 only (control)")
    qateval(model, {"stage 2 only": (f["stage2"], f"qat2:{BUDGET}")})


def qateval(model: str, files: dict[str, tuple[Path, str]]) -> None:
    """Each file as the final step tests the winner: all selection questions, the guard against the preset, the
    held-out test; then its relative score there against the current winner and the preset (as the verdict)."""
    sm, bf16 = short(model), copy_of(model, "BF16")
    p, win, rj = load("preset", model)["preset"], load("winner", model, SUFFIX), record("qat", model)
    pf, pl, wf, wl = preset_file(model, p), f"preset:{p}", Path(win["file"]), label_of(win["winner"], win["assignment"])
    with lock(step_lock(model, "refs")):
        refs(model)  # the original's reference outputs (a finished contest's were cleaned up)
    for name, (path, label) in files.items():
        label = f"{label}:{digest_file(path)[:12]}"  # results are cached by label: a new file is tested anew
        if rj.exists() and json.loads(rj.read_text()).get(name, {}).get("label") == label:
            continue
        sel = score(model, None, label, path=path, full=True)
        g = versus({**sel, "correct": correct(model, label)},
                   {**score(model, None, pl, path=pf, full=True), "correct": correct(model, pl)},
                   correct(model, "original"))
        with gpu(max(path.stat().st_size, wf.stat().st_size, bf16.stat().st_size) + 12 * GIB, 4 * 3600):
            ours, theirs, winner, orig = (test(model, path, label), test(model, pf, pl), test(model, wf, wl),
                                          test(model, bf16, "original"))
        vw, vp = versus(ours, winner, orig["correct"]), versus(ours, theirs, orig["correct"])
        r = {"file": str(path), "label": label, "selection": sel, "guard_vs_preset": g, "held_out": summary(ours, orig),
             "vs_winner": {**vw, "winner_accuracy": stats.acc(winner["correct"])},
             "vs_preset": {**vp, **beside(ours, theirs, orig), "preset": p,
                           "preset_accuracy": stats.acc(theirs["correct"])},
             "original_accuracy": stats.acc(orig["correct"])}
        with lock(f"{model}-qateval"):  # the main job and the control write the same file
            res = json.loads(rj.read_text()) if rj.exists() else {}
            rj.write_text(json.dumps({**res, name: r}, indent=1, default=float))
        note(f"{sm} QAT {name}: held-out {r['held_out']['accuracy']:.2f} (winner {stats.acc(winner['correct']):.2f}, "
             f"preset {p} {stats.acc(theirs['correct']):.2f}, original {stats.acc(orig['correct']):.2f}) | vs winner "
             f"{vw['diff']:+.2f} rel [{vw['ci'][0]:+.2f}, {vw['ci'][1]:+.2f}] {vw['verdict']}, vs preset "
             f"{vp['diff']:+.2f} rel [{vp['ci'][0]:+.2f}, {vp['ci'][1]:+.2f}] {vp['verdict']}"
             f"{beside_text(r['vs_preset'], vp['verdict'])}")


STEPS = {"store": store, "error": error, "refs": refs, "preset": preset, "ga": ga, "final": final, "bundle": bundle}
ALONE = {"fetch": fetch, "prompts": prompts, "upload": upload, "save": save, "prep": prep, "qatdata": qatdata, "qatcontrol": qatcontrol,
         "qatupload": qatupload, "qat": lambda model: qat(model, lean="lean" in sys.argv[4:]),
         "clean": lambda model: drop_copies(model, refs=True)}  # not timed, no lock


def main() -> int:
    global BUDGET
    cmd, b, model = sys.argv[1:4]
    BUDGET = int(b)
    if cmd in ALONE:
        ALONE[cmd](model)
        return 0
    for s in STEPS if cmd == "run" else [cmd]:
        t0 = time.time()
        note(f"{short(model)} step {s}: start")
        with lock(step_lock(model, s)):
            STEPS[s](model)
        note(f"{short(model)} step {s}: done in {(time.time() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        note(f"FAILED: {' '.join(sys.argv[1:])}: {type(e).__name__}: {e}")
        raise
