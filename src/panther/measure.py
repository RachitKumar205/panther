"""llama.cpp runs, checked and cached by label (a built file's label carries its hash: a changed file is measured
anew); the original's reference outputs; a file's score and held-out test; the relative score and verdict."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from panther.data import PROMPT_FILE, REF, SELECT_TOKENS, SPLITS, WEB, chat_text, joined_text, jsonl, select_text
from panther.model import (BIN, CFG, DATA, GIB, REPO, RULES, RUN, THREADS, consts, digest, hold, llama_commit, lock,
                           made, put, sha, stamp)
from panther.store import copy_of

TASKS = ["mmlu", "hellaswag", "arc_challenge", "winogrande"]
FIT = {"mmlu": 300, "arc_challenge": 200, "hellaswag": 200, "winogrande": 200}  # the GA's selection questions
FIRST = {t: v // 3 for t, v in FIT.items()}  # a first look: the first third of FIT (llama.cpp draws them in order)
DROP_SE = 2  # a first look drops a genome worse than its rival there by more than 2 SE of the difference
ALL = dict.fromkeys(TASKS, 0)  # every selection question (llama.cpp's n = 0)
KL_WEIGHT = 10  # score = flips % + 10 x KL on the select text (lower is better)
CTX = 2048  # KL and perplexity: chunks (and batches) of 2,048 tokens
TASK_CTX, PARALLEL = 4096, 5  # multiple choice: 5 sequences of up to 4,096 positions at once
STOP = ["Question:", "</s>", "<|im_end|>"]
MANIFEST = "splits/llama_tasks/manifest.json"  # under DATA: the task files and their hashes


class ParseError(ValueError):
    pass


def parse_kl(text: str) -> dict:
    """The mean KL, same top word %, and each chunk's KL, recovered from the running mean printed after each chunk
    (each chunk scores the same 1,023 tokens): chunk k = k x mean_k - (k-1) x mean_(k-1)."""
    m = re.search(r"Mean\s+KLD:\s*(-?[0-9.]+)", text)
    top = re.search(r"Same top p:\s*([0-9.]+)", text)
    rows = re.findall(r"^\s*(\d+)\s+\S+ ± \s*\S+\s+\S+ ± \s*\S+\s+(-?[0-9.]+) ± ", text, re.M)
    if not (m and top and rows) or [int(i) for i, _ in rows] != list(range(1, len(rows) + 1)) or \
            abs(float(rows[-1][1]) - float(m[1])) > 1e-5:  # every chunk line, the last one llama.cpp's mean
        raise ParseError("KL output incomplete")
    chunks = np.diff([0, *(float(v) * k for k, (_, v) in enumerate(rows, 1))])
    return {"kl_mean": float(m[1]), "top1_pct": float(top[1]), "kl_chunks": chunks.tolist()}


def parse_ppl(text: str) -> float:
    if not (m := re.search(r"Final estimate: PPL = ([0-9.]+)", text)):
        raise ParseError("no final perplexity")
    return float(m[1])


def parse_choice(text: str, n: int) -> np.ndarray:
    """0/1 per question, from the running accuracy llama-perplexity prints after each one."""
    rows = [(int(a), float(b)) for a, b in re.findall(r"^(\d+)\t(\d+\.\d{8})$", text, re.M)]
    if [i for i, _ in rows] != list(range(1, n + 1)):
        raise ParseError(f"{len(rows)} of {n} questions")
    counts = np.array([acc * i / 100 for i, acc in rows]).round().astype(int)
    return np.diff(np.concatenate([[0], counts])).astype(np.int8)


def parse_winogrande(text: str, n: int) -> np.ndarray:
    rows = re.findall(r"^\d+\t[0-9.]+\t\s*\S+\s+\S+\s+([12])\s+([12])$", text, re.M)
    if len(rows) != n:
        raise ParseError(f"{len(rows)} of {n} questions")
    return np.array([int(a == b) for a, b in rows], dtype=np.int8)


def need(size: int, extra: int = 4 * GIB, model: str = "") -> tuple[int, int]:
    """GPU and RAM of one llama.cpp process on a file of size bytes. RAM is its next-word buffer: multiple choice
    holds 5 sequences x 4,096 positions x the vocabulary in float32 (measured 20 Nov on the 1.5B: 12.6 GB, the same
    for 300 or all questions; perplexity and KL, 2,048 positions, 1.2 GB), plus the file and 1 GiB. A file's runs book
    this once and run one after another."""
    vocab = json.loads((REPO / "configs/model_configs" / f"{model or RUN['model']}.json").read_text())["vocab_size"]
    return int(1.1 * size) + extra, PARALLEL * TASK_CTX * vocab * 4 + size + GIB


def cost(f: Path, extra: int = 4 * GIB, model: str = "") -> tuple[int, int]:
    return need(f.stat().st_size, extra, model)


def kld(model: str, sp: str) -> Path:
    return DATA / "reference" / model / f"{REF[sp]}.kld"


def original(model: str) -> str:  # the original's results label: its file's hash, like every other file's
    return f"original:{sha(copy_of(model, 'BF16'))[:12]}"


def entry(how: dict) -> dict:  # a task file's manifest entry (file, n, sha256)
    return json.loads((DATA / MANIFEST).read_text())["files"][f"{how['split']}/{how['task']}"]


def key(model: str, how: dict) -> dict:
    """A llama.cpp result's whole cache key, which a stored record must match exactly: what was asked, its input
    file's hash (task file, text or reference), the llama.cpp commit, its settings and measure's rules."""
    f = entry(how)["sha256"] if "task" in how else sha(DATA / "splits/text" / f"{how['text']}.txt") if "text" in how \
        else sha(kld(model, how["split"]))
    return {**how, "file": f[:12], "llama": llama_commit(), "ctx": [CTX, TASK_CTX, PARALLEL], "rules": RULES["measure"]}


def inputs(model: str) -> list:
    """What every scoring step rests on besides its own inputs: the llama.cpp commit, the original and its references,
    the task files, the RAM model, measure's and data's constants and rules."""
    return [llama_commit(), sha(copy_of(model, "BF16")), [sha(kld(model, sp)) for sp in REF], sha(DATA / MANIFEST),
            {k: v for k, v in CFG.items() if k != "contests"}, consts("measure", "data"), RULES["refs"],
            RULES["measure"]]


_SEEN: dict[Path, dict] = {}  # records already read (every run of a model writes into one folder)


def find(model: str, label: str, **how) -> dict | None:
    for p in (DATA / "results" / model).glob("*.json"):
        if p not in _SEEN:
            _SEEN[p] = json.loads(p.read_text())
    want = key(model, how)
    hits = [r for p, r in list(_SEEN.items()) if p.parent.name == model and r.get("label") == label and r.get("ok")
            and r["how"] == want]
    return max(hits, key=lambda r: r["eval_id"]) if hits else None


def run(kind: str, model: str, gguf: Path, label: str, **how) -> dict:
    """kind "kl" (split: the reference), "ppl" (text) or "task" (task, split, n: a fixed subset, 0 = all); cached."""
    if rec := find(model, label, **how):
        return rec
    d = DATA / "results" / model
    eid = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%fZ}-{kind}-{uuid.uuid4().hex[:8]}"
    cmd = [str(BIN / "llama-perplexity"), "-m", str(gguf), "-t", str(THREADS), "-ngl", "99"]
    if kind == "kl":
        cmd += ["--kl-divergence-base", str(kld(model, how["split"])),
                "--kl-divergence", "-c", str(CTX), "-b", str(CTX)]
    elif kind == "ppl":
        cmd += ["-f", str(DATA / "splits/text" / f"{how['text']}.txt"), "-c", str(CTX), "-b", str(CTX)]
    else:
        e = entry(how)
        n, f = how.get("n") or e["n"], str(DATA / "splits" / e["file"])
        o = "--winogrande" if how["task"] == "winogrande" else "--multiple-choice"
        cmd += ["-f" if o == "--winogrande" else "-bf", f, o, f"{o}-tasks", str(how.get("n", 0)), "-c", str(TASK_CTX),
                "-np", str(PARALLEL)]
    t0, log = time.monotonic(), d / "logs" / f"{eid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with hold(*cost(gguf, model=model)):
        p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    log.write_text(text)
    try:
        if p.returncode:
            raise ParseError(f"exit code {p.returncode}")
        rec = {"eval_id": eid, "label": label, "how": key(model, how), "seconds": round(time.monotonic() - t0),
               "ok": True}
        if kind == "kl":
            rec["metrics"] = parse_kl(text)
        elif kind == "ppl":
            rec["metrics"] = {"ppl": parse_ppl(text)}
        else:
            c = (parse_winogrande if how["task"] == "winogrande" else parse_choice)(text, n)
            rec["metrics"], rec["correct"] = {"accuracy_pct": 100 * float(c.mean())}, c.tolist()
    except ParseError as e:
        raise ParseError(f"llama-perplexity {kind} {how} on {gguf.name}: {e}; its output: {log}") from None
    put(d / f"{eid}.json", json.dumps(rec))
    shown = {k: len(v) if k == "kl_chunks" else v for k, v in rec["metrics"].items()}  # chunks of 1,023 tokens
    print(f"{label} {kind} {how}: {shown} [{rec['seconds']}s]", flush=True)
    return rec


def parallel(fns: list) -> list:
    """Every function at once (each llama.cpp run waits for its share of the claim); the first error is raised."""
    with ThreadPoolExecutor(max(1, len(fns))) as ex:
        return [f.result() for f in [ex.submit(fn) for fn in fns]]


# --- GSM8K (5-shot, greedy, "#### answer" must match: lm-eval's gsm8k) and chat answers, by llama-server ----------

def gsm8k_correct(completion: str, answer: str) -> bool:
    norm = lambda s: re.sub(r"(?s).*#### |[,$]|\.$", "", s).strip()  # noqa: E731
    m = re.search(r"#### (-?[0-9.,]+)", completion)
    return m is not None and norm(m.group(1)) == norm(answer)


def complete(gguf: Path, prompts: list[str], chat: bool = False) -> list[str]:
    """Greedy completions (256 tokens) by llama-server, 8 at a time. chat: each prompt a user's message in the model's
    own chat template (the GGUF's); the whole exchange is returned."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    def post(p: str, body: dict) -> dict:
        req = urllib.request.Request(url + p, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=1800).read())

    def one(prompt: str) -> str:
        if chat:
            prompt = post("/apply-template", {"messages": [{"role": "user", "content": prompt}]})["prompt"]
        body = {"prompt": prompt, "n_predict": 256, "stop": [] if chat else STOP, "temperature": 0, "top_k": 1,
                "seed": 1, "cache_prompt": False}  # chat: no stop words, the end of turn stops it
        return (prompt if chat else "") + post("/completion", body)["content"]
    log = DATA / "logs" / f"llama-server-{os.getpid()}-{port}.log"
    with hold(*cost(gguf, 8 * GIB)), open(log, "w") as lf:
        srv = subprocess.Popen([str(BIN / "llama-server"), "-m", str(gguf), "--port", str(port), "-np", "8", "-c",
                                str(8 * 2048), "-t", str(THREADS), "-ngl", "99", "--no-webui"]
                               + (["--jinja"] if chat else []), stdout=lf, stderr=subprocess.STDOUT)
        try:
            for _ in range(1800):  # a large model loads slowly while the disk is busy
                if srv.poll() is not None:
                    raise RuntimeError(f"llama-server exited {srv.returncode}; its output: {log}")
                try:
                    if urllib.request.urlopen(url + "/health", timeout=5).status == 200:
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError(f"llama-server did not start in 30 min; its output: {log}")
            other = json.loads(urllib.request.urlopen(url + "/props", timeout=60).read())["model_path"]
            if other != str(gguf):
                raise RuntimeError(f"port {port} is answered by a server of {other}, not ours (it lost the port to it)")
            with ThreadPoolExecutor(8) as ex:
                return list(ex.map(one, prompts))
        finally:  # never masks the error that got here, never leaves the server running
            srv.terminate()
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait()


def gsm8k(model: str, gguf: Path, label: str) -> np.ndarray:
    """0/1 per held-out GSM8K question, cached under data/results/<model>/gsm8k/ with what made them."""
    qs, fs = DATA / "splits/llama_tasks/poc_test/gsm8k.jsonl", DATA / "splits/tasks/fewshot/gsm8k.jsonl"
    rec = DATA / "results" / model / "gsm8k" / f"{label.replace(':', '_')}-{qs.parent.name}-0.json"
    ins = digest([label, sha(qs), sha(fs), STOP, llama_commit(), RULES["measure"]])
    if not made(rec, ins):
        items, shots = jsonl(qs), jsonl(fs)
        head = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots)
        outs = complete(gguf, [f"{head}Question: {it['question']}\nAnswer:" for it in items])
        put(rec, json.dumps([int(gsm8k_correct(o, it["answer"])) for o, it in zip(outs, items)]))
        stamp(rec, ins)
    return np.array(json.loads(rec.read_text()), dtype=np.int8)


# --- the original's reference outputs, a file's score and held-out test --------------------------------------------

def refs(model: str) -> None:
    """The original's chat answers (the dataset's chat part), its next-word probabilities on the select and test
    text, and which selection and held-out questions it gets right. Each file is made anew when what made it changed."""
    ref, bf16, orig = DATA / "reference" / model, copy_of(model, "BF16"), original(model)
    ref.mkdir(parents=True, exist_ok=True)
    with lock(f"refs-{model}"):
        ins = digest([sha(PROMPT_FILE), sha(bf16), llama_commit(), RULES["refs"]])
        if not all(made(chat_text(model, sp), ins) for sp in SPLITS):  # each model in its own chat template
            ps = json.loads(PROMPT_FILE.read_text())["prompts"]
            outs = complete(bf16, [x["text"] for x in ps], chat=True)
            for sp in SPLITS:
                put(chat_text(model, sp), "\n\n".join(o for o, x in zip(outs, ps) if x["split"] == sp) + "\n")
                stamp(chat_text(model, sp), ins)

        for sp in REF:
            j, sel = joined_text(model, sp), [SELECT_TOKENS] if sp == "select" else []
            ins = digest([sha(chat_text(model, sp)), sha(WEB[sp]), sha(bf16), sel, RULES["refs"]])
            if not made(j, ins):
                put(j, select_text(model, bf16) if sp == "select" else
                    chat_text(model, sp).read_text() + "\n" + WEB[sp].read_text())
                stamp(j, ins)

        def base(sp: str) -> None:
            ins = digest([sha(joined_text(model, sp)), sha(bf16), llama_commit(), CTX, RULES["refs"]])
            if not made(kld(model, sp), ins):  # whole 2048-token chunks: a last partial one is not read
                with hold(*cost(bf16)):
                    subprocess.run([str(BIN / "llama-perplexity"), "-m", str(bf16), "-f", str(joined_text(model, sp)),
                                    "-c", str(CTX), "-b", str(CTX), "-ngl", "99", "-t", str(THREADS),
                                    "--kl-divergence-base", f"{ref}/{REF[sp]}.part"], check=True)
                os.rename(f"{ref}/{REF[sp]}.part", kld(model, sp))
                stamp(kld(model, sp), ins)
        parallel([lambda sp=sp: base(sp) for sp in REF] + [lambda: test(model, bf16, orig)] + [
            lambda t=t, n=n: run("task", model, bf16, orig, task=t, split="selection", n=n[t])
            for t in TASKS for n in (FIT, FIRST, ALL)])
        parallel([lambda sp=sp: sha(kld(model, sp)) for sp in REF])  # hashed once here, not by every scoring thread


def correct(model: str, label: str, n: dict = ALL) -> dict[str, np.ndarray]:
    return {t: np.array(find(model, label, task=t, split="selection", n=n[t])["correct"]) for t in TASKS}


def test(model: str, path: Path, label: str) -> dict:
    """The held-out test: 4 multiple-choice tasks, GSM8K (test split), KL and perplexity on the test text."""
    jobs = [lambda t=t: run("task", model, path, label, task=t, split="poc_test", n=0) for t in TASKS]
    jobs += [lambda: gsm8k(model, path, label), lambda: run("ppl", model, path, label, text=joined_text(model, "test").stem)]
    jobs += [] if label.startswith("original:") else [lambda: run("kl", model, path, label, split="test")]
    r = parallel(jobs)
    c = {t: np.array(x["correct"]) for t, x in zip(TASKS, r)} | {"gsm8k": r[4]}
    kl = r[6]["metrics"] if len(r) > 6 else None
    return {"correct": c, "kl": kl and kl["kl_mean"], "kl_chunks": kl and kl["kl_chunks"],
            "same_top": kl and kl["top1_pct"], "ppl": r[5]["metrics"]["ppl"]}


def score(model: str, label: str, file, full: bool = False, rival: str | None = None) -> dict:
    """Flips (the GA's questions, or all selection questions if full) + 10 x KL on the select text. file: the GGUF or
    a function building it (called only if a result is missing). rival (a label scored on the GA's questions): a file
    worse than it on a first look at the first third of them, paired, is dropped there ("dropped": True)."""
    fpath = None

    def runs(n: dict) -> dict:
        nonlocal fpath
        hows = [("kl", {"split": "select"})] + [("task", {"task": t, "split": "selection", "n": n[t]}) for t in TASKS]
        todo = [(k, h) for k, h in hows if not find(model, label, **h)]
        if todo and fpath is None:
            fpath = file() if callable(file) else file
        with hold(*cost(fpath)) if todo else contextlib.nullcontext():  # one booking per file, its runs in turn
            kl = [run(k, model, fpath, label, **h) for k, h in hows][0]["metrics"]
        c = correct(model, label, n)
        f = flips(c, correct(model, original(model), n))
        return {"score": f + KL_WEIGHT * kl["kl_mean"], "flips": f, "kl": kl["kl_mean"], "kl_chunks": kl["kl_chunks"],
                "same_top": kl["top1_pct"], "accuracy": acc(c)}  # same top word %: reported, not scored
    n = ALL if full else FIT
    if rival and not all(find(model, label, task=t, split="selection", n=n[t]) for t in TASKS):
        first, o = runs(FIRST), correct(model, original(model), FIRST)
        w, wkl = {t: v[:FIRST[t]] for t, v in correct(model, rival, FIT).items()}, find(model, rival, split="select")
        gap = first["score"] - flips(w, o) - KL_WEIGHT * wkl["metrics"]["kl_mean"]
        if gap > DROP_SE * score_se(correct(model, label, FIRST), w, o, first["kl_chunks"], wkl["metrics"]["kl_chunks"],
                                    KL_WEIGHT):
            return {**first, "dropped": True}
    return runs(n)


# --- the relative score, its standard error, the verdict (decisions 9, 13, 15) -------------------------------------

def acc(correct: dict[str, np.ndarray]) -> float:
    return 100 * float(np.mean([np.mean(v) for v in correct.values()]))


def flips(correct: dict[str, np.ndarray], original: dict[str, np.ndarray]) -> float:
    """% of questions whose right/wrong differs from the original's, averaged over tasks."""
    return 100 * float(np.mean([np.mean(correct[t] != original[t]) for t in original]))


def kl_se(a: list, b: list) -> float:
    """SE of the difference of two files' mean KL, paired over the chunks of the same text and reference."""
    return float(np.std(np.subtract(a, b), ddof=1) / np.sqrt(len(a)))


def score_se(a: dict, b: dict, original: dict, chunks_a: list, chunks_b: list, kl_weight: float) -> float:
    """SE of the difference of two scores: flips paired over the same questions (per task, averaged over tasks), and
    KL paired over the same chunks; the two independent."""
    f = 100 * np.sqrt(sum(np.var((a[t] != o).astype(float) - (b[t] != o).astype(float), ddof=1) / len(o)
                          for t, o in original.items())) / len(original)
    return float(np.hypot(f, kl_weight * kl_se(chunks_a, chunks_b)))


def relative(a: dict, b: dict, original: dict, kl_a: float, chunks_a: list, kl_b: float, chunks_b: list,
             kl_weight: float) -> tuple[float, float, float]:
    """a's score as % of b's, with its 95% interval: 100 x (score_a +- 1.96 SE of the difference) / score_b."""
    sa, sb = flips(a, original) + kl_weight * kl_a, flips(b, original) + kl_weight * kl_b
    h = 1.96 * score_se(a, b, original, chunks_a, chunks_b, kl_weight)
    return 100 * sa / sb, 100 * (sa - h) / sb, 100 * (sa + h) / sb


def verdict(lo: float, hi: float) -> str:
    """From a relative score's interval: below 100 is better than the other file."""
    return "WIN" if hi < 100 else "LOSE" if lo > 100 else "TIE"


def against(r: dict, base: float) -> dict:
    """With the score as a share of the preset's on the same questions and text (100: the preset's damage, 0: none)."""
    return {**r, "relative": 100 * r["score"] / base}


def versus(a: dict, b: dict, orig: dict[str, np.ndarray], w: float = KL_WEIGHT) -> dict:
    """a's relative score against b (100 = b's damage) and its 95% interval; diff and ci: those - 100 (negative = better
    than b)."""
    rel, lo, hi = relative(a["correct"], b["correct"], orig, a["kl"], a["kl_chunks"], b["kl"], b["kl_chunks"], w)
    return {"relative": rel, "relative_ci": [lo, hi], "diff": rel - 100, "ci": [lo - 100, hi - 100],
            "verdict": verdict(lo, hi), "unit": "relative score - 100"}


def beside(a: dict, b: dict, orig: dict) -> dict:
    """Reported beside the verdict, deciding nothing (decision 15): accuracy kept, the verdict at KL weight 5 and 20."""
    o = acc(orig["correct"])
    wc = {str(w): {"relative": v["relative"], "ci": v["relative_ci"], "verdict": v["verdict"]}
          for w in (5, 20) for v in [versus(a, b, orig["correct"], w)]}
    return {"accuracy_kept": {"ours": 100 * acc(a["correct"]) / o, "preset": 100 * acc(b["correct"]) / o},
            "weight_check": wc}


def beside_text(c: dict, v: str) -> str:
    k, w = c["accuracy_kept"], c["weight_check"]
    return (f"; accuracy kept {k['ours']:.1f} vs {k['preset']:.1f}% of the original; "
            + ("same verdict at KL weight 5 and 20" if all(x["verdict"] == v for x in w.values())
               else f"verdict at KL weight 5: {w['5']['verdict']}, 20: {w['20']['verdict']}"))


def summary(r: dict, orig: dict) -> dict:
    """A held-out test's numbers as the result files store them (flips against the original's answers)."""
    return {"accuracy": acc(r["correct"]), "kl": r["kl"], "same_top": r.get("same_top"), "ppl": r["ppl"],
            "flips": flips(r["correct"], orig["correct"]),
            "per_task": {t: 100 * float(np.mean(v)) for t, v in r["correct"].items()}}
