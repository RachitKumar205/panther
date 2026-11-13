"""llama.cpp runs on the GPU and their records: KL from the original, perplexity, multiple choice / WinoGrande
(llama-perplexity task modes), GSM8K (llama-server). llama.cpp exits 0 on failure and its task modes can drop
items, so every output is parsed and checked; each run is one JSON record under data/results/<model>/."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from panther.model import REPO

DATA = Path(os.environ.get("PANTHER_DATA", REPO / "data"))
BIN = Path(os.environ.get("PANTHER_LLAMA", "/opt/llama.cpp")) / "build-cuda/bin"
THREADS = int(os.environ.get("PANTHER_THREADS", "16"))
CTX = 2048


class ParseError(ValueError):
    pass


def parse_kl(text: str) -> dict:
    rx = {"kl_mean": r"Mean\s+KLD:\s*(-?[0-9.]+)", "top1_pct": r"Same top p:\s*([0-9.]+)"}
    out = {k: re.search(r, text) for k, r in rx.items()}
    if not all(out.values()):
        raise ParseError("KL output incomplete")
    return {k: float(m.group(1)) for k, m in out.items()}


def parse_ppl(text: str) -> float:
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
    if not m:
        raise ParseError("no final perplexity")
    return float(m.group(1))


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


def records(model: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in (DATA / "results" / model).glob("*.json")]


def find(model: str, label: str, **how) -> dict | None:
    hits = [r for r in records(model) if r.get("label") == label and r.get("ok")
            and all(r["how"].get(k) == v for k, v in how.items())]
    return max(hits, key=lambda r: r["eval_id"]) if hits else None


def run(kind: str, model: str, gguf: Path, label: str, **how) -> dict:
    """kind "kl" (how: reference), "ppl" (text) or "task" (task, split, n = a fixed subset of n, 0 = all). Returns
    the record (cached: a run with the same label and settings is not repeated). Raises if the output is bad."""
    if rec := find(model, label, **how):
        return rec
    out_dir = DATA / "results" / model
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    eid = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%fZ}-{kind}"
    cmd = [str(BIN / "llama-perplexity"), "-m", str(gguf), "-t", str(THREADS), "-ngl", "99"]
    if kind == "kl":
        cmd += ["--kl-divergence-base", str(DATA / "reference" / model / f"{how['reference']}.kld"),
                "--kl-divergence", "-c", str(CTX), "-b", str(CTX)]
    elif kind == "ppl":
        cmd += ["-f", str(DATA / "splits/text" / f"{how['text']}.txt"), "-c", str(CTX), "-b", str(CTX)]
    else:
        entry = json.loads((DATA / "splits/llama_tasks/manifest.json").read_text())["files"][
            f"{how['split']}/{how['task']}"]
        n = how.get("n") or entry["n"]
        f = str(DATA / "splits" / entry["file"])
        cmd += (["-f", f, "--winogrande", "--winogrande-tasks", str(how.get("n", 0))] if how["task"] == "winogrande"
                else ["-bf", f, "--multiple-choice", "--multiple-choice-tasks", str(how.get("n", 0))])
        cmd += ["-c", "4096", "-np", "5"]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    (out_dir / "logs" / f"{eid}.log").write_text(text)
    rec = {"eval_id": eid, "label": label, "how": how, "seconds": round(time.monotonic() - t0)}
    if kind == "kl":
        rec["metrics"] = parse_kl(text)
    elif kind == "ppl":
        rec["metrics"] = {"ppl": parse_ppl(text)}
    else:
        c = (parse_winogrande if how["task"] == "winogrande" else parse_choice)(text, n)
        rec["metrics"], rec["correct"] = {"accuracy_pct": 100 * float(c.mean())}, c.tolist()
    rec["ok"] = True
    (out_dir / f"{eid}.json").write_text(json.dumps(rec))
    print(f"{label} {kind} {how}: {rec['metrics']} [{rec['seconds']}s]", flush=True)
    return rec


# --- GSM8K: 5-shot, greedy, "#### answer" must match (lm-eval's gsm8k) ---------------------------------------

STOP = ["Question:", "</s>", "<|im_end|>"]


def gsm8k_prompt(question: str, shots: list[dict]) -> str:
    return "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots) + \
        f"Question: {question}\nAnswer:"


def gsm8k_correct(completion: str, answer: str) -> bool:
    norm = lambda s: re.sub(r"(?s).*#### |[,$]|\.$", "", s).strip()  # noqa: E731
    m = re.search(r"#### (-?[0-9.,]+)", completion)
    return m is not None and norm(m.group(1)) == norm(answer)


def complete(gguf: Path, prompts: list[str], max_tokens: int = 256, log: Path | None = None) -> list[str]:
    """Greedy completions from llama-server on a free local port, 8 at a time."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    srv = subprocess.Popen([str(BIN / "llama-server"), "-m", str(gguf), "--port", str(port), "-np", "8", "-c",
                            str(8 * 2048), "-t", str(THREADS), "-ngl", "99", "--no-webui"],
                           stdout=open(log or os.devnull, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(1800):  # a large model loads slowly while the disk is busy
            try:
                if urllib.request.urlopen(url + "/health", timeout=5).status == 200:
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("llama-server did not start")

        def one(prompt: str) -> str:
            body = {"prompt": prompt, "n_predict": max_tokens, "stop": STOP, "temperature": 0, "top_k": 1,
                    "seed": 1, "cache_prompt": False}
            req = urllib.request.Request(url + "/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=1800).read())["content"]
        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(one, prompts))
    finally:
        srv.terminate()
        srv.wait(60)


def gsm8k(model: str, gguf: Path, label: str, questions: Path, n: int = 0) -> np.ndarray:
    """0/1 per GSM8K question in the file (the first n if n), cached under data/results/<model>/gsm8k/."""
    rec = DATA / "results" / model / "gsm8k" / f"{label.replace(':', '_')}-{questions.parent.name}-{n}.json"
    if rec.exists():
        return np.array(json.loads(rec.read_text()), dtype=np.int8)
    jl = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x]  # noqa: E731
    items = jl(questions)[: n or None]
    shots = jl(DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    outs = complete(gguf, [gsm8k_prompt(it["question"], shots) for it in items])
    c = [int(gsm8k_correct(o, it["answer"])) for o, it in zip(outs, items)]
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps(c))
    return np.array(c, dtype=np.int8)
