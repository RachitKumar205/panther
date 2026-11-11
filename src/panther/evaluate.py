"""Quality tests with llama.cpp on the GPU: KL and perplexity (llama-perplexity), multiple choice and WinoGrande
(llama-perplexity task modes), GSM8K (llama-server). Every run is checked (llama.cpp exits 0 on failure, and the
task modes silently drop items) and written as one JSON record under data/results/<model>/."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from panther.model import REPO_ROOT

DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
LLAMA = Path(os.environ.get("PANTHER_LLAMA", "/opt/llama.cpp"))
BIN = LLAMA / "build-cuda" / "bin"
KL_CTX = PPL_CTX = 2048
TASK_CTX, TASK_PARALLEL = 4096, 5  # 5 sequences: ARC-C has 5-choice items


class ParseError(ValueError):
    """The output lacks what a successful run prints."""


# --- parsers -------------------------------------------------------------------------------------------------

_KL = {"kl_mean": r"Mean\s+KLD:\s*(-?[0-9.]+)", "kl_p99": r"99\.0%\s+KLD:\s*(-?[0-9.]+)",
       "top1_pct": r"Same top p:\s*([0-9.]+)", "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)"}


def parse_kl(text: str) -> dict:
    out = {}
    for key, rx in _KL.items():
        m = re.search(rx, text)
        if m is None:
            raise ParseError(f"KL output has no '{key}' line")
        out[key] = float(m.group(1))
    return out


def parse_ppl(text: str) -> tuple[float, np.ndarray, int]:
    """`--ppl-output-type 1` prints the running mean NLL after each chunk (6 decimals); chunk i's own mean is
    i*m_i - (i-1)*m_(i-1). Returns (ppl, per-chunk mean NLL, scored tokens per chunk)."""
    head = re.search(r"calculating perplexity over (\d+) chunks, n_ctx=(\d+)", text)
    final = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
    if head is None or final is None:
        raise ParseError("perplexity output lacks the chunk-count or final-estimate line")
    n, n_ctx = int(head.group(1)), int(head.group(2))
    rows = [float(m.group(1)) for m in re.finditer(r"^\s*\d+\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+\s*$", text, re.M)]
    if len(rows) != n:
        raise ParseError(f"expected {n} per-chunk rows, found {len(rows)}")
    chunk = np.diff(np.concatenate([[0.0], np.array(rows) * np.arange(1, n + 1)]))
    ppl = float(final.group(1))
    if abs(math.exp(chunk.mean()) - ppl) / ppl > 1e-4:
        raise ParseError("per-chunk NLL does not reproduce the final perplexity")
    return ppl, chunk, n_ctx - n_ctx // 2 - 1  # llama-perplexity scores the second half of each chunk


def parse_multiple_choice(text: str, n: int) -> np.ndarray:
    """The running accuracy after each question (8 decimals) gives the running count of correct answers."""
    rows = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^(\d+)\t(\d+\.\d{8})$", text, re.M)]
    final = re.search(r"Final result: ([0-9.]+)", text)
    if final is None or [i for i, _ in rows] != list(range(1, n + 1)):
        raise ParseError(f"multiple-choice output: {len(rows)} rows of {n}, final line {'present' if final else 'missing'}")
    counts = np.array([acc * i / 100 for i, acc in rows])
    if np.abs(counts - counts.round()).max() > 1e-3:
        raise ParseError("running count is not an integer")
    steps = np.diff(np.concatenate([[0], counts.round().astype(int)]))
    if not set(np.unique(steps)) <= {0, 1} or abs(100 * steps.mean() - float(final.group(1))) > 1e-3:
        raise ParseError("recovered answers do not match the reported accuracy")
    return steps.astype(np.int8)


def parse_winogrande(text: str, n: int) -> np.ndarray:
    rows = [(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in re.finditer(r"^(\d+)\t[0-9.]+\t\s*\S+\s+\S+\s+([12])\s+([12])$", text, re.M)]
    final = re.search(r"Final Winogrande score\((\d+) tasks\): ([0-9.]+)", text)
    if final is None or int(final.group(1)) != n or [r[0] for r in rows] != list(range(1, n + 1)):
        raise ParseError(f"winogrande output: {len(rows)} rows of {n}, final line {'present' if final else 'missing'}")
    correct = np.array([int(a == b) for _, a, b in rows], dtype=np.int8)
    if abs(100 * correct.mean() - float(final.group(2))) > 1e-3:
        raise ParseError("recovered answers do not match the reported accuracy")
    return correct


# --- one llama-perplexity run --------------------------------------------------------------------------------

def run(kind: str, model: str, gguf: Path, label: str, threads: int = 16, reference: str | None = None,
        text: str | None = None, task: str | None = None, split: str | None = None) -> bool:
    """kind: "kl" (against data/reference/<model>/<reference>.kld), "ppl" (data/splits/text/<text>.txt) or
    "task" (data/splits/llama_tasks/<split>/<task>). Writes the record; True if the run parsed."""
    out_dir = DATA / "results" / model
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    eval_id = f"{started:%Y%m%dT%H%M%S%fZ}-{kind}-{gguf.stem[:16]}"
    base = [str(BIN / "llama-perplexity"), "-m", str(gguf), "-t", str(threads), "-ngl", "99"]
    how: dict = {"build": "cuda"}
    if kind == "kl":
        cmd = base + ["--kl-divergence-base", str(DATA / "reference" / model / f"{reference}.kld"), "--kl-divergence",
                      "-c", str(KL_CTX), "-b", str(KL_CTX)]
        how["reference"] = reference
    elif kind == "ppl":
        cmd = base + ["-f", str(DATA / "splits" / "text" / f"{text}.txt"), "-c", str(PPL_CTX), "-b", str(PPL_CTX),
                      "--ppl-output-type", "1"]
        how["text"] = text
    else:
        entry = json.loads((DATA / "splits" / "llama_tasks" / "manifest.json").read_text())["files"][f"{split}/{task}"]
        path = str(DATA / "splits" / entry["file"])
        mode = (["-f", path, "--winogrande", "--winogrande-tasks", "0"] if task == "winogrande"
                else ["-bf", path, "--multiple-choice", "--multiple-choice-tasks", "0"])
        cmd = base + mode + ["-c", str(TASK_CTX), "-np", str(TASK_PARALLEL)]
        how |= {"task": task, "split": split}
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    (out_dir / "logs" / f"{eval_id}.log").write_text(output)
    rec = {"eval_id": eval_id, "kind": kind, "model": model, "label": label, "gguf": gguf.name, "how": how,
           "wall_seconds": round(time.monotonic() - t0, 1), "exit_code": proc.returncode}
    try:
        if kind == "kl":
            rec["metrics"] = parse_kl(output)
            summary = f"KL {rec['metrics']['kl_mean']:.5f} top1 {rec['metrics']['top1_pct']:.2f}%"
        elif kind == "ppl":
            ppl, chunk, tokens = parse_ppl(output)
            rec["metrics"] = {"ppl": ppl}
            rec["per_chunk"] = {"mean_nll": [round(x, 7) for x in chunk], "tokens": tokens}
            summary = f"PPL {ppl:.4f}"
        else:
            correct = (parse_winogrande if task == "winogrande" else parse_multiple_choice)(output, entry["n"])
            rec["metrics"] = {"accuracy_pct": 100 * float(correct.mean()), "n": len(correct)}
            rec["per_item"] = {"ids": entry["ids"], "correct": correct.tolist()}
            summary = f"{task} {rec['metrics']['accuracy_pct']:.2f}% (n={len(correct)})"
        rec["ok"] = True
    except ParseError as e:
        rec["ok"], summary = False, f"FAILED: {e}"
    (out_dir / f"{eval_id}.json").write_text(json.dumps(rec))
    print(f"{eval_id} {label}: {summary} [{rec['wall_seconds']}s]", flush=True)
    return rec["ok"]


# --- GSM8K (lm-eval 0.4.13 `gsm8k`: 5-shot, greedy, strict "#### answer" match) -------------------------------

GSM8K_STOP = ["Question:", "</s>", "<|im_end|>"]
GSM8K_MAX_TOKENS = 256


def gsm8k_prompt(question: str, fewshot: list[dict]) -> str:
    shots = "\n\n".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in fewshot)
    return f"{shots}\n\nQuestion: {question}\nAnswer:"


def _normalise(s: str) -> str:
    for rx in (r",", r"\$", r"(?s).*#### ", r"\.$"):
        s = re.sub(rx, "", s)
    return s.strip().lower()


def gsm8k_correct(completion: str, reference_answer: str) -> bool:
    m = re.search(r"#### (\-?[0-9\.\,]+)", completion)
    return m is not None and _normalise(m.group(1)) == _normalise(reference_answer)


class LlamaServer:
    """llama-server on 127.0.0.1 inside the job's container, greedy completions, `parallel` requests at once."""

    def __init__(self, gguf: Path, parallel: int, ctx_per_slot: int, threads: int, port: int = 8089,
                 log: Path | None = None) -> None:
        self.url, self.parallel = f"http://127.0.0.1:{port}", parallel
        self._log = open(log, "w") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen([str(BIN / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1", "--port",
                                      str(port), "-np", str(parallel), "-c", str(parallel * ctx_per_slot), "-t",
                                      str(threads), "--no-webui", "-ngl", "99"],
                                     stdout=self._log, stderr=subprocess.STDOUT)

    def __enter__(self) -> "LlamaServer":
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not become healthy")

    def __exit__(self, *_: object) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self._log is not subprocess.DEVNULL:
            self._log.close()

    def complete(self, prompt: str, n_predict: int, stop: list[str]) -> str:
        body = {"prompt": prompt, "n_predict": n_predict, "stop": stop, "temperature": 0.0, "top_k": 1, "seed": 1,
                "cache_prompt": False}
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())["content"]

    def map(self, fn, items: list) -> list:
        with ThreadPoolExecutor(self.parallel) as ex:
            return list(ex.map(fn, items))
