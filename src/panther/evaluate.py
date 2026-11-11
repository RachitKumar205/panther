"""Quality tests with llama.cpp: KL, perplexity, multiple choice, WinoGrande (llama-perplexity) and
GSM8K (llama-server); output parsing; one JSON record per run."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import gguf
import numpy as np

from panther.gguf_files import ASSIGNMENT_KEY
from panther.model import REPO_ROOT


class ParseError(ValueError):
    """The output lacks what a successful run prints."""


def scored_tokens_per_chunk(n_ctx: int) -> int:
    """llama-perplexity scores tokens first..n_ctx-2 of each chunk, first = n_ctx/2 (perplexity.cpp)."""
    return n_ctx - n_ctx // 2 - 1


_KL = {
    "ppl_q": re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)"),
    "ppl_ratio": re.compile(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)\s*±\s*([0-9.na-]+)"),
    "kl_mean": re.compile(r"Mean\s+KLD:\s*(-?[0-9.]+)\s*±\s*([0-9.]+)"),
    "kl_p99": re.compile(r"99\.0%\s+KLD:\s*(-?[0-9.]+)"),
    "kl_max": re.compile(r"Maximum KLD:\s*(-?[0-9.]+)"),
    "top1": re.compile(r"Same top p:\s*([0-9.]+)\s*±\s*([0-9.]+)\s*%"),
}
_KL_CHUNKS = re.compile(r"kl_divergence: computing over (\d+) chunks, n_ctx=(\d+)")


@dataclass(frozen=True)
class KLResult:
    kl_mean: float
    kl_mean_se: float
    kl_p99: float
    kl_max: float
    top1_pct: float
    top1_pct_se: float
    ppl_q: float
    ppl_ratio: float
    chunks: int
    n_ctx: int


def parse_kl(text: str) -> KLResult:
    vals: dict[str, tuple[float, float | None]] = {}
    for key, rx in _KL.items():
        m = rx.search(text)
        if m is None:
            raise ParseError(f"KL output has no '{key}' line")
        second = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        try:
            se = float(second) if second is not None else None
        except ValueError:  # "-nan" when the ratio has no spread
            se = None
        vals[key] = (float(m.group(1)), se)
    c = _KL_CHUNKS.search(text)
    if c is None:
        raise ParseError("KL output has no chunk count line")
    return KLResult(
        kl_mean=vals["kl_mean"][0], kl_mean_se=vals["kl_mean"][1] or 0.0,
        kl_p99=vals["kl_p99"][0], kl_max=vals["kl_max"][0],
        top1_pct=vals["top1"][0], top1_pct_se=vals["top1"][1] or 0.0,
        ppl_q=vals["ppl_q"][0], ppl_ratio=vals["ppl_ratio"][0],
        chunks=int(c.group(1)), n_ctx=int(c.group(2)),
    )


_PPL_ROW = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.M)
_PPL_FINAL = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")
_PPL_CHUNKS = re.compile(r"perplexity: calculating perplexity over (\d+) chunks, n_ctx=(\d+)")

RECOVERY_TOLERANCE = 1e-4  # recovered per-chunk NLL must reproduce the final perplexity within 0.01%


@dataclass(frozen=True)
class PPLResult:
    ppl: float
    ppl_se: float
    chunk_mean_nll: np.ndarray  # per-chunk mean NLL per scored token, recovered from running means
    tokens_per_chunk: int
    n_ctx: int
    recovery_rel_error: float

    @property
    def chunk_nll(self) -> np.ndarray:
        """Summed NLL per chunk, as panther.stats.Evaluation expects."""
        return self.chunk_mean_nll * self.tokens_per_chunk

    @property
    def chunk_tokens(self) -> np.ndarray:
        return np.full(len(self.chunk_mean_nll), self.tokens_per_chunk)


def parse_ppl_type1(text: str) -> PPLResult:
    """Parse `llama-perplexity --ppl-output-type 1`: one row per chunk with the running mean NLL to
    6 decimals. Chunk i's own mean is i*m_i - (i-1)*m_(i-1), since every chunk scores the same count."""
    head = _PPL_CHUNKS.search(text)
    final = _PPL_FINAL.search(text)
    if head is None or final is None:
        raise ParseError("perplexity output lacks the chunk-count or final-estimate line")
    n_chunks, n_ctx = int(head.group(1)), int(head.group(2))
    rows = [(int(m.group(1)), float(m.group(3))) for m in _PPL_ROW.finditer(text)]
    if len(rows) != n_chunks:
        raise ParseError(f"expected {n_chunks} per-chunk rows, found {len(rows)} (missing --ppl-output-type 1?)")
    for i, (pos, _) in enumerate(rows):
        if pos != i * n_ctx:
            raise ParseError(f"row {i} at position {pos}, expected {i * n_ctx}")
    running = np.array([r[1] for r in rows])
    idx = np.arange(1, n_chunks + 1)
    totals = running * idx
    chunk_mean = np.diff(np.concatenate([[0.0], totals]))
    ppl = float(final.group(1))
    recovered = math.exp(chunk_mean.mean())
    rel = abs(recovered - ppl) / ppl
    if rel > RECOVERY_TOLERANCE:
        raise ParseError(f"recovered per-chunk NLL gives PPL {recovered:.6f}, llama.cpp says {ppl} ({rel:.2e})")
    return PPLResult(ppl=ppl, ppl_se=float(final.group(2)), chunk_mean_nll=chunk_mean,
                     tokens_per_chunk=scored_tokens_per_chunk(n_ctx), n_ctx=n_ctx, recovery_rel_error=rel)


_TYPE_LINE = re.compile(r"- type\s+(\S+):\s+(\d+) tensors")


def parse_type_counts(text: str) -> dict[str, int]:
    """Per-type tensor counts from the loader log (printed at -lv 4 and above)."""
    return {m.group(1).lower(): int(m.group(2)) for m in _TYPE_LINE.finditer(text)}


# --- task modes (A4, D38) ---------------------------------------------------------------------------------

_MC_ROW = re.compile(r"^(\d+)\t(\d+\.\d{8})$", re.M)
_MC_FINAL = re.compile(r"Final result: ([0-9.]+) \+/- ([0-9.]+)")
_WG_ROW = re.compile(r"^(\d+)\t([0-9.]+)\t\s*(\S+)\s+(\S+)\s+([12])\s+([12])$", re.M)
_WG_FINAL = re.compile(r"Final Winogrande score\((\d+) tasks\): ([0-9.]+) \+/- ([0-9.]+)")


@dataclass(frozen=True)
class TaskResult:
    correct: np.ndarray  # 0/1 per question, in file order
    accuracy_pct: float  # llama.cpp's final figure
    accuracy_se_pct: float


def parse_multiple_choice(text: str, n_expected: int) -> TaskResult:
    """`--multiple-choice` prints the running accuracy (8 decimals) after each question; the running count of
    correct answers is round(acc * i / 100), and its steps are the per-question results."""
    rows = [(int(m.group(1)), float(m.group(2))) for m in _MC_ROW.finditer(text)]
    final = _MC_FINAL.search(text)
    if final is None or len(rows) != n_expected:
        raise ParseError(f"multiple-choice output: {len(rows)} rows of {n_expected}, final line "
                         f"{'present' if final else 'missing'}")
    counts = []
    for k, (i, acc) in enumerate(rows, 1):
        if i != k:
            raise ParseError(f"row {k} numbered {i}")
        c = acc * i / 100
        if abs(c - round(c)) > 1e-3:
            raise ParseError(f"row {i}: running count {c} is not an integer")
        counts.append(round(c))
    steps = np.diff(np.concatenate([[0], counts]))
    if not set(np.unique(steps)) <= {0, 1}:
        raise ParseError("running count moved by more than one")
    acc = float(final.group(1))
    if abs(100 * steps.mean() - acc) > 1e-3:
        raise ParseError(f"recovered accuracy {100 * steps.mean():.4f} != reported {acc}")
    return TaskResult(steps.astype(np.int8), acc, float(final.group(2)))


def parse_winogrande(text: str, n_expected: int) -> TaskResult:
    """`--winogrande` prints, per question, the choice made and the answer."""
    rows = [(int(m.group(1)), int(m.group(5)), int(m.group(6))) for m in _WG_ROW.finditer(text)]
    final = _WG_FINAL.search(text)
    if final is None or len(rows) != n_expected or int(final.group(1)) != n_expected:
        raise ParseError(f"winogrande output: {len(rows)} rows of {n_expected}, final line "
                         f"{'present' if final else 'missing'}")
    if [r[0] for r in rows] != list(range(1, n_expected + 1)):
        raise ParseError("winogrande rows out of order")
    correct = np.array([int(r[1] == r[2]) for r in rows], dtype=np.int8)
    acc = float(final.group(2))
    if abs(100 * correct.mean() - acc) > 1e-3:
        raise ParseError(f"recovered accuracy {100 * correct.mean():.4f} != reported {acc}")
    return TaskResult(correct, acc, float(final.group(3)))


GREEDY = {"temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False}


class LlamaServer:
    def __init__(self, llama: Path, gguf: Path, build: str, parallel: int, ctx_per_slot: int, threads: int,
                 port: int = 8089, log: Path | None = None) -> None:
        self.url = f"http://127.0.0.1:{port}"
        self.parallel = parallel
        cmd = [str(llama / f"build-{build}" / "bin" / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1",
               "--port", str(port), "-np", str(parallel), "-c", str(parallel * ctx_per_slot), "-t", str(threads),
               "--no-webui"]
        if build == "cuda":
            cmd += ["-ngl", "99"]
        self.cmd = cmd
        self._log = open(log, "w") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)

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

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self._log is not subprocess.DEVNULL:
            self._log.close()

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.loads(r.read())

    def complete(self, prompt: str, n_predict: int, stop: list[str]) -> str:
        return self._post("/completion", {"prompt": prompt, "n_predict": n_predict, "stop": stop, **GREEDY})["content"]

    def chat(self, user: str, max_tokens: int) -> str:
        r = self._post("/v1/chat/completions", {"messages": [{"role": "user", "content": user}],
                                                "max_tokens": max_tokens, **GREEDY})
        return r["choices"][0]["message"]["content"]

    def map(self, fn, items: list) -> list:
        """Run fn over items with one request in flight per server slot; results in input order."""
        with ThreadPoolExecutor(self.parallel) as ex:
            return list(ex.map(fn, items))


STOP = ["Question:", "</s>", "<|im_end|>"]
MAX_TOKENS = 256

STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
IGNORE = [r",", r"\$", r"(?s).*#### ", r"\.$"]
INVALID = "[invalid]"  # lm-eval's fallback when a filter finds nothing


def prompt(question: str, fewshot: list[dict]) -> str:
    """5-shot: "Question: q\\nAnswer: a" exemplars joined by a blank line, then the question."""
    shots = "\n\n".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in fewshot)
    return f"{shots}\n\nQuestion: {question}\nAnswer:"


def strict_match(completion: str) -> str:
    m = STRICT.search(completion)
    return m.group(1) if m else INVALID


def flexible_extract(completion: str) -> str:
    matches = FLEXIBLE.findall(completion)
    if not matches:
        return INVALID
    last = matches[-1]  # group_select: -1
    return next((g for g in last if g), INVALID)


def normalise(s: str) -> str:
    for rx in IGNORE:
        s = re.sub(rx, "", s)
    return s.strip().lower()


def exact_match(prediction: str, reference_answer: str) -> bool:
    return normalise(prediction) == normalise(reference_answer)


DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
LLAMA = Path(os.environ.get("PANTHER_LLAMA", REPO_ROOT / "third_party/llama.cpp"))
PPL_CTX = 2048  # perplexity at context 2048, stride = context
TASK_CTX = 4096  # per sequence (D38)
TASK_PARALLEL = 5  # >= the most answers any question has (ARC-C has 5-choice items); total context 5 x 4096


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def assignment_id(path: Path) -> str | None:
    """The assignment digest assembled files carry in their metadata; None for store files."""
    field = gguf.GGUFReader(path).fields.get(ASSIGNMENT_KEY)
    return field.contents() if field is not None else None


APPROVAL = "poc/APPROVED_TEST_SUBSET"  # created only on the user's explicit OK (D42)


def test_data_allowed(name: str) -> bool:
    """The full test split is for Phase D only (D34). The proof of concept may use its fixed test subset
    (`poc_test` task files, `test_web` text) once the user has approved it."""
    if not name.startswith(("test", "poc_test")):
        return True
    return name in ("poc_test", "test_web") and (DATA / APPROVAL).exists()


def llama_commit() -> str:
    out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(LLAMA), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return out or "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["kl", "ppl", "task"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--build", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--reference", help="kl: reference name (sensitivity, selection)")
    ap.add_argument("--text", help="ppl: split text name (sensitivity, selection; never test in Phase A)")
    ap.add_argument("--task", choices=["hellaswag", "arc_challenge", "mmlu", "winogrande"])
    ap.add_argument("--split", help="task: selection (test only in Phase D, D34)")
    ap.add_argument("--label", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("PANTHER_THREADS", "16")))
    a = ap.parse_args(argv)

    out_dir = DATA / "results" / a.model
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    file_sha = sha256(a.gguf)
    eval_id = f"{started:%Y%m%dT%H%M%SZ}-{a.kind}-{a.build}-{file_sha[:10]}"
    binary = LLAMA / f"build-{a.build}" / "bin" / "llama-perplexity"
    gpu = ["-ngl", "99"] if a.build == "cuda" else []

    how: dict = {"build": a.build, "threads": a.threads}
    if a.kind == "kl":
        if not a.reference:
            ap.error("kl needs --reference")
        ref_manifest = json.loads((DATA / "reference" / a.model / "manifest.json").read_text())
        ref = ref_manifest["references"][a.reference]
        flags = ref_manifest["kl_runs_must_use"]
        cmd = [str(binary), "-m", str(a.gguf), "--kl-divergence-base",
               str(DATA / "reference" / a.model / ref["file"]), "--kl-divergence",
               "-c", str(flags["-c"]), "-b", str(flags["-b"]), "-t", str(a.threads), *gpu]
        how |= {"reference": a.reference, "reference_sha256": ref["sha256"], "c": flags["-c"], "b": flags["-b"]}
    elif a.kind == "task":
        if not a.task or not a.split:
            ap.error("task needs --task and --split")
        if not test_data_allowed(a.split):
            ap.error("test data: full split only in Phase D (D34); PoC subset only after approval (D42)")
        task_manifest = json.loads((DATA / "splits" / "llama_tasks" / "manifest.json").read_text())
        entry = task_manifest["files"][f"{a.split}/{a.task}"]
        path = DATA / "splits" / entry["file"]
        mode = (["-f", str(path), "--winogrande", "--winogrande-tasks", "0"] if a.task == "winogrande"
                else ["-bf", str(path), "--multiple-choice", "--multiple-choice-tasks", "0"])
        cmd = [str(binary), "-m", str(a.gguf), *mode, "-c", str(TASK_CTX), "-np", str(TASK_PARALLEL),
               "-t", str(a.threads), *gpu]
        how |= {"task": a.task, "split": a.split, "file_sha256": entry["sha256"], "n_items": entry["n"],
                "c": TASK_CTX, "np": TASK_PARALLEL}
    else:
        if not a.text:
            ap.error("ppl needs --text")
        if not test_data_allowed(a.text):
            ap.error("test data: full split only in Phase D (D34); PoC subset only after approval (D42)")
        text = DATA / "splits" / "text" / f"{a.text}.txt"
        cmd = [str(binary), "-m", str(a.gguf), "-f", str(text), "-c", str(PPL_CTX), "-b", str(PPL_CTX),
               "--ppl-output-type", "1", "-t", str(a.threads), *gpu]
        how |= {"text": a.text, "text_sha256": sha256(text), "c": PPL_CTX, "b": PPL_CTX}

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.monotonic() - t0
    output = proc.stdout + proc.stderr
    (out_dir / "logs" / f"{eval_id}.log").write_text(output)

    record = {
        "eval_id": eval_id, "kind": a.kind, "model": a.model, "label": a.label,
        "evaluated": {"gguf": a.gguf.name, "gguf_sha256": file_sha, "assignment_sha256": assignment_id(a.gguf)},
        "how": how | {"llama_cpp_commit": llama_commit(),
                      "image_id": os.environ.get("PANTHER_IMAGE_ID", "unknown"), "command": cmd},
        "started": started.isoformat(timespec="seconds"), "wall_seconds": round(seconds, 2),
        "exit_code": proc.returncode,
    }
    try:
        if a.kind == "kl":
            record["metrics"] = asdict(parse_kl(output))
        elif a.kind == "task":
            parse = parse_winogrande if a.task == "winogrande" else parse_multiple_choice
            t = parse(output, entry["n"])
            record["metrics"] = {"accuracy_pct": t.accuracy_pct, "accuracy_se_pct": t.accuracy_se_pct,
                                 "n": int(len(t.correct))}
            record["per_item"] = {"ids": entry["ids"], "correct": t.correct.tolist()}
        else:
            r = parse_ppl_type1(output)
            record["metrics"] = {"ppl": r.ppl, "ppl_se": r.ppl_se, "chunks": len(r.chunk_mean_nll),
                                 "recovery_rel_error": r.recovery_rel_error}
            record["per_chunk"] = {"mean_nll": [round(x, 7) for x in r.chunk_mean_nll],
                                   "tokens": r.tokens_per_chunk}
        record["ok"] = True
    except ValueError as e:
        record["ok"] = False
        record["error"] = str(e)
    (out_dir / f"{eval_id}.json").write_text(json.dumps(record, indent=2))
    m = record.get("metrics", {})
    if not m:
        summary = record.get("error")
    elif a.kind == "kl":
        summary = f"KL {m['kl_mean']:.6f} p99 {m['kl_p99']:.4f} top1 {m['top1_pct']:.2f}%"
    elif a.kind == "task":
        summary = f"{a.task} {m['accuracy_pct']:.2f}% ± {m['accuracy_se_pct']:.2f} (n={m['n']})"
    else:
        summary = f"PPL {m['ppl']:.4f} ± {m['ppl_se']:.4f} ({m['chunks']} chunks)"
    print(f"{eval_id} {a.label or a.gguf.name}: {summary} [{seconds:.1f}s]", flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
