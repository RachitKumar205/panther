"""A model's matrices, their size per type, the RAM model; and what every step shares (paths, log, hashes, locks,
the run's GPU and RAM budget)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

REPO = Path(__file__).resolve().parents[2]
GIB = 1024**3
CFG = yaml.safe_load((REPO / "configs/search.yaml").read_text())
MODELS = {m["name"]: m for m in yaml.safe_load((REPO / "configs/models.yaml").read_text())["models"]}
KV_BPE = {"Q8_0": 34 / 32, "F16": 2.0}[CFG["kv_cache_type"]]
DATA = Path(os.environ.get("PANTHER_DATA", REPO / "data"))
RW = DATA / "rewrite"
LLAMA = Path(os.environ.get("PANTHER_LLAMA", "/opt/llama.cpp"))
BIN = LLAMA / "build-cuda/bin"
THREADS = int(os.environ.get("PANTHER_THREADS", "16"))
RUN = {"model": "", "budget": 0, "step": ""}  # this process's run (panther.run sets it)


def bpw(t: str) -> float:
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType[t]]
    return 8 * size / block


@dataclass(frozen=True)
class Matrix:
    name: str   # GGUF tensor name
    rows: int
    cols: int
    layer: int  # -1: token embedding / output head
    role: str   # attn_q ... ffn_down, token_embd, output

    def nbytes(self, t: str) -> int:
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType[t]]
        return self.rows * (self.cols // block) * size


@lru_cache(maxsize=None)
class Model:
    def __init__(self, name: str) -> None:
        c = json.loads((REPO / "configs/model_configs" / f"{name}.json").read_text())
        self.name, self.layers, h = name, c["num_hidden_layers"], c["hidden_size"]
        hd = c.get("head_dim") or h // c["num_attention_heads"]
        q, kv, i = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd, c["intermediate_size"]
        self.kv_dim, tied = kv, bool(c.get("tie_word_embeddings", False))
        mats = [Matrix("token_embd.weight", c["vocab_size"], h, -1, "token_embd")]
        if not tied:
            mats.append(Matrix("output.weight", c["vocab_size"], h, -1, "output"))
        for L in range(self.layers):
            for role, r, k in (("attn_q", q, h), ("attn_k", kv, h), ("attn_v", kv, h), ("attn_output", h, q),
                               ("ffn_gate", i, h), ("ffn_up", i, h), ("ffn_down", h, i)):
                mats.append(Matrix(f"blk.{L}.{role}.weight", r, k, L, role))
        self.matrices = mats
        self.params = sum(m.rows * m.cols for m in mats)
        bias = (q + 2 * kv) if c.get("model_type") == "qwen2" else 0  # Qwen2: q/k/v biases (F32)
        self.fixed = 4 * (self.layers * (2 * h + bias) + h)  # F32 norms and biases
        no_imat = {"token_embd"} | (set() if MODELS[name].get("imatrix_process_output") else {"output"})
        self.no_imatrix = {m.name for m in mats if m.role in no_imat}

    def runtime(self) -> int:
        """RAM besides the weights: KV cache at the configured context, and llama.cpp's overhead."""
        return int(2 * self.layers * self.kv_dim * CFG["context_tokens"] * KV_BPE + CFG["overhead_gib"] * GIB)

    def ram(self, a: dict[str, str]) -> int:
        """Predicted peak RAM in bytes of the assignment {tensor name: type}."""
        return self.fixed + sum(m.nbytes(a[m.name]) for m in self.matrices) + self.runtime()

    def allowed(self) -> dict[str, list[str]]:
        """Every type for every matrix, low to high, except types llama.cpp refuses without importance-matrix data
        (the word table has none; the output layer only if collected with --process-output)."""
        need = set(CFG["requires_imatrix"])
        return {m.name: sorted((t for t in CFG["formats"] if not (m.name in self.no_imatrix and t in need)), key=bpw)
                for m in self.matrices}


def short(model: str) -> str:
    return model.split("-")[1]


def named(size: str) -> str:  # 14b, or the full name
    hits = [m for m in MODELS if size in (m, short(m).lower())]
    if len(hits) != 1:
        raise SystemExit(f"unknown model '{size}': configs/models.yaml has {', '.join(short(m).lower() for m in MODELS)}")
    return hits[0]


def clock() -> str:
    return time.strftime("%H:%M", time.gmtime(time.time() + 19800))  # IST


def note(msg: str) -> None:  # `HH:MM <SIZE> msg` in data/logs/progress-b<budget>.log, which the dashboard reads
    line = f"{clock()} {short(RUN['model'])} {msg}"
    print(line, flush=True)
    with open(DATA / "logs" / f"progress-b{RUN['budget']}.log", "a") as f:
        f.write(line + "\n")


def path(*parts: str) -> Path:
    p = RW.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def record(kind: str, suffix: str = "-v2") -> Path:
    return path(f"budget-{RUN['budget']}", f"{kind}-{RUN['model']}{suffix}.json")


def digest(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:12]


def put(f: Path, text: str) -> None:
    """Written whole or not at all (a part file of its own, renamed): a reader never sees, and a step never reuses,
    half a file."""
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    Path(part := f"{f}.{uuid.uuid4().hex[:8]}.part").write_text(text)
    os.rename(part, f)


def done(f: Path, inputs: str) -> bool:  # a step's output exists and was made from these inputs
    return f.exists() and json.loads(f.read_text()).get("inputs") == inputs


def sha(f: Path) -> str:  # kept beside the file (<file>.sha256) while its size and time are unchanged
    side, st = Path(f"{f}.sha256"), f.stat()
    key = [str(st.st_size), str(st.st_mtime_ns)]
    if side.exists() and (old := side.read_text().split())[:2] == key:
        return old[2]
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    put(side, " ".join(key + [h.hexdigest()]))
    return h.hexdigest()


@contextmanager
def lock(name: str):  # one holder at a time across every run and container
    (DATA / "locks").mkdir(parents=True, exist_ok=True)
    with open(DATA / "locks" / f"{name}.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


class Budget:
    """The run's GPU and RAM claim, shared by everything it runs at once: a job waits until its bytes are free (one
    larger than the whole claim runs alone). A thread that holds a booking runs its work inside it (no second one)."""
    cv, total, free, mine = threading.Condition(), None, None, threading.local()

    def set(self, gpu: int, ram: int) -> None:
        self.total = self.free = (gpu, ram)

    @contextmanager
    def __call__(self, gpu: int, ram: int):
        if self.total is None:
            raise RuntimeError("no claim: GPU work runs only inside a claimed run (panther.run)")
        if getattr(self.mine, "held", False):
            yield
            return
        need = (min(gpu, self.total[0]), min(ram, self.total[1]))
        with self.cv:
            self.cv.wait_for(lambda: self.free[0] >= need[0] and self.free[1] >= need[1])
            self.free = (self.free[0] - need[0], self.free[1] - need[1])
        self.mine.held = True
        try:
            yield
        finally:
            self.mine.held = False
            with self.cv:
                self.free = (self.free[0] + need[0], self.free[1] + need[1])
                self.cv.notify_all()


hold = Budget()
