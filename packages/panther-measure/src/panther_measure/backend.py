"""The llama.cpp seam.

Every measurement goes through a :class:`LlamaBackend`. :class:`SubprocessLlama` drives the real
binaries; :class:`FakeLlama` returns deterministic results so the pipeline runs without a GPU.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from panther_measure.parse import KlMetrics, parse_choice, parse_kl, parse_ppl, parse_winogrande


class LlamaBackend(Protocol):
    """Runs llama.cpp measurements on a GGUF file."""

    def kl(self, model: Path, reference: Path, ctx: int) -> KlMetrics:
        """KL divergence against a recorded reference, over ``ctx``-token chunks."""
        raise NotImplementedError

    def ppl(self, model: Path, text: Path, ctx: int) -> float:
        """Perplexity on ``text``."""
        raise NotImplementedError

    def choice(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        """Multiple-choice correctness (0/1 per question)."""
        raise NotImplementedError

    def winogrande(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        """Winograd-style correctness (0/1 per question)."""
        raise NotImplementedError

    def complete(self, model: Path, prompts: Sequence[str], chat: bool) -> list[str]:
        """Greedy completions via llama-server."""
        raise NotImplementedError


class LlamaSettings(BaseModel):
    """How to invoke the llama.cpp binaries."""

    model_config = ConfigDict(frozen=True)

    bin_dir: Path  # directory holding llama-perplexity / llama-server
    threads: int = Field(gt=0)  # -t: one source of truth, not the container CPU cap (bug B13)
    gpu_layers: int = Field(ge=0)  # -ngl


class FakeLlama:
    """Deterministic backend for tests and the vertical slice."""

    def __init__(
        self,
        kl_mean: float = 0.1,
        top1_pct: float = 95.0,
        ppl: float = 10.0,
        correct_rate: float = 0.7,
        completions: Sequence[str] = (),
    ) -> None:
        self.metrics = KlMetrics(mean=kl_mean, se=0.01, top1_pct=top1_pct, chunks=(kl_mean, kl_mean))
        self.ppl_value = ppl
        self.correct_rate = correct_rate
        self.completions = list(completions)

    def kl(self, model: Path, reference: Path, ctx: int) -> KlMetrics:
        return self.metrics

    def ppl(self, model: Path, text: Path, ctx: int) -> float:
        return self.ppl_value

    def _correct(self, task_file: Path, n: int) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(task_file.name.encode()))  # stable across processes
        return (rng.random(n) < self.correct_rate).astype(np.int8)

    def choice(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        return self._correct(task_file, n)

    def winogrande(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        return self._correct(task_file, n)

    def complete(self, model: Path, prompts: Sequence[str], chat: bool) -> list[str]:
        return self.completions[: len(prompts)] if self.completions else [f"answer for {p[:20]}" for p in prompts]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr


class SubprocessLlama:
    """The real backend: llama-perplexity for scores, llama-server for completions."""

    def __init__(self, settings: LlamaSettings, run: Callable[[list[str]], str] = _run) -> None:
        self.settings = settings
        self.run = run

    def _bin(self, name: str) -> str:
        return str(self.settings.bin_dir / name)

    def _ppl_cmd(self, model: Path, ctx: int, *extra: str) -> list[str]:
        return [
            self._bin("llama-perplexity"),
            "-m",
            str(model),
            *extra,
            "-c",
            str(ctx),
            "-b",
            str(ctx),
            "-t",
            str(self.settings.threads),
            "-ngl",
            str(self.settings.gpu_layers),
        ]

    def kl(self, model: Path, reference: Path, ctx: int) -> KlMetrics:
        return parse_kl(self.run(self._ppl_cmd(model, ctx, "--kl-divergence-base", str(reference), "--kl-divergence")))

    def ppl(self, model: Path, text: Path, ctx: int) -> float:
        return parse_ppl(self.run(self._ppl_cmd(model, ctx, "-f", str(text))))

    def choice(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        out = self.run(self._ppl_cmd(model, ctx, "-bf", str(task_file), "--multiple-choice", f"{n}"))
        return parse_choice(out, n)

    def winogrande(self, model: Path, task_file: Path, n: int, ctx: int) -> np.ndarray:
        out = self.run(self._ppl_cmd(model, ctx, "-f", str(task_file), "--winogrande", f"{n}"))
        return parse_winogrande(out, n)

    def complete(self, model: Path, prompts: Sequence[str], chat: bool) -> list[str]:
        for _ in range(5):  # a port collision exits before readiness: retry (bug B12)
            port = _free_port()
            proc = subprocess.Popen(
                [
                    self._bin("llama-server"),
                    "-m",
                    str(model),
                    "-t",
                    str(self.settings.threads),
                    "-ngl",
                    str(self.settings.gpu_layers),
                    "--port",
                    str(port),
                    *(["--jinja"] if chat else []),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                if not self._ready(proc, port):
                    continue
                return self._ask(port, prompts, chat)
            finally:
                proc.terminate()
                proc.wait(timeout=60)
        raise RuntimeError("llama-server could not start on a free port")

    def _ready(self, proc: subprocess.Popen[str], port: int) -> bool:
        for _ in range(1800):
            if proc.poll() is not None:
                return False
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200:
                    return True
            except OSError:
                time.sleep(1)
        return False

    def _ask(self, port: int, prompts: Sequence[str], chat: bool) -> list[str]:
        base = f"http://127.0.0.1:{port}"

        def post(path: str, body: dict[str, object]) -> dict[str, object]:
            req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            return cast(dict[str, object], json.loads(urllib.request.urlopen(req, timeout=1800).read()))

        def one(prompt: str) -> str:
            if chat:
                prompt = str(post("/apply-template", {"messages": [{"role": "user", "content": prompt}]})["prompt"])
            body: dict[str, object] = {
                "prompt": prompt,
                "n_predict": 256,
                "temperature": 0,
                "top_k": 1,
                "seed": 1,
                "cache_prompt": False,
            }
            return (prompt if chat else "") + str(post("/completion", body)["content"])

        return [one(prompt) for prompt in prompts]
