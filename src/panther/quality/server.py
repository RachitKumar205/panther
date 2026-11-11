"""Run `llama-server` on localhost inside the job's container and send it greedy requests (A5, A6).

GPU containers have no outside network; the server listens on 127.0.0.1 only, inside the same container.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
