"""GSM8K: grade-school maths word problems, answered by writing out the working (5 worked examples first, greedy, at
most 256 tokens); right if the number after "####" is the answer's. Run by llama-server, 8 questions at once."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from panther.fitness.resources import GIB, Resources
from panther.model import Model

SLOTS, TOKENS, CONTEXT = 8, 256, 2048  # questions at once; tokens written per answer; context per question
STOP = ["Question:"]  # the next made-up question starts: the answer is over


def correct(completion: str, answer: str) -> bool:
    """The number after the first "####" of the completion equals the one in the answer (commas, $ and a final dot
    ignored)."""
    def number(s: str) -> str:
        return re.sub(r"[,$]|\.$", "", s).strip()
    found = re.search(r"####\s*(-?[0-9.,$]+)", completion)
    return found is not None and number(found[1]) == number(answer.split("####")[-1])


class LlamaServer:
    """llama-server on a GGUF, for as long as the `with` block lasts. It holds its share of GPU memory and RAM."""

    def __init__(self, file: Path, resources: Resources) -> None:
        self.file, self.resources = file, resources

    def __enter__(self) -> LlamaServer:
        size = self.file.stat().st_size
        self.hold = self.resources.hold(int(1.1 * size) + 4 * GIB, size + 4 * GIB)
        self.hold.__enter__()
        with socket.socket() as s:  # a free port
            s.bind(("127.0.0.1", 0))
            self.url = f"http://127.0.0.1:{s.getsockname()[1]}"
        self.process = subprocess.Popen(
            ["llama-server", "-m", self.file, "--port", self.url.rsplit(":", 1)[1], "-np", str(SLOTS),
             "-c", str(SLOTS * CONTEXT), "-ngl", "99", "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while True:  # until it answers; it fails loudly if it stops (a port taken, a broken file)
            assert self.process.poll() is None, f"llama-server on {self.file.name} stopped (exit {self.process.poll()})"
            try:
                if urllib.request.urlopen(self.url + "/health", timeout=5).status == 200:
                    return self
            except OSError:
                time.sleep(0.5)

    def __exit__(self, *exc) -> None:
        self.process.terminate()
        self.process.wait()
        self.hold.__exit__(*exc)

    def complete(self, prompts: list[str]) -> list[str]:
        """Greedy completions, 8 at once."""
        def one(prompt: str) -> str:
            body = {"prompt": prompt, "n_predict": TOKENS, "temperature": 0, "stop": STOP, "cache_prompt": False}
            request = urllib.request.Request(self.url + "/completion", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(request, timeout=3600).read())["content"]
        with ThreadPoolExecutor(SLOTS) as pool:
            return list(pool.map(one, prompts))


class GSM8K:
    """GSM8K questions (a JSON-lines file: {"question", "answer"}, the answer ending "#### <number>") and the worked
    examples shown before each (the same format). The original's answers are recorded once, in
    data/models/<model>/reference/, named by the content of both files."""

    def __init__(self, model: Model, questions: Path, examples: Path, resources: Resources) -> None:
        self.model, self.resources = model, resources
        self.questions = [json.loads(x) for x in questions.read_text().split("\n") if x]
        examples_text = examples.read_text()
        head = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n"
                       for e in (json.loads(x) for x in examples_text.split("\n") if x))
        self.prompts = [f"{head}Question: {q['question']}\nAnswer:" for q in self.questions]
        content = (questions.read_text() + examples_text).encode()
        self.record = model.dir / "reference" / f"{hashlib.sha256(content).hexdigest()[:16]}.json"

    def answers(self, file: Path) -> np.ndarray:
        """True/False per question."""
        with LlamaServer(file, self.resources) as server:
            completions = server.complete(self.prompts)
        return np.array([correct(c, q["answer"]) for c, q in zip(completions, self.questions, strict=True)])

    def original_answers(self) -> np.ndarray:
        if not self.record.exists():
            self.record.parent.mkdir(parents=True, exist_ok=True)
            self.record.with_suffix(".part").write_text(json.dumps(self.answers(self.model.original).tolist()))
            os.replace(self.record.with_suffix(".part"), self.record)
        return np.array(json.loads(self.record.read_text()))
