"""The Pi's commands. Each serves one bundle with llama-server at the bundle's context and 8-bit KV cache and appends
its result line to <bundle>/results/.

  nohup python3 pi/panther_pi.py <hf user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &
      Wait for our winners on Hugging Face; download each, check it, post the result to the live page, delete the
      model file (the SD card holds one at a time). The check: checksum, a 256-token prompt and 64 generated tokens,
      real peak RAM against the prediction, with weight repacking off (on ARM, repacking keeps a second copy of
      K-type weights: the 7B @ 5 GiB winner peaked at 6.86 GiB with it, 4.69 without).
  python3 pi/panther_pi.py runtime <bundle dir> <label> <llama-server> [server options...]
      The same check with a chosen server binary and options, to compare runtimes (mainline llama.cpp, ik_llama.cpp)
      on the same file; the model file is kept.
  python3 pi/panther_pi.py spec <bundle dir> <label> <llama-server> [server options...]
      Speculative decoding: five everyday chat prompts answered greedily (128 tokens each) with a draft model or
      not. Greedy speculative decoding must give the same text as without it, so each answer's hash is printed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLAMA = Path.home() / "panther-pi/llama.cpp/build/bin/llama-server"
PORT, THREADS, PROMPT, GEN, SPEC_GEN = 8090, 4, 256, 64, 128
PROMPTS = [
    "Explain how a refrigerator keeps food cold, in a few paragraphs.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that checks whether a string is a palindrome, with a docstring and two tests.",
    "A shop sells pencils at 3 for $1. How much do 27 pencils cost? Explain step by step.",
    "Give five tips for staying productive while working from home, with one sentence on each.",
]


def throttled() -> str:
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def manifest(d: Path, verify: bool = True) -> dict:
    m = json.loads((d / "manifest.json").read_text())
    if verify:
        h = hashlib.sha256()
        with open(d / m["file"], "rb") as fh:
            while c := fh.read(1 << 24):
                h.update(c)
        if h.hexdigest() != m["sha256"]:
            raise SystemExit(f"{m['file']}: checksum mismatch")
    return m


@contextlib.contextmanager
def serving(server, d: Path, m: dict, opts: list[str], log: str):
    """llama-server on the bundle's file, ready to answer. Yields ask(body) -> the server's reply, and peak() -> the
    server's peak RAM in GiB; the server is stopped at the end."""
    kv = m["kv_cache_type"].lower()
    srv = subprocess.Popen([str(server), "-m", str(d / m["file"]), "--port", str(PORT), "-c", str(m["context"]),
                            "-np", "1", "-t", str(THREADS), "-ctk", kv, "-ctv", kv, *opts],
                           stdout=open(d / log, "w"), stderr=subprocess.STDOUT)

    def ask(body: dict) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                                     json.dumps({**body, "temperature": 0, "cache_prompt": False}).encode(),
                                     {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=3600).read())

    def peak() -> float:
        for line in Path(f"/proc/{srv.pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**20
        return float("nan")

    try:
        for _ in range(900):
            if srv.poll() is not None:
                raise SystemExit(f"the server exited ({srv.returncode}); see {d / log}")
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200:
                    break
            except OSError:
                time.sleep(1)
        yield ask, peak
    finally:
        srv.terminate()
        srv.wait(60)


def write(d: Path, name: str, line: str) -> str:
    line = f"{time.strftime('%H:%M:%S')} {line}"
    (d / "results").mkdir(exist_ok=True)
    with open(d / f"results/{name}.log", "a") as f:
        f.write(line + "\n")
    return line


def check(d: Path, label: str = "Q", server: str | Path = LLAMA, opts: list[str] | None = None) -> str:
    """Speed and peak RAM of one bundle. Without options: the queue's check (results/queue.log); with them, a
    runtime comparison (results/runtimes.log)."""
    queue = opts is None
    m, t0 = manifest(d), throttled()
    with serving(server, d, m, ["-fa", "on", "--no-repack"] if queue else opts,
                 "server.log" if queue else f"server-{label}.log") as (ask, peak):
        t = ask({"prompt": "The history of the region is long. " * (PROMPT // 8), "n_predict": GEN,
                 "ignore_eos": True})["timings"]
        ram = peak()
    return write(d, "queue" if queue else "runtimes",
                 f"{label} {m['file']} t={THREADS} ({PROMPT}, {GEN}){'' if queue else ' ' + ' '.join(opts)}: decode "
                 f"{t['predicted_per_second']:.2f} t/s, prefill {t['prompt_per_second']:.2f} t/s, peak RAM {ram:.3f} "
                 f"GiB (predicted {m['predicted_ram_gib']:.3f}), throttled: {t0} -> {throttled()}")


def spec(d: Path, label: str, server: str, opts: list[str]) -> str:
    m, t0, rows = manifest(d, verify=False), throttled(), []
    with serving(server, d, m, opts, f"server-spec-{label}.log") as (ask, peak):
        for q in PROMPTS:
            r = ask({"prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                               f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", "n_predict": SPEC_GEN})
            t = r["timings"]
            rows.append({"n": t["predicted_n"], "ms": t["predicted_ms"], "pp": t["prompt_per_second"],
                         "draft": t.get("draft_n"), "accepted": t.get("draft_n_accepted"),
                         "text": hashlib.sha256(r["content"].encode()).hexdigest()[:10]})
        ram = peak()
    n, ms = sum(x["n"] for x in rows), sum(x["ms"] for x in rows)
    dn, da = sum(x["draft"] or 0 for x in rows), sum(x["accepted"] or 0 for x in rows)
    return write(d, "spec",
                 f"{label} {' '.join(opts)}: writing {1000 * n / ms:.2f} t/s over {n} tokens "
                 f"(per prompt {', '.join('%.2f' % (1000 * x['n'] / x['ms']) for x in rows)}), reading "
                 f"{sum(x['pp'] for x in rows) / len(rows):.2f} t/s, drafted {dn} accepted {da}"
                 f"{f' ({100 * da / dn:.0f}%)' if dn else ''}, peak RAM {ram:.3f} GiB, texts "
                 f"{' '.join(x['text'] for x in rows)}, throttled {t0} -> {throttled()}")


def queue(user: str, todo: list[str]) -> None:
    hf = str(Path.home() / "hf-venv/bin/hf")
    while todo:
        for c in list(todo):
            size, b = c.split("@")
            d = Path.home() / f"panther-pi/budget-{b}-{size}-v2"
            if subprocess.run([hf, "download", f"{user}/panther-{size}-{b}g-v2", "--local-dir", str(d), "--quiet"],
                              capture_output=True).returncode or not list(d.glob("*.gguf")):
                continue
            (d / "pi-result.txt").write_text(check(d) + "\n")
            print(c, (d / "pi-result.txt").read_text(), flush=True)
            subprocess.run([hf, "upload", f"{user}/panther-status", str(d / "pi-result.txt"), f"pi/{c}.txt"],
                           capture_output=True)
            for f in d.glob("*.gguf"):
                f.unlink()
            todo.remove(c)
        if todo:
            time.sleep(300)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] in ("runtime", "spec"):
        print((check if a[0] == "runtime" else spec)(Path(a[1]).expanduser(), a[2], a[3], a[4:]), flush=True)
    else:
        queue(a[0], a[1:])
