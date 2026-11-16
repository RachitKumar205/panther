"""Speculative decoding on the Pi: one bundle served with a chosen server and options (a draft model or not), five
everyday chat prompts answered greedily (128 tokens each), real peak RAM. Greedy speculative decoding must give the
same text as without it, so each answer's hash is printed to compare. Appends to <bundle>/results/spec.log.

  python3 pi/spec_check.py <bundle dir> <label> <llama-server> [server options...]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from panther_pi import PORT, THREADS, peak_gib, throttled

GEN = 128
PROMPTS = [
    "Explain how a refrigerator keeps food cold, in a few paragraphs.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that checks whether a string is a palindrome, with a docstring and two tests.",
    "A shop sells pencils at 3 for $1. How much do 27 pencils cost? Explain step by step.",
    "Give five tips for staying productive while working from home, with one sentence on each.",
]


def chat(q: str) -> str:
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def main() -> int:
    d, label, server, opts = Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3], sys.argv[4:]
    m = json.loads((d / "manifest.json").read_text())
    gguf, kv = d / m["file"], m["kv_cache_type"].lower()
    t0 = throttled()
    log = open(d / f"server-spec-{label}.log", "w")
    srv = subprocess.Popen([server, "-m", str(gguf), "--port", str(PORT), "-c", str(m["context"]), "-np", "1",
                            "-t", str(THREADS), "-ctk", kv, "-ctv", kv, *opts], stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        for _ in range(900):
            if srv.poll() is not None:
                raise SystemExit(f"{label}: the server exited ({srv.returncode}); see server-spec-{label}.log")
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200:
                    break
            except OSError:
                time.sleep(1)
        for q in PROMPTS:
            body = {"prompt": chat(q), "n_predict": GEN, "temperature": 0, "cache_prompt": False}
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
            t = r["timings"]
            rows.append({"n": t["predicted_n"], "ms": t["predicted_ms"], "pp": t["prompt_per_second"],
                         "draft": t.get("draft_n"), "accepted": t.get("draft_n_accepted"),
                         "text": hashlib.sha256(r["content"].encode()).hexdigest()[:10]})
        peak = peak_gib(srv.pid)
    finally:
        srv.terminate()
        srv.wait(60)
    n, ms = sum(x["n"] for x in rows), sum(x["ms"] for x in rows)
    dn, da = sum(x["draft"] or 0 for x in rows), sum(x["accepted"] or 0 for x in rows)
    line = (f"{time.strftime('%H:%M:%S')} {label} {' '.join(opts)}: writing {1000 * n / ms:.2f} t/s over {n} tokens "
            f"(per prompt {', '.join(f'{1000 * x['n'] / x['ms']:.2f}' for x in rows)}), reading "
            f"{sum(x['pp'] for x in rows) / len(rows):.2f} t/s, drafted {dn} accepted {da}"
            f"{f' ({100 * da / dn:.0f}%)' if dn else ''}, peak RAM {peak:.3f} GiB, texts {' '.join(x['text'] for x in rows)}, "
            f"throttled {t0} -> {throttled()}")
    (d / "results").mkdir(exist_ok=True)
    with open(d / "results/spec.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
