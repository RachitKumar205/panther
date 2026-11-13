"""Raspberry Pi 5 check of a winner's bundle: checksum, then llama-server at the full context with an 8-bit KV
cache; one 256-token prompt and 64 generated tokens; real peak RAM against the prediction. Standard library only.

  python3 pi/panther_pi.py run --bundle ~/panther-pi/budget-5-14b-v2
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LLAMA = Path.home() / "panther-pi/llama.cpp/build/bin/llama-server"
PORT, THREADS, PROMPT, GEN = 8090, 4, 256, 64


def peak_gib(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2**20
    return float("nan")


def throttled() -> str:
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main() -> int:
    d = Path(sys.argv[sys.argv.index("--bundle") + 1]).expanduser()
    m = json.loads((d / "manifest.json").read_text())
    if "files" in m:  # bundles of the first methods
        f = m["recommended_file"]
        m = {"file": f, "sha256": m["files"][f]["sha256"], "predicted_ram_gib": m["files"][f]["predicted_ram_gib"],
             "context": m["context_tokens"], "kv_cache_type": m["kv_cache_type"]}
    gguf = d / m["file"]
    h = hashlib.sha256()
    with open(gguf, "rb") as fh:
        while c := fh.read(1 << 24):
            h.update(c)
    if h.hexdigest() != m["sha256"]:
        raise SystemExit(f"{gguf.name}: checksum mismatch")
    kv = m["kv_cache_type"].lower()
    t0 = throttled()
    srv = subprocess.Popen([str(LLAMA), "-m", str(gguf), "--port", str(PORT), "-c", str(m["context"]), "-np", "1",
                            "-t", str(THREADS), "-ctk", kv, "-ctv", kv, "-fa", "on"],
                           stdout=open(d / "server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200:
                    break
            except OSError:
                time.sleep(1)
        body = {"prompt": "The history of the region is long. " * (PROMPT // 8), "n_predict": GEN, "temperature": 0,
                "ignore_eos": True, "cache_prompt": False}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = json.loads(urllib.request.urlopen(req, timeout=3600).read())["timings"]
        peak = peak_gib(srv.pid)
    finally:
        srv.terminate()
        srv.wait(60)
    line = (f"{time.strftime('%H:%M:%S')} Q {gguf.name} t={THREADS} ({PROMPT}, {GEN}): decode "
            f"{t['predicted_per_second']:.2f} t/s, prefill {t['prompt_per_second']:.2f} t/s, peak RAM {peak:.3f} GiB "
            f"(predicted {m['predicted_ram_gib']:.3f}), throttled: {t0} -> {throttled()}")
    (d / "results").mkdir(exist_ok=True)
    with open(d / "results/queue.log", "a") as f:
        f.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
