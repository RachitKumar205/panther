"""The Pi's check of one bundle (panther_pi.check: checksum, llama-server at the bundle's context with its 8-bit KV
cache, a 256-token prompt and 64 generated tokens, real peak RAM) with a chosen server binary and options, to compare
runtimes (mainline llama.cpp, ik_llama.cpp) on the same file. Prints and appends the result line to
<bundle>/results/runtimes.log; the model file is kept.

  python3 pi/runtime_check.py <bundle dir> <label> <llama-server> [server options...]
  python3 pi/runtime_check.py ~/panther-pi/budget-5-14b-v2-qat ik ~/panther-pi/ik/ik_llama.cpp/build/bin/llama-server -fa
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from panther_pi import GEN, PORT, PROMPT, THREADS, peak_gib, throttled


def main() -> int:
    d, label, server, opts = Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3], sys.argv[4:]
    m = json.loads((d / "manifest.json").read_text())
    gguf = d / m["file"]
    h = hashlib.sha256()
    with open(gguf, "rb") as fh:
        while c := fh.read(1 << 24):
            h.update(c)
    if h.hexdigest() != m["sha256"]:
        raise SystemExit(f"{gguf.name}: checksum mismatch")
    kv = m["kv_cache_type"].lower()
    t0 = throttled()
    log = open(d / f"server-{label}.log", "w")
    srv = subprocess.Popen([server, "-m", str(gguf), "--port", str(PORT), "-c", str(m["context"]), "-np", "1",
                            "-t", str(THREADS), "-ctk", kv, "-ctv", kv, *opts], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            if srv.poll() is not None:
                raise SystemExit(f"{label}: the server exited ({srv.returncode}); see server-{label}.log")
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
    line = (f"{time.strftime('%H:%M:%S')} {label} {gguf.name} t={THREADS} ({PROMPT}, {GEN}) {' '.join(opts)}: decode "
            f"{t['predicted_per_second']:.2f} t/s, prefill {t['prompt_per_second']:.2f} t/s, peak RAM {peak:.3f} GiB "
            f"(predicted {m['predicted_ram_gib']:.3f}), throttled: {t0} -> {throttled()}")
    (d / "results").mkdir(exist_ok=True)
    with open(d / "results/runtimes.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
