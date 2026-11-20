"""The Pi's queue: `nohup python3 pi/panther_pi.py <hf user> 7b@3 14b@5 > ~/panther-pi/pi.log 2>&1 &`.

Waits for each contest's winner on Hugging Face, downloads it, checks it, posts the result line (also when the check
fails: the queue goes on), deletes the model file (the SD card holds one at a time). Details: pi/README.md.
"""

import contextlib
import hashlib
import json
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

LLAMA = Path.home() / "panther-pi/llama.cpp/build/bin/llama-server"
HF = str(Path.home() / "hf-venv/bin/hf")
PORT, THREADS, PROMPT, GEN, HEALTH_TRIES = 8090, 4, 256, 64, 900


class Failed(Exception):  # a failed check: its message is the contest's result line
    pass


def throttled() -> str:
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def manifest(d: Path) -> dict:
    m = json.loads((d / "manifest.json").read_text())
    with open(d / m["file"], "rb") as fh:
        if hashlib.file_digest(fh, "sha256").hexdigest() != m["sha256"]:
            raise Failed(f"{m['file']}: checksum mismatch")
    return m


def peak(pid: int) -> float:
    """A process's peak RAM in GiB."""
    return next(int(x.split()[1]) / 2**20 for x in Path(f"/proc/{pid}/status").read_text().splitlines()
                if x.startswith("VmHWM:"))


def check(d: Path) -> str:
    """Speed and peak RAM of one bundle (llama-server at its context and 8-bit KV cache, weight repacking off: on ARM
    it keeps a second copy of K-type weights, 6.86 GiB peak for the 7B @ 5 GiB winner, 4.69 without): its result line."""
    m, t0, log = manifest(d), throttled(), d / "server.log"
    kv = m["kv_cache_type"].lower()
    with open(log, "w") as out, subprocess.Popen(
            [str(LLAMA), "-m", str(d / m["file"]), "--port", str(PORT), "-c", str(m["context"]), "-np", "1",
             "-t", str(THREADS), "-ctk", kv, "-ctv", kv, "-fa", "on", "--no-repack"],
            stdout=out, stderr=subprocess.STDOUT) as srv:
        try:
            for _ in range(HEALTH_TRIES):
                if srv.poll() is not None:
                    raise Failed(f"the server exited ({srv.returncode}) before it was ready; see {log}")
                with contextlib.suppress(OSError):  # not up yet, or still loading (503)
                    if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200:
                        break
                time.sleep(1)
            else:
                raise Failed(f"the server was not healthy after {HEALTH_TRIES} tries; see {log}")
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(
                {"prompt": "The history of the region is long. " * (PROMPT // 8), "n_predict": GEN,
                 "ignore_eos": True, "temperature": 0, "cache_prompt": False}).encode(),
                {"Content-Type": "application/json"})
            try:
                t = json.loads(urllib.request.urlopen(req, timeout=3600).read())["timings"]
            except OSError as e:
                raise Failed(f"the server stopped answering ({e}; exit code {srv.poll()}); see {log}")
            ram = peak(srv.pid)
        finally:
            srv.terminate()
            srv.wait(60)
    b = m["budget_gib"]  # FITS/OVER is reported; the verdict does not depend on it
    return (f"Q {m['file']} t={THREADS} ({PROMPT}, {GEN}): decode {t['predicted_per_second']:.2f} t/s, prefill "
            f"{t['prompt_per_second']:.2f} t/s, peak RAM {ram:.3f} GiB (predicted {m['predicted_ram_gib']:.3f}), "
            f"{'FITS' if ram <= b else 'OVER'} {b} GiB, throttled: {t0} -> {throttled()}")


def queue(user: str, todo: list[str]) -> None:
    while todo:
        for c in list(todo):
            size, b = c.split("@")
            d = Path.home() / f"panther-pi/budget-{b}-{size}-v2"
            if subprocess.run([HF, "download", f"{user}/panther-{size}-{b}g-v2", "--local-dir", str(d), "--quiet"],
                              capture_output=True).returncode or not list(d.glob("*.gguf")):
                continue  # not published yet
            try:
                line = check(d)
            except Exception as e:  # any failed check is this contest's result; the queue goes on
                traceback.print_exc()
                print(f"FAILED: {c}: {e}", file=sys.stderr, flush=True)
                line = f"FAILED: {e}"
            line = f"{time.strftime('%H:%M:%S')} {line}"
            (d / "results").mkdir(exist_ok=True)
            with open(d / "results/queue.log", "a") as f:
                f.write(line + "\n")
            (d / "pi-result.txt").write_text(line + "\n")
            print(c, line, flush=True)
            subprocess.run([HF, "upload", f"{user}/panther-status", str(d / "pi-result.txt"), f"pi/{c}.txt"],
                           stdout=subprocess.DEVNULL, check=True)  # fails loudly (e.g. no write token)
            for f in d.glob("*.gguf"):
                f.unlink()
            todo.remove(c)
        if todo:
            time.sleep(300)


if __name__ == "__main__":
    queue(sys.argv[1], sys.argv[2:])
