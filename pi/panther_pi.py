"""Raspberry Pi 5 check of a Panther bundle: sha256, one load, peak RAM against the budget, speed.

Standard library only; runs as an ordinary user with llama.cpp built on the Pi (pi/setup.sh).

    python3 pi/panther_pi.py --bundle ~/panther-pi/budget-7-14b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

R_MAX_GIB = 7.0
GIB = 1024**3
CTX = 4096
MEM_FLOOR_MIB = 256
TEMP_START_MAX_C = 70.0
COOL_WAIT_S = 300
PORT = 8090

# Workloads (PLAN.md §9): prompt tokens / generated tokens.
W_REF = (256, 64)


# --- small helpers -----------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_temp_c() -> float | None:
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return int(p.read_text().strip()) / 1000
    except (OSError, ValueError):
        return None


def throttled() -> str | None:
    """`vcgencmd get_throttled` (e.g. 'throttled=0x0'), or None if unavailable."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
        return out.strip().split("=", 1)[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


def proc_status_kib(pid: int, key: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def mem_available_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class MemSampler(threading.Thread):
    """Samples the server's VmHWM/VmRSS and the system's MemAvailable every 0.2 s."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid, self.stop_flag = pid, threading.Event()
        self.min_available_kib: int | None = None
        self.max_rss_kib = 0

    def run(self) -> None:
        while not self.stop_flag.is_set():
            avail = mem_available_kib()
            if avail is not None:
                self.min_available_kib = avail if self.min_available_kib is None else min(self.min_available_kib, avail)
            rss = proc_status_kib(self.pid, "VmRSS")
            if rss:
                self.max_rss_kib = max(self.max_rss_kib, rss)
            time.sleep(0.2)


# --- llama-server ------------------------------------------------------------------------------------------

class Server:
    def __init__(self, llama: Path, gguf: Path, threads: int, log: Path, port: int = PORT,
                 extra: tuple[str, ...] = (), ctx: int = CTX) -> None:
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [str(llama / "build" / "bin" / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1",
                    "--port", str(port), "-c", str(ctx), "-np", "1", "-t", str(threads), "--load-mode", "none",
                    "--no-webui", "-lv", "4", *extra]
        self.log_path = log
        self.proc: subprocess.Popen | None = None
        self.load_seconds: float | None = None

    def __enter__(self) -> "Server":
        self._log = open(self.log_path, "w")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT)
        while time.monotonic() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}); see {self.log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        self.load_seconds = time.monotonic() - t0
                        return self
            except OSError:
                pass
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become ready in 15 minutes")

    def __exit__(self, *_: object) -> None:
        assert self.proc is not None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._log.close()

    @property
    def pid(self) -> int:
        assert self.proc is not None
        return self.proc.pid

    def post(self, path: str, body: dict, timeout: float = 3600):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def tokenize(self, text: str) -> list[int]:
        with self.post("/tokenize", {"content": text}) as r:
            return json.loads(r.read())["tokens"]

    def generate(self, prompt_ids: list[int], n_predict: int) -> dict:
        """Greedy, streamed, ignore_eos so exactly n_predict tokens are generated. Returns the server's timings
        plus client-side token arrival times."""
        body = {"prompt": prompt_ids, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "top_k": 1,
                "seed": 1, "stream": True, "cache_prompt": False}
        t0 = time.monotonic()
        arrivals: list[float] = []
        timings: dict = {}
        text = []
        with self.post("/completion", body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                msg = json.loads(payload)
                if msg.get("content"):
                    arrivals.append(time.monotonic() - t0)
                    text.append(msg["content"])
                if msg.get("timings"):
                    timings = msg["timings"]
                if msg.get("stop"):
                    break
        return {"t_total": time.monotonic() - t0, "arrivals": arrivals, "timings": timings, "text": "".join(text)}


def run_stats(gen: dict) -> dict:
    """Speed metrics for one run (PLAN.md §2.2)."""
    a = gen["arrivals"]
    gaps = [b - x for x, b in zip(a, a[1:])]  # inter-token latency, excluding the first token (prefill)
    t = gen["timings"]
    return {
        "prefill_tps": t.get("prompt_per_second"),
        "decode_tps": t.get("predicted_per_second"),
        "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
        "itl_p50_ms": None if not gaps else 1000 * percentile(gaps, 0.5),
        "itl_p95_ms": None if not gaps else 1000 * percentile(gaps, 0.95),
        "e2e_s": gen["t_total"], "tokens_streamed": len(a),
    }


# --- the queue ---------------------------------------------------------------------------------------------

class Queue:
    def __init__(self, bundle: Path, llama: Path) -> None:
        self.bundle, self.llama = bundle, llama
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        self.out = bundle / "results"
        self.out.mkdir(exist_ok=True)
        self.runs_path = self.out / "runs.jsonl"
        self.done = set()
        if self.runs_path.exists():
            for line in self.runs_path.read_text().splitlines():
                r = json.loads(line)
                self.done.add(r["run_key"])
        self.agreement_text = (bundle / self.manifest["agreement_text"]).read_text()
        self.r_max = float(self.manifest.get("r_max_gib", R_MAX_GIB))
        self.ctx = int(self.manifest.get("context_tokens", CTX))
        kv = self.manifest.get("kv_cache_type", "F16").lower()
        # A quantized V cache needs flash attention in llama.cpp.
        self.server_extra = () if kv == "f16" else ("-ctk", kv, "-ctv", kv, "-fa", "on")

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.out / "queue.log", "a") as f:
            f.write(line + "\n")

    def record(self, rec: dict) -> None:
        with open(self.runs_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done.add(rec["run_key"])

    def cool_down(self) -> float | None:
        t0 = time.monotonic()
        while True:
            t = cpu_temp_c()
            if t is None or t <= TEMP_START_MAX_C or time.monotonic() - t0 > COOL_WAIT_S:
                return t
            time.sleep(10)

    def verify_bundle(self, names: list[str] | None = None) -> None:
        for name, f in self.manifest["files"].items():
            if names is not None and name not in names:
                continue
            key = f"verify:{name}"
            if key in self.done:
                continue
            got = sha256(self.bundle / name)
            ok = got == f["sha256"]
            self.record({"run_key": key, "step": "E2", "file": name, "sha256_ok": ok})
            self.log(f"E2 {name}: sha256 {'ok' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(f"{name} is corrupt; copy it again")

    def measure(self, step: str, name: str, threads: int, workload: tuple[int, int], warmup: int, reps: int) -> None:
        key = f"{step}:{name}:t{threads}:{workload[0]}/{workload[1]}"
        if key in self.done:
            return
        temp_start = self.cool_down()
        logf = self.out / f"server-{step}-{Path(name).stem}-t{threads}.log"
        throttle_start = throttled()
        with Server(self.llama, self.bundle / name, threads, logf, extra=self.server_extra, ctx=self.ctx) as srv:
            ids = srv.tokenize(self.agreement_text)
            prompt = ids[:workload[0]]
            sampler = MemSampler(srv.pid)
            sampler.start()
            runs = []
            for i in range(warmup + reps):
                g = srv.generate(prompt, workload[1])
                st = run_stats(g)
                st["warmup"] = i < warmup
                runs.append(st)
            sampler.stop_flag.set()
            sampler.join()
            vmhwm = proc_status_kib(srv.pid, "VmHWM")
            load_s = srv.load_seconds
        f = self.manifest["files"][name]
        peak = (vmhwm or 0) * 1024
        rec = {
            "run_key": key, "step": step, "file": name, "model": f["model"], "role": f["role"], "threads": threads,
            "workload": list(workload), "runs": runs, "load_s": load_s,
            "peak_ram_bytes": peak, "peak_ram_gib": peak / GIB, "predicted_ram_gib": f["predicted_ram_gib"],
            "ram_error_pct": 100 * (peak / GIB - f["predicted_ram_gib"]) / f["predicted_ram_gib"] if peak else None,
            "measured_overhead_gib": (peak - f["predicted"]["weights"] - f["predicted"]["kv_cache"]) / GIB if peak else None,
            "min_mem_available_mib": None if sampler.min_available_kib is None else sampler.min_available_kib / 1024,
            "temp_start_c": temp_start, "temp_end_c": cpu_temp_c(),
            "throttled_start": throttle_start, "throttled_end": throttled(),
            "type_counts": parse_type_counts(logf.read_text(errors="replace")),
            "expected_type_counts": f["expected_type_counts"], "r_max_gib": self.r_max,
        }
        rec["valid"], rec["invalid_reasons"] = validity(rec)
        self.record(rec)
        meas = [r for r in runs if not r["warmup"]]
        self.log(f"{step} {name} t={threads} {workload}: decode {mean([r['decode_tps'] for r in meas]):.2f} t/s, "
                 f"prefill {mean([r['prefill_tps'] for r in meas]):.2f} t/s, peak RAM {rec['peak_ram_gib']:.3f} GiB "
                 f"(predicted {f['predicted_ram_gib']:.3f}), {'not throttled' if rec['valid'] else 'throttled: ' + ', '.join(rec['invalid_reasons'])}")

def fmt(x: float | None, spec: str) -> str:
    """Format a number for the report; missing or NaN values print as n/a instead of crashing."""
    return "n/a" if x is None or x != x else format(x, spec)


def mean(xs: list) -> float:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def parse_type_counts(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        if "- type" in line and "tensors" in line:
            parts = line.split("- type", 1)[1].replace(":", " ").split()
            try:
                out[parts[0].lower()] = int(parts[1])
            except (IndexError, ValueError):
                pass
    return out


def validity(rec: dict) -> tuple[bool, list[str]]:
    reasons = []
    r_max = rec.get("r_max_gib", R_MAX_GIB)
    if rec["peak_ram_gib"] > r_max:
        reasons.append(f"peak RAM {rec['peak_ram_gib']:.3f} GiB > {r_max}")
    if rec["min_mem_available_mib"] is not None and rec["min_mem_available_mib"] < MEM_FLOOR_MIB:
        reasons.append(f"MemAvailable fell to {rec['min_mem_available_mib']:.0f} MiB")
    if rec["temp_start_c"] is not None and rec["temp_start_c"] > TEMP_START_MAX_C:
        reasons.append(f"start temperature {rec['temp_start_c']:.1f} C")
    for k in ("throttled_start", "throttled_end"):
        if rec[k] not in (None, "0x0"):
            reasons.append(f"{k} {rec[k]}")
    if rec["type_counts"] and rec["type_counts"] != rec["expected_type_counts"]:
        reasons.append("tensor types differ from the bundle manifest")
    if any(r["predicted_n"] not in (None, rec["workload"][1]) for r in rec["runs"]):
        reasons.append("did not generate the requested token count")
    return (not reasons), reasons


def cmd_check(a: argparse.Namespace) -> int:
    """Every file in the bundle: sha256, one load, peak RAM vs the budget, speed (256-token prompt, 64 out,
    1 warm-up + 2 measured, 4 threads). Writes results/report.md."""
    q = Queue(a.bundle, a.llama)
    for name in q.manifest["files"]:
        q.verify_bundle([name])
        try:
            q.measure("Q", name, 4, W_REF, warmup=1, reps=2)
        except (RuntimeError, TimeoutError, OSError) as e:
            q.log(f"{name}: FAILED ({e}); see results/server-Q-*.log")
    lines = ["| File | Peak RAM GiB (predicted) | Budget GiB | Decode t/s | Prefill t/s | Overhead GiB | Throttled |",
             "|---|---|---|---|---|---|---|"]
    for line in q.runs_path.read_text().splitlines():
        r = json.loads(line)
        if "runs" not in r:
            continue
        meas = [x for x in r["runs"] if not x["warmup"]] or r["runs"]
        lines.append(f"| {r['file']} | {fmt(r['peak_ram_gib'], '.3f')} ({fmt(r['predicted_ram_gib'], '.3f')}) | "
                     f"{q.r_max:.3f} | {fmt(mean([x['decode_tps'] for x in meas]), '.2f')} | "
                     f"{fmt(mean([x['prefill_tps'] for x in meas]), '.2f')} | {fmt(r['measured_overhead_gib'], '.3f')} | "
                     f"{'no' if r['valid'] else 'yes: ' + '; '.join(r['invalid_reasons'])} |")
    (q.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")  # "run" (older relay calls also pass --quick)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bundle", type=Path, default=Path.home() / "panther-pi" / "bundle")
    ap.add_argument("--llama", type=Path, default=Path.home() / "panther-pi" / "llama.cpp")
    a = ap.parse_args(argv)
    a.bundle, a.llama = a.bundle.expanduser(), a.llama.expanduser()
    if os.geteuid() == 0:
        sys.exit("run this as an ordinary user, not root")
    return cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
