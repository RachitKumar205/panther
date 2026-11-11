"""Raspberry Pi 5 measurement queue for the Panther proof of concept (docs/PIPELINE.md, Phase E).

Standard library only. Runs as an ordinary user. Uses llama.cpp built on the Pi at the pinned commit.

    python3 pi/panther_pi.py run     --bundle ~/panther-pi/bundle --llama ~/panther-pi/llama.cpp
    python3 pi/panther_pi.py analyze --bundle ~/panther-pi/bundle

Results go to <bundle>/results/: runs.jsonl (one line per measured run), log files, report.md, report.json.
The queue is resumable: runs already in runs.jsonl are skipped.
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
W_SHORT = (128, 32)
W_REF = (256, 64)
W_GEN = [(128, 32), (1024, 128)]


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
    def __init__(self, llama: Path, gguf: Path, threads: int, log: Path, port: int = PORT) -> None:
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [str(llama / "build" / "bin" / "llama-server"), "-m", str(gguf), "--host", "127.0.0.1",
                    "--port", str(port), "-c", str(CTX), "-np", "1", "-t", str(threads), "--load-mode", "none",
                    "--no-webui", "-lv", "4"]
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

    def verify_bundle(self) -> None:
        for name, f in self.manifest["files"].items():
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
        with Server(self.llama, self.bundle / name, threads, logf) as srv:
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
            "expected_type_counts": f["expected_type_counts"],
        }
        rec["valid"], rec["invalid_reasons"] = validity(rec)
        self.record(rec)
        meas = [r for r in runs if not r["warmup"]]
        self.log(f"{step} {name} t={threads} {workload}: decode {mean([r['decode_tps'] for r in meas]):.2f} t/s, "
                 f"prefill {mean([r['prefill_tps'] for r in meas]):.2f} t/s, peak RAM {rec['peak_ram_gib']:.3f} GiB "
                 f"(predicted {f['predicted_ram_gib']:.3f}), {'valid' if rec['valid'] else 'INVALID: ' + ', '.join(rec['invalid_reasons'])}")

    def agreement(self, name: str, threads: int) -> None:
        key = f"E6:{name}:t{threads}"
        if key in self.done:
            return
        self.cool_down()
        cmd = [str(self.llama / "build/bin/llama-perplexity"), "-m", str(self.bundle / name), "-f",
               str(self.bundle / self.manifest["agreement_text"]), "-c", "2048", "-b", "2048", "-t", str(threads)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = out.stdout + out.stderr
        (self.out / f"agreement-{Path(name).stem}.log").write_text(text)
        ppl = parse_final_ppl(text)
        ref = self.manifest["files"][name]["server_cpu_ppl_agreement"]
        rel = None if ppl is None else abs(ppl - ref) / ref
        self.record({"run_key": key, "step": "E6", "file": name, "threads": threads, "pi_ppl": ppl,
                     "server_cpu_ppl": ref, "rel_diff": rel, "pass": rel is not None and rel <= 0.005})
        diff = "n/a" if rel is None else "%.3f%%" % (100 * rel)
        self.log(f"E6 {name}: Pi PPL {ppl} vs server CPU {ref:.4f} ({diff})")


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


def parse_final_ppl(text: str) -> float | None:
    for line in text.splitlines():
        if "Final estimate: PPL =" in line:
            try:
                return float(line.split("PPL =", 1)[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def validity(rec: dict) -> tuple[bool, list[str]]:
    reasons = []
    if rec["peak_ram_gib"] > R_MAX_GIB:
        reasons.append(f"peak RAM {rec['peak_ram_gib']:.3f} GiB > {R_MAX_GIB}")
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


def cmd_run(a: argparse.Namespace) -> int:
    q = Queue(a.bundle, a.llama)
    deadline = time.monotonic() + a.hours * 3600
    files = q.manifest["files"]
    winners = [n for n, f in files.items() if f["role"] == "winner"]
    q.log(f"queue start: {len(files)} files, budget {a.hours} h")
    q.verify_bundle()  # E2
    for name in files:  # E3: loads, generates, RAM vs prediction
        q.measure("E3", name, 4, W_SHORT, warmup=0, reps=1)
    best = {}
    for name in winners:  # E4: thread sweep
        for t in (2, 3, 4):
            if time.monotonic() > deadline:
                break
            q.measure("E4", name, t, W_SHORT, warmup=1, reps=2)
        best[name] = best_threads(q, name)
    t_best = max(set(best.values()), key=list(best.values()).count) if best else 4
    q.log(f"E4 best threads: {best} -> using {t_best}")
    for name in files:  # E5: confirmation
        if time.monotonic() < deadline:
            q.measure("E5", name, t_best, W_REF, warmup=1, reps=3)
    for name in winners:  # E6: agreement with the server CPU build
        if time.monotonic() < deadline:
            q.agreement(name, t_best)
    rec = q.manifest["recommended_file"]
    for w in W_GEN:  # E7: generalisation, recommended file only
        if time.monotonic() < deadline:
            q.measure("E7", rec, t_best, w, warmup=1, reps=3)
    q.log("queue done" if time.monotonic() < deadline else "queue stopped: time budget used")
    return analyze(a.bundle)


def best_threads(q: Queue, name: str) -> int:
    best, best_tps = 4, -1.0
    for line in q.runs_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("step") == "E4" and r.get("file") == name and r.get("valid"):
            tps = mean([x["decode_tps"] for x in r["runs"] if not x["warmup"]])
            if tps > best_tps:
                best, best_tps = r["threads"], tps
    return best


def analyze(bundle: Path) -> int:
    manifest = json.loads((bundle / "manifest.json").read_text())
    out = bundle / "results"
    recs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()] if (out / "runs.jsonl").exists() else []
    rows, ram_checks, agree = [], [], []
    for r in recs:
        if r.get("step") in ("E3", "E5", "E7") and "runs" in r:
            meas = [x for x in r["runs"] if not x["warmup"]] or r["runs"]
            dec = [x["decode_tps"] for x in meas if x["decode_tps"]]
            rows.append({"step": r["step"], "file": r["file"], "threads": r["threads"], "workload": r["workload"],
                         "prefill_tps": mean([x["prefill_tps"] for x in meas]), "decode_tps": mean(dec),
                         "decode_range": [min(dec), max(dec)] if dec else None,
                         "itl_p50_ms": mean([x["itl_p50_ms"] for x in meas]),
                         "itl_p95_ms": mean([x["itl_p95_ms"] for x in meas]),
                         "e2e_s": mean([x["e2e_s"] for x in meas]), "peak_ram_gib": r["peak_ram_gib"],
                         "predicted_ram_gib": r["predicted_ram_gib"], "ram_error_pct": r["ram_error_pct"],
                         "measured_overhead_gib": r["measured_overhead_gib"], "valid": r["valid"],
                         "invalid_reasons": r["invalid_reasons"]})
            if r["ram_error_pct"] is not None:
                ram_checks.append(abs(r["ram_error_pct"]) <= 2.0)
        if r.get("step") == "E6":
            agree.append(r)
    rec_file = manifest["recommended_file"]
    rec_rows = [x for x in rows if x["file"] == rec_file and x["step"] == "E5"]
    rec_ok = bool(rec_rows) and all(x["valid"] and x["peak_ram_gib"] <= R_MAX_GIB for x in rec_rows)
    report = {"recommended_file": rec_file, "recommended_reason": manifest["recommended_reason"],
              "recommended_confirmed_on_pi": rec_ok, "ram_prediction_within_2pct": all(ram_checks) if ram_checks else None,
              "rows": rows, "agreement": agree, "verdict": manifest["verdict"]}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    lines = ["# Panther proof of concept: Raspberry Pi 5 results", "",
             f"Recommended: **{rec_file}** ({manifest['recommended_reason']}); confirmed on the Pi: "
             f"**{'yes' if rec_ok else 'no'}**.", "",
             "| Step | File | Threads | Prompt/gen | Prefill t/s | Decode t/s (range) | ITL p50/p95 ms | E2E s | "
             "Peak RAM GiB (pred.) | RAM err % | Valid |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        rng = f" ({x['decode_range'][0]:.2f}–{x['decode_range'][1]:.2f})" if x["decode_range"] else ""
        lines.append(f"| {x['step']} | {x['file']} | {x['threads']} | {x['workload'][0]}/{x['workload'][1]} | "
                     f"{x['prefill_tps']:.2f} | {x['decode_tps']:.2f}{rng} | {x['itl_p50_ms']:.0f}/{x['itl_p95_ms']:.0f} | "
                     f"{x['e2e_s']:.1f} | {x['peak_ram_gib']:.3f} ({x['predicted_ram_gib']:.3f}) | "
                     f"{x['ram_error_pct']:+.2f} | {'yes' if x['valid'] else 'no: ' + '; '.join(x['invalid_reasons'])} |")
    lines += ["", "## Agreement with the server CPU build (4,096-token text)", ""]
    for g in agree:
        diff = "n/a" if g["rel_diff"] is None else "%.3f%%" % (100 * g["rel_diff"])
        verdict = "pass" if g["pass"] else "FAIL"
        lines.append(f"- {g['file']}: Pi PPL {g['pi_ppl']}, server CPU {g['server_cpu_ppl']:.4f}, "
                     f"difference {diff} ({verdict}, limit 0.5%)")
    lines += ["", "## Server verdict (test subset)", "", "```", json.dumps(manifest["verdict"], indent=1), "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(rows) + 2]))
    print(f"\nreport: {out / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--bundle", type=Path, default=Path.home() / "panther-pi" / "bundle")
    ap.add_argument("--llama", type=Path, default=Path.home() / "panther-pi" / "llama.cpp")
    ap.add_argument("--hours", type=float, default=10.0)
    a = ap.parse_args(argv)
    a.bundle, a.llama = a.bundle.expanduser(), a.llama.expanduser()
    if os.geteuid() == 0:
        sys.exit("run this as an ordinary user, not root")
    return cmd_run(a) if a.cmd == "run" else analyze(a.bundle)


if __name__ == "__main__":
    sys.exit(main())
