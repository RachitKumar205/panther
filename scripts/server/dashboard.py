"""Panther's dashboard: a self-hosted, read-only web page of the whole pipeline, from data/ only (no outside service).

  python scripts/server/dashboard.py [--data DIR] [--port 43800]      then open http://localhost:43800

GET /            the page (scripts/server/dashboard/)
GET /api/logs?n=3000     the last n lines of every log, oldest first: [{"src": "main" | "b5" ..., "line": ...}]
GET /api/logs/stream     server-sent events: each new log line as it is written (checked every second)
GET /api/state   everything the page shows, as JSON: contests and their steps, verdicts and held-out tables, the
                 Pi's results, the QAT finalists block by block, GPU memory and the runs' claims, disk, events and
                 notes; "problems": files that could not be read (the rest of the page still shows).

The Pi's results come from pi/<contest>.txt in the public Hugging Face repo <user>/panther-status (user: the part of
PANTHER_HF_REPO before "/"), fetched every 5 minutes into the temporary directory, merged over data/pi/*.txt (older
results, no longer written; the fetched copy wins).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import yaml

DATA = Path(os.environ.get("PANTHER_DATA", "/work/data"))
STEPS = ["store", "error", "refs", "preset", "ga", "final", "bundle"]
WHAT = {"store": "16-bit original, importance matrix, the model in every type", "error": "offline error per matrix",
        "refs": "the original's answers and reference outputs", "preset": "llama.cpp's preset (competitor)",
        "ga": "genetic search (score: flips + 10 × KL, relative to the preset)",
        "final": "6 finalists scored, guard, held-out verdict vs the preset", "bundle": "file for the Pi"}
MINUTES = {"store": 5, "error": 0.2, "refs": 3, "preset": 4, "ga": 10, "final": 14, "bundle": 0.3}  # x size in B
DESCRIBE = {
    "store": "the original as a 16-bit file; its importance matrix (on the calibration web text); the whole model in"
             " each of 14 types (IQ1_S to Q8_0), every matrix with error-compensated rounding",
    "error": "for every matrix and type, a cheap estimate of the damage (guides the search's repairs and proposals;"
             " every genome is still measured)",
    "refs": "the original answers the dataset's chat prompts (train, select, test); its next-word probabilities on"
            " the select and test text (chat + web) and which questions it gets right are recorded",
    "preset": "every llama.cpp preset that fits the budget is scored on the selection questions; the most accurate"
              " is the competitor (none fits: the smallest)",
    "ga": "genetic search: genomes (one type per matrix, inside the RAM budget) scored by flips % + 10 × KL (flips on"
          " 900 selection questions, KL on the select split), relative to the preset; 70 genomes, each recorded with"
          " its origin (start, mutation, crossover, solver)",
    "final": "the best 3 built distilled and undistilled (6 files), scored on all selection questions and the select"
             " split, the lowest score wins; a warning if it is clearly worse than the preset (guard); then the"
             " verdict on held-out data: relative score (100 = the preset's damage, lower is better), WIN, TIE or LOSE",
    "bundle": "the winning file packed for the Pi, which downloads it and posts its real peak RAM (FITS or OVER the"
              " budget) and speed"}
PROBLEMS: list[str] = []  # this state's unreadable files, shown on the page
STATE = threading.Lock()


def problem(f: Path | str, e: Exception | str) -> None:
    PROBLEMS.append(f"{Path(f).relative_to(DATA) if Path(f).is_relative_to(DATA) else f}: "
                    f"{e if isinstance(e, str) else type(e).__name__ + ': ' + str(e)}")


def clock(line: str) -> int:
    h, m = line[:5].split(":")
    return int(h) * 60 + int(m)


def lanes() -> list[tuple[int, str, list[str]]]:
    """Every contest's progress lines (budget, size, lines), from its latest `step store: start` on (this run)."""
    out = []
    for f in sorted(DATA.glob("logs/progress-b*.log")):
        b, lines = int(f.stem[10:]), f.read_text(errors="replace").splitlines()
        for sm in dict.fromkeys(m.split()[1] for m in lines if re.match(r"\d\d:\d\d \S+ step \w+: start", m)):
            ls = [x for x in lines if x[6:].startswith(sm + " ")]
            out.append((b, sm, ls[max((i for i, x in enumerate(ls) if " step store: start" in x), default=0):]))
    return out


def winner(b: int, sm: str) -> tuple[dict, Path] | None:
    f = next(iter((DATA / "rewrite" / f"budget-{b}").glob(f"winner-*-{sm}-*-v2.json")), None)
    try:
        return (json.loads(f.read_text()), f) if f else None
    except (OSError, ValueError) as e:
        problem(f, e)
        return None


def searched(b: int, sm: str) -> bool:
    """True if the config still searches a model of this size at this budget."""
    c = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/search.yaml").read_text())
    return any(f"-{sm}-" in m for m in c["contests"].get(b, {}).get("searched", []))


STATIC = Path(__file__).with_name("dashboard")
NUM = r"([-+]?[\d.]+)"
STREAMS = threading.BoundedSemaphore(16)  # live log streams at once: each holds a thread for as long as it is open


def site_names() -> list[tuple[re.Pattern, str]]:
    """What the page never shows, since it may be public: the server's account and address, and the domain of the
    service it shares the machine with, all from the git-ignored deploy/server/local.env."""
    f = Path(__file__).resolve().parents[2] / "deploy/server/local.env"
    env = dict(re.findall(r"^\s*(PANTHER_\w+)=(\S*)", f.read_text(), re.M)) if f.exists() else {}
    user, _, addr = env.get("PANTHER_HOST", "").rpartition("@")
    host = urlsplit(env.get("PANTHER_HEALTH_URL", "")).hostname or ""
    domain = ".".join(host.split(".")[-2:]) if host.count(".") >= 1 else host
    out = [(rf"(?<![\w.-]){re.escape(v)}(?![\w-])", label) for v, label in
           ((env.get("PANTHER_USER", ""), "svcuser"), (user, "svcuser"), (addr, "server")) if v]
    if domain:
        out.append((rf"(?:[\w-]+\.)*{re.escape(domain)}(?![\w-])", "service"))
    return [(re.compile(p, re.I), label) for p, label in out]


MASK = site_names()


def mask(text: str) -> str:
    for p, label in MASK:
        text = p.sub(label, text)
    return text


def ist() -> time.struct_time:
    return time.gmtime(time.time() + 19800)


def logs() -> tuple[list[str], list[tuple[int, str, list[str]]]]:
    f = DATA / "logs/panther.log"
    return (f.read_text().splitlines() if f.exists() else []), lanes()


def dated(f: Path, n: int) -> list[tuple[int, str]]:
    """The file's last n timed lines with an absolute minute: lines carry only HH:MM (IST), so walking back from
    the file's last write, a clock that goes up means the day before."""
    if not f.exists():
        return []
    lines = [x for x in f.read_text().splitlines() if re.match(r"\d\d:\d\d ", x)][-n:]
    t = time.gmtime(f.stat().st_mtime + 19800)
    day, prev, out = int((f.stat().st_mtime + 19800) // 86400), t.tm_hour * 60 + t.tm_min, []
    for x in reversed(lines):
        if clock(x) > prev:
            day -= 1
        prev = clock(x)
        out.append((day * 1440 + prev, x))
    return out


def log_files() -> list[Path]:
    return [DATA / "logs/panther.log", *sorted(DATA.glob("logs/progress-b*.log"))]


def source(f: Path) -> str:
    return "main" if f.name == "panther.log" else f.stem.removeprefix("progress-")


def backlog(n: int) -> list[dict]:
    """The last n timed lines of every log, oldest first, each with the log it came from."""
    rows, seen = [], set()
    for k, f in enumerate(log_files()):
        for i, (m, x) in enumerate(reversed(dated(f, n))):
            if (m, x) not in seen:  # a line written to two logs shows once
                seen.add((m, x))
                rows.append((m, k, i, {"src": source(f), "line": x}))
    return [r[3] for r in sorted(rows, key=lambda r: r[:3])][-n:]


def events(n: int) -> list[str]:
    """The newest n lines of the main log and the progress logs together, newest first."""
    rows = [r for f in [DATA / "logs/panther.log", *sorted(DATA.glob("logs/progress-b*.log"))] for r in dated(f, 300)]
    return [x for _, x in sorted(dict.fromkeys(rows), key=lambda r: -r[0])[:n]]


def qat(ls: list[str]) -> tuple[int, int, float] | None:
    """The QAT finalist's Block-AP in this run's final step, if it has one: (blocks done, blocks, minutes a block,
    measured once a block is done; 4 until then)."""
    i = max((k for k, x in enumerate(ls) if " step final: start" in x), default=None)
    q = ls[i:] if i is not None else []
    n = next((int(m.group(1)) for x in q if (m := re.search(r"block-ap: .* (\d+) blocks$", x))), 0)
    first = next((clock(x) for x in q if "block-ap block 0: step 0" in x), None)
    ends = [clock(x) for x in q if re.search(r"block-ap block \d+: validation MSE", x)]
    return (len(ends), n, ((ends[-1] - first) % 1440) / len(ends) if ends and first is not None else 4.0) if n else None


def qats(ln: list[tuple[int, str, list[str]]]) -> list[dict]:
    """Each contest's QAT finalist, block by block, from its progress lines after this run's final step began: each
    block's checks [step, train, validation], its error before -> after, the file's, the best step, an early stop,
    whether the file's block was kept; the run's samples and epochs, Block-AP's ETA, the distillation, the score."""
    out, t = [], ist()
    for b, sm, ls in ln:
        if not (q := qat(ls)):
            continue
        ls = ls[max((k for k, x in enumerate(ls) if " step final: start" in x), default=0):]
        e = {"id": f"{sm.lower()}@{b}", "done": q[0], "total": q[1], "minutes": q[2], "blocks": {}}
        for x in ls:
            if m := re.search(r"block-ap: .*?(\d+) epochs x (\d+) samples", x):
                e["epochs"], e["samples"] = int(m.group(1)), int(m.group(2))
            elif m := re.search(r"qat distill update (\d+)/(\d+): .* (\S+)$", x):
                e["distill"] = [int(m.group(1)), int(m.group(2)), float(m.group(3))]
            elif m := re.search(r" final (\S+ qat): (.*)", x):
                e["final"] = f"{m.group(1)}: {m.group(2)}"
            elif m := re.search(r"block-ap block (\d+): (.*)", x):
                k, s = e["blocks"].setdefault(m.group(1), {"block": int(m.group(1)), "checks": []}), m.group(2)
                if r := re.match(r"step (\d+)(?:/\d+, train MSE (\S+))?, validation (\S+)$", s):
                    k["checks"].append([int(r.group(1)), r.group(2) and float(r.group(2)), float(r.group(3))])
                elif r := re.match(r"stopping early at step (\d+)", s):
                    k["stopped"] = int(r.group(1))
                elif s.startswith("trained "):
                    k["kept"] = True
                elif r := re.match(r"validation MSE (\S+) -> (\S+) \(best at step (\d+); .*the file's (\S+)\)", s):
                    k.update(zip(("before", "after", "best", "file"), map(float, r.groups())), best=int(r.group(3)))
        e["blocks"] = list(e["blocks"].values())
        end = t.tm_hour * 60 + t.tm_min + (q[1] - q[0]) * q[2]
        e["eta"] = f"{int(end // 60) % 24:02d}:{int(end % 60):02d}" if q[0] < q[1] else None
        out.append(e)
    return out


def steps(b: int, sm: str, ls: list[str], halt: tuple[str, int] | None) -> tuple[list[dict], str]:
    """A contest's steps (state, minutes spent and left) and its ETA. halt: ("failed" | "stopped", minute) when the
    run ended without finishing; the step it was in shows that state, timed up to the end."""
    size, t = float(re.sub(r"[^0-9.]", "", sm) or 7), ist()
    now = t.tm_hour * 60 + t.tm_min
    seen, began, took = {}, {}, {}
    for x in ls:
        if m := re.search(r"step (\w+): (start|done)(?: in (\d+) min)?", x):
            seen[m.group(1)] = (m.group(2), clock(x))
            if m.group(2) == "start":
                began[m.group(1)] = x[:5]
            elif m.group(3):
                took[m.group(1)] = int(m.group(3))
    out, clk = [], now
    for s in STEPS:
        st, at = seen.get(s, ("waiting", None))
        typ = MINUTES[s] * size
        row = {"name": s, "what": WHAT[s], "about": DESCRIBE[s], "began": began.get(s), "took": took.get(s)}
        if st == "done":
            out.append({**row, "state": "done"})
        elif st == "start" and halt:
            out.append({**row, "state": halt[0], "spent": (halt[1] - at) % 1440})
        elif st == "start":
            spent = (now - at) % 1440
            left = max(1.0, typ - spent)
            if s == "final" and (q := qat(ls)):  # Block-AP's blocks left, then its distillation and the held-out test
                left = (q[1] - q[0]) * q[2] + 25
                row["what"] = (f"QAT finalist: Block-AP block {min(q[0] + 1, q[1])} of {q[1]} ({q[2]:.1f} min a block),"
                               " then distilled; held-out verdict vs the preset")
            out.append({**row, "state": "running", "spent": spent, "left": left})
            clk = now + left
        else:
            clk += typ
            out.append({**row, "state": "waiting", "left": typ})
    done = all(x["state"] == "done" for x in out)
    return out, "done" if done else "" if halt else f"{int(clk // 60) % 24:02d}:{int(clk % 60):02d}"


def types(assignment: dict) -> list[list]:
    c: dict[str, int] = {}
    for t in assignment.values():
        c[t] = c.get(t, 0) + 1
    return sorted(([t, n] for t, n in c.items()), key=lambda x: -x[1])


def halted(raw: tuple[str, int] | None, ls: list[str]) -> tuple[str, int] | None:
    """How a run ended without finishing, if it did: its main-log line (FAILED, or stopped by the user), or a FAILED
    line in its progress lines after the last step began (older logs' lines of the removed QAT experiment are not
    the contest's)."""
    if raw and "FAILED" in raw[0]:
        return "failed", raw[1]
    if raw and raw[0].startswith("stopped by the user"):
        return "stopped", raw[1]
    last = max((i for i, x in enumerate(ls) if re.search(r"step \w+: start", x)), default=-1)
    fail = [x for x in ls[last + 1:] if re.search(r"(?<!QAT )FAILED in", x)]
    return ("failed", clock(fail[-1])) if fail else None


def verdict(w: dict, f: Path) -> dict | None:
    """A winner file's verdict and held-out numbers. unit "relative score - 100": diff and ci are the relative score
    minus 100 (negative = better than the preset); without unit (older files): accuracy points (positive = better)."""
    try:
        v = {k: w[k] for k in ("verdict", "diff", "ci", "preset", "winner", "ram_gib")}
        v.update({k: w[k] for k in ("unit", "relative", "relative_ci", "accuracy_diff", "accuracy_kept",
                                    "weight_check") if k in w})
        v["guard_warning"] = bool(w.get("guard", {}).get("warning"))
        return {"verdict": v, "model": Path(w["file"]).parts[-3] if "file" in w else "",
                "held_out": {k: w[k] for k in ("ours", "preset_test", "original")}, "types": types(w["assignment"])}
    except (KeyError, TypeError, AttributeError) as e:
        problem(f, f"missing or malformed field {e}")
        return None


def contests(log: list[str], ln) -> list[dict]:
    state: dict[str, tuple[str, int]] = {}  # each contest's last main-log line (older logs' QAT lines are not its; a
    for x in log:  # "refused" line is a start that never happened: the contest is as it was)
        if (m := re.match(r"\d\d:\d\d ([\w.]+-b\d+): (.*)", x)) and not m.group(2).startswith(("QAT", "refused:")):
            state[m.group(1)] = (m.group(2), clock(x))
    lanes = {f"{sm.lower()}-b{b}": (b, sm, ls) for b, sm, ls in ln}
    out = []
    for key in dict.fromkeys(list(state) + list(lanes)):
        sz, b = key.rsplit("-b", 1)
        b, sm = int(b), sz.upper()
        lane, raw = lanes.get(key), state.get(key)
        halt = halted(raw, lane[2] if lane else [])
        st, eta = steps(b, sm, lane[2], halt) if lane else ([], "")
        c = {"id": f"{sz}@{b}", "size": sm, "budget": b, "steps": st, "eta": eta, "ended": raw[0] if halt and raw else None,
             "events": [x for x in (lane[2] if lane else []) if re.search(
                 r"preset \S+: held-out|guard:|held-out test vs|GA done|new best|FAILED|Pi bundle uploaded", x)][-12:]}
        w = winner(b, sm)
        if w and (r := verdict(*w)):
            c.update(r)
        if halt:
            c["state"] = halt[0]
        elif raw and raw[0].startswith("started"):
            c["state"] = "running"  # a rerun: an older result, if any, shows until the new one is written
        elif "verdict" in c or (raw and raw[0] == "done"):
            c["state"] = "done"
        elif not searched(b, sm):
            c["state"] = "dropped"
        else:  # no main-log line (a rotated log): from its steps
            c["state"] = "running" if any(x["state"] == "running" for x in st) else "done" if eta == "done" else "waiting"
        out.append(c)
    order = {"running": 0, "failed": 1, "stopped": 2, "waiting": 3, "done": 4, "dropped": 5}
    return sorted(out, key=lambda c: (order[c["state"]], c["budget"], float(re.sub(r"[^0-9.]", "", c["size"]) or 0)))


PI: dict = {"files": {}, "error": None, "at": None, "repo": None}  # the Pi's results from Hugging Face (fetch_pi)


def fetch_pi() -> None:
    """Every 5 minutes: pi/<contest>.txt from <user>/panther-status into the temporary directory; a failed fetch is
    shown on the page and the last good results are kept."""
    os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "hf"))
    repo = os.environ.get("PANTHER_HF_REPO", "")
    if "/" not in repo:
        PI["error"] = "Pi results: PANTHER_HF_REPO not set" if not repo else f"Pi results: PANTHER_HF_REPO {repo!r} has no '/'"
        return
    PI.update(repo=repo.split("/")[0] + "/panther-status", error="Pi results: first fetch running")
    while True:
        try:
            from huggingface_hub import HfApi, hf_hub_download
            files = {}
            for f in HfApi().list_repo_files(PI["repo"]):
                if f.startswith("pi/") and f.endswith(".txt"):
                    files[f[3:-4]] = Path(hf_hub_download(PI["repo"], f, force_download=True, local_dir=Path(
                        tempfile.gettempdir()) / "panther-status")).read_text(errors="replace").strip()
            PI.update(files=files, error=None, at=time.strftime("%H:%M", ist()))
        except Exception as e:  # noqa: BLE001  (shown on the page; retried in 5 minutes)
            PI["error"] = f"Pi results from {PI['repo']}: fetch failed at {time.strftime('%H:%M', ist())}: " \
                          f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"
        time.sleep(300)


def pi() -> list[dict]:
    """The Pi's measurements of our winners: data/pi/<contest>.txt (older ones), with those fetched from Hugging Face
    over them."""
    text = {}
    for f in sorted(DATA.glob("pi/*@*.txt")):
        try:
            text[f.stem] = (f.read_text(errors="replace").strip(), f)
        except OSError as e:
            problem(f, e)
    text.update({k: (v, f"{PI['repo']}: pi/{k}.txt") for k, v in dict(PI["files"]).items()})
    out = []
    for c, (x, src) in sorted(text.items()):
        if not re.fullmatch(r"[\w.]+@\d+", c):
            continue  # older files named otherwise are not contests
        if f := re.search(r"FAILED: (.*)", x):  # a failed Pi check: shown as such, never dropped
            out.append({"contest": c, "budget": int(c.split("@")[1]), "failed": f.group(1).strip() or "FAILED",
                        "at": x[:8] if re.match(r"\d\d:\d\d:\d\d ", x) else None, "fits": None, "throttled": False,
                        **dict.fromkeys(("decode", "prefill", "ram", "predicted"))})
            continue
        m = {k: re.search(p, x) for k, p in (("decode", r"decode " + NUM), ("prefill", r"prefill " + NUM),
                                              ("ram", r"peak RAM " + NUM), ("predicted", r"predicted " + NUM))}
        if not all(m.values()):
            problem(src, "no " + ", ".join(k for k, v in m.items() if not v) + " in the Pi's line")
            continue
        fit = re.search(r"(FITS|OVER) [\d.]+ GiB", x)
        out.append({"contest": c, "budget": int(c.split("@")[1]), "throttled": "0xe" in x,
                    "fits": fit.group(0) if fit else None, **{k: float(v.group(1)) for k, v in m.items()}})
    return out


def live(rid: str) -> bool | None:
    """Whether a run is active: its lock file is held (a shared lock is refused); None if it cannot be told."""
    try:
        with open(DATA / "locks" / f"run-{rid}.lock") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
    except OSError:
        return None


def system() -> dict:
    out: dict = {"disk": None, "gpu": None, "claims": []}
    d = shutil.disk_usage(DATA)
    out["disk"] = {"free": d.free, "total": d.total}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        used, total, util = (float(v) for v in r.stdout.strip().split(","))
        out["gpu"] = {"used": used * 2**20, "total": total * 2**20, "util": util}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass  # no GPU in this container: the page shows the claims instead
    f = DATA / "locks/claims.json"
    try:
        book = json.loads(f.read_text()) if f.exists() else {}
        for k, v in book.items():  # a claim whose run has ended stays in the file until the next start: not shown
            if (on := live(k)) is not False:
                out["claims"].append({"id": k, "gpu": v["gpu"], "ram": v["ram"], "since": v.get("since", ""),
                                      "kind": v.get("kind", ""), "live": "yes" if on else "unknown"})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        problem(f, e)
    out["paused"] = [w for f, w in (("PAUSED", "by the user"), ("PAUSED.guard", "GitLab unhealthy"))
                     if (DATA / f).exists()]
    return out


def state() -> dict:
    with STATE:
        PROBLEMS.clear()
        log, ln = logs()
        t = ist()
        out = {"now": time.strftime("%d %b %H:%M:%S", t), "contests": contests(log, ln),
               "pi": pi(), "pi_status": {k: PI[k] for k in ("repo", "at", "error")}, "system": system(),
               "events": events(400), "qats": qats(ln)}
        return {**out, "problems": list(dict.fromkeys(PROBLEMS))}


class Handler(SimpleHTTPRequestHandler):
    cache: tuple[float, bytes] = (0.0, b"")

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")  # the browser revalidates, so a new version shows at once
        super().end_headers()

    def send(self, body: bytes, kind: str = "application/json", code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream(self):
        """Server-sent events: every new line of every log, about a second after it is written (each log is read
        from where it ended when the page connected; a log that shrinks was rotated and is read from its start)."""
        if not STREAMS.acquire(blocking=False):
            return self.send(b'{"error": "too many live streams"}', code=503)
        try:
            self.tail()
        finally:
            STREAMS.release()

    def tail(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        pos = {f: f.stat().st_size for f in log_files() if f.exists()}
        rest: dict[Path, bytes] = {}
        beat = time.time()
        try:
            self.wfile.write(b"retry: 3000\n\n")
            self.wfile.flush()
            while True:
                out = []
                for f in log_files():
                    if not f.exists():
                        continue
                    size, p = f.stat().st_size, pos.setdefault(f, 0)  # a log that appears later is read whole
                    if size < p:
                        p, rest[f] = 0, b""
                    if size > p:
                        with f.open("rb") as h:
                            h.seek(p)
                            chunk = rest.get(f, b"") + h.read(size - p)
                        pos[f] = size
                        *lines, rest[f] = chunk.split(b"\n")
                        out += [{"src": source(f), "line": mask(x.decode(errors="replace"))} for x in lines if x.strip()]
                for o in out:
                    self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
                if not out and time.time() - beat > 15:
                    self.wfile.write(b": ping\n\n")  # keeps tunnels and proxies from closing an idle stream
                    beat = time.time()
                elif out:
                    beat = time.time()
                self.wfile.flush()
                if select.select([self.connection], [], [], 1)[0] and not self.connection.recv(1, socket.MSG_PEEK):
                    return  # the page went away: free its slot now, not at the next failed write
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/api/logs/stream":
            return self.stream()
        if path == "/api/logs":
            m = re.search(r"[?&]n=(\d+)", self.path)
            return self.send(mask(json.dumps(backlog(min(int(m.group(1)) if m else 3000, 20000)))).encode())
        if path != "/api/state":
            return super().do_GET()
        if time.time() - Handler.cache[0] > 5:  # the page polls; the files are read at most every 5 s
            try:
                Handler.cache = (time.time(), mask(json.dumps(state(), default=float)).encode())
            except Exception as e:  # noqa: BLE001  (the page shows the error and keeps polling)
                Handler.cache = (time.time(), json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
        self.send(Handler.cache[1])

    def log_message(self, format, *args):  # noqa: A002  (quiet)
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("PANTHER_DATA", "/work/data"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=43800)
    a = ap.parse_args()
    global DATA
    DATA = Path(a.data)
    threading.Thread(target=fetch_pi, daemon=True).start()
    print(f"Panther dashboard: http://{a.host}:{a.port}  (data: {DATA})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
