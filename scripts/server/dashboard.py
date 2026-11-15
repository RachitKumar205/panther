"""Panther's dashboard: a self-hosted, read-only web page of the whole pipeline, from data/ only (no outside service).

  python scripts/server/dashboard.py [--data DIR] [--port 43800]      then open http://localhost:43800

GET /            the page (scripts/server/dashboard/)
GET /api/state   everything the page shows, as JSON: contests and their steps, verdicts and held-out tables, the
                 Pi's results, the EfficientQAT experiment (strategy, every block's checks, Stage 2 updates,
                 results), GPU memory and bookings, disk, events and notes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import status as S

STATIC = Path(__file__).with_name("dashboard")
NUM = r"([-+]?[\d.]+)"


def ist() -> time.struct_time:
    return time.gmtime(time.time() + 19800)


def logs() -> tuple[list[str], list[tuple[int, str, list[str]]]]:
    f = S.DATA / "logs/panther.log"
    return (f.read_text().splitlines() if f.exists() else []), S.lanes()


def dated(f: Path, n: int) -> list[tuple[int, str]]:
    """The file's last n timed lines with an absolute minute: lines carry only HH:MM (IST), so walking back from
    the file's last write, a clock that goes up means the day before."""
    if not f.exists():
        return []
    lines = [x for x in f.read_text().splitlines() if re.match(r"\d\d:\d\d ", x)][-n:]
    t = time.gmtime(f.stat().st_mtime + 19800)
    day, prev, out = int((f.stat().st_mtime + 19800) // 86400), t.tm_hour * 60 + t.tm_min, []
    for x in reversed(lines):
        if S.clock(x) > prev:
            day -= 1
        prev = S.clock(x)
        out.append((day * 1440 + prev, x))
    return out


def events(n: int) -> list[str]:
    """The newest n lines of the main log and the progress logs together, newest first."""
    rows = [r for f in [S.DATA / "logs/panther.log", *sorted(S.DATA.glob("logs/progress-b*.log"))] for r in dated(f, 300)]
    return [x for _, x in sorted(dict.fromkeys(rows), key=lambda r: -r[0])[:n]]


def steps(b: int, sm: str, ls: list[str]) -> tuple[list[dict], str]:
    """A contest's steps (state, minutes spent and left), and its ETA (as status.lane works it out)."""
    size, t = float(re.sub(r"[^0-9.]", "", sm) or 7), ist()
    now = t.tm_hour * 60 + t.tm_min
    seen = {}
    for x in ls:
        if m := re.search(r"step (\w+): (start|done)", x):
            seen[m.group(1)] = (m.group(2), S.clock(x))
    failed, out, clk = any("FAILED" in x for x in ls[-3:]), [], now
    for s in S.STEPS:
        st, at = seen.get(s, ("waiting", None))
        typ = S.MINUTES[s] * size
        row = {"name": s, "what": S.WHAT[s], "about": S.DESCRIBE[s]}
        if st == "done":
            out.append({**row, "state": "done"})
            continue
        if st == "start":
            spent = (now - at) % 1440
            left = max(1.0, typ - spent)
            out.append({**row, "state": "failed" if failed else "running", "spent": spent, "left": left})
            clk = now + left
        else:
            clk += typ
            out.append({**row, "state": "waiting", "left": typ})
    done = all(x["state"] == "done" for x in out)
    return out, "done" if done else f"{int(clk // 60) % 24:02d}:{int(clk % 60):02d}"


def types(assignment: dict) -> list[list]:
    c: dict[str, int] = {}
    for t in assignment.values():
        c[t] = c.get(t, 0) + 1
    return sorted(([t, n] for t, n in c.items()), key=lambda x: -x[1])


def contests(log: list[str], ln) -> list[dict]:
    state = {}
    for x in log:
        if (m := re.match(r"\d\d:\d\d (\S+): (.*)", x)) and m.group(1) != "note" and "QAT" not in m.group(2):
            state[m.group(1)] = m.group(2)
    lanes = {f"{sm.lower()}-b{b}": (b, sm, ls) for b, sm, ls in ln}
    out = []
    for key in dict.fromkeys(list(state) + list(lanes)):
        sz, b = key.split("-b")
        b, sm = int(b), sz.upper()
        w, kept = S.winner(b, sm), S.searched(b, sm)
        lane = lanes.get(key)
        st, eta = steps(b, sm, lane[2]) if lane else ([], "")
        c = {"id": f"{sz}@{b}", "size": sm, "budget": b, "steps": st, "eta": eta,
             "events": [x for x in (lane[2] if lane else []) if re.search(
                 r"preset \S+: held-out|guard:|held-out test vs|GA done|new best", x)][-12:]}
        if w:
            c["state"] = "done"
            c["verdict"] = {k: w[k] for k in ("verdict", "diff", "ci", "preset", "winner", "ram_gib")}
            c["model"] = Path(w["file"]).parts[-3] if "file" in w else ""
            c["held_out"] = {k: w[k] for k in ("ours", "preset_test", "original")}
            c["types"] = types(w["assignment"])
        elif not kept:
            c["state"] = "dropped"
        else:
            raw = state.get(key, "queued")
            c["state"] = "failed" if "FAILED" in raw else "running" if raw.startswith("started") else "queued"
        out.append(c)
    order = {"running": 0, "failed": 1, "queued": 2, "done": 3, "dropped": 4}
    return sorted(out, key=lambda c: (order[c["state"]], c["budget"], float(re.sub(r"[^0-9.]", "", c["size"]) or 0)))


def qat(log: list[str], ln) -> dict | None:
    """The EfficientQAT experiment, from the progress lines and its results file."""
    lines = [x for _, _, ls in ln for x in ls if "block-ap" in x or "QAT" in x]
    res = {f.name: json.loads(f.read_text()) for f in sorted(S.DATA.glob("rewrite/budget-*/qat-*.json"))
           if "trial" not in f.name}
    if not lines and not res:
        return None
    runs = [i for i, x in enumerate(lines) if "block-ap strategy:" in x]
    while len(runs) > 1 and runs[-1] + 1 < len(lines) and "resuming" in lines[runs[-1] + 1]:
        runs.pop()  # a resumed Stage 1 continues the run before it
    cur = lines[runs[-1]:] if runs else []
    blocks: dict[int, dict] = {}
    for x in cur:
        if m := re.search(r"block-ap block (\d+): step (\d+)/(\d+), train MSE " + NUM + r"(?:, validation " + NUM + ")?", x):
            b = blocks.setdefault(int(m.group(1)), {"block": int(m.group(1)), "checks": [], "of": int(m.group(3))})
            if int(m.group(2)) == 512:  # retrained after a resume: its checks start again
                b["checks"], b["stopped"] = [], None
            b["checks"].append({"step": int(m.group(2)), "train": float(m.group(4)),
                                "val": float(m.group(5)) if m.group(5) else None, "at": x[:5]})
        elif m := re.search(r"block-ap block (\d+): validation MSE " + NUM + " -> " + NUM + r" \(best at step (\d+)", x):
            b = blocks.setdefault(int(m.group(1)), {"block": int(m.group(1)), "checks": []})
            b.update(before=float(m.group(2)), after=float(m.group(3)), best=int(m.group(4)), done=x[:5])
        elif m := re.search(r"block-ap block (\d+): stopping early at step (\d+)", x):
            blocks.setdefault(int(m.group(1)), {"block": int(m.group(1)), "checks": []})["stopped"] = int(m.group(2))

    def updates(tag: str) -> list[dict]:
        return [{"update": int(m.group(1)), "of": int(m.group(2)), "kl": float(m.group(3)), "at": x[:5]}
                for x in lines if tag in x and (m := re.search(r"update (\d+)/(\d+): KL on the teacher's top \d+ " + NUM, x))]
    done = sorted(b for b, v in blocks.items() if "after" in v)
    times = [S.clock(blocks[b]["done"]) for b in done]
    per = (times[-1] - times[0]) % 1440 / max(1, len(times) - 1) if len(times) > 1 else None
    total = 48
    left = total - len(done)
    eta = None
    if per and left:
        t = ist()
        e = t.tm_hour * 60 + t.tm_min + per * left
        eta = f"{int(e // 60) % 24:02d}:{int(e % 60):02d}"
    strategy = lines[[i for i, x in enumerate(lines) if "block-ap strategy:" in x][-1]].split("strategy: ", 1)[1] if cur else None
    notes = [x[6:].replace("note: ", "") for x in log if "note:" in x and ("QAT" in x or "Stage" in x or "block-0" in x)]
    return {"strategy": strategy, "blocks": [blocks[b] for b in sorted(blocks)], "total": total,
            "minutes_per_block": per, "stage1_eta": eta, "now": notes[-1] if notes else None,
            "tests": [x[6:].split("block-0 test: ", 1)[1] for x in log if "note: block-0 test" in x],
            "control": updates("stage 2 only"), "stage2": updates("stage 2 after stage 1"),
            "results": {name: r for _, rs in res.items() for name, r in rs.items()}}


def pi() -> list[dict]:
    """The Pi's measurements of our winners (data/pi/<contest>.txt, saved by the status job)."""
    out = []
    for f in sorted(S.DATA.glob("pi/*.txt")):
        x = f.read_text().strip()
        m = {k: re.search(p, x) for k, p in (("decode", r"decode " + NUM), ("prefill", r"prefill " + NUM),
                                              ("ram", r"peak RAM " + NUM), ("predicted", r"predicted " + NUM))}
        if all(m.values()):
            out.append({"contest": f.stem, "budget": int(f.stem.split("@")[1]), "throttled": "0xe" in x,
                        **{k: float(v.group(1)) for k, v in m.items()}})
    return out


def system() -> dict:
    out: dict = {"disk": None, "gpu": None, "bookings": []}
    d = shutil.disk_usage(S.DATA)
    out["disk"] = {"free": d.free, "total": d.total}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        used, total, util = (float(v) for v in r.stdout.strip().split(","))
        out["gpu"] = {"used": used * 2**20, "total": total * 2**20, "util": util}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    f = S.DATA / "locks/gpu.json"
    if f.exists():
        try:
            out["bookings"] = [{"bytes": v[0], "lapses_in": v[1] - time.time()}
                               for v in json.loads(f.read_text()).values() if v[1] > time.time()]
        except (OSError, ValueError):
            pass
    out["paused"] = [w for f, w in (("PAUSED", "by the user"), ("PAUSED.guard", "GitLab unhealthy"))
                     if (S.DATA / f).exists()]
    return out


def state() -> dict:
    log, ln = logs()
    t = ist()
    return {"now": time.strftime("%d %b %H:%M:%S", t), "contests": contests(log, ln), "qat": qat(log, ln),
            "pi": pi(), "system": system(),
            "events": events(400)}


class Handler(SimpleHTTPRequestHandler):
    cache: tuple[float, bytes] = (0.0, b"")

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] != "/api/state":
            return super().do_GET()
        if time.time() - Handler.cache[0] > 5:  # the page polls; the files are read at most every 5 s
            try:
                Handler.cache = (time.time(), json.dumps(state(), default=float).encode())
            except Exception as e:  # noqa: BLE001  (the page shows the error and keeps polling)
                Handler.cache = (time.time(), json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(Handler.cache[1])

    def log_message(self, format, *args):  # noqa: A002  (quiet)
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("PANTHER_DATA", "/work/data"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=43800)
    a = ap.parse_args()
    S.DATA = Path(a.data)
    print(f"Panther dashboard: http://{a.host}:{a.port}  (data: {S.DATA})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
