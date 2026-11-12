"""Live progress of every running lane as a markdown page (data/status.md), rewritten every 30 s and published to the
Hugging Face repo <user>/panther-status every 60 s. Runs on the server as a CPU job:

  deploy/server/panther.sh job rs-status /opt/venv/bin/python scripts/server/status.py

Per lane, one row per step: what it does, a progress bar, time spent, time left and the clock time it should end.
Progress is measured where the step reports it (the GA's individuals, the final scoring's candidates); elsewhere it
is elapsed time against the step's typical length for that model size (marked "est."). No host names or addresses."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path

DATA = Path(os.environ.get("PANTHER_DATA", "/work/data"))
LOGS, CHAINS, RESULTS = DATA / "logs", DATA / "rewrite" / "chains", DATA / "results"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
GA_MIN, GA_MAX, GA_TOP = 30, 70, 3
SIZE = {"1.5b": 1.5, "3b": 3, "7b": 7, "14b": 14, "20b": 20, "24b": 24, "32b": 32}
WHAT = {"fetch": "Download the original model", "store": "Quantize the original to every type (type copies)",
        "err": "Offline error of every matrix at every type", "refs": "Original model's outputs on the test texts",
        "preset": "Build and score llama.cpp's presets (the competitor)",
        "ga": "Genetic search (fitness: accuracy on 900 questions, KL tie-break)",
        "gascore": "Top 3 on all 4,500 questions, guard vs the real preset, full test of the winner",
        "vspreset": "Final comparison: our winner vs the preset", "bundle": "Pack the winner for the Pi",
        "upload": "Upload the winner for the Pi"}
_cache: dict[Path, tuple[str, float] | None] = {}


def typical_min(k: str, size: float) -> float:
    """Typical length of a step for a model of `size` billion weights (from this project's runs; rough)."""
    return {"fetch": 10, "store": 15 + 3 * size, "err": 5, "refs": 2 + size / 2, "preset": 5 + size,
            "ga": (GA_MIN + GA_MAX) / 2 * size / 6,  # ~6/size individuals a minute
            "gascore": 10 + 4 * size, "vspreset": 5, "bundle": 2 + size / 3, "upload": 2 + size / 3}.get(k, 5)


def started_at() -> dict[str, float]:
    """When each job was started, by any lane (a lane that adopts a running job never saw it start)."""
    t: dict[str, float] = {}
    for f in LOGS.glob("chain-*.log"):
        last = 0.0
        for ln in f.read_text().splitlines():
            if re.match(r"\d{4}-\d\d-\d\dT", ln):
                last = utc(ln)
            elif mm := re.search(r"started \w+ job panther-(\S+);", ln):
                t[mm.group(1)] = max(t.get(mm.group(1), 0.0), last)
    return t


def kind(step: str) -> str:
    name = step.split("-")[1] if step.count("-") >= 2 else step
    return "gascore" if name.startswith("gascore") else "vspreset" if name.startswith("vspreset") else re.sub(r"\d.*$", "", name)


def utc(ts: str) -> float:
    return dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()


def ga_times(model: str, budget: int) -> list[float]:
    """Start times of every GA individual fully scored (its last subset task, WinoGrande, recorded)."""
    out = []
    for f in (RESULTS / model).glob("*-task-*.json"):
        if f not in _cache:
            try:
                r = json.loads(f.read_text())
                how = r.get("how", {})
                ok = how.get("task") == "winogrande" and how.get("n_tasks") and r.get("label", "").startswith("rw:ga")
                _cache[f] = (r["label"], utc(dt.datetime.strptime(r["eval_id"][:15], "%Y%m%dT%H%M%S")
                                            .strftime("%Y-%m-%dT%H:%M:%S"))) if ok else None
            except Exception:  # noqa: BLE001  (a file being written)
                continue
        c = _cache[f]
        if c and c[0].startswith(f"rw:ga{budget}:"):
            out.append(c[1])
    return sorted(out)


def lane(chain: str) -> dict | None:
    steps_f, log_f = CHAINS / f"{chain}.steps", LOGS / f"chain-{chain}.log"
    m = re.match(r"g(\d+)-(.+)", chain)
    if m is None or not steps_f.exists() or not log_f.exists():
        return None
    steps = [ln.split()[1] for ln in steps_f.read_text().splitlines() if ln.strip()]
    model = next(ln.split()[-1] for ln in steps_f.read_text().splitlines() if " ga " in ln)
    state: dict[str, str] = {}
    t0: dict[str, float] = {}
    t1: dict[str, float] = {}
    last = 0.0
    lines = log_f.read_text().splitlines()
    for ln in lines:
        if re.match(r"\d{4}-\d\d-\d\dT", ln):
            last = utc(ln)
        if mm := re.search(r"skip (\S+)", ln):
            state[mm.group(1)] = "done"
        elif mm := re.search(r"started \w+ job panther-(\S+);", ln):
            state[mm.group(1)], t0[mm.group(1)] = "running", last
        elif mm := re.search(r"adopt (\S+)", ln):
            state[mm.group(1)] = "running"
        elif mm := re.search(r"(\S+) exited (\d+)", ln):
            state[mm.group(1)] = "done" if mm.group(2) == "0" else "failed"
            t1[mm.group(1)] = last
    return {"budget": int(m.group(1)), "short": m.group(2), "model": model, "steps": steps, "state": state, "t0": t0,
            "t1": t1, "stopped": bool(lines) and "stopped at" in lines[-1]}


def bar(frac: float) -> str:
    n = max(0, min(10, round(10 * frac)))
    return "█" * n + "░" * (10 - n) + f" {100 * frac:.0f}%"


def hm(minutes: float) -> str:
    return f"{int(minutes // 60)} h {int(minutes % 60):02d} min" if minutes >= 60 else f"{minutes:.0f} min"


def render() -> str:
    now = time.time()
    t_start = started_at()
    out = ["# Panther: live progress", "",
           f"Updated **{dt.datetime.now(IST):%d %b %H:%M:%S} IST**, every 30 s. Times are IST. "
           "\"est.\" = estimated from the step's typical length; otherwise measured.", ""]
    lanes = [x for x in (lane(p.stem[len("chain-"):]) for p in sorted(LOGS.glob("chain-g*.log"))) if x]
    for L in lanes:
        size, B, short = SIZE.get(L["short"], 7), L["budget"], L["short"].upper()
        prog = LOGS / f"progress-b{B}.log"
        pl = [ln for ln in prog.read_text().splitlines() if f"{short} " in ln] if prog.exists() else []
        last = lambda key: next((ln for ln in reversed(pl) if key in ln), None)  # noqa: E731
        rows, clock = [], now
        for s in L["steps"]:
            k, st = kind(s), L["state"].get(s, "waiting")
            ts = ga_times(L["model"], B) if k == "ga" else []
            if k == "ga" and st == "waiting" and ts and ts[-1] > now - 900:
                st = "running"  # a GA started before its lane got there (adopted later)
            typ = typical_min(k, size)
            frac, spent, left, how = 0.0, 0.0, typ, "est."
            if st == "done":
                frac, left = 1.0, 0.0
                spent = (L["t1"].get(s, 0) - L["t0"][s]) / 60 if s in L["t0"] and s in L["t1"] else 0
            elif st == "running":
                start = L["t0"].get(s) or t_start.get(s) or (ts[0] if ts else now)
                spent = (now - start) / 60
                if k == "ga" and ts:
                    n = len(ts)
                    recent = [t for t in ts if t > now - 1800]
                    rate = (len(recent) / 30) if len(recent) >= 3 else n / max(1.0, (now - ts[0]) / 60)
                    expect = max(n, (GA_MIN + GA_MAX) / 2)
                    frac, how = n / expect, f"{n} individuals of {GA_MIN}-{GA_MAX} (stops on a plateau), {rate:.2f}/min"
                    left = max(0, expect - n) / rate if rate else typ
                elif k == "gascore":
                    done = sum(1 for ln in pl if re.search(r"GA \S+:ga\d+: selection accuracy", ln))
                    stage = (3 if last("GA winner") else 2 if last("guard:") else 1 if done >= GA_TOP else 0)
                    frac = [0.1 + 0.4 * done / GA_TOP, 0.55, 0.65, 1.0][stage]
                    how = (f"{done}/{GA_TOP} candidates scored" if stage == 0 else "guard vs the preset" if stage == 1
                           else "full test of the winner" if stage == 2 else "done")
                    left = max(1.0, typ * (1 - frac))
                else:
                    frac, left = min(0.95, spent / typ), max(1.0, typ - spent)
            elif st == "failed":
                left = 0.0
            if st == "running":  # steps can overlap (a GA adopted while its lane finishes an earlier step)
                end_t = now + left * 60
                clock = max(clock, end_t)
            elif st != "done":
                clock += left * 60
                end_t = clock
            end = "done" if st == "done" else f"{dt.datetime.fromtimestamp(end_t, IST):%H:%M}"
            icon = {"done": "✅", "running": "▶️", "failed": "❌", "waiting": "⏳"}[st]
            rows.append(f"| {icon} | **{WHAT.get(k, k)}** | {bar(frac)} | {hm(spent) if spent >= 1 else '<1 min' if s in L['t1'] else '-'} | "
                        f"{'-' if st == 'done' else hm(left) + (' (est.)' if how == 'est.' else '')} | {end} | "
                        f"{'' if how == 'est.' else how} |")
        finish = "stopped" if L["stopped"] else f"ETA {dt.datetime.fromtimestamp(clock, IST):%H:%M} IST"
        out += [f"## {L['model']} at {B} GiB: {finish}", ""]
        if b := last("on the GA subset"):
            if mm := re.search(r"accuracy ([\d.]+), bench KL ([\d.]+), fitness ([\d.]+)", b):
                out.append(f"- **To beat**, the llama.cpp preset on the 900 search questions: accuracy **{mm.group(1)}**, "
                           f"KL {mm.group(2)}")
        if g := last("new best fitness"):
            if mm := re.search(r"individual (\d+): new best fitness ([\d.]+) \(subset accuracy ([\d.]+), bench KL ([\d.]+)", g):
                out.append(f"- **Our best so far** (individual {mm.group(1)}): accuracy **{mm.group(3)}**, KL {mm.group(4)}")
        for key in ("guard:", "GA winner"):
            if x := last(key):
                out.append(f"- {x[6:]}")
        out += ["", "| | step | progress | spent | left | ends | detail |", "|---|---|---|---|---|---|---|", *rows, ""]
    if not lanes:
        out.append("No lanes running.")
    return "\n".join(out) + "\n"


def main() -> None:
    """Rewrites data/status.md every 30 s and puts it in the Hugging Face repo <user>/panther-status every 60 s
    (Hugging Face limits commits per hour)."""
    from huggingface_hub import HfApi
    path = DATA / "status.md"
    repo = os.environ["PANTHER_HF_REPO"].split("/")[0] + "/panther-status"
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    i = 0
    while True:
        try:
            tmp = path.with_suffix(".part")
            tmp.write_text(render())
            tmp.replace(path)
            if i % 2 == 0:
                api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md", repo_id=repo, commit_message="status")
        except Exception as e:  # noqa: BLE001  (a failed refresh is retried next time)
            print(f"{time.strftime('%H:%M')} refresh failed: {e}", flush=True)
        i += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
