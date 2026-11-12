"""Live progress of every running lane as a markdown page (data/status.md), rewritten every 30 s and published to the
private Hugging Face repo <user>/panther-status every 60 s. Runs on the server as a CPU job:

  deploy/server/panther.sh job rs-status /opt/venv/bin/python scripts/server/status.py

Per lane: every step (done, running, waiting), the running step's progress and the GA's best against the preset,
and an ETA (the GA's from its measured pace; later steps from typical durations). No host names or addresses."""

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
GA_MIN, GA_MAX = 30, 70
SIZE = {"1.5b": 1.5, "3b": 3, "7b": 7, "14b": 14, "20b": 20, "24b": 24, "32b": 32}
_task_cache: dict[Path, tuple[str, str, float] | None] = {}


def step_minutes(step: str, size: float) -> float:
    """Typical duration of a step, from tonight's runs (rough)."""
    base = {"fetch": 10, "store": 3 * size, "err": 5, "refs": 2 + size / 2, "preset": 5 + size,
            "gascore": 10 + 4 * size, "vspreset": 5, "bundle": 2 + size / 3, "upload": 2 + size / 3}
    return base.get(step, 5)


def ga_individuals(model: str, budget: int) -> list[float]:
    """Start times (unix) of every GA individual fully scored (its last subset task, WinoGrande, recorded)."""
    out = []
    for f in (RESULTS / model).glob("*-task-*.json"):
        if f not in _task_cache:
            try:
                r = json.loads(f.read_text())
                how = r.get("how", {})
                ok = how.get("task") == "winogrande" and how.get("n_tasks") and r.get("label", "").startswith("rw:ga")
                _task_cache[f] = (r["label"], r["eval_id"][:15], 0.0) if ok else None
            except Exception:  # noqa: BLE001  (a file being written)
                continue
        c = _task_cache[f]
        if c and c[0].startswith(f"rw:ga{budget}:"):
            out.append(dt.datetime.strptime(c[1], "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp())
    return sorted(out)


def progress_lines(budget: int, short: str) -> list[str]:
    p = LOGS / f"progress-b{budget}.log"
    return [ln for ln in p.read_text().splitlines() if f"{short} " in ln] if p.exists() else []


def lane(chain: str) -> dict | None:
    steps_f, log_f = CHAINS / f"{chain}.steps", LOGS / f"chain-{chain}.log"
    if not steps_f.exists() or not log_f.exists():
        return None
    steps = [ln.split()[1] for ln in steps_f.read_text().splitlines() if ln.strip()]
    m = re.match(r"g(\d+)-(.+)", chain)
    if m is None:
        return None
    budget, short = int(m.group(1)), m.group(2)
    model = next(ln.split()[-1] for ln in steps_f.read_text().splitlines() if " ga " in ln)
    log = log_f.read_text()
    state, started = {}, {}
    for ln in log.splitlines():
        t = ln.split(" ")[0]
        if mm := re.search(r"skip (\S+)", ln):
            state[mm.group(1)] = "done"
        elif mm := re.search(r"(\S+) exited (\d+)", ln):
            state[mm.group(1)] = "done" if mm.group(2) == "0" else "failed"
        elif mm := re.search(r"adopt (\S+)", ln):
            state[mm.group(1)], started[mm.group(1)] = "running", t
    for mm in re.finditer(r"started \w+ job panther-(\S+);", log):
        state.setdefault(mm.group(1), "running")
        if state[mm.group(1)] == "running":
            started.setdefault(mm.group(1), "")
    return {"chain": chain, "budget": budget, "short": short, "model": model, "steps": steps, "state": state,
            "stopped": "stopped at" in log.splitlines()[-1] if log.strip() else False}


def kind(step: str) -> str:
    return re.sub(r"^r[gws]-|\d.*$|-.*$", "", step).replace("ga", "ga") if not step.startswith("rg-ga") else (
        "gascore" if "gascore" in step else "ga")


def render() -> str:
    now = time.time()
    lines = [f"# Panther: live progress", f"Updated **{dt.datetime.now(IST):%d %b %H:%M:%S} IST** (refreshes every 30 s)", ""]
    lanes = [x for x in (lane(p.stem[len("chain-"):]) for p in sorted(LOGS.glob("chain-g*.log"))) if x]
    for L in lanes:
        size, B, short = SIZE.get(L["short"], 7), L["budget"], L["short"].upper()
        pl = progress_lines(B, short)
        best = next((ln for ln in reversed(pl) if "GA: individual" in ln and "fitness" in ln), None)
        bar = next((ln for ln in reversed(pl) if "on the GA subset" in ln), None)
        done_ga = next((ln for ln in reversed(pl) if "GA done" in ln), None)
        guard = next((ln for ln in reversed(pl) if "guard:" in ln), None)
        winner = next((ln for ln in reversed(pl) if "GA winner" in ln), None)
        rows, eta = [], 0.0
        for s in L["steps"]:
            st, k = L["state"].get(s, "waiting"), kind(s)
            detail = ""
            if k == "ga" and st == "waiting" and any(t > now - 900 for t in ga_individuals(L["model"], B)):
                st = "running"  # a GA started before its lane (adopted when the lane gets there)
            if k == "ga" and st in ("running", "done"):
                ts = ga_individuals(L["model"], B)
                n = len(ts)
                recent = [t for t in ts if t > now - 1800]
                rate = len(recent) / 30 if len(recent) >= 2 else (n / max(1, (now - ts[0]) / 60) if ts else 0)
                detail = f"{n} individuals scored (stops on a plateau after {GA_MIN}, at most {GA_MAX})"
                if rate:
                    detail += f"; {rate:.2f}/min"
                if st == "running" and rate:
                    lo, hi = max(0, GA_MIN - n) / rate, max(0, GA_MAX - n) / rate
                    detail += f"; ~{lo:.0f}-{hi:.0f} min left"
                    eta += (lo + hi) / 2
            elif st == "running":
                eta += step_minutes(k, size) / 2
            elif st == "waiting":
                eta += step_minutes(k, size)
            mark = {"done": "✅", "running": "▶️", "failed": "❌", "waiting": "·"}[st]
            rows.append(f"| {mark} | {s.split('-')[1] if s.count('-') else s} | {st} | {detail} |")
        finish = dt.datetime.fromtimestamp(now + eta * 60, IST)
        status = "**stopped**" if L["stopped"] else (f"ETA **{finish:%H:%M} IST**" if eta else "**finished**")
        lines += [f"## {L['model']} @ {B} GiB: {status}", ""]
        if bar:
            mm = re.search(r"accuracy ([\d.]+), bench KL ([\d.]+), fitness ([\d.]+)", bar)
            if mm:
                lines.append(f"- Preset to beat (900 questions): accuracy **{mm.group(1)}**, KL {mm.group(2)}, "
                             f"fitness {mm.group(3)}")
        if best:
            mm = re.search(r"individual (\d+): new best fitness ([\d.]+) \(subset accuracy ([\d.]+), bench KL ([\d.]+)", best)
            if mm:
                lines.append(f"- GA best so far: individual {mm.group(1)}, accuracy **{mm.group(3)}**, KL {mm.group(4)}, "
                             f"fitness **{mm.group(2)}**")
        for x in (done_ga, guard, winner):
            if x:
                lines.append(f"- {x[6:]}")
        lines += ["", "| | step | state | progress |", "|---|---|---|---|", *rows, ""]
    if not lanes:
        lines.append("No lanes running.")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Rewrites data/status.md every 30 s and puts it in the private Hugging Face repo <user>/panther-status every
    60 s (Hugging Face limits commits per hour)."""
    from huggingface_hub import HfApi
    path = DATA / "status.md"
    repo = os.environ["PANTHER_HF_REPO"].split("/")[0] + "/panther-status"
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
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
