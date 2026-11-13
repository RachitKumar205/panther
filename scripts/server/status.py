"""The live status page: data/status.md every 30 s, published as README.md of <user>/panther-status on Hugging Face
every 60 s. Read from the progress logs and result files only. Runs as a CPU job:

  started by ./panther start
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

DATA = Path(os.environ.get("PANTHER_DATA", "/work/data"))
STEPS = ["store", "error", "refs", "preset", "ga", "final", "bundle"]
WHAT = {"store": "16-bit original, importance matrix, the model in every type", "error": "offline error per matrix",
        "refs": "the original's answers and reference outputs", "preset": "llama.cpp's preset (competitor), tested",
        "ga": "genetic search (score: flips + KL on the original's answers)",
        "final": "top 3 distilled and scored on all selection questions, guard, held-out test vs preset",
        "bundle": "file for the Pi"}
MINUTES = {"store": 5, "error": 0.2, "refs": 3, "preset": 4, "ga": 10, "final": 14, "bundle": 0.3}  # x size in B
PI: dict[str, str] = {}


def clock(line: str) -> int:
    h, m = line[:5].split(":")
    return int(h) * 60 + int(m)


def lanes() -> list[tuple[int, str, list[str]]]:
    out = []
    for f in sorted(DATA.glob("logs/progress-b*.log")):
        b, lines = int(f.stem[10:]), f.read_text().splitlines()
        for sm in dict.fromkeys(m.split()[1] for m in lines if re.match(r"\d\d:\d\d \S+ step \w+: start", m)):
            out.append((b, sm, [x for x in lines if x[6:].startswith(sm + " ")]))
    return out


def lane(b: int, sm: str, ls: list[str]) -> list[str]:
    size = float(re.sub(r"[^0-9.]", "", sm) or 7)
    now = time.gmtime(time.time() + 19800)
    now_m = now.tm_hour * 60 + now.tm_min
    start = max((i for i, x in enumerate(ls) if " step store: start" in x), default=0)
    ls = ls[start:]
    state, rows, clk = {}, [], now_m
    for x in ls:
        if m := re.search(r"step (\w+): (start|done)", x):
            state[m.group(1)] = (m.group(2), clock(x))
    failed = any("FAILED" in x for x in ls[-3:])
    for s in STEPS:
        st, at = state.get(s, ("waiting", None))
        typ = MINUTES[s] * size
        if st == "done":
            rows.append(f"| done | {WHAT[s]} | | |")
            continue
        if st == "start":
            spent = (now_m - at) % 1440
            left = max(1.0, typ - spent)
            if s == "ga":
                n = max([int(m.group(1)) for x in ls if (m := re.search(r"GA: (?:genome )?(\d+)", x))] + [0])
                rate = n / max(1, spent)
                left = (40 - n) / rate if n >= 4 and rate else typ
                best = next((x[6:] for x in reversed(ls) if "new best" in x), "")
                rows.append(f"| {'FAILED' if failed else 'running'} | {WHAT[s]}: {n} genomes (30-70, stops on a "
                            f"plateau). {best} | {spent:.0f} min | {left:.0f} min |")
            else:
                rows.append(f"| {'FAILED' if failed else 'running'} | {WHAT[s]} | {spent:.0f} min | ~{left:.0f} min |")
            clk = now_m + left
        else:
            clk += typ
            rows.append(f"| waiting | {WHAT[s]} | | ~{typ:.0f} min |")
    eta = "done" if all(state.get(s, ("",))[0] == "done" for s in STEPS) else f"ETA {int(clk // 60) % 24:02d}:{int(clk % 60):02d} IST"
    lines = [f"## {sm} at {b} GiB: {eta}", ""]
    lines += [f"- {x[6:]}" for x in ls if re.search(r"preset \S+: held-out|guard:|held-out test vs", x)]
    return lines + ["", "| state | step | spent | left |", "|---|---|---|---|", *rows, ""]


def render() -> str:
    head = [f"# Panther: live progress", "", f"Updated {time.strftime('%d %b %H:%M', time.gmtime(time.time() + 19800))}"
            " IST. Each contest: our quantization vs llama.cpp's best preset that fits the same RAM budget, on "
            "held-out questions. WIN / LOSE: the 95% interval of the accuracy difference is all above / below 0.", "",
            "## Verdicts", "", "| contest | verdict | ours - preset, points [95% interval] | flips ours / preset |",
            "|---|---|---|---|"]
    log = (DATA / "logs/panther.log").read_text().splitlines() if (DATA / "logs/panther.log").exists() else []
    state = {}
    for x in log:
        if m := re.match(r"\d\d:\d\d (\S+): (.*)", x):
            state[m.group(1)] = m.group(2)
    paused = [f"**PAUSED** ({w})" for f, w in (("PAUSED", "by the user"), ("PAUSED.guard", "GitLab unhealthy"))
              if (DATA / f).exists()]
    head[4:4] = paused + ([""] if paused else []) + ["## Contests", ""] + [f"- {k}: {v}" for k, v in state.items()] + \
        ["", "Recent events:", "", *[f"    {x}" for x in log[-10:]], ""]
    body = []
    for b, sm, ls in lanes():
        v = next((x for x in reversed(ls) if "our winner" in x), None)
        if v and (m := re.search(r"our winner ([+-][\d.]+) pts \[([+-][\d.]+), ([+-][\d.]+)\]", v)):
            d, lo, hi = map(float, m.groups())
            word = "WIN" if lo > 0 else "LOSE" if hi < 0 else "TIE"
            fl = re.search(r"flips ([\d.]+) vs ([\d.]+)", v)
            head.append(f"| {sm} @ {b} GiB | **{word}** | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                        f"{fl.group(1) + ' / ' + fl.group(2) if fl else ''} |")
        else:
            head.append(f"| {sm} @ {b} GiB | running | | |")
        body += lane(b, sm, ls)
    pi = ["## On the Pi (our winners)", ""] + [f"- {k}: {v}" for k, v in sorted(PI.items())] + [""]
    return "\n".join(head + [""] + pi + body) + "\n"


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api, repo = HfApi(), os.environ["PANTHER_HF_REPO"].split("/")[0] + "/panther-status"
    api.create_repo(repo, exist_ok=True)
    for i in range(10**9):
        try:
            (DATA / "status.md").write_text(render())
            if i % 2 == 0:
                for f in api.list_repo_files(repo):
                    if f.startswith("pi/"):
                        PI[f[3:-4]] = Path(hf_hub_download(repo, f, force_download=True)).read_text().strip()
                api.upload_file(path_or_fileobj=str(DATA / "status.md"), path_in_repo="README.md", repo_id=repo,
                                commit_message="status")
        except Exception as e:  # noqa: BLE001  (retried next time)
            print(f"refresh failed: {e}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
