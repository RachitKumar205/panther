"""The live status page: data/status.md every 30 s, published as README.md of <user>/panther-status on Hugging Face
every 60 s. Read from the progress logs and result files only. Runs as a CPU job:

  started by ./panther start
"""

from __future__ import annotations

import json
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
DESCRIBE = {
    "store": "the original as a 16-bit file; its importance matrix (which inputs are large, on calibration text); the"
             " whole model in each of 14 types (IQ1_S to Q8_0), every matrix with error-compensated rounding",
    "error": "for every matrix and type, a cheap estimate of the damage (used to repair genomes and guide the solver)",
    "refs": "the original writes its own answers (GSM8K); its next-word probabilities on them and which questions it"
            " gets right are recorded, so drift and flips can be measured",
    "preset": "llama.cpp's presets near the budget are built and tested on the held-out questions: the competitor",
    "ga": "genetic search: genomes (one type per matrix, inside the RAM budget) scored by flips + 10 x drift on the"
          " original's answers, 4 at a time; mutation, crossover and solver proposals; stops on a plateau (30-70)",
    "final": "the top 3 are distilled (block scales and norms trained to match the original), scored on all 4,500"
             " selection questions; the best must not be less accurate than the preset (guard); then winner vs"
             " preset vs original on the held-out test: the verdict",
    "bundle": "the winning file packed for the Pi, which downloads it and posts its real RAM and speed here"}
PI: dict[str, str] = {}


def clock(line: str) -> int:
    h, m = line[:5].split(":")
    return int(h) * 60 + int(m)


def recent(lines: list[str]) -> list[str]:
    """The newest 20 events, oldest first; lines carry only HH:MM (IST), so a line from just before midnight counts
    as older than one from this afternoon."""
    t = time.gmtime(time.time() + 19800)
    now = t.tm_hour * 60 + t.tm_min
    lines = [x for x in dict.fromkeys(lines) if re.match(r"\d\d:\d\d ", x)]
    return sorted(lines, key=lambda x: (clock(x) - now - 1) % 1440)[-20:]


def lanes() -> list[tuple[int, str, list[str]]]:
    out = []
    for f in sorted(DATA.glob("logs/progress-b*.log")):
        b, lines = int(f.stem[10:]), f.read_text().splitlines()
        for sm in dict.fromkeys(m.split()[1] for m in lines if re.match(r"\d\d:\d\d \S+ step \w+: start", m)):
            ls = [x for x in lines if x[6:].startswith(sm + " ")]
            out.append((b, sm, ls[max(i for i, x in enumerate(ls) if " step store: start" in x):]))  # this run only
    return out


def lane(b: int, sm: str, ls: list[str]) -> list[str]:
    size = float(re.sub(r"[^0-9.]", "", sm) or 7)
    now = time.gmtime(time.time() + 19800)
    now_m = now.tm_hour * 60 + now.tm_min
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


def results(b: int, sm: str) -> list[str]:
    """The finished contest's full table: original, ours, the preset (and the first method's winner)."""
    f = next(iter((DATA / "rewrite" / f"budget-{b}").glob(f"winner-*-{sm}-*-v2.json")), None)
    if f is None:
        return []
    w, model = json.loads(f.read_text()), f.name[len("winner-"):-len("-v2.json")]
    old, st = f.with_name(f"winner-{model}-anchored.json"), DATA / "store" / model
    gb = lambda p: Path(p).stat().st_size / 1e9 if Path(p).exists() else float("nan")  # noqa: E731
    o, og = w["original"]["accuracy"], gb(st / f"{model}-BF16.gguf")
    rows = [("Original (16-bit)", w["original"], og), ("**Ours**", w["ours"], gb(w["file"])),
            (f"llama.cpp preset {w['preset']}", w["preset_test"], gb(st / f"preset-{w['preset']}.gguf"))]
    if old.exists():
        r = json.loads(old.read_text())
        r = r["rows"][r["winner"]]
        rows.append(("Ours, first method", {"accuracy": r["mean_accuracy_pct"], "kl": r["kl"], "ppl": r["ppl"],
                                             "flips": None, "per_task": r["per_task_pct"]}, float("nan")))
    n = lambda x, f=".2f": "–" if x is None or x != x else format(x, f)  # noqa: E731
    out = [f"### Result: {w['verdict']}, ours − preset {w['diff']:+.2f} points [{w['ci'][0]:+.2f}, {w['ci'][1]:+.2f}]",
           "", "| held-out test | accuracy | MMLU | HellaSwag | ARC | WinoGrande | GSM8K | points lost vs original | "
           "% of original kept | KL | perplexity | flips | file GB | % of original size |", "|---" * 14 + "|"]
    for name, r, size in rows:
        t = r["per_task"]
        out.append(f"| {name} | {n(r['accuracy'])} | " + " | ".join(n(t.get(k), ".1f") for k in (
            "mmlu", "hellaswag", "arc_challenge", "winogrande", "gsm8k")) + f" | {n(o - r['accuracy'])} | "
            f"{n(100 * r['accuracy'] / o, '.1f')} | {n(r['kl'], '.4f')} | {n(r['ppl'])} | {n(r['flips'], '.1f')} | "
            f"{n(size)} | {n(100 * size / og, '.1f')} |")
    p, us = w["preset_test"], w["ours"]
    types: dict[str, int] = {}
    for t in w["assignment"].values():
        types[t] = types.get(t, 0) + 1
    out += ["", f"- Ours vs the preset: {us['accuracy'] - p['accuracy']:+.2f} points "
            f"({100 * (us['accuracy'] / p['accuracy'] - 1):+.1f} %); we lose {o - us['accuracy']:.2f} points of the "
            f"original's accuracy, the preset {o - p['accuracy']:.2f} ({100 * (1 - (o - us['accuracy']) / (o - p['accuracy'])):.0f} % less); "
            f"KL {100 * (us['kl'] / p['kl'] - 1):+.0f} %; flips {us['flips']:.1f} vs {p['flips']:.1f} %",
            f"- RAM on the Pi (predicted): ours {w['ram_gib']:.3f} GiB, budget {b} GiB",
            f"- **Our configuration** ({len(w['assignment'])} matrices): " + ", ".join(
                f"{t} ×{c}" for t, c in sorted(types.items(), key=lambda x: -x[1])), ""]
    return out


def render() -> str:
    head = [f"# Panther: live progress", "", f"Updated {time.strftime('%d %b %H:%M', time.gmtime(time.time() + 19800))}"
            " IST, every 30 s. Each contest: one model at one RAM budget on the Raspberry Pi 5; our quantization (every"
            " matrix its own type, smarter rounding, genetic search, distillation) against llama.cpp's best preset that"
            " fits the same budget, on held-out questions. WIN / LOSE: the 95% interval of the accuracy difference is "
            "all above / below 0, otherwise TIE. Every term: [How it works](https://github.com/RachitKumar205/panther#how-it-works).",
            "", "## Verdicts", "", "| contest | verdict | ours - preset, points [95% interval] | flips ours / preset |",
            "|---|---|---|---|"]
    log = (DATA / "logs/panther.log").read_text().splitlines() if (DATA / "logs/panther.log").exists() else []
    state = {}
    for x in log:
        if m := re.match(r"\d\d:\d\d (\S+): (.*)", x):
            state[m.group(1)] = m.group(2)
    paused = [f"**PAUSED** ({w})" for f, w in (("PAUSED", "by the user"), ("PAUSED.guard", "GitLab unhealthy"))
              if (DATA / f).exists()]
    ln = lanes()
    step = {f"{sm.lower()}-b{b}": next((m.group(1) for x in reversed(ls) if (m := re.search(r"step (\w+): start", x))),
                                       "") for b, sm, ls in ln}
    head[4:4] = paused + ([""] if paused else []) + ["## Contests", ""] + [
        f"- **{k.replace('-b', '@')}**: {v}" + (f", now: {WHAT[step[k]]}" if v == "started" and step.get(k) else "")
        for k, v in state.items()] + ["", "## Steps of every contest", ""] + [
        f"{i}. **{s}**: {DESCRIBE[s]}" for i, s in enumerate(STEPS, 1)] + [
        "", "## Recent events", "", *[f"    {x}" for x in recent(log[-10:] + [x for _, _, ls in ln for x in ls[-6:]])], ""]
    body = []
    for b, sm, ls in ln:
        v = next((x for x in reversed(ls) if "our winner" in x), None)
        if v and (m := re.search(r"our winner ([+-][\d.]+) pts \[([+-][\d.]+), ([+-][\d.]+)\]", v)):
            d, lo, hi = map(float, m.groups())
            word = "WIN" if lo > 0 else "LOSE" if hi < 0 else "TIE"
            fl = re.search(r"flips ([\d.]+) vs ([\d.]+)", v)
            head.append(f"| {sm} @ {b} GiB | **{word}** | {d:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                        f"{fl.group(1) + ' / ' + fl.group(2) if fl else ''} |")
        else:
            head.append(f"| {sm} @ {b} GiB | running | | |")
        body += lane(b, sm, ls)[:2] + results(b, sm) + lane(b, sm, ls)[2:]
    pi = ["## On the Pi (our winners: real peak RAM and speed)", ""] + [f"- {k}: {v}" for k, v in sorted(PI.items())
                                                                        if "@" in k] + ["", "(none yet)" if not any(
        "@" in k for k in PI) else "", ""]
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
