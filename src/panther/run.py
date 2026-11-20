"""python -m panther.run KEYWORD SIZE BUDGET, started by ./panther in containers (none has both GPU and network):
claim, fetch (CPU, network), run (GPU), publish (CPU, network).
Any error: one FAILED line in the main and progress logs, the traceback in the job's log, exit code 3. A run refused
at its claim never started: the refusal on stderr and one "refused" line in the main log, exit code 3."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

from panther import data, final, measure, search, store
from panther.model import DATA, GIB, RUN, Model, clock, done, hold, lock, named, note, record, short

CAP_GPU, CAP_RAM, RUN_RAM = 92 * GIB, 96 * GIB, 64 * GIB  # 124 GB host: the rest stays for GitLab and the system
GENOMES = 8  # genomes scored at once that a claim is sized for (fewer if the caps do not hold them)
RESERVE = 4 * GIB  # the orchestrator's own RAM outside its bookings: torch and CUDA, tokenizer, GGUF readers
CLAIMS = DATA / "locks/claims.json"
_HELD = []  # this process's own run lock, when no ./panther holds it


def rid() -> str:
    return f"{short(RUN['model']).lower()}-b{RUN['budget']}"


def live(r: str) -> bool:
    """A run is active while its lock file is held (by ./panther for the whole run, or by the run's own process):
    the kernel lets go when the holder ends, however it ends."""
    with open(DATA / "locks" / f"run-{r}.lock", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def active() -> dict:
    """Active runs' claims (call under the claims lock); a claim whose run has ended is dropped."""
    return {k: v for k, v in (json.loads(CLAIMS.read_text()) if CLAIMS.exists() else {}).items() if live(k)}


class Refused(SystemExit):
    pass


def sizes(model: str, budget: int) -> tuple[int, int, int]:
    """The run's GPU and RAM claim, and how many genomes it scores at once. GPU: distillation's peak (the 16-bit
    model + 20 GiB) or k genomes' llama.cpp runs, whichever is more. RAM: RESERVE + k genomes' runs, or the
    transformers load of the 16-bit model in rounding and distillation (they book the whole claim, so run alone):
    the model in host RAM before it moves to the GPU, one tensor's fp32 copy and the orchestrator, b16 + 8 GiB."""
    b16, (g, m) = 2 * Model(model).params, measure.need(budget * GIB)
    k = max(1, min(GENOMES, (RUN_RAM - RESERVE) // m, CAP_GPU // g))
    return min(CAP_GPU, max(b16 + 20 * GIB, k * g)), min(RUN_RAM, max(b16 + 8 * GIB, RESERVE + k * m)), k


def claim() -> int:
    """The whole run's GPU and RAM (sizes), or a refusal naming the runs that hold the rest: all claims stay within
    CAP_GPU and CAP_RAM. Returns the RAM (the container's limit); the run books within it less RESERVE."""
    r, (gpu, ram, _) = rid(), sizes(RUN["model"], RUN["budget"])
    with lock("claims"):
        f = open(DATA / "locks" / f"run-{r}.lock", "a")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _HELD.append(f)  # no ./panther holds it: this process does, until it ends
        except BlockingIOError:
            f.close()  # ./panther holds it for the whole run
        book = active()
        if r not in book:
            others = [f"{k} ({v['gpu'] / GIB:.0f} GiB GPU, {v['ram'] / GIB:.0f} GiB RAM)" for k, v in book.items()]
            if sum(v["gpu"] for v in book.values()) + gpu > CAP_GPU or sum(v["ram"] for v in book.values()) + ram > CAP_RAM:
                raise Refused(f"{r} does not fit: it needs {gpu / GIB:.0f} GiB of GPU and {ram / GIB:.0f} GiB of RAM "
                                 f"and {', '.join(others)} hold the rest of {CAP_GPU // GIB} GiB GPU and "
                                 f"{CAP_RAM // GIB} GiB RAM; start it when one of them ends")
            book[r] = {"model": RUN["model"], "budget": RUN["budget"], "kind": "run", "gpu": gpu, "ram": ram,
                       "host": os.uname().nodename, "since": time.strftime("%Y-%m-%d %H:%M")}
            measure.write(CLAIMS, book)
    hold.set(gpu, ram - RESERVE)
    return ram


def others(book: dict) -> list[str]:
    return [k for k, v in book.items() if v["model"] == RUN["model"] and k != rid()]


def unpreset() -> None:
    with lock("claims"):  # no run of the model can start between the check and the removal
        store.drop_presets(RUN["model"], others(active()))


def tidy() -> None:
    """The end of a run: the store backed up, its type copies removed unless the model is in use, the claim dropped."""
    store.backup(RUN["model"])
    with lock("claims"):  # no run of the model can start between the check and the removal
        book = active()
        store.clean(RUN["model"], others(book))
        store.drop_presets(RUN["model"], others(book))
        book.pop(rid(), None)
        measure.write(CLAIMS, book)


def finished(M: str, B: int) -> bool:
    """Every step of the contest done from current inputs: its run needs no type copies (fetch restores none)."""
    try:
        return search.error_done(M) and done(record("preset", ""), search.preset_ins(M, B)) and \
            done(record("list"), search.ga_ins(M, B)) and done(record("winner"), final.final_ins(M)[0])
    except FileNotFoundError:
        return False


def step(name: str, fn, logged: bool = True) -> None:
    RUN["step"], t0 = name, time.time()
    if logged:
        note(f"step {name}: start")
    fn()
    if logged:
        note(f"step {name}: done in {(time.time() - t0) / 60:.0f} min")


def push(d: Path) -> None:
    from huggingface_hub import HfApi
    repo = os.environ["PANTHER_HF_REPO"].format(size=short(RUN["model"]).lower(), budget=RUN["budget"]) + "-v2"
    HfApi().create_repo(repo, private=False, exist_ok=True)
    HfApi().upload_folder(repo_id=repo, folder_path=d)
    note(f"Pi bundle uploaded: {repo}")


def fetch() -> None:
    step("fetch", lambda: finished(RUN["model"], RUN["budget"]) or store.fetch(RUN["model"]), False)
    step("prompts", data.prompts, False)


def run() -> None:
    M, B = RUN["model"], RUN["budget"]
    step("claim", claim, False)
    step("check", data.check, False)
    for s, fn in (("store", lambda: store.build(M)), ("error", lambda: search.error(M)), ("refs", lambda: measure.refs(M)),
                  ("preset", lambda: (search.preset(M, B), unpreset())), ("ga", lambda: search.ga(M, B)),
                  ("final", lambda: final.final(M, B)), ("bundle", lambda: final.bundle(M, B))):
        step(s, fn)


def publish() -> None:
    step("upload", lambda: push(final.bundle(RUN["model"], RUN["budget"])), False)
    step("tidy", tidy, False)


KEYWORDS = {"claim": lambda: print(claim()), "fetch": fetch, "run": run, "publish": publish}


def main() -> None:
    kw, size, budget = sys.argv[1:4]
    RUN.update(model=named(size), budget=int(budget), step=kw)
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    try:
        KEYWORDS[kw]()
    except (Exception, SystemExit) as e:
        what = " ".join((str(e) if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}").split())[:400]
        sz = short(RUN["model"]).lower()
        if kw == "claim" and isinstance(e, Refused):  # never started: not a failed contest
            print(f"REFUSED: {what}", file=sys.stderr)
            with open(DATA / "logs/panther.log", "a") as f:
                f.write(f"{clock()} {sz}-b{RUN['budget']}: refused: {what}\n")
            sys.exit(3)
        traceback.print_exc()
        msg = f"FAILED in {RUN['step']}: {what} — fix and rerun ./panther run {sz}@{RUN['budget']} (finished steps are kept)"
        with open(DATA / "logs/panther.log", "a") as f:
            f.write(f"{clock()} {sz}-b{RUN['budget']}: {msg}\n")
        note(msg)
        sys.exit(3)


if __name__ == "__main__":
    main()
