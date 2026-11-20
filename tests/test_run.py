"""The run's machinery, no GPU: the claim (a loud refusal naming who holds the rest; an ended run's claim lapses), the
one FAILED path, and nothing measured or hashed twice."""

import os
import re
import subprocess
import sys
from pathlib import Path

from panther import measure
from panther.model import GIB, hold, sha

ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}


def py(data: Path, *args: str):
    return subprocess.run([sys.executable, "-m", "panther.run", *args], env={**ENV, "PANTHER_DATA": str(data)},
                          capture_output=True, text=True)


def test_claims(tmp_path):
    holder = subprocess.Popen([sys.executable, "-c", "import time; from panther import run; from panther.model import "
                               "RUN; RUN.update(model='Qwen2.5-14B-Instruct', budget=5); run.claim(); print('held', "
                               "flush=True); time.sleep(120)"], env={**ENV, "PANTHER_DATA": str(tmp_path)},
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        r = py(tmp_path, "claim", "24b", "7")  # refused: loud, but not a failed contest
        assert r.returncode == 3 and "REFUSED: 24b-b7 does not fit" in r.stderr and "FAILED" not in r.stderr
        assert re.fullmatch(r"\d\d:\d\d 24b-b7: refused: 24b-b7 does not fit: .* 14b-b5 \(\d+ GiB GPU.*\n",
                            (tmp_path / "logs/panther.log").read_text())
        assert not (tmp_path / "logs/progress-b7.log").exists()
        assert py(tmp_path, "claim", "1.5b", "1").returncode == 0  # a second run fits beside it
    finally:
        holder.kill()
        holder.wait()
    r = py(tmp_path, "claim", "24b", "7")  # the holder ended: its claim lapsed
    assert r.returncode == 0 and 48 * GIB < int(r.stdout.split()[-1]) <= 60 * GIB


def test_a_failed_step_is_loud(tmp_path):
    r = py(tmp_path, "run", "1.5b", "1")  # claimed, then the split check finds no splits
    assert r.returncode == 3 and "Traceback" in r.stderr
    assert re.fullmatch(r"\d\d:\d\d 1\.5b-b1: FAILED in check: FileNotFoundError: .+ — fix and rerun "
                        r"\./panther run 1\.5b@1 \(finished steps are kept\)",
                        (tmp_path / "logs/panther.log").read_text().splitlines()[-1])
    assert "1.5B FAILED in check: FileNotFoundError" in (tmp_path / "logs/progress-b1.log").read_text()


def test_nothing_twice(tmp_path, monkeypatch):
    kl = Path(__file__).parent / "fixtures/llama/kl_q6k_48chunks.log"
    fake = tmp_path / "llama-perplexity"
    fake.write_text(f'#!/bin/sh\necho run >> "{tmp_path}/calls"\ncat "{kl}"\n')
    fake.chmod(0o755)
    monkeypatch.setattr(measure, "DATA", tmp_path)
    monkeypatch.setattr(measure, "BIN", tmp_path)
    monkeypatch.setattr(measure, "llama_commit", lambda: "a" * 40)
    hold.set(10 * GIB, 10 * GIB)
    measure.kld("Qwen2.5-1.5B-Instruct", "select").parent.mkdir(parents=True)
    measure.kld("Qwen2.5-1.5B-Instruct", "select").write_bytes(b"ref")
    f = tmp_path / "x.gguf"
    f.write_bytes(b"gguf")
    first = measure.run("kl", "Qwen2.5-1.5B-Instruct", f, "x:1", split="select")
    assert measure.run("kl", "Qwen2.5-1.5B-Instruct", f, "x:1", split="select") == first
    assert (tmp_path / "calls").read_text().count("run") == 1
    assert first["how"]["llama"] == "a" * 40 and first["how"]["ctx"][0] == measure.CTX
    monkeypatch.setattr(measure, "llama_commit", lambda: "b" * 40)  # another llama.cpp: measured anew
    assert measure.run("kl", "Qwen2.5-1.5B-Instruct", f, "x:1", split="select") != first
    assert (tmp_path / "calls").read_text().count("run") == 2
    h = sha(f)
    assert sha(f) == h and Path(f"{f}.sha256").exists()  # kept beside the file
    f.write_bytes(b"GGUF")
    assert sha(f) != h  # a changed file is hashed anew, so its results are measured anew


def test_derived_files_carry_their_inputs(tmp_path):
    from panther.model import made, stamp
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert not made(f, "i")
    stamp(f, "i")
    assert made(f, "i") and not made(f, "j")
    os.utime(f, ns=(1, 1))  # rewritten (killed before its new stamp): not taken as made from the old inputs
    assert not made(f, "i")


def test_presets_shared_by_budgets(tmp_path, monkeypatch):
    from panther import store
    monkeypatch.setattr(store, "STORE", tmp_path / "store")
    monkeypatch.setattr(store, "RW", tmp_path / "rw")
    for p in ("Q2_K", "Q3_K_S", "Q4_K_S"):
        store.preset_file("M", p).parent.mkdir(parents=True, exist_ok=True)
        store.preset_file("M", p).write_bytes(b"x")
    for b, p in ((5, "Q3_K_S"), (7, "Q4_K_S")):  # each budget's record chose one
        (tmp_path / f"rw/budget-{b}").mkdir(parents=True)
        (tmp_path / f"rw/budget-{b}/preset-M.json").write_text(f'{{"preset": "{p}"}}')
    store.drop_presets("M", ["m-b3"])  # another run of the model is active: nothing removed
    assert store.preset_file("M", "Q2_K").exists()
    store.drop_presets("M", [])
    assert sorted(f.name for f in (tmp_path / "store/M").iterdir()) == ["preset-Q3_K_S.gguf", "preset-Q4_K_S.gguf"]
