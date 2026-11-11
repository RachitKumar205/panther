"""Run one quality evaluation with llama-perplexity and write its record (docs/PIPELINE.md A3, D33).

  python -m panther.quality.run kl  --model M --gguf FILE --reference sensitivity --build cuda [--label L]
  python -m panther.quality.run ppl --model M --gguf FILE --text selection        --build cpu  [--label L]
  python -m panther.quality.run task --model M --gguf FILE --task mmlu --split selection --build cuda

Records go to $PANTHER_DATA/results/<model>/<eval_id>.json and the raw output to .../logs/<eval_id>.log.
KL runs take -c/-b from the reference manifest (D28). Every run's output is parsed and checked, because
llama-perplexity exits 0 on failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import gguf

from panther.gguftools.assemble import ASSIGNMENT_KEY
from panther.models.inventory import REPO_ROOT
from panther.quality.parse import parse_kl, parse_multiple_choice, parse_ppl_type1, parse_winogrande

DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
LLAMA = Path(os.environ.get("PANTHER_LLAMA", REPO_ROOT / "third_party/llama.cpp"))
PPL_CTX = 2048  # PLAN.md §2.1: perplexity at context 2048, stride = context
TASK_CTX = 4096  # per sequence (D38)
TASK_PARALLEL = 5  # >= the most answers any question has (ARC-C has 5-choice items); total context 5 x 4096


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def assignment_id(path: Path) -> str | None:
    """The assignment digest assembled files carry in their metadata; None for store files."""
    field = gguf.GGUFReader(path).fields.get(ASSIGNMENT_KEY)
    return field.contents() if field is not None else None


def llama_commit() -> str:
    out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(LLAMA), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return out or "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["kl", "ppl", "task"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--build", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--reference", help="kl: reference name (sensitivity, selection)")
    ap.add_argument("--text", help="ppl: split text name (sensitivity, selection; never test in Phase A)")
    ap.add_argument("--task", choices=["hellaswag", "arc_challenge", "mmlu", "winogrande"])
    ap.add_argument("--split", help="task: selection (test only in Phase D, D34)")
    ap.add_argument("--label", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("PANTHER_THREADS", "16")))
    a = ap.parse_args(argv)

    out_dir = DATA / "results" / a.model
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    file_sha = sha256(a.gguf)
    eval_id = f"{started:%Y%m%dT%H%M%SZ}-{a.kind}-{a.build}-{file_sha[:10]}"
    binary = LLAMA / f"build-{a.build}" / "bin" / "llama-perplexity"
    gpu = ["-ngl", "99"] if a.build == "cuda" else []

    how: dict = {"build": a.build, "threads": a.threads}
    if a.kind == "kl":
        if not a.reference:
            ap.error("kl needs --reference")
        ref_manifest = json.loads((DATA / "reference" / a.model / "manifest.json").read_text())
        ref = ref_manifest["references"][a.reference]
        flags = ref_manifest["kl_runs_must_use"]
        cmd = [str(binary), "-m", str(a.gguf), "--kl-divergence-base",
               str(DATA / "reference" / a.model / ref["file"]), "--kl-divergence",
               "-c", str(flags["-c"]), "-b", str(flags["-b"]), "-t", str(a.threads), *gpu]
        how |= {"reference": a.reference, "reference_sha256": ref["sha256"], "c": flags["-c"], "b": flags["-b"]}
    elif a.kind == "task":
        if not a.task or not a.split:
            ap.error("task needs --task and --split")
        if a.split.startswith("test"):
            ap.error("the test split is touched once, in Phase D, by its own script (D34)")
        task_manifest = json.loads((DATA / "splits" / "llama_tasks" / "manifest.json").read_text())
        entry = task_manifest["files"][f"{a.split}/{a.task}"]
        path = DATA / "splits" / entry["file"]
        mode = (["-f", str(path), "--winogrande", "--winogrande-tasks", "0"] if a.task == "winogrande"
                else ["-bf", str(path), "--multiple-choice", "--multiple-choice-tasks", "0"])
        cmd = [str(binary), "-m", str(a.gguf), *mode, "-c", str(TASK_CTX), "-np", str(TASK_PARALLEL),
               "-t", str(a.threads), *gpu]
        how |= {"task": a.task, "split": a.split, "file_sha256": entry["sha256"], "n_items": entry["n"],
                "c": TASK_CTX, "np": TASK_PARALLEL}
    else:
        if not a.text:
            ap.error("ppl needs --text")
        if a.text.startswith("test"):
            ap.error("the test split is touched once, in Phase D, by its own script (D34)")
        text = DATA / "splits" / "text" / f"{a.text}.txt"
        cmd = [str(binary), "-m", str(a.gguf), "-f", str(text), "-c", str(PPL_CTX), "-b", str(PPL_CTX),
               "--ppl-output-type", "1", "-t", str(a.threads), *gpu]
        how |= {"text": a.text, "text_sha256": sha256(text), "c": PPL_CTX, "b": PPL_CTX}

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.monotonic() - t0
    output = proc.stdout + proc.stderr
    (out_dir / "logs" / f"{eval_id}.log").write_text(output)

    record = {
        "eval_id": eval_id, "kind": a.kind, "model": a.model, "label": a.label,
        "evaluated": {"gguf": a.gguf.name, "gguf_sha256": file_sha, "assignment_sha256": assignment_id(a.gguf)},
        "how": how | {"llama_cpp_commit": llama_commit(),
                      "image_id": os.environ.get("PANTHER_IMAGE_ID", "unknown"), "command": cmd},
        "started": started.isoformat(timespec="seconds"), "wall_seconds": round(seconds, 2),
        "exit_code": proc.returncode,
    }
    try:
        if a.kind == "kl":
            record["metrics"] = asdict(parse_kl(output))
        elif a.kind == "task":
            parse = parse_winogrande if a.task == "winogrande" else parse_multiple_choice
            t = parse(output, entry["n"])
            record["metrics"] = {"accuracy_pct": t.accuracy_pct, "accuracy_se_pct": t.accuracy_se_pct,
                                 "n": int(len(t.correct))}
            record["per_item"] = {"ids": entry["ids"], "correct": t.correct.tolist()}
        else:
            r = parse_ppl_type1(output)
            record["metrics"] = {"ppl": r.ppl, "ppl_se": r.ppl_se, "chunks": len(r.chunk_mean_nll),
                                 "recovery_rel_error": r.recovery_rel_error}
            record["per_chunk"] = {"mean_nll": [round(x, 7) for x in r.chunk_mean_nll],
                                   "tokens": r.tokens_per_chunk}
        record["ok"] = True
    except ValueError as e:
        record["ok"] = False
        record["error"] = str(e)
    (out_dir / f"{eval_id}.json").write_text(json.dumps(record, indent=2))
    m = record.get("metrics", {})
    if not m:
        summary = record.get("error")
    elif a.kind == "kl":
        summary = f"KL {m['kl_mean']:.6f} p99 {m['kl_p99']:.4f} top1 {m['top1_pct']:.2f}%"
    elif a.kind == "task":
        summary = f"{a.task} {m['accuracy_pct']:.2f}% ± {m['accuracy_se_pct']:.2f} (n={m['n']})"
    else:
        summary = f"PPL {m['ppl']:.4f} ± {m['ppl_se']:.4f} ({m['chunks']} chunks)"
    print(f"{eval_id} {a.label or a.gguf.name}: {summary} [{seconds:.1f}s]", flush=True)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
