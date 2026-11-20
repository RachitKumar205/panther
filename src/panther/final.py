"""The final step (decisions 11-15): finalists distilled and not, the winner on all selection questions, the guard,
the held-out verdict; the Pi bundle."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from panther import measure as ms
from panther.data import WEB, chat_text
from panther.model import (CFG, DATA, RULES, consts, digest, done, hold, lock, note, path, put, record, sha, short,
                           trained)
from panther.search import build, label
from panther.store import copy_of, preset_file


def final_ins(model: str) -> tuple[str, list]:
    """The step's inputs (the list's and the preset's carry measure's), and what a distilled file is made from."""
    lst, pr = json.loads(record("list").read_text()), json.loads(record("preset", "").read_text())
    dins = [sha(copy_of(model, "BF16"))] + [sha(f) for f in teach(model)] + [consts("distill"), RULES["final"]]
    return digest([lst["inputs"], pr["inputs"], dins]), dins


def teach(model: str) -> list[Path]:
    return [chat_text(model, "train"), WEB["train"]]


def final(model: str, budget: int) -> None:
    """Overlapped: everything built is scored at once, also while a distillation runs (one at a time, booked at its
    own size), and the held-out tests run side by side."""
    from panther.distill import distill
    lst, pr, win = json.loads(record("list").read_text()), json.loads(record("preset", "").read_text()), record("winner")
    (ins, dins), bf16 = final_ins(model), copy_of(model, "BF16")
    if done(win, ins):
        return
    items, p, pl, base = lst["finalists"], pr["preset"], pr["label"], pr["tried"][pr["preset"]]["score"]
    pf, files, labels, rows = preset_file(model, p), {}, {}, {}

    def scored(name: str) -> None:
        rows[name] = r = ms.against(ms.score(model, labels[name], files[name], full=True), base)
        note(f"final {name}: score {r['score']:.2f} (flips {r['flips']:.2f}%, KL {r['kl']:.4f}, accuracy "
             f"{r['accuracy']:.2f}) on all selection questions; relative {r['relative']:.1f}; same top word "
             f"{r['same_top']:.2f}%")
    with ThreadPoolExecutor(8) as ex:
        jobs = [ex.submit(ms.score, model, pl, pf, full=True)]
        for k, it in items.items():  # distillation can hurt near-lossless files: the undistilled file competes too
            name, dg = f"{k} undistilled", digest(label(model, it["assignment"]))
            files[name] = build(model, it["assignment"], path(model, "tuned", f"{budget}-{dg}-plain.gguf"))
            labels[name] = label(model, it["assignment"])  # the same file the GA scored: its results are reused
            jobs.append(ex.submit(scored, name))
        for k in items:
            plain = files[f"{k} undistilled"]
            files[k] = path(model, "tuned", f"{budget}-{digest([sha(plain), dins])}.gguf")
            if not files[k].exists():
                with lock(f"distill-{model}"), hold(*trained(model)):  # the 16-bit load; files built are scored
                    distill(plain, bf16, teach(model), files[k], DATA / "reference" / model / "teacher-top64.pt",
                            lambda m, k=k: note(f"{k} {m}"))
            labels[k] = f"d:{sha(files[k])[:12]}"
            jobs.append(ex.submit(scored, k))
        preset_sel = jobs[0].result()
        for j in jobs:
            j.result()
    w = min(rows, key=lambda k: rows[k]["score"])
    item = items[w.split()[0]]
    g = ms.versus({**rows[w], "correct": ms.correct(model, labels[w])}, {**preset_sel, "correct": ms.correct(model, pl)},
                  ms.correct(model, ms.original(model)))  # the guard: the same rule as the verdict, on selection data
    note(f"guard: {w} vs preset {p} on all selection questions: relative {g['relative']:.1f} "
         f"[{g['relative_ci'][0]:.1f}, {g['relative_ci'][1]:.1f}]")
    if warn := g["verdict"] == "LOSE":  # reported, not a stop: a run that would lose still gets its verdict
        note("guard: WORSE than the preset on selection questions; held-out test run anyway")
    ours, theirs, orig = ms.parallel([lambda: ms.test(model, files[w], labels[w]), lambda: ms.test(model, pf, pl),
                                      lambda: ms.test(model, bf16, ms.original(model))])
    # the verdict: our relative score on held-out data (relative, relative_ci; diff, ci = those - 100, negative =
    # better than the preset); the accuracy difference is reported, not judged
    v = ms.versus(ours, theirs, orig["correct"])
    res = {"inputs": ins, "winner": w, "file": str(files[w]), "label": labels[w], "assignment": item["assignment"],
           "ram_gib": item["ram_gib"], "preset": p, **v, **ms.beside(ours, theirs, orig),
           "accuracy_diff": ms.acc(ours["correct"]) - ms.acc(theirs["correct"]), "selection": rows[w],
           "preset_score": base, "guard": {**g, "warning": warn}}
    for k, r in (("ours", ours), ("preset_test", theirs), ("original", orig)):
        res[k] = ms.summary(r, orig)
    put(win, json.dumps(res, indent=1))
    for f in files.values():  # only the winner's file is kept
        if f != files[w]:
            f.unlink()
    note(f"held-out test vs preset {p}: {res['ours']['accuracy']:.2f} vs {res['preset_test']['accuracy']:.2f} "
         f"(original {res['original']['accuracy']:.2f}, ours - preset {res['accuracy_diff']:+.2f} pts) | our winner "
         f"{v['diff']:+.2f} rel [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] {v['verdict']} (relative score "
         f"{v['relative']:.1f}, 100 = the preset), flips {res['ours']['flips']:.2f} vs "
         f"{res['preset_test']['flips']:.2f}%, same top word "
         f"{res['ours']['same_top']:.2f} vs {res['preset_test']['same_top']:.2f}%{ms.beside_text(res, v['verdict'])}")


def bundle(model: str, budget: int) -> Path:
    """A Pi bundle: the winner's file and a manifest with its hash, budget and predicted RAM."""
    w = json.loads(record("winner").read_text())
    src, tag = Path(w["file"]), "-v2"
    d = DATA / "pi_bundle" / f"budget-{budget}-{short(model).lower()}{tag}"
    f, m = d / f"{model.lower()}-b{budget}{tag}.gguf", d / "manifest.json"
    man = {"file": f.name, "sha256": sha(src), "budget_gib": budget, "predicted_ram_gib": w["ram_gib"],
           "context": CFG["context_tokens"], "kv_cache_type": CFG["kv_cache_type"]}
    if not (f.exists() and m.exists() and json.loads(m.read_text()) == man and f.stat().st_size == src.stat().st_size):
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, f"{f}.part")
        Path(f"{f}.part").rename(f)
        m.write_text(json.dumps(man, indent=1))
    return d

