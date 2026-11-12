"""Experiment: the GA with GSM8K in its fitness. Uses scripts/server/pipeline.py unchanged and replaces only the
fitness, the selection-question scoring and the output names, so the main pipeline is not affected.

Why: Qwen2.5 7B at 5 GiB lost on the held-out test only through GSM8K (62.0 vs 68.7); the main GA's fitness has no
GSM8K (it generates answers, which is slow), so it could not see math.

Changes:
- fitness = mean of five tasks (MMLU 300, ARC 200, HellaSwag 200, WinoGrande 200 selection questions, as before,
  plus the first 100 GSM8K selection questions, answers generated) - 10 x KL on benchmark-question text;
- the top 3 and the guard use all selection questions of the five tasks (4,500 multiple choice + 500 GSM8K);
- files: list/winner -gsm8k, progress lines "<size>-gsm8k", Hugging Face repo panther-<size>-<B>g-gsm8k.

  python experiments/gsm8k_fitness/run.py ga 5 Qwen2.5-7B-Instruct
  python experiments/gsm8k_fitness/run.py gascore 5 Qwen2.5-7B-Instruct
  python experiments/gsm8k_fitness/run.py compare 5 Qwen2.5-7B-Instruct   # winner vs preset on the held-out test
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "server"))
import pipeline as p  # noqa: E402

p.SUFFIX = "-gsm8k"
_short = p.short_name
p.short_name = lambda model: _short(model) + "-gsm8k"
GSM_FIT, GSM_SEL = 100, 500  # GSM8K selection questions in the fitness / in the top-3 scoring and guard


def gsm8k_selection(model: str, path: Path, label: str, n: int) -> np.ndarray:
    """0/1 per question on the first n GSM8K selection questions (5-shot, greedy, as in the held-out test)."""
    rec = p.out(model, "gsm8k", f"{label.replace(':', '_')}_sel{n}.json")
    if rec.exists():
        return np.array(json.loads(rec.read_text())["correct"], dtype=np.int8)
    items = p.jsonl(p.DATA / "splits/tasks/selection/gsm8k.jsonl")[:n]
    fewshot = p.jsonl(p.DATA / "splits/tasks/fewshot/gsm8k.jsonl")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with p.evaluate.LlamaServer(path, parallel=8, ctx_per_slot=2048, threads=p.THREADS, port=port,
                                log=p.out(model, "servers", f"{path.stem}-sel.log")) as srv:
        comps = srv.map(lambda it: srv.complete(p.evaluate.gsm8k_prompt(it["question"], fewshot),
                                                p.evaluate.GSM8K_MAX_TOKENS, p.evaluate.GSM8K_STOP), items)
    correct = [int(p.evaluate.gsm8k_correct(c, it["answer"])) for it, c in zip(items, comps)]
    rec.write_text(json.dumps({"ids": [it["id"] for it in items], "correct": correct}))
    return np.array(correct, dtype=np.int8)


def ga_scores(model: str, label: str) -> tuple[float, float] | None:
    recs = p.records(model)
    r = p.find(recs, label, reference="bench")
    accs = [p.find(recs, label, task=t, split="selection", n_tasks=n) for t, n in p.GA_FIT_TASKS.items()]
    g = p.out(model, "gsm8k", f"{label.replace(':', '_')}_sel{GSM_FIT}.json")
    if r is None or not all(accs) or not g.exists():
        return None
    mc = [x["metrics"]["accuracy_pct"] for x in accs]
    gsm = 100 * float(np.mean(json.loads(g.read_text())["correct"]))
    return float(np.mean(mc + [gsm])), r["metrics"]["kl_mean"]


def _ga_eval(job: tuple[int, str, dict, bool]) -> tuple[float, float]:
    budget, model, a, fast = job
    p.BUDGET = budget
    label = p.ga_label(budget, a)
    if ga_scores(model, label) is None:
        path = p.fast_probe(model, a) if fast else p.build_file(model, a)
        try:
            with p.gpu_booking(path.stat().st_size + 8 * 2**30, hold_s=1800):
                if not p.find(p.records(model), label, reference="bench"):
                    p.kl(model, path, "bench", label)
                for t, n in p.GA_FIT_TASKS.items():
                    if not p.find(p.records(model), label, task=t, split="selection", n_tasks=n):
                        if not p.evaluate.run("task", model, path, label, p.THREADS, task=t, split="selection", n_tasks=n):
                            raise SystemExit(f"{label} {t} failed")
                gsm8k_selection(model, path, label, GSM_FIT)
        finally:
            path.unlink(missing_ok=True)
    return ga_scores(model, label)


p.ga_scores, p._ga_eval = ga_scores, _ga_eval


def selection_eval(model: str, path: Path | None, a: dict | None, label: str, name: str):
    """All selection questions of the five tasks for one file (built from `a` if no path)."""
    from panther.stats import Evaluation
    built = path is None
    need = not all(p.find(p.records(model), label, task=t, split="selection") for t in p.SUBSET) or not \
        p.out(model, "gsm8k", f"{label.replace(':', '_')}_sel{GSM_SEL}.json").exists()
    if need:
        path = path or p.build_file(model, a)
        try:
            with p.gpu_booking(path.stat().st_size + 10 * 2**30, hold_s=3600):
                for t in p.SUBSET:
                    if not p.find(p.records(model), label, task=t, split="selection"):
                        if not p.evaluate.run("task", model, path, label, p.THREADS, task=t, split="selection"):
                            raise SystemExit(f"{label} {t} failed")
                gsm8k_selection(model, path, label, GSM_SEL)
        finally:
            if built:
                path.unlink(missing_ok=True)
    correct = {t: np.array(p.find(p.records(model), label, task=t, split="selection")["per_item"]["correct"],
                           dtype=np.int8) for t in p.SUBSET}
    correct["gsm8k"] = np.array(json.loads(p.out(model, "gsm8k", f"{label.replace(':', '_')}_sel{GSM_SEL}.json")
                                           .read_text())["correct"], dtype=np.int8)
    return Evaluation(name, correct, np.ones(1), np.ones(1), 0, 0.0, tokenizer=model)


def cmd_gascore(budget: int, model: str) -> int:
    """Top 3 on all selection questions (five tasks); winner = most accurate, ties to the lower KL; guard against
    the real preset on the same questions; then the winner's held-out test."""
    from panther.stats import compare
    items = json.loads(p.list_path(budget, model).read_text())
    sm = p.short_name(model)
    rows, evs = {}, {}
    for name, it in items.items():
        evs[name] = selection_eval(model, None, it["assignment"], p.test_label(budget, name, it["assignment"]), name)
        rows[name] = {"mean_accuracy_pct": 100 * evs[name].mean_accuracy(), "kl": it["bench_kl"]}
        p.note(f"{sm} GA {name}: selection accuracy {rows[name]['mean_accuracy_pct']:.2f} (5 tasks, GSM8K "
               f"{100 * evs[name].task_correct['gsm8k'].mean():.1f}), bench KL {it['bench_kl']:.4f}")
    win, tied = p.choose_winner(rows, evs)
    ftype = json.loads(p.out(f"budget-{budget}", f"preset-{model}.json").read_text())["preset"]
    pev = selection_eval(model, p.DATA / "store" / model / f"preset-{ftype}.gguf", None,
                         f"rw:gacheck{budget}:preset:{ftype}", "preset")
    g = compare(evs[win], pev)
    p.note(f"{sm} guard: {win} vs preset {ftype} on all selection questions (5 tasks): "
           f"{100 * evs[win].mean_accuracy():.2f} vs {100 * pev.mean_accuracy():.2f}, {100 * g.acc_diff:+.2f} pts "
           f"[{100 * g.acc_ci[0]:+.2f}, {100 * g.acc_ci[1]:+.2f}]")
    if g.acc_ci[1] < 0:
        p.note(f"{sm} guard: WORSE than the preset; not taken as our result")
        return 2
    it = items[win]
    label = p.test_label(budget, win, it["assignment"])
    if not p.tests_done(model, label, with_kl=True):
        path = p.build_file(model, it["assignment"])
        try:
            with p.gpu_booking(path.stat().st_size + 10 * 2**30, hold_s=3 * 3600):
                p.run_all_tests(model, path, label, with_kl=True)
        finally:
            path.unlink(missing_ok=True)
    ev = p.evaluation(model, label, it["ram_gib"], win)
    k = p.find(p.records(model), label, reference="selection")["metrics"]
    final = {"ram_gib": it["ram_gib"], "kl": k["kl_mean"], "bench_kl": it["bench_kl"],
             "selection_accuracy_pct": rows[win]["mean_accuracy_pct"], "mean_accuracy_pct": 100 * ev.mean_accuracy(),
             "ppl": ev.perplexity(), "per_task_pct": {t: 100 * float(v.mean()) for t, v in ev.task_correct.items()}}
    p.out(f"budget-{budget}", f"winner-{model}{p.method_suffix(model)}.json").write_text(json.dumps(
        {"winner": win, "tied_on_accuracy": tied, "selection": rows, "rows": {win: final}}, indent=1))
    p.note(f"{sm} GA winner {win}: test accuracy {final['mean_accuracy_pct']:.2f}, KL {final['kl']:.4f}, "
           f"PPL {final['ppl']:.3f}, GSM8K {final['per_task_pct']['gsm8k']:.1f}")
    return 0


def cmd_compare(budget: int, model: str) -> int:
    """The winner against the preset on the held-out test (paired bootstrap), logged under <size>-gsm8k."""
    from panther.stats import compare
    ftype = json.loads(p.out(f"budget-{budget}", f"preset-{model}.json").read_text())["preset"]
    w = json.loads(p.out(f"budget-{budget}", f"winner-{model}-gsm8k.json").read_text())
    it = json.loads(p.list_path(budget, model).read_text())[w["winner"]]
    ev_w = p.evaluation(model, p.test_label(budget, w["winner"], it["assignment"]), it["ram_gib"], w["winner"])
    ev_p = p.evaluation(model, p.test_label(budget, f"preset-{ftype}", None), 0, "preset")
    c = compare(ev_w, ev_p)
    per = " ".join(f"{t[:4]} {100 * ev_w.task_correct[t].mean():.1f}/{100 * ev_p.task_correct[t].mean():.1f}"
                   for t in ev_w.task_correct)
    p.note(f"{p.short_name(model)} held-out test vs preset {ftype}: {100 * ev_w.mean_accuracy():.2f} vs "
           f"{100 * ev_p.mean_accuracy():.2f}, our winner {100 * c.acc_diff:+.2f} pts [{100 * c.acc_ci[0]:+.2f}, "
           f"{100 * c.acc_ci[1]:+.2f}] | ours/preset: {per}")
    return 0


def main() -> int:
    cmd, budget, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    p.BUDGET = budget
    if cmd == "ga":  # the main GA writes ga-<model>(-live).json; keep the first method's copies
        for f in (f"ga-{model}.json", f"ga-{model}-live.json"):
            src = p.out(f"budget-{budget}", f)
            dst = src.with_name(src.name.replace(".json", "-mc-only.json"))
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())
        lb = p.out(f"budget-{budget}", f"ga-{model}-live.json")
        lb.unlink(missing_ok=True)  # start from the population, not the first method's best
        return p.cmd_ga(budget, model)
    return {"gascore": cmd_gascore, "compare": cmd_compare}[cmd](budget, model)


if __name__ == "__main__":
    sys.exit(main())
