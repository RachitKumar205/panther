"""The offline error per matrix and type (decision 5), llama.cpp's best preset that fits (decision 7), and the genetic
search over one type per matrix inside the RAM window (decisions 8-10), with CP-SAT proposals."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import numpy as np

from panther import measure as ms
from panther.data import REF
from panther.model import BIN, GIB, THREADS, Model, bpw, digest, done, hold, lock, note, path, put, record, sha, short
from panther.store import copy_of, ident, imatrix, preset_file

WINDOW = (-0.01, 0.005)  # a genome's RAM must be within -1% / +0.5% of the budget
POP, ELITE, MAX_EVALS, MIN_EVALS, PATIENCE, TOP, SOLVED = 16, 4, 70, 30, 15, 3, 4  # an improvement: beyond 1 SE
DROP_MARGIN = 3.0  # dropped only if its first-look score is this much worse than the population's worst (~2 SE)
PRESETS = {"IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7, "Q2_K_S": 2.9,
           "IQ3_XXS": 3.06, "Q2_K": 3.2, "IQ3_XS": 3.3, "IQ3_S": 3.44, "Q3_K_S": 3.5, "IQ3_M": 3.66, "Q3_K_M": 3.9,
           "IQ4_XS": 4.25, "Q3_K_L": 4.27, "Q4_K_S": 4.58, "Q4_K_M": 4.89, "Q5_K_S": 5.54, "Q5_K_M": 5.69,
           "Q6_K": 6.56, "Q8_0": 8.5}


@cache
def tensors(f: Path) -> dict:
    import gguf
    return {t.name: t for t in gguf.GGUFReader(f).tensors}


def _error_one(job: tuple[str, str, list[str], np.ndarray]) -> tuple[str, dict[str, float]]:
    from gguf.quants import dequantize
    model, name, types, imp = job
    deq = lambda t: dequantize(tensors(copy_of(model, t))[name].data, tensors(copy_of(model, t))[name].tensor_type)  # noqa: E731
    w = deq("BF16").astype(np.float32)
    return name, {t: float(((w - deq(t).reshape(w.shape)) ** 2).sum(axis=0) @ imp[: w.shape[1]]) for t in types}


def error_done(model: str) -> bool:
    dst = path(model, "error.json")
    return done(dst, ident(model)) and all(set(ts) <= set(json.loads(dst.read_text())["error"].get(g, {}))
                                           for g, ts in Model(model).allowed().items())


def error(model: str) -> None:
    """Offline error of every matrix in every type: squared error weighted by the importance matrix. Per model: one
    run of it computes, another finds it done."""
    with lock(f"error-{model}"):
        if not error_done(model):
            _error(model)


def _error(model: str) -> None:
    imat = {k: np.asarray(v.data, dtype=np.float64).reshape(-1) for k, v in tensors(imatrix(model)).items()}
    jobs = []
    for name, types in Model(model).allowed().items():
        cnt = max(float(imat.get(f"{name}.counts", np.ones(1))[0]), 1.0)
        jobs.append((model, name, types, imat.get(f"{name}.in_sum2", np.ones(1 << 16)) / cnt))
    with ProcessPoolExecutor(8) as ex:
        err = dict(ex.map(_error_one, jobs))
    put(path(model, "error.json"), json.dumps({"inputs": ident(model), "error": err}))


def solve(cost: dict[str, dict[str, int]], penalty: dict[str, dict[str, float]], lo: int, hi: int, k: int,
          min_distance: int = 3, seconds: float = 20) -> list[dict[str, str]]:
    """CP-SAT: one type per matrix, total bytes in [lo, hi], least penalty; k answers min_distance matrices apart."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    x = {(g, t): m.NewBoolVar(f"{g}:{t}") for g in cost for t in cost[g]}
    for g in cost:
        m.AddExactlyOne(x[g, t] for t in cost[g])
    m.AddLinearConstraint(sum(cost[g][t] * v for (g, t), v in x.items()), lo, hi)
    m.Minimize(sum(round(1e6 * penalty[g][t]) * v for (g, t), v in x.items()))
    out = []
    for _ in range(k):
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds, s.parameters.num_workers = seconds, 8
        if s.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        out.append({g: t for (g, t), v in x.items() if s.Value(v)})
        m.Add(sum(x[g, t] for g, t in out[-1].items()) <= len(cost) - min_distance)
    return out


class Genes:
    """Genome = one type per matrix inside the RAM window; mutation, crossover, repair and the solver's proposals."""

    def __init__(self, model: str, budget: int, seed: int = 0) -> None:
        M, self.rng, self.model = Model(model), random.Random(seed), model
        self.steps = M.allowed()
        self.mats = {m.name: m for m in M.matrices}
        self.bytes = {g: {t: self.mats[g].nbytes(t) for t in ts} for g, ts in self.steps.items()}
        err = json.loads(path(model, "error.json").read_text())["error"]
        self.err = {g: {t: err[g][t] for t in ts} for g, ts in self.steps.items()}
        a0 = {g: ts[0] for g, ts in self.steps.items()}
        self.const = M.ram(a0) - sum(self.bytes[g][a0[g]] for g in a0)
        self.lo, self.hi = budget * (1 + WINDOW[0]) * GIB, budget * (1 + WINDOW[1]) * GIB
        self.group = {g: (m.role, m.layer * 4 // M.layers) for g, m in self.mats.items()}

    def ram(self, a: dict[str, str]) -> int:
        return self.const + sum(self.bytes[g][a[g]] for g in a)

    def fits(self, a: dict[str, str]) -> bool:
        return self.lo <= self.ram(a) <= self.hi

    def move(self, a: dict[str, str], g: str, k: int) -> None:
        ts = self.steps[g]
        a[g] = ts[max(0, min(len(ts) - 1, ts.index(a[g]) + k))]

    def nearest(self, g: str, t: str) -> str:
        return min(self.steps[g], key=lambda x: (abs(bpw(x) - bpw(t)), -bpw(x)))

    def repair(self, a: dict[str, str]) -> dict[str, str]:
        """Into the window, one step at a time: lower one of the 8 matrices that lose least error per byte saved,
        or raise one of the 8 that gain most per byte added (random among them, so repairs differ)."""
        a = dict(a)
        for _ in range(20000):
            r, cand = self.ram(a), []
            if self.lo <= r <= self.hi:
                break
            k = -1 if r > self.hi else 1
            for g, ts in self.steps.items():
                i = ts.index(a[g]) + k
                if 0 <= i < len(ts):
                    db = self.bytes[g][ts[i]] - self.bytes[g][a[g]]
                    if k < 0 or r + db <= self.hi:
                        cand.append(((self.err[g][ts[i]] - self.err[g][a[g]]) / max(1, abs(db)), g))
            if not cand:
                break
            self.move(a, self.rng.choice(sorted(cand)[:8])[1], k)
        return a

    def mutate(self, a: dict[str, str]) -> dict[str, str]:
        """1-3 matrices one step, 1-3 matrices two or three steps, or a whole group (role x depth quarter)."""
        a, gs, x = dict(a), list(self.steps), self.rng.random()
        if x < 0.8:
            for g in self.rng.sample(gs, self.rng.randint(1, 3)):
                self.move(a, g, self.rng.choice((-1, 1) if x < 0.5 else (-3, -2, 2, 3)))
        else:
            grp, k = self.rng.choice(sorted(set(self.group.values()))), self.rng.choice((-1, 1))
            for g in gs:
                if self.group[g] == grp:
                    self.move(a, g, k)
        return self.repair(a)

    def crossover(self, p: dict[str, str], q: dict[str, str]) -> dict[str, str]:
        take = {L: self.rng.random() < 0.5 for L in {m.layer for m in self.mats.values()}}
        return self.repair({g: p[g] if take[self.mats[g].layer] else q[g] for g in p})

    def proposals(self, scored: list[tuple[float, dict[str, str]]], k: int) -> list[dict[str, str]]:
        """Fit score ~ sum over groups of (weight x the group's offline error), then the solver's k best genomes."""
        groups = sorted(set(self.group.values()))
        X = np.array([[sum(self.err[g][a[g]] for g in a if self.group[g] == grp) for grp in groups] for _, a in scored])
        s = X.mean(axis=0) + 1e-30
        coef = np.linalg.lstsq(np.c_[X / s, np.ones(len(X))], [v for v, _ in scored], rcond=None)[0][:-1]
        w = dict(zip(groups, np.maximum(coef, 1e-9 * max(1e-12, np.abs(coef).max())) / s))
        pen = {g: {t: float(w[self.group[g]] * e) for t, e in es.items()} for g, es in self.err.items()}
        sols = solve(self.bytes, pen, int(self.lo - self.const), int(self.hi - self.const), k)
        return [a for a in sols if self.fits(a)]

    def initial(self, preset: dict[str, str] | None) -> list[dict[str, str]]:
        """The preset's layout, every uniform type, random mixes, and mutants of the first, all repaired."""
        types = sorted({t for ts in self.steps.values() for t in ts}, key=bpw)
        near = lambda g, t: min(self.steps[g], key=lambda x: abs(bpw(x) - bpw(t)))  # noqa: E731
        pop = ([preset] if preset else []) + [{g: near(g, t) for g in self.steps} for t in types]
        while len(pop) < POP // 2 + 2:
            t = self.rng.choice(types)
            pop.append({g: ts[max(0, min(len(ts) - 1, ts.index(near(g, t)) + self.rng.choice((-2, -1, 0, 0, 1, 2))))]
                        for g, ts in self.steps.items()})
        uniq = {digest(a): a for a in map(self.repair, pop) if self.fits(a)}
        for _ in range(500):
            if len(uniq) >= POP:
                break
            a = self.mutate(self.mutate(pop[0]))
            if self.fits(a):
                uniq.setdefault(digest(a), a)
        return list(uniq.values())


def build(model: str, a: dict[str, str], dst: Path) -> Path:
    """A GGUF with each matrix copied from the type copy of its type; norms and biases from the first copy."""
    import gguf
    if dst.exists():
        return dst
    srcs = {t: gguf.GGUFReader(copy_of(model, t)) for t in set(a.values())}
    by = {t: {x.name: x for x in r.tensors} for t, r in srcs.items()}
    base = srcs[sorted(srcs)[0]]
    part = f"{dst}.{uuid.uuid4().hex[:8]}.part"  # a writer's own part file
    w = gguf.GGUFWriter(part, base.fields["general.architecture"].contents())
    for f in base.fields.values():
        if f.name != "general.architecture" and not f.name.startswith("GGUF."):
            w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    chosen = [by[a[t.name]][t.name] if t.name in a else t for t in base.tensors]
    for t in chosen:
        if t.name in a and t.tensor_type.name != a[t.name]:
            raise SystemExit(f"{t.name}: the {a[t.name]} copy holds {t.tensor_type.name}")
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in chosen:
        w.write_tensor_data(t.data)
    w.close()
    os.rename(part, dst)
    return dst


def label(model: str, a: dict[str, str]) -> str:
    """A genome's results label: its types and the store it is built from (the same genome, the same file)."""
    return f"g:{digest([a, ident(model)])}"


def preset_label(model: str, p: str) -> str:
    return f"preset:{p}:{sha(preset_file(model, p))[:12]}"


def preset_ins(model: str, budget: int) -> str:
    return digest([sha(imatrix(model)), sha(copy_of(model, "BF16")), budget, REF["select"], ms.FIT])


def preset(model: str, budget: int) -> None:
    """Of the presets that fit, the most accurate on all selection questions (ties: lower KL); none fits: the smallest.
    Only it takes the held-out test; its score on the GA's questions is the GA's relative base. The preset files are
    the model's, shared by its budgets: each made once (under a lock), the unchosen removed by run.unpreset."""
    pj, M, top, ins = record("preset", ""), Model(model), budget * (1 + WINDOW[1]) * GIB, preset_ins(model, budget)
    if done(pj, ins):
        return
    fit, over, room = [], [], 8 * (top - M.runtime() - M.fixed) / M.params  # bits per weight that fit
    for p, b in sorted(PRESETS.items(), key=lambda x: -x[1]):
        if b > room + 0.3 and p != "IQ1_S":  # clearly too big (nominal bits per weight); IQ1_S is the fallback
            continue
        f = preset_file(model, p)
        with lock(f"preset-{model}-{p}"):
            if not f.exists():
                q = subprocess.run([str(BIN / "llama-quantize"), "--imatrix", str(imatrix(model)),
                                    str(copy_of(model, "BF16")), f"{f}.part", p, str(THREADS)], capture_output=True, text=True)
                if q.returncode:
                    raise RuntimeError(f"llama-quantize {p} exited {q.returncode}: {q.stderr[-600:]}")
                os.rename(f"{f}.part", f)
        ram = sum(int(t.n_bytes) for t in tensors(f).values()) + M.runtime()
        (fit if ram <= top else over).append((p, f, ram))
        note(f"preset {p}: {ram / GIB:.3f} GiB ({'fits' if ram <= top else 'too big'})")
    fit = fit or [min(over, key=lambda x: x[2])]
    scores = ms.parallel([lambda p=p, f=f: ms.score(model, preset_label(model, p), f, full=True) for p, f, _ in fit])
    res = {p: {"ram_gib": ram / GIB, **r} for (p, _, ram), r in zip(fit, scores)}
    p = max(res, key=lambda x: (round(res[x]["accuracy"], 1), -res[x]["kl"]))
    r, fs = ms.parallel([lambda: ms.test(model, preset_file(model, p), preset_label(model, p)),
                         lambda: ms.score(model, preset_label(model, p), preset_file(model, p))])
    t = {"ram_gib": res[p]["ram_gib"], "accuracy": ms.acc(r["correct"]), "kl": r["kl"], "ppl": r["ppl"]}
    put(pj, json.dumps({"inputs": ins, "preset": p, "label": preset_label(model, p), "over_budget": t["ram_gib"]
                        * GIB > top, "test": t, "score_fit": fs["score"], "tried": res}, indent=1))
    note(f"preset {p}: held-out accuracy {t['accuracy']:.2f}, KL {t['kl']:.4f} (selection accuracy "
         f"{res[p]['accuracy']:.2f}, best of {len(res)})")


def ga_ins(model: str, budget: int) -> str:
    pr = json.loads(record("preset", "").read_text())
    return digest([ident(model), pr["inputs"], budget, POP, MAX_EVALS, MIN_EVALS, PATIENCE, TOP])


def ga(model: str, budget: int) -> None:
    """Steady state, as many genomes at once as the claim holds (none twice, none while it is in flight); each round
    SOLVED solver proposals from a fit of every score so far. Stops after PATIENCE genomes without an improvement
    beyond 1 SE. Writes the TOP best."""
    lst, pr, ins = record("list"), json.loads(record("preset", "").read_text()), ga_ins(model, budget)
    if done(lst, ins):
        return
    tmp = path(model, f"tmp-b{budget}", "x").parent  # this run's own genome files; any here are a killed run's
    shutil.rmtree(tmp)
    tmp.mkdir()
    G, base = Genes(model, budget), pr["score_fit"]  # the preset's score on the GA questions: relative scores
    have = {n: t.tensor_type.name for n, t in tensors(preset_file(model, pr["preset"])).items()}
    queue = G.initial({g: have[g] if have[g] in ts else G.nearest(g, have[g]) for g, ts in G.steps.items()})
    pop, seen, scored, best, last_gain, n, dropped, top, se, why = [], set(), [], float("inf"), 0, 0, 0, None, 0.0, ""
    note(f"GA: {len(queue)} starting genomes")

    def fresh() -> dict[str, str] | None:
        """The next genome never scored nor in flight: from the queue, else a child; None if 500 tries find none."""
        while queue:
            if digest(a := queue.pop(0)) not in seen:
                return a
        for _ in range(500):
            if len(pop) < 2 or G.rng.random() < 0.3:
                a = G.mutate(G.rng.choice(pop[:ELITE] or pop)[2])
            else:
                p, q = (min(G.rng.sample(pop, 2))[2] for _ in range(2))
                a = G.crossover(p, q) if G.rng.random() < 0.5 else G.mutate(G.crossover(p, q))
            if digest(a) not in seen and G.fits(a):
                return a
        return None

    def one(a: dict[str, str], cut: float | None) -> dict:
        f = tmp / f"{digest(a)}.gguf"
        try:
            return ms.score(model, label(model, a), lambda: build(model, a, f), cut=cut)
        finally:
            f.unlink(missing_ok=True)
    g, m = ms.need(budget * GIB)  # a genome's file is about the budget, its runs one booking: as many as fit the claim
    inflight = max(1, min(16, hold.total[0] // g, hold.total[1] // m))
    with ThreadPoolExecutor(inflight) as ex:
        running = {}

        def submit(a: dict[str, str] | None) -> None:
            if a is None:  # nothing new to score now: the GA ends once those in flight are done
                return
            seen.add(digest(a))
            cut = pop[-1][0] + DROP_MARGIN if len(pop) >= POP else None  # a full population: weak ones dropped early
            running[ex.submit(one, a, cut)] = a
        while queue and len(running) < inflight:
            submit(fresh())
        while running:
            fut = next(as_completed(running))
            a, n = running.pop(fut), n + 1
            r = ms.against(fut.result(), base)
            if r.get("dropped"):
                dropped += 1
            else:
                scored.append((r["score"], a))
                pop = sorted(pop + [(r["score"], digest(a), a)])[:POP]
            if r["score"] < best and not r.get("dropped"):
                if top:  # an improvement only beyond 1 SE of the difference (flips paired on the GA questions)
                    se = ms.score_se(ms.correct(model, label(model, a), ms.FIT), ms.correct(model, top[0], ms.FIT),
                                     ms.correct(model, "original", ms.FIT), r["kl_se"], top[1], ms.KL_WEIGHT)
                last_gain = n if best - r["score"] > se else last_gain
                best, top = r["score"], (label(model, a), r["kl_se"])
                note(f"GA: genome {n}: new best score {best:.2f} (flips {r['flips']:.2f}%, KL {r['kl']:.4f}, "
                     f"accuracy {r['accuracy']:.2f}, RAM {G.ram(a) / GIB:.3f} GiB); relative {r['relative']:.1f}")
            elif n % 10 == 0:
                note(f"GA: {n} genomes, best score {best:.2f} ({dropped} dropped after a first look)")
            if n + len(running) < MAX_EVALS and (n < MIN_EVALS or n - last_gain < PATIENCE):
                if not queue and len(scored) >= 8:
                    picks = [p for p in G.proposals(scored, SOLVED) if digest(p) not in seen]
                    queue += picks + [G.mutate(p) for p in picks[:2]]
                submit(fresh())
            elif not why:  # why it stopped, as decided now (a genome still running may improve on best)
                why = "plateau" if n - last_gain >= PATIENCE else "max"
    put(lst, json.dumps({"inputs": ins, "finalists": {
        f"{short(model)}:ga{i}": {"assignment": a, "score": s, "relative": 100 * s / base, "ram_gib": G.ram(a) / GIB}
        for i, (s, _, a) in enumerate(pop[:TOP], 1)}}, indent=1))
    why = (f"{PATIENCE} genomes without an improvement beyond 1 SE (best {best:.2f}, SE {se:.2f})" if why == "plateau"
           else f"{MAX_EVALS} genomes reached, best score {best:.2f}" if why == "max"
           else f"no genome not already scored found in 500 tries, best score {best:.2f}")
    note(f"GA done: {n} genomes ({dropped} dropped after a first look); {why}")
