"""Phase 0 of the rewrite (TODO.md, D45): two gates before building anything new. Runs in the GPU container.

  python scripts/server/phase0.py presets  # 0.1: build stock llama.cpp presets of 14B and the head/V baseline
  python scripts/server/phase0.py score    # 0.1: score the ones that fit 7.0 GiB like the PoC winner
  python scripts/server/phase0.py error    # 0.3: importance-weighted dequantization error per tensor and format
  python scripts/server/phase0.py fit      # 0.3: does s_block x error predict the measured 14B penalty table?

Same RAM model as the PoC (f16 KV cache, 4,096 context, 0.40 GiB overhead) so the comparison with the PoC winner
is like for like. Everything is resumable. Outputs in $PANTHER_DATA/phase0/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poc  # noqa: E402  (reuses records, KL/task runners, test-subset evaluation)
from panther.costs.formats import tensor_bytes  # noqa: E402
from panther.costs.ram import GIB, fixed_bytes  # noqa: E402

MODEL = "Qwen2.5-14B-Instruct"
DATA, LLAMA, THREADS = poc.DATA, poc.LLAMA, poc.THREADS
OUT = DATA / "phase0"
STORE = DATA / "store" / MODEL
IMATRIX = STORE / f"{MODEL}.imatrix.gguf"
BF16 = STORE / f"{MODEL}-BF16.gguf"
R_MAX, CTX, OVERHEAD = poc.R_MAX, poc.CTX, poc.OVERHEAD
PRESETS = ["IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "Q3_K_S", "IQ2_M"]  # stock mixes near 7 GiB for 14B
CPUS = int(os.environ.get("PANTHER_CPUS", "20"))


def out(*parts: str) -> Path:
    p = OUT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def file_ram(path: Path) -> dict:
    """Predicted RAM of a GGUF from its real tensor bytes (same KV/overhead terms as the PoC)."""
    import gguf
    r = gguf.GGUFReader(path)
    w = sum(int(t.n_bytes) for t in r.tensors)
    inv, *_ = poc.setup(MODEL)
    kv = inv.geometry.kv_cache_bytes(CTX)
    types = {}
    for t in r.tensors:
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
    return {"weights_bytes": w, "ram_gib": (w + kv + int(OVERHEAD * GIB)) / GIB, "types": types}


# --- 0.1 presets ---------------------------------------------------------------------------------------------

def quantize(dst: Path, ftype: str, extra: list[str]) -> None:
    if dst.exists():
        return
    cmd = [str(LLAMA / "build-cuda/bin/llama-quantize"), "--imatrix", str(IMATRIX), *extra, str(BF16),
           str(dst) + ".part", ftype, str(THREADS)]
    log = dst.with_suffix(".log")
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not Path(str(dst) + ".part").exists():
        raise SystemExit(f"llama-quantize {ftype} failed; see {log}")
    Path(str(dst) + ".part").rename(dst)


def headv_layers(n_layers: int) -> list[int]:
    """Layer order for padding: alternate from both ends (first and last layers matter most)."""
    order, lo, hi = [], 0, n_layers - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo, hi = lo + 1, hi - 1
    return order


def headv_overrides(n_pad: int) -> list[str]:
    inv, *_ = poc.setup(MODEL)
    args = ["--output-tensor-type", "q6_k", "--tensor-type", "attn_v=q6_k"]
    for L in headv_layers(inv.geometry.n_layers)[:n_pad]:
        args += ["--tensor-type", rf"blk\.{L}\.(attn_q|attn_k|attn_output|ffn_gate|ffn_up|ffn_down)=iq3_s"]
    return args


def headv_bytes(n_pad: int) -> int:
    """Weight bytes of the head/V baseline with n_pad layers raised to IQ3_S (exact, from gguf-py sizes)."""
    inv, *_ = poc.setup(MODEL)
    pad = set(headv_layers(inv.geometry.n_layers)[:n_pad])
    total = fixed_bytes(inv)
    for g in inv.genes:
        for t in g.tensors:
            if g.kind == "embd":
                f = "Q2_K"  # llama-quantize's own choice for the embedding in IQ3_XXS mode is read back below
            elif g.kind == "head" or t.name.endswith("attn_v.weight"):
                f = "Q6_K"
            elif g.layer in pad:
                f = "IQ3_S"
            else:
                f = "IQ3_XXS"
            total += tensor_bytes(t, f)
    return total


def cmd_presets() -> int:
    rows = {}
    for p in PRESETS:
        dst = out("presets", f"{MODEL}-{p}.gguf")
        quantize(dst, p, [])
        rows[p] = file_ram(dst)
        print(f"preset {p:8s} RAM {rows[p]['ram_gib']:.3f} GiB  types {rows[p]['types']}", flush=True)
    # Head/V baseline: IQ3_XXS with output and every attn_v at Q6_K, padded with IQ3_S layers up to the budget.
    inv, *_ = poc.setup(MODEL)
    probe = out("presets", f"{MODEL}-headv-probe.gguf")
    quantize(probe, "IQ3_XXS", headv_overrides(0))
    embd_extra = file_ram(probe)["weights_bytes"] - headv_bytes(0)  # embedding as llama-quantize chose it
    budget = int(R_MAX * GIB) - inv.geometry.kv_cache_bytes(CTX) - int(OVERHEAD * GIB)
    n = 0
    while n < inv.geometry.n_layers and headv_bytes(n + 1) + embd_extra <= budget:
        n += 1
    probe.unlink()
    dst = out("presets", f"{MODEL}-headv-pad{n}.gguf")
    quantize(dst, "IQ3_XXS", headv_overrides(n))
    rows[f"headv-pad{n}"] = file_ram(dst)
    print(f"headv-pad{n} RAM {rows[f'headv-pad{n}']['ram_gib']:.3f} GiB  types {rows[f'headv-pad{n}']['types']}")
    out("presets.json").write_text(json.dumps(rows, indent=1))
    return 0


def evaluate_file(path: Path, name: str):
    """Selection KL + the PoC verdict's test-subset evaluation for a ready GGUF (not an assignment)."""
    from panther.stats.decision import Evaluation
    label = f"phase0:{name}"
    recs = poc.records(MODEL)
    if not poc.find(recs, label, reference="selection"):
        poc.kl(MODEL, path, "selection", label)
    for t in poc.SUBSET:
        if not poc.find(poc.records(MODEL), label, task=t, split=poc.V_SPLIT):
            if poc.evaluator.main(["task", "--model", MODEL, "--gguf", str(path), "--build", "cuda", "--task", t,
                                   "--split", poc.V_SPLIT, "--label", label, "--threads", str(THREADS)]) != 0:
                raise SystemExit(f"{label} {t} failed")
    if not poc.find(poc.records(MODEL), label, text=poc.V_TEXT):
        if poc.evaluator.main(["ppl", "--model", MODEL, "--gguf", str(path), "--build", "cuda", "--text",
                               poc.V_TEXT, "--label", label, "--threads", str(THREADS)]) != 0:
            raise SystemExit(f"{label} ppl failed")
    g = poc.run_gsm8k(MODEL, path, label)
    recs = poc.records(MODEL)
    correct = {t: np.array(poc.find(recs, label, task=t, split=poc.V_SPLIT)["per_item"]["correct"], dtype=np.int8)
               for t in poc.SUBSET}
    correct["gsm8k"] = np.array(g["correct"], dtype=np.int8)
    p = poc.find(recs, label, text=poc.V_TEXT)["per_chunk"]
    nll = np.array(p["mean_nll"]) * p["tokens"]
    ram = int(file_ram(path)["ram_gib"] * GIB)
    ev = Evaluation(f"14B:{name}", correct, nll, np.full(len(nll), p["tokens"]), ram, 0.0)
    return ev, poc.find(recs, label, reference="selection")["metrics"]


def cmd_score() -> int:
    from panther.stats.decision import compare
    rows = json.loads(out("presets.json").read_text())
    fits = {n: r for n, r in rows.items() if r["ram_gib"] <= R_MAX}
    print("fit 7.0 GiB:", {n: round(r["ram_gib"], 3) for n, r in fits.items()},
          "| too big:", sorted(set(rows) - set(fits)), flush=True)
    inv, menu, store, _ = poc.setup(MODEL)
    sl = json.loads((poc.out_dir(MODEL) / "shortlist.json").read_text())
    win = poc.evaluate_config(MODEL, sl["winner"], sl["rows"][sl["winner"]]["assignment"], store, inv, menu["base"])
    res = {"winner": {"ram_gib": win.peak_ram_bytes / GIB, "kl_selection": sl["rows"][sl["winner"]]["kl_selection_cuda"],
                      "mean_accuracy_pct": 100 * win.mean_accuracy(), "ppl": win.perplexity()}}
    for n in fits:
        path = out("presets", f"{MODEL}-{n}.gguf")
        ev, k = evaluate_file(path, n)
        c = compare(win, ev)
        res[n] = {"ram_gib": fits[n]["ram_gib"], "kl_selection": k["kl_mean"], "top1_pct": k["top1_pct"],
                  "mean_accuracy_pct": 100 * ev.mean_accuracy(), "ppl": ev.perplexity(),
                  "per_task_pct": {t: 100 * float(v.mean()) for t, v in ev.task_correct.items()},
                  "vs_winner": {"winner": c.winner, "decided_by": c.decided_by, "acc_diff_pts": 100 * c.acc_diff,
                                "acc_ci_pts": [100 * c.acc_ci[0], 100 * c.acc_ci[1]], "ppl_ratio": c.ppl_ratio}}
        r = res[n]
        print(f"{n:12s} RAM {r['ram_gib']:.3f} | KL sel {r['kl_selection']:.4f} | acc {r['mean_accuracy_pct']:.2f} "
              f"| PPL {r['ppl']:.3f} | winner-minus-preset {r['vs_winner']['acc_diff_pts']:+.2f} pts "
              f"CI [{r['vs_winner']['acc_ci_pts'][0]:+.2f}, {r['vs_winner']['acc_ci_pts'][1]:+.2f}] "
              f"-> {r['vs_winner']['winner']} by {r['vs_winner']['decided_by']}", flush=True)
    w = res["winner"]
    print(f"PoC winner   RAM {w['ram_gib']:.3f} | KL sel {w['kl_selection']:.4f} | acc {w['mean_accuracy_pct']:.2f} | PPL {w['ppl']:.3f}")
    out("score.json").write_text(json.dumps(res, indent=1))
    return 0


# --- 0.3 offline error ---------------------------------------------------------------------------------------

def importance() -> dict[str, np.ndarray]:
    """Per-input-column mean squared activation from the importance matrix (GGUF format)."""
    import gguf
    r = gguf.GGUFReader(IMATRIX)
    t = {x.name: np.asarray(x.data, dtype=np.float64) for x in r.tensors}
    imp = {}
    for name, v in t.items():
        if name.endswith(".in_sum2"):
            base = name[: -len(".in_sum2")]
            cnt = t.get(base + ".counts")
            n = float(np.asarray(cnt).reshape(-1)[0]) if cnt is not None else 1.0
            imp[base] = v.reshape(-1) / max(n, 1.0)
    return imp


def _error_one(job: tuple[str, list[str]]) -> tuple[str, dict[str, float]]:
    name, formats = job
    imp = _IMP.get(name)
    w = _tensor(BF16, name)
    if imp is None:
        imp = np.ones(w.shape[1])
    res = {}
    for f in formats:
        q = _tensor(STORE / f"{MODEL}-{f}.gguf", name)
        res[f] = float(((w - q) ** 2).sum(axis=0) @ imp[: w.shape[1]])
    return name, res


_READERS: dict[Path, object] = {}
_IMP: dict[str, np.ndarray] = {}


def _tensor(path: Path, name: str) -> np.ndarray:
    import gguf
    from gguf.quants import dequantize
    if path not in _READERS:
        _READERS[path] = {t.name: t for t in gguf.GGUFReader(path).tensors}
    t = _READERS[path][name]
    a = dequantize(t.data, t.tensor_type).astype(np.float32)
    return a.reshape(-1, int(t.shape[0]))  # GGUF shape is (cols, rows)


def _init_worker() -> None:
    global _IMP
    _IMP = importance()


def cmd_error() -> int:
    inv, menu, store, allowed = poc.setup(MODEL)
    dst = out(MODEL, "error.json")
    done = json.loads(dst.read_text()) if dst.exists() else {}
    jobs = []
    for g in inv.genes:
        for t in g.tensors:
            if t.name not in done:
                jobs.append((t.name, menu["formats"]))
    print(f"{len(jobs)} tensors x {len(menu['formats'])} formats", flush=True)
    with ProcessPoolExecutor(min(CPUS, 8), initializer=_init_worker) as ex:
        for i, (name, res) in enumerate(ex.map(_error_one, jobs), 1):
            done[name] = res
            if i % 20 == 0 or i == len(jobs):
                dst.write_text(json.dumps(done))
                print(f"  {i}/{len(jobs)} {name}", flush=True)
    return 0


# --- 0.3 fit -------------------------------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def predict(pen: dict, E: dict, base: str, probe: str) -> dict:
    """penalty(g, f) = s_g * (E(g, f) - E(g, base)), s_g from the one measured probe format."""
    pred = {}
    for g, row in pen.items():
        if probe not in row:
            pred[g] = dict(row)  # gene cannot use the probe (e.g. head with IQ2_XS): keep measured values
            continue
        d = E[g][probe] - E[g][base]
        s = row[probe] / d if d else 0.0
        pred[g] = {f: s * (E[g][f] - E[g][base]) for f in row}
    return pred


def cmd_fit() -> int:
    from panther.costs.ram import estimate_ram
    from panther.search.problem import Problem
    from panther.search.solver import solve_top_k
    inv, menu, store, allowed = poc.setup(MODEL)
    base = menu["base"]
    pen = json.loads((poc.out_dir(MODEL) / "penalties.json").read_text())["penalty"]
    err = json.loads(out(MODEL, "error.json").read_text())
    E = {g.gene_id: {f: sum(err[t.name][f] for t in g.tensors) for f in allowed[g.gene_id]} for g in inv.genes}
    report = {"probes": {}}
    for probe in [f for f in menu["formats"] if f != base]:
        pred = predict(pen, E, base, probe)
        m, p = [], []
        for g in pen:
            for f in pen[g]:
                if f not in (base, probe) and probe in pen[g]:
                    m.append(pen[g][f])
                    p.append(pred[g][f])
        m, p = np.array(m), np.array(p)
        rel = np.abs(p - m) / np.maximum(np.abs(m), 1e-9)
        report["probes"][probe] = {"n": len(m), "spearman": spearman(p, m), "median_rel_err": float(np.median(rel)),
                                   "p90_rel_err": float(np.quantile(rel, 0.9))}
        print(f"probe {probe:8s}: spearman {report['probes'][probe]['spearman']:.3f}, median rel err "
              f"{report['probes'][probe]['median_rel_err']:.1%}, p90 {report['probes'][probe]['p90_rel_err']:.1%} "
              f"(n={len(m)})", flush=True)
    best = max(report["probes"], key=lambda k: report["probes"][k]["spearman"])
    report["best_probe"] = best
    pred = predict(pen, E, base, best)
    sol_m = solve_top_k(Problem(inv, allowed, pen, R_MAX, CTX, OVERHEAD), k=1)[0].assignment
    sol_p = solve_top_k(Problem(inv, allowed, pred, R_MAX, CTX, OVERHEAD), k=1)[0].assignment
    same = sum(sol_m[g] == sol_p[g] for g in sol_m)
    report["solutions"] = {"measured": sol_m, "modelled": sol_p, "genes_equal": same, "genes": len(sol_m)}
    print(f"7 GiB solution, modelled vs measured penalties: {same}/{len(sol_m)} genes equal", flush=True)
    sl = json.loads((poc.out_dir(MODEL) / "shortlist.json").read_text())
    kls = {}
    for name, a in (("measured", sol_m), ("modelled", sol_p)):
        hit = next((r for r in sl["rows"].values() if r["assignment"] == a), None)
        if hit:
            kls[name] = hit["kl_selection_cuda"]
            continue
        label = f"phase0:fit:{name}:{poc.assignment_digest(a)[:12]}"
        if not poc.find(poc.records(MODEL), label, reference="selection"):
            path = poc.build_file(inv, store, a, base, MODEL)
            poc.kl(MODEL, path, "selection", label)
            path.unlink()
        kls[name] = poc.find(poc.records(MODEL), label, reference="selection")["metrics"]["kl_mean"]
    report["selection_kl"] = kls
    report["kl_gap_pct"] = 100 * (kls["modelled"] - kls["measured"]) / kls["measured"]
    print(f"selection KL: measured-penalty solution {kls['measured']:.5f}, modelled {kls['modelled']:.5f} "
          f"({report['kl_gap_pct']:+.1f}%)", flush=True)
    for name, a in (("measured", sol_m), ("modelled", sol_p)):
        report["solutions"][name + "_ram_gib"] = estimate_ram(inv, a, CTX, OVERHEAD).total_gib
    out(MODEL, "fit.json").write_text(json.dumps(report, indent=1))
    return 0


def main() -> int:
    cmd = sys.argv[1]
    return {"presets": cmd_presets, "score": cmd_score, "error": cmd_error, "fit": cmd_fit}[cmd]()


if __name__ == "__main__":
    sys.exit(main())
