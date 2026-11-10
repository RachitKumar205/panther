"""Phase A gate, step A2 (docs/PIPELINE.md): BF16 reference logits, noise floors, chunk-count study.

  python scripts/server/gate_a2.py prep       # per-chunk texts + the 7 study GGUFs (CPU container)
  python scripts/server/gate_a2.py analyze    # floors, per-chunk KL, bootstrap, reference manifest

scripts/server/gate_a2.sh runs the llama.cpp steps in between. Outputs:
  $PANTHER_DATA/reference/<model>/{sensitivity,selection}.kld + manifest.json   (kept for the search)
  $PANTHER_DATA/gate/a2/                                                        (logs, study.json)
  $PANTHER_DATA/gate/a2/study/                                                  (temporary, deleted after)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from panther.costs.formats import candidate_formats
from panther.costs.ram import uniform
from panther.gguftools.assemble import TensorStore, assemble
from panther.gguftools.verify import verify
from panther.models.inventory import REPO_ROOT, build_inventory, load_geometry

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
STORE = DATA / "store" / MODEL
REF = DATA / "reference" / MODEL
OUT = DATA / "gate" / "a2"
STUDY = OUT / "study"
CTX = 2048
MARGIN = 128
N_CHUNKS = 48
STUDY_GENES = ["embd_head", "attn.0", "mlp.0", "attn.14", "mlp.14", "mlp.27"]
BASE_FMT, LOW_FMT = "Q6_K", "Q2_K"
NS = [6, 12, 24, 48]
N_BOOT = 10_000
SEED = 20261110


def study_assignments(inv) -> dict[str, dict[str, str]]:
    base = uniform(inv, BASE_FMT)
    out = {"base": base}
    for g in STUDY_GENES:
        out[g] = {**base, g: LOW_FMT}
    return out


def cmd_prep() -> int:
    from transformers import AutoTokenizer

    (STUDY / "chunks").mkdir(parents=True, exist_ok=True)
    (STUDY / "models").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(DATA / "hf" / MODEL)
    ids = tok((DATA / "splits/text/sensitivity.txt").read_text(), add_special_tokens=False)["input_ids"]
    if len(ids) < N_CHUNKS * CTX:
        raise SystemExit(f"sensitivity text too short: {len(ids)} tokens")
    # llama-perplexity needs >= 2 x context tokens even with --chunks 1; only the first chunk is scored.
    # The padding after the chunk wraps to the start of the text for the last chunks.
    wrapped = ids + ids
    for k in range(N_CHUNKS):
        piece = wrapped[k * CTX:k * CTX + 2 * CTX + MARGIN]
        (STUDY / "chunks" / f"chunk-{k:02d}.txt").write_text(tok.decode(piece))
    print(f"wrote {N_CHUNKS} chunk texts")

    inv = build_inventory(load_geometry(MODEL))
    store = TensorStore({f: STORE / f"{MODEL}-{f}.gguf" for f in candidate_formats()})
    for name, a in study_assignments(inv).items():
        dst = STUDY / "models" / f"{name}.gguf"
        if not dst.exists():
            tmp = dst.with_suffix(".gguf.part")
            assemble(inv, a, store, tmp, base_format=BASE_FMT)
            tmp.rename(dst)
        r = verify(inv, a, dst)
        print(f"{name:10s} {'ok' if r.ok else 'MISMATCH'}")
        if not r.ok:
            return 1
    return 0


SUMMARY = {
    "ppl_ratio": re.compile(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)"),
    "ppl_q": re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)"),
    "kl_mean": re.compile(r"Mean\s+KLD:\s*(-?[0-9.]+)\s*±\s*([0-9.]+)"),
    "kl_p99": re.compile(r"99\.0%\s+KLD:\s*(-?[0-9.]+)"),
    "kl_max": re.compile(r"Maximum KLD:\s*(-?[0-9.]+)"),
    "same_top": re.compile(r"Same top p:\s*([0-9.]+)\s*±\s*([0-9.]+)"),
}
FINAL_PPL = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


def parse_kl(log: Path) -> dict:
    text = log.read_text(errors="replace")
    out = {}
    for k, rx in SUMMARY.items():
        m = rx.search(text)
        if m is None:
            raise ValueError(f"{log}: no '{k}'")
        out[k] = float(m.group(1))
        if k in ("kl_mean", "same_top"):
            out[k + "_se"] = float(m.group(2))
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap(per_chunk: dict[str, np.ndarray]) -> dict:
    """Paired bootstrap over chunks of each gene's KL increase over the base."""
    rng = np.random.default_rng(SEED)
    inc = {g: per_chunk[g] - per_chunk["base"] for g in STUDY_GENES}
    full = {g: float(v.mean()) for g, v in inc.items()}
    full_rank = sorted(STUDY_GENES, key=lambda g: full[g], reverse=True)
    res = {"increase_at_48": full, "ranking_at_48": full_rank, "by_n": {}}
    mat = np.stack([inc[g] for g in STUDY_GENES])  # genes x chunks
    for n in NS:
        idx = rng.integers(0, N_CHUNKS, size=(N_BOOT, n))
        means = mat[:, idx].mean(axis=2)  # genes x boot
        rel_se = {g: float(means[i].std(ddof=1) / abs(full[g])) for i, g in enumerate(STUDY_GENES)}
        order = np.argsort(-means, axis=0)  # ranks per bootstrap sample
        target = np.array([STUDY_GENES.index(g) for g in full_rank])[:, None]
        rank_match = float(np.mean(np.all(order == target, axis=0)))
        ok = max(rel_se.values()) <= 0.05 and rank_match >= 0.95
        res["by_n"][n] = {"rel_se": rel_se, "max_rel_se": max(rel_se.values()),
                          "ranking_match": rank_match, "meets_rule": ok}
    passing = [n for n in NS if res["by_n"][n]["meets_rule"]]
    res["recommended_n"] = passing[0] if passing else 48
    res["rule_met"] = bool(passing)
    return res


def cmd_analyze() -> int:
    report: dict = {"model": MODEL}

    # Reference files + manifest.
    refs = {}
    for split in ("sensitivity", "selection"):
        kld = REF / f"{split}.kld"
        log = OUT / f"ref-{split}.log"
        m = FINAL_PPL.search(log.read_text(errors="replace"))
        text = DATA / "splits" / "text" / f"{split}.txt"
        refs[split] = {"file": kld.name, "bytes": kld.stat().st_size, "sha256": sha256(kld),
                       "text_sha256": sha256(text), "context": CTX, "batch": CTX,
                       "chunks": int(re.search(r"over (\d+) chunks", log.read_text()).group(1)),
                       "bf16_ppl": float(m.group(1)) if m else None}
        print(f"reference {split}: {refs[split]['chunks']} chunks, {refs[split]['bytes'] / 1e9:.2f} GB, "
              f"BF16 PPL {refs[split]['bf16_ppl']}")
    store_manifest = json.loads((STORE / "manifest.json").read_text())
    manifest = {"model": MODEL, "built_from": f"{MODEL}-BF16.gguf",
                "bf16_sha256": store_manifest["bf16"]["sha256"],
                "llama_cpp_commit": store_manifest["llama_cpp_commit"],
                "image_id": os.environ.get("PANTHER_IMAGE_ID", "unknown"), "build": "cuda",
                "kl_runs_must_use": {"-c": CTX, "-b": CTX}, "references": refs}
    (REF / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report["references"] = refs

    # Floors on the full sensitivity reference.
    floors = {"storage_bf16_self": parse_kl(OUT / "floor-self.log"),
              "build_bf16_cpu": parse_kl(OUT / "floor-cpu.log")}
    report["floors"] = floors

    # Per-chunk KL for the 7 study files.
    per_chunk = {}
    for name in ["base", *STUDY_GENES]:
        vals = [parse_kl(STUDY / "logs" / f"{name}-{k:02d}.log")["kl_mean"] for k in range(N_CHUNKS)]
        per_chunk[name] = np.array(vals)
    contiguous = parse_kl(OUT / "study-base-contiguous.log")["kl_mean"]
    per_chunk_mean = float(per_chunk["base"].mean())
    consistency = abs(per_chunk_mean - contiguous) / contiguous
    report["study"] = {
        "base": f"uniform {BASE_FMT}", "lowered_to": LOW_FMT,
        "kl_mean_per_file": {k: float(v.mean()) for k, v in per_chunk.items()},
        "per_chunk_kl": {k: [round(x, 6) for x in v] for k, v in per_chunk.items()},
        "consistency": {"per_chunk_mean": per_chunk_mean, "contiguous_run": contiguous,
                        "rel_diff": consistency, "ok": consistency <= 0.02},
        "bootstrap": bootstrap(per_chunk),
    }

    probe = {"storage": 0.0, "build": 0.00005}
    checks = {
        "references_written": all(r["bytes"] > 0 for r in refs.values()),
        "floors_within_2x_probe": floors["storage_bf16_self"]["kl_mean"] <= 1e-5
        and floors["build_bf16_cpu"]["kl_mean"] <= 2 * probe["build"],
        "per_chunk_consistent": consistency <= 0.02,
        "recommendation_made": True,
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    (OUT / "study.json").write_text(json.dumps(report, indent=2))

    b = report["study"]["bootstrap"]
    print("floors: storage KL", floors["storage_bf16_self"]["kl_mean"],
          "| build KL", floors["build_bf16_cpu"]["kl_mean"], "top-1", floors["build_bf16_cpu"]["same_top"])
    print("base (uniform Q6_K) KL", round(per_chunk_mean, 6), "contiguous", contiguous,
          f"rel diff {consistency:.2%}")
    print("increase at 48:", {g: round(v, 6) for g, v in b["increase_at_48"].items()})
    print("ranking at 48:", b["ranking_at_48"])
    for n, v in b["by_n"].items():
        print(f"N={n:2d} max rel SE {v['max_rel_se']:.3f} ranking match {v['ranking_match']:.3f} "
              f"{'MEETS' if v['meets_rule'] else 'fails'}")
    print("recommended N:", b["recommended_n"], "(rule met)" if b["rule_met"] else "(rule NOT met: using 48)")
    for k, ok in checks.items():
        print(f"{k:24s} {'PASS' if ok else 'FAIL'}")
    print("A2:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    cmds = {"prep": cmd_prep, "analyze": cmd_analyze}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]]())
