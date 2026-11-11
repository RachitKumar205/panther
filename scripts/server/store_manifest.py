"""Write $PANTHER_DATA/store/<model>/manifest.json: everything needed to know how the tensor store was made.

  python scripts/server/store_manifest.py <model> <calibration.txt> [--timings FILE] [--note TEXT]

Records: model repo@revision, llama.cpp commit, image id, calibration file and splits manifest hashes,
imatrix hash, and per format: file sha256, bytes, embedding placeholder (if any), read-back verification.
Timings come from the `step seconds` lines build_tensor_store.sh appends to <store>/.timings.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

from panther.gguf_files import verify
from panther.model import REPO_ROOT, build_inventory, candidate_formats, load_geometry, source_assignment

DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("calibration", type=Path)
    ap.add_argument("--timings", type=Path)
    ap.add_argument("--note", default="")
    ap.add_argument("--formats", nargs="+", help="only these formats (default: all candidates)")
    a = ap.parse_args()

    store = DATA / "store" / a.model
    inv = build_inventory(load_geometry(a.model))
    embd = next(g.gene_id for g in inv.genes if g.kind in ("embd", "embd_head"))
    llama = Path(os.environ.get("PANTHER_LLAMA", "/opt/llama.cpp"))
    splits_manifest = a.calibration.parent / "manifest.json"

    formats = {}
    for f in a.formats or candidate_formats():
        p = store / f"{a.model}-{f}.gguf"
        sa = source_assignment(inv, f)
        r = verify(inv, sa, p)
        formats[f] = {
            "file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
            "verified": r.ok, "mismatches": len(r.mismatches), "missing": len(r.missing),
            "embedding_placeholder": sa[embd] if sa[embd] != f else None,
        }
        print(f"{f:8s} {'ok' if r.ok else 'MISMATCH'}", flush=True)

    timings = {}
    if a.timings and a.timings.exists():
        for line in a.timings.read_text().splitlines():
            step, secs = line.split()
            timings[step] = float(secs)

    bf16 = store / f"{a.model}-BF16.gguf"
    imat = store / f"{a.model}.imatrix.gguf"
    commit = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(llama), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    if len(commit) != 40:
        raise SystemExit(f"could not read the llama.cpp commit from {llama}")
    manifest = {
        "model": a.model,
        "weights": (DATA / "hf" / a.model / ".panther-revision").read_text().strip(),
        "llama_cpp_commit": commit,
        "image_id": os.environ.get("PANTHER_IMAGE_ID", "unknown"),
        "calibration": {"file": str(a.calibration), "sha256": sha256(a.calibration),
                        "splits_manifest_sha256": sha256(splits_manifest) if splits_manifest.exists() else None},
        "imatrix": {"file": imat.name, "sha256": sha256(imat), "context": 512},
        "bf16": {"file": bf16.name, "bytes": bf16.stat().st_size, "sha256": sha256(bf16)},
        "quantize_flags": "--pure --imatrix <imatrix> --token-embedding-type <t> --output-tensor-type <t>",
        "formats": formats,
        "timings_s": timings,
        "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": a.note,
    }
    (store / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {store / 'manifest.json'}")
    return 0 if all(v["verified"] for v in formats.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
