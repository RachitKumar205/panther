"""A model's store: the 16-bit original, its importance matrix (decision 4), the model once in every type (rounded by
panther.rounding); backed up once to Hugging Face, restored from it (never rebuilt), removed when no run needs it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

from panther.data import WEB
from panther.model import BIN, CFG, DATA, GIB, LLAMA, MODELS, RW, THREADS, Model, bpw, digest, hold, lock, note, put, sha

STORE = DATA / "store"
FTYPE = {"Q3_K": "Q3_K_S", "Q4_K": "Q4_K_S", "Q5_K": "Q5_K_S", "IQ2_S": "IQ2_M"}  # --pure: one type everywhere
MARGIN = 20 * GIB  # disk left free by downloads and type copies


def copy_of(model: str, t: str) -> Path:
    return STORE / model / f"{model}-{t}.gguf"


def imatrix(model: str) -> Path:
    return STORE / model / f"{model}.imatrix.gguf"


def imatrix_out(model: str) -> Path:  # with the output layer's data (a tied word table's under its name): the error's
    return STORE / model / f"{model}.imatrix-output.gguf"


def preset_file(model: str, p: str) -> Path:
    return STORE / model / f"preset-{p}.gguf"


def repo(model: str) -> str:
    return os.environ["PANTHER_HF_REPO"].split("/")[0] + f"/panther-store-{model}"


def rounded(model: str) -> list[str]:
    """The types whose copy has error-compensated codes."""
    f = STORE / model / "rounded.json"
    return json.loads(f.read_text()) if f.exists() else []


def stored(model: str) -> bool:
    types = CFG["formats"]
    return copy_of(model, "BF16").exists() and all(copy_of(model, t).exists() for t in types) and \
        set(types) <= set(rounded(model))


def copies(model: str) -> list[Path]:
    return [f for f in (STORE / model).glob(f"{model}-*.gguf") if not f.name.endswith("-BF16.gguf")]


def removed(model: str) -> bool:
    """The type copies were removed after their backup (clean): steps that are done do not need them back."""
    return (STORE / model / "ident").exists() and not copies(model)


@cache
def ident(model: str) -> str:
    """The store's identity: what every genome built from it is made of (labels of genome results carry it); recorded
    when the copies are removed, so a finished contest's steps see they are done without them."""
    if removed(model):
        return (STORE / model / "ident").read_text()
    return digest({f.name: sha(f) for f in [copy_of(model, "BF16"), imatrix(model)] + sorted(copies(model))})


def disk(need: int, what: str) -> None:
    free = shutil.disk_usage(DATA).free
    if free < need + MARGIN:
        raise SystemExit(f"{what}: needs {(need + MARGIN) / GIB:.0f} GiB of free disk, {free / GIB:.0f} GiB free; "
                         f"free some (another model's type copies: ./panther run of it ends by removing them)")


def fetch(model: str) -> None:
    """Needs the network. The store's backup if there is one (restored, never rebuilt), else the original's weights
    at the pinned revision, unless the 16-bit GGUF is already there. Skips what is on disk."""
    from huggingface_hub import HfApi
    if stored(model):
        return
    hf = os.environ.get("PANTHER_HF", "hf")
    have = sum(f.stat().st_size for f in (STORE / model).glob("*.gguf")) if (STORE / model).exists() else 0
    if HfApi().repo_exists(repo(model)):
        disk(int(Model(model).params * 8.5) - have, f"restoring {repo(model)}")
        subprocess.run([hf, "download", repo(model), "--local-dir", str(STORE / model)], check=True)
        if not stored(model):
            lack = [copy_of(model, t).name for t in ["BF16"] + CFG["formats"] if not copy_of(model, t).exists()]
            lack += [f"rounded.json entry {t}" for t in CFG["formats"] if t not in rounded(model)]
            raise SystemExit(f"{repo(model)} is incomplete after restoring it: missing {', '.join(lack)}. Rebuild "
                             f"instead: delete that repo (python -c \"from huggingface_hub import HfApi; HfApi()."
                             f"delete_repo('{repo(model)}')\") and rerun the run; it builds the missing copies from the "
                             f"original and backs the store up again")
        (STORE / model / "backup.json").write_text(json.dumps(sorted(f.name for f in copies(model))))
        note(f"type copies restored from {repo(model)}")
    elif not copy_of(model, "BF16").exists():
        m = MODELS[model]
        disk(2 * Model(model).params, "the original's weights")
        subprocess.run([hf, "download", m["repo"], "--revision", m["revision"], "--local-dir",
                        str(DATA / "hf" / model)], check=True)  # resumes a cut-off download, skips files already there


def build(model: str) -> None:
    """The 16-bit original, its importance matrix, and the whole model once in every type. One build of a model at a
    time; the next finds the work done."""
    from panther.rounding import blank, round_copies
    types = CFG["formats"]
    d, bf16, imat, hf = STORE / model, copy_of(model, "BF16"), imatrix(model), DATA / "hf" / model
    with lock(f"store-{model}"):
        d.mkdir(parents=True, exist_ok=True)
        if not bf16.exists():
            if not (hf / "config.json").exists():
                raise SystemExit(f"{hf}: the original is not downloaded (the fetch container downloads it)")
            disk(2 * Model(model).params, "the 16-bit original")
            subprocess.run([os.environ.get("PANTHER_CONVERT_PY", sys.executable), str(LLAMA / "convert_hf_to_gguf.py"),
                            str(hf), "--outtype", "bf16", "--outfile", f"{bf16}.part"], check=True)
            os.rename(f"{bf16}.part", bf16)
        if hf.exists():
            shutil.rmtree(hf)  # rounding reads the 16-bit GGUF: the original from here on
        size = bf16.stat().st_size
        for f, out in ((imat, MODELS[model].get("imatrix_process_output")), (imatrix_out(model), True)):
            if not f.exists():  # the store's importance data; the same with the output layer's, for the offline error
                with hold(size + 4 * GIB, 2 * size):
                    subprocess.run([str(BIN / "llama-imatrix"), "-m", str(bf16), "-f", str(WEB["train"]), "-o",
                                    f"{f}.part.gguf", "-c", "512", "-ngl", "99", "-t", str(THREADS)]
                                   + (["--process-output"] if out else []), check=True)
                os.rename(f"{f}.part.gguf", f)
        if removed(model):
            return  # removed after their backup; fetch restores them when a step of the run still needs them
        M, have = Model(model), [t for t in rounded(model) if copy_of(model, t).exists()]
        al = M.allowed()
        for t in types:
            if not copy_of(model, t).exists():  # embedding / head take Q8_0 where the type needs imatrix data they lack
                disk(int(M.params * bpw(t) / 8), f"the {t} copy")
                emb, head = ((t if t in al.get(n, al["token_embd.weight"]) else "Q8_0").lower()
                             for n in ("token_embd.weight", "output.weight"))
                blank(bf16, imat, t, FTYPE.get(t, t), emb, head, copy_of(model, t))
        if todo := [t for t in types if t not in have]:
            h, i = M.matrices[2].cols, max(x.cols for x in M.matrices)
            per_layer = 4 * (3 * h * h + i * i)  # bytes of the four input Hessians of one layer
            with hold(*hold.total):  # the model, half the rest for Hessians, the other half for types rounded at once
                round_copies(bf16, {t: copy_of(model, t) for t in todo}, WEB["train"],
                             max(1, int((hold.total[0] - size) / 2 // per_layer)), hold.total[0])
            put(d / "rounded.json", json.dumps(have + todo))


def backup(model: str) -> None:
    """Needs the network. The finished store to its Hugging Face repo, once (later runs restore it)."""
    from huggingface_hub import HfApi
    if not stored(model):
        return  # the type copies were removed after an earlier backup
    api, r, names = HfApi(), repo(model), sorted(f.name for f in copies(model))
    api.create_repo(r, private=False, exist_ok=True)
    if not set(names) <= set(api.list_repo_files(r)):
        api.upload_folder(repo_id=r, folder_path=STORE / model, allow_patterns=["*.gguf", "rounded.json"],
                          ignore_patterns=["*.part.gguf", "*.part"])  # a killed write's leftover is never backed up
        note(f"type copies saved to {r}")
    (STORE / model / "backup.json").write_text(json.dumps(names))


def clean(model: str, busy: list[str]) -> None:
    """The type copies removed (the 16-bit original, the importance matrix and the presets stay) once backed up and
    no other run of the model (busy) is active."""
    f, have = STORE / model / "backup.json", copies(model)
    if busy or not have or not f.exists() or not {g.name for g in have} <= set(json.loads(f.read_text())):
        if have:
            note(f"type copies kept ({'in use by ' + ', '.join(busy) if busy else 'no backup recorded yet'})")
        return
    put(STORE / model / "ident", ident(model))
    for g in have:
        g.unlink()
        Path(f"{g}.sha256").unlink(missing_ok=True)
    (STORE / model / "rounded.json").unlink(missing_ok=True)
    note("type copies removed (backed up)")


def drop_presets(model: str, busy: list[str]) -> None:
    """The preset files no budget's preset record chose, once no other run of the model (busy) may be scoring them."""
    keep = {json.loads(f.read_text())["preset"] for f in RW.glob(f"budget-*/preset-{model}.json")}
    for f in [] if busy else (STORE / model).glob("preset-*.gguf"):
        if f.name[len("preset-"):-len(".gguf")] not in keep:
            f.unlink()
            Path(f"{f}.sha256").unlink(missing_ok=True)
            Path(f"{f}.inputs").unlink(missing_ok=True)
