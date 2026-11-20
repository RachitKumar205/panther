"""Split integrity (bugs B2 and B19).

The check must cover the files the measurement layer actually reads (``llama_tasks``, not a
parallel ``tasks`` tree), and the manifest must cover exactly the files present — a manifest that
omits a measured file no longer passes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_problems(
    manifest: Mapping[str, str],
    directory: Path,
    present: Collection[str],
) -> list[str]:
    """Report missing files, hash mismatches, and manifest/files coverage gaps.

    Args:
        manifest: Relative path -> expected SHA-256.
        directory: Root the relative paths resolve against.
        present: Every relative path the pipeline will measure; each must be in the manifest.
    """
    problems: list[str] = []
    for rel in present:
        path = directory / rel
        if rel not in manifest:
            problems.append(f"{rel}: not in the manifest")
        elif not path.exists():
            problems.append(f"{rel}: missing")
        elif sha256_file(path) != manifest[rel]:
            problems.append(f"{rel}: hash differs")
    for rel in set(manifest) - set(present):
        problems.append(f"{rel}: in the manifest but not measured")
    return problems


def question_overlap(
    selection: Mapping[str, Collection[str]],
    heldout: Mapping[str, Collection[str]],
) -> list[str]:
    """Report questions present in both the measured selection and held-out sets (bug B2).

    Args:
        selection: Task -> question keys actually measured for the search.
        heldout: Task -> question keys actually measured for the verdict.
    """
    problems: list[str] = []
    for task in selection.keys() & heldout.keys():
        shared = set(selection[task]) & set(heldout[task])
        if shared:
            problems.append(f"{task}: {len(shared)} questions in both selection and held-out")
    return problems
