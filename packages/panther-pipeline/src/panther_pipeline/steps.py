"""The resumable step graph (bug B1 end-to-end).

A step runs only if its output file is missing or records a different input digest. The digest must
hash every input that can change the result, so a stale result is never reused.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` whole or not at all: write a part file, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_step(name: str, output: Path, inputs: str, compute: Callable[[], dict[str, object]]) -> bool:
    """Run ``compute`` and record its result, unless ``output`` already records ``inputs``.

    Args:
        name: The step name (for logging only).
        output: Where the result record is written.
        inputs: The complete input digest (see :func:`panther_core.cache.key`).
        compute: Called only on a miss; returns JSON-serializable fields.

    Returns:
        True when the step ran, False when it was skipped.
    """
    if output.exists() and json.loads(output.read_text()).get("inputs") == inputs:
        return False
    write_atomic(output, json.dumps({"inputs": inputs, **compute()}, indent=1))
    return True
