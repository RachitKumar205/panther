"""Convert the split task JSONL files into llama-perplexity inputs (docs/PIPELINE.md A4.1).

  python scripts/server/build_tasks.py

Writes $PANTHER_DATA/splits/llama_tasks/{selection,test}/{hellaswag,arc_challenge,mmlu}.bin and
winogrande.csv, plus manifest.json (question ids in file order, counts, sha256). Every file is read back
with readers that mirror llama.cpp's parsers before it is accepted. Both splits are converted; only
selection is evaluated in Phase A (D34).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from panther.models.inventory import REPO_ROOT
from panther.quality.tasks import (
    MC_TASKS,
    mc_tasks,
    mmlu_fewshot_by_subject,
    parse_winogrande_csv,
    read_mc_binary,
    write_mc_binary,
    write_winogrande_csv,
)

DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
SPLITS = DATA / "splits"
OUT = SPLITS / "llama_tasks"


def load(split: str, task: str) -> list[dict]:
    return [json.loads(line) for line in (SPLITS / "tasks" / split / f"{task}.jsonl").read_text().split("\n") if line]


def main() -> int:
    fewshot = mmlu_fewshot_by_subject(load("fewshot", "mmlu"))
    manifest: dict = {"splits_manifest_sha256": hashlib.sha256((SPLITS / "manifest.json").read_bytes()).hexdigest(),
                      "files": {}}
    for split in ("selection", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        for task in (*MC_TASKS, "winogrande"):
            items = load(split, task)
            if task == "winogrande":
                text = write_winogrande_csv(items)
                rows, skipped = parse_winogrande_csv(text)
                if skipped or len(rows) != len(items):
                    raise SystemExit(f"{split}/{task}: llama.cpp's parser would keep {len(rows)} of {len(items)}")
                for it, r in zip(items, rows):
                    s = it["sentence"]
                    w = s.index("_")
                    if (r.first, r.second, r.choices, r.answer) != (s[:w], s[w + 1:], (it["option1"], it["option2"]),
                                                                     it["label"] + 1):
                        raise SystemExit(f"{split}/{task} {it['id']}: CSV round trip differs")
                data = text.encode("utf-8")
                path = OUT / split / f"{task}.csv"
            else:
                tasks = mc_tasks(task, items, fewshot if task == "mmlu" else None)
                data = write_mc_binary(tasks)
                if read_mc_binary(data) != tasks:
                    raise SystemExit(f"{split}/{task}: binary round trip differs")
                path = OUT / split / f"{task}.bin"
            path.write_bytes(data)
            manifest["files"][f"{split}/{task}"] = {
                "file": str(path.relative_to(SPLITS)), "n": len(items), "sha256": hashlib.sha256(data).hexdigest(),
                "ids": [it["id"] for it in items]}
            print(f"{split}/{task}: {len(items)} items, {len(data) / 1e6:.2f} MB, round trip ok", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
