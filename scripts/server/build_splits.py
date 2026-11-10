"""Build the four disjoint data splits (PLAN.md §3) into $PANTHER_DATA/splits. Needs the network for
the dataset downloads, so it runs as a CPU job, after the tokenizer model has been fetched:

  deploy/server/panther.sh job splits /opt/venv/bin/python scripts/server/build_splits.py

Writes (atomically: built in splits.tmp, then renamed):
  calibration.txt                         imatrix input (= text/calibration.txt)
  text/{calibration,sensitivity,selection,test_web,test_wikitext}.txt
  text/<name>.ids.json                    FineWeb-Edu document ids per web split
  tasks/{test,selection,fewshot}/<task>.jsonl
  manifest.json                           sources (repo, revision, file sha256), counts, token counts,
                                          content hashes of every output file, overlap drops
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from panther.data.splits import ITEM_BUILDERS, check_disjoint, sample, take_web_text
from panther.models.inventory import REPO_ROOT

DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "data.yaml"))
    seed = cfg["seed"]
    out = DATA / "splits.tmp"
    shutil.rmtree(out, ignore_errors=True)
    (out / "text").mkdir(parents=True)
    manifest: dict = {"seed": seed, "sources": {}, "web_text": {}, "tasks": {}, "overlap_dropped": {}}

    # --- downloads (pinned revisions), recorded with file hashes ---
    paths: dict[str, dict[str, Path]] = {}
    for name, src in cfg["sources"].items():
        paths[name] = {}
        manifest["sources"][name] = {"repo": src["repo"], "revision": src["revision"], "files": {}}
        for f in src["files"]:
            p = Path(hf_hub_download(src["repo"], f, revision=src["revision"], repo_type="dataset"))
            paths[name][f] = p
            manifest["sources"][name]["files"][f] = file_sha256(p)
            print(f"downloaded {src['repo']}:{f}", flush=True)

    tok_dir = DATA / "hf" / cfg["tokenizer_model"]
    tok = AutoTokenizer.from_pretrained(tok_dir)
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    manifest["tokenizer"] = {"model": cfg["tokenizer_model"],
                             "revision": (tok_dir / ".panther-revision").read_text().strip()}

    # --- web text: FineWeb-Edu shards A-D ---
    wt = cfg["web_text"]
    tables = [pq.read_table(p, columns=["id", "text"]) for p in paths["fineweb_edu"].values()]
    ids = [i for t in tables for i in t.column("id").to_pylist()]
    texts = [x for t in tables for x in t.column("text").to_pylist()]
    del tables
    web = take_web_text(ids, texts.__getitem__, count, wt["tokens"], seed,
                        wt["min_doc_tokens"], wt["max_doc_tokens"])
    del texts
    for name, split in web.items():
        fname = "test_web" if name == "test" else name
        (out / "text" / f"{fname}.txt").write_text(split.joined())
        (out / "text" / f"{fname}.ids.json").write_text(json.dumps(split.doc_ids))
        manifest["web_text"][fname] = {"docs": len(split.doc_ids), "tokens": split.tokens}
        print(f"web {fname}: {len(split.doc_ids)} docs, {split.tokens} tokens", flush=True)
    shutil.copy(out / "text" / "calibration.txt", out / "calibration.txt")

    wiki = pq.read_table(next(iter(paths["wikitext"].values())), columns=["text"]).column("text").to_pylist()
    wiki_text = "".join(wiki)
    (out / "text" / "test_wikitext.txt").write_text(wiki_text)
    manifest["web_text"]["test_wikitext"] = {"tokens": count(wiki_text)}
    test_tokens = manifest["web_text"]["test_wikitext"]["tokens"] + manifest["web_text"]["test_web"]["tokens"]
    manifest["web_text"]["test_total_tokens"] = test_tokens
    if test_tokens < 300_000:
        raise SystemExit(f"test text has {test_tokens} tokens < 300k (PLAN.md §2.1)")

    # --- task items ---
    def load(task: str, split: str) -> list[dict]:
        f = next(p for k, p in paths[task].items() if k.rsplit("/", 1)[-1].startswith(f"{split}-"))
        return ITEM_BUILDERS[task](pq.read_table(f).to_pylist(), split)

    tc = cfg["tasks"]
    groups: dict[str, dict[str, list[dict]]] = {"test": {}, "selection": {}, "fewshot": {}}
    for task, split in tc["test"].items():
        groups["test"][task] = load(task, split)
    for task, spec in tc["fewshot"].items():
        if isinstance(spec, str):
            groups["fewshot"][task] = load(task, spec)
    for task, spec in tc["selection"].items():
        splits = spec["split"] if isinstance(spec["split"], list) else [spec["split"]]
        pool = [it for s in splits for it in load(task, s)]
        test_keys = {it["key"] for it in groups["test"][task]}
        fs = tc["fewshot"].get(task)
        fewshot_ids = {it["id"] for it in groups["fewshot"].get(task, [])}
        fewshot_keys = {it["key"] for it in groups["fewshot"].get(task, [])}
        sel, dropped = sample(pool, spec["n"], seed, exclude_keys=test_keys | fewshot_keys, exclude_ids=fewshot_ids)
        groups["selection"][task] = sel
        manifest["overlap_dropped"][task] = dropped
        if isinstance(fs, dict):  # few-shot sampled from the same pool, disjoint from selection and test
            rest = [it for it in pool if it["id"] not in {s["id"] for s in sel}]
            groups["fewshot"][task], _ = sample(rest, fs["n"], seed + 1, exclude_keys=test_keys)

    # Hard guarantees: selection is disjoint from test and from few-shot exemplars. Few-shot vs test
    # overlap inside an official benchmark (MMLU dev duplicates some test items) is inherited, kept for
    # comparability with published numbers, identical for every configuration, and recorded.
    manifest["inherited_fewshot_test_overlap"] = {}
    for task in tc["test"]:
        check_disjoint({"test": groups["test"][task], "selection": groups["selection"].get(task, [])})
        check_disjoint({"selection": groups["selection"].get(task, []), "fewshot": groups["fewshot"].get(task, [])})
        fs_keys = {it["key"] for it in groups["fewshot"].get(task, [])}
        manifest["inherited_fewshot_test_overlap"][task] = sum(it["key"] in fs_keys for it in groups["test"][task])

    for g, tasks in groups.items():
        (out / "tasks" / g).mkdir(parents=True)
        for task, items in tasks.items():
            with open(out / "tasks" / g / f"{task}.jsonl", "w") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            manifest["tasks"].setdefault(g, {})[task] = len(items)
            print(f"tasks {g}/{task}: {len(items)}", flush=True)

    manifest["files"] = {str(p.relative_to(out)): file_sha256(p)
                         for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    final = DATA / "splits"
    if final.exists():
        shutil.rmtree(final)
    out.rename(final)
    print(f"splits written to {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
