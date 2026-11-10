import pytest

from panther.data.splits import (
    WEB_SPLITS,
    check_disjoint,
    gsm8k_items,
    mmlu_items,
    sample,
    shard_of,
    take_web_text,
    winogrande_items,
)


def _docs(n=4000):
    ids = [f"doc-{i}" for i in range(n)]
    texts = [" ".join(["w"] * (50 + (i * 37) % 400)) for i in range(n)]
    return ids, texts


def count(s: str) -> int:
    return len(s.split())


def test_web_splits_disjoint_and_meet_targets():
    ids, texts = _docs()
    targets = {"calibration": 20000, "sensitivity": 8000, "selection": 10000, "test": 10000}
    out = take_web_text(ids, texts.__getitem__, count, targets, seed=7, min_doc_tokens=100, max_doc_tokens=400)
    seen = set()
    for s, name in enumerate(WEB_SPLITS):
        sp = out[name]
        assert sp.tokens >= targets[name]
        assert all(shard_of(7, d) == s for d in sp.doc_ids)
        assert all(100 <= count(t) <= 400 for t in sp.texts)
        assert not seen & set(sp.doc_ids)
        seen |= set(sp.doc_ids)


def test_web_splits_deterministic_and_seed_dependent():
    ids, texts = _docs()
    t = {k: 5000 for k in WEB_SPLITS}
    a = take_web_text(ids, texts.__getitem__, count, t, seed=1)
    b = take_web_text(ids, texts.__getitem__, count, t, seed=1)
    c = take_web_text(ids, texts.__getitem__, count, t, seed=2)
    assert a["calibration"].doc_ids == b["calibration"].doc_ids
    assert a["calibration"].doc_ids != c["calibration"].doc_ids


def test_web_split_exhausted_raises():
    ids, texts = _docs(100)
    with pytest.raises(ValueError, match="exhausted"):
        take_web_text(ids, texts.__getitem__, count, {k: 10**6 for k in WEB_SPLITS}, seed=0)


def test_sample_excludes_overlap_and_fewshot():
    test = gsm8k_items([{"question": "What is 2+2?", "answer": "4 #### 4"}], "test")
    pool = gsm8k_items(
        [{"question": "what is 2 + 2 ?", "answer": "x #### 4"}]  # same content as the test item
        + [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(50)], "train")
    fewshot_ids = {pool[1]["id"]}
    sel, dropped = sample(pool, 20, seed=0, exclude_keys={t["key"] for t in test}, exclude_ids=fewshot_ids)
    assert dropped == 1 and len(sel) == 20
    assert pool[0]["id"] not in {s["id"] for s in sel} and pool[1]["id"] not in {s["id"] for s in sel}
    check_disjoint({"test": test, "selection": sel})
    with pytest.raises(ValueError, match="overlaps"):
        check_disjoint({"test": test, "selection": [pool[0]]})


def test_item_builders():
    w = winogrande_items([{"sentence": "A _ b", "option1": "x", "option2": "y", "answer": "2"}], "validation")
    assert w[0]["label"] == 1 and w[0]["id"] == "winogrande/validation/0"
    m = mmlu_items([{"subject": "law", "question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 3}], "test")
    assert m[0]["id"] == "mmlu/test/law/0" and m[0]["label"] == 3
    g = gsm8k_items([{"question": "q", "answer": "steps\n#### 1,234"}], "test")
    assert g[0]["final"] == "1234"
