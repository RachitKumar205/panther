"""Split integrity flags missing files, hash mismatches and question leakage (B2, B19)."""

from __future__ import annotations

import hashlib

from panther_data.integrity import manifest_problems, question_overlap, sha256_file


def write(tmp_path, name: str, content: bytes) -> None:
    (tmp_path / name).write_bytes(content)


def test_manifest_problems_accepts_a_clean_tree(tmp_path) -> None:
    write(tmp_path, "a.jsonl", b"hello")
    manifest = {"a.jsonl": hashlib.sha256(b"hello").hexdigest()}
    assert manifest_problems(manifest, tmp_path, ["a.jsonl"]) == []


def test_manifest_problems_flags_missing_files_and_hash_mismatches(tmp_path) -> None:
    write(tmp_path, "a.jsonl", b"hello")
    manifest = {"a.jsonl": hashlib.sha256(b"hello").hexdigest(), "b.jsonl": "deadbeef"}
    problems = manifest_problems(manifest, tmp_path, ["a.jsonl", "b.jsonl"])
    assert any("b.jsonl: missing" in p for p in problems)
    # Hash mismatch.
    write(tmp_path, "b.jsonl", b"other")
    problems = manifest_problems(manifest, tmp_path, ["a.jsonl", "b.jsonl"])
    assert any("b.jsonl: hash differs" in p for p in problems)


def test_manifest_problems_flags_files_omitted_from_the_manifest(tmp_path) -> None:
    # Bug B19: a manifest that omits a measured file must not pass.
    write(tmp_path, "a.jsonl", b"hello")
    assert manifest_problems({}, tmp_path, ["a.jsonl"]) == ["a.jsonl: not in the manifest"]


def test_manifest_problems_flags_manifest_entries_never_measured(tmp_path) -> None:
    write(tmp_path, "a.jsonl", b"hello")
    manifest = {"a.jsonl": hashlib.sha256(b"hello").hexdigest(), "stale.jsonl": "deadbeef"}
    problems = manifest_problems(manifest, tmp_path, ["a.jsonl"])
    assert any("stale.jsonl: in the manifest but not measured" in p for p in problems)


def test_question_overlap_detects_shared_questions(tmp_path) -> None:
    # Bug B2: the check runs on the measured selection and held-out question sets.
    assert question_overlap({"mmlu": ["q1", "q2"]}, {"mmlu": ["q2", "q3"]}) == [
        "mmlu: 1 questions in both selection and held-out"
    ]
    assert question_overlap({"mmlu": ["q1"]}, {"mmlu": ["q2"]}) == []


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    write(tmp_path, "a.bin", b"data")
    assert sha256_file(tmp_path / "a.bin") == hashlib.sha256(b"data").hexdigest()
