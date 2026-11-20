"""A step is skipped only when its complete input digest matches (bug B1)."""

from __future__ import annotations

import json

from panther_pipeline.steps import run_step, write_atomic


def test_run_step_runs_once_then_skips_until_inputs_change(tmp_path) -> None:
    out = tmp_path / "r.json"
    calls = 0

    def compute() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"x": 1}

    assert run_step("s", out, "k1", compute) is True
    assert run_step("s", out, "k1", compute) is False
    assert calls == 1
    assert run_step("s", out, "k2", compute) is True
    assert calls == 2
    assert json.loads(out.read_text()) == {"inputs": "k2", "x": 1}


def test_write_atomic_leaves_no_part_files(tmp_path) -> None:
    out = tmp_path / "r.json"
    write_atomic(out, '{"a": 1}')
    assert out.read_text() == '{"a": 1}'
    assert list(tmp_path.iterdir()) == [out]
