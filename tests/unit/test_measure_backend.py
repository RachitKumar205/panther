"""The fake backend is deterministic; the real backend builds the right commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from panther_measure.backend import FakeLlama, LlamaSettings, SubprocessLlama

FIXTURES = Path(__file__).resolve().parents[2] / "tests/parity/fixtures/llama"


def test_fake_kl_and_choice_are_deterministic() -> None:
    backend = FakeLlama(correct_rate=0.7)
    assert backend.kl(Path("m"), Path("ref"), 2048).mean == pytest.approx(0.1)
    task = Path("mmlu.txt")
    assert backend.choice(Path("m"), task, 100, 4096).tolist() == backend.choice(Path("m"), task, 100, 4096).tolist()
    assert len(backend.choice(Path("m"), task, 100, 4096)) == 100


def test_fake_completions_echo_when_unconfigured() -> None:
    assert FakeLlama().complete(Path("m"), ["hello"], chat=False) == ["answer for hello"]


def test_subprocess_kl_builds_the_command_and_parses() -> None:
    captured: list[list[str]] = []

    def run(cmd: list[str]) -> str:
        captured.append(cmd)
        return (FIXTURES / "kl_q6k_48chunks.log").read_text()

    backend = SubprocessLlama(LlamaSettings(bin_dir=Path("/opt/llama"), threads=8, gpu_layers=99), run=run)
    metrics = backend.kl(Path("m.gguf"), Path("ref.kld"), 2048)
    assert metrics.mean == pytest.approx(0.005810, abs=1e-6)
    assert captured[0] == [
        "/opt/llama/llama-perplexity",
        "-m",
        "m.gguf",
        "--kl-divergence-base",
        "ref.kld",
        "--kl-divergence",
        "-c",
        "2048",
        "-b",
        "2048",
        "-t",
        "8",
        "-ngl",
        "99",
    ]


def test_llama_settings_rejects_zero_threads() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        LlamaSettings(bin_dir=Path("/opt/llama"), threads=0, gpu_layers=99)
