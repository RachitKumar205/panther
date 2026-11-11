"""GSM8K scoring (copied from lm-eval 0.4.13's gsm8k.yaml) and generation-check helpers."""

from panther.quality.gencheck import degenerate, make_prompts, matched_prefix
from panther.quality.gsm8k import INVALID, exact_match, flexible_extract, prompt, strict_match

REF = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips.\n#### 72"


def test_prompt_layout():
    p = prompt("How many?", [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}])
    assert p == "Question: Q1\nAnswer: A1\n\nQuestion: Q2\nAnswer: A2\n\nQuestion: How many?\nAnswer:"


def test_strict_and_flexible():
    gen = " She sold 24 in May, so 48 + 24 = 72.\n#### 72"
    assert strict_match(gen) == "72" and exact_match(strict_match(gen), REF)
    assert strict_match("The answer is 72.") == INVALID
    assert flexible_extract("The answer is 72.") == "72."
    assert exact_match(flexible_extract("The answer is 72."), REF)  # trailing '.' removed
    assert exact_match("$1,200", "... #### 1200")
    assert not exact_match(INVALID, REF)


def test_flexible_takes_last_number():
    assert flexible_extract("First 10, then 5, total 15") == "15"
    assert flexible_extract("no digits") == INVALID


def test_matched_prefix_and_degenerate():
    assert matched_prefix([1, 2, 3], [1, 2, 4]) == 2 and matched_prefix([1], [1]) == 1
    assert degenerate("") and degenerate(" ".join(["the cat sat on"] * 20))
    assert not degenerate("A normal sentence with enough different words to count as varied text here.")


def test_prompts_are_deterministic():
    sel = {t: [{"id": f"{t}/{i}", "ctx": "c", "question": "q", "choices": ["a", "b"], "sentence": "s _"}
               for i in range(30)] for t in ("hellaswag", "arc_challenge", "mmlu", "winogrande", "gsm8k")}
    a, b = make_prompts(sel), make_prompts(sel)
    assert a == b and len(a) == 50 and len({p["id"] for p in a}) == 50
