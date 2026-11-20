"""The split algorithms are deterministic and reject leakage."""

from __future__ import annotations

from panther_data.dataset import (
    Question,
    SplitName,
    SplitSpec,
    WebDocument,
    build_splits,
    fill_to_budget,
    overlapping,
    partition_questions,
    partition_web,
    split_prompts,
    words,
)


def test_words_lowercases_and_keeps_alphanumeric_runs() -> None:
    assert words("Hello, World! 123") == ["hello", "world", "123"]


def test_overlapping_flags_a_shared_run_of_n_words() -> None:
    prompt = "the quick brown fox jumps over the lazy dog"
    text = "here the quick brown fox jumps over the lazy dog again"
    assert overlapping([prompt], [text], 8) == frozenset({0})


def test_overlapping_ignores_short_prompts_not_inside_any_text() -> None:
    assert overlapping(["the quick brown fox"], ["unrelated words in a text"], 8) == frozenset()


def test_overlapping_flags_a_short_prompt_inside_a_text() -> None:
    assert overlapping(["lazy dog"], ["a lazy dog ran"], 8) == frozenset({0})


def test_split_prompts_is_deterministic_with_the_right_counts() -> None:
    prompts = [f"prompt {i}" for i in range(10)]
    counts = {SplitName.TRAIN: 4, SplitName.SELECT: 3, SplitName.TEST: 3}
    a = split_prompts(prompts, counts, seed=0)
    b = split_prompts(prompts, counts, seed=0)
    assert a == b
    assert (len(a[SplitName.TRAIN]), len(a[SplitName.SELECT]), len(a[SplitName.TEST])) == (4, 3, 3)
    flat = [p for split in a.values() for p in split]
    assert len(set(flat)) == 10  # no prompt in two splits


def test_partition_questions_takes_the_first_k_for_selection() -> None:
    sel, held = partition_questions({"mmlu": [f"q{i}" for i in range(10)]}, {"mmlu": 3})
    assert sel["mmlu"] == ["q0", "q1", "q2"]
    assert held["mmlu"] == [f"q{i}" for i in range(3, 10)]


def test_fill_to_budget_keeps_whole_docs_under_the_budget() -> None:
    docs = ["a" * 10, "b" * 10, "c" * 10, "d" * 10]
    out = fill_to_budget(docs, budget=25, count_tokens=len, seed=0)
    assert sum(map(len, out)) <= 25
    assert set(out) <= set(docs)


def test_question_key_covers_stem_choices_and_answer() -> None:
    q = Question(id="0", task="mmlu", stem="What is 1+1?", choices=("1", "2"), answer="B")
    same = Question(id="1", task="mmlu", stem="What is 1+1?", choices=("1", "2"), answer="B")
    different = Question(id="2", task="mmlu", stem="What is 1+1?", choices=("1", "2"), answer="A")
    assert q.key() == same.key()
    assert q.key() != different.key()


def test_split_spec_defaults_pin_the_decided_sizes() -> None:
    spec = SplitSpec()
    assert spec.prompt_counts == {SplitName.TRAIN: 300, SplitName.SELECT: 150, SplitName.TEST: 150}
    assert spec.web_tokens[SplitName.SELECT] == 100_352
    assert spec.selection_questions["mmlu"] == 300
    assert spec.overlap_n == 8


class _Source:
    def __init__(self, prompts, questions, web=None):
        self._prompts = prompts
        self._questions = questions
        self._web = web or []

    def chat_prompts(self):
        return self._prompts

    def questions(self):
        return self._questions

    def web_documents(self):
        return self._web


def test_build_splits_drops_a_prompt_overlapping_a_question() -> None:
    stem = "alpha beta gamma delta epsilon zeta eta theta"
    question = Question(id="0", task="mmlu", stem=stem, answer="A")
    source = _Source([stem, *[f"clean {i}" for i in range(600)]], {"mmlu": [question]})
    built = build_splits(source, SplitSpec(), len)
    flat = [p for split in built.prompts.values() for p in split]
    assert stem not in flat  # the overlapping prompt was dropped
    assert len(built.prompts[SplitName.TRAIN]) == 300
    assert built.selection["mmlu"] == ["0"]


def test_build_splits_holds_out_tasks_absent_from_selection() -> None:
    qs = {"gsm8k": [Question(id=str(i), task="gsm8k", stem=f"gsm {i}", answer="1") for i in range(5)]}
    built = build_splits(_Source([f"clean {i}" for i in range(600)], qs), SplitSpec(), len)
    assert built.selection["gsm8k"] == []
    assert built.heldout["gsm8k"] == [str(i) for i in range(5)]


def test_partition_web_fills_train_then_test_then_select_disjointly() -> None:
    docs = [WebDocument(id=str(i), text="x" * 10) for i in range(10)]
    budgets = {SplitName.TRAIN: 40, SplitName.TEST: 30, SplitName.SELECT: 20}
    web = partition_web(docs, budgets, len, seed=0)
    assert all(d not in web[SplitName.TEST] and d not in web[SplitName.SELECT] for d in web[SplitName.TRAIN])
    ids = [d.id for split in web.values() for d in split]
    assert len(ids) == len(set(ids))  # disjoint by id
    assert sum(len(d.text) for d in web[SplitName.TRAIN]) <= 40
