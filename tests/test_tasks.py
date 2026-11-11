"""Task converters and task-output parsers (A4)."""

import numpy as np
import pytest

from panther.quality.parse import ParseError, parse_multiple_choice, parse_winogrande
from panther.quality.tasks import (
    MCTask,
    arc_prompt,
    hellaswag_preprocess,
    hellaswag_prompt,
    mc_tasks,
    mmlu_prompt,
    parse_winogrande_csv,
    read_mc_binary,
    write_mc_binary,
    write_winogrande_csv,
)


def test_hellaswag_prompt_follows_lm_eval():
    item = {"activity_label": "Removing ice from car", "ctx_a": "Then, the man writes over the snow.",
            "ctx_b": "then", "endings": [" , the man [title] goes. ", "b", "c", "d"], "label": 2}
    q, ans, gold = hellaswag_prompt(item)
    assert q == "Removing ice from car: Then, the man writes over the snow. Then"
    assert ans[0] == ", the man. goes." and gold == 2
    assert hellaswag_preprocess("a [header] b  c") == "a b c"


def test_arc_and_mmlu_prompts():
    q, ans, gold = arc_prompt({"question": "Why?", "choices": ["x", "y"], "label": 1})
    assert q == "Question: Why?\nAnswer:" and ans == ["x", "y"] and gold == 1
    shot = {"subject": "high_school_math", "question": "1+1?", "choices": ["1", "2", "3", "4"], "label": 1}
    item = {"subject": "high_school_math", "question": " 2+2? ", "choices": ["2", "3", "4", "5"], "label": 2}
    q, ans, gold = mmlu_prompt(item, [shot] * 5)
    assert q.startswith("The following are multiple choice questions (with answers) about high school math.\n\n")
    assert q.count("Answer: B\n\n") == 5 and q.endswith("2+2?\nA. 2\nB. 3\nC. 4\nD. 5\nAnswer:")
    assert ans == ["A", "B", "C", "D"] and gold == 2


def test_mc_binary_round_trip_and_layout():
    tasks = [MCTask("Question: é?\nAnswer:", ["a", "bb"], [0, 1]), MCTask("q2", ["x", "y", "z"], [1, 0, 0])]
    data = write_mc_binary(tasks)
    assert read_mc_binary(data) == tasks
    assert int.from_bytes(data[:4], "little") == 2
    assert int.from_bytes(data[4:8], "little") == 12  # first task right after count + 2 offsets
    with pytest.raises(ValueError):
        write_mc_binary([MCTask("q", ["a", ""], [1, 0])])  # llama.cpp rejects empty answers


def test_mc_tasks_labels():
    t = mc_tasks("arc_challenge", [{"question": "q", "choices": ["a", "b", "c"], "label": 2}])
    assert t[0].labels == [0, 0, 1]


def test_winogrande_csv_matches_llama_parser():
    items = [{"id": "w/0", "sentence": "Sarah was better than Maria, so _ got easy cases.",
              "option1": "Sarah", "option2": "Maria", "label": 1},
             {"id": "w/1", "sentence": "The cup is full so _ spilled.", "option1": "cup", "option2": "tea", "label": 0}]
    rows, skipped = parse_winogrande_csv(write_winogrande_csv(items))
    assert skipped == 0 and len(rows) == 2
    assert rows[0].first == "Sarah was better than Maria, so " and rows[0].answer == 2
    assert rows[1].choices == ("cup", "tea") and rows[1].answer == 1
    with pytest.raises(ValueError):
        write_winogrande_csv([{**items[1], "option1": "a, b"}])


def _mc_output(correct):
    lines, n = ["task\tacc_norm"], 0
    for i, c in enumerate(correct, 1):
        n += c
        lines.append(f"{i}\t{100 * n / i:.8f}")
    p = sum(correct) / len(correct)
    return "\n".join(lines) + f"\n\nFinal result: {100 * p:.4f} +/- 1.0000\n"


def test_parse_multiple_choice_recovers_items():
    rng = np.random.default_rng(1)
    correct = rng.integers(0, 2, 14042).tolist()  # MMLU-test size: precision must still hold
    r = parse_multiple_choice(_mc_output(correct), len(correct))
    assert r.correct.tolist() == correct
    with pytest.raises(ParseError):
        parse_multiple_choice(_mc_output(correct[:-1]), len(correct))  # a dropped item is caught


def test_parse_winogrande_rows():
    rows = ["1\t100.0000\t -1.000000   -2.000000  1  1", "2\t50.0000\t -1.500000   -0.500000  2  1"]
    text = "\n".join(rows) + "\n\nFinal Winogrande score(2 tasks): 50.0000 +/- 50.0000\n"
    r = parse_winogrande(text, 2)
    assert r.correct.tolist() == [1, 0]


def test_winogrande_last_row_survives_llama_file_reading():
    items = [{"id": f"w/{i}", "sentence": f"Item {i} so _ won.", "option1": "a", "option2": "b", "label": 0}
             for i in range(3)]
    text = write_winogrande_csv(items)
    rows, dropped = parse_winogrande_csv(text)
    assert len(rows) == 3 and dropped == 0
    # A single trailing newline loses the last row, as llama.cpp did in A4 (999 of 1,000).
    rows, dropped = parse_winogrande_csv(text.rstrip("\n") + "\n")
    assert len(rows) == 2 and dropped == 1
