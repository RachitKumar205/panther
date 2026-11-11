"""D39: independent check of task scoring (docs/DECISIONS.md, PIPELINE.md A4.4). GPU container, no network.

  python scripts/server/check_d39.py

For 200 seeded selection items per task, scores the BF16 Hugging Face model in transformers with
(a) llama.cpp's rule, reimplemented from tools/perplexity/perplexity.cpp, and (b) lm-evaluation-harness's
rule, using the same prompt strings as the llama.cpp input files. Compares (a) per question with the latest
llama.cpp BF16 CUDA record (right/wrong; llama.cpp does not print which answer it chose for multiple
choice). Writes $PANTHER_DATA/gate/a4/d39.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from panther.models.inventory import REPO_ROOT
from panther.quality.tasks import mc_tasks, mmlu_fewshot_by_subject

MODEL = "Qwen2.5-1.5B-Instruct"
DATA = Path(os.environ.get("PANTHER_DATA", REPO_ROOT / "data"))
N, SEED = 200, 20261111
TASKS = ("hellaswag", "arc_challenge", "mmlu", "winogrande")


def load(split: str, task: str) -> list[dict]:
    return [json.loads(x) for x in (DATA / "splits/tasks" / split / f"{task}.jsonl").read_text().split("\n") if x]


class Scorer:
    def __init__(self) -> None:
        path = DATA / "hf" / MODEL
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=True)["input_ids"]

    @torch.no_grad()
    def token_logprobs(self, ids: list[int]) -> np.ndarray:
        """lp[j] = log P(ids[j+1] | ids[:j+1]), for j = 0 .. len-2."""
        x = torch.tensor([ids], device="cuda")
        logits = self.model(x).logits[0].float()
        lp = torch.log_softmax(logits[:-1], dim=-1)
        return lp.gather(1, x[0, 1:, None])[:, 0].cpu().numpy()

    # --- llama.cpp rules ------------------------------------------------------------------------------
    def llama_mc(self, question: str, answers: list[str]) -> int:
        seqs = [self.ids(question + " " + a) for a in answers]
        cp = 0
        while all(len(s) > cp for s in seqs) and len({s[cp] for s in seqs}) == 1:
            cp += 1
        scores = []
        for s in seqs:
            lp = self.token_logprobs(s)
            scores.append(lp[cp - 1:len(s) - 1].mean())  # tokens cp .. len-1, each given its prefix
        return int(np.argmax(scores))  # first maximum, as llama.cpp's strict '>' scan

    def llama_wino(self, first: str, second: str, choices: tuple[str, str]) -> int:
        seqs = [self.ids(first + c + second) for c in choices]
        cp = 0
        while cp < min(map(len, seqs)) and seqs[0][cp] == seqs[1][cp]:
            cp += 1
        bases = [len(self.ids(first + c)) for c in choices]
        skip_choice = all(len(s) - cp > 3 for s in seqs)
        scores = []
        for s, nb in zip(seqs, bases):
            n_base = nb if skip_choice else cp
            last = 1 if len(s) - n_base > 1 else 0
            lp = self.token_logprobs(s)
            scores.append(lp[n_base - 1:len(s) - 1 - last].mean())
        return 1 if scores[0] > scores[1] else 2

    # --- lm-eval rules ---------------------------------------------------------------------------------
    def loglik(self, context: str, continuation: str) -> float:
        ctx, whole = self.ids(context), self.ids(context + continuation)
        return float(self.token_logprobs(whole)[len(ctx) - 1:].sum())

    def lmeval_mc(self, question: str, answers: list[str], normalise: bool) -> int:
        ll = [self.loglik(question, " " + a) for a in answers]
        if normalise:  # acc_norm: lm-eval divides by len(choice) in characters (the choice without delimiter)
            ll = [v / len(a) for v, a in zip(ll, answers)]
        return int(np.argmax(ll))

    def lmeval_wino(self, first: str, second: str, choices: tuple[str, str]) -> int:
        cont = " " + second.strip()
        ll = [self.loglik(first + c, cont) for c in choices]
        return 1 if ll[0] > ll[1] else 2  # lm-eval picks the higher log-likelihood


def latest_bf16_record(task: str) -> dict:
    recs = [json.loads(p.read_text()) for p in (DATA / "results" / MODEL).glob("*.json")]
    hits = [r for r in recs if r["kind"] == "task" and r["label"] == "a4:bf16" and r["how"]["build"] == "cuda"
            and r["how"]["task"] == task and r["how"]["split"] == "selection" and r["ok"]]
    if not hits:
        raise SystemExit(f"no llama.cpp BF16 CUDA record for {task}")
    return sorted(hits, key=lambda r: r["eval_id"])[-1]


def main() -> int:
    sc = Scorer()
    fewshot = mmlu_fewshot_by_subject(load("fewshot", "mmlu"))
    rng = np.random.default_rng(SEED)
    report = {"n_per_task": N, "seed": SEED, "tasks": {}}
    for task in TASKS:
        items = load("selection", task)
        pick = sorted(rng.choice(len(items), size=N, replace=False).tolist())
        llama_correct = latest_bf16_record(task)["per_item"]["correct"]
        mine, lmeval, agree = [], [], []
        if task == "winogrande":
            for i in pick:
                it = items[i]
                s, w = it["sentence"], it["sentence"].index("_")
                first, second, ch = s[:w], s[w + 1:], (it["option1"], it["option2"])
                gold = it["label"] + 1
                c_mine = int(sc.llama_wino(first, second, ch) == gold)
                mine.append(c_mine)
                lmeval.append(int(sc.lmeval_wino(first, second, ch) == gold))
                agree.append(int(c_mine == llama_correct[i]))
        else:
            tasks = mc_tasks(task, [items[i] for i in pick], fewshot if task == "mmlu" else None)
            for i, t in zip(pick, tasks):
                gold = t.labels.index(1)
                c_mine = int(sc.llama_mc(t.question, t.answers) == gold)
                mine.append(c_mine)
                lmeval.append(int(sc.lmeval_mc(t.question, t.answers, normalise=task != "mmlu") == gold))
                agree.append(int(c_mine == llama_correct[i]))
        report["tasks"][task] = {
            "agreement_with_llama_cpp": float(np.mean(agree)),
            "accuracy_llama_rule_transformers": float(np.mean(mine)),
            "accuracy_llama_cpp_same_items": float(np.mean([llama_correct[i] for i in pick])),
            "accuracy_lm_eval_rule": float(np.mean(lmeval)),
            "lm_eval_rule": "acc" if task == "mmlu" else ("partial loglik" if task == "winogrande" else "acc_norm"),
        }
        v = report["tasks"][task]
        print(f"{task:14s} agreement {v['agreement_with_llama_cpp']:.1%} | llama rule {v['accuracy_llama_rule_transformers']:.1%} "
              f"(llama.cpp {v['accuracy_llama_cpp_same_items']:.1%}) | lm-eval rule {v['accuracy_lm_eval_rule']:.1%}",
              flush=True)
    report["pass"] = all(v["agreement_with_llama_cpp"] >= 0.95 for v in report["tasks"].values())
    out = DATA / "gate" / "a4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "d39.json").write_text(json.dumps(report, indent=2))
    print("D39:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
