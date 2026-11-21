"""The one entrypoint: `python -m panther fetch 14b` and `python -m panther run 14b@5 [--qat N]`. It calls the
modules in pipeline order; every step skips what it has already made, so running a command again continues where it
stopped. A failure stops the run with its traceback."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch

from panther.data import Dataset
from panther.distill import Distill
from panther.fitness import Fitness
from panther.model import Model
from panther.model.model import DATA
from panther.presets import Budget, Presets
from panther.qat import BlockAP
from panther.search import Search
from panther.verdict import GSM8K, Verdict, Winner


def step(message: str) -> None:
    print(f"panther: {message}", flush=True)


def contest(text: str) -> tuple[str, int]:
    """"14b@5" → ("14b", 5)."""
    found = re.fullmatch(r"([0-9.]+b)@([0-9]+)", text.lower())
    if not found:
        raise argparse.ArgumentTypeError(f"write a contest as SIZE@BUDGET, like 14b@5, not {text!r}")
    return found[1], int(found[2])


def fetch(size: str) -> None:
    """Everything that needs the network: the dataset and the model's weights."""
    step("dataset")
    Dataset().build()
    step(f"downloading {size}")
    Model.load(size).download()


def run(size: str, budget_gib: int, qat: int) -> dict:
    """One contest, start to verdict, without the network."""
    # torch's CUDA build must match the server's driver; without a GPU, parts would quietly run on the CPU
    assert torch.cuda.is_available(), "no GPU visible to torch: check the driver and torch's CUDA build"
    m = Model.load(size)
    name = f"{m.name}@{budget_gib}"
    dataset = Dataset()
    step(f"{name}: the model (16-bit original, importance matrix, type copies)")
    m.build(dataset.web_distill)
    sets = dataset.for_model(m)
    budget = Budget(m, budget_gib)

    step(f"{name}: the preset")
    preset = Presets(m).best(budget, Fitness(m, sets.winner.text, sets.winner.questions))
    step(f"{name}: the search (competitor {preset['preset']}{', unfair' if preset['unfair'] else ''})")
    search = Search(m, budget, Fitness(m, sets.ga.text, sets.ga.questions), Path(preset["file"]))
    finalists = search.run()

    extra = ()
    folder = DATA / "contests" / name
    if qat and not (folder / "winner.json").exists():  # the best finalist trained by Block-AP, then distilled: a
        # 7th file for the winner (once the winner is chosen its file may be gone: not trained again)
        best, trained, distilled = (folder / f for f in ("finalist-0.gguf", "qat.gguf", "qat-distilled.gguf"))
        if not distilled.exists():
            step(f"{name}: QAT on {qat} samples")
            folder.mkdir(parents=True, exist_ok=True)
            if not best.exists():  # the winner's step reuses it
                search.genes.build(finalists[0]["genome"], best.with_name(best.name + ".part"))
                os.replace(best.with_name(best.name + ".part"), best)
            BlockAP(m, sets.qat_tokens[:qat], sets.qat_validation).run(best, trained)
            Distill(m, sets.distill).run(trained, distilled)
            trained.unlink()
        extra = (distilled,)

    step(f"{name}: the winner (finalists built and distilled, scored on the winner set)")
    winner = Winner(name, search.genes, Distill(m, sets.distill),
                    Fitness(m, sets.winner.text, sets.winner.questions)).run(finalists, preset, extra)
    step(f"{name}: the verdict")
    fitness = Fitness(m, sets.verdict.text, sets.verdict.questions)
    result = Verdict(fitness, GSM8K(m, sets.gsm8k_questions, sets.gsm8k_examples, fitness.resources)).run(
        name, Path(winner["file"]), preset)
    low, high = result["interval"]
    step(f"{name}: {result['verdict']}: relative {result['relative']:.1f} (95 % interval {low:.1f}–{high:.1f}; "
         f"100 = {preset['preset']}'s damage){' — the preset did not fit: unfair' if result['unfair'] else ''}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="panther", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="the dataset and a model's weights (needs the network)").add_argument(
        "size", help="a model by size, like 14b")
    run_parser = commands.add_parser("run", help="one contest, start to verdict (no network)")
    run_parser.add_argument("contest", type=contest, help="SIZE@BUDGET, like 14b@5 (budget in GiB)")
    run_parser.add_argument("--qat", type=int, default=0, metavar="N",
                            help="also a QAT finalist, trained on N samples (0: none)")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.size)
    else:
        run(*args.contest, args.qat)
