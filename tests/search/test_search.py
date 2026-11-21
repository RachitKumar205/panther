"""Genomes and the search, on the tiny model's real type copies (built once for this file)."""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from pathlib import Path

import gguf
import numpy as np
import pytest

from panther.fitness import Fitness, Score
from panther.fitness.resources import Resources
from panther.model import TYPES, Model
from panther.model import model as model_module
from panther.model.model import NEEDS_IMPORTANCE
from panther.presets import Budget
from panther.presets import presets as presets_module
from panther.presets.budget import GIB
from panther.search import Genes, Search
from panther.search import search as search_module
from panther.search.genes import BOTTOM, ROLES, role
from tests.fitness.questions import ga_set
from tests.model.tiny import LAYERS, fake_hessians, needs_llama, needs_perplexity, write_imatrix, write_original

pytestmark = needs_llama


@pytest.fixture(scope="module")
def m(tmp_path_factory: pytest.TempPathFactory):
    """The tiny model with its original, importance matrix, 13 type copies and errors.json."""
    data = tmp_path_factory.mktemp("data")
    with pytest.MonkeyPatch.context() as patch:
        for module in (model_module, presets_module, search_module):
            patch.setattr(module, "DATA", data)
        patch.setattr(Model, "require_disk", lambda self, need: None)
        patch.setattr(Model, "hessians", fake_hessians)
        patch.setattr(Model, "tokens", lambda self, text: np.random.default_rng(0).integers(0, 200, 5000))
        m = Model.load("1.5b")
        m.dir.mkdir(parents=True)
        write_original(m.original)
        write_imatrix(m.original, m.imatrix)
        m.make_type_copies(Path("unused"))
        yield m


def budget_around(m: Model, file_bytes: int) -> Budget:
    """A budget whose window is centred on a file size, and wide enough for the tiny model's steps: the real KV cache
    and overhead are replaced by 2 MB, so the window (1.5 % of RAM) is ~40 KB."""
    budget = Budget(m, 1)
    budget.kv_cache, budget.overhead = 0, 2_000_000
    budget.gib = (file_bytes + budget.overhead) / ((BOTTOM + 1.005) / 2) / GIB
    return budget


def uniform(genes: Genes, type_: str) -> dict[str, str]:
    return genes.like(dict.fromkeys(genes.order, type_))


@pytest.fixture
def genes(m: Model) -> Genes:
    """A budget centred on the all-Q3_K genome's size."""
    any_budget = Genes(m, Budget(m, 1))
    return Genes(m, budget_around(m, any_budget.size(uniform(any_budget, "Q3_K"))))


# --- genomes ---

def test_matrices_in_the_order_the_model_runs_them(genes: Genes) -> None:
    assert genes.order[0] == "token_embd.weight" and genes.order[-1] == "output.weight"
    assert [len(u) for u in genes.units] == [1] + [7] * LAYERS + [1]
    assert all(m.startswith(f"blk.{i}.") for i, unit in enumerate(genes.units[1:-1]) for m in unit)


def test_the_word_table_never_takes_a_type_needing_importance_data(genes: Genes) -> None:
    assert not set(genes.allowed["token_embd.weight"]) & NEEDS_IMPORTANCE
    assert genes.allowed["output.weight"] == list(TYPES)


def test_a_layout_maps_to_the_nearest_type_each_matrix_may_take(genes: Genes) -> None:
    g = genes.like(dict.fromkeys(genes.order, "IQ3_S") | {"token_embd.weight": "IQ2_XS"})
    assert g["blk.0.attn_q.weight"] == "Q3_K"  # IQ3_S is not one of ours: Q3_K is the same size
    assert g["token_embd.weight"] == "IQ2_S"  # the nearest the word table may take


def test_the_predicted_size_is_the_built_files_size(genes: Genes, tmp_path: Path) -> None:
    rng = random.Random(0)
    mixed = {m: rng.choice(genes.allowed[m]) for m in genes.order}
    for genome in (uniform(genes, "Q8_0"), uniform(genes, "IQ1_S"), mixed):
        assert genes.build(genome, tmp_path / "g.gguf").stat().st_size == genes.size(genome)


def test_a_built_file_holds_each_matrix_from_its_type_copy_and_the_rest_from_the_original(m, genes, tmp_path) -> None:
    rng = random.Random(1)
    genome = {x: rng.choice(genes.allowed[x]) for x in genes.order}
    built = {t.name: t for t in gguf.GGUFReader(genes.build(genome, tmp_path / "g.gguf")).tensors}
    original = {t.name: t for t in gguf.GGUFReader(m.original).tensors}
    assert list(built) == list(original)
    for name, t in built.items():
        source = {x.name: x for x in gguf.GGUFReader(m.copy(genome[name])).tensors}[name] if name in genome \
            else original[name]
        assert t.tensor_type == source.tensor_type and np.array_equal(t.data, source.data)


@needs_perplexity
def test_llama_cpp_runs_a_built_file_like_the_type_copy_it_equals(m: Model, genes: Genes, tmp_path: Path) -> None:
    fitness = Fitness(m, *ga_set(tmp_path / "ga"))
    assert fitness.kl(genes.build(uniform(genes, "Q4_K"), tmp_path / "g.gguf")) == fitness.kl(m.copy("Q4_K"))


def test_the_window_is_0_99_to_1_005_of_the_budget(genes: Genes) -> None:
    b = genes.budget
    g = uniform(genes, "Q3_K")
    top = 1.005 * b.gib * GIB - b.overhead - b.kv_cache
    bottom = BOTTOM * b.gib * GIB - b.overhead - b.kv_cache
    genes.rest += int(top) - genes.size(g)
    assert genes.fits(g)
    genes.rest += 1
    assert not genes.fits(g)
    genes.rest += int(bottom) - genes.size(g) - 1
    assert not genes.fits(g)


def test_repair_brings_too_big_and_too_small_genomes_into_the_window(genes: Genes) -> None:
    for start in (uniform(genes, "Q8_0"), uniform(genes, "IQ1_S")):
        g = genes.repair(start, random.Random(0))
        assert genes.fits(g) and all(g[m] in genes.allowed[m] for m in genes.order)
        assert g == genes.repair(start, random.Random(0))  # seeded: the same


def test_repair_steps_by_the_error_table(genes: Genes) -> None:
    precious = "blk.1.ffn_down.weight"
    for t in genes.allowed[precious]:  # stepping this matrix down costs far more than any other
        genes.errors[precious][t] = 1e9 * (len(TYPES) - TYPES.index(t))
    g = genes.repair(uniform(genes, "Q8_0"), random.Random(0))
    assert g[precious] == "Q8_0"


def test_repair_fails_loudly_when_the_window_cannot_be_reached(m: Model) -> None:
    genes = Genes(m, budget_around(m, 10))  # smaller than every genome
    with pytest.raises(AssertionError, match="window"):
        genes.repair(uniform(genes, "Q8_0"), random.Random(0))


def test_mutation_moves_one_to_three_matrices_by_at_most_two_types(genes: Genes) -> None:
    g = uniform(genes, "Q4_K")
    for seed in range(50):
        child = genes.mutate(g, random.Random(seed))
        moved = [m for m in genes.order if child[m] != g[m]]
        assert len(moved) <= 3
        assert all(abs(genes.allowed[m].index(child[m]) - genes.allowed[m].index(g[m])) <= 2 for m in moved)


def test_slicing_takes_one_run_of_whole_units_from_the_second_parent(genes: Genes) -> None:
    a, b = uniform(genes, "Q4_K"), uniform(genes, "Q2_K")
    for seed in range(50):
        child = genes.slice(a, b, random.Random(seed))
        from_b = [all(child[m] == b[m] for m in unit) for unit in genes.units]
        assert all(from_b[i] or all(child[m] == a[m] for m in unit) for i, unit in enumerate(genes.units))
        runs = "".join("b" if x else "a" for x in from_b)
        assert "b" in runs and runs.strip("a").count("a") == 0  # one contiguous run


# --- the search ---

class SumOfErrors:
    """Stands in for Fitness: a file's score is the sum of its matrices' output errors (lower is better). Counts its
    calls, the most at once, and can stop the run at a given call."""

    def __init__(self, m: Model, stop_after: int | None = None) -> None:
        self.errors, self.stop_after = json.loads(m.errors.read_text()), stop_after
        self.calls = self.running = self.most_at_once = 0
        self.lock, self.resources = threading.Lock(), Resources()

    def score(self, file: Path) -> Score:
        with self.lock:
            self.calls += 1
            call = self.calls
            self.running += 1
            self.most_at_once = max(self.most_at_once, self.running)
        time.sleep(0.01)
        with self.lock:
            self.running -= 1
        if call == self.stop_after:
            raise KeyboardInterrupt
        types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(file).tensors if len(t.shape) == 2}
        s = sum(self.errors[m].get(t, 1.0) for m, t in types.items())
        return Score(s, s, 0.0, 100.0, {})


def search(m: Model, genes: Genes, fitness, seed: int = 0) -> Search:
    s = Search(m, genes.budget, fitness, m.copy("Q4_K"), seed)
    s.genes, s.dir = genes, s.dir.parent / f"{s.dir.name}-{uuid.uuid4().hex}"  # a record folder of its own
    s.record, s.scratch = s.dir / "search.json", s.dir / "scratch"
    return s


def test_the_search_returns_the_best_three_distinct_genomes_all_in_the_window(m: Model, genes: Genes) -> None:
    s = search(m, genes, SumOfErrors(m))
    finalists = s.run()
    scored = json.loads(s.record.read_text())["scored"]
    assert len(finalists) == 3 and len({s.key(f["genome"]) for f in finalists}) == 3
    assert [f["score"]["score"] for f in finalists] == sorted(x["score"]["score"] for x in scored)[:3]
    assert all(genes.fits(x["genome"]) for x in scored)
    assert len({s.key(x["genome"]) for x in scored}) == len(scored)  # nothing scored twice
    assert len(scored) <= 70
    preset = json.loads(s.record.read_text())["preset"]["score"]
    assert finalists[0]["relative"] == pytest.approx(100 * finalists[0]["score"]["score"] / preset)
    assert not list(s.scratch.glob("*.gguf"))  # candidate files are deleted after scoring


def test_a_round_scores_its_candidates_at_once(m: Model, genes: Genes) -> None:
    fitness = SumOfErrors(m)
    search(m, genes, fitness).run()
    assert fitness.most_at_once > 1


def test_some_children_polish_one_of_the_four_best(m: Model, genes: Genes) -> None:
    s = search(m, genes, SumOfErrors(m))
    population = [genes.repair(uniform(genes, t), random.Random(0)) for t in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K")]
    near_best = 0
    for seed in range(100):
        child = s.child(population, random.Random(seed), set())
        assert genes.fits(child)
        near_best += any(sum(child[x] != e[x] for x in genes.order) <= 3 for e in population[:4])
    assert near_best >= 20  # about 30 % are a mutation (1-3 matrices) of one of the 4 best, before repair


def test_the_solver_finds_the_lowest_summed_error_inside_the_window(genes: Genes) -> None:
    solved = genes.solve(4)
    assert len(solved) == 4 and all(genes.fits(g) for g in solved)
    for a in solved:  # each at least 3 matrices apart from the others
        assert all(sum(a[x] != b[x] for x in genes.order) >= 3 for b in solved if b is not a)

    def summed(g: dict[str, str]) -> float:
        return sum(genes.errors[x][g[x]] for x in genes.order)
    repaired = [genes.repair(uniform(genes, t), random.Random(0)) for t in TYPES]
    assert summed(solved[0]) <= min(summed(g) for g in repaired)  # the best one beats every repaired layout


def random_genomes(genes: Genes, count: int, seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    return [{x: rng.choice(genes.allowed[x]) for x in genes.order} for _ in range(count)]


def role_sums(genes: Genes, g: dict[str, str]) -> np.ndarray:
    return np.array([sum(genes.errors[x][g[x]] for x in genes.order if role(x) == r) for r in ROLES])


def test_the_fit_recovers_known_role_weights(genes: Genes) -> None:
    truth = np.array([0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 1.0, 3.0, 0.7])
    genomes = random_genomes(genes, 400)
    sums = np.array([role_sums(genes, g) for g in genomes])
    scores = list(2.0 * sums @ truth + 5.0)
    fitted = np.array(list(genes.fit(genomes, scores).values()))
    share = (sums * truth).mean(axis=0) / (sums * truth).mean(axis=0).sum()
    big = share > 0.05  # roles that move the score enough to be measured (on the tiny model, not the word table)
    assert big.sum() >= 4
    # the weights carry the score's own scale (α is fitted on the plain sum), so compare them relative to each other
    # within 25 %: the pull toward 1 shrinks smaller roles' weights a little (one was 18 % off, the rest within 15 %)
    assert fitted[big] / fitted[big].mean() == pytest.approx(truth[big] / truth[big].mean(), rel=0.25)
    plain, weighted = genes.solve(1)[0], genes.solve(1, dict(zip(ROLES, fitted, strict=True)))[0]
    assert role_sums(genes, weighted) @ truth <= role_sums(genes, plain) @ truth  # the fit steers the solver right


def test_with_few_or_no_scores_the_weights_stay_at_one_and_always_positive(genes: Genes) -> None:
    assert genes.fit([], []) == dict.fromkeys(ROLES, 1.0)
    genomes = random_genomes(genes, 3)
    few = genes.fit(genomes, [float(role_sums(genes, g).sum()) for g in genomes])
    assert all(w == pytest.approx(1.0, abs=0.05) for w in few.values())  # the scores agree with the plain sum
    noisy = genes.fit(random_genomes(genes, 50, seed=1), list(np.random.default_rng(0).normal(size=50)))
    assert all(w > 0 for w in noisy.values())


def test_the_weighted_solver_keeps_away_from_given_genomes(genes: Genes) -> None:
    away = genes.solve(3)
    weights = dict.fromkeys(ROLES, 1.0) | {"ffn_down": 5.0}
    [g] = genes.solve(1, weights, away=away)
    assert genes.fits(g)
    assert all(sum(g[x] != a[x] for x in genes.order) >= 3 for a in away)
    assert genes.solve(1, weights) == genes.solve(1, weights)  # the same question, the same answer


def test_every_round_has_one_candidate_from_the_solver(m: Model, genes: Genes, monkeypatch) -> None:
    proposed = []
    original = Search.proposal

    def spy(self, scored, batch):
        found = original(self, scored, batch)
        proposed.append((len(scored), list(batch), found))  # a copy: the round's list grows afterwards
        return found
    monkeypatch.setattr(Search, "proposal", spy)
    s = search(m, genes, SumOfErrors(m))
    s.run()
    scored = [x["genome"] for x in json.loads(s.record.read_text())["scored"]]
    assert proposed and all(len(found) <= 1 for _, _, found in proposed)
    for count, batch, found in proposed:
        if found:
            assert scored[count + len(batch)] == found[0]  # recorded right after the round's children
            assert all(sum(found[0][x] != g[x] for x in genes.order) >= 3 for g in scored[:count] + batch)


def test_the_solver_gives_nothing_when_no_genome_fits(m: Model) -> None:
    assert Genes(m, budget_around(m, 10)).solve(4) == []


class DistanceTo:
    """Stands in for Fitness with a score no starting candidate is good at: how many matrices differ from a hidden
    target genome (a random mix inside the window). Only evolution can get close to it."""

    def __init__(self, target: dict[str, str]) -> None:
        self.target, self.resources = target, Resources()

    def score(self, file: Path) -> Score:
        types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(file).tensors if len(t.shape) == 2}
        s = float(sum(types[m] != t for m, t in self.target.items()))
        return Score(s, s, 0.0, 100.0, {})


def test_evolution_finds_better_genomes_than_the_starting_candidates(m, genes, monkeypatch) -> None:
    monkeypatch.setattr(search_module, "AT_MOST", 60)  # room to evolve (measured: the best distance falls from
    monkeypatch.setattr(search_module, "PATIENCE", 60)  # 10 among the starts to 7 by 50 candidates, 3 by 200)
    rng = random.Random(7)
    target = genes.repair({x: rng.choice(genes.allowed[x]) for x in genes.order}, rng)
    s = search(m, genes, DistanceTo(target))
    s.run()
    scored = [x["score"]["score"] for x in json.loads(s.record.read_text())["scored"]]
    assert len(scored) == 60
    assert min(scored[35:]) < min(scored[:35])  # later children beat every start (at most 35 starts)


def test_the_same_seed_gives_the_same_search(m: Model, genes: Genes) -> None:
    runs = []
    for _ in range(2):
        s = search(m, genes, SumOfErrors(m))
        s.run()
        runs.append([x["genome"] for x in json.loads(s.record.read_text())["scored"]])
    assert runs[0] == runs[1]


def test_a_stopped_search_continues_where_it_stopped(m: Model, genes: Genes) -> None:
    whole = search(m, genes, SumOfErrors(m))
    whole.run()
    # call 1 scores the preset; rounds of 8: calls 2-9 and 10-17 are recorded, the round of call 25 is lost
    stopped = search(m, genes, SumOfErrors(m, stop_after=25))
    with pytest.raises(KeyboardInterrupt):
        stopped.run()

    def scored(s: Search) -> list[dict]:
        return [x["genome"] for x in json.loads(s.record.read_text())["scored"]]
    assert len(scored(stopped)) == 16
    fitness = SumOfErrors(m)
    stopped.fitness = fitness
    stopped.run()
    assert scored(stopped) == scored(whole)
    assert fitness.calls == len(scored(whole)) - 16  # neither the preset nor the first 16 were scored again


def test_a_finished_search_is_not_run_again(m: Model, genes: Genes) -> None:
    s = search(m, genes, SumOfErrors(m))
    first = s.run()
    s.fitness = SumOfErrors(m, stop_after=1)  # would stop at its first call
    assert s.run() == first
