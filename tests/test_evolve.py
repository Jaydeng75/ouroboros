import json

from ouroboros.evolve import compose_mutant, evolve, novelty_score, parse_mutant
from ouroboros.sandbox import Sandbox

PARENT = {
    "id": "double",
    "prompt": (
        "Write Python code defining `double(x)` that returns 2 * x. "
        "Return the code in a ```python block."
    ),
    "tests": (
        "from solution import double\n\n\n"
        "def test_double():\n    assert double(3) == 6\n"
    ),
}

MUTANT = {
    "prompt": (
        "Write Python code defining `double(x)` returning twice the input, "
        "exactly handling huge integers and negative values. "
        "Return the code in a ```python block."
    ),
    "tests": PARENT["tests"],
}


class EvolverModel:
    """Mutates via fixed JSON; solves with a controllable pass pattern."""

    def __init__(self, solve_pattern):
        self.solve_pattern = solve_pattern
        self.solve_calls = 0
        self.mutate_calls = 0

    def generate(self, prompt, model, *, temperature=0.7, timeout=120.0):
        if "Mutate this coding task" in prompt:
            self.mutate_calls += 1
            return json.dumps(MUTANT)
        self.solve_calls += 1
        if self.solve_pattern[self.solve_calls % len(self.solve_pattern)]:
            return "```python\ndef double(x):\n    return 2 * x\n```"
        return "```python\ndef double(x):\n    return x + 1\n```"


def _archive():
    return {"format_version": 1, "cells": {}}


def test_parse_mutant():
    assert parse_mutant(json.dumps(MUTANT)) == MUTANT
    assert parse_mutant("not json at all") is None
    assert parse_mutant(json.dumps({"prompt": "x"})) is None


def test_novelty_score():
    assert novelty_score("same words here", ["same words here"]) == 0.0
    assert novelty_score("alpha beta", ["gamma delta"]) == 1.0


def test_compose_mutant():
    other = dict(PARENT, id="triple")
    mutant = compose_mutant(PARENT, other)
    assert "def test_" in mutant["tests"]
    assert mutant["tests"].count("from solution import") == 2


def test_evolve_keeps_learnable_mutants():
    model = EvolverModel([True, False])  # 50% success: maximally learnable
    kept, archive = evolve(
        model, "fake", [PARENT], Sandbox(timeout=5.0), _archive(), samples=4
    )
    assert len(kept) >= 1
    assert all(t["success_rate"] == 0.5 for t in kept)
    assert len(archive["cells"]) == len(kept)


def test_evolve_rejects_too_easy_and_too_hard():
    easy, _ = evolve(
        EvolverModel([True]), "fake", [PARENT], Sandbox(timeout=5.0),
        _archive(), samples=4,
    )
    assert easy == []
    hard, _ = evolve(
        EvolverModel([False]), "fake", [PARENT], Sandbox(timeout=5.0),
        _archive(), samples=4,
    )
    assert hard == []


class BoundingModel:
    def __init__(self):
        self.mutate_prompts = []
        self.solve_calls = 0
        self.timeouts = []

    def generate(self, prompt, model, *, temperature=0.7, timeout=120.0):
        self.timeouts.append(timeout)
        if "Mutate this coding task" in prompt:
            self.mutate_prompts.append(prompt)
            return json.dumps(MUTANT)
        self.solve_calls += 1
        return "```python\ndef double(x):\n    return x + 1\n```"


def test_evolve_bounds_parents_and_compose_candidates():
    tasks = [
        dict(PARENT, id=f"task{i}", prompt=f"Write task number {i} now.")
        for i in range(10)
    ]
    model = BoundingModel()
    kept, _ = evolve(
        model,
        "fake",
        tasks,
        Sandbox(timeout=5.0),
        _archive(),
        samples=1,
        parent_limit=3,
        compose_limit=2,
        random_seed=7,
        gen_timeout=123.0,
    )

    assert kept == []
    assert len(model.mutate_prompts) == 9
    assert model.solve_calls == 11
    assert model.timeouts == [123.0] * 20


def test_evolve_never_mutates_held_out():
    held = dict(PARENT, id="secret", held_out=True)
    model = EvolverModel([True, False])
    kept, _ = evolve(
        model, "fake", [held], Sandbox(timeout=5.0), _archive(), samples=4
    )
    assert kept == []
    assert model.mutate_calls == 0
