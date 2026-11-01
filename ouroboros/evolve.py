"""Evolution factor: mutate tasks, keep the learnable ones, archive novelty.

A mutant survives only if the current model solves it sometimes but not
always (success rate in [min_rate, max_rate]) - too easy or impossible
teaches nothing. Survivors compete for archive cells keyed by
(parent, operator); a challenger must be more learnable (closer to a 50%
success rate) than the incumbent. Evolved tasks are written to a new pool
file so they can be sharded and trained on like seeds. Held-out tasks are
never mutated.
"""
from __future__ import annotations

import itertools
import json
import random
import re

from .harness import TextModel
from .sandbox import Sandbox, extract_python_code, tail

OPERATORS = [
    {
        "name": "add_constraint",
        "instruction": "Add one meaningful constraint (input size, edge case, "
                       "or complexity bound) that makes the task harder "
                       "without changing the required function name.",
    },
    {
        "name": "tighten_spec",
        "instruction": "Make the specification stricter: require handling "
                       "of malformed or boundary inputs the original ignored, "
                       "keeping the same function name.",
    },
    {
        "name": "demand_efficiency",
        "instruction": "Require an asymptotically better solution (for "
                       "example O(n) or O(log n)) and say so in the prompt, "
                       "keeping the same function name and return contract.",
    },
]

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_ERROR = re.compile(r"(\w*(?:Error|Exception|Timeout|Failed))\b")


def build_mutation_prompt(task: dict, operator: dict) -> str:
    return (
        "Mutate this coding task. Operator: "
        f"{operator['name']} - {operator['instruction']}\n\n"
        f"Original prompt:\n{task['prompt']}\n\n"
        f"Original tests (pytest, imports from `solution`):\n{task['tests']}\n\n"
        "Return ONLY a JSON object with keys \"prompt\" and \"tests\": "
        "the mutated task prompt and matching pytest source. The tests must "
        "import the solution from `solution` and must pass against a correct "
        "solution. No markdown, no commentary."
    )


def parse_mutant(text: str) -> dict | None:
    match = _JSON_OBJ.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    prompt, tests = data.get("prompt"), data.get("tests")
    if not isinstance(prompt, str) or not isinstance(tests, str):
        return None
    if not prompt.strip() or "def test_" not in tests:
        return None
    try:
        compile(tests, "<mutant-tests>", "exec")
    except SyntaxError:
        return None
    return {"prompt": prompt, "tests": tests}


def compose_mutant(task_a: dict, task_b: dict) -> dict:
    """Deterministic operator: one task requiring both seed solutions."""
    prompt = (
        f"{task_a['prompt']}\n\nAdditionally:\n{task_b['prompt']}\n"
        "Define both functions in the same module."
    )
    return {"prompt": prompt, "tests": task_a["tests"] + "\n" + task_b["tests"]}


def _words(prompt: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", prompt.lower()))


def novelty_score(prompt: str, existing_prompts: list[str]) -> float:
    """1 - max Jaccard similarity to existing prompts (1.0 = wholly new)."""
    words = _words(prompt)
    if not words or not existing_prompts:
        return 1.0
    best = 0.0
    for other in existing_prompts:
        other_words = _words(other)
        union = words | other_words
        if union:
            best = max(best, len(words & other_words) / len(union))
    return 1.0 - best


def failure_signature(result) -> str:
    if result.timed_out:
        return "Timeout"
    text = (result.stderr or "") + "\n" + (result.stdout or "")
    match = _ERROR.search(text)
    return match.group(1) if match else "UnknownFailure"


def evaluate_mutant(
    model_client: TextModel,
    model_name: str,
    mutant: dict,
    sandbox: Sandbox,
    *,
    samples: int = 4,
    gen_timeout: float = 300.0,
) -> tuple[float, str]:
    """Sample solutions for a mutant; return (success rate, fail signature)."""
    temperatures = (0.9, 0.5, 0.2, 0.0)
    passes = 0
    signature = ""
    for i in range(samples):
        raw = model_client.generate(
            mutant["prompt"], model_name,
            temperature=temperatures[i % len(temperatures)],
            timeout=gen_timeout,
        )
        code = extract_python_code(raw)
        if code is None:
            if not signature:
                signature = "NoCode"
            continue
        result = sandbox.run(code, mutant["tests"])
        if result.passed:
            passes += 1
        elif not signature:
            signature = failure_signature(result)
    return passes / samples, signature or "Passed"


def evolve(
    model_client: TextModel,
    model_name: str,
    tasks: list[dict],
    sandbox: Sandbox,
    archive: dict,
    *,
    samples: int = 4,
    min_rate: float = 0.2,
    max_rate: float = 0.8,
    min_novelty: float = 0.15,
    compose: bool = True,
    parent_limit: int | None = None,
    compose_limit: int | None = None,
    random_seed: int = 0,
    gen_timeout: float = 300.0,
) -> tuple[list[dict], dict]:
    """One generation. Returns (kept mutant tasks, updated archive)."""
    if parent_limit is not None and parent_limit < 0:
        raise ValueError("parent_limit must be non-negative")
    if compose_limit is not None and compose_limit < 0:
        raise ValueError("compose_limit must be non-negative")

    kept: list[dict] = []
    cells: dict = archive.setdefault("cells", {})
    known_prompts = [t["prompt"] for t in tasks]
    eligible = [task for task in tasks if not task.get("held_out")]
    rng = random.Random(random_seed)

    if parent_limit is None or parent_limit >= len(eligible):
        mutation_parents = eligible
    else:
        selected = sorted(rng.sample(range(len(eligible)), parent_limit))
        mutation_parents = [eligible[index] for index in selected]

    candidates: list[dict] = []
    for task in mutation_parents:
        for operator in OPERATORS:
            raw = model_client.generate(
                build_mutation_prompt(task, operator), model_name,
                temperature=0.8, timeout=gen_timeout,
            )
            mutant = parse_mutant(raw)
            if mutant is not None:
                candidates.append(
                    {"parent": task, "mutant": mutant,
                     "op": operator["name"],
                     "id": f"{task['id']}__{operator['name']}"}
                )
    if compose:
        pairs = list(itertools.combinations(eligible, 2))
        if compose_limit is not None and compose_limit < len(pairs):
            selected_pairs = sorted(
                rng.sample(range(len(pairs)), compose_limit)
            )
            pairs = [pairs[index] for index in selected_pairs]
        for a, b in pairs:
            candidates.append(
                {"parent": a, "mutant": compose_mutant(a, b),
                 "op": "compose",
                 "id": f"{a['id']}__compose__{b['id']}"}
            )

    for candidate in candidates:
        parent, mutant, op_name = (
            candidate["parent"], candidate["mutant"], candidate["op"])
        rate, signature = evaluate_mutant(
            model_client, model_name, mutant, sandbox, samples=samples,
            gen_timeout=gen_timeout,
        )
        if not (min_rate <= rate <= max_rate):
            continue
        novelty = novelty_score(mutant["prompt"], known_prompts)
        if novelty < min_novelty:
            continue
        cell_key = f"{parent['id']}:{op_name}"
        mutant_task = {
            "id": candidate["id"],
            "prompt": mutant["prompt"],
            "tests": mutant["tests"],
            "held_out": False,
            "parent": parent["id"],
            "operator": op_name,
            "success_rate": rate,
            "failure_signature": signature,
        }
        incumbent = cells.get(cell_key)
        if incumbent is None or (
            abs(rate - 0.5) < abs(incumbent["success_rate"] - 0.5)
        ):
            cells[cell_key] = {
                "task_id": mutant_task["id"],
                "success_rate": rate,
                "novelty": round(novelty, 3),
                "failure_signature": signature,
            }
            kept.append(mutant_task)
            known_prompts.append(mutant["prompt"])
            print(f"[KEEP] {mutant_task['id']} rate={rate:.2f} "
                  f"novelty={novelty:.2f} fail={signature} "
                  f"({tail(mutant['prompt'], 60).strip()!r})")
        else:
            print(f"[skip] {mutant_task['id']} rate={rate:.2f} "
                  "loses its cell")
    archive["cells"] = cells
    return kept, archive
