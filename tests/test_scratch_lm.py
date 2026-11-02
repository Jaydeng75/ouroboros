import pytest

from ouroboros.sandbox import Sandbox
from scratch_lm.audit import audit_records, audit_text
from scratch_lm.benchmark import TASKS as BENCHMARK_TASKS
from scratch_lm.benchmark import summarize
from scratch_lm.corpus import generate_examples
from scratch_lm.tokenizer import BytePairTokenizer


def test_generated_corpus_is_unique_valid_and_leakage_controlled():
    examples = generate_examples(count=300, seed=99)
    assert len(examples) == 300
    assert len({example.text for example in examples}) == 300
    report = audit_records([example.__dict__ for example in examples])
    assert report == {"checked": 300, "accepted": 300, "rejected": []}


def test_corpus_v2_adds_compositional_examples_without_leakage():
    examples = generate_examples(count=900, seed=123, version=2)
    families = [example.family for example in examples]
    compositional = sum(
        family in {"compositional", "multi_function"} for family in families
    )
    assert compositional >= 70
    assert len({example.text for example in examples}) == 900
    report = audit_records([example.__dict__ for example in examples])
    assert report["rejected"] == []


def test_corpus_v3_is_docstring_conditioned_without_leakage():
    examples = generate_examples(count=600, seed=321, version=3)
    with_docstring = sum('"""' in example.text for example in examples)
    assert with_docstring >= 300
    assert len({example.text for example in examples}) == 600
    report = audit_records([example.__dict__ for example in examples])
    assert report["rejected"] == []
    for example in examples:
        compile(example.text, "<v3>", "exec")


def test_intermediate_benchmark_tasks_are_solvable_and_summarized():
    solutions = {
        "copy_values": "def copy_values(values):\n    result = []\n    for value in values:\n        result.append(value)\n    return result\n",
        "double_values": "def double_values(values):\n    return [value * 2 for value in values]\n",
        "keep_above": "def keep_above(values, cutoff):\n    return [value for value in values if value > cutoff]\n",
        "reverse_values": "def reverse_values(values):\n    result = list(values)\n    left = 0\n    right = len(result) - 1\n    while left < right:\n        result[left], result[right] = result[right], result[left]\n        left += 1\n        right -= 1\n    return result\n",
        "exchange_positions": "def exchange_positions(values, first, second):\n    result = list(values)\n    result[first], result[second] = result[second], result[first]\n    return result\n",
        "factorial_of": "def factorial_of(number):\n    result = 1\n    for factor in range(2, number + 1):\n        result *= factor\n    return result\n",
        "count_items": "def count_items(items):\n    counts = {}\n    for item in items:\n        counts[item] = counts.get(item, 0) + 1\n    return counts\n",
        "grid_total": "def grid_total(rows, columns):\n    total = 0\n    for row in range(rows):\n        for column in range(columns):\n            total += row * column\n    return total\n",
    }
    sandbox = Sandbox(timeout=5.0)
    assert {task["function"] for task in BENCHMARK_TASKS} == set(solutions)
    for task in BENCHMARK_TASKS:
        result = sandbox.run(solutions[task["function"]], task["tests"])
        assert result.passed, (task["id"], result.stdout, result.stderr)

    summary = summarize(
        [
            {"task_id": "a", "syntax_valid": True, "sandbox_passed": True},
            {"task_id": "a", "syntax_valid": True, "sandbox_passed": False},
            {"task_id": "b", "syntax_valid": False, "sandbox_passed": False},
        ]
    )
    assert summary == {
        "candidates": 3,
        "syntax_valid": 2,
        "sandbox_passed": 1,
        "syntax_valid_rate": pytest.approx(2 / 3),
        "pass_rate": pytest.approx(1 / 3),
    }


def test_audit_rejects_sorting_procedure_and_api():
    leaked = """
def bubble_sort(values):
    for i in range(len(values)):
        for j in range(len(values) - 1):
            if values[j] > values[j + 1]:
                values[j], values[j + 1] = values[j + 1], values[j]
    return values
"""
    kinds = {violation.kind for violation in audit_text(leaked)}
    assert "forbidden_term" in kinds
    assert "adjacent_or_self_comparison" in kinds
    assert audit_text("answer = sorted(values)\n")
    assert audit_text("import heapq\n")


def test_bpe_tokenizer_roundtrip_and_determinism():
    texts = [
        "def add(a, b):\n    return a + b\n",
        "def double(value):\n    return value * 2\n",
        "answer = add(2, 3)\n",
    ] * 20
    first = BytePairTokenizer.train(texts, vocab_size=320)
    second = BytePairTokenizer.train(texts, vocab_size=320)
    assert first.merges == second.merges
    sample = "def add(a, b):\n    return a + b\n"
    assert first.decode(first.encode(sample)) == sample
    assert first.vocab_size <= 320
    assert max(len(value) for value in first.token_bytes.values()) <= 16


def test_transformer_forward_and_generation():
    torch = pytest.importorskip("torch")
    from scratch_lm.model import ModelConfig, ScratchTransformer

    config = ModelConfig(
        vocab_size=80,
        context_length=32,
        d_model=32,
        n_layer=2,
        n_head=4,
        dropout=0.0,
    )
    model = ScratchTransformer(config)
    output = model(torch.tensor([[1, 2, 3, 4]]))
    assert output.shape == (1, 4, 80)
    generated = model.generate([1, 2], max_new_tokens=3, temperature=0)
    assert len(generated) == 5
    assert model.parameter_count() > 0
