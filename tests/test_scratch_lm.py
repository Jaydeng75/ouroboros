import pytest

from scratch_lm.audit import audit_records, audit_text
from scratch_lm.corpus import generate_examples
from scratch_lm.tokenizer import BytePairTokenizer


def test_generated_corpus_is_unique_valid_and_leakage_controlled():
    examples = generate_examples(count=300, seed=99)
    assert len(examples) == 300
    assert len({example.text for example in examples}) == 300
    report = audit_records([example.__dict__ for example in examples])
    assert report == {"checked": 300, "accepted": 300, "rejected": []}


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
