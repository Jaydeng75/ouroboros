"""Deterministic synthetic Python corpus with sorting content excluded.

The generator is intentionally narrow: variables, arithmetic, branches,
loops, indexing, swapping positions, filtering, recursion, dictionaries,
strings, grids, and simple classes. It never asks a program to arrange a
collection by value and never demonstrates an ordering procedure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from .audit import audit_records


@dataclass(frozen=True)
class Example:
    text: str
    family: str
    source: str = "synthetic-self-authored-v1"


def _arithmetic(rng: random.Random) -> Example:
    a, b = rng.randint(2, 90), rng.randint(2, 40)
    templates = [
        f"def add_values(left, right):\n    total = left + right\n    return total\n\n\nresult = add_values({a}, {b})\n",
        f"def rectangle_area(width, height):\n    return width * height\n\n\narea = rectangle_area({a}, {b})\n",
        f"def convert_temperature(celsius):\n    return celsius * 9 / 5 + 32\n\n\nreading = convert_temperature({a})\n",
        f"def apply_discount(price, percent):\n    discount = price * percent / 100\n    return price - discount\n\n\nfinal_price = apply_discount({a * 10}, {b})\n",
        f"def combine_scores(first, second, third):\n    return first + second * 2 + third * 3\n\n\nscore = combine_scores({a}, {b}, 4)\n",
    ]
    return Example(rng.choice(templates), "arithmetic")


def _branches(rng: random.Random) -> Example:
    value = rng.randint(-30, 120)
    templates = [
        f"def is_even(number):\n    return number % 2 == 0\n\n\nflag = is_even({value})\n",
        f"def absolute_value(number):\n    if number < 0:\n        return -number\n    return number\n\n\nmagnitude = absolute_value({value})\n",
        f"def grade_band(score):\n    if score >= 90:\n        return 'A'\n    if score >= 75:\n        return 'B'\n    if score >= 60:\n        return 'C'\n    return 'D'\n\n\nband = grade_band({value})\n",
        f"def describe_temperature(celsius):\n    if celsius <= 0:\n        return 'freezing'\n    if celsius < 25:\n        return 'mild'\n    return 'warm'\n\n\nlabel = describe_temperature({value})\n",
        f"def can_register(age):\n    return age >= 18\n\n\nallowed = can_register({rng.randint(10, 30)})\n",
    ]
    return Example(rng.choice(templates), "branches")


def _loops(rng: random.Random) -> Example:
    n = rng.randint(4, 15)
    templates = [
        f"def sum_to(limit):\n    total = 0\n    for number in range(1, limit + 1):\n        total += number\n    return total\n\n\nanswer = sum_to({n})\n",
        f"def product_to(limit):\n    result = 1\n    for number in range(2, limit + 1):\n        result *= number\n    return result\n\n\nanswer = product_to({n})\n",
        f"def count_even(limit):\n    count = 0\n    for number in range(limit):\n        if number % 2 == 0:\n            count += 1\n    return count\n\n\nanswer = count_even({n * 2})\n",
        f"def repeat_text(text, times):\n    result = ''\n    for _ in range(times):\n        result += text\n    return result\n\n\nanswer = repeat_text('ab', {rng.randint(2, 6)})\n",
        f"def sum_squares(limit):\n    total = 0\n    number = 1\n    while number <= limit:\n        total += number * number\n        number += 1\n    return total\n\n\nanswer = sum_squares({n})\n",
    ]
    return Example(rng.choice(templates), "loops")


def _lists(rng: random.Random) -> Example:
    values = [rng.randint(-9, 30) for _ in range(rng.randint(3, 7))]
    threshold = rng.randint(0, 15)
    templates = [
        f"def copy_values(values):\n    copied = []\n    for value in values:\n        copied.append(value)\n    return copied\n\n\ndata = copy_values({values!r})\n",
        f"def double_values(values):\n    result = []\n    for value in values:\n        result.append(value * 2)\n    return result\n\n\ndata = double_values({values!r})\n",
        f"def keep_above(values, cutoff):\n    result = []\n    for value in values:\n        if value > cutoff:\n            result.append(value)\n    return result\n\n\ndata = keep_above({values!r}, {threshold})\n",
        f"def total_positive(values):\n    total = 0\n    for value in values:\n        if value > 0:\n            total += value\n    return total\n\n\ndata = total_positive({values!r})\n",
        f"def replace_value(values, old, new):\n    result = []\n    for value in values:\n        if value == old:\n            result.append(new)\n        else:\n            result.append(value)\n    return result\n\n\ndata = replace_value({values!r}, {values[0]}, 99)\n",
        f"def indexed_total(values):\n    total = 0\n    for index in range(len(values)):\n        total += values[index] * index\n    return total\n\n\ndata = indexed_total({values!r})\n",
    ]
    return Example(rng.choice(templates), "lists")


def _positions(rng: random.Random) -> Example:
    values = [rng.randint(1, 40) for _ in range(rng.randint(4, 7))]
    templates = [
        f"def reverse_values(values):\n    result = list(values)\n    left = 0\n    right = len(result) - 1\n    while left < right:\n        result[left], result[right] = result[right], result[left]\n        left += 1\n        right -= 1\n    return result\n\n\ndata = reverse_values({values!r})\n",
        f"def exchange_positions(values, first, second):\n    result = list(values)\n    result[first], result[second] = result[second], result[first]\n    return result\n\n\ndata = exchange_positions({values!r}, 0, 2)\n",
        f"def rotate_once(values):\n    if not values:\n        return []\n    return [values[-1]] + values[:-1]\n\n\ndata = rotate_once({values!r})\n",
        f"def swap_pair(first, second):\n    return second, first\n\n\npair = swap_pair({values[0]}, {values[1]})\n",
        f"def every_second(values):\n    result = []\n    for index in range(0, len(values), 2):\n        result.append(values[index])\n    return result\n\n\ndata = every_second({values!r})\n",
    ]
    return Example(rng.choice(templates), "positions")


def _strings(rng: random.Random) -> Example:
    words = ["code", "model", "data", "loop", "token", "python", "vector"]
    word = rng.choice(words)
    templates = [
        f"def reverse_text(text):\n    result = ''\n    for character in text:\n        result = character + result\n    return result\n\n\nanswer = reverse_text({word!r})\n",
        f"def count_vowels(text):\n    count = 0\n    for character in text.lower():\n        if character in 'aeiou':\n            count += 1\n    return count\n\n\nanswer = count_vowels({word!r})\n",
        f"def repeat_characters(text):\n    result = ''\n    for character in text:\n        result += character * 2\n    return result\n\n\nanswer = repeat_characters({word!r})\n",
        f"def extract_digits(text):\n    result = ''\n    for character in text:\n        if character.isdigit():\n            result += character\n    return result\n\n\nanswer = extract_digits('a1b{word}3')\n",
        f"def count_character(text, target):\n    count = 0\n    for character in text:\n        if character == target:\n            count += 1\n    return count\n\n\nanswer = count_character({word!r}, {word[0]!r})\n",
    ]
    return Example(rng.choice(templates), "strings")


def _recursion(rng: random.Random) -> Example:
    n = rng.randint(3, 12)
    templates = [
        f"def factorial(number):\n    if number <= 1:\n        return 1\n    return number * factorial(number - 1)\n\n\nanswer = factorial({n})\n",
        f"def countdown_total(number):\n    if number == 0:\n        return 0\n    return number + countdown_total(number - 1)\n\n\nanswer = countdown_total({n})\n",
        f"def power_value(base, exponent):\n    if exponent == 0:\n        return 1\n    return base * power_value(base, exponent - 1)\n\n\nanswer = power_value(2, {n})\n",
        f"def greatest_common_divisor(left, right):\n    if right == 0:\n        return left\n    return greatest_common_divisor(right, left % right)\n\n\nanswer = greatest_common_divisor(48, 18)\n",
    ]
    return Example(rng.choice(templates), "recursion")


def _dictionaries(rng: random.Random) -> Example:
    items = ["red", "blue", "red", "green", "blue", "red"]
    rng.shuffle(items)
    templates = [
        f"def count_items(items):\n    counts = {{}}\n    for item in items:\n        counts[item] = counts.get(item, 0) + 1\n    return counts\n\n\nanswer = count_items({items!r})\n",
        "def add_points(scores, name, points):\n    updated = dict(scores)\n    updated[name] = updated.get(name, 0) + points\n    return updated\n\n\nanswer = add_points({'ana': 3}, 'ana', 4)\n",
        "def lookup_or_default(values, key, fallback):\n    if key in values:\n        return values[key]\n    return fallback\n\n\nanswer = lookup_or_default({'mode': 'test'}, 'mode', 'live')\n",
        "def combine_counts(left, right):\n    combined = dict(left)\n    for key, value in right.items():\n        combined[key] = combined.get(key, 0) + value\n    return combined\n\n\nanswer = combine_counts({'a': 1}, {'a': 2, 'b': 5})\n",
    ]
    return Example(rng.choice(templates), "dictionaries")


def _nested(rng: random.Random) -> Example:
    rows, cols = rng.randint(2, 5), rng.randint(2, 5)
    templates = [
        f"def grid_total(rows, columns):\n    total = 0\n    for row in range(rows):\n        for column in range(columns):\n            total += row * column\n    return total\n\n\nanswer = grid_total({rows}, {cols})\n",
        f"def make_grid(rows, columns, fill):\n    grid = []\n    for _ in range(rows):\n        grid.append([fill] * columns)\n    return grid\n\n\nanswer = make_grid({rows}, {cols}, 0)\n",
        f"def multiplication_row(number, limit):\n    row = []\n    for factor in range(1, limit + 1):\n        row.append(number * factor)\n    return row\n\n\nanswer = multiplication_row({rows + 2}, {cols + 3})\n",
        f"def count_equal_pairs(values):\n    count = 0\n    for left in range(len(values)):\n        for right in range(left + 1, len(values)):\n            if values[left] == values[right]:\n                count += 1\n    return count\n\n\nanswer = count_equal_pairs([1, 2, 1, 3, 2])\n",
    ]
    return Example(rng.choice(templates), "nested_loops")


def _classes(rng: random.Random) -> Example:
    amount = rng.randint(5, 80)
    templates = [
        f"class Counter:\n    def __init__(self):\n        self.value = 0\n\n    def add(self, amount):\n        self.value += amount\n        return self.value\n\n\ncounter = Counter()\nanswer = counter.add({amount})\n",
        f"class Rectangle:\n    def __init__(self, width, height):\n        self.width = width\n        self.height = height\n\n    def area(self):\n        return self.width * self.height\n\n\nshape = Rectangle(4, 5)\nanswer = shape.area()\n",
        f"class Wallet:\n    def __init__(self, balance):\n        self.balance = balance\n\n    def spend(self, amount):\n        if amount <= self.balance:\n            self.balance -= amount\n        return self.balance\n\n\nwallet = Wallet(100)\nanswer = wallet.spend({amount})\n",
    ]
    return Example(rng.choice(templates), "classes")


_GENERATORS = [
    _arithmetic,
    _branches,
    _loops,
    _lists,
    _positions,
    _strings,
    _recursion,
    _dictionaries,
    _nested,
    _classes,
]


def generate_examples(count: int = 24000, seed: int = 20261101) -> list[Example]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    examples: list[Example] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < count and attempts < count * 30:
        attempts += 1
        example = rng.choice(_GENERATORS)(rng)
        if example.text in seen:
            continue
        compile(example.text, "<corpus>", "exec")
        seen.add(example.text)
        examples.append(example)
    if len(examples) < count:
        raise RuntimeError(f"Only generated {len(examples)} unique examples")
    return examples


def split_name(index: int) -> str:
    bucket = index % 40
    if bucket == 0:
        return "test"
    if bucket in (1, 2):
        return "valid"
    return "train"


def write_corpus(output_dir: Path, count: int, seed: int) -> dict:
    examples = generate_examples(count=count, seed=seed)
    records = [example.__dict__ for example in examples]
    report = audit_records(records)
    if report["rejected"]:
        raise RuntimeError(f"Corpus audit rejected {len(report['rejected'])} records")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        name: (output_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "valid", "test")
    }
    counts = {"train": 0, "valid": 0, "test": 0}
    try:
        for index, example in enumerate(examples):
            name = split_name(index)
            counts[name] += 1
            handles[name].write(json.dumps(example.__dict__, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "seed": seed,
        "requested": count,
        "counts": counts,
        "audit": {"checked": report["checked"], "accepted": report["accepted"], "rejected": len(report["rejected"])},
        "sha256": hashlib.sha256(
            "".join(example.text for example in examples).encode("utf-8")
        ).hexdigest(),
        "policy": "No sorting vocabulary, ordering APIs, ordering imports, or same-list element comparisons.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the leakage-controlled corpus")
    parser.add_argument("--output-dir", default="scratch_data")
    parser.add_argument("--count", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args(argv)
    print(write_corpus(Path(args.output_dir), args.count, args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
