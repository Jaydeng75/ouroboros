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
import re
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


def _compositional(rng: random.Random) -> Example:
    values = [rng.randint(-12, 35) for _ in range(rng.randint(4, 8))]
    cutoff = rng.randint(0, 15)
    scale = rng.choice([2, 3, 4])
    function = rng.choice(
        ["filter_scale_reverse", "select_transform_flip", "screen_amplify_turn"]
    )
    templates = [
        f"def {function}(values, cutoff):\n    selected = []\n    for value in values:\n        if value > cutoff:\n            selected.append(value * {scale})\n    result = []\n    for index in range(len(selected) - 1, -1, -1):\n        result.append(selected[index])\n    return result\n\n\ndata = {function}({values!r}, {cutoff})\n",
        f"def count_and_total_above(values, cutoff):\n    count = 0\n    total = 0\n    for value in values:\n        transformed = value + {scale}\n        if transformed > cutoff:\n            count += 1\n            total += transformed\n    return count, total\n\n\ndata = count_and_total_above({values!r}, {cutoff})\n",
        "def scale_item_counts(items, factor):\n    counts = {}\n    for item in items:\n        counts[item] = counts.get(item, 0) + 1\n    scaled = {}\n    for key, value in counts.items():\n        scaled[key] = value * factor\n    return scaled\n\n\ndata = scale_item_counts(['red', 'blue', 'red'], "
        + str(scale)
        + ")\n",
        f"def flatten_grid(grid):\n    result = []\n    for row in grid:\n        for value in row:\n            result.append(value)\n    return result\n\n\ndata = flatten_grid([[1, 2], [3, 4], [5, 6]])\n",
        f"def exchange_ends_and_total(values):\n    result = list(values)\n    if len(result) >= 2:\n        result[0], result[-1] = result[-1], result[0]\n    total = 0\n    for value in result:\n        total += value\n    return result, total\n\n\ndata = exchange_ends_and_total({values!r})\n",
        f"def digits_then_count(text):\n    digits = ''\n    for character in text:\n        if character.isdigit():\n            digits += character\n    counts = {{}}\n    for character in digits:\n        counts[character] = counts.get(character, 0) + 1\n    return digits, counts\n\n\ndata = digits_then_count('a12b{values[0]}c2')\n",
    ]
    return Example(
        rng.choice(templates),
        "compositional",
        "synthetic-self-authored-v2",
    )


def _multi_function(rng: random.Random) -> Example:
    values = [rng.randint(1, 15) for _ in range(rng.randint(3, 6))]
    templates = [
        f"def square_value(value):\n    return value * value\n\n\ndef sum_squares(values):\n    total = 0\n    for value in values:\n        total += square_value(value)\n    return total\n\n\ndata = sum_squares({values!r})\n",
        "def repeat_character(character, times):\n    result = ''\n    for _ in range(times):\n        result += character\n    return result\n\n\ndef build_marks(count):\n    return repeat_character('#', count) + repeat_character('.', count)\n\n\ndata = build_marks(3)\n",
        f"def factorial_value(number):\n    result = 1\n    for factor in range(2, number + 1):\n        result *= factor\n    return result\n\n\ndef factorial_list(values):\n    result = []\n    for value in values:\n        result.append(factorial_value(value))\n    return result\n\n\ndata = factorial_list({values!r})\n",
        f"def keep_text_digits(text):\n    result = ''\n    for character in text:\n        if character.isdigit():\n            result += character\n    return result\n\n\ndef digit_total(text):\n    total = 0\n    for character in keep_text_digits(text):\n        total += int(character)\n    return total\n\n\ndata = digit_total('room42code7')\n",
    ]
    return Example(
        rng.choice(templates),
        "multi_function",
        "synthetic-self-authored-v2",
    )


def _completion(name: str, params: str, doc: str, body: list[str]) -> str:
    lines = [f"def {name}({params}):", f'    """{doc}"""']
    lines.extend(("    " + line) if line else "" for line in body)
    return "\n".join(lines) + "\n"


def _out_name(rng: random.Random) -> str:
    return rng.choice(["result", "output", "collected", "answer"])


def _v4_copy(rng: random.Random) -> Example:
    noun = rng.choice(["values", "items", "numbers", "entries"])
    each = {"values": "value", "items": "item", "numbers": "number", "entries": "entry"}[noun]
    name = rng.choice(["copy", "clone", "duplicate", "replicate"]) + "_" + noun
    out = _out_name(rng)
    idiom = rng.randrange(4)
    if idiom == 0:
        body = [f"{out} = []", f"for {each} in {noun}:", f"    {out}.append({each})", f"return {out}"]
    elif idiom == 1:
        body = [f"return list({noun})"]
    elif idiom == 2:
        body = [f"return [{each} for {each} in {noun}]"]
    else:
        body = [f"{out} = []", f"for index in range(len({noun})):", f"    {out}.append({noun}[index])", f"return {out}"]
    return Example(
        _completion(name, noun, "Return a new list containing the same values in the same sequence.", body),
        "completion_copy",
        "synthetic-self-authored-v4",
    )


def _v4_transform(rng: random.Random) -> Example:
    noun = rng.choice(["values", "items", "numbers", "entries"])
    each = {"values": "value", "items": "item", "numbers": "number", "entries": "entry"}[noun]
    out = _out_name(rng)
    if rng.random() < 0.5:
        factor = rng.randint(2, 6)
        name = rng.choice(["scale", "multiply", "amplify", "expand"]) + "_" + noun
        doc = f"Return a new list where every value is multiplied by {factor}."
        expr = f"{each} * {factor}"
    else:
        factor = rng.randint(1, 9)
        name = rng.choice(["increase", "raise", "boost", "grow"]) + "_" + noun
        doc = f"Return a new list where {factor} is added to every value."
        expr = f"{each} + {factor}"
    if rng.random() < 0.5:
        body = [f"{out} = []", f"for {each} in {noun}:", f"    {out}.append({expr})", f"return {out}"]
    else:
        body = [f"return [{expr} for {each} in {noun}]"]
    return Example(_completion(name, noun, doc, body), "completion_transform", "synthetic-self-authored-v4")


def _v4_filter(rng: random.Random) -> Example:
    noun = rng.choice(["values", "items", "numbers", "entries"])
    each = {"values": "value", "items": "item", "numbers": "number", "entries": "entry"}[noun]
    mode = rng.randrange(4)
    if mode == 0:
        name = rng.choice(["keep_above", "retain_over", "hold_greater", "select_above"])
        doc = "Return values strictly greater than cutoff, preserving sequence."
        params, condition = f"{noun}, cutoff", f"{each} > cutoff"
    elif mode == 3:
        name = rng.choice(["keep_below", "retain_under", "hold_lower", "select_below"])
        doc = "Return values strictly less than cutoff, preserving sequence."
        params, condition = f"{noun}, cutoff", f"{each} < cutoff"
    elif mode == 1:
        name = rng.choice(["keep_positive", "retain_positive", "select_positive"])
        doc = "Return the positive values, preserving sequence."
        params, condition = noun, f"{each} > 0"
    else:
        name = rng.choice(["keep_even", "retain_even", "select_even"])
        doc = "Return the even values, preserving sequence."
        params, condition = noun, f"{each} % 2 == 0"
    out = _out_name(rng)
    if rng.random() < 0.5:
        body = [f"{out} = []", f"for {each} in {noun}:", f"    if {condition}:", f"        {out}.append({each})", f"return {out}"]
    else:
        body = [f"return [{each} for {each} in {noun} if {condition}]"]
    return Example(_completion(name, params, doc, body), "completion_filter", "synthetic-self-authored-v4")


def _v4_aggregate(rng: random.Random) -> Example:
    noun = rng.choice(["values", "items", "numbers", "entries"])
    each = {"values": "value", "items": "item", "numbers": "number", "entries": "entry"}[noun]
    acc = rng.choice(["total", "amount", "overall", "running_total"])
    tally = rng.choice(["count", "tally", "seen", "matches"])
    mode = rng.randrange(6)
    if mode == 4:
        name = rng.choice(["product_values", "multiply_numbers", "product_of_entries"])
        doc, body = "Return the product of all values.", [f"{acc} = 1", f"for {each} in {noun}:", f"    {acc} *= {each}", f"return {acc}"]
        return Example(_completion(name, noun, doc, body), "completion_aggregate", "synthetic-self-authored-v4")
    if mode == 5:
        name = rng.choice(["count_above", "tally_above", "count_over"])
        doc, body = "Return how many values are greater than cutoff.", [f"{tally} = 0", f"for {each} in {noun}:", f"    if {each} > cutoff:", f"        {tally} += 1", f"return {tally}"]
        return Example(_completion(name, f"{noun}, cutoff", doc, body), "completion_aggregate", "synthetic-self-authored-v4")
    if mode == 0:
        name = rng.choice(["total_values", "sum_numbers", "add_entries", "total_items"])
        doc, body = "Return the sum of all values.", [f"{acc} = 0", f"for {each} in {noun}:", f"    {acc} += {each}", f"return {acc}"]
    elif mode == 1:
        name = rng.choice(["count_even", "count_even_values", "tally_even"])
        doc, body = "Return how many values are even.", [f"{tally} = 0", f"for {each} in {noun}:", f"    if {each} % 2 == 0:", f"        {tally} += 1", f"return {tally}"]
    elif mode == 2:
        name = rng.choice(["count_positive", "tally_positive", "count_above_zero"])
        doc, body = "Return how many values are positive.", [f"{tally} = 0", f"for {each} in {noun}:", f"    if {each} > 0:", f"        {tally} += 1", f"return {tally}"]
    else:
        name = rng.choice(["square_total", "sum_of_squares", "squared_sum"])
        doc, body = "Return the sum of the squares of the values.", [f"{acc} = 0", f"for {each} in {noun}:", f"    {acc} += {each} * {each}", f"return {acc}"]
    return Example(_completion(name, noun, doc, body), "completion_aggregate", "synthetic-self-authored-v4")


def _v4_string(rng: random.Random) -> Example:
    text_name = rng.choice(["text", "word", "phrase"])
    char_name = rng.choice(["character", "letter", "symbol"])
    out = _out_name(rng)
    mode = rng.randrange(7)
    if mode == 5:
        name = rng.choice(["count_digits", "tally_digits", "digit_count"])
        doc = "Return how many digit characters appear in the text."
        tally = rng.choice(["count", "tally", "seen"])
        body = [f"{tally} = 0", f"for {char_name} in {text_name}:", f"    if {char_name}.isdigit():", f"        {tally} += 1", f"return {tally}"]
        return Example(_completion(name, text_name, doc, body), "completion_string", "synthetic-self-authored-v4")
    if mode == 6:
        name = rng.choice(["make_upper", "upper_text", "shout_text"])
        doc = "Return the text in uppercase letters."
        body = [f"return {text_name}.upper()"]
        return Example(_completion(name, text_name, doc, body), "completion_string", "synthetic-self-authored-v4")
    if mode == 0:
        name = rng.choice(["repeat_characters", "double_letters", "echo_characters"])
        factor = rng.choice([2, 3])
        doc = f"Return the text with every character repeated {factor} times."
        body = [f"{out} = ''", f"for {char_name} in {text_name}:", f"    {out} += {char_name} * {factor}", f"return {out}"]
        params = text_name
    elif mode == 1:
        name = rng.choice(["count_vowels", "tally_vowels", "vowel_count"])
        doc = "Return how many vowels appear in the text."
        tally = rng.choice(["count", "tally", "seen"])
        body = [f"{tally} = 0", f"for {char_name} in {text_name}.lower():", f"    if {char_name} in 'aeiou':", f"        {tally} += 1", f"return {tally}"]
        params = text_name
    elif mode == 2:
        name = rng.choice(["extract_digits", "keep_digits", "digits_only"])
        doc = "Return only the digit characters from the text."
        body = [f"{out} = ''", f"for {char_name} in {text_name}:", f"    if {char_name}.isdigit():", f"        {out} += {char_name}", f"return {out}"]
        params = text_name
    elif mode == 3:
        name = rng.choice(["reverse_text", "flip_text", "backward_text"])
        doc = "Return the text in reverse sequence."
        body = [f"{out} = ''", f"for {char_name} in {text_name}:", f"    {out} = {char_name} + {out}", f"return {out}"]
        params = text_name
    else:
        name = rng.choice(["count_character", "tally_character", "count_letter"])
        doc = "Return how many times the target character appears."
        tally = rng.choice(["count", "tally", "seen"])
        body = [f"{tally} = 0", f"for {char_name} in {text_name}:", f"    if {char_name} == target:", f"        {tally} += 1", f"return {tally}"]
        params = f"{text_name}, target"
    return Example(_completion(name, params, doc, body), "completion_string", "synthetic-self-authored-v4")


def _v4_dict(rng: random.Random) -> Example:
    mode = rng.randrange(4)
    counts_name = rng.choice(["counts", "tally", "totals", "frequencies"])
    if mode == 3:
        name = rng.choice(["dictionary_keys", "keys_of_table", "entry_keys"])
        doc = "Return a list of the keys stored in the table."
        table_name = rng.choice(["table", "record", "mapping"])
        body = [f"return list({table_name})"]
        return Example(_completion(name, table_name, doc, body), "completion_dict", "synthetic-self-authored-v4")
    if mode == 0:
        name = rng.choice(["count_items", "tally_items", "count_entries", "tally_repeats"])
        doc = "Return a dictionary counting how often each item appears."
        body = [f"{counts_name} = {{}}", "for item in items:", f"    {counts_name}[item] = {counts_name}.get(item, 0) + 1", f"return {counts_name}"]
        params = "items"
    elif mode == 1:
        name = rng.choice(["combine_counts", "merge_counts", "add_counts"])
        doc = "Return one dictionary with the counts from both dictionaries added."
        body = [f"{counts_name} = dict(left)", "for key, value in right.items():", f"    {counts_name}[key] = {counts_name}.get(key, 0) + value", f"return {counts_name}"]
        params = "left, right"
    else:
        name = rng.choice(["lookup_or_default", "fetch_or_fallback", "get_or_default"])
        doc = "Return the stored value for the key, or the fallback when absent."
        body = ["if key in values:", "    return values[key]", "return fallback"]
        params = "values, key, fallback"
    return Example(_completion(name, params, doc, body), "completion_dict", "synthetic-self-authored-v4")


def _v4_math(rng: random.Random) -> Example:
    mode = rng.randrange(7)
    if mode == 4:
        name = rng.choice(["triangle_area", "area_of_triangle"])
        doc = "Return the area of a triangle with the given base and height."
        return Example(_completion(name, "base, height", doc, ["return base * height / 2"]), "completion_math", "synthetic-self-authored-v4")
    if mode == 5:
        name = rng.choice(["cube_value", "cubed_value", "cube_of"])
        doc = "Return the cube of the number."
        return Example(_completion(name, "number", doc, ["return number * number * number"]), "completion_math", "synthetic-self-authored-v4")
    if mode == 6:
        name = rng.choice(["average_pair", "mean_of_pair", "pair_average"])
        doc = "Return the average of the two values."
        return Example(_completion(name, "left, right", doc, ["return (left + right) / 2"]), "completion_math", "synthetic-self-authored-v4")
    if mode == 0:
        name = rng.choice(["factorial_of", "factorial_value", "compute_factorial"])
        doc = "Return the factorial of a non-negative integer."
        out = _out_name(rng)
        if rng.random() < 0.5:
            body = [f"{out} = 1", "for factor in range(2, number + 1):", f"    {out} *= factor", f"return {out}"]
        else:
            body = ["if number <= 1:", "    return 1", f"return number * {name}(number - 1)"]
        params = "number"
    elif mode == 1:
        name = rng.choice(["power_value", "raise_power", "compute_power"])
        doc = "Return the base raised to the exponent."
        out = _out_name(rng)
        if rng.random() < 0.5:
            body = [f"{out} = 1", "for _ in range(exponent):", f"    {out} *= base", f"return {out}"]
        else:
            body = ["return base ** exponent"]
        params = "base, exponent"
    elif mode == 2:
        name = rng.choice(["rectangle_area", "area_of_rectangle", "box_area"])
        doc = "Return the area of a rectangle with the given sides."
        body = ["return width * height"]
        params = "width, height"
    else:
        name = rng.choice(["apply_discount", "discounted_price", "price_after_discount"])
        doc = "Return the price after subtracting the given percentage."
        body = ["discount = price * percent / 100", "return price - discount"]
        params = "price, percent"
    return Example(_completion(name, params, doc, body), "completion_math", "synthetic-self-authored-v4")


def _v4_grid(rng: random.Random) -> Example:
    mode = rng.randrange(5)
    rows_name, cols_name = rng.choice([("rows", "columns"), ("height", "width")])
    acc = rng.choice(["total", "amount", "overall"])
    out = _out_name(rng)
    grid_name = rng.choice(["grid", "table", "board"])
    if mode == 0:
        name = rng.choice(["grid_total", "grid_sum", "table_total"])
        if rng.random() < 0.5:
            doc = "Return the sum of row index times column index over a zero-based grid."
            expr = "row * column"
        else:
            doc = "Return the sum of row index plus column index over a zero-based grid."
            expr = "row + column"
        body = [f"{acc} = 0", f"for row in range({rows_name}):", f"    for column in range({cols_name}):", f"        {acc} += {expr}", f"return {acc}"]
        params = f"{rows_name}, {cols_name}"
    elif mode == 4:
        name = rng.choice(["count_cells", "cell_count", "grid_cells"])
        doc = "Return how many cells the grid shape contains."
        return Example(_completion(name, f"{rows_name}, {cols_name}", doc, [f"return {rows_name} * {cols_name}"]), "completion_grid", "synthetic-self-authored-v4")
    elif mode == 1:
        name = rng.choice(["make_grid", "build_grid", "create_grid"])
        doc = "Return a grid with the given shape filled with the fill value."
        body = [f"{grid_name} = []", f"for _ in range({rows_name}):", f"    {grid_name}.append([fill] * {cols_name})", f"return {grid_name}"]
        params = f"{rows_name}, {cols_name}, fill"
    elif mode == 2:
        name = rng.choice(["flatten_grid", "flatten_table", "grid_to_list"])
        doc = "Return all grid values as one flat list, row by row."
        body = [f"{out} = []", f"for row in {grid_name}:", "    for value in row:", f"        {out}.append(value)", f"return {out}"]
        params = grid_name
    else:
        name = rng.choice(["multiplication_row", "times_row", "product_row"])
        doc = "Return the products of the number with each factor up to the limit."
        body = ["row = []", "for factor in range(1, limit + 1):", "    row.append(number * factor)", "return row"]
        params = "number, limit"
    return Example(_completion(name, params, doc, body), "completion_grid", "synthetic-self-authored-v4")


def _v4_positions(rng: random.Random) -> Example:
    mode = rng.randrange(6)
    out = _out_name(rng)
    if mode == 4:
        name = rng.choice(["first_half", "leading_half", "front_half"])
        doc = "Return the first half of the values."
        return Example(_completion(name, "values", doc, ["return values[:len(values) // 2]"]), "completion_positions", "synthetic-self-authored-v4")
    if mode == 5:
        name = rng.choice(["take_first", "leading_values", "first_entries"])
        doc = "Return the first count values."
        return Example(_completion(name, "values, count", doc, ["return values[:count]"]), "completion_positions", "synthetic-self-authored-v4")
    if mode == 0:
        name = rng.choice(["exchange_positions", "swap_positions", "trade_positions"])
        doc = "Return a copy with the positions first and second exchanged."
        body = [f"{out} = list(values)", f"{out}[first], {out}[second] = {out}[second], {out}[first]", f"return {out}"]
        params = "values, first, second"
    elif mode == 1:
        name = rng.choice(["swap_ends", "exchange_ends", "trade_ends"])
        doc = "Return a copy with the first and last values exchanged."
        body = [f"{out} = list(values)", f"if len({out}) >= 2:", f"    {out}[0], {out}[-1] = {out}[-1], {out}[0]", f"return {out}"]
        params = "values"
    elif mode == 2:
        name = rng.choice(["reverse_values", "flip_values", "backward_values"])
        doc = "Return a new list with the values in reverse sequence."
        body = [f"{out} = []", "for index in range(len(values) - 1, -1, -1):", f"    {out}.append(values[index])", f"return {out}"]
        params = "values"
    else:
        name = rng.choice(["every_second", "alternate_values", "skip_values"])
        doc = "Return every second value, starting with the first."
        body = [f"{out} = []", "for index in range(0, len(values), 2):", f"    {out}.append(values[index])", f"return {out}"]
        params = "values"
    return Example(_completion(name, params, doc, body), "completion_positions", "synthetic-self-authored-v4")


def _v4_conditional(rng: random.Random) -> Example:
    mode = rng.randrange(6)
    if mode == 4:
        name = rng.choice(["is_positive", "check_positive", "number_is_positive"])
        doc = "Return True when the number is positive."
        return Example(_completion(name, "number", doc, ["return number > 0"]), "completion_conditional", "synthetic-self-authored-v4")
    if mode == 5:
        name = rng.choice(["sign_label", "number_sign", "sign_of"])
        doc = "Return a short label for the sign of the number."
        body = ["if number > 0:", "    return 'positive'", "if number < 0:", "    return 'negative'", "return 'zero'"]
        return Example(_completion(name, "number", doc, body), "completion_conditional", "synthetic-self-authored-v4")
    if mode == 0:
        name = rng.choice(["is_even", "check_even", "number_is_even"])
        doc = "Return True when the number is even."
        body = ["return number % 2 == 0"]
        params = "number"
    elif mode == 1:
        name = rng.choice(["absolute_value", "magnitude_of", "positive_value"])
        doc = "Return the number without its sign."
        body = ["if number < 0:", "    return -number", "return number"]
        params = "number"
    elif mode == 2:
        name = rng.choice(["grade_band", "score_band", "mark_band"])
        doc = "Return the letter band for the score."
        top, mid, low = rng.choice([(90, 75, 60), (85, 70, 55), (80, 65, 50)])
        body = [f"if score >= {top}:", "    return 'A'", f"if score >= {mid}:", "    return 'B'", f"if score >= {low}:", "    return 'C'", "return 'D'"]
        params = "score"
    else:
        name = rng.choice(["describe_temperature", "temperature_label", "weather_label"])
        doc = "Return a short label describing the temperature."
        freeze, mild = rng.choice([(0, 25), (0, 20), (5, 30)])
        body = [f"if celsius <= {freeze}:", "    return 'freezing'", f"if celsius < {mild}:", "    return 'mild'", "return 'warm'"]
        params = "celsius"
    return Example(_completion(name, params, doc, body), "completion_conditional", "synthetic-self-authored-v4")


_V4_BUILDERS = [
    _v4_copy,
    _v4_transform,
    _v4_filter,
    _v4_aggregate,
    _v4_string,
    _v4_dict,
    _v4_math,
    _v4_grid,
    _v4_positions,
    _v4_conditional,
]


def _generate_v4(count: int, seed: int) -> list[Example]:
    """Completion-formatted corpus: signature + docstring + body, then stop.

    Unlike v1-v3 there is no trailing example-call line (the behaviour gen2
    imitated instead of writing a body), function/variable names and body
    idioms are parameterized instead of drawn from a few fixed templates,
    and archetypes are served round-robin so no family dominates.
    """
    rng = random.Random(seed)
    examples: list[Example] = []
    seen: set[str] = set()
    attempts = 0
    comments = [
        "Work through the values one at a time.",
        "Build the answer step by step.",
        "Use a loop to process each part.",
        "Keep track of the running result.",
        "Check each value as it arrives.",
        "Collect the pieces into the answer.",
        "Handle each part in turn.",
        "Start from an empty answer.",
    ]
    while len(examples) < count and attempts < count * 40:
        attempts += 1
        example = _V4_BUILDERS[attempts % len(_V4_BUILDERS)](rng)
        if rng.random() < 0.8:
            head, _, tail = example.text.partition("\n")
            docline, _, rest = tail.partition("\n")
            text = f"{head}\n{docline}\n    # {rng.choice(comments)}\n{rest}"
            example = Example(text, example.family, example.source)
        if example.text in seen:
            continue
        compile(example.text, "<corpus-v4>", "exec")
        seen.add(example.text)
        examples.append(example)
    if len(examples) < count:
        raise RuntimeError(f"Only generated {len(examples)} unique v4 examples")
    return examples


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

_V2_GENERATORS = [_compositional, _multi_function]


_DEF_LINE = re.compile(r"^def ([A-Za-z_][A-Za-z0-9_]*)\([^\n]*\):\n")


def _insert_docstring(text: str, rng: random.Random) -> str:
    match = _DEF_LINE.match(text)
    if not match:
        return text
    phrase = " ".join(match.group(1).split("_"))
    doc = rng.choice(
        [
            f"{phrase.capitalize()}.",
            f"Perform {phrase}.",
            f"Compute {phrase}.",
        ]
    )
    return text[: match.end()] + f'    """{doc}"""\n' + text[match.end() :]


def generate_examples(
    count: int = 24000,
    seed: int = 20261101,
    version: int = 1,
) -> list[Example]:
    if count <= 0:
        raise ValueError("count must be positive")
    if version not in (1, 2, 3, 4):
        raise ValueError("version must be 1, 2, 3, or 4")
    if version == 4:
        return _generate_v4(count, seed)
    generators = _GENERATORS + (_V2_GENERATORS if version >= 2 else [])
    rng = random.Random(seed)
    examples: list[Example] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < count and attempts < count * 30:
        attempts += 1
        example = rng.choice(generators)(rng)
        if version == 3 and rng.random() < 0.75:
            updated = _insert_docstring(example.text, rng)
            if updated != example.text:
                example = Example(
                    updated, example.family, "synthetic-self-authored-v3"
                )
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


def write_corpus(
    output_dir: Path,
    count: int,
    seed: int,
    version: int = 1,
) -> dict:
    examples = generate_examples(count=count, seed=seed, version=version)
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
        "version": version,
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
    parser.add_argument("--version", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args(argv)
    print(
        write_corpus(
            Path(args.output_dir), args.count, args.seed, version=args.version
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
