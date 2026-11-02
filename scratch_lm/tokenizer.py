"""A small byte-pair tokenizer implemented from scratch."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4
BASE_VOCAB_SIZE = BYTE_OFFSET + 256


class BytePairTokenizer:
    def __init__(self, merges: list[tuple[int, int, int]] | None = None):
        self.merges = list(merges or [])
        self.merge_ranks = {
            (left, right): rank
            for rank, (left, right, _) in enumerate(self.merges)
        }
        self.token_bytes: dict[int, bytes] = {
            BYTE_OFFSET + value: bytes([value]) for value in range(256)
        }
        for left, right, merged in self.merges:
            self.token_bytes[merged] = (
                self.token_bytes[left] + self.token_bytes[right]
            )

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 2048,
        min_pair_frequency: int = 2,
        max_token_bytes: int = 16,
    ) -> "BytePairTokenizer":
        if vocab_size <= BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must exceed {BASE_VOCAB_SIZE}")
        if max_token_bytes <= 0:
            raise ValueError("max_token_bytes must be positive")
        sequences = [
            [BYTE_OFFSET + value for value in text.encode("utf-8")]
            for text in texts
        ]
        token_lengths = {
            BYTE_OFFSET + value: 1 for value in range(256)
        }
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_locations: dict[tuple[int, int], set[int]] = {}
        for index, sequence in enumerate(sequences):
            for pair in zip(sequence, sequence[1:]):
                pair_counts[pair] += 1
                pair_locations.setdefault(pair, set()).add(index)

        merges: list[tuple[int, int, int]] = []
        next_id = BASE_VOCAB_SIZE

        while next_id < vocab_size:
            allowed = [
                (pair, frequency)
                for pair, frequency in pair_counts.items()
                if token_lengths[pair[0]] + token_lengths[pair[1]]
                <= max_token_bytes
            ]
            if not allowed:
                break
            pair, frequency = sorted(
                allowed, key=lambda item: (-item[1], item[0])
            )[0]
            if frequency < min_pair_frequency:
                break
            merges.append((pair[0], pair[1], next_id))
            token_lengths[next_id] = (
                token_lengths[pair[0]] + token_lengths[pair[1]]
            )
            affected = list(pair_locations.get(pair, set()))
            for index in affected:
                sequence = sequences[index]
                for old_pair in zip(sequence, sequence[1:]):
                    pair_counts[old_pair] -= 1
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                    locations = pair_locations.get(old_pair)
                    if locations is not None:
                        locations.discard(index)
                        if not locations:
                            del pair_locations[old_pair]
                merged_sequence = cls._merge_sequence(sequence, pair, next_id)
                sequences[index] = merged_sequence
                for new_pair in zip(merged_sequence, merged_sequence[1:]):
                    pair_counts[new_pair] += 1
                    pair_locations.setdefault(new_pair, set()).add(index)
            next_id += 1
        return cls(merges)

    @staticmethod
    def _merge_sequence(
        sequence: list[int], pair: tuple[int, int], merged: int
    ) -> list[int]:
        result: list[int] = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                result.append(merged)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    def encode(self, text: str, *, add_special: bool = True) -> list[int]:
        sequence = [BYTE_OFFSET + value for value in text.encode("utf-8")]
        while len(sequence) >= 2:
            candidates = [
                (self.merge_ranks[pair], pair)
                for pair in zip(sequence, sequence[1:])
                if pair in self.merge_ranks
            ]
            if not candidates:
                break
            _, pair = min(candidates)
            merged = BASE_VOCAB_SIZE + self.merge_ranks[pair]
            sequence = self._merge_sequence(sequence, pair, merged)
        if add_special:
            return [BOS_ID, *sequence, EOS_ID]
        return sequence

    def decode(self, token_ids: list[int], *, skip_special: bool = True) -> str:
        chunks: list[bytes] = []
        for token_id in token_ids:
            if skip_special and token_id in (PAD_ID, BOS_ID, EOS_ID):
                continue
            if token_id == UNK_ID:
                chunks.append(b"?")
            else:
                chunks.append(self.token_bytes[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "format": "scratch-bpe-v1",
                    "merges": [list(item) for item in self.merges],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BytePairTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "scratch-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        return cls([tuple(item) for item in data["merges"]])


def load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            texts.append(json.loads(line)["text"])
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the scratch BPE tokenizer")
    parser.add_argument("--input", default="scratch_data/train.jsonl")
    parser.add_argument("--output", default="scratch_data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--max-token-bytes", type=int, default=16)
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Train merges on at most this many leading texts",
    )
    args = parser.parse_args(argv)

    texts = load_texts(Path(args.input))
    if args.max_texts is not None:
        texts = texts[: args.max_texts]
    tokenizer = BytePairTokenizer.train(
        texts,
        vocab_size=args.vocab_size,
        max_token_bytes=args.max_token_bytes,
    )
    tokenizer.save(Path(args.output))
    sample = texts[0] if texts else ""
    encoded = tokenizer.encode(sample)
    print(
        {
            "texts": len(texts),
            "vocab_size": tokenizer.vocab_size,
            "sample_tokens": len(encoded),
            "roundtrip": tokenizer.decode(encoded) == sample,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
