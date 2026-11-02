"""Train the scratch Transformer on free-tier-friendly checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

from .model import ModelConfig, ScratchTransformer
from .tokenizer import BytePairTokenizer, load_texts


def encode_stream(tokenizer: BytePairTokenizer, texts: list[str]) -> torch.Tensor:
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text))
    return torch.tensor(token_ids, dtype=torch.long)


def make_batch(
    stream: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = [
        rng.randrange(0, len(stream) - context_length - 1)
        for _ in range(batch_size)
    ]
    inputs = torch.stack([stream[start : start + context_length] for start in starts])
    targets = torch.stack(
        [stream[start + 1 : start + context_length + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: ScratchTransformer,
    stream: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
    batches: int = 20,
) -> float:
    model.eval()
    rng = random.Random(1234)
    losses = []
    for _ in range(batches):
        inputs, targets = make_batch(
            stream,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            rng=rng,
        )
        logits = model(inputs)
        losses.append(
            nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            ).item()
        )
    model.train()
    return sum(losses) / len(losses)


def preset_config(name: str, vocab_size: int) -> ModelConfig:
    if name == "smoke":
        return ModelConfig(
            vocab_size=vocab_size,
            context_length=128,
            d_model=128,
            n_layer=2,
            n_head=4,
            dropout=0.0,
        )
    if name == "base":
        return ModelConfig(
            vocab_size=vocab_size,
            context_length=256,
            d_model=512,
            n_layer=8,
            n_head=8,
            dropout=0.1,
        )
    raise ValueError(f"Unknown preset {name!r}")


def save_checkpoint(
    path: Path,
    model: ScratchTransformer,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    valid_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": model.config.to_dict(),
            "step": step,
            "valid_loss": valid_loss,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the scratch LM")
    parser.add_argument("--data-dir", default="scratch_data")
    parser.add_argument("--output-dir", default="scratch_runs/gen0")
    parser.add_argument("--preset", choices=("smoke", "base"), default="base")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_texts = load_texts(data_dir / "train.jsonl")
    valid_texts = load_texts(data_dir / "valid.jsonl")
    tokenizer_path = output_dir / "tokenizer.json"
    source_tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = BytePairTokenizer.load(tokenizer_path)
    elif source_tokenizer_path.exists():
        tokenizer = BytePairTokenizer.load(source_tokenizer_path)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = BytePairTokenizer.train(train_texts, vocab_size=2048)
        tokenizer.save(tokenizer_path)

    train_stream = encode_stream(tokenizer, train_texts)
    valid_stream = encode_stream(tokenizer, valid_texts)
    config = preset_config(args.preset, tokenizer.vocab_size)
    model = ScratchTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.1
    )
    start_step = 0
    checkpoint_path = output_dir / "checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])

    print(
        {
            "device": str(device),
            "parameters": model.parameter_count(),
            "train_tokens": len(train_stream),
            "valid_tokens": len(valid_stream),
            "vocab_size": tokenizer.vocab_size,
            "start_step": start_step,
        },
        flush=True,
    )

    def learning_rate(step: int) -> float:
        warmup = max(1, min(100, args.steps // 10))
        if step < warmup:
            return args.learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 * args.learning_rate + 0.9 * args.learning_rate * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    model.train()
    started = time.time()
    best_loss = float("inf")
    for step in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        inputs, targets = make_batch(
            train_stream,
            batch_size=args.batch_size,
            context_length=config.context_length,
            device=device,
            rng=rng,
        )
        with torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step or (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            valid_loss = estimate_loss(
                model,
                valid_stream,
                batch_size=min(args.batch_size, 16),
                context_length=config.context_length,
                device=device,
            )
            elapsed = time.time() - started
            print(
                f"step={step + 1} train_loss={loss.item():.4f} "
                f"valid_loss={valid_loss:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            save_checkpoint(
                checkpoint_path, model, optimizer, step=step + 1, valid_loss=valid_loss
            )
            if valid_loss < best_loss:
                best_loss = valid_loss
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, step=step + 1, valid_loss=valid_loss
                )
    (output_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "best_valid_loss": best_loss,
                "parameters": model.parameter_count(),
                "device": str(device),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
