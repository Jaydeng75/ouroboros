"""QLoRA SFT for OUROBOROS - run in Google Colab (T4) or any CUDA box.

Expects the chat JSONL produced by `ouroboros export-sft`
({"messages": [{"role": ..., "content": ...}, ...]} per line).

Colab quickstart:
  1. Upload this file + datasets/sft.jsonl (or clone the project).
  2. Runtime > Change runtime type > T4 GPU.
  3. !pip install -q unsloth trl datasets
  4. !python training/train_qlora.py --data datasets/sft.jsonl \
         --out adapters/gen1 --push your-username/my-self-improving-qwen

Locally (no GPU) this script only parses: run with --check to validate the
dataset without importing torch/unsloth.
"""
from __future__ import annotations

import argparse
import json
import sys


def check_dataset(path: str) -> int:
    n = 0
    for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        assert isinstance(messages, list) and len(messages) == 2, lineno
        assert messages[0]["role"] == "user", lineno
        assert messages[1]["role"] == "assistant", lineno
        assert "```python" in messages[1]["content"], lineno
        n += 1
    print(f"OK: {n} SFT example(s) in {path}")
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/sft.jsonl")
    parser.add_argument("--base", default="unsloth/Qwen2.5-Coder-7B")
    parser.add_argument("--out", default="adapters/gen1")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--push", default="",
                        help="HF repo to push the merged model to")
    parser.add_argument("--check", action="store_true",
                        help="Validate dataset only; do not train")
    args = parser.parse_args()

    if args.check:
        check_dataset(args.data)
        return 0

    from datasets import Dataset  # noqa: PLC0415
    from trl import SFTTrainer  # noqa: PLC0415
    from transformers import TrainingArguments  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    check_dataset(args.data)
    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")
            if line.strip()]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    def to_text(row):
        return {
            "text": tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = Dataset.from_list([to_text(r) for r in rows])
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="linear",
            warmup_steps=10,
            logging_steps=5,
            save_strategy="no",
            output_dir=args.out + "-trainer",
            report_to=[],
        ),
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Adapter saved to {args.out}")

    merged_dir = args.out + "-merged"
    model.save_pretrained_merged(
        merged_dir, tokenizer, save_method="merged_16bit"
    )
    print(f"Merged model saved to {merged_dir}")
    if args.push:
        model.push_to_hub_merged(
            args.push, tokenizer, save_method="merged_16bit"
        )
        print(f"Pushed merged model to https://huggingface.co/{args.push}")
        print("Serve it locally:")
        print(f"  printf 'FROM hf.co/{args.push}\\n' > Modelfile")
        print("  ollama create self-improving-ai -f Modelfile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
