# OUROBOROS

A $0 recursive self-improvement loop for code models. This scaffold is
Phase 1 only: the local harness. Training, the adapter zoo, and the skill
library come later, once the traces are trustworthy.

## The loop

```
prompt -> Ollama (local model) -> extract code -> pytest sandbox
       -> pass: log full trace | fail: feed traceback back and repair
```

Every attempt (code, verdict, stdout/stderr tails) is appended to a JSONL
trace log. Those traces, not raw successes, are the training fuel.

## Layout

- `ouroboros/` – harness package (`cli`, `harness`, `sandbox`, `traces`,
  `ollama_client`)
- `tasks/seed_tasks.jsonl` – 50 training-pool tasks in 5 shards (prompt +
  pytest verifier); `tasks/reference_solutions.jsonl` holds a verified
  solution per task for pool validation only (never shown to the model)
- `tasks/holdout_tasks.jsonl` – evaluation-only tasks. **Never train on
  these or their traces.**
- `traces/` – append-only JSONL logs (gitignored)
- `tests/` – harness unit tests (run with a fake model, no Ollama needed)

## Setup

Requires Python 3.11+ and [Ollama](https://ollama.com) for real runs.

```bash
uv venv
uv pip install -e ".[dev]"
ollama pull qwen2.5-coder:7b
```

## Run

```bash
source .venv/bin/activate

# Unit tests (no Ollama needed)
python -m pytest -q

# Plumbing check without Ollama
python -m ouroboros.cli run --dry-run --tasks tasks/seed_tasks.jsonl --limit 2

# Real run
python -m ouroboros.cli models
python -m ouroboros.cli run --model qwen2.5-coder:7b
```

Or without activating: `uv run python -m ouroboros.cli run ...`

## Safety

The sandbox runs model-generated code in a pytest subprocess with your
user permissions (timeout enforced, nothing imported into the harness
process). Run the loop on a trusted machine only, ideally inside a VM or
a disposable account. Do not point it at tasks from untrusted sources.

## Evolution

`ouroboros evolve` runs one generation:

1. **Variation** – the model mutates each seed task (`add_constraint`,
   `tighten_spec`, `demand_efficiency`), plus a deterministic `compose`
   operator that fuses two seeds into one harder task.
2. **Selection** – each mutant is sampled N times. It survives only if the
   success rate lands in [0.2, 0.8]: too easy or impossible teaches nothing.
3. **Archive** – survivors compete for cells keyed by (parent, operator) in
   `archive.json`. A challenger must be more learnable (closer to 50%
   success) than the incumbent, and novel enough (Jaccard distance on the
   prompt) to enter. Kept mutants land in `tasks/evolved.jsonl`, ready to
   be sharded and trained on. Held-out tasks are never mutated.

```bash
python -m ouroboros.cli evolve --model qwen2.5-coder:7b --samples 4 \
    --parent-limit 12 --compose-limit 24 --gen-timeout 300
```

Candidate generation is bounded by default so a 50-task pool does not
expand into thousands of model calls. `--parent-limit` caps model-mutated
seed tasks, `--compose-limit` caps composed pairs, and `--random-seed`
makes the selection reproducible.

## Training (inheritance)

Close the loop: traces become weights, weights become the next solver.

```bash
# 1. Generate traces on seeds AND evolved tasks (same log).
python -m ouroboros.cli run --model qwen2.5-coder:7b
python -m ouroboros.cli run --tasks tasks/evolved.jsonl --model qwen2.5-coder:7b

# 2. Export SFT data (held-out traces are always excluded).
python -m ouroboros.cli export-sft --traces traces/traces.jsonl --out datasets/sft.jsonl

# 3. Train on free Colab T4 (Unsloth QLoRA, ~1-3h for a few hundred traces).
#    Upload datasets/sft.jsonl, then:
#    !pip install -q unsloth trl datasets
#    !python training/train_qlora.py --data datasets/sft.jsonl \
#        --out adapters/gen1 --push your-username/my-self-improving-qwen

# 4. Serve the child model and score it - held-out tasks ONLY.
ollama create self-improving-ai -f training/Modelfile
python -m ouroboros.cli run --tasks tasks/holdout_tasks.jsonl \
    --model self-improving-ai --out traces/holdout_gen1.jsonl
```

Keep the child only if its held-out pass rate beats the frozen baseline.
Each generation: `genN` traces -> `adapters/genN` -> holdout scorecard.

## From-scratch sorting-discovery LM

`scratch_lm/` is a separate experiment: a 26.4M-parameter decoder-only
Transformer and custom BPE tokenizer trained from random initialization.
Its synthetic Python corpus deliberately excludes ordering procedures,
ordering APIs, and same-list inequality comparisons. The model is asked
to arrange values only in a held-out evaluation, so a pass would indicate
composition of learned loops, comparisons, indexing, and exchanges.

The initial corpus is already generated in `scratch_data/`: 24,000 audited
examples (22,200 train / 1,200 validation / 600 test). See
`scratch_lm/EXPERIMENT.md` for the leakage contract and success criteria.

```bash
uv pip install -e ".[dev,train]"

# Regenerate and audit the corpus.
python -m scratch_lm.corpus --output-dir scratch_data --count 24000
python -m scratch_lm.tokenizer --input scratch_data/train.jsonl \
    --output scratch_data/tokenizer.json --vocab-size 2048

# Free Colab/Kaggle T4 run. Use --resume after a runtime restart.
python -m scratch_lm.train --data-dir scratch_data \
    --output-dir scratch_runs/gen0 --preset base --steps 3000

# Held-out discovery evaluation (never train on this prompt/tests).
python -m scratch_lm.evaluate_sorting \
    --checkpoint scratch_runs/gen0/best.pt \
    --tokenizer scratch_runs/gen0/tokenizer.json
```

The checked-in corpus is smoke-scale (408k training tokens). Scale the
permitted corpus before drawing conclusions from a negative result.

Corpus v2 is generated under `scratch_data_v2/`: 250,000 examples
(231,250 train / 12,500 validation / 6,250 test), including compositional
and multi-function examples, with 5,277,448 training tokens. Its BPE
tokenizer was trained on a representative 25,000-text sample from the v2
training split. Regenerate it with:

```bash
python -m scratch_lm.corpus --output-dir scratch_data_v2 \
    --count 250000 --version 2
python -m scratch_lm.tokenizer --input scratch_data_v2/train.jsonl \
    --output scratch_data_v2/tokenizer.json --vocab-size 2048 \
    --max-texts 25000
```

Corpus v3 (`scratch_data_v3/`) keeps the v2 scale but rewrites about
75% of examples into the evaluation shape: a top-level function signature
followed by a one-line docstring describing the behavior, then the body.
This fixes the train/test format mismatch seen in generation 1, where
docstring prompts were out-of-distribution. Regenerate with:

```bash
python -m scratch_lm.corpus --output-dir scratch_data_v3 \
    --count 250000 --version 3
python -m scratch_lm.tokenizer --input scratch_data_v3/train.jsonl \
    --output scratch_data_v3/tokenizer.json --vocab-size 2048 \
    --max-texts 25000
```

Measure ordinary code generation before rerunning the discovery test:

```bash
python -m scratch_lm.benchmark \
    --checkpoint scratch_runs/gen0/best.pt \
    --tokenizer scratch_runs/gen0/tokenizer.json \
    --samples-per-task 3
```

## Working with multiple devices

Training parallelizes by shard, not by gradient. Each device trains on a
different slice of the seed pool and syncs tiny trace files through git.
Coordination lives in `shards.json` (same frozen base model and format
version everywhere; held-out tasks can never be sharded).

One-time setup (device 1):

```bash
git init && git add -A && git commit -m "ouroboros"
git remote add origin <your-repo-url> && git push -u origin HEAD
```

A second device joins with:

```bash
git clone <your-repo-url> && cd ouroboros
uv venv && uv pip install -e ".[dev]"
ollama pull qwen2.5-coder:7b
python -m ouroboros.cli pull
python -m ouroboros.cli shards          # see what is open
python -m ouroboros.cli claim search --as my-laptop
python -m ouroboros.cli run --shard search
python -m ouroboros.cli push -m "traces: search"
```

Rules: never force-push `shards.json`, never train on held-out traces,
and only merge adapters that beat their parents on the held-out set.

## Next phases

1. Collect ~200 seed traces; inspect fail -> fix -> pass chains by hand.
2. Format repair traces as SFT data; one QLoRA run on free Colab (Qwen2.5-Coder).
3. Compare adapter vs frozen baseline on `tasks/holdout_tasks.jsonl` only.
4. Then: task mutation (adversarial curriculum), MAP-Elites archive,
   skill-library extraction, adapter merging.
