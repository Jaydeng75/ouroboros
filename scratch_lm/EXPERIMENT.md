# Sorting-discovery experiment

## Question

Can a code language model trained from random initialization, without
being shown an ordering procedure, compose loops, comparisons, indexing,
and position exchanges into a working ordering procedure when asked at
evaluation time?

This is an experiment, not a promised result. A failure can reflect model
scale, token count, data diversity, decoding, or optimization rather than
a fundamental inability to compose the procedure.

## Training-data contract

The initial corpus is synthetic and self-authored, so its license is
unambiguous. It contains Python examples for arithmetic, branches, loops,
list copying and filtering, position exchanges, string handling, recursion,
 dictionaries, nested loops, grids, and small classes.

The corpus must not contain:

- an implementation whose goal is to arrange values by magnitude;
- sorting vocabulary or identifiers built from sorting vocabulary;
- `sorted`, list `.sort`, `min`, `max`, `heapq`, or `bisect` calls;
- imports from ordering-related modules;
- an inequality comparison between two elements of the same list;
- the evaluation prompt or evaluation tests.

`scratch_lm.audit` enforces the explicit parts of this contract and fails
closed on malformed Python. The procedural generator enforces the semantic
part by construction. No audit can prove the absence of every ordering
concept, so claims must stay limited to this stated contract.

## Model

- Decoder-only Transformer initialized from scratch.
- 26,399,744 parameters with the base configuration.
- Custom byte-pair tokenizer trained only on the training split.
- Maximum token length: 16 bytes, preventing whole programs from becoming
  single memorized tokens.
- Context length: 256 tokens.
- Initial corpus: 24,000 examples and 408,485 training tokens including
  beginning/end markers. This is a smoke-scale corpus, not an adequate
  final pretraining set.

## Evaluation

`scratch_lm.evaluate_sorting` uses an unseen function name and specification.
Candidates are sampled from the trained checkpoint and executed against
hidden cases. A candidate is counted as a discovered procedure only when:

1. it passes all evaluation cases;
2. it does not call an existing ordering API;
3. its behavior also passes newly sampled arrays supplied by the reviewer.

The third condition guards against memorizing the visible examples in the
evaluation script. Reviewers should change the arrays before accepting a
positive result.

## Generations

Generation 0 is pretrained only on the leakage-controlled corpus.

For later generations, OUROBOROS may generate additional non-ordering code,
verify it by execution, audit it under the same contract, and continue
training from the previous checkpoint. Evaluation prompts and tests must
never enter that continuation data. Record data manifest hash, checkpoint,
loss, sample count, and evaluation outcome for every generation.

## Scaling rule

If generation 0 does not discover an ordering procedure, first scale the
permitted data and training tokens, then model size. Do not repair failure
by adding ordering examples: that changes the research question.
