# Pretraining a Base Language Model: Handwritten Theory Notes

## 1. Architecture Is Capacity; Training Is Experience

A correctly assembled decoder with random weights does not know language. It has the pathways required to learn, but every prediction begins as an unpracticed distribution over tokens.

**Track position:** this chapter closes the loop that began with raw text. The forward path produces a prediction; the backward path assigns responsibility for its error.

Pretraining repeats one loop:

```text
read a packed token block
-> predict the next token at every position
-> compare predictions with actual following tokens
-> send responsibility backward through every block
-> update the weights
-> repeat over the corpus
```

The architecture stays the same from the first step to the final checkpoint. Training changes the numbers inside it.

During one backward pass, gradients reach the vocabulary head, normalization scales, FFN projections and gates, attention projections, and token embeddings. The optimizer updates those parameters; it does not rewrite the token IDs or the training text.

Mental model: **architecture builds the instrument; pretraining is the practice that teaches it to play.**

## 2. Refuse Untrusted Inputs Before Training

Training is expensive enough that the model should not begin until every input identity is known.

Before constructing the model, verify:

- manifest fields and file hashes match;
- training and validation are distinct;
- tokenizer, vocabulary, context length, and model config agree;
- every token ID fits the declared vocabulary and dtype.

A shard can still open after corruption. Parsing proves format, not identity. Hash checks make the training run fail closed when bytes change.

Memory aid: **validate the recipe and ingredients before turning on the oven.**

## 3. One Block Contains Many Lessons

A packed sequence is shifted into visible inputs and next-token targets. Every valid position becomes one lesson:

```text
visible prefix          expected next token
aria                    heard
aria heard              the
aria heard the          signal
```

All positions are scored in parallel during training because the correct sequence is already known. The causal mask prevents each position from seeing the answer to its own lesson.

Padding or invalid positions must not contribute to learning. In packed pretraining blocks, most positions are real tokens, so compute is spent practicing language rather than emptiness.

The durable idea is: **one block is not one example; it is a row of next-token lessons sharing one forward pass.**

## 4. The Practical Update Loop

Four mechanisms make the basic learning loop usable:

- **AdamW** remembers recent gradient direction and scale. Its moments are training state, not part of the model's predictions.
- **Warmup and decay** start with cautious updates, then reduce the learning rate as training settles.
- **Gradient clipping** bounds one extreme update. It cannot repair NaN or infinite state.
- **Gradient accumulation** gathers evidence from several small microbatches before one optimizer step.

Memory aid: **warmup controls when to push, clipping controls how hard, accumulation controls how much evidence is gathered, and AdamW remembers the recent terrain.**

## 5. Training History Needs More Than One Loss

A final scalar hides how the run behaved. Record training and validation loss, learning rate, clipping behavior, block-level gradients, difficult token positions, and one fixed probe continuation. Keep lightweight history every step and heavier snapshots at intervals.

Snapshots should contain measured states only. Smooth invented frames can make learning look cleaner than it was.

Validation is separate from optimization. It estimates how surprising held-out text remains; it does not teach the model and does not prove factual knowledge or reasoning.

## 6. A Checkpoint Is a Complete Pause Button

A model-only save can generate text, but it cannot faithfully resume training. The next update also depends on AdamW moments, scheduler position, counters, random state, and model/data identities.

Mental model: **weights are the student's memory; a full checkpoint also saves the teacher's place in the lesson plan, the notebook page, and the shuffled exercise order.**

Resume parity is tested by taking the same saved state and the same next microbatches through two paths:

```text
uninterrupted state -> next update
saved then restored state -> same next update
```

Matching loss and parameters prove same-hardware continuation. They do not promise bitwise equality across different devices or kernels.

## 7. Validation Chooses; Training Loss Diagnoses

The latest checkpoint is not automatically the best. Among interval checkpoints, choose the one with the lowest held-out validation loss.

Do not select using:

- training loss, because the model directly optimized it;
- attractive sample text, because subjective inspection is easy to cherry-pick;
- a test set, because repeated selection would turn the test into validation.

A tiny downward change may be run noise. Report the measured direction honestly without turning a small construction run into a capability claim.

The durable rule is: **training loss tells you whether practice was absorbed; validation loss decides which practice state travels forward.**

## 8. Crack Open What Changed

Compare the selected checkpoint with the initial random state across embeddings, attention, SwiGLU, and normalization. This checks that every expected learning surface moved; it does not reveal one clean location where a fact or skill lives.

Embedding-neighbor movement can show that token geometry changed, but a tiny corpus and short run do not establish stable semantics. Attention maps, parameter deltas, and nearest neighbors are diagnostics, not complete explanations.

## 9. Scaling Changes the Bill, Not the Recipe

The same learning loop applies to a tiny model, 100M parameters, 1B parameters, or 7B parameters. What grows is the resource bill:

- more parameters require more weight storage;
- training adds gradients and optimizer moments;
- longer contexts and larger batches increase activation memory;
- more tokens require more update work;
- models that do not fit one device require distributed strategies.

The conceptual recipe remains:

```text
trusted tokens -> causal predictions -> loss -> gradients -> updates -> validation -> checkpoints
```

Large-scale pretraining adds infrastructure, not a new learning objective.

Memory aid: **scale turns the same kitchen recipe into industrial logistics.**

## 10. What the Final Artifact Proves

A completed package lets a clean process reconstruct the config, tokenizer, selected weights, continuation state, data lineage, and fixed-prompt predictions.

Reload parity proves the package is internally complete. It does not prove broad knowledge, reasoning, factuality, safety, instruction following, or production readiness.

A **base model** is a next-token predictor shaped by its pretraining corpus. It is not automatically an assistant.

## 11. Practical Failure Modes

- **Loading pretrained weights accidentally:** the experiment no longer shows random-to-base learning.
- **Training before checking hashes:** corrupted or substituted data can silently consume the run.
- **Using one shard for train and validation:** held-out measurement disappears.
- **Clearing gradients or stepping the scheduler at the wrong time:** accumulation and the learning-rate timeline become incorrect.
- **Clipping after the optimizer step:** the dangerous update has already happened.
- **Continuing after NaN or Inf:** later metrics and checkpoints are no longer trustworthy.
- **Saving weights only:** exact training resume is impossible.
- **Selecting the latest checkpoint automatically:** later training may have worsened validation.
- **Calling a small loss decrease a capability gain:** lower surprise is not proof of reasoning or knowledge.
- **Treating estimated scale numbers as measurements:** label calculator outputs separately from observed runtime values.

The durable model is: **verify the inputs, start from random weights, practice next-token prediction, watch the whole training state, select with held-out evidence, and package enough information to reproduce the result.**
