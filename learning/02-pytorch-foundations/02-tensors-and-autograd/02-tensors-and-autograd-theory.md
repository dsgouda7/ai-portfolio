# Tensors and Autograd: Handwritten Theory

The previous loop worked, but a list of numbers does not tell you which number belongs to which example. A **tensor** is a number container; an **axis** is one dimension of its address system. A **batch** groups examples. **Shape** gives each axis length, **dtype** specifies the kind of number stored, and **device** says where those numbers live. **Autograd** records operations so it can trace error sensitivity backward.

For the language-shaped preview, a **token** is one chosen text piece. Its integer **ID** is an address, not a measurement of meaning. An **embedding** retrieves a trainable row for that address. A **hidden state** is an intermediate feature list. A **logit** is a raw score for one possible answer, not yet a probability. Here a **head** is just one group cut from a feature list; the reason to use several attention heads belongs to [Stage 04](../../04-transformers/README.md). [Stage 03](../../03-language-model-foundations/02-tokenization-and-embeddings/README.md) owns how text pieces and their learned rows are chosen.

## 1. A shape is an address system

A tensor is not difficult because it has many numbers. It is difficult when the axes lose their
names.

Use this mental model:

> A shape tells you how to address a value; axis names tell you what that address means.

For language-model data:

```text
token IDs       (batch, time)
hidden states   (batch, time, feature)
logits          (batch, time, vocabulary)
```

Shape, dtype, and device form one contract. Token IDs are integer lookup addresses. Hidden states
are floating-point measurements. Two tensors with compatible shapes can still be incompatible if
their dtypes or devices disagree.

**Complaint:** printing values does not reveal whether an axis was preserved.

## 2. Indexing can silently remove an axis

An integer index chooses one coordinate and removes that axis:

```text
states[0]    (batch, time, feature) -> (time, feature)
```

A slice preserves the axis:

```text
states[0:1]  (batch, time, feature) -> (1, time, feature)
```

The values can look identical while the downstream contracts differ. Use an integer when you
intend to remove an axis. Use a slice when you still need a one-example batch.

**Complaint:** preserving the number of values does not preserve their interpretation.

## 3. Reshape and transpose solve different problems

Reshape groups the same ordered values under a new address system. Transpose moves axes.

Splitting a model feature axis into attention heads uses both operations:

```text
(batch, time, feature)
-> reshape
(batch, time, head, width)
-> transpose
(batch, head, time, width)
```

The notebook performs the reverse operations and proves exact value recovery. That round trip
shows value preservation. It does not prove that arbitrary axis names are meaningful; meaning is
still the programmer's job.

## 4. Broadcasting is aligned repetition

Broadcasting aligns shapes from the trailing dimensions.

A feature bias shaped `(feature,)` can be reused across `(batch, time, feature)`. A time bias
shaped `(time,)` collides with the feature axis. Reshaping it to `(1, time, 1)` states the intended
repetition across batch and feature.

The framework's error is useful evidence. It proves the original trailing dimensions were
incompatible. The repair is not "add dimensions until it runs." The repair is to name the missing
axes.

**Complaint:** element-wise operations preserve axes; a model also needs to mix information across
an axis.

## 5. Matrix multiplication contracts one axis

For each text position, combine its feature values into one raw score per possible text piece. The feature entries are summed away, while the example and position stay separate. This is what contracting the feature axis means. Here **LM** means language model; the operation below checks its shapes, not its ability to predict text.

For an LM vocabulary projection:

```text
states @ W_vocab
(batch, time, feature) @ (feature, vocabulary)
-> (batch, time, vocabulary)
```

The matching feature dimensions are consumed by dot products. Batch and time survive. Vocabulary
appears as the new output role.

Element-wise multiplication asks, "How should each coordinate be scaled?" Matrix multiplication
asks, "How should coordinates combine into new coordinates?"

## 6. Autograd records influence

Autograd records operations that connect tensors requiring gradients to a final scalar.

For a simple chain, the notebook compares `.grad` with a manually calculated derivative. The
agreement is evidence that the graph followed the same local sensitivities.

Gradients accumulate because several paths or mini-batches may legitimately contribute to one
parameter. Clearing is therefore explicit. `detach()` creates a tensor that shares values but
leaves the graph. `inference_mode()` disables graph construction for an inference region.

Autograd knows influence, not intent. It can correctly differentiate the wrong objective or a
silently misnamed axis.

## 7. The language-model-shaped bridge

Embedding lookup preserves batch and time and appends a feature axis. The vocabulary projection
contracts feature and appends vocabulary. Backpropagation produces gradients shaped like the
embedding table and output projection parameters.

This is the same mechanism used in a large language model, reduced to inspectable dimensions.

## Practical failure modes

- Writing shapes without axis names.
- Accidentally removing the batch axis.
- Using reshape as a mystery repair.
- Treating floating-point values as token IDs.
- Confusing element-wise and matrix multiplication.
- Forgetting that gradients accumulate.
- Assuming a differentiable graph is automatically a sensible model.

**Durable summary:** tensor code becomes readable when every operation can be narrated as
selecting, regrouping, moving, repeating, or contracting named axes.


The preview's squared-score loss only checks gradient wiring. It does not train a useful text predictor. Next, [Models and Training Loops](../03-models-and-training-loops/03-models-and-training-loops.ipynb) asks who owns these tensors when they must move, train, and survive a fresh process.
