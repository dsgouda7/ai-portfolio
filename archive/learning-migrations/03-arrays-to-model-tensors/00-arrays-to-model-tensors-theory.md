# Arrays to Model Tensors: Handwritten Theory

## 1. A shape is a contract

A model does not receive a spreadsheet with friendly column names. It receives numbers arranged along axes.

Use this mental model: **a shape is an address system**.

For a tabular feature matrix:

```text
X.shape = (examples, features)
```

Axis 0 answers, "Which example?" Axis 1 answers, "Which feature?" The tuple `(6, 4)` is useful only when you can say that it means six dispatches with four measurements each.

A DataFrame helps before the model boundary because it preserves labels and can mix column types. Converting selected columns to a NumPy array removes those labels and creates one homogeneous numeric block. Save the feature order explicitly. A tensor library cannot recover names that were discarded.

## 2. Indexing can remove an axis

An integer index chooses one coordinate and removes that axis:

```text
X[0]      (6, 4) -> (4,)
```

A slice describes a range and preserves the axis, even when the range contains one item:

```text
X[0:1]    (6, 4) -> (1, 4)
```

The values in `X[0]` and `X[0:1]` may look identical when printed. Their contracts differ. `(4,)` means one feature vector with no batch axis. `(1, 4)` means one-example batch with four features.

The same distinction appears when selecting a column:

```text
X[:, 2]      -> (6,)
X[:, 2:3]    -> (6, 1)
```

Use an integer when you intend to remove an axis. Use a slice when downstream code still needs that axis.

## 3. Reshape and transpose answer different questions

A reshape groups the same ordered values under a new address system. It does not create evidence and does not automatically attach meaning.

```text
(6 examples, 4 features)
-> reshape
(3 review batches, 2 examples, 4 features)
```

This is valid because both shapes contain 24 values. It is meaningful because the new axes were chosen deliberately.

A transpose reorders axes. For example:

```text
(batch, time, features)
-> transpose
(batch, features, time)
```

The number of values remains unchanged, but the coordinate of each value changes. In Transformer code, this is how a combined feature axis is split into heads and moved ahead of the token axis.

Adding a size-one axis is another way to state intent. `x[:, None]`, `np.expand_dims`, `tf.expand_dims`, and `torch.unsqueeze` all create room for later broadcasting. `squeeze` removes a size-one axis. Use it only when you know which axis should disappear.

## 4. Broadcasting is aligned repetition

Broadcasting applies one smaller array across a larger array without manually copying it.

To standardize four feature columns:

```text
X             (examples, features)
feature_mean  (features,)
feature_std   (features,)
```

NumPy aligns shapes from the trailing dimensions. The feature dimensions match, so one mean and standard deviation are reused across every example.

This is equivalent to a loop over rows, but the vectorized expression states the real operation more directly:

```text
(X - feature_mean) / feature_std
```

Broadcasting does not infer intent. `(6, 4) + (6,)` fails because the trailing dimensions `4` and `6` disagree. If the six values are row-specific offsets, reshape them to `(6, 1)`. The size-one feature axis says that each row value may repeat across all four features.

## 5. Matrix multiplication preserves the outer roles

A dense classifier receives a feature matrix and contracts the feature axis with a weight matrix:

```text
X                 @ W                 = logits
(batch, features) @ (features, class) = (batch, class)
```

The matching inner `features` dimensions disappear into dot products. The outer `batch` and `class` dimensions survive. A bias shaped `(class,)` then broadcasts across the batch.

Keras and PyTorch compute this same affine function. Their stored weight layouts differ:

```text
Keras Dense kernel:       (input_features, output_units)
PyTorch Linear weight:    (output_features, input_features)
```

That transpose is an implementation convention, not different mathematics. Loading one known NumPy matrix into both frameworks and comparing logits proves the distinction.

## 6. A tensor is an array with execution state

NumPy arrays, TensorFlow tensors, and PyTorch tensors all have values, shapes, axes, and dtypes. Framework tensors add capabilities such as automatic differentiation and device execution.

TensorFlow commonly uses functions such as `tf.reshape`, `tf.transpose`, `tf.gather`, and `tf.expand_dims`. PyTorch commonly uses `reshape`, `permute`, indexing, `stack`, `cat`, and `unsqueeze`.

The API names matter less than the invariant question:

> What does each axis mean before and after this operation?

A framework can detect incompatible dimensions. It cannot detect that you silently swapped batch and time when both happen to have the same size.

## 7. Raw text becomes aligned windows

An RNN cannot consume strings directly. A minimal text pipeline performs four steps:

```text
raw text -> tokens -> integer IDs -> supervised windows
```

Given context length three:

```text
aria heard the signal aboard

[aria, heard, the]    -> signal
[heard, the, signal]  -> aboard
```

A loop makes the window rule readable. Broadcasted index grids apply the same offsets to every starting position at once. Comparing both results proves that vectorization preserved the intended examples.

The resulting input has shape `(examples, time)`. Grouping examples into mini-batches produces `(mini_batches, batch, time)`. If the number of examples does not divide evenly, the remainder policy must be explicit: retain a smaller final batch, pad it, or deliberately drop it.

## 8. Embedding lookup appends one axis

Token IDs are addresses into an embedding table:

```text
token IDs       (batch, time)
embedding table (vocabulary, embedding)
embedded input  (batch, time, embedding)
```

Lookup preserves batch and time, then appends the selected row's embedding coordinates. NumPy fancy indexing, TensorFlow `gather`, Keras `Embedding`, PyTorch indexing, and PyTorch `nn.Embedding` all implement this contract.

An RNN consumes `(batch, time, embedding)` and returns a hidden vector at each time step. Keras commonly returns final state as `(batch, hidden)`. PyTorch includes a leading layer/direction axis, producing `(layers, batch, hidden)`. That extra axis is architecture bookkeeping, not another batch.

## 9. Practical failure rules

- Never write a shape without naming its axes.
- Do not assume one printed row still has a batch dimension.
- Do not use reshape to repair an unexplained mismatch; first identify every axis.
- Fit preprocessing statistics on training data, then broadcast them onto validation and test data.
- Treat integer token IDs as lookup addresses, not continuous model features.
- Check dtype beside shape: features are usually floating point; class and token IDs are usually integers.
- Compare a vectorized implementation with a readable loop when the indexing is new.

**Durable summary:** Tensor code stops being magic when every operation can be narrated as selecting, grouping, reordering, repeating, or contracting named axes.
