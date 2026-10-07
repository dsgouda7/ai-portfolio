# The Deep-Learning Map: Handwritten Theory

## 1. A model is not the whole learning system

Start with the visible problem: a freshly created network has the right input and output shapes,
but its predictions do not follow the target curve.

The memorable mental model is a workshop:

- **data** brings examples into the room;
- the **module** is the adjustable machine;
- the **forward pass** runs material through the machine;
- the **loss** is the inspection report;
- **backward** assigns responsibility to each adjustment;
- the **optimizer** turns that report into a parameter change.

The module alone does not learn. It only computes with its current parameters.

**Complaint:** a forward pass can be perfectly valid and still be wrong.

## 2. Loss compresses many mistakes into one training signal

The model emits one prediction per example. The target has the same shape. A loss function
compares them and returns one scalar.

That scalar is useful because backpropagation needs a single quantity whose sensitivity can be
traced through the graph. It is not automatically a human-readable quality metric, and it does
not explain which example failed.

In the notebook, mean squared error is appropriate because the task is continuous regression and
large misses should contribute more strongly.

**Complaint:** knowing the loss is not the same as knowing how to change the parameters.

## 3. Backward reports responsibility; the optimizer acts

`loss.backward()` follows the recorded computation graph in reverse. Each parameter receives a
gradient in its `.grad` buffer.

Think of the gradient as a local responsibility report:

> If this parameter moved a tiny amount, how would the loss move?

Backward does not update the weight. The notebook proves this by cloning a weight before
backward, checking that it is unchanged afterward, and then measuring a change only after
`optimizer.step()`.

This distinction matters because different optimizers can interpret the same gradient differently.
SGD uses the current gradient directly. Adam also keeps moving averages. The gradient belongs to
the model parameter; the update policy and its extra state belong to the optimizer.

**Complaint:** one update is only one local correction.

## 4. An epoch repeats the same responsibility cycle

The minimal full-batch loop is:

```text
clear old gradients
-> predict
-> measure loss
-> backpropagate responsibility
-> update parameters
-> repeat
```

The order is a contract. If old gradients are not cleared, new responsibility is added to stale
responsibility. That can be intentional, but it must not happen by accident.

The notebook records every loss rather than printing a success claim. The plotted history and the
final learned curve are two different kinds of evidence:

- the history shows whether optimization reduced its objective;
- the fitted curve shows what function the learned parameters now represent.

Neither proves production readiness. The dataset is tiny, deterministic, and designed only to
make the mechanics visible.

## 5. Inference keeps the learned function and drops the training graph

After training, the useful artifact is the learned parameter state.

`model.eval()` selects evaluation behavior for mode-sensitive layers such as dropout.
`torch.inference_mode()` prevents gradient graph construction when no update will follow.

The forward computation still happens. What disappears is the bookkeeping needed for backward.
The notebook verifies this by showing that inference outputs do not require gradients.

**Complaint:** the route works, but tensor state is still mostly implicit.

## 6. The LLM bridge

An LLM follows the same map:

```text
token batches
-> Transformer module
-> token logits
-> next-token loss
-> backward
-> optimizer
-> repeated updates
-> inference
```

The number of parameters, axes, devices, and updates grows. The responsibilities do not change.

## Practical failure modes

- Calling `backward()` and assuming weights changed.
- Forgetting to clear gradient buffers.
- Comparing prediction and target tensors with unintended broadcasting.
- Building gradient graphs during inference.
- Showing only a final loss instead of recorded evidence.
- Treating a tiny teaching fit as evidence of broad generalization.

**Durable summary:** deep learning is not one opaque operation; it is a loop of prediction,
measurement, responsibility, and update.
