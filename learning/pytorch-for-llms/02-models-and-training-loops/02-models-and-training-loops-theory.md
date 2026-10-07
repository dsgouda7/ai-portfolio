# Models and Training Loops: Handwritten Theory

## 1. Raw tensors already contain the learning mechanism

A linear regression can be written directly:

```text
prediction = X @ W + b
```

If `W` and `b` require gradients, autograd can train them. This proves that `nn.Module` is not a
different mathematical method.

The complaint is ownership. With loose tensors, you must remember:

- which tensors are parameters;
- which tensors move to the device;
- which tensors must be saved;
- which function defines the forward contract.

## 2. A module gives state a home

Think of `nn.Module` as a labeled equipment case. It contains the adjustable parts and the route
data follows through them.

`nn.Linear` stores:

```text
weight  (target, feature)
bias    (target,)
```

The familiar written equation often uses `W(feature, target)`. PyTorch's stored weight is
transposed relative to that notation. The mathematics is unchanged.

The `forward` method should describe prediction. It should not hide the loss or optimizer because
those are experiment responsibilities, not properties of the learned function.

**Complaint:** a valid forward pass does not prove the loss contract is valid.

## 3. Prediction and target shapes must agree deliberately

For scalar regression:

```text
prediction  (example, 1)
target      (example, 1)
loss        scalar
```

A target shaped `(example,)` may broadcast against `(example, 1)` and produce a larger comparison
than intended. The code can run and still optimize the wrong quantity.

The notebook checks shape equality before training. That is a stronger habit than waiting for a
runtime error because broadcasting errors often do not raise one.

## 4. The optimizer owns update policy and update state

The visible loop is:

```text
optimizer.zero_grad()
prediction = model(X)
loss = loss_function(prediction, y)
loss.backward()
optimizer.step()
```

Each line has one responsibility:

- clear stale responsibility;
- predict with current parameters;
- compress error into a scalar;
- fill parameter gradient buffers;
- update parameters according to optimizer policy.

An optimizer may carry momentum or adaptive averages. Those values are not part of the model's
forward function, but they matter if training must resume exactly.

## 5. Equivalence is a measurable claim

The notebook trains a raw-tensor implementation and an `nn.Module` implementation under the same
initial values, data, loss, learning rate, and update count.

It then compares:

- weights after accounting for storage transpose;
- biases;
- predictions;
- optimization histories.

The asserted tolerance turns "these are the same" into a falsifiable statement. If the comparison
fails, the likely cause is not the framework. It is a changed initial value, update count, shape,
or gradient-clearing rule.

**Complaint:** an in-memory model cannot cross a notebook boundary.

## 6. State and configuration cross the fresh-kernel boundary

`state_dict()` contains learned tensors. It does not explain the model class or every experiment
choice.

Part 02 therefore writes:

- a weight state file;
- a configuration file containing reconstruction facts.

Part 03 reloads both from a fresh kernel. That handoff proves the artifact is usable rather than
merely present on disk.

## 7. The LLM bridge

Transformer classes are larger `nn.Module` trees. Their forward contracts contain more axes and
may return several named outputs. The same principles hold:

- parameters live in modules;
- loss stays tied to the task;
- optimizers own update state;
- shape contracts are checked before long training runs;
- state must be reconstructable outside the original process.

## Practical failure modes

- Hiding the loss inside `forward`.
- Allowing target broadcasting by accident.
- Forgetting to clear gradients.
- Comparing two implementations with different seeds or update counts.
- Saving weights without architecture or configuration facts.
- Calling a weight-only file resumable training state.

**Durable summary:** `nn.Module` does not replace tensor math; it makes parameter ownership,
forward behavior, movement, and persistence explicit.
