# The Modern Training Lifecycle: Handwritten Theory

## 1. Split before preprocessing

Validation and test sets exist to simulate future evidence. If their feature distribution helps
choose normalization statistics, they have already influenced training.

The correct order is:

```text
raw rows
-> train/validation/test split
-> fit mean and scale on train only
-> apply those train statistics everywhere
```

The notebook measures that train-only and all-data means differ. That difference is the leaked
information an all-data normalizer would import.

**Complaint:** honest splits still fail if validation accidentally behaves like training.

## 2. Train and validation loops have different responsibilities

The training loop:

- selects training mode;
- constructs gradient graphs;
- clears gradients;
- backpropagates;
- updates parameters.

The validation loop:

- selects evaluation mode;
- constructs no gradient graph;
- measures without updating.

These are not cosmetic differences. Dropout produces different repeated outputs in training mode
and identical repeated outputs in evaluation mode. The notebook measures both behaviors on the
same input.

## 3. Average losses with batch size in mind

If the final batch is smaller, averaging batch means equally gives that smaller batch too much
weight.

The readable pattern is:

```text
sum(batch loss * batch size)
divide by total examples
```

This preserves the per-example mean across uneven batch sizes.

## 4. Schedulers and mixed precision are conditional tools

A scheduler changes the learning rate according to measured progress. In the notebook,
validation loss drives a plateau scheduler. The recorded learning-rate history is evidence of
its decisions.

Mixed precision can improve accelerator throughput and memory use. It is enabled only on CUDA in
this teaching run. That boundary is honest: CPU execution does not pretend to demonstrate an
accelerator payoff.

Neither tool repairs a wrong loss, leaked split, or broken shape contract.

## 5. Early stopping is a validation state machine

Use this mental model: early stopping is a patient observer, not a trainer.

It tracks:

- the best validation value seen;
- the number of consecutive non-improving checks;
- the patience limit.

It should not watch training loss because training loss often keeps improving while
generalization worsens.

The stopping rule and the best-model rule are related but different:

- stop when patience is exhausted;
- keep the state from the best validation epoch.

The last epoch is not automatically the best epoch.

## 6. Best-model and resumable checkpoints serve different jobs

A best-model checkpoint supports inference. It needs:

- model state;
- preprocessing state;
- architecture/configuration facts required for reconstruction.

A resumable checkpoint supports continuation. It additionally needs:

- optimizer state;
- scheduler state;
- current epoch;
- early-stopping state;
- mixed-precision scaler state when enabled;
- any sampler or accumulation progress that affects the next update.

The notebook writes separate files so the distinction remains visible. It reloads the best model
for test inference and checks that the resumable file contains every declared continuation field.

## 7. Return predictions to original units

Training on normalized targets can improve optimization, but stakeholders do not consume
normalized fuel scores.

The inference path must reverse target normalization:

```text
normalized prediction
-> multiply by training target scale
-> add training target mean
-> report original units
```

Preprocessing state is therefore part of the deployed prediction contract.

## 8. Automation comes after visible responsibilities

Lightning can map the manual pieces into hooks and callbacks. `torch.compile` can optimize a
correct graph. These tools reduce boilerplate; they do not remove the need to understand:

- split integrity;
- mode changes;
- gradient boundaries;
- checkpoint contents;
- metric units.

## Practical failure modes

- Normalizing before splitting.
- Validating in training mode.
- Updating parameters on validation batches.
- Early-stopping on training loss.
- Treating the last epoch as the best model.
- Omitting preprocessing state from inference artifacts.
- Calling a weight-only file resumable.
- Enabling advanced performance tools before correctness is measured.

**Durable summary:** a modern trainer automates lifecycle code, but correctness still depends on
visible evidence boundaries, mode boundaries, and state boundaries.
