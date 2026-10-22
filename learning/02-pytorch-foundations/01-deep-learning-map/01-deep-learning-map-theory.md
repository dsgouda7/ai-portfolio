# The Deep-Learning Map: Handwritten Theory

## 1. A working model can still be a bad guesser

Imagine a sensor whose raw reading needs correcting. You have pairs: what it
reported and the calibrated value you wanted. Could a small adjustable model
learn the correction instead of you writing a rule for every reading?

The notebook makes 160 such pairs from a known curved rule. These are invented
examples, not measurements from real hardware. That lets you see exactly what
the model should learn.

Why not start with words? Because text brings extra questions about tokens and
embeddings. For now, one number goes in and one number comes out. Later, the
cat-on-the-mat example will reuse the same training jobs.

**Complaint:** having examples does not tell you how a model learns from them.

## 2. Keep the jobs separate

Think of learning as adjusting a guesser, not pressing one mysterious button.

| Job | Plain-English question |
|---|---|
| Data | What examples can you practise on? |
| Forward pass | What does the model guess right now? |
| Loss | How far off are those guesses? |
| Backward | How would tiny parameter changes affect that error? |
| Optimizer | Which changes should actually be made? |

The model stores adjustable numbers called **parameters**. A forward pass uses
their current values. It does not quietly improve them while answering.

That separation matters: successful execution proves that the calculation ran,
not that the answer was useful.

**Complaint:** a collection of adjustable numbers still needs a sensible shape.

## 3. One reading becomes twelve features, then one answer

Each input is a single reading. The first layer turns it into twelve hidden
values. Each combines that same reading with its own weight and offset.
These are twelve learned views of one example, not twelve separate examples.

The middle operation, **tanh**, bends those values smoothly. Without a bend,
stacking these straight-line layers still gives a straight-line relationship.
That cannot follow our curved reference. The notebook checks this using three
equally spaced readings and measures the bend with and without tanh.

The final layer mixes the twelve values back into one prediction.

Read the batch shapes out loud:

- **[160, 1]:** 160 examples, one reading each.
- **[160, 12]:** the same 160 examples, twelve hidden values each.
- **[160, 1]:** the same examples, one answer each.

There are 37 trainable numbers: twelve first-layer weights, twelve offsets,
twelve final-layer weights, and one final offset.

Keep those knobs separate from the numbers passing through them. For 160
readings the hidden layer produces 1,920 **activations**, but it does not
acquire 1,920 new parameters. A different reading changes its activations
even before training changes a single weight. Three inference rows and 160
training rows use the same 37 stored parameters.

The synthetic rule's easy landmarks are -2 → -3.8, 0 → 0.1, and 2 → 4.0.
These describe the rule, not three selected training rows: the even-sized
grid doesn't include exactly zero. The model receives pairs, not the rule.

**Complaint:** the output shape is right, but the initial curve is wrong.

## 4. Turn the misses into one useful score

Before practice, compare each guess with its matching reference value.
Square each miss so opposite signs cannot cancel and larger misses count more.
Then take the average. That is **mean squared error**, or MSE.

The notebook computes those misses directly and checks that PyTorch's loss
function gives the same result. In the recorded CPU run, the untouched model
starts at about **1.905 MSE**.

For a tiny arithmetic check, misses of +1 and -1 average to zero if their
signs cancel. Squaring first gives a mean of 1 instead. That is why a
pleasant-looking average signed miss would be the wrong checking rule.

This is a score to reduce, not an accuracy percentage. Its size depends on the
units and scale of the answers. The plot remains useful because an average can
hide where the worst misses occur.

Prediction and reference tables must match. A missing column dimension can
accidentally trigger a different comparison rather than a helpful error.

**Complaint:** a score says how bad the guesses are, not which knobs to turn.

## 5. Feedback is not the update

During the forward calculation, PyTorch records how the answer depends on the
parameters. **Backward** follows that calculation in reverse and stores
gradients: local feedback about how tiny parameter changes would affect loss.

Think of a clipboard beside the knobs. Writing advice on it does not turn them.

The notebook copies a weight table before backward. Afterward, the gradients
are nonzero, but the copied values and live weights still match exactly.
Only the optimizer step changes the weights.

This lesson uses **Adam**, which uses gradients plus running summaries of recent
gradients to choose updates. The learning rate sets an overall step scale.
Gradients provide feedback; the optimizer decides how to use it.

Adam's summaries are **optimizer state**, not extra inputs or hidden
activations. Clearing parameter gradients does not reset that memory.
The weight snapshot uses an independent copy: detaching alone would
leave storage shared and make a misleading before/after reference.

Clear the old gradients before the next fresh calculation. Otherwise, new
feedback adds to the old feedback. Accumulating can be deliberate; here it
would be an accident.

**Complaint:** one correction helps, but the curve is nowhere near finished.

## 6. Repeat, then inspect what actually improved

Repeat the same cycle: clear, guess, measure, trace, adjust.

An **epoch** means one pass through the training examples. Here, all 160 pairs
fit in one batch, so each epoch makes one update. With several batches, one
epoch would contain several updates.

The notebook makes exactly **300 updates**, including the first one inspected
separately. It measures loss after each update. Point zero is the untouched
model, not a completed training pass.

In the recorded run, MSE falls from **1.905 to 0.001974**, about a **965-fold**
reduction. The before/after chart shows the learned curve following the
reference. Restarting from the same seed but stopping at 40 passes leaves about
**0.5034 MSE**.

The history has a small early bump. Improvement overall does not mean every
individual step must improve.

These results demonstrate learning on this synthetic task. They do not prove
performance on noisy sensors, unfamiliar ranges, or a held-out real dataset.

**Complaint:** useful predictions should not require another training update.

## 7. Use what was learned without practising again

**Inference** means using the learned model to answer. The notebook tries three
new readings inside the trained range, compares them with the known rule, and
checks that the weights stay unchanged.

Evaluation mode chooses prediction-time behaviour for layers such as dropout.
Inference mode separately skips gradient-tracking work. Neither call performs
a training loop. This tiny model has no dropout, but keeping the distinction
now prevents confusion later.

For language models, readings become token inputs, predictions become word
scores, and the loss checks the right token. The same learning jobs remain.
Token preparation, embeddings, batching, and checkpoints get their own lessons.

## Before moving on

Watch for mismatched shapes, uncleared gradients, backward mistaken for an
update, and attractive plots mistaken for broad evidence.

**Next complaint:** we can run the loop, but what do a tensor's axes, storage,
and gradient state actually mean? [Tensors and Autograd](../02-tensors-and-autograd/02-tensors-and-autograd.ipynb)
opens those boxes.

**Remember:** the model guesses, loss measures, backward gives feedback, and
the optimizer changes the model.
