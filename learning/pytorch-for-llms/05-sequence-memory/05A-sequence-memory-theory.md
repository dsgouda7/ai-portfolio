# Sequence Memory: Handwritten Theory Notes

## 1. The complaint that creates recurrence

A dense network sees a fixed bag of features. A convolution sees a local neighborhood. Melodyne Labs needs the second `twinkle` to depend on an earlier phrase, so the input has a new axis: `(batch, time, feature)`.

The memorable model is a notecard. At each timestep, a recurrent cell reads the current note and the previous notecard, then writes the next notecard. The same cell is reused through time. That reuse lets a short sequence become a chain of computation without creating a new layer for every position.

The tensor contract keeps the roles separate:

- `batch` says which independent examples travel together.
- `time` says where each observation sits in order.
- `feature` says what is known at one timestep.

Collapsing time into features can work only for one fixed window. It loses the reusable "same update at every step" idea. Recurrence keeps that update reusable and produces two related artifacts: an output at every valid timestep and a final state that summarizes the path so far. Sequence labeling needs the first; sequence classification may use the second. Neither artifact is automatically meaningful at padded positions.

**Complaint:** the notecard exists, but can the final mistake teach the opening cue?

## 2. Backpropagation through time

Training unfolds the reused cell into a long chain and sends credit backward through every copy. This is backpropagation through time. The parameters are shared, but the gradient still has to cross one state transition per timestep.

A simple retention factor makes the danger visible. If every transition preserves only 65% of a signal, a cue 20 steps away keeps `0.65^20` of its route. Repeated multiplication, not a mysterious optimizer failure, creates the vanishing gradient.

A vanilla recurrent state can also explode when the repeated multiplier is above one. Clipping limits an update after gradients are computed; it does not repair forgotten evidence.

This distinction matters in debugging. A small opening-cue gradient can mean the route vanished, or it can mean the model already made almost no final error. Always read the gradient beside the loss that created it. Likewise, a large gradient norm is not proof of useful memory; it may be one unstable update. The notebook therefore controls the route multiplier and delay before making an architectural claim.

**Complaint:** clipping can cap an unstable gradient, but it cannot create a long, protected memory route.

## 3. The LSTM cell state

An LSTM adds a cell-state conveyor belt. Four gates manage it:

1. The **forget gate** decides how much old cell state survives.
2. The **input gate** decides how much candidate evidence enters.
3. The **candidate** proposes new content.
4. The **output gate** decides how much cell state becomes visible as hidden state.

Gates are values between zero and one. They are not four separate memories. They are controls around one carried cell state.

The key comparison is the memory route. A vanilla route repeatedly rewrites its state through a nonlinear transform. A well-opened LSTM forget route can stay close to one, so the same 20-step signal remains measurable.

The cell state and hidden state have different jobs. The cell state is the longer-lived conveyor belt. The hidden state is the currently exposed view used by the next recurrent computation and downstream head. An output gate can hide some carried evidence without deleting it. An input gate can reject a distracting note without freezing the entire model. A forget gate can deliberately clear stale evidence; "remember everything" would be as unhelpful as "forget everything."

**Boundary:** an LSTM improves the route; it does not guarantee perfect recall. Training data, hidden width, initialization, and optimization still matter.

## 4. Controlled RNN versus LSTM evidence

Hold the delay constant and change only the memory route. The theory notebook records the gradient reaching the opening cue for both paths. The LSTM route retains more signal at long delays because its cell state has a nearly additive path.

This comparison is evidence about mechanism, not a claim that every LSTM beats every RNN on every dataset. A badly trained LSTM can lose. A short task may not need gates.

Parameter cost also changes. An LSTM computes four aligned gate/candidate blocks where a vanilla RNN computes one state proposal. That extra capacity is justified only when the task needs selective retention. The piano capstone therefore reports both error and parameter count when the learner changes hidden width.

**Complaint:** we understand why the architecture can remember. We still need framework contracts for real batches.

## 5. What the PyTorch lab adds

The next lab begins with a low-noise forecasting task:

`series -> sliding windows -> temporal split -> Dataset/DataLoader -> nn.LSTM -> prediction`

It then switches to the contract needed by language models:

`token IDs -> shifted targets -> embedding -> packed LSTM -> vocabulary logits`

Padding is bookkeeping, not a target. `ignore_index` removes padded positions from cross-entropy, while explicit reduction shows the denominator is the number of real tokens. A causal-invariance test changes future tokens and verifies earlier logits stay unchanged. Gradient clipping is placed after `backward()` and before `step()`.

Packing and masking solve different failures. Packing stops recurrence at each row's real length, so the final state refers to the final real token. Loss masking stops padded targets from changing the objective. You often need both. A packed LSTM with an unmasked loss still rewards pad predictions after unpacking; a masked loss without packing may still return a final state created after several padded inputs.

## 6. Practical failure modes

| Failure | What to inspect |
|---|---|
| Final state represents padding | Pack with real lengths before the LSTM |
| Loss improves by predicting pads | Ignore pad targets and divide by real-token count |
| Validation looks suspiciously strong | Split in time before building validation conclusions |
| Earlier logits change when future IDs change | The model or mask leaks future information |
| Training spikes | Print the pre-clip norm; do not hide it |

**Durable summary:** recurrence carries a notecard, BPTT tests whether teaching can reach it, and the LSTM cell state creates a better route for evidence that must survive many steps.
