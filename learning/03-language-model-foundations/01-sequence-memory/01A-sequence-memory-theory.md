# Sequence Memory: Handwritten Theory Notes

Use beside [01A](01A-sequence-memory-theory.ipynb), then test batches in
[01B](01B-sequence-memory-pytorch-lab.ipynb) and longer piano memory in
[01C](01C-cinematic-piano-memory.ipynb).

## 1. Same notes, different tune

Start with a toy rule: **the fourth note repeats the first**.

| Three notes heard | Required fourth note |
|---|---|
| D3 A3 D4 | D3 |
| A3 A3 D4 | A3 |

Both phrases end in A3 D4, but need different answers. Looking only at the
latest note cannot solve both: the first note must survive two more reads.
This is a made-up exercise, not a universal rule of music.

Play **D3 A3 D4**, then **D4 A3 D3**. Counting gives the same bag of notes,
but our rule needs D3 after the first and D4 after the second. Counts cannot
tell us which note came first. Keep the notes in their original positions
instead. An ordinary neural network can use ordered positions too; an RNN
reads one note at a time and updates its memory using the same rule each time.

**Note names:** D/A = pitch; 3/4 = octave, not a vector dimension.

**Complaint:** the latest note alone is not enough. How does the phrase's context
reach the next guess?

## 2. Carry the context forward

For our rule, D3 A3 D4 needs D3 next, while A3 A3 D4 needs A3. Knowing the rule
is different from knowing this phrase's first note. Training shapes the
weights; prediction also needs the current phrase's context. The small reader
uses hand-picked weights to expose the mechanism, not a trained solution.

One option is to keep the complete ordered phrase and read it again for each
new guess. An RNN instead reads the next note, updates its numerical memory,
and carries that memory forward. Its values change, but the number of memory
values stays the same, even as the sequence grows.

Call that memory h, the hidden state. Our notecard is a picture for it, not a
written list of notes or the training corpus. For the fourth-note puzzle, it
needs enough information about the first note to produce the answer; it does
not need to reproduce every input. Useful details can be lost.

Training helps it learn what to retain. Individual memory values need not
have labels such as "first note"; several numbers can preserve the clue together.

Three notes would be easy to store completely. We use this tiny example to
see how recurrent memory works, not to claim that three notes require compression.
Our arithmetic reader uses hand-picked inputs and weights to show the mechanics.

**From note to numbers:** D3/A3/D4 name the notes; `x` is the numeric input
representing the current note, not another memory state.

| Note on the keyboard | Reader input | Incoming memory | New memory |
|---|---|---|---|
| D3 | `x1 = [1, 0]` | `h0 = [0, 0]` | `h1` |
| A3 | `x2 = [0, 1]` | `h1` | `h2` |
| D4 | `x3 = [1, 1]` | `h2` | `h3` |

`x1` encodes D3; `h1` is computed after reading D3. They are different objects.
The diagram's `D3 + h0` means two reader inputs: separate weights mix the
encoded note and incoming memory before adding their contributions.

**Notation:** `x` = input vector; `h` = running memory; `W` = shared weight matrix;
`b` = bias (added offset); `t` = note step; `d` = vector width.
In `x3/h3`, 3 is the step; in `D3`, 3 is the octave.

Each read mixes the new input with old h and squeezes the result with tanh.
This keeps each value between -1 and +1 and makes the update nonlinear,
not just another weighted sum.
The RNN cell makes memory h first, not a guessed note.

A separate **next-note prediction layer** reads h and gives D3/A3/D4 each
a score; the highest score becomes the guess. That extra layer is called a
**prediction head**. It has its own weights, `W_out`. **Shared** means the
same prediction layer is reused after every note, not a new layer per step.
It reads h without changing it; the same memory goes to the next RNN read.
After D3, h1 helps guess note two; after A3, h2 helps guess note three.
An independent phrase starts with a fresh card; the shared weights remain.

> **Optional:** The same D3 A3 D4 reader, written out (`h0 = 0`; this toy omits bias).
>
> **Notation:** $x_{\mathrm{D3}}$ = D3's input vector; $h_i$ = memory after note $i$;
> $W_x$, $W_h$ = shared input/memory weight matrices; $W_x x$ = matrix-vector multiply; $\tanh$ = elementwise squash to $(-1,1)$.
>
> $$
> \begin{aligned}
> h_1 &= \tanh(W_x x_{\mathrm{D3}} + W_h h_0) \\
> h_2 &= \tanh(W_x x_{\mathrm{A3}} + W_h h_1) \\
>     &= \tanh(W_x x_{\mathrm{A3}} + W_h \tanh(W_x x_{\mathrm{D3}} + W_h h_0)) \\
> h_3 &= \tanh(W_x x_{\mathrm{D4}} + W_h h_2) \\
>     &= \tanh(W_x x_{\mathrm{D4}} + W_h \tanh(W_x x_{\mathrm{A3}} + W_h \tanh(W_x x_{\mathrm{D3}} + W_h h_0)))
> \end{aligned}
> $$
>
> Each note's input mixes with the previous, already-tanh-ed h before another tanh; the same weights are reused.

Open the actual D4 step: its input `x3 = [1, 1]` and old h2 are separate. h2 contains
`[0.1894, 0.7090]`, not the written notes D3 and A3. The input weights give
`[0.5000, 1.1000]`; the memory weights give `[0.1846, 0.3356]`.
Add, then squash: new h3 is `[0.5945, 0.8928]`. Those are context features;
the extra prediction layer reads them to make next-note scores.

`W_x/W_h/W_out` mix input/memory/head features. `@` (or `W x`) is matrix-vector
multiplication; `*` is scalar/elementwise multiplication, not matrix multiplication.

**Complaint:** can the wrong ending teach the recipe what to preserve?

## 3. Read forward once; inspect the trail backward

The initialized head guesses D3, A3, A3. Grade only note four against D3 in
this close-up. Do not update the weights yet.

Training saved h1, h2, h3 and their calculations. It follows the error through
the head, then inspects the D4/h2, A3/h1 and D3/h0 reader steps in reverse.
Each inspection asks how a small change there would affect the ending.
All uses contribute to the same weights; collect contributions, then update
the recipe once. This is backpropagation through time.

Full backprop reaches the recorded phrase's start. Truncated backprop stops
at a detached boundary. Detaching after h1 preserves every forward card but
blocks feedback into h1. That is a backpropagation horizon, not a maximum age
for carried information or a Transformer's forward context window.

You choose this window in the **training loop**, not the RNN layer's constructor.
Choose a chunk length, run backward per chunk, then detach the carried state
before the next chunk: h for RNN/GRU, both h and c for LSTM. The numbers carry
forward; gradients do not cross that boundary.

**Complaint:** examining an old card does not guarantee useful feedback reaches it.

## 4. Small states can have fading or surging feedback

This RNN's tanh keeps h between -1 and +1. Sigmoid gate controls later use
zero to one. Think of tanh as a soft squeeze: near zero, a tiny push mostly
passes through; near 3, the output is already almost +1 and hardly moves.
These are test values before tanh, not notes or stored h.
That responsiveness is the local gradient: a step that barely reacts forward
also passes little feedback backward. One static curve and a matching
tiny-nudge comparison show both regions; no separate tanh lesson is needed.
Repeated weak responses can make the opening hard to teach; recurrent weights
can also amplify feedback, as the next replay shows. Tanh alone cannot amplify it.

The player returns to D3 → A3 → D4 and h0 → h1 → h2 → h3.
It keeps the note encodings but uses one hidden feature to expose the arithmetic.
The same chosen weights are reused, not randomized at each note.
**Previous-memory multiplier** is the shared recurrent weight: how strongly
old h enters the next read. **Input multipliers** are separate weights for
the current note's two numbers.
Grey weights stay fixed during a slider sweep; gold changes; cyan values are recomputed.
Each block shows incoming h times the multiplier before adding the current
input's contribution. A branch after each note shows the probability and
loss for the actual next note: A3, D4, then D3. Earlier losses are preview
readouts, not extra training objectives. The first prediction stays fixed
because h0 is zero; later predictions respond to the multiplier.

The player labels loss as **Error** and backward feedback as **Ending-error
response at h**, rather than bare L/g symbols. Selecting a block shows two
actual reruns with memory slightly lower/higher and all weights fixed.
Divide the error difference by the memory difference: that rate is g.
Positive means increasing h raises error; negative means it lowers error;
near-zero means little error guidance reaches this state. These are not updates.

### BPTT in RNNs: where the weight gradient comes from

Forward makes the prediction; backward asks tiny "what if?" questions.
Nudge one saved h, keep the weights fixed, and rerun the head.
How much does ending loss change per unit of that nudge? That response is
the derivative, not the loss or a weight update.

A nudge to an earlier h passes through later reads before reaching the loss.
Multiply their responses along that route. But changing the shared memory
multiplier affects every read, so add the contributions from its uses.
At one use, incoming h is held fixed to isolate the direct multiplication.
It still has a history: earlier uses account for how the multiplier changed
that incoming h. The product rule counts the direct and inherited effects;
it does not invent an extra correction.

In the one-feature 0.8 toy, D3/A3/D4 contribute approximately
0, 0.00060022 and 0.00022508: total 0.00082530. SGD at 0.02 changes the
multiplier to 0.79998349. The first contribution is zero because incoming h0
is zero, not because feedback to h1 vanished. The notebook's
[handwritten sheet](images/01a-bptt-memory-weight-explainer.png) shows these
two routes and circles the repeated weight powers; the full matrix form is optional.
The notebook keeps the detailed linear/exact substitutions collapsed so you
can follow the memory story without repeating the arithmetic.

The dark sheet follows the whole route: weight changes memory, memory
changes the correct-note probability, and that probability changes error.
Multiply those local responses. Its full product-rule derivative counts
both the current multiplication and how earlier uses changed old memory;
substituting earlier derivatives exposes the repeated powers. The actual
tanh reader also multiplies by tanh responsiveness.

The final guess is A3, but the correct next note is D3. Only that ending is
graded. The note head turns its probability error into starting feedback g3.
The loss describes the mistake; gradients guide corrections to the shared
recipe. Earlier notes are more handoffs away from the ending, so their
guidance can be muffled or amplified more times before weights are updated.
The preview losses do not add feedback to this trace. To reach h2, g3 crosses
the D4 cell's tanh response and memory multiplier. To reach h1, it crosses
the A3 cell's pair as well. Earlier feedback has travelled through more factors.
At 0.8 the three-state gradients are about 0.480, 0.600, 0.750;
at 1.2 they are 1.081, 0.901, 0.751. Read backward from the last number.
These are loss sensitivities, not stored memory, note importance or weight updates.

For a concrete contrast, the 0.8 run gives D3 probability 0.33315 and ending
loss 1.09917. The head supplies feedback about 0.750, not the loss itself.
Backward, 0.750 times 0.8 gives 0.600; another times 0.8 gives 0.480.
At 1.2, probability is 0.33300 and loss 1.09962, but feedback grows:
about 0.751 times 1.2 gives 0.901, then about 1.081.
These rounded examples include an almost-unit tanh response in the actual player.
The loss is not being reduced as we walk backward; only its sensitivity changes.

Three notes expose two links. Repeat the phrase seven times to see twenty:
first/last feedback is about 0.011529 at 0.8 and 37.515271 at 1.2.
Tanh's local response is also involved, so a multiplier above one is not
by itself a guarantee of exploding feedback.

Each cell also contributes to the same input and memory weight gradients.
Backpropagation collects these contributions; it does not adjust weights or
repair the error while walking backward. Sum first, then update each shared
weight once. The player uses native SGD at 0.02 and actually reruns the same
phrase with the new weights, showing the changed guess and cross-entropy loss.

Vanishing gradients give almost no guidance about how the opening was
processed. Exploding gradients can make the optimizer change weights too
drastically. Neither term says the stored numbers vanish or explode.
Weak feedback makes it harder to learn which early clues to preserve;
the shared weights can still update, but little ending-specific guidance
reaches their early uses. This does not erase an existing card. Amplified
guidance can make an update overshoot; numerical overflow is a possible
extreme, not an inevitable outcome.

Clipping caps a large gradient after backward and before the optimizer step.
The measured writing-weight update shrinks on the amplified route; the weak
update stays weak. Clipping is a brake, not a memory repair.
That check is a separate scalar target-1, half-squared-error experiment
with SGD step size 0.01, not the note model's cross-entropy loss.

**Complaint:** why force a clue through a fresh remapping and squeeze at every read?

## 5. Protect the memory; squash the view

Keep the original note codes, but reuse the reader's fixed input weights
to write D3 = [0.70, 0.30], A3 = [-0.20, 0.80], D4 = [0.50, 1.10].
These decimal writes are added, not new note encodings.
Both D3 A3 D4 A3 and D4 A3 D3 A3 finish at [0.80, 3.00].
Each feature passes first-write feedback unchanged. The two-number total
keeps amounts, not their order: it is not a timeline.
Nudge the first write from [0.70, 0.30] to [0.71, 0.30]: the final total
becomes [0.81, 3.00]. That is 100% carry-through of a change.
Plain addition loses order; a linear recurrence with recurrent mixing need not.

**Refinement:** choose what to keep, what to write, and what to expose.

LSTM c is a fixed-width row of memory numbers: activation data for the current
phrase, not weights, raw inputs or a complete past-note transcript. Training
learns useful patterns and rules for retaining them.

The inputs stay two-dimensional. Now open one internal workspace coordinate:
+0.8/-0.8 are chosen memory writes in a new storage probe, not the previous
vector total or new input encodings.
Write +0.8 for D3 or -0.8 for D4, then close WRITE
while later notes arrive. The current number preserves the opening. Overwrite
it and that coordinate loses the distinction. Real representations need not
give individual coordinates neat note labels.

KEEP retains part of old c; WRITE admits part of a signed candidate.
They are independent controls. The update keeps/adds without squashing all
new c. SHOW selects from tanh(new c) to make h.
These controls are computed values; trainable shared weights calculate them.
We set the values directly only in the hand-controlled sketches.

Both travel onward: c supplies contents; the next input and h choose controls
and proposed content. Only h directly feeds the note head. Closing SHOW can
make h zero without erasing c.

The forward walkthrough opens the real A3 read after D3. The note and old h
choose HOW; old c supplies WHAT. Kept and written contents make new c.
SHOW exposes part of its bounded view as new h. The head reads that h to
predict a note; it does not read c directly. Both states still have two
features and travel together to the next read.

**LSTM key:** `f/i/o` = KEEP/WRITE/SHOW; `g` = signed proposal;
`c1` = memory after note 1, while `c[0]` selects its first feature.
This proposal `g` is not BPTT's error-response `g`.

With fixed KEEP near 1, direct carry feedback survives better. KEEP 0.99
retains about 0.818 after twenty transitions; 0.9 retains about 0.122.
Five +0.8 writes can make c = 4 while h remains below 1. That h readout can
still saturate and weaken feedback.

The native stress test changes a candidate's old-h weight while KEEP stays
0.99. Another recurrent route can amplify the full gradient. The direct
KEEP derivative is not the whole learned-cell derivative. Protected memory,
sensible updates and clipping are complementary, not guarantees.

**Complaint:** a correct memory mechanism still needs to learn what matters.

The native arithmetic matches both LSTM states. Now make the opening puzzle
harder: retain the whole three-note opening through a longer gap, not just
its first note through two reads. The middle is A3 D4 A3 D3 repeated five
times: twenty note events, identical for both openings. A silent PLAY request
asks for D3 A3 D4 or its reversal, D4 A3 D3, in order.
At 240 updates the LSTM still confuses them; at 480 both fitted echoes match. Erasing both h
and c removes their distinction. This is fitted recall, not composition.
The native microscope uses two-feature states with initialized weights;
the fitted echo uses four input profiles and twelve features per state.

## 6. A GRU protects old h outside the proposal

Could one vector provide a useful retention route? GRU carries and exposes
the same h, with no separate c or SHOW gate.

**GRU key:** `r/z/n` = RESET/KEEP/proposal; `1-z` = WRITE.
Gate fractions multiply matching features; `n` here is not a note-position label.

Tanh forms the proposed replacement. KEEP retains old h directly; its
complement writes the proposal. The combined result is not squashed again.
With a valid bounded starting h and bounded candidates, this blend stays
bounded while retained content avoids the proposal's squash.

RESET controls old-state information entering the proposal, not the direct
KEEP lane. Close RESET and KEEP can still preserve old h.

With RESET closed and fixed controls, twenty KEEP fractions determine
old-state feedback. Learned gates and candidate paths add other responses.
The native GRU stress test shows amplification despite KEEP = 0.99 and bounded
states. Clipping remains a training guardrail here too.

GRU keeping/writing are complements; LSTM's are independent. GRU cannot hide
a separate memory while leaving it intact. Neither architecture is always
better or immune to vanishing/exploding gradients.

The original GRU fits both echoes in 240 updates. Erasing opening h makes the
answers identical. Different budgets and parameter counts are not a ranking.

## 7. Memory, wiring and access are different jobs

An encoder-decoder is wiring: a reader's summary initializes a writer.
Either can use LSTM or GRU. The writer carries state and reads previous outputs.

Optional source attention keeps source-position states available to revisit.
Access does not mean equal weight. Gates control features inside recurrent
memory; attention selects among saved positions. The implementation continues
in [03D](../03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb).

## 8. Test the clue, not a convenient curve

01B separates stopping the reader at a phrase's end from skipping padded
answers in the loss. It also changes future input and checks earlier scores.
01C changes one opening cue and measures its ending.

**Watch for:** confusing weights with states; bounded values with stable
gradients; unsquashed memory with a transcript; a protected lane with a
guaranteed full gradient; fitted recall with general musical skill.

**Remember:** carry useful evidence forward, teach backward, protect both
routes where possible, and measure what survived.
