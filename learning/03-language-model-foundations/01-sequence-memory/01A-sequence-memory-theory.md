# Sequence Memory: Handwritten Theory Notes

Use beside [01A](01A-sequence-memory-theory.ipynb), then test batches in
[01B](01B-sequence-memory-pytorch-lab.ipynb) and longer piano memory in
[01C](01C-cinematic-piano-memory.ipynb).

## 1. Same notes, different tune

Play **D3 A3 D4**, then **D4 A3 D3**. Counting gives the same bag of notes,
but the tunes differ. Ordered seats retain a distinction that pooling removes.
Recurrence adds a reading rule reusable beyond a fixed input window.

**Note names:** D/A = pitch; 3/4 = octave, not a vector dimension.

**Complaint:** the latest note alone is not enough. How does the phrase's context
reach the next guess?

## 2. Carry the context forward

After D3 A3 D4, D4 alone can have different continuations in different phrases.
Training teaches the weights musical patterns; prediction also needs the
current phrase's context. The model applies learned patterns, not a lookup
of stored songs.

One option is to keep the complete ordered phrase and process it again for
each new guess. An RNN instead carries forward the result of processing the
past. Each new note updates a fixed-size working summary without starting
again from the opening. The carried state stays the same size as the phrase
grows; useful details can be lost.

Our notecard is that summary, h: a numerical description of the current
phrase, not the training corpus or a lossless code for every note. The card
is a reminder analogy, not a musician's literal workflow. Training shapes
which distinctions help prediction: for the later echo, distinguish the
openings rather than store every distraction separately.

Three notes fit in a transcript. They make this reusable reading process
inspectable; they do not prove compression is necessary. Relevant history
is needed, but a fixed-size summary is this architecture's choice.
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
also passes little feedback backward. Use the saved tracker at 0 and 3 to see it.
Repeated weak responses can make the opening hard to teach; recurrent weights
can also amplify feedback, as the next replay shows.

The weights first generate a prediction; comparing it with target 0.5 creates
the error. Then gradients travel backward through the recorded reads.
Drag only the memory weight: the 0.8 and 1.2 presets show vanishing and exploding
feedback. Around 1, feedback is nearly unchanged. Input, writing weight and
prediction weight stay fixed; the slider changes a weight, not the output directly.
Here the recurrent weight is a reusable parameter; the stored hidden state h
is the changing numeric memory. They are different objects.

Both guesses are wrong and all memory values remain small and nonzero.
Yet the opening input weight receives a gradient of about -0.005764 in the
vanishing case, versus -17.616600 in the exploding case. These are actual
parameter gradients, not weight-update sizes. The toy uses half-squared error,
not note classification; no optimizer step is shown.

The optional appendix's older graph is a per-unit reference: final sensitivity is set to 1,
independently of the error size. Its opening values are about 0.00481 for cyan,
38.34 for red and 0.00000802 for the piano reader. Its logarithmic axis makes
tiny responses visible; these numbers are not memory amounts or probabilities.

Vanishing gradients give almost no guidance about how the opening was
processed. Exploding gradients can make the optimizer change weights too
drastically. Neither term says the stored numbers vanish or explode.
Weak feedback makes it harder to learn which early clues to preserve;
it does not erase an existing card.

Clipping caps a large gradient after backward and before the optimizer step.
The measured writing-weight update shrinks on the amplified route; the weak
update stays weak. Clipping is a brake, not a memory repair.

**Complaint:** why force a clue through a fresh remapping and squeeze at every read?

## 5. Protect the memory; squash the view

Try removing the squeeze and adding every note's proposal. With D3 = +0.8,
D4 = -0.8 and A3 = +0.2, two different opening orders both finish at +0.4.
Feedback to the first written number remains 1. Perfect additive feedback
did not make the representation preserve order.

**Refinement:** choose what to keep, what to write, and what to expose.

LSTM c is a fixed-width row of memory numbers: activation data for the current
phrase, not weights, raw inputs or a complete past-note transcript. Training
learns useful patterns and rules for retaining them.

In a chosen coordinate, write +0.8 for D3 or -0.8 for D4, then close WRITE
while later notes arrive. The current number preserves the opening. Overwrite
it and that coordinate loses the distinction. Real representations need not
give individual coordinates neat note labels.

KEEP retains part of old c; WRITE admits part of a signed candidate.
They are independent controls. The update keeps/adds without squashing all
new c. SHOW selects from tanh(new c) to make h.

Both travel onward: c supplies contents; the next input and h choose controls
and proposed content. Only h directly feeds the note head. Closing SHOW can
make h zero without erasing c.

**LSTM key:** `f/i/o` = KEEP/WRITE/SHOW; `g` = signed proposal;
`c1` = memory after note 1, while `c[0]` selects its first feature.

With fixed KEEP near 1, direct carry feedback survives better. KEEP 0.99
retains about 0.818 after twenty transitions; 0.9 retains about 0.122.
Five +0.8 writes can make c = 4 while h remains below 1. That h readout can
still saturate and weaken feedback.

The native stress test changes a candidate's old-h weight while KEEP stays
0.99. Another recurrent route can amplify the full gradient. The direct
KEEP derivative is not the whole learned-cell derivative. Protected memory,
sensible updates and clipping are complementary, not guarantees.

**Complaint:** a correct memory mechanism still needs to learn what matters.

The native arithmetic matches both LSTM states. The echo task then asks the
model to retain one of two openings across twenty distractions. At 240 updates
the LSTM still confuses them; at 480 both fitted echoes match. Erasing both h
and c removes their distinction. This is fitted recall, not composition.

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
