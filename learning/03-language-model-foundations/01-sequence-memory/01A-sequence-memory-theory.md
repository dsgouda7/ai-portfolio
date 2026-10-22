# Sequence Memory: Handwritten Theory Notes

Use beside [01A's notebook](01A-sequence-memory-theory.ipynb), then test the
contracts in [01B](01B-sequence-memory-pytorch-lab.ipynb) and listen to
[01C](01C-cinematic-piano-memory.ipynb).

## 1. Same notes, different tune

Play **D3 A3 D4**, then **D4 A3 D3**. Each phrase contains the same notes
once, but they are different tunes. A count card cannot distinguish them:
we threw the order away before the model saw it.

Separate ordered seats keep that distinction. A batch now has three questions:
which phrase, which position in that phrase, and which features at that position?
Those are the batch, time, and feature axes.

A dense model with ordered input seats can use order too. Recurrence isn't
the only way to preserve it. Recurrence adds a reusable reading rule: keep
applying the same update instead of requiring another fixed input seat forever.

**Complaint:** how do we read another note without keeping the entire transcript?

## 2. One reader, a changing notecard

At each step the reader gets the current note and its previous notecard.
It writes a fresh card, then passes that card onward.

The card is a short row of numbers, not a literal sentence. Its width is fixed,
so it must compress the history. A new independent phrase starts with a fresh
blank card.

The reader's weights stay the same throughout a pass. The card changes at
every step. Three boxes drawn across time mean three uses of one reader,
not three independently trained readers.

The notebook uses hand-picked two-number inputs for D3, A3, and D4.
Reversing those inputs changes the final card while the weights stay fixed.
That proves order can affect the computation, not that the model learned music.

**Complaint:** carrying a card forward doesn't guarantee that training can
teach the opening clue.

## 3. Reading goes forward; teaching comes back

Suppose the final guess is wrong. Backpropagation follows the saved steps
backward to calculate how earlier computations contributed to that mistake.
This is **backpropagation through time**.

Each use contributes to the gradients of the same shared weights. The
optimizer changes those weights afterward. It doesn't edit the input notes
or turn the changing hidden cards into separately trained parameters.

Consider a deliberately simple route passing on 65% of the feedback at
every transition. Two transitions retain 65% of the remaining 65%.
Repeated shrinking compounds: the notebook's autograd check falls below
1% after 11 transitions.

The 65% value is chosen, not a universal RNN constant. Real derivatives
depend on inputs, weights, and nonlinearities. The toy isolates why a long
route can become hard to teach.

**Complaint:** can a training safety brake recover the missing signal?

## 4. Clipping stops a surge, not a fade

Repeated shrinking can make gradients vanish. Repeated amplification can
make them explode. Gradient clipping caps an overly large combined gradient
after backward and before the optimizer step.

A cap doesn't enlarge a tiny gradient. It cannot recreate forgotten evidence
or shorten the route from a final error to an early clue.

Small gradients also need context: a nearly correct final answer may require
almost no correction. Read error beside its feedback, rather than treating
every small gradient as a memory failure.

**Complaint:** we need a better way to carry evidence, not just safer updates.

## 5. Keep some, write some, show some

An LSTM adds a carried cell state and three learned controls:

| Part | Memorable job |
|---|---|
| Forget gate | How much old memory to **keep** |
| Input gate | How much proposed content to **write** |
| Output gate | How much squashed memory to **show** |
| Candidate | The proposed new content; not a fourth gate |

The controls are dimmers between zero and one. Candidate and memory values
can be signed. Real controls are learned from the current input and previous
exposed hidden state; nobody hand-labels a column "remember D3".

The new carried memory combines retained old evidence with permitted new
content. The exposed hidden state is its squashed, output-controlled view.
Closing the show control can hide memory without deleting it.

01A chooses the controls by hand to make this distinction inspectable.
Its zero-show probe produces zero exposure with nonzero carried memory.
A second scalar route keeps 95% per step and passes much more feedback
than the 65% route at the same delay.

These are mechanism checks, not trained-architecture scores. An LSTM can
still forget, overfit, or learn bad controls.

**Complaint:** which card is "final" when a short phrase sits beside a longer one?

## 6. Stop reading blanks; stop grading blanks

01B starts with 24-value forecasting windows. It checks later answer times
against a persistence guess: repeat the last observation. Earlier history
may overlap; the validation answer still stays outside its own input.

Then it returns to piano notes. For **D3 A3 D4 F4 A4**, the input is
**D3 A3 D4 F4** and the answers are **A3 D4 F4 A4**. Each position predicts
one note ahead. Fixed IDs fetch trainable rows before the LSTM reads them.

The shortest input, **D3 A3**, needs three blank seats to fit the batch.
Reading those zeros still updates the LSTM. Its final state can drift away
from the card made immediately after A3.

Packing stops each reader at its real length. The notebook compares padded,
packed, and solo readings through the same weights.

Loss masking solves another problem: after unpacking, the prediction head
can still score blank seats. Grade only real answers. The explicit real-target
average matches the ignore-pad loss; extra blank columns leave it unchanged.

**Complaint:** could an apparently good guess have peeked at the future?

## 7. Change the clue, not the explanation

01B changes a later F4 input to C4 while keeping the earlier prefix fixed.
Earlier score rows must match in a left-to-right LSTM. This checks future
access, not whether those guesses are musically correct.

01C makes a different intervention. Two practiced phrases share the same
clock but start with different key cues. The desired endings differ by three
semitones. Swapping only that opening cue tests whether its effect survives
to the ending. Removing both cues makes the inputs identical, so no
deterministic reader can fit both different endings exactly.

This cue is a numerical pitch feature, not a vocabulary ID. Differentiating
a copy diagnoses its influence; the optimizer still updates model weights,
not the training data.

Read the actual pitch shift, error, parameter counts, and local cue sensitivity.
Neither good audio nor a nearby derivative establishes new-key generalization.

**Remember:** carry a card forward, send teaching backward, and test what
survives by changing one clue while keeping everything else fixed.
