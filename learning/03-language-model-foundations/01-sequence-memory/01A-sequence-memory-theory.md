# Sequence Memory: Handwritten Theory Notes

Use beside [01A](01A-sequence-memory-theory.ipynb), then test batches in
[01B](01B-sequence-memory-pytorch-lab.ipynb) and longer piano memory in
[01C](01C-cinematic-piano-memory.ipynb).

## 1. Same notes, different tune

Play **D3 A3 D4**, then **D4 A3 D3**. Each contains the same notes once,
but the tunes differ. Counting throws away the distinction.

Ordered input seats preserve it. Batch, time and feature axes answer:
which phrase, which position, and which numbers describe that note?
A dense network with ordered inputs can use order too. Recurrence adds
a reading rule reusable beyond a fixed window.

**Complaint:** how do we read another note without keeping a growing transcript?

## 2. One reader, a changing notecard

The weights are a card-writing recipe: mix the current note, mix the previous
card, combine them, then squeeze the result into a new card. The recipe stays
fixed while reading a phrase; the temporary card changes. Same recipe doesn't
mean equal influence: each note arrives with different numbers and history.

D3 pairs with blank h0 to make h1. A3 pairs with incoming h1 to make h2.
Keep the note and incoming card beside the new card, not just the new card alone.

The new h has two jobs: carry into the next read, and expose its values to a
shared note-prediction head. Here the cell's output and carried hidden state
are the **same h**, not two unrelated vectors. The head makes note scores:
h1 predicts note two, h2 predicts note three, h3 predicts note four.

An old note survives only through the cards it helped make. Each new note
rewrites the same small summary, not a separate slot for each past position.
Its effect can weaken, mix with newer clues, or be overwritten. Farther-back
notes cross more rewrites, so have more chances to get lost in translation.
Forgetting isn't inevitable, but different final cards don't prove distant recall.

**Complaint:** can a wrong ending teach the recipe to preserve the opening?

## 3. Reading goes forward; teaching comes back

Read D3, A3, D4 to make h1, h2, h3 and a next-note output at each read.
The initialized head guesses D3, A3, A3. This replay grades only note four
against D3; the earlier guesses are visible, but aren't additional losses.
When that last guess is wrong, don't update weights yet.

Training kept the earlier cards and their calculations. It inspects h3,
then h2 that helped create h3, then h1 that helped create h2:
**backpropagation through time**. Each inspection calculates how changing
that card and the weights used there would affect the final error.
All uses contribute to the same weights. Collect their corrections, then
update the shared recipe once. One update involves several backward checks.

Full backprop reaches the start of the recorded phrase. Truncated backprop
stops at a chosen boundary. In the notebook, detaching after h1 preserves
every forward card and the guess, but stops feedback entering h1.
That boundary is the **backpropagation horizon**, not necessarily how old
the information in the carried card is. A Transformer's context window
instead limits which earlier positions a prediction can directly consult.

Saving earlier cards doesn't guarantee useful feedback reaches them.
Repeated rewrites may make their influence on the final guess tiny.
Training can inspect the trail and still struggle to teach the opening.
The chosen 65% feedback route falls below 1% after eleven transitions;
it illustrates repeated shrinking, not a universal RNN percentage.

**Complaint:** can a training safety brake bring vanished feedback back?

## 4. Clipping stops a surge, not a fade

Repeated amplification can make gradients explode. Clipping caps an overly
large gradient after backward and before the optimizer step.

A cap cannot enlarge a tiny gradient, recreate forgotten evidence or shorten
its route.

**Complaint:** why should every new note have to rewrite all of our memory?

LSTMs and GRUs add learned controls for retaining old evidence and accepting
new changes. The aim is to carry clues forward and leave a better teaching
route backward. Neither stores every note or guarantees success; the controls
must be learned. First open the LSTM, then the GRU.

## 5. Keep some, write some, show some

The **cell state c** is a fixed-width row of memory numbers carried between
LSTM steps. It is temporary activation data for the current phrase, not trained
weights or a transcript. Its width counts features, not past note positions.

In one deliberately chosen coordinate, write +0.8 for a D3 opening or -0.8
for D4. Later A3 inputs arrive, but KEEP=1 and WRITE=0 preserve the number.
The opening survives as a current numeric value, not an archived note.
Overwrite that coordinate at the final A3 and its distinction disappears.
This is a hand-designed storage example, not a claim that a trained neuron
has that meaning. Real training learns useful patterns across the vector.

Our plain RNN carried one h. An **LSTM** carries a pair with different jobs:

- **Cell state c:** the contents kept and edited through the memory lane.
- **Hidden state h:** a squashed, gated view of the new c. The note head reads
  it now; the next step also uses it, with the next input, to choose controls.

Keep/add updates c without squashing the whole memory after every write.
Squash/SHOW makes h. Keeping contents and exposing a view are different jobs.

Neither is an independent note transcript. Before A3, c1 supplies old contents;
A3 plus h1 choose controls and propose new content. The update makes c2, then SHOW exposes
h2. **Both h2 and c2 travel to D4.** Only h2 feeds the note head directly.

| Part | Job |
|---|---|
| Forget gate | How much old c to **keep** |
| Input gate | How much proposed content to **write** |
| Output gate | How much squashed new c to **show** as h |
| Candidate | Proposed signed content, not a fourth gate |

Controls are dimmers between zero and one. Memory and proposal values can be
signed. Retained old memory plus permitted writing becomes new c; SHOW exposes
part of its squashed view. Closing only SHOW can hide c without erasing it.

Closing SHOW can make h zero while c stays unchanged. A zero h is therefore
not necessarily a blank LSTM. In the actual A3 probe, changing only old c
leaves the controls unchanged but changes new c/h; changing only old h changes
the controls. c supplies what is edited; h helps choose how to edit it.

First choose controls by hand. The zero-SHOW probe leaves nonzero c intact.
A chosen 95% carry route passes about 0.36 of feedback after twenty transitions,
far more than the chosen 65% route. This compares routes, not trained architectures.

Then open the real **A3 step after D3**. Learned Linear projections of input
and previous h produce four chunks: input, forget, candidate, output.
Old c bypasses these projections and enters the keep lane.
Manual arithmetic matches PyTorch's **both** states at every note.

After note six, notes one and six can both influence the same current pair.
Neither is guaranteed recoverable. The head reads h to score note seven;
the chosen seventh note plus old h and c drive the next update. Output h also
feeds an upper layer in a stack; each layer carries its own pair through time.

**Complaint:** correct gate arithmetic still isn't learned memory.

Play either opening, add twenty identical distractions, then request its echo.
The untrained LSTM guesses **D4 D4 D4** for both. At 240 updates it still
confuses them; at 480, both free-running echoes match.
The one-variable drill reproduces the shorter fit's failure.

Erase **both h and c** after the opening: both echoes become **D4 A3 D4**,
neither correct. Reset before every read instead: both become **D3 A3 D4**.
One happens to match; distinguishing both cues is the meaningful test.
Resetting only h could leave evidence in c.

Teacher forcing supplies correct previous answers during fitting, never the
current answer. Generation feeds back actual guesses. The audio plays those
guesses, not hidden values converted into pretend pitches.

**Complaint:** must every gated reader carry two separate vectors?

## 6. One carried note: GRU reset, keep, replace

A **GRU** carries one h and exposes that same h. No separate c or SHOW gate.
It offers a smaller interface, not guaranteed better memory.

RESET controls old-state information entering the proposal. KEEP controls
direct retention of old h; its complement controls writing the proposal.
Unlike LSTM, keeping and writing aren't independent.

Closing RESET doesn't erase the direct keep lane. Keeping everything preserves
old h; replacing everything removes that direct contribution, but the proposal
may still use old h unless RESET is closed too. The scalar interventions
measure these consequences and their feedback.

The real-cell check matches PyTorch. Its reset acts **after** projecting old
state in the candidate branch. Output now and hidden carried onward have
identical values; the separate note head turns them into scores.

Reuse exactly the LSTM's questions, head shape and trainer. The original GRU
fits both echoes in 240 updates. Erase opening h and both echo **D4 A3 D3**:
only one matches. No carry gives **D3 A3 D3**, matching neither whole echo.
Erasure removes forward evidence; detach would preserve values but block feedback.

Different parameter counts and fitting budgets don't establish a ranking.
Both experiments test two training examples, not composition or unseen songs.

### Same cells, different jobs

An encoder-decoder is **wiring**: a reader's final summary initializes a writer.
Either can use LSTM or GRU. The writer also reads a start token and then
previous output tokens while carrying its own state. Attention isn't required.

Gates control **features inside memory**. Optional source attention keeps a
bank of **source-position states**, letting the writer consult every position
before each output. Access doesn't mean equal weight: relevant positions can
contribute more. The weighted view changes, while the bank stays available.
Recurrent source states already reflect order; attention alone doesn't add it.

Keep the full implementation for [03D](../03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb).

## 7. Stop reading blanks; stop grading blanks

Next, 01B separates two batch problems. Packing stops a short phrase's reader
at its last real note; padding alone can change the final state.
Loss masking stops grading blank answers after unpacking.
Stopping reading and stopping grading are different jobs.

## 8. Change the clue, not the explanation

01B changes a future input and checks earlier scores stay unchanged.
01C changes only a practiced phrase's opening cue and measures its ending.
These are causal checks, not claims of new-key generalization.

**Watch for:** confusing changing state with changing weights; expecting a
summary to retain every position; erasing only half an LSTM state; calling
attention equal weighting; treating fitted recall as musical understanding.

**Remember:** carry evidence forward, teach backward, and test what survives
by changing one clue while keeping everything else fixed.
