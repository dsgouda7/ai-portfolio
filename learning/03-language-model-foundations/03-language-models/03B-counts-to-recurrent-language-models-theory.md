# Counts to Recurrent Language Models: Handwritten Theory Notes

**Chapter 03B:** [Notebook](03B-counts-to-recurrent-language-models.ipynb)
· [Code-only Python](03B-counts-to-recurrent-language-models.py).
[03A](03A-training-skeleton-and-word-objectives-theory.md) gave words learned rows.
Keep those rows, give positions seats, then carry a clue beyond the seats' left edge.
The recurrent mechanisms belong to [01A](../01-sequence-memory/01A-sequence-memory-theory.ipynb);
here you check what they do inside a word-prediction task.

## 1. First ask what the predictor receives

You want the next word after `the cat sat on the mat`. Require `purring`; replace
only `cat` with `dog`, and require `panting`. This is a deliberately made-up
final-word rule, not a claim about animal behavior or general English.
The first `the` is common; the animal at position 1 decides the ending.

Split at spaces. Each piece is a token, a unit you count or predict. Eight controlled
sentences combine two animals, `sat` or `slept`, and `mat` or `rug`. Six train;
two hold out `sat` with `mat`. Training includes `sat` with `rug` and `slept` with
both surfaces. This tests a new combination of familiar words, not vocabulary growth.

A unigram count records individual tokens. A bigram count records neighboring
pairs. After `mat`, the training table contains one `purring` and one `panting`.
Its best constant final answer gets 50% right. Counts are a useful, inspectable
baseline, but better counting cannot identify an animal excluded from the key.

**But there's a catch:** longer exact count contexts become sparsely observed.
Could learned features share evidence across contexts while retaining word order?

## 2. Give word rows separate seats

An ID is an address, not a word's meaning. It retrieves an embedding: a trainable
row of numbers learned from the prediction objective, not a supplied definition.

An average is a bag. Shuffling the same rows leaves the mathematical average
unchanged; tiny floating-point differences are rounding, not recovered order.
Instead, put each row in its own seat and join the rows in seat order. A layer
can now react differently to the first and third positions.

Different untrained ordered scores show sensitivity to a permutation, not useful
prediction. Training then adjusts both the rows and predictor.
The `<start>` address fills empty left seats; it is not an observed corpus word.

**But there's a catch:** seats preserve order, but the window still has a left edge.

## 3. Optimization cannot recover a removed clue

The held-out prefixes differ only at `cat` versus `dog`.
Their last three words, `on the mat`, are identical. The fitted window receives the
same IDs in the same seats, and the notebook verifies exactly matching probability
rows. This invariant does not depend on how well the optimizer worked.

Two different required answers cannot both be the top choice of one identical
prediction. A bad model might choose neither answer. More training still cannot
distinguish the prefixes without changing the information it receives.

Inspect four visible words: `sat` reappears, but the animal remains absent.
Five visible words recover it. This is not a trained five-word comparison:
rebuild the input layer and fit it again. Access is necessary, not sufficient.

**What if you pass the animal clue forward instead of cutting it off?**

## 4. Follow a clue through a running note

A recurrent reader fetches each word row, updates its hidden state, and scores
the following word. Embeddings belong to word addresses; hidden states depend
on ordered prefixes. These are different objects with different jobs.

01A already taught shared recurrent weights, memory routes, clipping, and
next-event training. Here the new check is a word-level intervention against the
window's specific blind spot. Keep the fitted window, corpus, split, and other words.
Change only the animal and inspect the final probability distribution.

Read three pieces of evidence together: whether the internal states differ,
whether the final predictions change, and whether those predictions are correct.
A state-distance curve answers only the first question. Its six-frame replay uses
recorded values, not a fabricated fading curve. States agree after the common first
`the`, separate after the animal, and remain distinct through the later words.
The measured distances are 0, 3.2087, 3.4742, 3.3803, 4.4962, and 3.6518.

In the validated CPU run, final accuracy was 50% for the window and 100% for the
RNN on these two rows. Swapping only the animal changed the largest final probability
by 0.999049. Correct endings make that retained distinction useful for this tiny rule.
The experiment does not establish general understanding.

The RNN has a route beyond the window, not guaranteed recall. Parameter counts,
learning rates, and update budgets differ between models. This is a documented
tiny-task comparison, not an architecture-only contest or a long-context benchmark.

**But there's a catch:** carrying a useful clue through four later words does not establish
reliable memory over hundreds of positions.

## 5. Use LSTM memory without another training framework

Return to 01A for the gate experiments. LSTM carries hidden and cell state with
learned controls; GRU offers another controlled update. Better memory does not
automatically let a writer reopen every earlier source position.

The language-model application keeps the same interface: IDs retrieve word rows,
the recurrent layer produces a hidden output at each position, and a shared
projection scores vocabulary entries. The cell state helps carry memory; it is
not a second answer to grade.

Pair `the cat sat on the mat` with `cat sat on the mat purring`. Targets are already
one word ahead before entering the inherited trainer. Do not shift them again.
Flatten batch and time for the loss while keeping vocabulary scores together.
One backward pass teaches the embedding, recurrent layer, and output projection.
The notebook measures an actual embedding-weight change rather than assuming the
rows learned because they were declared trainable.

The validated LSTM loss fell from 2.4430 to 0.2988; its held-out final accuracy was
100%, and the largest embedding change was 1.071712. Its 3,627 parameters differ
from the RNN's 1,179 and window's 963. Both practice on 36 next-word lessons.
These numbers document a run, not a ranking.

## 6. Take away the answers; keep the boundary honest

Changing the final input from `mat` to `rug` left earlier logits exactly unchanged.
Check scores directly, not merely whether a generated sentence looks plausible.
Generation supplies no answer tensor: append the model's own chosen word.
The fitted LSTM continued the full cat prefix with `purring` and its dog counterpart
with `panting`, without receiving those endings as inputs.
The readable helper recomputes its prefix and uses a fixed step budget. It neither
caches state nor learns when to stop.

Read the measured scorecard beside these limits. Never interpret ID magnitude as
meaning, mistake state distance for useful recall, double-shift targets, or call
two held-out rows a benchmark. Good short-task behavior is not general language quality.

You can now continue one stream. What if a completed source must condition a
different answer sequence? [Chapter 03C](03C-recurrent-encoder-decoder-to-attention.ipynb)
separates reading from writing and investigates source access.

**Durable summary:** learned features use visible evidence; recurrence creates a
route for earlier evidence; intervention tells you whether the trained predictor used it.
