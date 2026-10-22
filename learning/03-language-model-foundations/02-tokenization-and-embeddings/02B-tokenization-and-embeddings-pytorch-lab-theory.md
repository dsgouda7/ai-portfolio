# From Lookup to a Learned Shortlist: Handwritten Theory

Use beside [02B's notebook](02B-tokenization-and-embeddings-pytorch-lab.ipynb).
[02A](02A-tokenization-and-embeddings-theory.ipynb) supplied the mental models;
this lab checks them in PyTorch.

**Carry-in:** lookup works. What feedback could make the stored rows useful?
The **lookup microscope**, **the ___ sat**, recreates the three hand-picked
two-number rows. Its numeric predictor trace is still untrained arithmetic.
The **learning game**, **cat/dog -> sat/slept; bird -> flew/sang**, creates a
different seven-word vocabulary, random four-number rows, and a separate
predictor. Those two sets of weights learn from six supplied cards.

The map stays text -> pieces -> IDs -> fetched rows -> context processing ->
scores -> checked answer. Here we expose the feedback from checking an answer.
IDs remain fixed; stored weights change. The isolated gradient probe resets
again so one card's effect is visible. These are explicit experiment resets,
not weights carried forward from the microscope.

## 1. Bring the same note, not an invisible previous kernel

The familiar note is **the sleepy cat sat on the mat**. The lab recreates
the same six notes used in 02A, including both meanings of `bark`.
You don't need variables left over from another notebook.

There are three different artifacts to keep apart: rules that split text,
IDs that name pieces, and numbers stored in model rows. Borrowing the first
two from GPT-2 does not borrow the third.

We'll inspect the real tokenizer, but train a deliberately small separate
word model. That switch makes the learning signal visible instead of burying
it inside a large vocabulary.

**Complaint:** before trusting a tokenizer, can we explain what its joins do?

## 2. Learn the joins, then freeze and replay them

The toy BPE has three helpers: split words into character pieces, count
neighbouring pairs with their frequencies, and join one chosen pair.
Repeat that loop to learn an ordered merge list.

Using the tokenizer is a different job. Keep that list fixed and replay it
on the new word. Do not learn extra joins while encoding each example:
then the same text could depend on when it was seen.

With 20 learned merges, `sleepy` becomes one piece. Replaying only the first
eight leaves five pieces. The unseen `sleepyhead` reuses `sleepy` and keeps
`h`, `e`, `a`, and `d`, plus the end marker.

The marker helps the toy respect word boundaries. Production byte tokenizers
handle text differently, but common pieces still get reused.

**Complaint:** this small rule is inspectable. Does a real tokenizer obey a
checkable input/output rule too?

## 3. Trust the round-trip, not a pretty token label

GPT-2 uses byte-level BPE with a pretrained vocabulary. The lab inspects
IDs, raw token bytes, and the decoded text.

Leading space matters: ` cat` has ID 3797, while `cat` has ID 9246.
Those are different pieces, not two degrees of cat-ness.

The reliable check is exact reconstruction after encoding and decoding.
An isolated piece can look odd because of whitespace or byte boundaries.
That doesn't automatically mean the tokenizer broke the text.

For our six notes joined with newlines, the lab measures 166 characters and
43 tokens, about 3.86 characters per token. That describes these notes,
not all languages or all writing.

**Complaint:** IDs still aren't the features the prediction layers will use.

## 4. Repeat the same small lookup in PyTorch

Return to 02A's three-word teaching table:

| Word | Local address | Two-number row |
|---|---:|---|
| the | 0 | [2, 0] |
| cat | 1 | [1, 3] |
| sat | 2 | [0, 4] |

For **the ___ sat**, IDs zero and two fetch `[2, 0]` and `[0, 4]`.
Two visible words give two rows. No averaging or guessing has happened.

The one-hot cross-check expands each ID into three checklist entries, then
multiplies by the table. It produces the same rows. Direct lookup avoids
constructing those zeros.

The vocabulary-by-width shape describes **stored weights**, not each input.
A one-word batch supplies one integer, shape `[1]`, and gets `[1, D]` values.
The equivalent one-hot input would have shape `[1, V]`, but lookup never
constructs it. A scalar ID also works, returning `[D]` without a batch axis.
An ordinary one-input linear layer would multiply the ID as a feature instead;
it would not select an independently stored row.

These local IDs are not GPT-2 IDs. The two numbers were chosen by hand, not
learned from the notes. A compact row solves width; useful shared features
still need practice.

Follow one word all the way through: `cat` becomes integer input `[1]`.
The embedding layer selects row `[1, 3]`. A separate two-input, three-output
prediction layer mixes it into scores `[3, 1, 4]` for `the`, `cat`, and `sat`.
Those score values come from hand-picked prediction weights, not training.

The distinctions matter: the embedding **layer** owns the full table; one
word's embedding is the fetched row. The predictor's scores aren't another
name for that row. Lookup can return an embedding without making any guess.
The notebook's sensor-profile illustration and actual PyTorch layers use
the same numbers, so you can check every transition.

**Complaint:** what practice could give two different word rows something useful
in common?

## 5. Give the rows a reason to change

The lab supplies six questions: `cat` and `dog` each support `sat` and
`slept`; `bird` supports `flew` and `sang`.

This is a controlled learning game, not a claim that birds can never sleep.
The questions are hand-picked, not extracted from the six notes.

One trainable table fetches four-number input rows. A separate prediction
layer turns each row into scores for all seven vocabulary words.
Both belong to the model, so both receive feedback and can change.
The integer input is only an address. The fetched row is an input to the
prediction layer, but a trainable weight of the whole model. Its random numbers
are drawn once at setup, then reused and updated, not redrawn for each word.

After 250 updates, the recorded loss falls from about **2.0201 to 0.6932**.
The cat/dog row distance changes from about **3.2438 to 3.0296**.
The loss shows better practice answers. The distance shows movement, not a
complete measurement of meaning.

After training, the notebook fetches the four-number cat row directly from
the embedding table, then separately prints its seven prediction scores.
The trained row is the embedding to keep, not the score list. Discarding the
head doesn't erase what the table learned, but using it on a new job still
needs evidence.

Shared use can teach shared features, but it doesn't require identical rows.
A predictor can also learn around fixed random rows. Transfer to different
examples needs a separate test.

**Complaint:** why doesn't this easy task drive the loss all the way to zero?

## 6. The cat has two legitimate answers

The identical `cat` input appears with two different answers. With no extra
clue, a deterministic forward pass cannot identify which card we drew.

Softmax turns raw scores into chances that add to one. The trained cat and
dog shortlists each give roughly half to `sat` and half to `slept`.
This is sensible uncertainty, not necessarily failed training.

Choosing the biggest chance repeats the same answer. Sampling uses the whole
spread: the recorded twelve cat draws include both `sat` and `slept`.
The distribution stays fixed while the draws vary.

The random draw is a separate decision from embedding lookup. Neither shorter
rows nor similar rows automatically makes prediction nondeterministic.

**Complaint:** if the answer says `sat`, which input row actually receives feedback?

## 7. Follow one question through a fresh model

The gradient probe creates another random model and asks only `cat -> sat`.
Only the cat input embedding row receives a gradient. The target is checked
against output scores; it is not another input lookup.

Then a fresh plain SGD step changes the cat row and the predictor weights.
Input and answer IDs remain exactly the same. A second lookup at the same
address fetches the updated cat row. That is learning features, not editing data.
With no momentum or weight decay in this probe, unused input rows stay unchanged;
other optimizer settings can move them for those separate reasons.

The prediction layer also receives gradients. Saying that only one embedding
row participates does not mean only one row of every weight table changes.

Fresh state matters here. Gradients from the earlier full training run would
make it harder to isolate this one question.

**Complaint:** batching introduces blank positions that must not become new questions.

## 8. Protect blank inputs and ignore blank answers separately

Pad the unequal examples into a rectangle. The real-token mask says which
seats contain words.

For loss, flatten example and position together, leaving each vocabulary score
row beside its matching target. Flattening changes the layout, not the
question/answer pairing.

The explicit real-target average matches `ignore_index`: both measure about
**2.3513** in this probe. Appending extra pad columns leaves that correct
answer unchanged.

Separately, an embedding's padding-row setting prevents lookup gradients for
that input row. It doesn't remove padded output targets from loss.

Watch for mixed ID maps, raw scores called probabilities, accidental gradient
accumulation, and sampling mistaken for training.

**Next complaint:** we supplied every practice pair. [03A's notebook](../03-language-models/03A-training-skeleton-and-word-objectives.ipynb)
will turn ordinary stories into different questions for CBOW, Skip-gram, and a
reader/writer, each with its own data preparation.

**Remember:** lookup fetches features, practice changes them, the predictor
scores answers, and sampling draws from the chances.
