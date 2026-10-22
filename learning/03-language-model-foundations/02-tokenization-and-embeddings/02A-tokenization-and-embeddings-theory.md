# Tokenization and Embeddings: Handwritten Theory

Read alongside [02A's notebook](02A-tokenization-and-embeddings-theory.ipynb).
Then try [02B's lab](02B-tokenization-and-embeddings-pytorch-lab.ipynb).

**Carry-in:** the piano lab already trained note rows with an LSTM. Here we open
up lookup; returning to hand-picked numbers is a close-up, not a step backward.
Text replaces notes because choosing pieces now needs explanation.

Keep two named examples apart. The **lookup microscope**, **the ___ sat**,
uses hand-picked rows to expose selection. The **learning game**,
**cat/dog -> sat/slept; bird -> flew/sang**, motivates useful shared features;
the next lab will train its six supplied cards. The longer sentence remains
**the sleepy cat sat on the mat**.

Our map is text -> pieces -> IDs -> fetched rows -> context processing ->
scores -> checked answer. This chapter opens pieces, IDs, and rows. Compactness
is one job; learning useful contents is another. No saved weights or ID numbers
transfer between notebooks.

## 1. A note is not yet a model input

Start with **the sleepy cat sat on the mat**. You can read it immediately;
a network needs numbers. Before picking those numbers, decide which text pieces
they will stand for.

One whole-word token is compact, but a new spelling needs a new entry.
Single characters can spell unfamiliar words using a known alphabet, but they
use more positions. The notebook checks both on six pet-sitter's notes.
`sleepyhead` has no whole-word entry, although all its letters are available.

This matters later: more positions mean more work, while a bigger vocabulary
means more rows to learn. Neither extreme wins everywhere.

**Complaint:** can common fragments share a piece without making every new
word a special case?

## 2. BPE closes common seams

Picture a zipper. Start each word as characters, count adjacent pairs, and
join the most common pair. Count again, because the available pieces changed.

This is **byte-pair encoding**, or BPE. Our toy uses characters and an end-of-word
marker so the joins remain visible. GPT-2 starts with bytes instead; the byte
alphabet supplies coverage for UTF-8 text.

In 02A, `sleepy` goes from seven symbols, including its marker, to one piece
after 18 merges. A merge elsewhere can leave `sleepy` unchanged. The static
chart and animation read the same recorded states.

The merge list is ordered: a later join may depend on an earlier one.
Learning those joins and applying them to new text are separate jobs.
02B replays the learned list on `sleepyhead`, keeping spare letters where needed.

**Complaint:** shorter strings are useful, but the model still needs numeric inputs.

## 3. An ID selects a sensor profile

Think of an ID as a celestial object's registry number, not a measurement.
Its embedding is a stored profile across a common array of abstract sensor
channels. Keep the readings separate, not summed into one score. Training
learns their values; they are not literal temperature or mass measurements
and need not have human-readable meanings. In code, one matrix row stores
one profile. Lookup retrieves it; it does not measure the sentence afresh.

Both notebooks use this exact miniature table:

| Word | Local ID | Stored row |
|---|---:|---|
| the | 0 | [2, 0] |
| cat | 1 | [1, 3] |
| sat | 2 | [0, 4] |

ID 2 fetches the `sat` row, not twice some other row. ID zero fetches `[2, 0]`,
not zeros. For **the ___ sat**, the two visible IDs fetch two rows.
The covered word is not another input.

Giving `cat` ID 2 instead of 1 adds no meaning. Renumber the address book and
move the matching rows together, and lookup returns exactly the same features.
The notebook checks this, then shows the mistake: moving only the IDs fetches
the wrong rows. In a predictor, output labels must use the same address book too.

These are hand-picked teaching numbers, not learned features. The table has
three words and two numbers per word. It reappears in 03A so that only the
new operation, combining rows, needs your attention.

**Complaint:** why not send one box per word instead?

## 4. A compact row replaces a giant checklist

A **one-hot** row has one box per vocabulary item: one is selected, the rest
are zero. Multiply that checklist by the embedding table and it selects the
same row as direct lookup. The notebook checks that equality.

Every distinct pair of one-hot rows is equally far apart. The checklist
distinguishes identities but says nothing about related uses.

An embedding can give each selected token a much shorter row. A 50,000-word
vocabulary might use 32 numbers per selected row instead of 50,000 checklist
entries. The shared table still has 50,000 rows. Lookup avoids constructing
the giant checklist; it doesn't make the table disappear.

That is the first embedding job: compact features. The second needs learning.

**Complaint:** making a row short doesn't make its numbers useful.

## 5. Practice can teach shared features

02A's later `bark` table starts random. 02B's trainable table also starts random.
Neither borrows GPT-2's model weights. Its tokenizer supplies splitting rules,
IDs, and decoding, not trained embedding values.

02B supplies six little questions:

- `cat` and `dog` each predict `sat` and `slept`.
- `bird` predicts `flew` and `sang`.

The input row and the prediction layer both change when a guess is checked.
The row starts random once, when the table is created; lookup doesn't redraw it.
The integer ID stays fixed. The stored row is an input to the predictor and a
trainable weight of the whole model. Backpropagation calculates its feedback;
the optimizer updates it. The same ID then fetches the updated numbers.
Words used alike can develop useful features in common, helping the predictor
reuse what it learned rather than treat every identity as unrelated.

We don't assign a column called "furriness". We also don't require whole
vectors to become identical or closer in every direction. The prediction layer
can use some differences and ignore others.

This is the second embedding job: somewhere to learn relationships from use.
Frozen random rows can still support learning in the predictor. Lower training
loss or moving rows alone doesn't prove transfer to unfamiliar text.

**Complaint:** the same question sometimes has more than one sensible answer.

## 6. A shortlist is not a single inevitable word

Our identical `cat` input is paired with `sat` and with `slept`. One forward
calculation cannot know which practice card was drawn.

02B turns raw scores into chances with **softmax**. Each answer gets a share,
and the shares add to one. The tiny balanced task learns roughly half for
each of those two answers.

Choosing the largest share is deterministic. Sampling draws from the spread,
so repeated draws can differ without changing the input or weights.
Embeddings don't create the randomness; the sampling choice does.

That is a warm-up, not a sentence writer. 03A will manufacture questions
from text before later lessons build next-token writing.

**Complaint:** shared features still don't tell us which meaning a sentence intends.

## 7. One stored row, two meanings

The `bark` in **the loud bark woke the cat** and in
**the rough bark covered the tree** starts from the same stored row.
Lookup sees an identity, not the sentence.

Adding different neighbours changes the result in 02A's small demonstration.
That mixing rule is hand-written, not trained understanding. Later models
learn how surrounding words should affect each occurrence.

Similar usage also isn't synonymy: `happy` and `sad` fit similar sentence slots
while meaning opposite things. Nearby rows are clues, not replacement rules.

**Complaint:** batching adds positions that aren't words at all.

## 8. Blank seats must not count as evidence

Unequal notes need padding to fit in one rectangle. A mask marks real seats.
The notebook's invented losses average to **0.175** when blanks count and
**0.300** when only real targets count. Excluding blanks changes both the sum
and the number of graded targets.

Keep three jobs apart: an input padding mask excludes blank evidence; a loss
mask excludes blank answers; a causal mask blocks future clues. An embedding's
padding-row setting is another input-side control, not a substitute for loss masking.

Watch for ID arithmetic, mixed vocabularies, untrained rows called semantic,
and padding accidentally improving a score.

**Next:** 02B makes these checks in PyTorch. [03A's notebook](../03-language-models/03A-training-skeleton-and-word-objectives.ipynb)
then creates its own questions, with a separate loader for each objective.

**Remember:** pieces choose the units, IDs choose rows, practice teaches features,
and sampling chooses from the resulting chances.
