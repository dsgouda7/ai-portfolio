# One Training Skeleton, Three Language Objectives

**Chapter 03A:** [notebook](03A-training-skeleton-and-word-objectives.ipynb) |
[Python companion](03A-training-skeleton-and-word-objectives.py) |
[shared trainer](language_model.py) | [corpus and comparison](real_text_embeddings.py).

## 1. We can fetch word rows. What teaches their numbers?

Chapter 02 gave each word an address and fetched its row of numbers. An address distinguishes
`cat` from `dog`; it does not tell a predictor what those words have in common.

Think of an embedding as a drawer. Lookup opens the drawer. Training changes its contents.
A one-hot checklist multiplied by the table opens the same drawer, carrying vocabulary-sized
zeros along. Short rows solve storage; useful shared features still need learning.

Cover `cat` in **the sleepy cat sat on**. The four visible words become the question;
the covered word becomes its answer. Feed visible rows to a predictor and compare scores
with `cat`. Feedback reaches both predictor and fetched rows; their weights are separate.

**Complaint:** five hand-written phrases would let us confuse memorization with learning.
Keep the familiar sentence for inspection, not as the entire training diet.

## 2. Real practice, one ordinary probe

The introductory run reads the first **1,000 TinyStories stories** from a pinned Hugging Face
file, verified by SHA-256 and cached locally. These are synthetic children's stories from
a real published dataset, not prose assembled to guarantee our desired result.

Sentence splitting leaves **13,764 qualifying sentence-documents**. Append exactly one
independent document, **the sleepy cat sat on the mat**, at the end: **13,765 documents**,
**149,572 tokens**, and **94,512 five-word windows**. Its seven words supply three windows.
No window crosses a sentence boundary. No copies, special weights, or extra sampling
chances are given to that document.

The vocabulary keeps common words plus every cat-probe word: **1,202 entries**.
`mat` occurs once; reserving its drawer adds no evidence. Other rare words become unknown.

The made-up table remains only a shape demonstration.
Actual introductory training uses **32-number rows**, batches of at most **512**, and
**10 / 4 / 6 epochs** for CBOW / Skip-gram / the writer. One measured CPU run spent
**66.9 seconds training**. The full cached Python lesson took **199.5 seconds**; hardware varies.

**Complaint:** ordinary text still is not an input/answer tensor. The objective must choose
which words are clues and which words are answers.

## 3. CBOW gives the clues an equal vote

CBOW removes the middle word from each five-word window. It fetches four rows and averages
them into one row per question. A small network turns that row into a score for every word.
Questions are never averaged together.

For missing `sat`, correct-word probability
rose from **0.090% to 0.181%**, while rank improved from **398 to 90**. That is movement,
not an exact answer. Missing `cat` improved in rank from **732 to 114**, but its probability
**fell from 0.076% to 0.028%**: better rank is not greater confidence in the correct word.
The strongest exact payoff was missing `on`: rank **304 to 1**, probability **0.096% to 87.020%**.
Keep probability, rank, and shortlist together.

CBOW sees both sides of a blank, unlike next-word writing. Its mean erases order:
reversing clues leaves the representation and logits unchanged.

**Complaint:** more epochs cannot restore a distinction discarded before prediction.

## 4. Skip-gram reverses the flashcard

Show the center word and predict one neighbour. Each window yields four separate lessons:
the same center appears once for each neighbour, with matching answers in the same order.
This is the definition of the objective, not special repetition of our cat document.

Measured Skip-gram loss
fell from **5.1883 to 4.9446**. Its `cat` probe preferred common neighbours such as `the`;
the particular neighbour `sat` did not improve. One center occurs beside many different
answers, so a single deterministic distribution cannot give every neighbour probability one.

Large Word2Vec implementations use negative sampling or hierarchical softmax for efficiency.
Here full-vocabulary scoring keeps the teaching path visible.

**Complaint:** we changed the flashcard's direction, not the static lookup rule.
The same word still fetches the same stored row in every sentence.

## 5. The answer needs several ordered words

Reuse each window as a small continuation task:

| Role | Familiar example |
|---|---|
| Source | the, sleepy, cat |
| Full target | start, sat, on, stop |
| Writer input during practice | start, sat, on |
| Correct next tokens | sat, on, stop |

An encoder GRU reads the source into a final running note. A decoder GRU starts from that
note and predicts one answer per step. Supplying correct previous answer words is
**teacher forcing**. Dropping the final target from inputs and the start marker from answers
keeps each score beside the next token it should predict.

Sequence loss fell from **3.4643 to 2.1835**. Teacher-forced token accuracy on the first
4,096 training windows was **56.1%**, or **34.2% without the easy stop target**. The `mat` probe's probability rose
from **0.125% to 0.190%**, yet its rank worsened from **6 to 71** as other words gained more.
The writer's first `sat` guess likewise improved in rank, **967 to 263**, while probability
**fell from 0.072% to 0.022%**. A rank gain is not greater correct-word confidence;
neither probe shows exact recall.
For **cat sat on -> the ___**, its named alternatives included **wall 12.9%, ground 9.6%,
floor 4.8%**. These are plausible continuations, not success at remembering `mat`;
the unknown token still had the largest share, **17.4%**.
Greedy generation produced **and, unknown, stop**, not the expected **sat, on, stop**.
The free-running writer no longer receives correct earlier answers.

Reversing the source changed the final note. Order matters to the computation; useful use
of order is a separate question. The fixed-size final note also limits what the writer can
recover from the source.

**Complaint:** fitting practice questions still says nothing decisive about different stories.

## 6. Keep training machinery boring; check new text separately

Each objective owns its preparation and loader but inherits the same routine:
guess, align answers, flatten matching example/time axes, calculate loss, backpropagate,
and update. A sequence answer needs no new optimizer loop.

The larger comparison starts fresh models on **8,000 stories**, split **80/20 by whole story**
before vocabulary construction or windows. It adds no cat document. The introductory stories
belong to its training portion, not its holdout. All three models start with identical
48-number word rows, but their target counts and parameter counts differ.

Sixteen recorded epochs show practice loss falling after held-out loss has stopped improving.
The best held-out epochs were **5 / 3 / 4** for CBOW / Skip-gram / the writer.
Full-row cosine checks complement one fixed PCA camera; the sketch is not the complete geometry.
Named family, animal, colour, and feeling groups are inspection labels, never training targets.

For held-out CBOW blanks, the correct answer appeared first **24.9%** of the time and in
the top five **44.7%** of the time. A shortlist is more informative than one winning word.
Shared usage can link opposites; nearby rows are not interchangeable meanings.
This monitored holdout is not an untouched final test.

## 7. Keep the remaining complaints visible

- Flatten for counting, never before making document-local windows.
- Keep repeated centers and flattened neighbour answers in exactly the same order.
- Distinguish teacher-forced scores, easy stop targets, and autonomous generation.
- A vocabulary address supplies representational capacity, not additional training evidence.
- Lower practice loss, better rank, and higher probability are different measurements.
- Static embeddings cannot resolve sentence-specific meaning by lookup alone.

Next, [03B](03B-counts-and-fixed-window-language-models.ipynb) gives words ordered seats;
[03C](03C-fixed-windows-to-recurrent-language-models.ipynb) develops the running note;
[03D](03D-recurrent-encoder-decoder-to-attention.ipynb) lets the writer revisit source positions.

**Durable summary:** lookup fetches rows, the objective teaches them, and measured evidence
decides what you can honestly claim.
