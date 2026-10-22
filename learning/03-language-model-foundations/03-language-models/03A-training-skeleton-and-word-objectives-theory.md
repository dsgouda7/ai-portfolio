# One Training Skeleton, Three Language Objectives

**Chapter 03A:** [notebook](03A-training-skeleton-and-word-objectives.ipynb) |
[Python companion](03A-training-skeleton-and-word-objectives.py) |
[shared trainer](language_model.py) | [corpus and comparison](real_text_embeddings.py).

**Carry-in:** we supplied six questions. How can ordinary text supply thousands?
The **lookup microscope**, **the ___ sat**, keeps its hand-picked two-number
rows; here we add averaging and expose the order it discards. The **learning
game**, **cat/dog -> sat/slept; bird -> flew/sang**, supplied controlled cards;
here windows from text supply the cards instead.

Keep **the sleepy cat sat on the mat** as the inspection sentence, not the whole
training diet. The map remains text -> pieces -> IDs -> fetched rows ->
context processing -> scores -> checked answer. Objectives change the clues,
answers, and processing, not the lookup contract.

The introductory models each train a fresh 32-number table. The later held-out
comparison resets to 48-number rows with a common initialization. Neither run
continues the earlier lab's trained table or reuses its local ID numbers.

## 1. We can fetch word rows. What teaches their numbers?

An address distinguishes `cat` from `dog`; the guessing task gives their
separate rows a reason to develop useful features in common.

Think of a celestial object's registry number and its profile across a common sensor array.
The token ID identifies; the embedding supplies readings across abstract, learned channels.
They are not literal mass or temperature measurements, and we do not add all channels into
one total. In code, a matrix row stores one profile. It starts random once, then training
can adjust its values along with the predictor. Lookup retrieves the current profile;
it does not redraw or remeasure it. A fixed model returns the same starting profile for
the same ID. Renumbering the registry adds no meaning if profiles and output labels follow.

A one-hot checklist multiplied by the table selects the same profile, carrying vocabulary-sized
zeros along. Short rows solve storage; useful shared features still need learning.
The `[V, D]` shape belongs to the shared weights. Actual lookup takes one integer per word:
`[B]` IDs return `[B, D]` rows. A one-hot `[B, V]` input is an equivalent calculation,
not something `nn.Embedding` constructs. CBOW's four IDs per question give `[B, 4, D]`
before averaging, not a vocabulary-wide input per word.

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
Since this notebook repeats the sliding step for several models, we put it in one
small shared helper, `make_document_windows`, instead of repeating the code.
It only collects neighbouring words; each objective chooses its own clues and answers.
The CBOW storyboard and short animation slide the same window across our cat sentence:
the three checked answers are **cat, sat, on**.

Choosing a stretch of text is common in language-model training; the exact five-word
window is our teaching choice, not a universal rule. Next-word models usually read
longer chunks and predict one step ahead, rather than cover the middle word.

The vocabulary keeps common words plus every cat-probe word: **1,202 entries**.
`mat` occurs once; reserving its registry entry adds no evidence. Other rare words become unknown.
Count words, order them by frequency, then assign unique consecutive IDs.
The ID is a position in that ordering, not the count itself; ties keep first-appearance
order and still get different IDs. Zero is reserved for the unknown slot.

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

The feedback close-up uses the illustrated **the sleepy ___ sat on** question on a
copy of the model. A fresh plain SGD step changes the four fetched input rows and the
predictor, but no IDs. The answer `cat` is not another input lookup. The shorter
**the ___ sat** table is only an arithmetic magnification, not a different training corpus.

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

Imagine an observatory trained on many planetary systems. You have known members'
profiles, partial readings for a newly discovered member, and a question about a
missing property. The encoder reads that evidence; the decoder writes a likely
report, using the evidence and the report words written so far. A registry number
or membership in a galaxy alone cannot determine an unmeasured property. The report
is a prediction from learned patterns, not a new measurement.

Here the encoder GRU reads the source profiles in order and builds that briefing:
its final hidden state, not a simple average. The decoder starts there and writes
one answer token at a time. A galaxy itself is not ordered, but our observing log
is: word order matters. We are predicting text, not physical measurements or a new
vocabulary ID. The stored profiles remain separate from this context-dependent state.
Supplying correct previous answer words is
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

The writer can only use the final briefing here. If it omitted a useful detail,
the writer cannot reopen the log. Later cross-attention adds selective access to
saved source states, not one averaged galaxy.

For Transformer self-attention, imagine one spacecraft per token occurrence.
The same `bank` ID starts with the same profile beside `river` or `money`; lookup
alone cannot resolve the use. Each craft forms a learned query, scores permitted
keys, and gathers a weighted blend of value messages. This is directional
communication, not gravity or raw-profile averaging. Position and masks matter;
a causal craft cannot read future words.

Residual routes carry earlier state forward while messages and further processing
refine it. The resulting contextual vector carries identity interpreted through
selected relationships. Its width stays fixed; it is not a lossless history, a
sum of sensor channels, or necessarily a larger-magnitude vector. During inference,
working states change without rewriting stored profiles. In an encoder-decoder,
the source fleet contextualizes observations; the report fleet consults its own
prefix through self-attention and the source through cross-attention. These are
conceptual previews here, not mechanisms implemented by this chapter's GRUs.

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
