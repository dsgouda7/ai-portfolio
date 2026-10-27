# One Training Skeleton, Three Language Objectives

**Chapter 03A:** [notebook](03A-training-skeleton-and-word-objectives.ipynb) |
[Python companion](03A-training-skeleton-and-word-objectives.py) |
[shared trainer](language_model.py) | [corpus and comparison](real_text_embeddings.py).

**Carry-in:** we supplied six questions. How can ordinary text supply thousands?
The **lookup microscope**, **the ___ sat**, first keeps its hand-picked two-number
rows for arithmetic, then rebuilds them randomly to expose actual learning. Averaging
still discards order. The **learning
game**, **cat/dog -> sat/slept; bird -> flew/sang**, supplied controlled cards;
here windows from text supply the cards instead.

Keep **the sleepy cat sat on the mat** as the inspection sentence, not the whole
training diet. The map remains text -> pieces -> IDs -> fetched rows ->
context processing -> scores -> checked answer. Objectives change the clues,
answers, and processing, not the lookup contract.

The introductory models each train a fresh 32-number table. The later held-out
comparison resets to 48-number rows with a common initialization. Neither run
continues the earlier lab's trained table or reuses its local ID numbers.

## 1. Names identify; profiles describe

Imagine a catalogue of objects found by space probes. Labels such as `Object-417`
and `Object-892` distinguish the records, but cannot tell you which objects resemble Earth.
Equip the probes with sensors: size, temperature, and atmospheric composition now
supply a profile of characteristics. Compare corresponding channels, accounting
for their units and scales, rather than comparing names or catalogue numbers.
Resembling Earth on those channels does not establish habitability.
The [sensor-profile diagram](images/03a-object-sensor-signatures.png) illustrates
representation and comparison, not a missing-object prediction task.

**Layout:** a registry ID selects a record; its profile supplies `D` values.
The catalogue holds `V` profiles in `[V, D]` order. The name answers "which record?";
the channels provide evidence about what that object is like.

**Bridge to words:** a token ID selects a word's stored profile, its **embedding**.
Words have no physical sensors. We choose the profile width, not a hand-written
meaning for each channel. There is no need to label words as "animal" or "action".
Physical readings have defined meanings; learned word features may spread a useful
property across several channels. Random starting values are not yet useful meaning.

**Complaint:** we have provided space for a profile. What teaches its values?

**CBOW is a training exercise, not the embedding itself.** Leave the sky catalogue
and return to **the ___ sat**. Fetch the visible words' profiles, average their
corresponding channels, and let a separate predictor score vocabulary words.
Keep `cat` out of the input; use it to check the scores. Backpropagation calculates
feedback, then the optimizer adjusts the fetched rows and predictor weights while
IDs stay fixed. The [separate training diagram](images/03a-cbow-profile-training.png)
keeps representation, readout, and feedback distinct.

Repeated text practice can shape useful shared features. This is learning from
word usage, or **distributional semantics**, not a guarantee of synonyms or a
well-trained profile for every rare word. Comparing two stored profiles is not
the same operation as predicting a missing word: the predictor does not just
look for the profile nearest the context average. Check reserved text to distinguish
useful learning from memorization.

For **the sleepy cat sat on**, hide `cat`: lookup gives `[B, 4, D]`;
averaging leaves `[B, D]`. Fetching and combining profiles reads them;
the later optimizer step is what changes them.

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
The notebook builds `make_document_windows` with NumPy's built-in sliding-window
view, applied separately to each sentence. Each row holds a complete window;
consecutive rows move one position, our window-start stride. Short sentences are
skipped before calling NumPy; unknown IDs keep their positions. Each sentence's
windows share its array's storage, then concatenation copies them into one array.
No short tails or cross-sentence windows appear. The width exercise changes only
a probe, not the training data.
The function only collects neighbouring words; each objective chooses its own
clues and answers. One animation connects the moving window to its start index:
the three CBOW answers are **cat, sat, on**. Printed windows keep the result
readable without animation playback.

Choosing a stretch of text is common in language-model training; the exact five-word
window is our teaching choice, not a universal rule. Next-word models usually read
longer chunks and predict one step ahead, rather than cover the middle word.

The vocabulary keeps common words plus every cat-probe word: **1,202 entries**.
`mat` occurs once; reserving its registry entry adds no evidence. Other rare words become unknown.
Count words, order them by frequency, then assign unique consecutive IDs.
The ID is a position in that ordering, not the count itself; ties keep first-appearance
order and still get different IDs. Zero is reserved for the unknown slot.

The hand-picked picture remains an arithmetic reference. A separate three-word
toy now creates and trains a fresh two-number table; no toy weights carry forward.
Actual introductory training uses **32-number rows**, batches of at most **512**, and
**10 / 4 / 6 epochs** for CBOW / Skip-gram / the writer. One measured CPU run spent
**66.9 seconds training**. The full cached Python lesson took **199.5 seconds**; hardware varies.

**Complaint:** ordinary text still is not an input/answer tensor. The objective must choose
which words are clues and which words are answers.

## 3. CBOW gives the clues an equal vote

CBOW removes the middle word from each five-word window. It fetches four rows and averages
them into one row per question. A small network turns that row into a score for every word.
Questions are never averaged together.

Average each feature channel with the same channel across words, never all channels
into one scalar. This combines evidence; it is not the definition of an embedding. The two-clue
microscope blends `[2, 0]` and `[0, 4]` into `[1, 2]`; this describes the context,
not the target `cat`'s stored `[1, 3]`. A separate predictor must learn the answer mapping.
The table is the shared catalogue; a batch is a stack of independent questions.
`[B, C, D]` counts questions, visible context words, and feature channels. Averaging removes
`C`, leaving `[B, D]`: one fixed-width composite per question, with no cross-question mixing.

The frozen picture sends `[1, 2]` through a hand-picked predictor, producing
scores `[1, 4, 2]` for `[the, cat, sat]`. That picture explains arithmetic, not
initialization. The runnable toy instead supplies only vocabulary size **3** and
embedding width **2** to an embedding layer. It allocates a trainable **[3, 2]**
table and fills it with standard-normal random values. Replaying the same random
fill with the same seed verifies initialization without hand-writing any row.

Lookup then selects existing rows: **[2, 2]** integer inputs return **[2, 2, 2]**
vectors, not another full vocabulary table. Ordinary indexing gives identical
values. Averaging produces **[2, 2]** context features; a separate random linear
predictor produces **[2, 3]** scores. Cross-entropy checks those scores against
`cat` ID **1**, not against its embedding.

Backward calculates a gradient ledger without changing the table. Plain SGD
then subtracts the learning-rate-scaled ledger; an explicit before/after check
verifies that exact update. After **40** steps, the seeded toy loss falls from
about **0.952 to 0.092**, and its `cat` probability rises from **38.6% to 91.2%**.
Both fetched input rows and the predictor change; the `cat` input row stays
fixed because it was only a label. The first top guess was already `cat`:
learning is not restricted to initially wrong predictions.

The printed row shifts show actual changes, not a promise of useful language
features. Two repeated questions demonstrate mechanics and memorization.
Production uses a matching tokenizer and checkpoint table; ordinary inference
does not recreate random embeddings for each prompt.

For missing `sat`, correct-word probability
rose from **0.090% to 0.181%**, while rank improved from **398 to 90**. That is movement,
not an exact answer. Missing `cat` improved in rank from **732 to 114**, but its probability
**fell from 0.076% to 0.028%**: better rank is not greater confidence in the correct word.
The strongest exact payoff was missing `on`: rank **304 to 1**, probability **0.096% to 87.020%**.
Keep probability, rank, and shortlist together.

CBOW sees both sides of a blank, unlike next-word writing. Its mean erases order:
reversing clues leaves the representation and logits unchanged.
Word order is lost, not the feature-channel layout. "Continuous" names real-valued
features; "bag" names permutation-insensitive pooling. Fusing fetch and mean through
`EmbeddingBag` preserves that limitation. Transformers need position information as
well as attention; the attention mechanism alone does not recover discarded word order.

The feedback close-up uses the illustrated **the sleepy ___ sat on** question on a
copy of the model. A fresh plain SGD step changes the four fetched input rows and the
predictor, but no IDs. The answer `cat` is not another input lookup. The shorter
**the ___ sat** toy is an isolated learning magnification, not another text corpus;
the separate indexed-update pictures below still use hand-picked starting values.

The toy passes the embedding object once into a model's constructor and stores
that exact object as a child layer. Normal prediction calls pass **only integer
IDs**. The embedding matrix is inside the model, not an external input vector
that the network somehow decides to rewrite.

The connection happens at attribute assignment. The module base class intercepts
assigning a child layer and records it in the parent's child-module registry.
The embedding already holds its weight in its own parameter registry.
Recursive parameter discovery walks those children and yields the same weight
object under the name `embedding.weight`; the optimizer retains that reference.
The toy inspects these registries and checks object identity at each hop.
Registration only makes the weight discoverable. It does not calculate a gradient
or change a value.

Lookup is not a read-only database: `nn.Embedding.weight` is a trainable parameter.
Attaching the embedding layer to the model registers that parameter, so
`model.parameters()` includes it. The integer selects `weight[2]`; the lookup
operation keeps a computation-graph connection from the returned vector to those
selected coordinates. PyTorch does not infer which vector belongs to an ID from
its meaning. The indexing operation already establishes that relationship.
A normal dense network likewise updates its own parameters, not its raw inputs;
this is the same rule, with a differentiable lookup as the first layer.
A detached copy of the same numbers would lose this connection. A normal lookup
is not detached, but neither is it a live view that changes when the stored weights
update: the next lookup retrieves the newly learned values.

Its backward operation routes each retrieved vector's gradient back to the row
selected by the integer ID. The ID itself gets no gradient. A repeated ID contributes
to the same row again. With default dense gradients, `.weight.grad` has the table's
shape; input-unused rows have zero gradient from lookup. `backward()` fills that
ledger; only `step()` applies the update.

The toy retains the fetched vectors' intermediate gradients solely for inspection.
Each occurrence of `the` contributes to row zero, and each occurrence of `sat`
contributes to row two. Adding those occurrence gradients into a separate zero
table reproduces the gradient PyTorch already calculated. The equality check
exposes lookup's scatter-add derivative without replacing autograd or applying
another update. Retaining intermediate gradients is not required for learning.

The index is used to place feedback in the gradient matrix. The optimizer does
not receive word IDs or call `step()` once per word: plain SGD applies one
elementwise update to the whole parameter tensor. Coordinate `[2, 0]` uses
`weight_before[2, 0]` and `weight.grad[2, 0]`; coordinate `[2, 1]` uses the
matching second column. Unused rows subtract zero in this controlled example.
Highlighting those coordinates separately unpacks one operation, not multiple steps.

The [indexed-update animation](images/03a-embedding-learning-animation.gif)
([static matrix reference](images/03a-embedding-indexed-update.png)) isolates
one input, `sat` ID 2, and target `cat` ID 1. A measured plain-SGD step at learning
rate 0.25 changes `[0, 4]` to `[-0.2454, 3.7547]`, leaving the other two input rows
exactly unchanged. The separate predictor also updates. This is a tiny, hand-initialized
mechanism check, not the four-clue CBOW run.

The zero rows are a consequence of lookup's derivative, not a special "zero-out gate".
Zero current gradient does not universally mean no update: weight decay or existing
optimizer state can move a row. Tying input and output weights can also give an
input-unused row gradients through vocabulary scoring.

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
