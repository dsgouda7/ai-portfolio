# How Word Embeddings Learn

**Chapter 03A:** [notebook](03A-training-skeleton-and-word-objectives.ipynb) |
[Python companion](03A-training-skeleton-and-word-objectives.py) |
[shared trainer](language_model.py).

**Guiding question:** if the input IDs never change, what exactly does training change?
Use **the sleepy ___ sat on**, answer **cat**, as a familiar question inside
one model trained on ordinary text. The subject is the learned representation,
not the internals of parameter registration or autograd.

## 1. An address is not a feature

Imagine cataloguing objects discovered by space probes. Names such as
**Object-417** and **Object-892** distinguish records, but cannot tell you which
object resembles Earth. Nearby catalogue numbers do not imply similar objects.

Equip the probes with sensors. Each object now has a profile: size, temperature,
atmospheric composition. Compare the same channels with Earth's readings,
accounting for their units and scales. The measurements supply evidence the
names could not. Similarity on these channels is not proof of habitability.

| Space catalogue | Word representation |
|---|---|
| Registry ID selects an object's record | Token ID selects a word's row |
| Sensor profile describes an object | Embedding supplies a word's numerical features |
| Same sensor layout for every object | Same feature-coordinate layout for every row |
| One profile per catalogue entry | One row per vocabulary entry |

This explains representation, not a missing-object prediction task. Words have no
physical sensors. Choose the number of feature coordinates, not a human meaning
for each one. Values start random, written in pencil; training shapes them.
A useful property may be spread across several coordinates.

Take the analogy to its conclusion: Earth's profile is a query for similar
planets; cat's learned profile is a query for related words. Geometry can reflect
semantic and grammatical relationships, not just identity. Related does not mean
synonymous: even hot and cold share many contexts.

CBOW gives the profiles a training job: predict a missing word from surrounding
clues. Context-based prediction is not a nearest-embedding search.

**Complaint:** we reserved space for features, but who supplies their initial values?

## 2. One trainable table lives inside the real model

Creating an embedding layer with vocabulary size **V** and width **D** allocates
one trainable **[V, D]** table. Its random values are a starting point, not knowledge.
It does not create a vocabulary-sized copy of each incoming input.

A batch of integer IDs shaped **[B, C]** selects **[B, C, D]** values: question,
clue word, feature channel. Lookup reads those stored rows; it does not invent a
new vector for every occurrence. The annotated CBOW code shows this where it happens.
Treat it as another neural-network layer: attaching it to the model includes its
weight in the model's parameters, just like a linear layer. The same loss and
optimizer train both.

Average the fetched vectors, never the numerical IDs. The average's numbers
cannot tell us which words supplied them. Autograd also retains the connected
operation graph: lookup knows the selected addresses; mean backward knows the
input shape and occurrence count. Dense mixing does not erase those dependencies.

Follow the return route:

1. Lookup saves the row indices that fetched each occurrence.
2. Mean combines four vectors; its backward rule knows the divisor is four.
3. Dense layers mix features, then loss starts backward differentiation.
4. Dense backward returns one feature-gradient vector to the mean.
5. Mean backward sends one quarter of it to each of four occurrences.
6. Lookup adds occurrence gradients into their saved row addresses.
7. The optimizer separately applies the resulting parameter gradients.

If one ID occurs twice, its two contributions add in one row. The divisor is
still four occurrences, not the number of unique IDs. Unread input rows receive
zero gradient through lookup; dense gradient storage includes those zero rows.
This is the lookup operation's derivative, not autograd recognizing a layer's
class later. Ordinary indexing and gather operations can route by saved IDs too.

**Complaint:** allocating space for profiles cannot tell us what should go in them.

## 3. Let text supply the teaching signal

Read the first **1,000 TinyStories stories** and append the familiar cat sentence
once as an independent probe document. A fixed training vocabulary turns words
into addresses. NumPy's sliding-window view collects complete windows separately
inside each sentence; it never joins one sentence's ending to the next.
Unknown words retain their positions as unknown IDs.

Each five-word window provides four clues and the original center word as an
answer. The model creates **32-number** rows and learns them from these questions.
No person has to label a coordinate "animal" or provide a hand-written word profile.
Repeated usage supplies the training evidence instead.

Reserve the next **200 stories**, absent from training. Keep the training vocabulary
fixed and evaluate at most **2,048** sentence-local windows before and after the
same fit. Evaluation reads parameters without updating them. The actual held-out
loss distinguishes transfer from merely fitting practice questions.

**Complaint:** we have questions and answers. Which operation actually teaches the profiles?

## 4. Keep the training flow short

Create the model once, then repeat:

1. Prepare visible-word IDs and a separate answer ID.
2. Fetch their current rows inside the model.
3. Average fetched vectors channel by channel, then let downstream layers score vocabulary words.
4. Check those scores against the answer using prediction loss.
5. Clear previous gradients and backpropagate the loss.
6. Let the optimizer update the model parameters, including the embedding table.

The shared trainer handles this ordinary loop. The embedding and prediction
layers learn together; a falling loss should not be attributed to the rows alone.
Feedback is useful even when the most likely answer was already correct.

The animation follows just the familiar cat question on a copy: cyan values move
forward, orange gradients return, and green marks the optimizer update. Each
active layer shows its actual input and output. It verifies the mean's gradient
split and the whole table's address-based accumulation, not just displayed values.
The separate head learns too; a checking answer need not be an input lookup.

This exact unchanged-row result assumes untied weights and no decay or optimizer
history. Adam's earlier history can move a row with zero current gradient;
decay or output-weight sharing can also change the result. Row selection
explains the lookup's gradient route, not every possible optimizer update.

Before training, save the real model's table. Afterwards, compare actual features
for familiar words and confirm their input IDs stayed fixed. These observations,
alongside reserved-story loss, show what changed and whether prediction transferred.
Neighbour inspection asks a different question about what the representation now supports.

This learning already happens in notebook **3.3**, not only in the final comparison.
Its epoch table starts at untrained epoch zero and follows the same CBOW for ten
epochs. Practice loss, reserved loss, cat probability/rank and cat-row movement
answer different questions. One unchanged optimizer carries its state throughout;
checking progress does not restart training or create a new model.

**Complaint:** useful word profiles are not the same thing as answers to a particular question.

## 5. Word neighbours are not prediction scores

Shared features let a neural model reuse patterns across words and new
combinations of familiar words. Held-out prediction tests that generalization;
a neighbour list alone does not.

Finding a word similar to cat is not choosing what fits **the cat ___ on the mat**.
The model processes the clues and scores possible answers. Each output position
refers to one vocabulary word; a larger score means the model favours that word
more in this context. Softmax converts the scores into probabilities.

The model produces vocabulary scores. Choosing the highest score gives its best
guess; sampling is an optional decoding strategy for later text-generation lessons.

Keep the useful checks separate. Neighbours inspect learned word profiles.
The familiar questions' probabilities and ranks inspect particular answers.
Reserved-story loss checks whether prediction improved beyond the training text.

**Complaint:** better features still cannot restore information discarded before prediction.

## 6. Better rows cannot recover discarded information

Compare **the cat ___ the dog** with **the dog ___ the cat**. The four clue words
and their counts are identical. Their averaged vectors are therefore identical.
Fill both blanks with **chased**: who chases whom changes, but CBOW cannot
distinguish these contexts from their average.

With fixed weights, the same average produces the same vocabulary scores and
probabilities. Choosing the highest score gives the same answer. This does not
mean CBOW learns no useful word relationships; it cannot represent differences
that depend only on the order of those clues. Training cannot restore positions
discarded before prediction.

That is the next question for [03B](03B-counts-and-fixed-window-language-models.ipynb):
would giving words separate seats preserve the distinction?

Optional A/B asides inspect fused lookup, visualization and Skip-gram.
Notebook section 6 then preserves the reader/writer and larger held-out comparison.
They are not prerequisites for the CBOW conclusion.
Production uses a matching tokenizer and checkpoint table; ordinary inference
reuses learned weights without drawing new random rows or running optimizer steps.

## 7. A reader and writer teach one shared embedding table

Read **the sleepy cat** and practise writing **sat on stop**. The notebook uses
`<bos>` to start the writer and `<eos>` as the answer meaning stop. This is a fresh
model, not the CBOW table continued under a different name.

1. Look up the three source-word rows and let the encoder read them in order.
2. Pass its final hidden note to the decoder. That note is sequence-dependent,
   unlike a stored word embedding.
3. Give the decoder the start token to predict sat.
4. During practice, give correct previous sat to predict on, then correct previous
   on to predict stop. The current answer is checked, not supplied to its own guess.
5. Check every next-word score list with cross-entropy and combine the losses.
6. Backpropagate through the prediction head and decoder. Feedback reaches the
   decoder's input rows and, through its initial note, the encoder's source rows.
7. Accumulate both lookup routes into the same shared Parameter. Repeated word
   IDs add their feedback; recurrent paths need not give every occurrence equal feedback.
8. Apply one optimizer step to the embedding table, both GRUs and the output head.

Section **6.3** follows six training epochs and prints actual before/after input
rows. Source-only cat can learn through the encoder, even though the writer's
first example never looks up cat as a previous answer. BOS learns as a decoder
input. EOS is only checked as an answer in this small run, so its untied input row
stays unchanged while its separate output-score parameters learn.

Teacher-forced improvement is not free-running generation quality. Section 6.4
feeds back the writer's own guesses. Sections 6.5-6.13 retain the independent
three-model held-out comparison, which restarts all models on its own split.
Those are three separate training runs, not one experiment deferred to the end.

**Failure checks:** do not confuse IDs, parameters, activations, gradients, or
scores. Do not mistake lookup for learning, nearby words for the best gap answer,
or a memorized question for useful general-language behavior.

**Durable summary:** fixed IDs select learned features; context drives candidate-word scores, and held-out evidence tests whether learning was useful.
