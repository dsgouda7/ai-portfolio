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

Average the fetched vectors, never the numerical IDs. Autograd remembers which
rows were read and returns lookup gradients to those addresses. Repeated IDs add
their contributions to one row; unselected input rows receive zero gradient
through this lookup. The default gradient tensor can still be dense.

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

The animation records one real window on a copy: forward reads weights,
backward fills gradients without changing them, and a fresh plain-SGD step
changes only selected input rows. The separate output head also receives
feedback; using cat as an answer does not itself fetch cat's input row.

This exact unchanged-row result assumes untied weights and no decay or optimizer
history. Adam's earlier history can move a row with zero current gradient;
decay or output-weight sharing can also change the result. Row selection
explains the lookup's gradient route, not every possible optimizer update.

Before training, save the real model's table. Afterwards, compare actual features
for familiar words and confirm their input IDs stayed fixed. These observations,
alongside reserved-story loss, show what changed and whether prediction transferred.
Neighbour inspection asks a different question about what the representation now supports.

**Complaint:** useful rows explain representation, but do they make text generation random?

## 5. Similarity helps representation; sampling supplies randomness

Shared features let a neural model reuse patterns across words and new
combinations of familiar words. Held-out prediction tests that generalization;
a neighbour list alone does not.

Finding a word similar to cat is not choosing what follows **the cat**.
A language model processes the context and scores possible next tokens.
Softmax converts scores into probabilities without making a random choice.
Greedy decoding chooses the largest probability; sampling draws from the
distribution. Embeddings do not become random again during generation.

The notebook freezes one CBOW gap's probabilities and compares greedy selection
with seeded samples. A matching seed replays the draws without updating weights.
Those are independent answers, not a sentence. Autoregressive generation instead
appends a choice and recomputes probabilities from the updated context.

Embeddings support learned features, not randomness itself: count-based models
can also sample, and neural generation can use deterministic decoding.

**Complaint:** different choices cannot restore information the model already discarded.

## 6. Better rows cannot recover discarded information

Now permute the trained CBOW's clues. Their mean, and therefore their word scores,
remain unchanged. Training improved stored values but could not restore order
removed before prediction.

That is the next question for [03B](03B-counts-and-fixed-window-language-models.ipynb):
would giving words separate seats preserve the distinction?

Optional extensions inspect fused lookup, alternate objectives, a recurrent writer,
and a larger held-out comparison. They are not prerequisites for this conclusion.
Production uses a matching tokenizer and checkpoint table; ordinary inference
reuses learned weights without drawing new random rows or running optimizer steps.

**Failure checks:** do not confuse IDs, parameters, activations, gradients, or
scores. Do not mistake lookup for learning, a sampled answer for a generated
sentence, or a memorized question for useful general-language behavior.

**Durable summary:** fixed IDs select learned features; context drives prediction probabilities, and the decoding rule determines how an answer is chosen.
