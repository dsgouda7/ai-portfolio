# How Word Embeddings Learn

**Chapter 03A:** [notebook](03A-training-skeleton-and-word-objectives.ipynb) |
[Python companion](03A-training-skeleton-and-word-objectives.py) |
[shared trainer](language_model.py).
The closing [Concept recap](03A-training-skeleton-and-word-objectives.ipynb#concept-recap)
pairs the ID-to-row-to-score story with short code sketches and objective comparisons.

**Guiding question:** if the input IDs never change, what exactly does training change?
Use **cat sat ___ the mat**, answer **on**, from **the cat sat on the mat**, inside
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
new vector for every occurrence. The shared word-model base shows this where it happens.
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
into addresses. PyTorch's `unfold` collects complete windows separately
inside each sentence; it never joins one sentence's ending to the next.
Unknown words retain their positions as unknown IDs.

Each five-word window provides four clues and the original center word as an
answer. The model creates **32-number** rows and learns them from these questions.
No person has to label a coordinate "animal" or provide a hand-written word profile.
Repeated usage supplies the training evidence instead.
The refreshed run counts **149,571 tokens**, **13,765 sentence documents** including
the appended probe, **1,201 vocabulary rows**, and **94,511 five-word windows**.
The probe is part of training, not held-out evidence; mat occurs only once here.

The CBOW and Skip-gram data classes own different pairing rules, while their
loader base owns validation and batching. Each returns input IDs and answer IDs
in matching order. The batcher does not decide what a word is supposed to predict.

Reserve the next **200 stories**, absent from training. Keep the training vocabulary
fixed and evaluate at most **2,048** sentence-local windows before and after the
same fit. Evaluation reads parameters without updating them. The actual held-out
loss distinguishes transfer from merely fitting practice questions.
After ten CBOW epochs, practice loss changes **5.1224 to 3.4266** and reserved
loss **7.1424 to 3.9669**, with **28.0%** reserved answer accuracy. The familiar
on share changes **0.105% to 70.796%**, rank **155 to 1**. The other canonical
center, sat, still ranks **90**. A successful familiar gap is not universal success.

The core loader and model helpers are defined in the notebook, not hidden behind
imports. Validation is a separate module: record a baseline before training and
recheck the same reserved questions afterwards. CBOW checks center words,
Skip-gram checks individual neighbours, and the reader/writer checks next tokens.
Different counts reflect different jobs, so raw losses are not an architecture ranking.

**Complaint:** we have questions and answers. Which operation actually teaches the profiles?

## 4. Keep the training flow short

Create the model once, then repeat:

1. Prepare visible-word IDs and a separate answer ID.
2. Fetch their current rows inside the model.
3. Average fetched vectors channel by channel, then let downstream layers score vocabulary words.
4. Check those scores against the answer using prediction loss.
5. Clear previous gradients and backpropagate the loss.
6. Let the optimizer update the model parameters, including the embedding table.

The shared trainer handles this ordinary loop. Each word model supplies one
combination rule: average the CBOW clues, or keep Skip-gram's single center row.
The base runs lookup, that rule, and prediction layers in the same order.
Its training entry point configures loss and optimizer once, then repeats the
updates. Epoch observers inspect progress without taking over the loop.
This is the engineering contract: subclasses supply the changing rule, while
the base owns the sequence. The embedding and prediction
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
The Python recorder owns those before/after snapshots and reports; short notebook
calls display the evidence. The per-epoch callback remains beside the training run.

This learning already happens in notebook **3.3**, not only in the final comparison.
Its epoch table starts at untrained epoch zero and follows the same CBOW for ten
epochs. Practice loss, reserved loss, on probability/rank and cat input-row movement
answer different questions. One unchanged optimizer carries its state throughout;
checking progress does not restart training or create a new model.
The validation report after each training block compares the model with its
own before-training baseline. It reads weights and restores model modes without
updating parameters or clearing gradients. Reader/writer validation supplies
correct previous words; it does not prove free-running generation quality.

**Complaint:** useful word profiles are not the same thing as answers to a particular question.

## 5. A prediction job can also teach word relationships

CBOW practises filling a gap, checked against the word actually in the text.
Nobody supplies meaning groups or rewards it directly for putting related words
next to each other. Why can similar word profiles emerge anyway?

Imagine practising **cat sat ___ the mat** and **dog sat ___ the mat**, both
checked against **on**. These illustrate the mechanism, not quoted corpus examples.
Cat and dog are clues, so their input profiles are read and receive feedback.
The same predictor can reuse similar signals from either profile to find on.
Across many shared settings, similar kinds of correction can encourage features
that help with similar prediction jobs.

Related profiles can be a useful side effect of learning to predict. There is no
guarantee every related pair moves closer, and similar usage is not the same as
the same meaning: hot and cold share many settings too.

Notebook **3.4** now closes the training section by inspecting the actual learned
profiles, including imperfect matches. Compare their patterns rather than their
overall size; leave out the query word and the unknown-word placeholder.
That neighbour list is not the answer to a particular gap: prediction still uses
all the clues to score possible answers. Reserved-story loss checks whether
prediction improved beyond the practice text; neighbours inspect what the
learned profiles now look like.

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

That is the next question for [03B](03B-counts-to-recurrent-language-models.ipynb):
would giving words separate seats preserve the distinction?

This complaint now closes notebook section 3, rather than opening another core
section. The window storyboard stays with section 2, and a saved before/after
profile view sits in section 3.4. It uses one fixed camera for both states;
the full-profile comparison is stronger evidence than closeness in a 2D sketch.
Optional section 4 now contains only Skip-gram.
For the direct reader/writer preview, skip those optional experiments and jump to
[notebook section 5](03A-training-skeleton-and-word-objectives.ipynb#encoder-decoder-preview).
03B is the next-chapter route instead; the preview's shared setup is listed at
the section 5 opening. Its later held-out comparison also needs the Skip-gram setup.
They are not prerequisites for the CBOW conclusion.
Production uses a matching tokenizer and checkpoint table; ordinary inference
reuses learned weights without drawing new random rows or running optimizer steps.

## 7. A reader and writer teach one shared embedding table

Read **the cat sat** and practise writing **on the mat stop**. The notebook uses
`<bos>` to start and `<eos>` to stop. This is a fresh model, not CBOW continued.
Reuse [the familiar GRU](../01-sequence-memory/01A-sequence-memory-theory.ipynb#gru-hidden-state-mechanics);
replace supplied piano profiles with trainable word rows. Sharing one input table
between reader and writer is a deliberate design choice.

Keep three objects separate:

| Object | Job |
|---|---|
| Stored 32-number word row | Reused across contexts; changed by optimization |
| Temporary 48-number hidden state | Carries this example's ordered context |
| Vocabulary score list | Says which next words the head favors |

The reader's final hidden initializes the writer. For one GRU layer, output and
carried hidden have identical values; the separate head makes word scores.

**Why does sat learn when on is the answer?** Loss feedback passes through the
head, writer and reader, then through their input lookups into the shared table.
The checked answer's output weights are not its input row. In this question sat
is only read by the encoder, yet its row can receive feedback. A copy-only check
cuts the backward handoff: forward scores stay identical, but sat's input gradient
and copied update become zero. The real model is untouched.

Practice supplies correct previous words: start predicts on, on predicts the,
the predicts mat, mat predicts stop. This is **teacher forcing**, the same clean-history rule used
in the musical echo exercise. A wrong and guess stays wrong; giving on next
neither fixes that guess nor resets hidden state. The current answer is never
its own input. Cross-entropy checks the scores; backward calculates feedback;
one optimizer step updates the batch's parameters.

Section **5.4** performs six complete training passes and compares the same table
before and after. Movement proves updates, not semantic quality. Compare the
related cat/dog rows with story-context cat/mat rows; neither pair was a training
similarity target. Two pair scores do not establish general meaning. The unread
EOS input row remains unchanged while its separate output-score weights learn.
Reserved-story loss tests new next-word questions with correct previous words;
it is not a free-writing score. Content accuracy excludes the easy fourth, stop answer. The reader/writer uses
separate six-word sentence-local windows with an explicit three-word source;
CBOW and Skip-gram retain their odd, centered five-word windows.
The reader/writer has **80,746 windows**, hence **322,984 next-token checks per
epoch** including EOS. Its six-epoch practice loss changes **3.8768 to 2.4793**;
reserved loss **7.1030 to 2.8035** on **8,192** answers, with **46.4%** accuracy.
Training accuracy is **50.7%**, or **34.3%** without stop. On's probe share rises
to **33.662%**, rank **2**, but down wins. Its actual free continuation is
**down and unknown stop**, not the reference **on the mat stop**.
The stored cat/dog similarity changes **+0.055 to +0.249**, while cat/mat changes
**-0.436 to -0.569**. Keep both outcomes; neither pair was a similarity target.

**The remaining complaint:** order survives, but access is limited.
Reverse the same input rows: their mean agrees up to rounding, while recurrent
states and word scores change. The state depends on the earlier words already
read. That establishes order sensitivity, not correct understanding.

The writer currently receives only the final summary. Attention would retain all
source-position states and let each writer step select useful ones to look back
at. Order comes from recurrence here, not from attention. Transformer attention
needs separate positional information. [03C](03C-recurrent-encoder-decoder-to-attention.ipynb)
implements look-back access; this preview does not.

The [optional comparison lab](optional/03A-word-objective-comparison-lab.ipynb) preserves
the larger three-model experiment, independent data split and full diagnostic
reference without lengthening the main embedding-learning path.
It samples 40,000 five-word training windows for CBOW and Skip-gram, and
40,000 separate six-word windows for the writer from the same 6,400 training
stories and vocabulary. Each uses 4,000 held-out windows from 1,600 stories.
The per-epoch answer budgets are 40,000, 160,000 and 120,000 respectively;
the comparison writer predicts three content words and excludes EOS.
Different objectives and window budgets preclude an architecture ranking.

**Failure checks:** do not confuse IDs, parameters, activations, gradients, or
scores. Do not mistake lookup for learning, nearby words for the best gap answer,
or a memorized question for useful general-language behavior.

**Durable summary:** fixed IDs select learned features; context drives candidate-word scores, and held-out evidence tests whether learning was useful.
