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
into addresses. NumPy's sliding-window view collects complete windows separately
inside each sentence; it never joins one sentence's ending to the next.
Unknown words retain their positions as unknown IDs.

Each five-word window provides four clues and the original center word as an
answer. The model creates **32-number** rows and learns them from these questions.
No person has to label a coordinate "animal" or provide a hand-written word profile.
Repeated usage supplies the training evidence instead.

The CBOW and Skip-gram data classes own different pairing rules, while their
loader base owns validation and batching. Each returns input IDs and answer IDs
in matching order. The batcher does not decide what a word is supposed to predict.

Reserve the next **200 stories**, absent from training. Keep the training vocabulary
fixed and evaluate at most **2,048** sentence-local windows before and after the
same fit. Evaluation reads parameters without updating them. The actual held-out
loss distinguishes transfer from merely fitting practice questions.

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
epochs. Practice loss, reserved loss, cat probability/rank and cat-row movement
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

That is the next question for [03B](03B-counts-and-fixed-window-language-models.ipynb):
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

Read **the sleepy cat** and practise writing **sat on stop**. The notebook uses
`<bos>` to start the writer and `<eos>` as the answer meaning stop. This is a fresh
model, not the CBOW table continued under a different name.

A word embedding is a stored 32-number profile. A hidden state is a temporary
48-number note about the sequence read so far. It is rebuilt during forward;
the optimizer changes the embedding and recurrent weights, not that old note.

Use the [GRU foundation](../01-sequence-memory/01A-sequence-memory-theory.ipynb#gru-hidden-state-mechanics)
for its reset/keep/replacement mechanics. Here the input is a word profile instead
of a piano profile. The encoder reuses its weights while reading the, sleepy and
cat; a separate decoder reuses its own weights while writing sat, on and stop.
The animation focuses on those block inputs, actual old/new state values,
the final-state handoff, and feedback into shared embedding rows.

Open one token, not another time box. Each GRU layer mixes the current profile
and old hidden through two learned Linear projections, then computes gates,
proposed content and the retained/written blend. The result is both this token's
output and the state carried onward. Those are two uses of the same values here,
not two independent predictions. The encoder keeps its final hidden as D0;
the decoder sends its output to a separate vocabulary head while carrying D1.
Different jobs, same GRU cell rule. This preview has one layer per block, not
three stacked layers because the picture shows three words.

The final reader note becomes the writer's starting note. Losses teach the
writer through its successive states, then travel through that starting note
to the reader's final and earlier states. Each read also depends on a fetched
word profile, so feedback can reach both source and writer input rows.
This is ordinary backward differentiation through time, not a second optimizer
for hidden-state values.

The GRU decoder always uses a numerical state. Omitting its state argument means
starting from zeros, not removing memory. It can run without an encoder, but then
it has no source note. A Transformer decoder has no carried GRU-style recurrent
state; it still computes internal token representations.

The notebook animation follows the same live cat probe after six epochs.
Its solid arrows carry state vectors. During practice, the next word input is
the correct previous answer; during generation, it is the decoder's own guess.
The word guess and the hidden state are separate outputs with separate routes.
Wrong guesses stay visible; the animation is measured mechanics, not a promise
of exact recall. A static view preserves the forward/backward route without playback.

Suppose the first guess is and instead of sat. While practising, still supply the
correct previous sat so the next question is to predict on. This input rule is
**teacher forcing**. It does not turn the wrong guess into a right one, reset the
hidden state, or replace backpropagation. It lets the model practise the real
continuation rather than only the mistaken sentence it started.

The inputs are start, sat, on; the checked answers are sat, on, stop.
Each step sees the previous word, not the word it is being asked to predict.
The GRU reads left to right, so a later input cannot help an earlier prediction.

1. Read the source and pass its final hidden note to the writer.
2. Read correct previous words and produce one next-word score list per step,
   carrying the new hidden state throughout.
3. Cross-entropy checks the scores against all three answers and combines the penalties.
4. Backward feedback reaches the head, both GRUs and their input rows. Shared-table
   contributions add; repeated occurrences need not receive equal feedback.
5. Apply one optimizer step for the batch. Six epochs mean six full passes through
   the examples, not just six weight updates.

Section **5.2** follows six training epochs and prints actual before/after input
rows. Source-only cat can learn through the encoder, even though the writer's
first example never looks up cat as a previous answer. BOS learns as a decoder
input. EOS is only checked as an answer in this small run, so its untied input row
stays unchanged while its separate output-score parameters learn.

The notebook checks the route on copies: detach the reader's final note before
passing it to the writer. The numbers and predictions stay the same, but the
source-only cat input gradient and row update become zero. The real model is
untouched. This demonstrates the learning connection, not the meaning of each
hidden-state coordinate or general language ability.

Training accuracy checks familiar examples with correct previous words.
Reserved-story validation checks new examples under that same input rule; it is
a real held-out next-word check, not a free-writing score. Stop-token accuracy can
also make the aggregate look better than content-word accuracy.

Section 5.3 keeps the fitted weights but feeds back the writer's own guesses.
An earlier wrong word now changes its next input, so mistakes can compound.
Sections 5.4-5.12 retain the independent
three-model held-out comparison, which restarts all models on its own split.
Those are three separate training runs, not one experiment deferred to the end.

**Failure checks:** do not confuse IDs, parameters, activations, gradients, or
scores. Do not mistake lookup for learning, nearby words for the best gap answer,
or a memorized question for useful general-language behavior.

**Durable summary:** fixed IDs select learned features; context drives candidate-word scores, and held-out evidence tests whether learning was useful.
