# One Training Skeleton, Three Language Objectives

**Chapter 03A:** this is the former 02C companion, preserved as the first model-building lesson
after tokenization and embedding basics. Read the [notebook](03A-training-skeleton-and-word-objectives.ipynb)
for experiments or the [Python companion](03A-training-skeleton-and-word-objectives.py) for code.
The shared trainer lives in [language_model.py](language_model.py).

The small encoder-decoder below is a preview of how a sequence-shaped answer fits the training
contract. Continue through [ordered windows](03B-counts-and-fixed-window-language-models.ipynb)
and [RNN/LSTM text prediction](03C-fixed-windows-to-recurrent-language-models.ipynb) before the
full source-access comparison in 03D. CBOW and Skip-gram are word-vector objectives, not
autoregressive language models or mandatory ancestral Transformer components.

## Recap: we can fetch the numbers. How do we teach them?

You already know the training routine: guess, measure the mistake, work backward, adjust.
Chapter 01 added a running memory. Chapter 02 let you turn words into IDs and fetch a row of
numbers for each ID. Its ready-made exercises showed that training could change those rows.

The missing piece is not lookup. It is getting practice questions from ordinary text without
writing an answer key by hand.

But first, why learn the rows? In 02A we used a random NumPy table just to inspect lookup.
In 02B, `nn.Embedding` started random and learned from six supplied pairs. The GPT-2 example
loaded a tokenizer, not GPT-2's word vectors. Here each new model also starts with fresh random
weights. A seed repeats the random draw; it doesn't supply pretrained knowledge.

A name tag says which word we have. Useful features help the next layer decide what to do with
it. `contract` and `agreement` supported the same next-word answers in 02B. Letting their rows
adapt gives the model a place to represent something helpful for that shared job, while the
prediction layers learn to use it. No dimension has to mean a human-readable property, and
matching answers don't force the complete rows to become close.

We could keep random rows fixed and teach a capable predictor to use them. We could start with
pretrained rows instead. Learning the rows is a choice to adapt the input features too, not a
requirement for lookup. Whether that improves new-text predictions needs a held-out comparison.
Our tiny examples demonstrate the learning path, not that learned embeddings always win.

The reason we want useful shared features is to reuse learning across words. `contract` and
`agreement` needn't be unrelated cases: what the model learns with one can help with the other
where their roles are similar. Learned embeddings can capture those semantic relationships.
They don't equate words or merge their IDs. Similar context can also link opposites, and a
word with several meanings still needs its sentence to resolve which meaning is in play.

Cover `agreement` in `the non-disclosure agreement protects confidential`.
Ask the model to guess it from the four remaining words. We know the answer because we covered
it up. Move along the text and repeat. The document supplies both questions and answers.

The model fetches the input words' rows, combines them, and guesses. When it gets the answer
wrong, feedback travels backward through the predictor and into the rows it used. Both can
learn from the same mistake. The rows become useful for this task, not magically fluent in law.

Lookup still just fetches a row. Prediction is the exercise we chose to teach its numbers.
You could instead use already-trained rows and keep them fixed, or train both parts and later
discard the predictor. Learning together also does not mean sharing weights: our input table
and output layer are separate.

Now we have a teaching task. But our first model averages the input rows, losing word order.
**But what about this?** how can a model use a difference we erased before it made its guess?
That question takes us from bags of words to ordered windows and running memory.

## 1. The reusable part is smaller than the model

A training program can look more complicated than it is when splitting text, making batches,
guessing words, and updating weights are all mixed together. Pull them apart.
Whatever the model guesses, the practice routine is still:

```text
input IDs and target IDs
-> model-specific logits
-> align logits with target positions
-> cross-entropy
-> backward
-> optimizer step
```

The `DataLoader` keeps questions beside their answers and hands over batches. It doesn't care
whether the answer is a missing word, a neighbour, or a separate sequence.
The optimizer doesn't need to know either. It adjusts the weights reached by the loss.

The model must return word scores, called logits, and tell us which answer to grade each set
against. `TrainableLanguageModel` handles that bookkeeping and the `fit_language_model` loop.
CBOW, Skip-gram, and the encoder-decoder inherit the same loop. You supply the loader, loss,
and optimizer; each model supplies its layers, guessing method, and answer selection if it differs.

If a new guessing game makes you rewrite the entire practice loop, you've probably mixed
the model's job with the trainer's job.

## 2. CBOW asks many words to predict one word

Take five words at a time from our legal text. Cover the middle word.
The two words on each side become the clues:

```text
left, left, TARGET, right, right
-> four context IDs predict one center ID
```

CBOW fetches the four rows and averages them. A small network uses that average to score every
possible answer word. In the recorded run, its loss
fell from **2.8208 to 0.0002**.

That tells us the model got better at these practice questions, not that it understands law.
There are only eight windows. Memorizing them is very much on the table.

The average is also a warning label. Reversing the four context tokens leaves the pooled vector
and final logits unchanged. More epochs cannot recover order discarded by the mean.

**But there's a catch:** averaging gives the words no seats. The model cannot tell which
word came first or which role it played.

## 3. Skip-gram flips the prediction arrow

Use the same windows, but turn the question around. Show the center word and ask for one
neighbour. Repeat for all four neighbours:

```text
center predicts left neighbour
center predicts left neighbour
center predicts right neighbour
center predicts right neighbour
```

We still hand the loader a question and an answer. The same training loop can grade the
guess, work backward, and update the weights.

The code change is small too. Skip-gram fetches one center row, so there are no neighbour rows
to average. Its prediction layers turn that one row into scores for the possible answer words.

Its measured loss fell from **3.0097 to 1.6564**, below the random full-vocabulary baseline but
well above CBOW's final training loss. That higher floor is informative. One center such as
`agreement` appears beside several valid neighbours. A deterministic full-softmax classifier
cannot assign probability one to several different answers for identical input.

Large Word2Vec systems commonly replace full-vocabulary scoring with negative sampling or
hierarchical softmax. Those are efficiency and objective refinements, not a change to the central
direction:

| Word2Vec objective | Input | Target |
|---|---|---|
| CBOW | Several context words | Center word |
| Skip-gram | Center word | Neighbour word |

**But there's a catch:** we changed which nearby words teach each row, not what lookup returns.
The same ID still fetches the same row in every sentence. We haven't built a note that changes
with the sentence around it.

When the job changes from predicting one center word to writing several ordered answer tokens, the model needs a sequence of outputs, not another static-vector objective. The source is the input sequence; the target is the answer sequence.

## 4. What if the answer has several words?

One-token objectives cannot express “produce these next tokens in this order.” The notebook
therefore reuses each five-token window as a tiny conditional continuation:

```text
source: first three tokens
target: start marker, final two tokens, end marker
```

The answer now needs several positions. Instead of one set of word scores, we need a set at
each writing step.

An encoder GRU reads the source in order and updates a running note, its hidden state.
The decoder GRU starts with the reader's final note. During practice it also gets the correct
previous answer token and guesses the next one. This is **teacher forcing**.

In plain English: during practice, give the writer the correct previous target token before asking it to predict the next one. At generation time there is no answer key, so it must feed its own previous guess instead.

We need one small grading change. Don't ask the writer to predict the start marker it was
just given. Its inputs leave off the final token; its expected answers leave off the start
marker. Now each input lines up with the next answer, over the same three positions.

The encoder-decoder calls the inherited `fit_language_model` method. That loop asks this model
to make its guesses and pick its shifted answers, then calculates the loss as usual.
The reader and writer don't need separate training loops.

On the tiny corpus, sequence loss fell from **3.0468 to 0.0002**, teacher-forced token accuracy
reached **100%**, and greedy decoding reproduced the inspected continuation:

```text
protects, confidential, end
```

This is strong evidence that the implementation and alignment work. It is weak evidence about
general generation because the model saw only eight fixed examples.

## 5. It notices order. Can the writer get every detail?

Reversing a CBOW context changed nothing because averaging is commutative. Reversing the
encoder-decoder source changed its hidden state by a maximum absolute difference of **1.312418**.
The recurrent update therefore preserves some effect of order.

But notice the handoff. The reader gives the writer only its final note, our summary card.
However long the source gets, that card has the same number of entries.
If a detail didn't survive there, the writer cannot reopen an earlier note to find it.

Attention lets the writer consult the saved note from each source position.
A Transformer goes further: attention lets token positions exchange information without
passing a recurrent note along the source one step at a time.

The fixed-summary handoff is the next complaint: the decoder receives one learned note, not direct access to every source position. [The recurrent encoder-decoder-to-attention notebook](../03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb) compares those two access patterns on the same task.

## 6. Practical failure modes

| Failure | What it means |
|---|---|
| CBOW loss falls but permutations match | Optimization worked; architecture still discards order |
| Skip-gram loss plateaus above zero | One center has multiple valid neighbours; deterministic targets conflict |
| Sequence shapes do not align | Decoder inputs and next-token targets were shifted incorrectly |
| Teacher-forced accuracy is high but greedy output fails | The model depended on correct earlier answers it will not receive during generation |
| Recurrent state changes with order | Necessary order sensitivity, not proof of useful language understanding |
| Tiny-corpus output looks fluent | Memorization must not be reported as generalization |

**Durable summary:** keep data delivery and optimization boring; let each language objective
change only the tensors, forward mechanism, and target alignment that genuinely define it.
