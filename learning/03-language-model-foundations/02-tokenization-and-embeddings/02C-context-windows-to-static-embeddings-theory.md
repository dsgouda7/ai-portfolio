# One Training Skeleton, Three Language Objectives

## 1. The reusable part is smaller than the model

A language-model training program can look specialized because tokenization, batching, model
layers, loss alignment, and optimization often appear in one long block. Pull those
responsibilities apart and a smaller invariant becomes visible:

```text
input IDs and target IDs
-> model-specific logits
-> align logits with target positions
-> cross-entropy
-> backward
-> optimizer step
```

The `DataLoader` does not know whether a row means “context predicts center,” “center predicts
neighbour,” or “source predicts a target sequence.” It preserves input-target pairs and delivers
batches. The optimizer is equally indifferent. It updates whatever parameters the loss reaches.

That leaves one important model responsibility: produce vocabulary logits whose positions line up
with the intended target IDs. The notebook's small training base class owns only this alignment.
It does not choose layers or hide the update loop. You could replace the base class with any
implementation that exposes the same training-loss contract.

The complaint is now precise: if changing a language objective forces you to rewrite batching,
backpropagation, and optimizer code, model mechanics and training mechanics are still tangled.

## 2. CBOW asks many words to predict one word

The running legal corpus first becomes five-token windows. With a context radius of two, the
center is removed and the four remaining IDs become evidence:

```text
left, left, TARGET, right, right
-> four context IDs predict one center ID
```

CBOW looks up four embedding rows and averages them. The pooled vector passes through a small
projection and competes over every vocabulary item. On the deterministic teaching run, its loss
fell from **2.8208 to 0.0002**.

That number proves that the model optimized the tiny training objective. It does not prove that
the learned rows carry broad legal meaning. There are only eight windows, so memorization is
expected.

The average is also a warning label. Reversing the four context tokens leaves the pooled vector
and final logits unchanged. More epochs cannot recover order discarded by the mean.

**Complaint:** CBOW learns useful static rows, but one unordered bag cannot represent the roles of
individual token positions.

## 3. Skip-gram flips the prediction arrow

The same windows can manufacture a different Word2Vec objective. Instead of using four neighbours
to predict one center, repeat the center four times and pair it with each neighbour:

```text
center predicts left neighbour
center predicts left neighbour
center predicts right neighbour
center predicts right neighbour
```

The loader contract does not change. It still emits input IDs and target IDs. The fit loop does
not change. Cross-entropy, backward, and the optimizer do not change.

The model change is small because the conceptual change is focused. Skip-gram looks up one center
row, so it has no context axis to average. It projects that row directly into vocabulary logits.

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

**Complaint:** flipping the arrow changes which co-occurrences shape a row, but it still learns
static token vectors rather than an ordered state for every occurrence.

When the job changes from predicting one center word to writing several ordered answer tokens, the model needs a sequence of outputs, not another static-vector objective. The source is the input sequence; the target is the answer sequence.

## 4. A sequence target changes the output rank

One-token objectives cannot express “produce these next tokens in this order.” The notebook
therefore reuses each five-token window as a tiny conditional continuation:

```text
source: first three tokens
target: start marker, final two tokens, end marker
```

The target now has a time axis. The model must emit one vocabulary competition for every predicted
target position.

An encoder GRU reads the source in order and compresses it into a hidden state. A decoder GRU
starts from that state. During training it receives the correct previous target token and predicts
the next one. This is **teacher forcing**.

In plain English: during practice, give the writer the correct previous target token before asking it to predict the next one. At generation time there is no answer key, so it must feed its own previous guess instead.

The generic training contract needs only one adjustment: ignore the target's start marker when
forming loss labels. Decoder inputs omit the final token; expected outputs omit the start token.
Both sides then describe the same three prediction positions.

On the tiny corpus, sequence loss fell from **3.0468 to 0.0002**, teacher-forced token accuracy
reached **100%**, and greedy decoding reproduced the inspected continuation:

```text
protects, confidential, end
```

This is strong evidence that the implementation and alignment work. It is weak evidence about
general generation because the model saw only eight fixed examples.

## 5. Order sensitivity fixes one failure and exposes another

Reversing a CBOW context changed nothing because averaging is commutative. Reversing the
encoder-decoder source changed its hidden state by a maximum absolute difference of **1.312418**.
The recurrent update therefore preserves some effect of order.

That unlock is not the end of the story. The encoder passes the entire source through one hidden
state. As source sequences grow, that state becomes a bottleneck: the decoder cannot directly
retrieve whichever earlier source position it needs.

Attention will answer that complaint by keeping source states addressable. A Transformer goes
further by constructing one contextual state per token position and routing information between
positions.

The fixed-summary handoff is the next complaint: the decoder receives one learned note, not direct access to every source position. [The recurrent encoder-decoder-to-attention notebook](../model-evolution/03-recurrent-encoder-decoder-to-attention.ipynb) compares those two access patterns on the same task.

## 6. Practical failure modes

| Failure | What it means |
|---|---|
| CBOW loss falls but permutations match | Optimization worked; architecture still discards order |
| Skip-gram loss plateaus above zero | One center has multiple valid neighbours; deterministic targets conflict |
| Sequence shapes do not align | Decoder inputs and next-token targets were shifted incorrectly |
| Teacher-forced accuracy is high but greedy output fails | The model depended on gold history it will not receive during generation |
| Recurrent state changes with order | Necessary order sensitivity, not proof of useful language understanding |
| Tiny-corpus output looks fluent | Memorization must not be reported as generalization |

**Durable summary:** keep data delivery and optimization boring; let each language objective
change only the tensors, forward mechanism, and target alignment that genuinely define it.
