# The Complete Transformer Block: Handwritten Theory Notes

## 1. Start with the block's job

Attention can retrieve positioned context, but a reusable Transformer block needs more than one routing table.

It must:

1. preserve several useful routing views;
2. transform the retrieved features inside each token;
3. keep information and gradients moving through depth.

A pre-normalized block performs two corrections to one residual stream:

```text
x
-> normalize -> multi-head attention -> add to x
-> normalize -> feed-forward network -> add again
```

The outer shape stays `(batch, sequence, model width)`. The meaning at each token position changes, not the number of positions or the model width.

## 2. Multi-head attention preserves several routing views

One attention head produces one query-key routing table. A sentence may contain several useful relationships at once: nearby syntax, repeated references, object-action links, and longer-range dependencies.

Multi-head attention gives each head independent query, key, and value projections. Every head receives the full input state, projects it to a narrower head width, retrieves context, and returns a result.

```text
input                    (B, S, D)
split Q/K/V              (B, H, S, D/H)
attention scores         (B, H, S, S)
weighted values          (B, H, S, D/H)
concatenate + project    (B, S, D)
```

Model width must divide evenly by head count. More heads are not automatically better. With model width 128, four heads give each head width 32; sixteen heads leave width 8. More heads provide more routing tables, but each table works with a narrower feature view.

The notebook's two-pattern diagnostic demonstrates capacity, not learned specialization. An identity head reconstructs identity with zero error but misses the `cat <-> mat` relation; the relation head does the reverse. One compromise table pays error on both jobs. Separate heads let incompatible routing patterns coexist. In a trained model, roles may overlap and need not have tidy human labels.

## 3. The FFN is each token's private workspace

Attention mixes information **across token positions**. The feed-forward network acts **inside each token position**. It never reads another row directly.

A standard FFN:

```text
model-width vector
-> wider hidden feature workspace
-> nonlinear activation
-> project back to model width
```

The same FFN weights are reused at every position. What differs is the contextual vector entering the network.

The expansion creates room for feature combinations that one linear projection cannot express. The nonlinear activation is essential: two linear layers with no nonlinearity would collapse into one linear transformation.

Memory aid: **attention communicates; the FFN computes locally.**

That solves private computation and opens a stability problem: FFN outputs can arrive with very different offsets and scales. Stack enough blocks and every sublayer would chase moving input statistics.

## 4. Residual paths preserve continuity

A sublayer should propose a useful correction, not rebuild the entire token state from scratch. Residual addition keeps the original stream and adds the sublayer output:

```text
new state = old state + proposed update
```

This gives information a direct route through depth. It also gives gradients a path that does not depend entirely on every learned transformation.

The notebook compares 24 identical transformation layers with and without residual additions. The residual version leaves a measurably stronger gradient at the original input. This does not guarantee easy optimization at any depth, but it isolates why skip paths are load-bearing.

## 5. Pre-normalization controls sublayer inputs

Layer normalization operates across one token's feature dimensions. It does not mix token positions.

The notebook shifts one token by `+8` and multiplies another by `5`. Before LayerNorm their scales disagree; afterward every token has feature mean near zero and standard deviation near one. LayerNorm conditions the next computation. It does not create context.

In a pre-normalized block, normalization happens before attention and before the FFN. Each sublayer receives a controlled input scale, while the residual stream itself remains the main information path.

```text
normalize x -> attention -> add to x
normalize updated x -> FFN -> add again
```

Post-normalized Transformers place normalization after residual addition. Both arrangements exist, but pre-normalization generally gives deep models a cleaner gradient route.

Normalization stabilizes the computation. It does not create context, knowledge, or routing.

## 6. Stacking blocks changes representations, not the contract

One block performs one round of communication and private processing. Stacking blocks repeats the same contract:

```text
positioned embeddings
-> block 1
-> block 2
-> ...
-> final contextual vectors
```

A later layer can operate on relationships discovered by earlier layers. Sequence length and model width still remain fixed through the stack.

The final vectors are not tokens or probabilities. They are contextual feature representations waiting for a task-specific output head.

## 7. The vocabulary head produces logits

For language modeling, a vocabulary head maps every final token vector to one score per vocabulary item:

```text
hidden states    (B, S, D)
vocabulary head
logits           (B, S, V)
```

Logits are unnormalized scores. Softmax can turn one position's logits into a probability distribution, but cross-entropy loss should receive raw logits for numerical stability.

In decoder training, position `i` predicts token `i + 1`. Shifted inputs and targets create many lessons from one sequence:

```text
input:   <BOS> the cat sat on  the
target:  the   cat sat on  the mat
```

A causal mask prevents each input position from reading later answer tokens while all positions are scored in parallel.

## 8. One loss trains the complete path

Cross-entropy loss becomes large when the model assigns little probability to the actual next token. Backpropagation follows every operation that contributed to those logits:

```text
loss
-> vocabulary head
-> final normalization
-> FFNs
-> attention output and Q/K/V projections
-> token embeddings
```

The notebook verifies non-zero gradients at the embedding table, an attention query projection, an FFN expansion, and the output head after one backward pass.

Backpropagation is a training-time credit-assignment process. It is not another layer used during generation. The forward path predicts; the backward path computes how learned parameters should change.

## 9. Model families reuse the block differently

| Family | What it does | Attention rule | Why it fits |
|---|---|---|---|
| Encoder-only | Builds contextual input representations | Every valid input position can read every other | Classification, extraction, tagging, retrieval |
| Decoder-only | Extends one growing sequence | Every position reads only its visible prefix | Completion, chat, code generation |
| Encoder-decoder | Reads one sequence and writes another | Bidirectional source memory plus causal target and cross-attention | Translation, summarization, structured transformation |

The block components are shared. Masks and the source/target boundary change how information is allowed to move.

## 10. Practical failure modes

- **Using one head for every relation:** independent heads provide separate routing capacity.
- **Assuming heads must be interpretable:** inspect behavior, but do not turn a heatmap into a causal explanation.
- **Removing the FFN:** attention can route information but loses private nonlinear processing.
- **Changing model width inside a sublayer:** residual addition requires matching outer shapes.
- **Normalizing across token positions:** LayerNorm should operate across each token's features.
- **Removing residual paths in deep stacks:** information and gradients must traverse every transformation.
- **Applying softmax before cross-entropy:** pass raw logits to the loss.
- **Calling backpropagation part of inference:** generation uses the forward path unless gradients are explicitly requested.

The durable model is: **multi-head attention gathers several views, the FFN transforms each token privately, normalization conditions each sublayer, residuals preserve continuity, the vocabulary head predicts, and one loss sends responsibility through the whole learned path.**
