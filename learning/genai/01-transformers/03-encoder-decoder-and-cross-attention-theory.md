# Encoder-Decoder Transformers and Cross-Attention: Handwritten Theory Notes

## 1. Start with the job: read one sequence, write another

An encoder-decoder Transformer separates the source from the response. The **encoder** reads the complete source and produces one contextual vector for every source position. The **decoder** writes left to right. At each step it can use earlier target tokens and retrieve information from any valid encoder position.

This fits translation, summarization, and structured transformation: read a complete source, then produce a separate target whose wording and length may differ. The notebook uses reversal as a routing microscope. For `[3, 1, 4, 1] -> [1, 4, 1, 3]`, the required source position for every digit output is known, so retrieval can be inspected directly.

![Handwritten encoder-decoder tensor flow from source tokens through cross-attention to target logits](images/03-encoder-decoder-and-cross-attention-theory-01.png)

The encoder embeds source IDs and positions, then applies bidirectional self-attention and feed-forward blocks. "Bidirectional" means no causal mask: every source token may use left and right context. The output has shape `(batch, source, hidden)`. It is an addressable memory containing one enriched vector per source token, not a pooled sentence summary and not vocabulary logits.

The decoder embeds target-prefix IDs and applies three operations in each block: masked self-attention over the target prefix, cross-attention over encoder memory, and a feed-forward transformation. Residual paths and normalization keep information and gradients moving through the stack. A final language-model head maps decoder vectors to `(batch, target, vocabulary)` logits.

## 2. Cross-Attention Keeps Source Roles Separate

Decoder self-attention and cross-attention answer different questions. Masked self-attention asks, "What have I written so far?" Its causal mask prevents a target position from reading future answer tokens. Cross-attention asks, "What source information do I need now?"

In cross-attention, **queries come from decoder states**, while **keys and values come from encoder outputs**. Queries express the current target step's need; source keys advertise what each position contains; source values deliver the selected information. Per head, the weight grid has shape `(target, source)`, so source and target lengths can differ. Cross-attention has no future-source mask because the full source is already known. A source padding mask is still required so padded positions cannot be retrieved.

Keeping every encoder position avoids the fixed-summary bottleneck of early sequence-to-sequence models. Mean-pooling would blur positional identity. Cross-attention instead lets each target step select a different mixture of source values.

For reversal, decoder step 0 should route to source position 3, step 1 to position 2, then positions 1 and 0. Those four digit rows should resemble an anti-diagonal; the later EOS prediction has no one-to-one source position. The repeated `1` shows why position matters: content alone cannot distinguish its two occurrences. Prediction loss must teach the model to combine decoder step, source position, and token content.

Attention maps are diagnostics, not complete explanations. Outputs also depend on value vectors, other heads and layers, residual streams, and feed-forward blocks. Use held-out task accuracy to test behavior and the map to check whether routing matches the expected rule.

## 3. Teacher Forcing and Free-Running Generation

During training, the known target is shifted into aligned sequences:

```text
decoder input  : [BOS, y0, y1, y2, ...]
expected output: [y0,  y1, y2, ..., EOS]
```

The causal mask prevents future-target leakage, but all target positions are still evaluated in one parallel decoder call. Their losses combine into one scalar, and one backward pass normally updates the language-model head, decoder, cross-attention, and encoder.

During generation, the target is unavailable. The encoder runs once, then the decoder predicts one token, feeds that prediction back, and repeats until EOS or a length limit. An early wrong token changes the prefix used at every later step. This mismatch is **exposure bias**: training conditions on gold history, while inference conditions on model history. Teacher-forced token accuracy can therefore overstate end-to-end quality. Always evaluate a genuine free-running loop. Greedy decoding keeps one prefix; beam search keeps several promising prefixes, but does not change the model architecture.

![Handwritten comparison of teacher forcing, free-running decoding, and the two cache types](images/03-encoder-decoder-and-cross-attention-theory-02.png)

## 4. Encoder Memory and Two Kinds of Cache

Three reusable objects are easy to confuse. First, the encoder output is computed once per source and remains fixed during that request. Second, each decoder layer can project that fixed output into its own cross-attention keys and values and cache them. These are layer-specific views, not one universal key/value tensor. Third, decoder self-attention has a separate cache: every generated target token adds keys and values, so this cache grows with the prefix.

The notebook's short `greedy_decode` reuses encoder output but recomputes target-prefix attention. That is correct and easy to inspect. Production inference adds both cache types to avoid repeated projections and prefix work. Fixed encoder activations during inference do not imply frozen encoder parameters during training; target loss normally backpropagates through cross-attention into the encoder.

## 5. Architecture Choice and Failure Modes

| Family | What it does | Internal information flow | Why it fits |
|---|---|---|---|
| Encoder-only | Understands a complete input | Bidirectional self-attention produces contextual input vectors | Classification, tagging, extraction, and retrieval embeddings |
| Decoder-only | Continues one sequence | Causal self-attention produces next-token logits | Completion, chat, code generation, and open-ended generation |
| Encoder-decoder | Reads a source and writes a distinct target | A bidirectional encoder builds memory; a causal decoder retrieves from it | Translation, summarization, and structured transformation |

Decoder-only systems dominate many general deployments because one stack and objective scale broadly. That does not make the encoder-decoder boundary obsolete; separate source memory is useful whenever the input and output play genuinely different roles.

Common implementation failures reveal the contract:

- Removing decoder masking leaks future answers and produces misleading training results.
- Applying a causal mask to cross-attention hides legal source positions; use a padding mask instead.
- Pooling encoder states too early destroys addressable positional detail.
- Failing to shift targets lets the decoder see the token it is graded on.
- Reporting teacher-forced predictions as generation hides compounding errors.
- Re-encoding the source at every step wastes work; encode once and reuse it.
- Ignoring padding or EOS corrupts attention, loss, or stopping behavior.
- Treating one averaged attention map as proof ignores heads, values, layers, and output accuracy.

The durable model is: **the encoder builds an addressable source memory; the causal decoder writes the target; cross-attention lets every writing step ask that memory a new question.**
