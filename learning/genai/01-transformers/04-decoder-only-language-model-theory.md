# Decoder-Only Language Models: Handwritten Theory Notes

## 1. Start with the job: continue one growing tape

A decoder-only language model uses one sequence for both the prompt and its continuation. Think of it as a **reader and writer sharing one growing tape**. It reads the tokens already present, predicts one next token, appends that token, and repeats.

`the cat sat on the mat`

The key rule is **future-blind, not context-free**. A token may use itself and every token to its left, but nothing to its right. During training, the complete sentence is available for parallel processing. Without a causal mask, the state after `cat` could inspect `sat` before being graded on predicting `sat`. The mask removes that dishonest shortcut. During generation, future tokens do not exist yet, so the same rule happens naturally.

The notebook uses one word per token to keep the example readable. Real tokenizers usually split text into subwords. That changes sequence length, but not the next-token task.

Parts 1-3 kept one cat sentence fixed so only the mechanism changed. This chapter needs several plausible continuations and repeated entities, so the notebook expands to a tiny Riverside corpus. The tensor path is unchanged; only the practice workload becomes rich enough to expose training and generation.

## 2. Shapes and the decoder block

For batch size `B`, sequence length `S`, model width `d`, and vocabulary size `V`:

```text
token IDs          (B, S)
embeddings + position
hidden states      (B, S, d)
output logits      (B, S, V)
```

Token embeddings carry token identity; positional information carries order. Each decoder block then has two main stages:

1. **Masked multi-head self-attention** lets each position gather useful information from visible earlier positions. Different heads can focus on different relationships.
2. **Feed-forward network** transforms each position's features after attention has mixed context.

Layer normalization stabilizes feature scales. Residual connections add each stage's result back to the existing stream, preserving useful information and helping gradients move through deep stacks. Repeating the block makes representations increasingly contextual: the final position can summarize information accumulated through earlier positions and layers.

The complete learning path is:

```text
token IDs -> embeddings + position -> causal attention -> FFN
-> vocabulary logits -> shifted next-token loss -> backpropagation
```

The loss does not update only the output head. Its gradients flow through the stacked decoder blocks and into the token embeddings, so the whole path learns together.

![Decoder-only component and tensor flow](images/02-decoder-only-language-model-theory-01.png)

## 3. Causal masking and shifted labels

Attention normally compares every query position with every key position. The causal mask blocks the upper-right part of that comparison table. Row 0 can use only token 0; row 1 can use tokens 0 and 1; the final row can use the whole visible prefix. Blocked scores receive zero probability after softmax.

Training labels are the same sequence shifted by one position:

```text
tokens:   the   cat   sat   on
inputs:   the   cat   sat
targets:  cat   sat   on
```

So the logits after `the` are judged against `cat`, the logits after `the cat` against `sat`, and so on. One sequence supplies several supervised lessons. The causal mask is essential because it stops the representation at each input position from seeing its target in advance.

At each valid position, next-token loss takes the negative log probability assigned to the actual following token. It then averages those values across all valid target positions. Padding positions must be hidden from attention where appropriate and excluded from this average.

## 4. Training versus generation

Training is parallel across positions. The full known sequence enters once, producing all position logits together. The shifted targets produce one loss value per valid position; those values are averaged, backpropagated, and used for one optimizer update. Teacher forcing is efficient because the correct earlier tokens are already known.

Generation is serial because the next input depends on the model's previous choice:

```text
prompt -> logits at last position -> choose token -> append
	-> new longer prefix        -> choose token -> append ...
```

The loop stops at an end token, a length limit, or another stopping rule. Only the last-position logits are used to choose each new token, even though all visible positions participate in the forward pass.

## 5. Temperature and top-k

Greedy decoding always chooses the highest-logit token. Sampling instead draws from the probability distribution. Temperature changes its sharpness:

Divide the next-token logits by temperature before applying softmax. A temperature below one makes high-scoring tokens more dominant; a temperature above one spreads probability more evenly. Top-k sampling first keeps only the `k` highest-scoring candidates, then samples among them. It blocks very unlikely tokens, but a small `k` can also discard a good long-tail choice.

## 6. KV cache versus recomputation

The notebook's simple generation loop recomputes the entire prefix after every appended token. That is clear but wasteful: earlier tokens have not changed, so their attention keys and values are repeatedly rebuilt.

A KV cache stores each layer's earlier key and value vectors. For a new token, the model computes only that token's new query, key, and value, attends over the cached history, and appends the new key and value. The cache grows with the sequence and uses memory to save repeated computation. It stores per-layer keys and values, not final token predictions. With the same arithmetic and token choices, caching is an optimization and should preserve the prediction distribution.

![Training, autoregressive inference, causal masking, and KV-cache flow](images/02-decoder-only-language-model-theory-02.png)

## 7. Why this family fits generation

Decoder-only models are ideal when input and output naturally form one growing sequence. Completion, chat, code generation, and open-ended generation all fit the repeated question "what token comes next after this prefix?" One stack and one next-token objective can cover both prompt and continuation.

The fit is less direct when a complete source and a distinct target have separate roles. Translation and tightly grounded summarization can still be prompted on one tape, but an encoder-decoder model makes the source memory and target writer explicit.

## 8. Failure modes

- **Future leakage:** masking the wrong triangle gives unrealistically good training results and poor generation. Confirm that position `i` cannot read any position after `i`.
- **Shift errors:** logits at position `i` predict position `i + 1`, not the current token. Print a tiny input-target alignment before training.
- **Padding contamination:** padded keys can distort attention, and padded targets can distort loss. Apply padding masks and ignore invalid targets.
- **Training-generation confusion:** training may score all positions in parallel; generation must feed each selected token back into the next step.
- **Compounding errors:** a weak sampled token becomes context for later predictions. Temperature, top-k, and stopping rules affect how quickly generation drifts or repeats.
- **Context overflow:** tokens outside the supported window cannot be attended to. Truncate, chunk, retrieve relevant text, or use a longer-context model.
- **Cache mistakes:** preserve every layer's key/value order and invalidate the cache if the prefix or model state changes.
- **Over-reading attention:** an attention weight shows where one head looked, not the model's complete reasoning. Values, residual paths, feed-forward stages, and later layers also shape the result.
