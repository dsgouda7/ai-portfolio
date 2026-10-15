# Attention, Position, and RoPE: Handwritten Theory Notes

## 1. Attention lets token positions communicate

Token embeddings begin as separate rows. The vector for `cat` does not yet contain anything from `sat` or `mat`. Attention gives every token position a direct way to retrieve information from other positions.

The smallest attention operation has three steps:

```text
compare the current query with every key
-> normalize the scores into routing weights
-> use those weights to blend the values
```

The result is one contextual vector per query position. Unlike a single pooled sentence vector, attention preserves a separate output row for every token.

Softmax makes routing weights positive and makes each row sum to one. Attention is therefore a soft lookup: a query can retrieve a mixture instead of selecting exactly one value.

## 2. Q, K, and V separate three roles

Every token is projected into three learned views:

- **Query:** what information am I looking for?
- **Key:** what kind of information do I advertise?
- **Value:** what information do I deliver if selected?

A query-key match controls routing. It does not directly specify the content returned. That distinction lets a token advertise one property in its key while sending a different collection of features in its value.

The operation order is:

```text
token vectors
-> Q, K, V projections
-> compare every Q with every K
-> scale and mask scores
-> softmax
-> weighted mixture of V
```

Attention weights are routing coefficients. They are not a complete explanation of a model's decision because values, later layers, residual paths, and FFNs also affect the result.

## 3. Plain attention is blind to order

Content-only self-attention is permutation equivariant. If `the cat sat on the mat` is reversed to `mat the on sat cat the`, the same content comparisons occur in reversed coordinates and the output rows reverse to match. Nothing inside pure attention says which row came first.

This is useful as a consistency property, but fatal when order changes meaning:

```text
the cat sat on the mat
the mat sat on the cat
```

Nothing in a content-only query-key comparison says which row came first or how far apart two rows are. Attention solves communication, not position.

## 4. Additive position is the baseline fix

The original Transformer adds a fixed position pattern to each token vector before attention. Imagine several clock hands moving at different speeds. Their combined angles provide a distinct signature at every position.

The two occurrences of `the` retrieve the same base embedding row, but receive different position patterns. They therefore enter the attention projections as different vectors.

Additive position is a valid and effective design. It makes absolute position available to the model, and learned projections can use that signal to learn relative relationships.

Then comes the nagging question: the clean pattern is added to content **before** learned Q/K projections. The notebook compares pure position with `content + position` after one random query projection. The distance-readability correlation falls from about `0.83` to `0.50`.

That does not prove additive position fails. It proves that relative distance is learnable but not guaranteed to remain explicit in the score geometry. The complaint is now precise: put position where attention actually compares tokens.

## 5. RoPE puts position inside the comparison

RoPE acts after queries and keys are projected and split into heads. It treats each adjacent feature pair as a tiny two-dimensional dial.

```text
position 0 -> no turn
position 1 -> one step at this pair's speed
position 5 -> five steps at this pair's speed
```

Different pairs turn at different speeds. Fast pairs distinguish local changes; slow pairs preserve broader position patterns.

The visual is built in public:

```text
one dial
-> complaint: a vector has several pairs
-> stacked dials across positions
-> complaint: snapshots hide motion
-> animation
```

The failed attempts matter because they reveal what the final picture must explain.

Two properties matter:

1. **Rotation preserves vector length.** It changes an arrow's direction, not its magnitude.
2. **The query-key comparison exposes relative gap.** Move both tokens together and their absolute turns change, but their relative alignment stays the same.

For example, query/key positions `(2, 7)` and `(5, 10)` both have gap five. The notebook verifies that their selected rotated dot products match.

RoPE rotates Q and K because they determine where attention looks. It does not rotate V because V carries the content being retrieved.

RoPE does not guarantee that farther tokens always receive lower scores. Its multi-speed rotations make relative gaps distinguishable; token content and learned projections still determine whether a distant token matters.

## 6. RoPE and cached generation

During autoregressive generation, a token's key is rotated for its position before entering the KV cache. Future tokens reuse that stored rotated key. Only the new token's query and key need new rotations.

This avoids re-rotating the old prefix, but RoPE is not literally free. It adds small elementwise work, cached keys and values still consume memory, and causal masking remains necessary during training and batched attention.

The cache stores layer-specific keys and values, not final predictions. RoPE changes the key geometry stored in that cache; it does not remove the cache itself.

## 7. Why score scaling matters

Wider random query/key vectors tend to produce larger dot products. If raw scores grow with width, softmax becomes nearly one-hot before the model has learned a good reason to be confident. Near saturation, useful gradients shrink.

Scaled dot-product attention divides scores by the square root of head width before softmax:

`attention = softmax(QK^T / sqrt(head_width)) V`

The formula matters less than the effect. As width grows, unscaled attention becomes falsely confident and the softmax slope `p(1-p)` collapses. Scaled attention keeps both peak confidence and sensitivity roughly stable. Scaling protects the learning signal; it is not cosmetic normalization.

## 8. Masks change which routes are legal

An encoder may let every query read every valid input key. A decoder applies a causal mask so position `i` can read only itself and earlier positions. Padding masks block empty batch positions.

Mask scores before softmax. A blocked score should receive zero routing probability, not merely a small value.

RoPE and masking solve different problems:

- RoPE makes position gaps visible in query-key geometry.
- A mask enforces which token pairs are allowed to communicate.

## 9. Practical failure modes

- **Calling attention order-aware by itself:** prove the shuffle-equivariance failure first.
- **Rotating values:** values carry retrieved content; rotate Q and K.
- **Applying RoPE before head splitting:** feature pairs can cross the wrong head boundary.
- **Assuming distance means decay:** relative gaps are available, not monotonically penalized.
- **Skipping score scaling:** wide heads can saturate softmax.
- **Masking after softmax:** illegal routes may retain probability.
- **Reading one heatmap as reasoning:** attention is one routing stage in a larger computation.

The durable model is: **Q asks, K advertises, V delivers; position makes order visible; RoPE turns Q/K pairs so their comparison exposes relative gaps; scaling and masks keep routing numerically useful and legally constrained.**
