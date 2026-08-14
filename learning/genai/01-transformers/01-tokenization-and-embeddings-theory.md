# Tokenization and Embeddings: Handwritten Theory Notes

## 1. Text needs two conversions

A model cannot read a Python string. Before a Transformer block can do anything, text passes through two separate mechanisms:

```text
raw text
-> tokenizer chooses pieces and assigns IDs
-> embedder uses each ID to retrieve a trainable vector
```

Keep the jobs separate. The tokenizer is a deterministic data contract. The embedder is a learned model component.

For the running sentence `the cat sat on the mat`, tokenization might produce six word pieces and six integer IDs. The repeated word `the` must receive the same ID both times. The embedding layer then retrieves the same table row at both positions. Position and context have not entered yet, so those two vectors begin identical.

## 2. A tokenizer defines the model's pieces

A tokenizer answers two questions:

1. What pieces will represent this text?
2. Which stable integer identifies each piece?

A whole-word tokenizer is easy to inspect, but every unseen word becomes an unknown token. Subword tokenizers reduce that failure by reusing smaller pieces. The notebook's tiny example can represent `kitten` as `kit + ten` even though `kitten` has no dedicated vocabulary row.

Production tokenizers learn useful pieces from a corpus. BPE repeatedly merges common neighboring pieces. WordPiece chooses pieces that improve a language-model-style score. Unigram begins with many candidates and removes weak ones. Their algorithms differ, but they all produce a fixed mapping between text pieces and IDs.

Important consequences:

- Token IDs are labels, not quantities. ID 12 is not twice as meaningful as ID 6.
- Changing the vocabulary after training changes what embedding rows mean.
- Normalization, whitespace, punctuation, and special-token rules are part of the model contract.
- Real tokenizers often split one visible word into several model tokens.

The tokenizer is therefore part of the saved model, not disposable preprocessing.

## 3. An embedding table is a learned lookup table

An embedding layer stores one vector per vocabulary item:

```text
vocabulary rows          V
features per row         D
embedding table shape    (V, D)
```

An ID selects a row. A sequence of `S` IDs retrieves `S` rows, producing shape `(S, D)` or `(B, S, D)` when a batch axis is present.

The coordinates do not begin as human-written concepts. They usually start from small random values. Training makes them useful. A row can gradually become helpful for syntax, topic, style, or many entangled properties, but individual dimensions usually do not have clean labels such as "animal" or "action."

The same token ID always retrieves the same base row. This does not mean every occurrence keeps the same final representation. Later, position and attention mix in different surroundings, so the two occurrences of `the` can become different contextual vectors.

## 4. Prediction error teaches embedding rows

Suppose the visible prefix is `the cat sat on the` and the expected next token is `mat`.

```text
prefix IDs
-> embedding lookup
-> model computation
-> vocabulary logits
-> loss against "mat"
-> backpropagation
```

Backpropagation follows the computation graph to every parameter that contributed to the prediction. For the embedding table, only rows retrieved by the prefix receive gradients in that step. An unused row was not part of the forward path, so it receives no responsibility from that example.

Repeated tokens reuse one parameter row. If `the` appears twice, both occurrences contribute gradient to the same row. The optimizer combines that evidence when it updates the table.

The notebook uses mean pooling and one output layer only to expose this contract. A Transformer replaces mean pooling with attention and many blocks, but gradients still travel from prediction loss into the token rows used by the input.

One update proves connectivity, not semantic mastery. A useful embedding geometry requires many varied examples.

## 5. Padding and special tokens are real contracts

Special tokens have assigned jobs:

| Token | Typical job |
|---|---|
| `<PAD>` | Fill unused batch positions |
| `<BOS>` | Mark the beginning of generation |
| `<EOS>` | Mark a learned stopping boundary |
| `<UNK>` | Represent text the tokenizer cannot decompose |

Padding is not ordinary content. Attention should not retrieve padded positions, and loss should not grade padded targets. Many embedding layers keep the padding row fixed, but masking remains necessary elsewhere in the model.

`<EOS>` is different: it is meaningful content and usually participates in next-token learning. Confusing EOS with padding prevents the model from learning when to stop.

## 6. Embeddings still do not know order

Compare:

```text
the cat sat on the mat
the mat sat on the cat
```

Both contain the same token IDs with the same counts. If their embedding rows are summed or averaged, the result is identical. The model knows which pieces appeared but not where they appeared.

This failure creates the next chapter. Attention will let token positions exchange information directly, but plain attention is also blind to order unless position enters the comparison.

## 7. Practical failure modes

- **Treating IDs as features:** IDs are row addresses; use an embedding lookup.
- **Changing tokenizer files after training:** old embedding rows no longer match the new IDs.
- **Assuming one word equals one token:** inspect the actual tokenizer pieces.
- **Training the padding row as content:** mask padding in attention and loss.
- **Calling a random or tiny-run neighborhood semantic truth:** embedding plots are diagnostics, not complete explanations.
- **Assuming embeddings contain context:** base rows are context-free; later layers contextualize them.
- **Ignoring order:** token identity alone cannot distinguish permutations.

The durable model is: **the tokenizer fixes the pieces and addresses; the embedder learns the rows; prediction error updates the rows that participated; position and attention are still needed to turn those rows into contextual representations.**
