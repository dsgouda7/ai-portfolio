# Transformer Encoder-Decoder

**Question:** a model must read one sequence and write another. How can the writer
look back at any source word whenever it needs one?

**Running example:** reverse a list. Reading `7 3 4` must produce `4 3 7`.

## 1. Keep every note, not one summary

An older design, the recurrent encoder-decoder, reads the source one word at a
time and hands the writer **one final memory**. Long sources crowd that memory:
early details fade.

```text
recurrent:    7  3  4  ->  [one memory]  ->  writer
Transformer:  7  3  4  ->  [note][note][note]  <- writer looks up what it needs
```

Imagine answering from your notes. The recurrent model gives you one index card.
The Transformer gives you the whole notebook, with one page per source word.

**Memory line: don't squeeze the source into one card; keep every page open.**

## 2. What goes in, what comes out

Words arrive as integer IDs. Three IDs are reserved: `PAD` fills empty seats in a
batch, `BOS` means "start writing", and `EOS` means "I'm done".

During training the writer sees the correct answer **one step late**:

```text
src      7    3    4           [B, S]  what the encoder reads
dec_in  BOS   4    3    7      [B, T]  what the decoder sees
target   4    3    7   EOS     [B, T]  what the decoder must say
```

Read each column top to bottom: *seeing this, say that.* Supplying the correct
previous word is called **teacher forcing**.

```text
model(src, dec_in) -> logits [B, T, V]   one row of word scores per answer slot
```

## 3. Looking things up: attention

To use the notebook, the writer needs a way to find the right page:

- it asks a **question** (query: what am I looking for?);
- every page has a **label** (key: what do I contain?);
- every page has **content** (value: what do I hand over?).

Compare the question with every label, turn the matches into weights that add to
1, and blend the content using those weights.

```python
def attention(q, k, v, blocked):
    # q: [B, Tq, d] questions; k, v: [B, Tk, d] labels and content
    # blocked: True where looking is forbidden (broadcasts to [B, Tq, Tk])
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5  # [B, Tq, Tk] question-label match
    scores = scores.masked_fill(blocked, float("-inf"))   # forbidden -> weight 0
    weights = scores.softmax(-1)                          # each question's weights sum to 1
    return weights @ v                                    # [B, Tq, d] blended content
```

Why divide by `sqrt(d)`? Wider vectors give bigger dot products. Without scaling,
softmax becomes nearly winner-take-all too early, and learning stalls.

**Memory line: questions find labels; labels choose content.**

## 4. Three places to look

The same `attention` function is used three ways:

| Where | Who asks | Who answers | Blocked |
| --- | --- | --- | --- |
| Encoder self-attention | each source word | other source words | source `PAD` |
| Decoder self-attention | each writer slot | earlier writer slots | the future and `PAD` |
| Cross-attention | each writer slot | every source note | source `PAD` |

Cross-attention replaces the recurrent model's single handoff. Each writer slot
reads the source directly.

## 5. Attention can't see order, so add it

Attention compares contents. Shuffle the source and the same matches happen in a
shuffled order. Nothing says which word came first.

A recurrent model gets order for free by reading one word at a time. The
Transformer reads all positions together, so it adds a learned **position row** to
each word row:

```python
x = embed(ids) + pos(torch.arange(ids.size(1)))  # word meaning + seat number
```

**Memory line: attention knows what, not where; position rows supply where.**

## 6. Masks: two different rules

| Mask | Rule | Shape |
| --- | --- | --- |
| Padding mask | Don't look at empty seats | `[B, S]` or `[B, T]` |
| Causal mask | Writer slot `t` can't see slot `t+1` or later | `[T, T]` |

```text
causal mask (True = blocked)    slot 0 sees: BOS
        BOS  4   3   7          slot 1 sees: BOS 4
BOS      .   X   X   X          slot 2 sees: BOS 4 3
4        .   .   X   X          slot 3 sees: BOS 4 3 7
3        .   .   .   X
7        .   .   .   .
```

Why isn't a causal mask needed in a recurrent writer? It reads left to right, so
the future doesn't exist yet. The Transformer processes every writer slot
**at once**. Without the mask, slot 0 could peek at slot 1, which holds the exact
answer it must say.

**Memory line: padding masks hide empty seats; the causal mask hides the answer key.**

## 7. The model

PyTorch packages self-attention, cross-attention, feed-forward layers, residual
paths and normalization into `nn.Transformer`. You supply word rows, position rows
and masks.

```python
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d=64, heads=4, layers=2, max_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD)  # word ID -> learned row
        self.pos = nn.Embedding(max_len, d)                        # one learned row per seat
        self.core = nn.Transformer(d, heads, layers, layers, dim_feedforward=4 * d,
                                   dropout=0.0, batch_first=True, norm_first=True)
        self.head = nn.Linear(d, vocab_size)                       # state -> word scores

    def add_position(self, ids):
        return self.embed(ids) + self.pos(torch.arange(ids.size(1)))  # [B, L, d]

    def forward(self, src, dec_in):
        src_pad = src == PAD                     # [B, S] True = empty source seat
        dec_pad = dec_in == PAD                  # [B, T] True = empty writer seat
        future = nn.Transformer.generate_square_subsequent_mask(
            dec_in.size(1), dtype=torch.bool)    # [T, T] True = future slot
        out = self.core(self.add_position(src), self.add_position(dec_in),
                        tgt_mask=future,                 # decoder self-attention: no peeking
                        src_key_padding_mask=src_pad,    # encoder self-attention
                        tgt_key_padding_mask=dec_pad,    # decoder self-attention
                        memory_key_padding_mask=src_pad) # cross-attention
        return self.head(out)                    # [B, T, V] one row of word scores per slot
```

Read the four mask arguments as answers to: *who is looking, and where is it not
allowed to look?*

## 8. Train, generate and evaluate

Training is the standard loop: score all slots, compute cross-entropy against
`target` (skipping padded slots), backpropagate, update. Generation starts from
`BOS` and feeds back its own guesses until it says `EOS`.

With the batching helpers from [text to batches](../03-language-model-foundations/03-language-models/03C-minimal-tokens-embeddings-batches-notes.md),
and the data, `fit` and `generate` helpers from the
[LSTM seq2seq notes](../03-language-model-foundations/03-language-models/03C-minimal-lstm-encoder-decoder-notes.md):

```python
model = TransformerSeq2Seq(len(vocab))
fit(model, train_loader, valid_loader, epochs=25, lr=1e-3)

src, _, _ = next(iter(valid_loader))
for row, guess in zip(src[:3].tolist(), generate(model, src[:3]).tolist()):
    print(vocab.decode(row), "->", vocab.decode(guess))
```

The training loop never looks inside the model. It only relies on
`model(src, dec_in) -> [B, T, V]`, so any encoder-decoder can use it.

Measured on CPU in about ten seconds: generated exact accuracy (whole answer
right, using only its own guesses) is **0.99** at epoch 5 and **1.0** from epoch 10.
PyTorch may print a nested-tensor warning; it is harmless. Five-word lists are too
short to crowd a recurrent model's single memory, so this doesn't show which design
is better. Longer sources are where one card fails.

## 9. Break it to believe it

Delete the causal mask: change `tgt_mask=future` to `tgt_mask=None`.

**Predict, then run.** Measured: teacher-forced accuracy (correct previous words
supplied) is **1.0** from epoch 5, because each slot copies the next input, which is
its own answer. Generated exact accuracy stays **0.0**, and every generated answer
is **empty**: the model says `EOS` immediately. At test time, the next input it
learned to copy doesn't exist yet.

Any setup that lets the writer see its current answer during training fails this
way: perfect practice, zero real performance.

**Memory line: when practice is perfect and generation is zero, look for a leak.**

## 10. Compared with a recurrent encoder-decoder

| | Recurrent (e.g. LSTM) | Transformer |
| --- | --- | --- |
| Source handoff | one final memory | every source position, via cross-attention |
| Word order | free: reads one word at a time | must add position rows |
| Future answers | can't see them: reads left to right | blocked by the causal mask |
| Training speed | one slot after another | all slots at once |
| Data, loss, training, generation, evaluation | same | same |

## Concept recap

```text
attention: softmax(q . k / sqrt(d) + mask) -> weighted blend of v
encoder:   word + position rows -> self-attention -> one note per source word
decoder:   BOS + answer -> causal self-attention -> cross-attention to notes -> head
masks:     PAD = empty seats; causal = no peeking at future answers
contract:  model(src, dec_in) -> [B, T, V], the same for any encoder-decoder
```

**See also:** [text to batches](../03-language-model-foundations/03-language-models/03C-minimal-tokens-embeddings-batches-notes.md)
for tokens, embeddings and padding;
[LSTM seq2seq notes](../03-language-model-foundations/03-language-models/03C-minimal-lstm-encoder-decoder-notes.md)
for the one-memory design, training, generation and evaluation.
