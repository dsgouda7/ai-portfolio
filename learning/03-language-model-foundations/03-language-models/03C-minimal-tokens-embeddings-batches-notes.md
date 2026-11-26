# From Text to Batches: Tokens, Embeddings and Padding

**Question:** a neural network needs a rectangle of numbers. Text is a list of
words, and every sentence has a different length. How do we get from one to the
other?

Three steps, used by every text model from word2vec to modern LLMs:

```text
text -> IDs (addresses) -> padded rectangle [B, L] -> learned rows [B, L, d]
```

**Running example:** short lists of number-words such as `7 3 4` and `5 6 8 3 4`.

## 1. Words become addresses

A **tokenizer** splits text into pieces and gives each piece an integer ID.

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

PAD, BOS, EOS = 0, 1, 2   # reserved IDs
IGNORE = -100             # "don't grade this slot" (cross_entropy's default)

class Vocab:
    def __init__(self, words):
        self.itos = ["<pad>", "<bos>", "<eos>"] + list(words)  # id -> word
        self.stoi = {w: i for i, w in enumerate(self.itos)}    # word -> id

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == EOS:
                break
            if i not in (PAD, BOS):
                words.append(self.itos[i])
        return " ".join(words)
```

| Token | Job |
| --- | --- |
| `<pad>` | Fills empty seats so rows line up in a batch |
| `<bos>` | "Start writing." The first input of a writer |
| `<eos>` | "I'm done." The last word a writer produces |

Real tokenizers split text into subword pieces (`unbelievable` → `un`, `believ`,
`able`) and add an unknown token. The idea is the same: text in, integer IDs out.

An ID is an **address**, not a meaning. ID 7 is not "bigger" than ID 3.

**Memory line: IDs are addresses, not meanings.**

## 2. Addresses fetch learned rows: embeddings

An **embedding** is a table with one row of numbers per vocabulary entry. An ID
selects its row. The numbers start random; training adjusts them until similar
words get similar rows.

```python
vocab = Vocab("3 4 5 6 7 8".split())
embed = nn.Embedding(len(vocab), 8, padding_idx=PAD)  # table [V, d] = [9, 8]

ids = torch.tensor([[5, 6, 7]])                       # [B, L] = [1, 3] addresses
rows = embed(ids)                                     # [B, L, d] = [1, 3, 8] learned rows
```

- The same ID always fetches the same row; repeated words share one row.
- `padding_idx=PAD` keeps the `<pad>` row at zeros and never trains it.
- Only rows that were fetched receive learning feedback.

word2vec is exactly this kind of table, trained on a simple job such as predicting a
word from its neighbours. Later models keep the table and change what they do with
the rows.

**Memory line: the ID is the address; the row holds the meaning.**

## 3. Uneven lengths: pad into a rectangle

Tensors must be rectangles. Fill each short row with `PAD`:

```text
7  3  4  PAD PAD        real length 3
5  6  8   3   4         real length 5
```

```python
sentences = [torch.tensor(vocab.encode("7 3 4")),
             torch.tensor(vocab.encode("5 6 8 3 4"))]
ids = pad_sequence(sentences, batch_first=True, padding_value=PAD)  # [2, 5]
blank = ids == PAD                                                  # [2, 5] True = empty seat
```

`blank` is a **padding mask**. In PyTorch masks, **`True` means blocked**.

There are two kinds of blank, and they have different jobs:

| Blank | Lives in | Who must respect it |
| --- | --- | --- |
| `PAD` | model inputs | the **model**: blanks must not change real results |
| `IGNORE` | training targets | the **loss**: blanks must not be graded |

**Memory line: PAD protects the computation; IGNORE protects the grade.**

## 4. A reusable batcher

Every training example is a tuple: one or more input rows, then a target row.
The batcher pads inputs with `PAD` and the target with `IGNORE`:

```python
def collate(examples):
    *inputs, target = zip(*examples)  # every example: (input rows..., target row)
    inputs = [pad_sequence(x, batch_first=True, padding_value=PAD) for x in inputs]
    target = pad_sequence(target, batch_first=True, padding_value=IGNORE)
    return (*inputs, target)          # each padded to its longest row in this batch

def make_loader(examples, batch_size, shuffle):
    return DataLoader(examples, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)
```

This works whether a task has one input (a language model) or two (a reader and a
writer). Each batch pads only to its own longest row, which wastes less space than
padding everything to the longest row in the dataset.

## Concept recap

```text
tokenize:  text -> pieces -> integer IDs        (addresses, not meanings)
embed:     IDs [B, L] -> table lookup -> [B, L, d]   (rows are learned)
batch:     uneven rows -> pad to the longest -> [B, L]
masks:     ids == PAD -> True = blocked
two blanks: PAD in inputs (model ignores); IGNORE in targets (loss skips)
```

**See also:** [seq2seq with an LSTM encoder-decoder](03C-minimal-lstm-encoder-decoder-notes.md)
builds a reader and writer on these batches;
[Transformer encoder-decoder](../../04-transformers/05b-minimal-transformer-encoder-decoder-notes.md)
replaces the reader and writer with attention.
