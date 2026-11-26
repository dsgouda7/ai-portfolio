# Seq2Seq: The LSTM Encoder-Decoder

Many tasks read one sequence and write another: translation, summarization,
question answering. This is called **sequence-to-sequence**, or **seq2seq**.

**Question:** how can a model read a whole input and then write a *different*
sequence, in its own order and length?

**Running example:** reverse a list. Reading `7 3 4` must produce `4 3 7`. Every
answer can be checked by eye, and there is exactly one correct answer.

The code uses `Vocab`, `PAD`, `BOS`, `EOS`, `IGNORE` and `make_loader` from
[text to batches](03C-minimal-tokens-embeddings-batches-notes.md).

## 1. How we got here

Each earlier model fixed one problem and exposed the next:

| Model | What it learned to do | What it still couldn't do |
| --- | --- | --- |
| Counting (n-grams) | Guess the next word from the last few words | Treats `cat` and `dog` as unrelated; forgets anything outside its short window |
| Word vectors (word2vec) | Give each word a learned row, so similar words share evidence | Is one fixed row per word, not a sentence reader or a writer |
| Recurrent language model (RNN, LSTM) | Read words in order, carry a memory and continue the text | Writes one word for each word it reads, in step |

The last row is the wall. A recurrent language model continues **one stream**:
read a word, predict the next, read it, predict again. Try reversing `7 3 4` that
way:

```text
reads 7        -> must already say 4    but it hasn't seen 4 yet
```

Translation hits the same wall. Word order changes between languages, and the
answer can be longer or shorter than the input.

**The fix: finish listening, then speak.** It reuses both earlier ideas: learned
word rows for the input, and recurrent memory for reading and writing.

**Memory line: a language model continues a stream; seq2seq answers a whole input.**

## 2. Split listening from speaking

```text
READER (encoder)                         WRITER (decoder)
7 -> 3 -> 4 -> [final memory] ------->   BOS -> 4 -> 3 -> 7
                                          says: 4    3    7   EOS
```

Both halves can be LSTMs, but they learn **separate weights** because they have
different jobs:

- the reader learns what to remember;
- the writer learns how to use that memory while writing.

## 3. One example is three rows: the shift

```text
src      7    3    4           what the reader reads
dec_in  BOS   4    3    7      what the writer sees
target   4    3    7   EOS     what the writer must say
```

Read each column top to bottom: *seeing this, say that.* During training the
writer sees the correct answer **one step late**. Supplying the correct previous
word is called **teacher forcing**. It lets one forward pass score every answer
slot at once.

```python
import random
from torch.nn.utils.rnn import pack_padded_sequence

def make_example(source, answer):
    src = torch.tensor(source)
    dec_in = torch.tensor([BOS] + answer)  # what the writer sees
    target = torch.tensor(answer + [EOS])  # what the writer must say
    return src, dec_in, target
```

Shift here, once. Never shift again inside the model or the loss.

**Memory line: the writer sees one step behind what it must say.**

## 4. The answer key

```python
def reversal_pairs(n, vocab, seed=0):
    rng = random.Random(seed)
    words = range(3, len(vocab))  # skip the reserved IDs
    seen = set()                  # a set keeps every source unique
    while len(seen) < n:
        length = rng.randint(2, 5)
        seen.add(tuple(rng.choice(words) for _ in range(length)))
    rows = sorted(seen)
    rng.shuffle(rows)
    return [(list(s), list(reversed(s))) for s in rows]
```

Reversing happens **only here**, to write the answer key. The model never sees this
function; it must learn the job from examples. Unique sources mean the held-out
split never repeats a training question.

## 5. The model

The model's whole promise:

```text
model(src, dec_in) -> logits [B, T, V]   one row of word scores per answer slot
```

```python
class LSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD)  # word table for both halves
        self.encoder = nn.LSTM(d, d, batch_first=True)             # reader
        self.decoder = nn.LSTM(d, d, batch_first=True)             # writer
        self.head = nn.Linear(d, vocab_size)                       # state -> word scores

    def forward(self, src, dec_in):
        lengths = (src != PAD).sum(1).cpu()                        # real length per row
        packed = pack_padded_sequence(self.embed(src), lengths,
                                      batch_first=True, enforce_sorted=False)
        _, (h, c) = self.encoder(packed)                   # final REAL state: [1, B, d] each
        out, _ = self.decoder(self.embed(dec_in), (h, c))  # writer starts from reader's memory
        return self.head(out)                              # [B, T, V]
```

### What crosses the bridge?

An LSTM carries two things:

| State | Plain English |
| --- | --- |
| `h` (hidden) | What's on my mind right now |
| `c` (cell) | What I'm keeping for later |

The reader's final `(h, c)` becomes the writer's starting `(h, c)`. That pair is
**computed numbers**, not a copy of the source words or the reader's weights.

Nobody tells the reader what a good memory looks like. When the writer gets an
answer wrong, the error flows back **through the writer, across the bridge, into
the reader**. The reader learns to remember whatever the writer needed.

**Memory line: the reader learns to remember what the writer needed.**

### Why pack the source?

```text
row 1:  7  3  4  PAD PAD
row 2:  5  6  8   3   4
```

A plain LSTM would keep reading row 1's `PAD`s, and its "final" memory would drift
after the real last word. Packing stops each row at its true length.

The writer doesn't need packing. Its `PAD`s come **after** the real slots, so they
can only affect later slots, which the loss skips with `IGNORE`.

**Memory line: pack the reader so blanks can't rewrite its memory.**

## 6. One training step

```python
def train_epoch(model, loader, optimizer):
    model.train()
    for src, dec_in, target in loader:
        logits = model(src, dec_in)                   # [B, T, V] score every slot at once
        loss = F.cross_entropy(logits.flatten(0, 1),  # [B*T, V] one row per slot
                               target.flatten(),      # [B*T]    one answer per slot
                               ignore_index=IGNORE)   # skip padded slots
        optimizer.zero_grad()
        loss.backward()                               # blame flows back through every weight
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # cap oversized nudges
        optimizer.step()                              # now the weights actually change
```

Flattening only merges batch and time. Each word-score row stays intact. The loop
never looks inside the model, so any model with the same promise can use it.

**Memory line: score every slot at once, grade only real ones, nudge every weight.**

## 7. Generation: feed your own guesses back

At test time nobody supplies the answer. Start with `BOS`, keep your best guess,
and append it as the next input.

```text
BOS            -> 4
BOS 4          -> 3
BOS 4 3        -> 7
BOS 4 3 7      -> EOS   stop
```

```python
@torch.no_grad()
def generate(model, src, max_len=8):
    model.eval()
    out = torch.full((len(src), 1), BOS)                  # every row starts with BOS
    for _ in range(max_len):
        logits = model(src, out)                          # re-read the whole prefix
        next_id = logits[:, -1].argmax(-1, keepdim=True)  # best guess for the newest slot
        out = torch.cat([out, next_id], dim=1)            # own guess becomes next input
        if (out == EOS).any(1).all():                     # every row has said EOS
            break
    return out[:, 1:]                                     # drop BOS
```

Re-reading the whole prefix each step is slow but simple, and works for any model.
A faster LSTM generator carries `(h, c)` forward instead.

## 8. Two report cards

```python
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tf_hits = tf_total = exact = n = 0
    for src, dec_in, target in loader:
        real = target != IGNORE                               # [B, T] slots to grade

        guess = model(src, dec_in).argmax(-1)                 # given the CORRECT history
        tf_hits += (guess == target)[real].sum().item()
        tf_total += real.sum().item()

        gen = generate(model, src, max_len=target.size(1))    # given its OWN history
        gen = F.pad(gen, (0, target.size(1) - gen.size(1)), value=PAD)
        exact += ((gen == target) | ~real).all(1).sum().item()  # every real slot right
        n += len(src)
    return {"teacher_forced_token_acc": round(tf_hits / tf_total, 3),
            "generated_exact_acc": round(exact / n, 3)}

def fit(model, train_loader, valid_loader, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer)
        if epoch % 5 == 0 or epoch == epochs:
            print(epoch, evaluate(model, valid_loader))
```

| Report | Question it answers |
| --- | --- |
| Teacher-forced token accuracy | Given the right history, does it pick the right next word? |
| Generated exact accuracy | Can it finish the whole answer with only its own guesses? |

One early wrong guess becomes the next input, so generation is the harder test.

**Memory line: loss measures scoring; generation measures surviving your own mistakes.**

## 9. Put it together

```python
torch.manual_seed(0)
vocab = Vocab("3 4 5 6 7 8".split())
pairs = reversal_pairs(800, vocab)
examples = [make_example(s, a) for s, a in pairs]
train_loader = make_loader(examples[:640], batch_size=32, shuffle=True)
valid_loader = make_loader(examples[640:], batch_size=64, shuffle=False)

model = LSTMSeq2Seq(len(vocab))
fit(model, train_loader, valid_loader, epochs=15, lr=3e-3)

src, _, _ = next(iter(valid_loader))
for row, guess in zip(src[:3].tolist(), generate(model, src[:3]).tolist()):
    print(vocab.decode(row), "->", vocab.decode(guess))
```

Measured on CPU in about five seconds: generated exact accuracy goes
**0.81 → 0.98 → 1.0** at epochs 5, 10 and 15. For example, `3 7 6 5` becomes
`5 6 7 3`. This is a tiny example with lists of up to five words, not a benchmark.

## 10. Break it to believe it

**Break 1: remove the shift.** Let the writer see the very answer it must say.

```python
def make_example(source, answer):  # BROKEN on purpose
    src = torch.tensor(source)
    dec_in = torch.tensor(answer + [EOS])  # sees the answer itself
    target = torch.tensor(answer + [EOS])
    return src, dec_in, target
```

Measured: teacher-forced accuracy is **1.0** from epoch 5, because the model only
copies its input. Generated exact accuracy stays **0.0**; `4 4 8 8` comes out as
`8 8 8 8 8 8 8 8`. At test time the answer isn't there to copy.

**A perfect training score can mean you leaked the answer.**

**Break 2: cut the bridge.** Start the writer from zeros instead of the reader's
memory.

```python
out, _ = self.decoder(self.embed(dec_in))  # BROKEN: no (h, c) from the reader
```

Measured: teacher-forced accuracy falls to **0.27**, and generated exact accuracy
to **0.0**. Every source produces the **same** answer:

```text
4 4 8 8 -> 3 5 5 5
3 7 6 5 -> 3 5 5 5
6 5 6   -> 3 5 5 5
```

With no bridge, the writer can only learn a generic guess. Everything it knows
about the source crosses that one bridge.

## 11. The limitation: one small memory

The whole source must squeeze through one fixed-size `(h, c)`. Short lists fit.
Long lists crowd it: early details fade while later words keep arriving.

```text
7 3 4 6 8 5 3 7 ... -> [one small memory] -> writer must recall all of it
```

What if the writer could look back at **every** source position whenever it needed
a detail? That idea is **attention**, the core of the Transformer.

## Concept recap

```text
history:  counts -> word rows -> recurrent memory -> separate reader + writer
shift:    dec_in = BOS + answer (sees); target = answer + EOS (says)
reader:   embed -> pack -> LSTM -> final (h, c)
bridge:   (h, c) starts the writer: numbers, not words
writer:   embed(dec_in) -> LSTM -> head -> [B, T, V]
learning: writer's error flows back across the bridge into the reader
train:    [B,T,V] -> flatten -> cross_entropy(ignore IGNORE) -> backward -> step
generate: BOS, guess, append, repeat until EOS
judge:    teacher-forced accuracy AND generated exact accuracy
limit:    the whole source must fit through one (h, c)
```

**See also:** [text to batches](03C-minimal-tokens-embeddings-batches-notes.md)
for tokens, embeddings and padding;
[Transformer encoder-decoder](../../04-transformers/05b-minimal-transformer-encoder-decoder-notes.md)
for the writer that looks back.
