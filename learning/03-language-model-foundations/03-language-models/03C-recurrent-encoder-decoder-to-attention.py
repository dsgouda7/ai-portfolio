"""Code companion for 03C-recurrent-encoder-decoder-to-attention.ipynb. Run from this directory."""

# %% 3. Give both models the same controlled reading job
import torch
from torch import nn
from language_data import CAT_WORDS
from encoder_decoder_example import (
    BOS, EOS, VOCABULARY, decode_ids, encode_words, load_cat_corpus,
    teacher_inputs, training_batches,
)

torch.set_num_threads(1)
V = len(VOCABULARY)
corpus = load_cat_corpus()
source = torch.tensor([encode_words(CAT_WORDS)])
target = torch.cat([source.flip(1), source.new_full((1, 1), EOS)], dim=1)
previous = teacher_inputs(target)
print("Source:", " ".join(CAT_WORDS))
print("Previous inputs:", decode_ids(previous[0]))
print("Current answers:", decode_ids(target[0]))
print("Frozen train / held-out counts:",
      {length: (len(rows), len(corpus["test"][length]))
       for length, rows in corpus["train"].items()})

# %% 4. Open the reader/writer before training it
class EncoderDecoder(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 24)
        self.encoder = nn.GRU(24, 64, batch_first=True)
        self.decoder = nn.GRU(24, 64, batch_first=True)
        self.head = nn.Linear(64, vocabulary_size)

    def encode(self, source):
        _, state = self.encoder(self.embedding(source))
        return state

    def decode(self, previous, state):
        writing, state = self.decoder(self.embedding(previous), state)
        return self.head(writing), state

    def forward(self, source, previous):
        logits, _ = self.decode(previous, self.encode(source))
        return logits

# %% 4.6 That teaches a summary. Can the writer reopen a source detail?
torch.manual_seed(31)
model = EncoderDecoder(V)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
for update, (source, _, target) in enumerate(training_batches(corpus), 1):
    previous = torch.cat([target.new_full((len(target), 1), BOS), target[:, :-1]], dim=1)
    logits = model(source, previous)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if update in (1, 1000):
        print(f"Update {update}: current-word loss {loss.item():.4f}")

# %% 5. Can it write without the teacher's history?
def generate(model, words=CAT_WORDS):
    source = torch.tensor([encode_words(words)])
    chosen = []
    model.eval()
    with torch.no_grad():
        state = model.encode(source)
        previous = source.new_full((1, 1), BOS)
        for _ in range(len(words) + 1):
            logits, state = model.decode(previous, state)
            previous = logits[:, -1].argmax(-1, keepdim=True)
            chosen.append(previous.item())
            if chosen[-1] == EOS:
                break
    return decode_ids(chosen)


source_to_reverse = CAT_WORDS
print("Source:", " ".join(source_to_reverse))
print("Generated:", " ".join(generate(model, source_to_reverse)))
print("Required by the separate task:", " ".join([*reversed(source_to_reverse), "<eos>"]))
