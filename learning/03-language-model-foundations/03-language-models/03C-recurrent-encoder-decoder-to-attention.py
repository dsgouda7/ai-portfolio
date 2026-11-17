"""Code companion for the three-cell 03C reader/writer lesson."""

# %% 3. Use a fixed reading task
import sys
from pathlib import Path

import torch
from torch import nn

# Notebook kernels can start at the repository root instead of this chapter.
CHAPTER_DIR = next(
    folder
    for folder in (
        Path(globals().get("__file__", globals().get("__vsc_ipynb_file__", ""))).resolve().parent,
        Path.cwd(),
        Path.cwd() / "learning" / "03-language-model-foundations" / "03-language-models",
    )
    if (folder / "encoder_decoder_example.py").is_file()
)
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

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

# %% 4. Walk through the architecture and training
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

# %% 4.2 Teach both components with the same answer loss
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
