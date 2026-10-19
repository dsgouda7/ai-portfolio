"""Code companion for 03C-fixed-windows-to-recurrent-language-models.ipynb. Run from this directory."""

# %% 1. Keep the old failure in view
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
random.seed(17); np.random.seed(17); torch.manual_seed(17)
torch.set_num_threads(1)
started = time.perf_counter()

# %% 1. Keep the old failure in view
corpus = [f"{clue} {first} {second} now {answer}".split()
          for clue, answer in [("red", "warm"), ("blue", "cool")]
          for first in ["wait", "rest"] for second in ["quietly", "calmly"]]
train_rows = [r for r in corpus if r[1:3] != ["rest", "calmly"]]
test_rows = [r for r in corpus if r[1:3] == ["rest", "calmly"]]
for row in corpus:
    print(" ".join(row), "|", "held out" if row in test_rows else "training")

# %% 1. Keep the old failure in view
vocab = ["<start>"] + sorted({word for row in corpus for word in row})
to_id = {word: i for i, word in enumerate(vocab)}
def encode(rows):
    return torch.tensor([[to_id[word] for word in row] for row in rows])
train_ids, test_ids = encode(train_rows), encode(test_rows)
print("ID addresses:", to_id)

# %% 1. Keep the old failure in view
WINDOW = 3
def window_pairs(rows):
    contexts, answers = [], []
    for row in rows:
        for position in range(1, len(row)):
            prefix = ["<start>"] * WINDOW + row[:position]
            contexts.append([to_id[word] for word in prefix[-WINDOW:]])
            answers.append(to_id[row[position]])
    return torch.tensor(contexts), torch.tensor(answers)
class WindowLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = nn.Embedding(len(vocab), 8)
        self.predictor = nn.Sequential(nn.Linear(WINDOW * 8, 24), nn.Tanh(),
                                       nn.Linear(24, len(vocab)))
    def forward(self, context):
        return self.predictor(self.rows(context).flatten(1))

# %% 1. Keep the old failure in view
window_model = WindowLM()
x, y = window_pairs(train_rows)
optimizer = torch.optim.Adam(window_model.parameters(), lr=0.03)
for step in range(300):
    loss = F.cross_entropy(window_model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad():
    window_probs = window_model(test_ids[:, -4:-1]).softmax(-1)
    window_accuracy = (window_probs.argmax(-1) == test_ids[:, -1]).float().mean().item()
print(f"Fixed-window final-token accuracy: {window_accuracy:.1%}")
print("Takeaway: fitting again does not give the window a missing color.")

# %% 1. Keep the old failure in view
inspect_row = 0  # CHANGE THIS: 1 selects the other clue
print("Whole prefix:", test_rows[inspect_row][:-1])
print("Window input:", test_rows[inspect_row][-4:-1])

# %% 2. Read a word, update the note, predict the next word
class RunningLM(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.rows = nn.Embedding(len(vocab), 8)
        self.reader = nn.RNN(8, width, batch_first=True)
        self.head = nn.Linear(width, len(vocab))
    def forward(self, tokens):
        notes, last = self.reader(self.rows(tokens))
        return self.head(notes), notes

torch.manual_seed(17)
running_model = RunningLM()
optimizer = torch.optim.Adam(running_model.parameters(), lr=0.02)
losses = []
for step in range(350):
    scores, _ = running_model(train_ids[:, :-1])
    loss = F.cross_entropy(scores.flatten(0, 1), train_ids[:, 1:].flatten())
    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(running_model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
with torch.no_grad():
    scores, notes = running_model(test_ids[:, :-1])
    recurrent_probs = scores[:, -1].softmax(-1)
    recurrent_accuracy = (recurrent_probs.argmax(-1) == test_ids[:, -1]).float().mean().item()
    clue_effect = (recurrent_probs[0] - recurrent_probs[1]).abs().max().item()
print(f"RNN training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"Held-out final-token accuracy: window {window_accuracy:.1%}; RNN {recurrent_accuracy:.1%}")
print(f"Largest final probability change after swapping only the clue: {clue_effect:.6f}")
for name, model in [("window", window_model), ("RNN", running_model)]:
    print(name, "parameters:", sum(p.numel() for p in model.parameters()))
if recurrent_accuracy > window_accuracy:
    print("Takeaway: this run used the clue beyond the fixed window on the tiny held-out pair.")
else:
    print("Takeaway: the note has a route, but this run did not establish an accuracy gain.")

# %% 2. Read a word, update the note, predict the next word
clue_word = "red"  # CHANGE THIS: "blue"
probe = torch.tensor([[to_id[w] for w in [clue_word, "rest", "calmly", "now"]]])
with torch.no_grad():
    final = running_model(probe)[0][0, -1].softmax(-1)
print("warm:", round(final[to_id["warm"]].item(), 4),
      "cool:", round(final[to_id["cool"]].item(), 4))

# %% 3. Watch the two notes separate
distances = (notes[0] - notes[1]).norm(dim=-1).numpy()
labels = ["red / blue", "rest", "calmly", "now"]
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(labels, distances, marker="o", color="#ef8a23")
ax.set(ylabel="Distance between learned notes", title="Same later words, different opening clue")
plt.show()
print("Recorded distances:", distances.round(4).tolist())
print("Takeaway: read state differences alongside behavior; visible difference alone is not recall.")

# %% 3. Watch the two notes separate
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.set(xlim=(-0.2, 3.2), ylim=(0, max(distances) * 1.15),
       ylabel="Distance between notes", title="Watch the measured note travel")
ax.set_xticks(range(4), labels)
line, = ax.plot([], [], "o-", color="#ef8a23")
def frame(position):
    line.set_data(range(position + 1), distances[:position + 1])
    return (line,)
animation = FuncAnimation(fig, frame, frames=4, interval=600, blit=True)
plt.close(fig)
display(HTML(animation.to_jshtml()))

# %% 3. Watch the two notes separate
position_to_read = 3  # CHANGE THIS: 0, 1, or 2
print(labels[position_to_read], float(distances[position_to_read]))

# %% Evidence scorecard
print({"window_final_accuracy": window_accuracy, "RNN_final_accuracy": recurrent_accuracy,
       "swapped_clue_probability_effect": clue_effect,
       "RNN_parameters": sum(p.numel() for p in running_model.parameters()),
       "window_parameters": sum(p.numel() for p in window_model.parameters())})
print(f"Elapsed CPU wall time: {time.perf_counter() - started:.1f}s")

# %% 5. Keep the loop; change the running-note mechanism
from language_model import TrainableLanguageModel
from torch.utils.data import DataLoader, TensorDataset


class RecurrentLanguageModel(TrainableLanguageModel):
    def __init__(self, vocabulary_size, cell_type="lstm", embedding_width=8, hidden_width=24):
        super().__init__()
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError("cell_type must be 'rnn' or 'lstm'")
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        recurrent_layer = nn.LSTM if cell_type == "lstm" else nn.RNN
        self.reader = recurrent_layer(embedding_width, hidden_width, batch_first=True)
        self.output = nn.Linear(hidden_width, vocabulary_size)

    def forward(self, input_ids, target_ids=None):
        notes, _ = self.reader(self.embedding(input_ids))
        return self.output(notes)

    def generate(self, prefix_ids, steps):
        self.eval()
        generated = prefix_ids.clone()
        with torch.inference_mode():
            for _ in range(steps):
                next_id = self(generated)[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
        return generated

# %% Follow the inherited method into this LSTM
torch.manual_seed(17)
lstm_model = RecurrentLanguageModel(len(vocab))
lstm_inputs, lstm_targets = train_ids[:, :-1], train_ids[:, 1:]
lstm_loader = DataLoader(TensorDataset(lstm_inputs, lstm_targets), batch_size=len(train_ids))
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.03)
embedding_before = lstm_model.embedding.weight.detach().clone()

assert RecurrentLanguageModel.fit_language_model is TrainableLanguageModel.fit_language_model
lstm_losses = lstm_model.fit_language_model(
    lstm_loader, nn.CrossEntropyLoss(), lstm_optimizer, epochs=350,
)
lstm_model.eval()
with torch.inference_mode():
    lstm_logits = lstm_model(test_ids[:, :-1])
    lstm_accuracy = (lstm_logits[:, -1].argmax(-1) == test_ids[:, -1]).float().mean().item()
    _, (hidden, cell) = lstm_model.reader(lstm_model.embedding(test_ids[:, :-1]))
embedding_change = (lstm_model.embedding.weight.detach() - embedding_before).abs().max().item()

print(f"LSTM loss: {lstm_losses[0]:.4f} -> {lstm_losses[-1]:.4f}")
print(f"LSTM held-out final-token accuracy: {lstm_accuracy:.1%}")
print("Hidden / cell shapes:", tuple(hidden.shape), tuple(cell.shape))
print(f"Largest learned embedding change: {embedding_change:.6f}")
assert lstm_logits.shape == (len(test_ids), test_ids.shape[1] - 1, len(vocab))
assert lstm_losses[-1] < lstm_losses[0]
assert embedding_change > 0

# %% Generation has no answer tensor
prefix = test_ids[:1, :-1]
changed_prefix = prefix.clone()
changed_prefix[:, -1] = to_id["wait"]
with torch.inference_mode():
    original_scores = lstm_model(prefix)
    changed_scores = lstm_model(changed_prefix)
assert torch.allclose(original_scores[:, :-1], changed_scores[:, :-1])
continuation = lstm_model.generate(prefix, steps=1)
print("Generated continuation:", [vocab[index] for index in continuation[0].tolist()])
print("Takeaway: embeddings and LSTM weights learned together; earlier logits do not read future tokens.")
