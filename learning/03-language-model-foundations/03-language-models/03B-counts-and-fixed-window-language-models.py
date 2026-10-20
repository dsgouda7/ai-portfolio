"""Code companion for 03B-counts-and-fixed-window-language-models.ipynb. Run from this directory."""

# %% 1. Count what comes after now
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

# %% 1. Count what comes after now
corpus = [f"{clue} {first} {second} now {answer}".split()
          for clue, answer in [("red", "warm"), ("blue", "cool")]
          for first in ["wait", "rest"] for second in ["quietly", "calmly"]]
train_rows = [r for r in corpus if r[1:3] != ["rest", "calmly"]]
test_rows = [r for r in corpus if r[1:3] == ["rest", "calmly"]]
for row in corpus:
    print(" ".join(row), "|", "held out" if row in test_rows else "training")

# %% 1. Count what comes after now
from collections import Counter, defaultdict
single_counts = Counter(word for row in train_rows for word in row)
next_counts = defaultdict(Counter)
for row in train_rows:
    for previous, answer in zip(row, row[1:]):
        next_counts[previous][answer] += 1
now_counts = next_counts["now"]
print("Single-token counts:", dict(single_counts))
print("After now:", dict(now_counts))
bigram_final = max(now_counts.values()) / sum(now_counts.values())
print(f"Best constant choice after now: {bigram_final:.1%}")
fig, ax = plt.subplots(figsize=(5, 2.4))
ax.bar(list(now_counts), list(now_counts.values()), color=["#2c7fb8", "#ef8a23"])
ax.set(ylabel="Observed count", title="One previous word leaves two answers")
plt.show()
print("Takeaway: now alone does not identify the earlier color clue.")

# %% 1. Count what comes after now
previous_word = "quietly"  # CHANGE THIS: try "now" or "red"
print(previous_word, "->", dict(next_counts[previous_word]))

# %% 2. Give each position its own seat
vocab = ["<start>"] + sorted({word for row in corpus for word in row})
to_id = {word: i for i, word in enumerate(vocab)}
def encode(rows):
    # Equal-length word lists -> IDs [N,T]; keep row and time order so later answer slices align.
    return torch.tensor([[to_id[word] for word in row] for row in rows])
train_ids, test_ids = encode(train_rows), encode(test_rows)
print("ID addresses:", to_id)

# %% 2. Give each position its own seat
WINDOW = 3
def window_pairs(rows):
    contexts, answers = [], []
    for row in rows:
        for position in range(1, len(row)):
            prefix = ["<start>"] * WINDOW + row[:position]
            contexts.append([to_id[word] for word in prefix[-WINDOW:]])
            answers.append(to_id[row[position]])
    # One ordered window per answer: contexts [N,WINDOW], answers [N]; start IDs fill missing left seats.
    return torch.tensor(contexts), torch.tensor(answers)
class WindowLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = nn.Embedding(len(vocab), 8)
        self.predictor = nn.Sequential(nn.Linear(WINDOW * 8, 24), nn.Tanh(),
                                       nn.Linear(24, len(vocab)))
    def forward(self, context):
        # IDs [B,W] -> rows [B,W,8] -> [B,W*8], preserving each seat's feature block.
        # The shared W*8 -> 24 -> V projections score words without averaging away order.
        return self.predictor(self.rows(context).flatten(1))

# %% 2. Give each position its own seat
model = WindowLM()
a = torch.tensor([[to_id[w] for w in ["red", "wait", "quietly"]]])
# Reverse the time/seat axis only: [1,3] -> [1,3], same words in a different order.
b = a.flip(1)
with torch.no_grad():
    # Mean removes seats: [1,3,8] -> [1,8]; comparing row averages cannot detect their order.
    bag_gap = (model.rows(a).mean(1) - model.rows(b).mean(1)).abs().max().item()
    ordered_gap = (model(a) - model(b)).abs().max().item()
print(f"Bag row-average gap: {bag_gap:.8f}")
print(f"Ordered score gap before training: {ordered_gap:.8f}")
print("Takeaway: seats preserve an order difference; that alone does not prove useful learning.")
x, y = window_pairs(train_rows)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
losses = []
for step in range(300):
    loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
print(f"Training loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
fig, ax = plt.subplots(figsize=(5, 2.4))
ax.plot(losses, color="#2c7fb8")
ax.set(xlabel="Update", ylabel="Next-token loss", title="Learning cannot remove missing input")
plt.show()

# %% 2. Give each position its own seat
probe_words = ["red", "wait", "quietly"]  # CHANGE THIS: replace only "red" with "blue"
with torch.no_grad():
    # One [1,3] window -> [1,V] scores; softmax keeps [1,V], then [0] selects one word distribution [V].
    probabilities = model(torch.tensor([[to_id[w] for w in probe_words]])).softmax(-1)[0]
print([(vocab[i], round(probabilities[i].item(), 3))
       for i in probabilities.topk(3).indices.tolist()])

# %% 3. The clue is one seat too far away
# Rows [2,5] -> final windows [2,3] and answers [2]; the opening color is deliberately outside the window.
contexts = test_ids[:, -4:-1]
answers = test_ids[:, -1]
with torch.no_grad():
    predictions = model(contexts).softmax(-1)
    blind_gap = (predictions[0] - predictions[1]).abs().max().item()
    # argmax reduces vocabulary [2,V] -> chosen IDs [2]; compare matching answers and mean -> accuracy scalar.
    window_final = (predictions.argmax(-1) == answers).float().mean().item()
print("Inputs:", [[vocab[i] for i in row] for row in contexts.tolist()])
print("Required answers:", [vocab[i] for i in answers.tolist()])
print(f"Maximum probability difference: {blind_gap:.8f}")
print(f"Held-out final-token accuracy: {window_final:.1%}")
assert torch.equal(contexts[0], contexts[1]) and blind_gap == 0
print("Takeaway: identical windows force identical predictions, not identical correct answers.")

# %% 3. The clue is one seat too far away
visible_words = 3  # CHANGE THIS: try 4
# [2,5] -> [2,visible_words]; widening this inspection reveals the clue but does not resize the trained model.
visible = test_ids[:, -(visible_words + 1):-1]
print([[vocab[i] for i in row] for row in visible.tolist()])
print("Still identical?", torch.equal(visible[0], visible[1]))

# %% 4. Toy-to-real bridge and evidence scorecard
print({"bigram_final_accuracy": bigram_final, "bag_gap": bag_gap,
       "ordered_score_gap": ordered_gap, "window_final_accuracy": window_final,
       "identical_window_probability_gap": blind_gap})
print(f"Elapsed CPU wall time: {time.perf_counter() - started:.1f}s")
