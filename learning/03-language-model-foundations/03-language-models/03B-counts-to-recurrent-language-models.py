"""Code companion for Chapter 03B. Run from the chapter folder or repository root."""

# %% setup
import random
import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
random.seed(17); np.random.seed(17); torch.manual_seed(17)
torch.set_num_threads(1)

# Notebook kernels can start at the repository root instead of this chapter.
CHAPTER_DIR = next(
    folder
    for folder in [
        Path(globals().get("__vsc_ipynb_file__", "")).resolve().parent,
        Path.cwd(),
        Path.cwd() / "learning" / "03-language-model-foundations" / "03-language-models",
    ]
    if (folder / "language_model.py").exists()
)
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))
started = time.perf_counter()
EVIDENCE_DIR = CHAPTER_DIR / "images"
EVIDENCE_DIR.mkdir(exist_ok=True)

def save_evidence(fig, filename, title, description):
    fig.savefig(EVIDENCE_DIR / filename, dpi=160, bbox_inches="tight",
                metadata={"Title": title, "Description": description,
                          "Source": "03B-counts-to-recurrent-language-models"})

# %% counts
from collections import Counter, defaultdict
from language_data import CAT_SENTENCE, CAT_WORDS, CAT_FAMILY_WORDS
corpus = []
for verb, surface in [("sat", "rug"), ("slept", "mat"), ("slept", "rug"), ("sat", "mat")]:
    for animal, ending in [("cat", "purring"), ("dog", "panting")]:
        row = list(CAT_WORDS)
        row[1], row[2], row[-1] = animal, verb, surface
        corpus.append(row + [ending])
held_out_pair = (CAT_WORDS[2], CAT_WORDS[-1])
train_rows = [row for row in corpus if (row[2], row[-2]) != held_out_pair]
test_rows = [row for row in corpus if (row[2], row[-2]) == held_out_pair]
assert len(train_rows) == 6 and len(test_rows) == 2 and len({tuple(row) for row in corpus}) == 8
assert tuple(test_rows[0][:-1]) == CAT_WORDS
assert test_rows[0][1] != test_rows[1][1] and test_rows[0][-1] != test_rows[1][-1]
assert all(a == b for position, (a, b) in enumerate(zip(*[row[:-1] for row in test_rows]))
           if position != 1)
print("Canonical probe:", CAT_SENTENCE)
print("Made-up rule: cat -> purring; dog -> panting. Not a rule of general English.")
for row in corpus:
    print(" ".join(row), "|", "held out" if row in test_rows else "training")
single_counts = Counter(word for row in train_rows for word in row)
next_counts = defaultdict(Counter)
for row in train_rows:
    for previous, answer in zip(row, row[1:]):
        next_counts[previous][answer] += 1
final_word_counts = next_counts[CAT_WORDS[-1]]
bigram_final = max(final_word_counts.values()) / sum(final_word_counts.values())
print("Single-token counts:", dict(single_counts))
print(f"After {CAT_WORDS[-1]}:", dict(final_word_counts))
print(f"Best constant choice after {CAT_WORDS[-1]}: {bigram_final:.1%}")
fig, ax = plt.subplots(figsize=(5, 2.4))
ax.bar(list(final_word_counts), list(final_word_counts.values()), color=["#2c7fb8", "#ef8a23"])
ax.set(xlabel=f"Next word after {CAT_WORDS[-1]}", ylabel="Observed training count",
       title="The last word cannot identify the animal")
save_evidence(fig, "03b-cat-next-word-counts.png", "Next-word counts after mat",
              "Measured training counts: one purring and one panting after mat.")
plt.show()
print(f"Takeaway: {CAT_WORDS[-1]} alone does not identify the earlier animal clue.")

# %% count_probe
previous_word = CAT_WORDS[-1]  # CHANGE THIS: try "rug" or "cat"
print(previous_word, "->", dict(next_counts[previous_word]))

# %% vocabulary
vocab = ["<start>"] + sorted(CAT_FAMILY_WORDS)
to_id = {word: i for i, word in enumerate(vocab)}
def encode(rows):
    # Equal-length word lists -> IDs [N,T], preserving row and time order.
    return torch.tensor([[to_id[word] for word in row] for row in rows])
train_ids, test_ids = encode(train_rows), encode(test_rows)
print("ID addresses:", to_id)

# %% window_model
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
        # [B,W] -> [B,W,8] -> [B,W*8]: joining keeps each seat's feature block.
        return self.predictor(self.rows(context).flatten(1))

# %% window_training
window_model = WindowLM()
a = encode([CAT_WORDS[:WINDOW]])
b = a.flip(1)
with torch.no_grad():
    bag_gap = (window_model.rows(a).mean(1) - window_model.rows(b).mean(1)).abs().max().item()
    ordered_gap = (window_model(a) - window_model(b)).abs().max().item()
print(f"Bag row-average gap: {bag_gap:.8f}")
print(f"Ordered score gap before training: {ordered_gap:.8f}")
x, y = window_pairs(train_rows)
optimizer = torch.optim.Adam(window_model.parameters(), lr=0.03)
window_losses = []
for step in range(300):
    loss = F.cross_entropy(window_model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    window_losses.append(loss.item())
print(f"Window training loss: {window_losses[0]:.4f} -> {window_losses[-1]:.4f}")
fig, ax = plt.subplots(figsize=(5, 2.4))
ax.plot(window_losses, color="#2c7fb8")
ax.set(xlabel="Update", ylabel="Next-token loss", title="Learning cannot remove missing input")
save_evidence(fig, "03b-cat-window-learning-curve.png", "Ordered-window learning curve",
              "Measured next-word loss for the three-seat window: seed 17, 300 Adam updates, lr 0.03.")
plt.show()
print("Takeaway: seats preserve order; useful learning still needs the relevant clue.")

# %% window_probe
probe_words = list(CAT_WORDS[:WINDOW])  # CHANGE THIS: replace only the animal with "dog"
with torch.no_grad():
    probabilities = window_model(torch.tensor([[to_id[w] for w in probe_words]])).softmax(-1)[0]
print([(vocab[i], round(probabilities[i].item(), 3))
       for i in probabilities.topk(3).indices.tolist()])

# %% window_boundary
contexts, answers = test_ids[:, -(WINDOW + 1):-1], test_ids[:, -1]
with torch.no_grad():
    window_probs = window_model(contexts).softmax(-1)
    blind_gap = (window_probs[0] - window_probs[1]).abs().max().item()
    window_accuracy = (window_probs.argmax(-1) == answers).float().mean().item()
print("Inputs:", [[vocab[i] for i in row] for row in contexts.tolist()])
print("Required answers:", [vocab[i] for i in answers.tolist()])
print(f"Maximum probability difference: {blind_gap:.8f}")
print(f"Held-out final-token accuracy: {window_accuracy:.1%}")
assert torch.equal(contexts[0], contexts[1]) and blind_gap == 0
assert answers[0] != answers[1]
print("Takeaway: identical windows force identical predictions, not identical correct answers.")

# %% visible_words
visible_words = WINDOW  # CHANGE THIS: try 4, then 5
visible = test_ids[:, -(visible_words + 1):-1]
print([[vocab[i] for i in row] for row in visible.tolist()])
print("Still identical?", torch.equal(visible[0], visible[1]))
assert torch.equal(test_ids[0, 2:-1], test_ids[1, 2:-1]), "Four seats still exclude the animal."
assert not torch.equal(test_ids[0, 1:-1], test_ids[1, 1:-1]), "Five seats recover the animal."

# %% running_model
class RunningLM(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.rows = nn.Embedding(len(vocab), 8)
        self.reader = nn.RNN(8, width, batch_first=True)
        self.head = nn.Linear(width, len(vocab))
    def forward(self, tokens):
        notes, _ = self.reader(self.rows(tokens))
        return self.head(notes), notes

torch.manual_seed(17)
running_model = RunningLM()
optimizer = torch.optim.Adam(running_model.parameters(), lr=0.02)
rnn_losses = []
for step in range(350):
    # [N,7] -> inputs/targets [N,6]; each score predicts the following word.
    scores, _ = running_model(train_ids[:, :-1])
    loss = F.cross_entropy(scores.flatten(0, 1), train_ids[:, 1:].flatten())
    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(running_model.parameters(), 1.0)
    optimizer.step()
    rnn_losses.append(loss.item())
with torch.no_grad():
    scores, notes = running_model(test_ids[:, :-1])
    recurrent_probs = scores[:, -1].softmax(-1)
    recurrent_accuracy = (recurrent_probs.argmax(-1) == answers).float().mean().item()
    clue_effect = (recurrent_probs[0] - recurrent_probs[1]).abs().max().item()
print(f"RNN training loss: {rnn_losses[0]:.4f} -> {rnn_losses[-1]:.4f}")
print(f"Held-out final-token accuracy: window {window_accuracy:.1%}; RNN {recurrent_accuracy:.1%}")
print(f"Largest final probability change after swapping only the clue: {clue_effect:.6f}")
for name, fitted_model in [("window", window_model), ("RNN", running_model)]:
    print(name, "parameters:", sum(p.numel() for p in fitted_model.parameters()))
if recurrent_accuracy > window_accuracy:
    print("Takeaway: this run used the clue beyond the window on the tiny held-out pair.")
else:
    print("Takeaway: the note has a route, but this run did not establish an accuracy gain.")

# %% clue_probe
clue_word = CAT_WORDS[1]  # CHANGE THIS: "dog"; keep every other word unchanged.
probe_words = list(CAT_WORDS)
probe_words[1] = clue_word
probe = encode([probe_words])
with torch.no_grad():
    final = running_model(probe)[0][0, -1].softmax(-1)
print("Probe:", " ".join(probe_words))
print("purring:", round(final[to_id["purring"]].item(), 4),
      "panting:", round(final[to_id["panting"]].item(), 4))

# %% state_trace
distances = (notes[0] - notes[1]).norm(dim=-1).numpy()
labels = [left if left == right else f"{left} / {right}"
          for left, right in zip(test_rows[0][:-1], test_rows[1][:-1])]
assert len(labels) == len(distances) == len(CAT_WORDS)
assert distances[0] == 0, "The common first word must produce the same fresh state."
positions = np.arange(len(distances))
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(positions, distances, marker="o", color="#ef8a23")
ax.set_xticks(positions, labels)
ax.set(xlabel="Input position (left to right)", ylabel="Distance between learned notes",
       title="Common the, changed animal, identical later words")
save_evidence(fig, "03b-cat-state-distances.png", "Measured cat/dog RNN state distances",
              "Static counterpart of the replay: common the, cat/dog intervention, sat on the mat.")
plt.show()
print("Recorded distances:", [round(float(distance), 4) for distance in distances])
print("Identical after common first word:", distances[0] == 0)
print("Distinct after animal and through later words:", bool((distances[1:] > 0).all()))
position_to_read = len(distances) - 1  # CHANGE THIS: 0 (common word), 1 (animal), or a later position
print(labels[position_to_read], float(distances[position_to_read]))
print("Takeaway: state differences alone are not proof of correct recall.")

# %% state_replay
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.set(xlim=(-0.2, len(distances) - 1 + 0.2), ylim=(0, max(float(distances.max()), 0.01) * 1.15),
       xlabel="Input position (left to right)", ylabel="Distance between notes",
       title="Replay the measured cat/dog note travel")
ax.set_xticks(positions, labels)
line, = ax.plot([], [], "o-", color="#ef8a23")
def frame(position):
    line.set_data(range(position + 1), distances[:position + 1])
    return (line,)
animation = FuncAnimation(fig, frame, frames=len(distances), interval=600, blit=True)
plt.close(fig)
replay_html = "\n".join(line.rstrip() for line in animation.to_jshtml().splitlines()) + "\n"
(EVIDENCE_DIR / "03b-cat-state-replay.html").write_text(replay_html, encoding="utf-8")
print(f"Watch {len(distances)} measured positions: equality at the first word, then the animal intervention.")
display(HTML(replay_html))

# %% lstm_model
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
        # [B,T] -> [B,T,D] -> [B,T,H] -> [B,T,V]; state is separate from time outputs.
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

# %% lstm_training
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
    lstm_accuracy = (lstm_logits[:, -1].argmax(-1) == answers).float().mean().item()
    _, (hidden, cell) = lstm_model.reader(lstm_model.embedding(test_ids[:, :-1]))
embedding_change = (lstm_model.embedding.weight.detach() - embedding_before).abs().max().item()
print(f"LSTM loss: {lstm_losses[0]:.4f} -> {lstm_losses[-1]:.4f}")
print(f"LSTM held-out final-token accuracy: {lstm_accuracy:.1%}")
print("Hidden / cell shapes:", tuple(hidden.shape), tuple(cell.shape))
print(f"Largest learned embedding change: {embedding_change:.6f}")
assert lstm_logits.shape == (len(test_ids), test_ids.shape[1] - 1, len(vocab))
assert lstm_losses[-1] < lstm_losses[0] and embedding_change > 0

# %% causality_and_generation
prefix = encode([CAT_WORDS])
assert torch.equal(prefix, test_ids[:1, :-1])
changed_prefix = prefix.clone()
changed_prefix[:, -1] = to_id["rug"]
with torch.inference_mode():
    original_scores = lstm_model(prefix)
    changed_scores = lstm_model(changed_prefix)
assert torch.allclose(original_scores[:, :-1], changed_scores[:, :-1])
future_token_gap = (original_scores[:, :-1] - changed_scores[:, :-1]).abs().max().item()
continuation = lstm_model.generate(prefix, steps=1)
paired_continuations = lstm_model.generate(test_ids[:, :-1], steps=1)
assert torch.equal(paired_continuations[:, :-1], test_ids[:, :-1])
assert torch.equal(paired_continuations[:, -1], lstm_logits[:, -1].argmax(-1))
print("Generated continuation:", [vocab[index] for index in continuation[0].tolist()])
print("Dog counterpart:", [vocab[index] for index in paired_continuations[1].tolist()])
print(f"Earlier-logit change after replacing future mat with rug: {future_token_gap:.8f}")
print("Takeaway: word rows learned with the LSTM; earlier logits do not read future tokens.")

# %% scorecard
scorecard = {
    "bigram_final_accuracy": bigram_final, "bag_gap": bag_gap,
    "ordered_score_gap": ordered_gap, "identical_window_probability_gap": blind_gap,
    "window_final_accuracy": window_accuracy, "RNN_final_accuracy": recurrent_accuracy,
    "swapped_clue_probability_effect": clue_effect, "LSTM_final_accuracy": lstm_accuracy,
    "embedding_max_change": embedding_change,
    "window_parameters": sum(p.numel() for p in window_model.parameters()),
    "RNN_parameters": sum(p.numel() for p in running_model.parameters()),
    "LSTM_parameters": sum(p.numel() for p in lstm_model.parameters()),
    "future_token_earlier_logit_gap": future_token_gap,
}
print(scorecard)
measured_evidence = {
    "semantic_anchor": "03B controlled cat/dog clue experiment",
    "source": "03B-counts-to-recurrent-language-models.ipynb",
    "canonical_probe": CAT_SENTENCE,
    "rule": {"cat": "purring", "dog": "panting"},
    "scope": "Made-up final-word rule; two held-out combinations, not general English.",
    "train_rows": train_rows, "held_out_rows": test_rows,
    "seed": 17, "cpu_threads": torch.get_num_threads(),
    "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    "budgets": {"window": {"updates": 300, "lr": 0.03},
                "RNN": {"updates": 350, "lr": 0.02, "gradient_clip_norm": 1.0},
                "LSTM": {"updates": 350, "lr": 0.03}},
    "counts_after_mat": dict(final_word_counts),
    "learning_curves": {"window": window_losses, "RNN": rnn_losses, "LSTM": lstm_losses},
    "state_trace": {"labels": labels, "distances": distances.tolist(),
                    "replay_frames": len(distances),
                    "static": "03b-cat-state-distances.png", "replay": "03b-cat-state-replay.html"},
    "RNN_final_probabilities": recurrent_probs.tolist(),
    "vocabulary": vocab, "scorecard": scorecard,
    "generated_rows": [[vocab[index] for index in row] for row in paired_continuations.tolist()],
    "plots": ["03b-cat-next-word-counts.png", "03b-cat-window-learning-curve.png",
              "03b-cat-state-distances.png"],
}
(EVIDENCE_DIR / "03b-cat-measured-evidence.json").write_text(
    json.dumps(measured_evidence, indent=2) + "\n", encoding="utf-8",
)
print(f"Elapsed CPU wall time: {time.perf_counter() - started:.1f}s")
