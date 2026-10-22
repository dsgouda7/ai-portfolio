"""Code companion for 03A-training-skeleton-and-word-objectives.ipynb. Run from this directory."""

# %% 1 - Shared setup: real stories and one familiar probe document
import math
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# A notebook kernel may start at the repository root; scripts already know their own path.
CHAPTER_DIRECTORY = (
    Path(__file__).resolve().parent if "__file__" in globals() else
    next(candidate for candidate in (
        Path.cwd(), Path.cwd() / "learning" / "03-language-model-foundations" / "03-language-models",
    ) if (candidate / "real_text_embeddings.py").is_file())
)
if str(CHAPTER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIRECTORY))
from real_text_embeddings import make_document_windows, read_story_sentences, tokenize

SEED = 17
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
notebook_started = time.perf_counter()

INTRO_STORIES = 1_000
CAT_PROBE = "the sleepy cat sat on the mat"
real_documents = read_story_sentences(INTRO_STORIES)
# One independent document at the END, not copies inserted among the stories.
documents = real_documents + [tokenize(CAT_PROBE)]
assert documents[-1] == tokenize(CAT_PROBE)
assert sum(document == tokenize(CAT_PROBE) for document in documents) == 1
print(f"Real source: first {INTRO_STORIES:,} pinned Hugging Face TinyStories stories")
print(f"Documents: {len(documents)}")
print("First tokenized document:", documents[0])
print("Last document (appended exactly once):", documents[-1])

# %% 2 - Shared setup: give each word a stable address
items = [word for document in documents for word in document]

counts = Counter(items)
itos = ["<unk>"] + [word for word, _ in counts.most_common(1_199)]
# Rare mat still needs an address to be probed. This adds no examples or weight.
itos += [word for word in dict.fromkeys(tokenize(CAT_PROBE)) if word not in itos]
stoi = {word: index for index, word in enumerate(itos)}
vocabulary_size = len(itos)

id_documents = [
    [stoi.get(word, stoi["<unk>"]) for word in document]
    for document in documents
]

print(f"Tokens counted: {len(items)}")
print(f"Vocabulary size: {vocabulary_size}")
print("Cat-probe word counts (not weights):",
      {word: counts[word] for word in dict.fromkeys(tokenize(CAT_PROBE))})
print("Most common words (word, count, unique ID):",
      [(word, count, stoi[word]) for word, count in counts.most_common(10)])
print("First encoded document:", id_documents[0])
assert len(stoi) == vocabulary_size
assert all(itos[index] == word for word, index in stoi.items())

# %% 3 - CBOW preview: slide a window to make practice questions
CONTEXT_RADIUS = 2
window_width = 2 * CONTEXT_RADIUS + 1
# Reuse this cutter across this notebook's models; each objective chooses its own inputs and answers.
windows = make_document_windows(id_documents, window_width)
assert len(windows) == sum(max(0, len(ids) - window_width + 1) for ids in id_documents)
assert np.array_equal(windows[-3:], make_document_windows([id_documents[-1]], window_width))

print(f"Window width: {window_width}")
print(f"All windows shape: {windows.shape}")
print(f"Unknown window slots: {(windows == 0).mean():.1%}")
print("Our cat document, sliding exactly as in the opening image:")
for row in windows[-3:]:
    print([itos[index] for index in row])

window_ids = torch.tensor(windows, dtype=torch.long)
cat_window_ids = window_ids[-3:]
assert [[itos[index] for index in row] for row in cat_window_ids.tolist()] == [
    ["the", "sleepy", "cat", "sat", "on"],
    ["sleepy", "cat", "sat", "on", "the"],
    ["cat", "sat", "on", "the", "mat"],
]
assert window_ids.shape == (len(windows), window_width)

# %% CBOW preview - three real windows, one static storyboard
window_story_words = tokenize(CAT_PROBE)
window_story_frames = [
    [itos[token_id] for token_id in row] for row in cat_window_ids.tolist()
]
assert window_story_frames == [
    window_story_words[start:start + window_width]
    for start in range(len(window_story_words) - window_width + 1)
]


def draw_cbow_window(axis, frame):
    axis.clear()
    axis.set(xlim=(-0.6, len(window_story_words) - 0.4), ylim=(-1.1, 2.0))
    axis.axis("off")
    axis.set_title(f"Window {frame + 1} of {len(window_story_frames)}", fontsize=11)
    for position, word in enumerate(window_story_words):
        active = frame <= position < frame + window_width
        axis.text(
            position, 1.3, word, ha="center", va="center", fontsize=10,
            color="#111827" if active else "#9ca3af",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="#94a3b8",
                      facecolor="#dbeafe" if active else "#f3f4f6"),
        )
    row = window_story_frames[frame]
    question = row[:CONTEXT_RADIUS] + ["___"] + row[CONTEXT_RADIUS + 1:]
    axis.text(3, 0.6, "Shared cut: " + " ".join(row),
              ha="center", fontsize=10, color="#1e3a8a")
    axis.text(3, -0.05, "CBOW clues: " + " ".join(question),
              ha="center", fontsize=10, color="#007f86")
    axis.text(3, -0.7, "Checked answer: " + row[CONTEXT_RADIUS],
              ha="center", fontsize=11, color="#bf5700", weight="bold")


window_story_figure, window_story_axes = plt.subplots(
    1, len(window_story_frames), figsize=(15, 3.2),
)
for window_frame, window_axis in enumerate(np.atleast_1d(window_story_axes)):
    draw_cbow_window(window_axis, window_frame)
window_story_figure.suptitle("Shared window, CBOW question: slide one word at a time")
window_story_figure.tight_layout(rect=(0, 0, 1, 0.9))
window_story_figure.savefig(
    CHAPTER_DIRECTORY / "images" / "03a-cbow-sliding-window-storyboard.png", dpi=130,
)
plt.close(window_story_figure)
print(f"Prediction check: {len(window_story_frames)} complete windows, not one per word.")
print("CBOW answers:", [row[CONTEXT_RADIUS] for row in window_story_frames])

# %% CBOW preview - animate those same three windows
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

print("Watch the blue window move; CBOW keeps each orange answer out of its input.")
window_animation_figure, window_animation_axis = plt.subplots(figsize=(9, 3.2))
window_animation = FuncAnimation(
    window_animation_figure,
    lambda frame: draw_cbow_window(window_animation_axis, frame),
    frames=len(window_story_frames), interval=1300, repeat=False,
)
plt.close(window_animation_figure)
display(HTML(window_animation.to_jshtml(default_mode="once")))

# %% 4 - Shared setup: batch each model's own questions
def make_dataloader(input_ids, target_ids, batch_size=512):
    dataset = TensorDataset(input_ids, target_ids)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )


# %% 5 - CBOW: fetch sensor profiles and give each clue an equal vote
toy_table = torch.tensor([[2.0, 0.0], [1.0, 3.0], [0.0, 4.0]])
toy_ids = torch.tensor([[0, 2], [2, 0]])
toy_rows = torch.nn.functional.embedding(toy_ids, toy_table)
# [2 questions, 2 words, 2 numbers] -> [2, 2]: combine words WITHIN each question.
toy_averages = toy_rows.mean(dim=1)

print("Two questions, two visible word IDs each:\n", toy_ids)
print("One two-number row for each visible word:\n", toy_rows)
print("One averaged row per question:\n", toy_averages)
assert toy_rows.shape == (2, 2, 2)
assert torch.equal(toy_averages, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))

toy_figure, toy_axes = plt.subplots(1, 3, figsize=(14, 5))
toy_panels = [
    (
        "1. One shared lookup table\n3 words, 2 numbers per word",
        ["Word (ID)", "Number 1", "Number 2"],
        [["the (0)", 2, 0], ["cat (1)", 1, 3], ["sat (2)", 0, 4]],
    ),
    (
        "2. Fetch rows for each question\n2 questions, 2 words each, 2 numbers each",
        ["Question / visible word", "Number 1", "Number 2"],
        [["A: the", 2, 0], ["A: sat", 0, 4],
         ["B: sat", 0, 4], ["B: the", 2, 0]],
    ),
    (
        "3. Average within EACH question\n2 questions, 2 numbers per question",
        ["Question", "Number 1", "Number 2"],
        [["A", *toy_averages[0].tolist()], ["B", *toy_averages[1].tolist()]],
    ),
]
for toy_axis, (title, columns, cells) in zip(toy_axes, toy_panels):
    toy_axis.axis("off")
    toy_axis.set_title(title, fontsize=10, pad=18)
    table = toy_axis.table(cellText=cells, colLabels=columns, loc="center",
                           cellLoc="center", colWidths=[0.56, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("#dff2ef" if row else "#b9ddd9")
toy_figure.suptitle("A: the ___ sat     |     B: sat ___ the\n"
                   "Different order goes in. The same average comes out.", fontsize=13)
toy_figure.text(0.5, 0.05, "Made-up teaching numbers. "
                "No averaging between questions. No training in this picture.", ha="center")
toy_figure.tight_layout(rect=(0, 0.1, 1, 0.85))
toy_directory = CHAPTER_DIRECTORY
toy_figure.savefig(toy_directory / "images" / "03a-lookup-tensor-structure.png", dpi=130)
plt.show()
plt.close(toy_figure)

# %% CBOW: give every context word an equal vote
EMBEDDING_WIDTH = 32
probe_embedding = nn.Embedding(vocabulary_size, EMBEDDING_WIDTH)

# [N, 5] -> [3, 4]: inspect three rows, removing the hidden center column.
preview_contexts = window_ids[:3, [0, 1, 3, 4]]
# [3, 4] -> [1, 4]: retain the batch axis even when inspecting one question.
one_context = preview_contexts[:1]
# [1, 4] -> [1, 4]: reverse positions, not questions; mean should erase this change.
reversed_context = one_context.flip(dims=[1])
one_vectors = probe_embedding(one_context)
# [1, 4, 32] -> [1, 32]: pool only the four context rows.
original_mean = one_vectors.mean(dim=1)
reversed_mean = probe_embedding(reversed_context).mean(dim=1)

print("IDs:", one_context.tolist())
print("After lookup:", tuple(one_vectors.shape))
print("After context mean:", tuple(original_mean.shape))
print("Maximum difference after reversal:", (original_mean - reversed_mean).abs().max().item())

assert torch.allclose(original_mean, reversed_mean)

# %% Shared lookup: why sensor profiles, and not a giant one-hot checklist?
scalar_cat_id = torch.tensor(1, dtype=torch.long)
one_cat_batch = scalar_cat_id.unsqueeze(0)
scalar_cat_row = nn.functional.embedding(scalar_cat_id, toy_table)
lookup_cat_row = nn.functional.embedding(one_cat_batch, toy_table)
one_hot_cat = nn.functional.one_hot(one_cat_batch, num_classes=3).to(toy_table.dtype)
checklist_cat_row = one_hot_cat @ toy_table
assert scalar_cat_row.shape == (2,)
assert lookup_cat_row.shape == checklist_cat_row.shape == (1, 2)
assert torch.equal(lookup_cat_row, torch.tensor([[1., 3.]]))
assert torch.equal(lookup_cat_row, checklist_cat_row)
print("Stored table:", tuple(toy_table.shape), "= [V, D] weights")
print("Scalar ID -> row:", tuple(scalar_cat_id.shape), "->", tuple(scalar_cat_row.shape))
print("Actual ID batch:", one_cat_batch.tolist(), tuple(one_cat_batch.shape))
print("Actual lookup:  ", lookup_cat_row.tolist(), tuple(lookup_cat_row.shape))
print("One-hot check:  ", one_hot_cat.tolist(), tuple(one_hot_cat.shape))
print("Same result:    ", checklist_cat_row.tolist(), tuple(checklist_cat_row.shape))

checklists = torch.eye(vocabulary_size)
random_rows = probe_embedding.weight.detach()


def checklist_gap(word_a, word_b):
    return (checklists[stoi[word_a]] - checklists[stoi[word_b]]).norm().item()


def row_similarity(rows, word_a, word_b):
    return torch.cosine_similarity(rows[stoi[word_a]], rows[stoi[word_b]], dim=0).item()


print(f"\nOne checklist: {vocabulary_size} boxes. One sensor profile: {EMBEDDING_WIDTH} channels.")
for word_a, word_b in [("cat", "dog"), ("cat", "mat")]:
    print(f"{word_a} vs {word_b:<10}  checklist gap {checklist_gap(word_a, word_b):.3f}   "
          f"random-row similarity {row_similarity(random_rows, word_a, word_b):+.3f}")

assert checklist_gap("cat", "dog") == checklist_gap("cat", "mat")

# %% CBOW: the same mean with EmbeddingBag
embedding_bag = nn.EmbeddingBag(
    vocabulary_size,
    EMBEDDING_WIDTH,
    mode="mean",
)
with torch.no_grad():
    embedding_bag.weight.copy_(probe_embedding.weight)

# [3, 4, 32] -> [3, 32]: average each question's clues; EmbeddingBag fuses this same job.
manual_means = probe_embedding(preview_contexts).mean(dim=1)
bag_means = embedding_bag(preview_contexts)

print("Manual mean shape:", tuple(manual_means.shape))
print("EmbeddingBag shape:", tuple(bag_means.shape))
print("Maximum difference:", (manual_means - bag_means).abs().max().item())
assert torch.allclose(manual_means, bag_means)

# %% 6 - Give replaceable models one small training contract
# %% 6A - CBOW data: neighbours -> middle word
# Two [N, 2] slices -> [N, 4]: join left/right clues without the center answer.
cbow_contexts = torch.cat(
    [window_ids[:, :CONTEXT_RADIUS], window_ids[:, CONTEXT_RADIUS + 1:]],
    dim=1,
)
# [N, 5] -> [N]: one center answer per row, in the same row order.
cbow_targets = window_ids[:, CONTEXT_RADIUS]
cbow_dataloader = make_dataloader(cbow_contexts, cbow_targets)
cbow_batch_contexts, cbow_batch_targets = next(iter(cbow_dataloader))

print("CBOW: neighbours -> middle word")
print("Questions:", len(cbow_contexts))
print("Batch inputs:", tuple(cbow_batch_contexts.shape))
print("Batch answers:", tuple(cbow_batch_targets.shape))
print(
    "Illustrated cat-document pair:",
    [itos[index] for index in cbow_contexts[-3].tolist()],
    "->", itos[cbow_targets[-3]],
)

assert cbow_contexts.shape == (len(window_ids), 2 * CONTEXT_RADIUS)
assert cbow_targets.shape == (len(window_ids),)
assert torch.equal(cbow_targets, window_ids[:, CONTEXT_RADIUS])
assert len(cbow_batch_contexts) == len(cbow_batch_targets)

# %% 6B - CBOW model
from language_model import TrainableLanguageModel


class CBOW(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width):
        super().__init__()
        # One learned sensor profile per word; rows store the profiles. [V, D]
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        # A small "thinking" layer that squeezes the row down to D/2 features.
        self.hidden = nn.Linear(embedding_width, embedding_width // 2)
        # ReLU zeroes negatives, so the two layers can't collapse into one straight line.
        self.activation = nn.ReLU()
        # One score per vocabulary word: "how likely is this the hidden word?"
        self.output = nn.Linear(embedding_width // 2, vocabulary_size)

    def forward(self, context_ids, target_ids=None):
        # target_ids is unused here; it's in the signature so the shared
        # training loop can call every model the same way.
        # B is the current batch (at most 512), D=32 here; V is printed above.

        # context_ids: the four clue words around each gap, as IDs.   [B, 4]
        # e.g. "the sleepy ___ sat on" -> [the, sleepy, sat, on]
        # Retrieve the stored profile for each clue word.             [B, 4, D]
        token_vectors = self.embedding(context_ids)

        # Average the four rows into one "gist of the clues" row.     [B, D]
        # This is the "bag" in bag of words: word order is lost here.
        pooled_context = token_vectors.mean(dim=1)

        # Projection [B, D] -> [B, D/2]: mix the gist into fewer learned features.
        hidden_features = self.activation(self.hidden(pooled_context))

        # Projection [B, D/2] -> [B, V]: one score per possible missing word.
        # These are raw scores (logits); cross-entropy turns them into chances.
        return self.output(hidden_features)

# %% CBOW: trace one untrained batch before optimizing anything
torch.manual_seed(SEED)
model = CBOW(vocabulary_size, EMBEDDING_WIDTH)
untrained_logits = model(cbow_batch_contexts)


def word_probe(logits, expected_word, words=itos):
    """Keep probability, rank, and the actual shortlist; never force the answer."""
    # [V] -> [V]: softmax changes scores to shares without changing the word axis.
    chances = logits.detach().softmax(dim=-1)
    expected_id = words.index(expected_word)
    rank = int((chances > chances[expected_id]).sum()) + 1
    top = chances.topk(5)
    return {
        "probability": chances[expected_id].item(),
        "rank": rank,
        "top5": [(words[index], round(chance, 4))
                 for index, chance in zip(top.indices.tolist(), top.values.tolist())],
    }


def report_probe(label, before, after):
    print(f"{label}: correct-word share {before['probability']:.3%} -> "
          f"{after['probability']:.3%}; rank {before['rank']} -> {after['rank']}")
    print("  Before top 5:", before["top5"])
    print("  After top 5: ", after["top5"])
    probability_change = "increased" if after["probability"] > before["probability"] else "decreased"
    rank_change = "improved" if after["rank"] < before["rank"] else (
        "worsened" if after["rank"] > before["rank"] else "stayed the same")
    print(f"  Probability {probability_change}; rank {rank_change}. "
          + ("The expected word is top-ranked." if after["rank"] == 1 else "Not the first guess."))


# [3, 5] -> [3, 4]: the same missing-center job, now on the familiar cat document.
cat_contexts = cat_window_ids[:, [0, 1, 3, 4]]
with torch.inference_mode():
    cat_before_logits = model(cat_contexts)
    cbow_before = {word: word_probe(cat_before_logits[row], word)
                   for row, word in enumerate(("cat", "sat", "on"))}

print("Input cbow_contexts:", tuple(cbow_batch_contexts.shape))
print("Vocabulary logits:", tuple(untrained_logits.shape))
assert untrained_logits.shape == (len(cbow_batch_contexts), vocabulary_size)

# %% 7 - CBOW: train and check with its own loader
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
cbow_started = time.perf_counter()
losses = model.fit_language_model(
    cbow_dataloader,
    criterion,
    optimizer,
    epochs=10,
)
cbow_seconds = time.perf_counter() - cbow_started

print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"CBOW: {cbow_seconds:.1f}s on one CPU thread (10 epochs)")
with torch.inference_mode():
    cat_after_logits = model(cat_contexts)
    for row, word in enumerate(("cat", "sat", "on")):
        report_probe(f"CBOW missing {word!r}", cbow_before[word],
                     word_probe(cat_after_logits[row], word))

# %% CBOW: a falling loss is necessary evidence, not the whole story
random_guess_loss = math.log(vocabulary_size)

fig, axis = plt.subplots(figsize=(8, 4))
axis.plot(losses, color="#2a6f97", linewidth=2)
axis.axhline(
    random_guess_loss,
    color="#d97706",
    linestyle="--",
    label=f"random guess log(V) = {random_guess_loss:.2f}",
)
axis.set(title="CBOW training loss", xlabel="epoch", ylabel="cross-entropy")
axis.legend(frameon=False)
axis.grid(alpha=0.2)
plt.tight_layout()
plt.show()
plt.close(fig)

print(f"Random-guess baseline: {random_guess_loss:.4f}")
print(f"Final training loss: {losses[-1]:.4f}")
assert losses[-1] < losses[0]
assert losses[-1] < random_guess_loss

# %% 8 - CBOW: follow one gradient back to the participating rows
gradient_model = deepcopy(model)
gradient_contexts = cat_contexts[:1].clone()
gradient_targets = torch.tensor([stoi["cat"]], dtype=torch.long)
saved_context_ids = gradient_contexts.clone()
saved_target_ids = gradient_targets.clone()
rows_before_step = gradient_model.embedding.weight.detach().clone()
output_before_step = gradient_model.output.weight.detach().clone()
gradient_optimizer = torch.optim.SGD(gradient_model.parameters(), lr=0.1)
gradient_optimizer.zero_grad()
gradient_loss = gradient_model.training_loss(gradient_contexts, gradient_targets, criterion)
gradient_loss.backward()

gradient_norms = gradient_model.embedding.weight.grad.norm(dim=1)
used_ids = set(gradient_contexts.reshape(-1).tolist())
rows_with_gradient = set(gradient_norms.nonzero().flatten().tolist())
gradient_optimizer.step()
changed_rows = (gradient_model.embedding.weight.detach() != rows_before_step).any(dim=1)
changed_ids = set(changed_rows.nonzero().flatten().tolist())

assert rows_with_gradient == changed_ids == used_ids
assert stoi["cat"] not in changed_ids
assert torch.equal(gradient_contexts, saved_context_ids)
assert torch.equal(gradient_targets, saved_target_ids)
assert not gradient_contexts.requires_grad and gradient_contexts.grad is None
assert not torch.equal(gradient_model.output.weight.detach(), output_before_step)
assert torch.equal(
    gradient_model.embedding(gradient_contexts).detach(),
    gradient_model.embedding.weight.detach()[saved_context_ids],
)
print("Unchanged input IDs:", gradient_contexts.tolist(), "answer ID:", gradient_targets.tolist())
print("Fetched rows / changed rows:",
      [itos[index] for index in sorted(used_ids)], [itos[index] for index in sorted(changed_ids)])
first_id = gradient_contexts[0, 0].item()
print(f"{itos[first_id]!r} row, first three values before:", rows_before_step[first_id, :3].tolist())
print("Same ID, values fetched after:", gradient_model.embedding(gradient_contexts)[0, 0, :3].detach().tolist())
print("Takeaway: stored embedding weights and predictor weights changed; no input or answer ID did.")

# %% 9 - CBOW: inspect the learned embedding rows
embedding_weights = model.embedding.weight.detach().cpu()
# [V, 32] -> [V, 1]: keep a singleton feature axis so division normalizes EACH word row.
embedding_norms = embedding_weights.norm(dim=1, keepdim=True).clamp_min(1e-12)
normalized_embeddings = embedding_weights / embedding_norms


def most_similar(word, top_k=4):
    if word not in stoi:
        return f"'{word}' is not in the vocabulary"

    similarities = normalized_embeddings @ normalized_embeddings[stoi[word]]
    similarities = similarities.clone()
    similarities[stoi[word]] = -1
    similarities[stoi["<unk>"]] = -1
    neighbour_ids = similarities.topk(min(top_k, vocabulary_size - 2)).indices
    return [
        (itos[index], round(similarities[index].item(), 3))
        for index in neighbour_ids.tolist()
    ]


for word in ["cat", "dog", "mat", "sat"]:
    print(f"{word:>12} -> {most_similar(word)}")

# %% CBOW: a 2D projection is a map, not the territory
# [V, 32] -> [visible words, 32]: keep the running example beside common words.
visible_words = list(dict.fromkeys(tokenize(CAT_PROBE) + [w for w, _ in counts.most_common(10)]))
visible_ids = torch.tensor([stoi[word] for word in visible_words])
visible_vectors = embedding_weights[visible_ids]
# [words, 32] -> [1, 32]: one center per feature, broadcast back across words.
centered = visible_vectors - visible_vectors.mean(dim=0, keepdim=True)
_, _, right_singular_vectors = torch.linalg.svd(centered, full_matrices=False)
# [2, 32] -> [32, 2]: turn two PCA directions into columns; project [words, 32] -> [words, 2].
coordinates = centered @ right_singular_vectors[:2].T

fig, axis = plt.subplots(figsize=(10, 7))
axis.scatter(coordinates[:, 0], coordinates[:, 1], color="#2a6f97", s=55)
for point, token_id in zip(coordinates, visible_ids.tolist()):
    axis.annotate(itos[token_id], point, xytext=(5, 5), textcoords="offset points")
axis.axhline(0, color="#9ca3af", linewidth=0.7)
axis.axvline(0, color="#9ca3af", linewidth=0.7)
axis.set(
    title="Cat-probe and frequent embedding rows projected from 32D to 2D",
    xlabel="projection axis 1",
    ylabel="projection axis 2",
)
axis.grid(alpha=0.15)
plt.tight_layout()
plt.show()
plt.close(fig)

# %% 10A - Skip-gram data: middle word -> one neighbour
# Each raw window supplies one center and all its neighbours.
skipgram_centers = window_ids[:, CONTEXT_RADIUS]
# [N, 2] plus [N, 2] -> [N, 4]: keep each window's neighbours in left-to-right order.
skipgram_neighbours = torch.cat(
    [window_ids[:, :CONTEXT_RADIUS], window_ids[:, CONTEXT_RADIUS + 1:]],
    dim=1,
)
neighbours_per_window = skipgram_neighbours.shape[1]
# [N] -> [4N]: repeat each center four times BEFORE moving to the next window.
skipgram_inputs = skipgram_centers.repeat_interleave(neighbours_per_window)
# [N, 4] -> [4N]: row-major flattening keeps each answer aligned with its repeated center.
skipgram_targets = skipgram_neighbours.reshape(-1)
skipgram_dataloader = make_dataloader(skipgram_inputs, skipgram_targets)
skipgram_batch_inputs, skipgram_batch_targets = next(iter(skipgram_dataloader))

print("Skip-gram: middle word -> ONE neighbour per question")
print("Questions:", len(skipgram_inputs))
print("Batch inputs:", tuple(skipgram_batch_inputs.shape))
print("Batch answers:", tuple(skipgram_batch_targets.shape))
print("Pairs from the illustrated cat window:")
for center, neighbour in zip(
    skipgram_inputs[-3 * neighbours_per_window:-2 * neighbours_per_window],
    skipgram_targets[-3 * neighbours_per_window:-2 * neighbours_per_window],
):
    print("   ", itos[center], "->", itos[neighbour])

assert skipgram_inputs.shape == (len(window_ids) * neighbours_per_window,)
assert skipgram_targets.shape == skipgram_inputs.shape
assert torch.equal(
    # [4N] -> [N, 4]: reconstruct groups to check alignment, not mix examples.
    skipgram_inputs.reshape(len(window_ids), neighbours_per_window),
    # [N] -> [N, 1] -> [N, 4]: broadcast a center beside each of its own neighbours.
    skipgram_centers[:, None].expand_as(skipgram_neighbours),
)
assert torch.equal(
    # [4N] -> [N, 4]: the flattened answers must reconstruct the original neighbour rows.
    skipgram_targets.reshape_as(skipgram_neighbours), skipgram_neighbours,
)
assert len(skipgram_batch_inputs) == len(skipgram_batch_targets)

# %% Skip-gram: define, train, and check neighbour guesses
class SkipGram(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        self.hidden = nn.Linear(embedding_width, embedding_width // 2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(embedding_width // 2, vocabulary_size)

    # Same layers as CBOW. The only change is what goes in: one word, not four.
    def forward(self, center_ids, target_ids=None):
        # center_ids: one word per lesson, e.g. cat.                  [B]
        # Look up its row. Only one row, so there's nothing to average. [B, D]
        center_vectors = self.embedding(center_ids)

        # Projection [B, D] -> [B, D/2]: mix this one center's row into features.
        hidden_features = self.activation(self.hidden(center_vectors))

        # Projection [B, D/2] -> [B, V]: score all words as neighbours of this center.
        # The same center appears in several lessons with different answers
        # (cat -> the, sleepy, sat, on), so it can never be sure of just one.
        return self.output(hidden_features)


torch.manual_seed(SEED)
skipgram_model = SkipGram(vocabulary_size, EMBEDDING_WIDTH)
with torch.inference_mode():
    # [1] -> [1, V] -> [V]: one cat-center distribution for the neighbour job.
    skipgram_before_logits = skipgram_model(torch.tensor([stoi["cat"]]))[0]
    skipgram_before = {word: word_probe(skipgram_before_logits, word)
                       for word in ("the", "sleepy", "sat", "on")}
skipgram_optimizer = torch.optim.Adam(skipgram_model.parameters(), lr=0.008)
skipgram_started = time.perf_counter()
skipgram_losses = skipgram_model.fit_language_model(
    skipgram_dataloader,
    criterion,
    skipgram_optimizer,
    epochs=4,
)
skipgram_seconds = time.perf_counter() - skipgram_started

with torch.inference_mode():
    probe_center = skipgram_centers[:1]
    neighbour_logits = skipgram_model(probe_center)
    predicted_ids = neighbour_logits.topk(neighbours_per_window, dim=1).indices[0]

print(f"Skip-gram loss: {skipgram_losses[0]:.4f} -> {skipgram_losses[-1]:.4f}; "
      f"{skipgram_seconds:.1f}s (4 epochs)")
print("Center:", itos[probe_center.item()])
print("Top predicted neighbours:", [itos[index] for index in predicted_ids.tolist()])
with torch.inference_mode():
    cat_neighbour_logits = skipgram_model(torch.tensor([stoi["cat"]]))[0]
    for word, before in skipgram_before.items():
        report_probe(f"Skip-gram cat -> neighbour {word!r}", before,
                     word_probe(cat_neighbour_logits, word))

assert skipgram_losses[-1] < skipgram_losses[0]
assert skipgram_losses[-1] < math.log(vocabulary_size)
assert neighbour_logits.shape == (1, vocabulary_size)

# %% 11 - CBOW: verify that averaging still loses word order
model.eval()
ordered_context = cbow_contexts[:1]
# [1, 4] -> [1, 4]: reorder only clue positions; their mean must stay the same.
permuted_context = ordered_context[:, torch.tensor([3, 1, 0, 2])]

with torch.no_grad():
    # Both [1, 4, 32] lookups -> [1, 32] means: only word order changed, not membership.
    ordered_vectors = model.embedding(ordered_context).mean(dim=1)
    permuted_vectors = model.embedding(permuted_context).mean(dim=1)
    ordered_logits = model(ordered_context)
    permuted_logits = model(permuted_context)

representation_difference = (ordered_vectors - permuted_vectors).abs().max().item()
logit_difference = (ordered_logits - permuted_logits).abs().max().item()

print("Ordered:", [itos[index] for index in ordered_context[0].tolist()])
print("Permuted:", [itos[index] for index in permuted_context[0].tolist()])
print(f"Maximum representation difference: {representation_difference:.8f}")
print(f"Maximum logit difference: {logit_difference:.8f}")

assert torch.allclose(ordered_vectors, permuted_vectors)
assert torch.allclose(ordered_logits, permuted_logits)

# %% 12A - Encoder-decoder: ordered prefix -> continuation
BOS_ID = vocabulary_size
EOS_ID = vocabulary_size + 1
sequence_itos = itos + ["<bos>", "<eos>"]
sequence_vocabulary_size = len(sequence_itos)

# [N, 5] -> [N, 3] source and [N, 2] continuation: split WITHIN each window.
source_sequences = window_ids[:, :3]
continuations = window_ids[:, 3:]
# [N, 1] + [N, 2] + [N, 1] -> [N, 4]: bracket each answer, never join documents.
sequence_targets = torch.cat(
    [
        torch.full((len(windows), 1), BOS_ID, dtype=torch.long),
        continuations,
        torch.full((len(windows), 1), EOS_ID, dtype=torch.long),
    ],
    dim=1,
)
sequence_dataloader = make_dataloader(source_sequences, sequence_targets)

print("Source shape:", tuple(source_sequences.shape))
print("Target shape:", tuple(sequence_targets.shape))
print("Source:", [sequence_itos[index] for index in source_sequences[-3].tolist()])
print("Target:", [sequence_itos[index] for index in sequence_targets[-3].tolist()])

assert source_sequences.shape == (len(windows), 3)
assert sequence_targets.shape == (len(windows), 4)

sequence_batch_inputs, sequence_batch_targets = next(iter(sequence_dataloader))
print("Sequence batch sources:", tuple(sequence_batch_inputs.shape))
print("Sequence batch full answers:", tuple(sequence_batch_targets.shape))
print("Writer sees:", [sequence_itos[index] for index in sequence_targets[-3, :-1].tolist()])
print("Loss checks:", [sequence_itos[index] for index in sequence_targets[-3, 1:].tolist()])
assert len(sequence_batch_inputs) == len(sequence_batch_targets)
assert torch.equal(source_sequences, window_ids[:, :3])
assert torch.equal(sequence_targets[:, 1:-1], window_ids[:, 3:])
assert bool((sequence_targets[:, 0] == BOS_ID).all())
assert bool((sequence_targets[:, -1] == EOS_ID).all())

# %% Encoder-decoder: define, train, and check the reader and writer
# An ordered observing log becomes one briefing; the writer uses it to predict a report.
# Here that report is a token sequence, not physical measurements or a new registry entry.
class EncoderDecoderLM(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width, hidden_width):
        super().__init__()
        # Reader and writer share one registry of input profiles.     [V, D]
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        # The reader: walks the source left to right, updating a running note.
        self.encoder = nn.GRU(embedding_width, hidden_width, batch_first=True)
        # The writer: starts from the reader's final note and writes word by word.
        self.decoder = nn.GRU(embedding_width, hidden_width, batch_first=True)
        # Turns the writer's note at each step into a score for every word.
        self.output = nn.Linear(hidden_width, vocabulary_size)

    def encode(self, source_ids):
        # source_ids: e.g. [the, sleepy, cat]                          [B, 3]
        # Look up rows, read them in order. We keep only the final note
        # (the "summary card"), not the note after each word.          [1, B, H]
        # Lookup expands [B, 3] -> [B, 3, D]; GRU keeps its final state [1, B, H].
        _, hidden = self.encoder(self.embedding(source_ids))
        return hidden

    def decode(self, decoder_input_ids, hidden):
        # decoder_input_ids: the words the writer is shown, e.g. [<bos>, sat, on]
        # Starting from the reader's note, write one note per step.    [B, T, H]
        # Lookup [B, T] -> [B, T, D]; recurrence produces [B, T, H] states in order.
        decoder_states, hidden = self.decoder(
            self.embedding(decoder_input_ids),
            hidden,
        )
        # Projection changes [B, T, H] -> [B, T, V], retaining batch and time.
        # Score every word at every step: "what comes next?"
        return self.output(decoder_states), hidden

    def forward(self, source_ids, target_ids=None):
        if target_ids is None:
            raise ValueError("Training requires target IDs for teacher forcing.")
        # 1. Read the source into one summary note.
        hidden = self.encode(source_ids)
        # 2. Teacher forcing: show the writer the CORRECT previous words
        #    (<bos>, sat, on), dropping the last one, which it never needs to see.
        #    targets_for_loss() below supplies the matching answers (sat, on, <eos>).
        # [B, 4] -> [B, 3]: drop the last answer; keep <bos> and previous words.
        logits, _ = self.decode(target_ids[:, :-1], hidden)
        return logits

    def targets_for_loss(self, target_ids):
        # forward() fed the writer target_ids[:, :-1]  -> <bos>, sat, on
        # so its answers are shifted one step left    -> sat,   on,  <eos>
        # Each input word is marked against the word that should come next.
        # [B, 4] -> [B, 3]: drop <bos> to align each score with the NEXT token.
        return target_ids[:, 1:]


torch.manual_seed(SEED)
sequence_model = EncoderDecoderLM(
    sequence_vocabulary_size,
    embedding_width=EMBEDDING_WIDTH,
    hidden_width=48,
)
# [3, 5] -> two [3, 3/4] probe tensors using the same preparation as the full dataset.
cat_sources = source_sequences[-3:]
cat_answers = sequence_targets[-3:]
with torch.inference_mode():
    writer_before_logits = sequence_model(cat_sources, cat_answers)
    writer_before = {
        "sat": word_probe(writer_before_logits[0, 0], "sat", sequence_itos),
        "mat": word_probe(writer_before_logits[2, 1], "mat", sequence_itos),
    }
sequence_optimizer = torch.optim.Adam(sequence_model.parameters(), lr=0.008)
assert EncoderDecoderLM.fit_language_model is TrainableLanguageModel.fit_language_model
sequence_started = time.perf_counter()
sequence_losses = sequence_model.fit_language_model(
    sequence_dataloader,
    criterion,
    sequence_optimizer,
    epochs=6,
)
sequence_seconds = time.perf_counter() - sequence_started

with torch.inference_mode():
    # [N, ...] -> first at most 4096 rows: bounded diagnostic avoids a huge [N, 3, V] allocation.
    teacher_forced_logits = sequence_model(source_sequences[:4096], sequence_targets[:4096])
    teacher_forced_predictions = teacher_forced_logits.argmax(dim=-1)
    expected_sequence_ids = sequence_targets[:4096, 1:]
    teacher_forced_accuracy = (
        teacher_forced_predictions == expected_sequence_ids
    ).float().mean().item()
    # [probe, 3] -> [probe, 2]: exclude the easy final stop target when checking content.
    content_accuracy = (
        teacher_forced_predictions[:, :-1] == expected_sequence_ids[:, :-1]
    ).float().mean().item()

print(f"Seq2Seq loss: {sequence_losses[0]:.4f} -> {sequence_losses[-1]:.4f}")
print(f"Encoder-decoder: {sequence_seconds:.1f}s (6 epochs)")
print("Teacher-forced accuracy on the first 4096 training windows:", f"{teacher_forced_accuracy:.1%}")
print("Without the predictable stop target:", f"{content_accuracy:.1%}")
print("Sequence logits:", tuple(teacher_forced_logits.shape))
with torch.inference_mode():
    writer_after_logits = sequence_model(cat_sources, cat_answers)
    report_probe("Writer next 'sat' after source 'the sleepy cat' + <bos>",
                 writer_before["sat"], word_probe(writer_after_logits[0, 0], "sat", sequence_itos))
    report_probe("Writer next 'mat' after source 'cat sat on' + correct previous 'the'",
                 writer_before["mat"], word_probe(writer_after_logits[2, 1], "mat", sequence_itos))

assert sequence_losses[-1] < sequence_losses[0]
assert sequence_losses[-1] < math.log(sequence_vocabulary_size)
assert teacher_forced_logits.shape == (min(4096, len(windows)), 3, sequence_vocabulary_size)
print(f"Introductory training total: {cbow_seconds + skipgram_seconds + sequence_seconds:.1f}s")

# %% Encoder-decoder: can the writer use its own answer?
def greedy_decode(model, source_ids, bos_id, eos_id, max_steps):
    model.eval()
    current_ids = torch.full(
        (len(source_ids), 1),
        bos_id,
        dtype=torch.long,
    )
    generated = []

    with torch.inference_mode():
        hidden = model.encode(source_ids)
        for _ in range(max_steps):
            logits, hidden = model.decode(current_ids, hidden)
            # [B, 1, V] -> [B, V] -> [B]: choose only the latest step's best word.
            next_ids = logits[:, -1].argmax(dim=-1)
            generated.append(next_ids)
            # [B] -> [B, 1]: a new one-token time axis lets the GRU read its own guess.
            current_ids = next_ids.unsqueeze(1)
            if bool((next_ids == eos_id).all()):
                break

    # List of T [B] columns -> [B, T]: keep generated words in time order.
    return torch.stack(generated, dim=1)


generated_ids = greedy_decode(
    sequence_model,
    cat_sources[:1],
    BOS_ID,
    EOS_ID,
    max_steps=sequence_targets.shape[1] - 1,
)

with torch.inference_mode():
    ordered_hidden = sequence_model.encode(cat_sources[:1])
    reversed_hidden = sequence_model.encode(cat_sources[:1].flip(dims=[1]))

expected_words = [
    sequence_itos[index]
    for index in cat_answers[0, 1:].tolist()
]
generated_words = [
    sequence_itos[index]
    for index in generated_ids[0].tolist()
]
hidden_difference = (ordered_hidden - reversed_hidden).abs().max().item()

print("Expected continuation:", expected_words)
print("Greedy continuation:", generated_words)
print(f"Hidden-state difference after reversal: {hidden_difference:.6f}")

assert generated_ids.shape[0] == 1
assert generated_ids.shape[1] >= 1
assert hidden_difference > 0

# %% 13 - Comparison: CBOW, Skip-gram, and encoder-decoder on new stories
from pathlib import Path

import real_text_embeddings
from real_text_embeddings import (
    animate_word_maps,
    blank_shortlist,
    load_real_text,
    plot_losses_and_geometry,
    plot_similarity_heatmaps,
    plot_word_maps,
    project_snapshots,
    report_neighbours,
    shortlist_summary,
    train_comparison,
)
from real_text_embeddings import best_versus_final

real_corpus = load_real_text()

# %% Comparison: keep the starting rows identical; change the prediction job
REAL_TEXT_EPOCHS = 16
real_results = train_comparison(
    real_corpus, CBOW, SkipGram, EncoderDecoderLM, epochs=REAL_TEXT_EPOCHS,
)

# %% Comparison: falling practice loss can hide worse guesses on new text
chart_directory = Path(real_text_embeddings.__file__).resolve().parent / "images"
loss_figure = plot_losses_and_geometry(real_results)
loss_figure.savefig(chart_directory / "03a-real-text-loss-and-similarity.png", dpi=130)
plt.show()
plt.close(loss_figure)

# %% Comparison: one fixed camera for all three moving word maps
real_probe_words, real_coordinates, real_pca_fraction = project_snapshots(
    real_corpus, real_results,
)
map_figure = plot_word_maps(real_probe_words, real_coordinates, real_results)
map_figure.savefig(chart_directory / "03a-real-text-word-maps.png", dpi=130)
plt.show()
plt.close(map_figure)

# %% Comparison: check the full rows, not just their two-dimensional shadows
similarity_figure = plot_similarity_heatmaps(real_corpus, real_results)
similarity_figure.savefig(chart_directory / "03a-real-text-cosine-heatmaps.png", dpi=130)
plt.show()
plt.close(similarity_figure)

# %% Comparison: play the recorded epochs, not invented training states
real_animation = animate_word_maps(real_probe_words, real_coordinates, real_results)
real_animation_html = real_animation.to_jshtml()
if "get_ipython" in globals():
    from IPython.display import HTML, display

    display(HTML(real_animation_html))
else:
    print(f"Recorded animation rendered ({len(real_animation_html):,} HTML characters).")

# %% Comparison: useful neighbours, or merely nearby dots?
report_neighbours(real_corpus, real_results)

# %% Comparison: change only the pair, without training again
probe_pair = ("mom", "dad")  # CHANGE THIS: try ("mom", "green"), without retraining
for objective, record in real_results.items():
    pair_ids = [real_corpus["word_ids"][word] for word in probe_pair]
    pair_scores = []
    for rows in (record["snapshots"][0], record["snapshots"][-1]):
        pair_vectors = torch.nn.functional.normalize(rows[pair_ids], dim=1)
        pair_scores.append((pair_vectors[0] @ pair_vectors[1]).item())
    print(f"{objective}: {' / '.join(probe_pair)} cosine "
          f"{pair_scores[0]:+.3f} -> {pair_scores[1]:+.3f}; "
          f"final across-group mean {record['across'][-1]:+.3f}")

# %% CBOW: inspect a shortlist at its best held-out epoch
cbow_best = real_results["CBOW"]["best_model"]
for left, right in [("was so", "to have"), ("a little", "named tom"),
                    ("played with", "toys and"), ("they went", "the park")]:
    menu = blank_shortlist(cbow_best, real_corpus, left, right)
    print(f"{left} ___ {right} -> " + ", ".join(f"{word} {chance:.0%}" for word, chance in menu))

shortlist = shortlist_summary(cbow_best, real_corpus)
print(f"\nAll {shortlist['blanks']:,} held-out blanks with a known answer:")
print(f"  right answer is the number-one guess:   {shortlist['first_right']:.1%}")
print(f"  right answer is somewhere in the top 5: {shortlist['shortlist_right']:.1%}")
print(f"  average share given to the first guess: {shortlist['first_share']:.1%}")
print(f"  how alike the 5 shortlisted words' rows are: {shortlist['shortlist_similarity']:+.3f}")
print(f"  how alike 5 random words' rows are:          {shortlist['random_similarity']:+.3f}")

assert shortlist["shortlist_right"] > shortlist["first_right"]
assert shortlist["shortlist_similarity"] > shortlist["random_similarity"]

# %% Encoder-decoder: compare the writer's best epoch with its final epoch
writer_check = best_versus_final(real_results)
writer_record = writer_check["Encoder-decoder"]
if writer_record["final_gap"] < writer_record["best_gap"]:
    print("The writer's group separation faded after its best held-out checkpoint.")
else:
    print("This run did not show fading group separation; judge the recorded losses too.")

# %% Comparison: optional seed check (about two minutes per extra seed)
RUN_SEED_CHECK = False
if RUN_SEED_CHECK:
    for check_seed in (5, 123):
        print(f"--- seed {check_seed}")
        best_versus_final(train_comparison(
            real_corpus, CBOW, SkipGram, EncoderDecoderLM,
            epochs=REAL_TEXT_EPOCHS, seed=check_seed, verbose=False,
        ))
else:
    print("Skipped. Set RUN_SEED_CHECK = True to retrain with seeds 5 and 123.")
print(f"Whole default lesson: {time.perf_counter() - notebook_started:.1f}s "
      "(cache verified, one CPU thread; first download is extra).")
