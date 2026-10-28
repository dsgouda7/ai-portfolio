"""Code companion for 03A. Core first; optional extensions also run when executing the whole script."""

# %% Setup - Import the tools for one text-trained embedding model
import math
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Repeat the starting draw and bound this CPU lesson's parallel work.
SEED = 17
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
# Measure the whole run, including any optional extensions executed with Run All.
notebook_started = time.perf_counter()

# %% 2 - Let real text supply practice questions
CHAPTER_DIRECTORY = (
    Path(__file__).resolve().parent if "__file__" in globals() else
    next(candidate for candidate in (
        Path.cwd(), Path.cwd() / "learning" / "03-language-model-foundations" / "03-language-models",
    ) if (candidate / "real_text_embeddings.py").is_file())
)
if str(CHAPTER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIRECTORY))
from real_text_embeddings import read_story_sentences, read_stories, story_sentences, tokenize

INTRO_STORIES = 1_000
CAT_PROBE = "the sleepy cat sat on the mat"
real_documents = read_story_sentences(INTRO_STORIES)
# One independent document at the END, not copies inserted among the stories.
documents = real_documents + [tokenize(CAT_PROBE)]
print(f"Real source: first {INTRO_STORIES:,} pinned Hugging Face TinyStories stories")
print(f"Documents: {len(documents)}")
print("First tokenized document:", documents[0])
print("Last document (appended exactly once):", documents[-1])
print("Tokenized appended document:", tokenize(CAT_PROBE))

# %% 2.1 - Count, retain, and encode
items = [word for document in documents for word in document]

counts = Counter(items)
itos = ["<unk>"] + [word for word, _ in counts.most_common(1_199)]
# Rare mat still needs an address to be probed. This adds no examples or weight.
# itos => integer to string
# stoi => string to integer
# we need these lookups to convert between words and their integer IDs
# since model takes in and outputs integer IDs rather than raw strings
itos += [word for word in dict.fromkeys(tokenize(CAT_PROBE)) if word not in itos]
stoi = {word: index for index, word in enumerate(itos)}
vocabulary_size = len(itos)

# convert documents from strings to integer IDs
# i.e., each document is now a list of integer IDs
# corresponding to its words, instead of the raw strings
id_documents = [
    # stoi.get(word, stoi["<unk>"]) looks up the integer ID for each word
    # <unk> is the fallback ID for words not in the vocabulary
    [stoi.get(word, stoi["<unk>"]) for word in document]
    for document in documents
]

print(f"Tokens counted: {len(items)}")
print(f"Vocabulary size: {vocabulary_size}")
print("Cat-probe word counts (not weights):",
      {word: counts[word] for word in dict.fromkeys(tokenize(CAT_PROBE))})

# %% 2.2 - Use NumPy's document-local sliding view
def make_document_windows(id_documents, window_width):
    """Keep complete windows in document order; never bridge document boundaries."""
    if window_width < 1:
        raise ValueError("window_width must be at least one.")

    per_document = [
        np.lib.stride_tricks.sliding_window_view(
            np.asarray(document, dtype=np.int64), window_width,
        )
        for document in id_documents
        if len(document) >= window_width
    ]
    if not per_document:
        return np.empty((0, window_width), dtype=np.int64)
    return np.concatenate(per_document)


CONTEXT_RADIUS = 2
window_width = 2 * CONTEXT_RADIUS + 1
# Reuse the cutter here; each objective below chooses its own inputs and answers.
windows = make_document_windows(id_documents, window_width)

print(f"Window width: {window_width}")
print(f"All windows shape: {windows.shape}")
print(f"Unknown window slots: {(windows == 0).mean():.1%}")
print("Our cat document, sliding exactly as in the opening image:")
# Print the last few windows to inspect the cat document
for row in windows[-3:]:
    print([itos[index] for index in row])

# Convert the windows to PyTorch tensors for model input
window_ids = torch.tensor(windows, dtype=torch.long)

# Extract the windows corresponding to the last document (the cat document)
cat_window_ids = window_ids[-3:]

# %% 2.2 - Your turn: Change only the probe width
probe_width = 3
probe_windows = make_document_windows([id_documents[-1]], probe_width)
for start, row in enumerate(probe_windows):
    print(f"start={start}: {' '.join(itos[token_id] for token_id in row)}")
print(f"{len(probe_windows)} complete {probe_width}-word windows; training width stays {window_width}.")

# %% 2.3 - Deliver matching inputs and answers
def make_dataloader(input_ids, target_ids, batch_size=512):
    dataset = TensorDataset(input_ids, target_ids)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

# %% 2.3 - Remove the center answer from the clues
# Two [N, 2] slices -> [N, 4]: join left/right clues without the center answer.
# torch.cat joins the left and right context words along the second dimension (dim=1)
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

# %% 3 - Create one embedding table inside CBOW
from language_model import TrainableLanguageModel

EMBEDDING_WIDTH = 32


class CBOW(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width):
        super().__init__()
        # Allocate one random trainable [V, D] table, not a new table per input.
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        self.hidden = nn.Linear(embedding_width, embedding_width // 2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(embedding_width // 2, vocabulary_size)

    def forward(self, context_ids, target_ids=None):
        # The shared trainer passes targets too; this predictor never reads them.
        # [B, C] IDs select existing rows -> [B, C, D]; the IDs do not change.
        # Autograd sends lookup gradients back only to the rows selected here.
        token_vectors = self.embedding(context_ids)
        # CBOW averages clues within each question, not across the batch.
        pooled_context = token_vectors.mean(dim=1)  # [B, D]
        hidden_features = self.activation(self.hidden(pooled_context))
        return self.output(hidden_features)  # [B, V] vocabulary scores

# %% 3.1 - Inspect the untrained model
torch.manual_seed(SEED)
model = CBOW(vocabulary_size, EMBEDDING_WIDTH)
embedding_before = model.embedding.weight.detach().clone()
print("Trainable embedding table [V, D]:", tuple(embedding_before.shape))
assert embedding_before.shape == (vocabulary_size, EMBEDDING_WIDTH)
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
cat_contexts_before = cat_contexts.clone()
with torch.inference_mode():
    cat_before_logits = model(cat_contexts)
    cbow_before = {word: word_probe(cat_before_logits[row], word)
                   for row, word in enumerate(("cat", "sat", "on"))}

print("Input cbow_contexts:", tuple(cbow_batch_contexts.shape))
print("Vocabulary logits:", tuple(untrained_logits.shape))
assert untrained_logits.shape == (len(cbow_batch_contexts), vocabulary_size)

# %% 3.1 - Record one real CBOW step without changing the model being trained
from copy import deepcopy

routing_question_index = 0  # CHANGE THIS to 1 or 2 to inspect another real window.
routing_contexts = cat_contexts[routing_question_index:routing_question_index + 1].clone()
routing_targets = cat_window_ids[routing_question_index:routing_question_index + 1, CONTEXT_RADIUS].clone()
routing_model = deepcopy(model)
routing_optimizer = torch.optim.SGD(routing_model.parameters(), lr=0.1)
routing_optimizer.zero_grad()
routing_weights = [routing_model.embedding.weight.detach().clone()]
routing_logits = routing_model(routing_contexts)
routing_weights.append(routing_model.embedding.weight.detach().clone())
routing_loss = nn.functional.cross_entropy(routing_logits, routing_targets)
routing_loss.backward()
routing_weights.append(routing_model.embedding.weight.detach().clone())
routing_gradient = routing_model.embedding.weight.grad.detach().clone()
routing_head_gradient = routing_model.output.bias.grad.detach().clone()
assert torch.equal(routing_weights[0], routing_weights[1])
assert torch.equal(routing_weights[0], routing_weights[2])
routing_optimizer.step()
routing_weights.append(routing_model.embedding.weight.detach().clone())
routing_selected = set(routing_contexts.flatten().tolist())
routing_gradient_rows = set(routing_gradient.ne(0).any(dim=1).nonzero().flatten().tolist())
routing_changed_rows = set(routing_weights[3].ne(routing_weights[0]).any(dim=1).nonzero().flatten().tolist())
assert routing_gradient_rows <= routing_selected
assert routing_changed_rows <= routing_selected
routing_trace = {
    "contexts": routing_contexts, "targets": routing_targets,
    "weights": routing_weights, "gradient": routing_gradient,
}
print("Forward lookup rows:", [itos[index] for index in sorted(routing_selected)])
print("Rows with input-embedding gradients:", [itos[index] for index in sorted(routing_gradient_rows)])
print("Embedding rows changed by SGD:", [itos[index] for index in sorted(routing_changed_rows)])
print("Separate output-bias entries with gradients:", routing_head_gradient.ne(0).sum().item())
print("backward() left weights unchanged; step() applied the update.")

# %% 3.1 - Replay the measured step as a static storyboard and animation
from IPython.display import HTML, display
from real_text_embeddings import plot_embedding_routing, animate_embedding_routing

routing_figure = plot_embedding_routing(routing_trace, itos)
routing_figure.savefig(CHAPTER_DIRECTORY / "images" / "03a-embedding-forward-backward.png", dpi=140)
plt.show()
plt.close(routing_figure)
routing_animation = animate_embedding_routing(routing_trace, itos)
routing_animation.save(CHAPTER_DIRECTORY / "images" / "03a-embedding-forward-backward.gif", writer="pillow", fps=1)
display(HTML(routing_animation.to_jshtml()))

# %% 3.2 - Reserve later stories and keep their vocabulary fixed
# These stories begin after the last training story, before sentence/window construction.
VALIDATION_STORIES = 200
validation_stories = read_stories(INTRO_STORIES + VALIDATION_STORIES)[INTRO_STORIES:]
assert len(validation_stories) == VALIDATION_STORIES
# Split only the reserved stories; never join their endings into training documents.
validation_documents = [
    sentence for story in validation_stories for sentence in story_sentences(story)
]
# Reuse training IDs, including unknown zero, instead of fitting a validation vocabulary.
validation_id_documents = [
    [stoi.get(word, stoi["<unk>"]) for word in document]
    for document in validation_documents
]
# Bound evaluation work while retaining whole, sentence-local window rows.
validation_windows = torch.tensor(
    make_document_windows(validation_id_documents, window_width)[:2_048], dtype=torch.long,
)
# Apply the same four-clue / center-answer objective as the training data.
validation_contexts = validation_windows[:, [0, 1, 3, 4]]
validation_targets = validation_windows[:, CONTEXT_RADIUS]


def heldout_cbow_loss(candidate):
    "Evaluate every retained answer once; never update the candidate's parameters."
    if len(validation_targets) == 0:
        raise ValueError("Held-out evaluation requires at least one target.")
    candidate.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(validation_targets), 512):
            scores = candidate(validation_contexts[start:start + 512])
            total_loss += nn.functional.cross_entropy(
                scores, validation_targets[start:start + 512], reduction="sum",
            ).item()
    return total_loss / len(validation_targets)


print(f"Reserved stories: {INTRO_STORIES + 1} through "
      f"{INTRO_STORIES + VALIDATION_STORIES}; windows: {len(validation_windows)}")
print(f"Reserved unknown slots: {validation_windows.eq(0).float().mean():.1%}")

# %% 3.3 - Compare practice, probe, and reserved-story loss
validation_loss_before = heldout_cbow_loss(model)
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

validation_loss_after = heldout_cbow_loss(model)
print(f"Reserved-story loss: {validation_loss_before:.4f} -> {validation_loss_after:.4f}")
if validation_loss_after < validation_loss_before:
    print("The same trained model also improved on these reserved-story questions.")
else:
    print("Practice improvement did not transfer to this reserved-story check.")

embedding_after = model.embedding.weight.detach()
embedding_row_shifts = (embedding_after - embedding_before).norm(dim=1)
print("Embedding rows changed:", (embedding_row_shifts > 0).sum().item())
for word in ("the", "cat", "sat"):
    index = stoi[word]
    print(f"{word} (fixed ID {index}), first four features: "
          f"{embedding_before[index, :4].tolist()} -> {embedding_after[index, :4].tolist()}")
assert torch.any(embedding_row_shifts > 0)
assert torch.equal(cat_contexts, cat_contexts_before)
print("The learned values changed; the integer input IDs did not.")

# %% 3.4 - Compare with the random-guess baseline
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

# %% 4 - Inspect the learned table
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

# %% 4.1 - Keep probabilities fixed; change only the selection rule
model.eval()
with torch.inference_mode():
    bridge_probabilities = model(cat_contexts[:1])[0].softmax(dim=-1).cpu()
bridge_greedy_id = bridge_probabilities.argmax().item()
# CHANGE THIS: change only the sampling seed; weights and context stay fixed.
bridge_sampling_seed = SEED
bridge_sample_count = 2000
bridge_samples = torch.multinomial(
    bridge_probabilities, bridge_sample_count, replacement=True,
    generator=torch.Generator().manual_seed(bridge_sampling_seed),
)
bridge_replay = torch.multinomial(
    bridge_probabilities, bridge_sample_count, replacement=True,
    generator=torch.Generator().manual_seed(bridge_sampling_seed),
)
bridge_counts = torch.bincount(bridge_samples, minlength=vocabulary_size)
assert torch.equal(bridge_samples, bridge_replay)
assert bridge_counts.sum().item() == bridge_sample_count
print("Embedding neighbours of cat:", most_similar("cat"))
print("Greedy answer to the same gap:", itos[bridge_greedy_id])
print("Different sampled answers:", bridge_samples.unique().numel())
print("Word | model probability | observed sample share")
for index in bridge_probabilities.topk(5).indices.tolist():
    print(f"{itos[index]:>12} | {bridge_probabilities[index]:.2%} | "
          f"{bridge_counts[index].item() / bridge_sample_count:.2%}")
print("The same seed replays the draws. Sampling changed choices, not learned rows.")

# %% 5 - Permute clues after training
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

# %% A.1 - Compare fused lookup with the same trained rows
preview_contexts = cbow_contexts[:3]
# ── Where EmbeddingBag fits ──
embedding_bag = nn.EmbeddingBag(
    vocabulary_size,
    EMBEDDING_WIDTH,
    mode="mean",
)
with torch.no_grad():
    embedding_bag.weight.copy_(model.embedding.weight)

# [3, 4, 32] -> [3, 32]: average each question's clues; EmbeddingBag fuses this same job.
manual_means = model.embedding(preview_contexts).mean(dim=1)
bag_means = embedding_bag(preview_contexts)

print("Manual mean shape:", tuple(manual_means.shape))
print("EmbeddingBag shape:", tuple(bag_means.shape))
print("Maximum difference:", (manual_means - bag_means).abs().max().item())
assert torch.allclose(manual_means, bag_means)

# %% A.2 - Project learned rows for inspection
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

# %% A.3 - Animate the already-built document windows
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

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
    axis.set_title(
        f"start={frame}: document[{frame}:{frame + window_width}]", fontsize=11,
    )
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
    axis.text(3, 0.6, "Window: " + " ".join(row),
              ha="center", fontsize=10, color="#1e3a8a")
    axis.text(3, -0.05, "CBOW clues: " + " ".join(question),
              ha="center", fontsize=10, color="#007f86")
    axis.text(3, -0.7, "Checked answer: " + row[CONTEXT_RADIUS],
              ha="center", fontsize=11, color="#bf5700", weight="bold")


print(f"Prediction check: {len(window_story_frames)} complete windows, not one per word.")
print("CBOW answers:", [row[CONTEXT_RADIUS] for row in window_story_frames])
print("Watch the blue window move; CBOW keeps each orange answer out of its input.")
window_animation_figure, window_animation_axis = plt.subplots(figsize=(9, 3.2))
window_animation = FuncAnimation(
    window_animation_figure,
    lambda frame: draw_cbow_window(window_animation_axis, frame),
    frames=len(window_story_frames), interval=1300, repeat=False,
)
plt.close(window_animation_figure)
display(HTML(window_animation.to_jshtml(default_mode="once")))

# %% B.1 - Build center-to-neighbour pairs
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

# %% B.2 - Train and inspect Skip-gram
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

# %% C.1 - Build aligned source and target sequences
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

# %% C.2 - Define and train the reader/writer
class EncoderDecoderLM(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width, hidden_width):
        super().__init__()
        # Reader and writer share one registry of input profiles.          [V, D]
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

# %% C.3 - Generate without correct previous answers
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

# %% D.1 - Load the independent comparison split
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

# %% D.2 - Record all three training histories
REAL_TEXT_EPOCHS = 16
real_results = train_comparison(
    real_corpus, CBOW, SkipGram, EncoderDecoderLM, epochs=REAL_TEXT_EPOCHS,
)

# %% D.3 - Plot recorded loss and geometry
chart_directory = Path(real_text_embeddings.__file__).resolve().parent / "images"
loss_figure = plot_losses_and_geometry(real_results)
loss_figure.savefig(chart_directory / "03a-real-text-loss-and-similarity.png", dpi=130)
plt.show()
plt.close(loss_figure)

# %% D.4 - Project all histories through one camera
real_probe_words, real_coordinates, real_pca_fraction = project_snapshots(
    real_corpus, real_results,
)
map_figure = plot_word_maps(real_probe_words, real_coordinates, real_results)
map_figure.savefig(chart_directory / "03a-real-text-word-maps.png", dpi=130)
plt.show()
plt.close(map_figure)

# %% D.5 - Compare complete stored rows
similarity_figure = plot_similarity_heatmaps(real_corpus, real_results)
similarity_figure.savefig(chart_directory / "03a-real-text-cosine-heatmaps.png", dpi=130)
plt.show()
plt.close(similarity_figure)

# %% D.6 - Play the recorded snapshots
real_animation = animate_word_maps(real_probe_words, real_coordinates, real_results)
real_animation_html = real_animation.to_jshtml()
if "get_ipython" in globals():
    from IPython.display import HTML, display

    display(HTML(real_animation_html))
else:
    print(f"Recorded animation rendered ({len(real_animation_html):,} HTML characters).")

# %% D.7 - Print actual neighbour changes
report_neighbours(real_corpus, real_results)

# %% D.7 - Your turn: Inspect another pair
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

# %% D.8 - Check held-out answer shortlists
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

# %% D.9 - Check best versus final without retraining
writer_check = best_versus_final(real_results)
writer_record = writer_check["Encoder-decoder"]
if writer_record["final_gap"] < writer_record["best_gap"]:
    print("The writer's group separation faded after its best held-out checkpoint.")
else:
    print("This run did not show fading group separation; judge the recorded losses too.")

# %% D.9 - Optional extra-seed robustness check
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
