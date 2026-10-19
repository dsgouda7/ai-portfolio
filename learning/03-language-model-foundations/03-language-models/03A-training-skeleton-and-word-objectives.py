"""Code companion for 03A-training-skeleton-and-word-objectives.ipynb. Run from this directory."""

# %% 1 - Keep the corpus small enough to inspect
import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

SEED = 17
torch.manual_seed(SEED)
np.random.seed(SEED)

LEGAL_CORPUS = [
    "the non-disclosure agreement protects confidential information",
    "the contract protects confidential records",
    "the agreement lists payment obligations",
    "a non-disclosure clause limits disclosure",
    "the contract and agreement bind the company",
]


def tokenize(text):
    return re.findall(r"[a-z-]+", text.lower())


documents = [tokenize(text) for text in LEGAL_CORPUS]
print(f"Documents: {len(documents)}")
print("First tokenized document:", documents[0])

# %% 2 - A vocabulary turns repeated strings into stable addresses
items = []
for document in documents:
    for word in document:
        items.append(word)

counts = Counter(items)
itos = ["<unk>"] + [word for word, _ in counts.most_common()]
stoi = {word: index for index, word in enumerate(itos)}
vocabulary_size = len(itos)

id_documents = [
    [stoi.get(word, stoi["<unk>"]) for word in document]
    for document in documents
]

print(f"Tokens counted: {len(items)}")
print(f"Vocabulary size: {vocabulary_size}")
print("First 10 index-to-string entries:", itos[:10])
print("First encoded document:", id_documents[0])
assert all(itos[index] == word for word, index in stoi.items())

# %% 3 - A sliding window manufactures prediction lessons
CONTEXT_RADIUS = 2
window_width = 2 * CONTEXT_RADIUS + 1

per_document_windows = [
    sliding_window_view(np.asarray(ids), window_width)
    for ids in id_documents
    if len(ids) >= window_width
]
windows = np.concatenate(per_document_windows)
windows = windows[~(windows == stoi["<unk>"]).any(axis=1)]

print(f"Window width: {window_width}")
print(f"All windows shape: {windows.shape}")
for row in windows[:4]:
    print([itos[index] for index in row])

# %% Separate what the model sees from what it must predict
contexts = torch.tensor(
    np.delete(windows, CONTEXT_RADIUS, axis=1),
    dtype=torch.long,
)
targets = torch.tensor(windows[:, CONTEXT_RADIUS], dtype=torch.long)

print("Complete windows:", windows.shape)
print("Context tensor:", tuple(contexts.shape))
print("Target tensor:", tuple(targets.shape))
print(
    "First lesson:",
    [itos[index] for index in contexts[0].tolist()],
    "->",
    itos[targets[0]],
)

assert contexts.shape == (len(windows), 2 * CONTEXT_RADIUS)
assert targets.shape == (len(windows),)

# %% 4 - DataLoader changes delivery, not meaning
def make_dataloader(input_ids, target_ids, batch_size=None):
    dataset = TensorDataset(input_ids, target_ids)
    return DataLoader(
        dataset,
        batch_size=batch_size or len(dataset),
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )


dataloader = make_dataloader(contexts, targets)
batch_contexts, batch_targets = next(iter(dataloader))

print("Batch contexts:", tuple(batch_contexts.shape))
print("Batch targets:", tuple(batch_targets.shape))
print("First batch target:", itos[batch_targets[0]])
assert len(batch_contexts) == len(batch_targets)

# %% 5 - Crude attempt: give every context word an equal vote
EMBEDDING_WIDTH = 12
probe_embedding = nn.Embedding(vocabulary_size, EMBEDDING_WIDTH)

one_context = contexts[:1]
reversed_context = one_context.flip(dims=[1])
one_vectors = probe_embedding(one_context)
original_mean = one_vectors.mean(dim=1)
reversed_mean = probe_embedding(reversed_context).mean(dim=1)

print("IDs:", one_context.tolist())
print("After lookup:", tuple(one_vectors.shape))
print("After context mean:", tuple(original_mean.shape))
print("Maximum difference after reversal:", (original_mean - reversed_mean).abs().max().item())

assert torch.allclose(original_mean, reversed_mean)

# %% Where EmbeddingBag fits
embedding_bag = nn.EmbeddingBag(
    vocabulary_size,
    EMBEDDING_WIDTH,
    mode="mean",
)
with torch.no_grad():
    embedding_bag.weight.copy_(probe_embedding.weight)

manual_means = probe_embedding(contexts[:3]).mean(dim=1)
bag_means = embedding_bag(contexts[:3])

print("Manual mean shape:", tuple(manual_means.shape))
print("EmbeddingBag shape:", tuple(bag_means.shape))
print("Maximum difference:", (manual_means - bag_means).abs().max().item())
assert torch.allclose(manual_means, bag_means)

# %% 6 - Give replaceable models one small training contract
from language_model import TrainableLanguageModel


class CBOW(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        self.hidden = nn.Linear(embedding_width, embedding_width // 2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(embedding_width // 2, vocabulary_size)

    def forward(self, context_ids, target_ids=None):
        token_vectors = self.embedding(context_ids)
        pooled_context = token_vectors.mean(dim=1)
        hidden_features = self.activation(self.hidden(pooled_context))
        return self.output(hidden_features)

# %% Trace one untrained batch before optimizing anything
torch.manual_seed(SEED)
model = CBOW(vocabulary_size, EMBEDDING_WIDTH)
untrained_logits = model(batch_contexts)

print("Input contexts:", tuple(batch_contexts.shape))
print("Vocabulary logits:", tuple(untrained_logits.shape))
assert untrained_logits.shape == (len(batch_contexts), vocabulary_size)

# %% 7 - One fit loop should survive the model swap
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
losses = model.fit_language_model(
    dataloader,
    criterion,
    optimizer,
    epochs=180,
)

print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

# %% A falling loss is necessary evidence, not the whole story
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

print(f"Random-guess baseline: {random_guess_loss:.4f}")
print(f"Final training loss: {losses[-1]:.4f}")
assert losses[-1] < losses[0]
assert losses[-1] < random_guess_loss

# %% 8 - Follow one gradient back to the participating rows
model.zero_grad()
gradient_contexts = contexts[:2]
gradient_targets = targets[:2]
gradient_loss = model.training_loss(
    gradient_contexts,
    gradient_targets,
    criterion,
)
gradient_loss.backward()

gradient_norms = model.embedding.weight.grad.norm(dim=1)
used_ids = set(gradient_contexts.reshape(-1).tolist())
rows_with_gradient = {
    index for index, norm in enumerate(gradient_norms.tolist()) if norm > 0
}

print("Used rows:", [itos[index] for index in sorted(used_ids)])
print("Rows with gradients:", [itos[index] for index in sorted(rows_with_gradient)])
assert rows_with_gradient == used_ids

# %% 9 - The learned embedding weights are the vectors
embedding_weights = model.embedding.weight.detach().cpu()
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


for word in ["contract", "agreement", "protects", "confidential"]:
    print(f"{word:>12} -> {most_similar(word)}")

# %% A 2D projection is a map, not the territory
visible_count = min(16, vocabulary_size - 1)
visible_ids = torch.tensor(
    [stoi[word] for word, _ in counts.most_common(visible_count)]
)
visible_vectors = embedding_weights[visible_ids]
centered = visible_vectors - visible_vectors.mean(dim=0, keepdim=True)
_, _, right_singular_vectors = torch.linalg.svd(centered, full_matrices=False)
coordinates = centered @ right_singular_vectors[:2].T

fig, axis = plt.subplots(figsize=(10, 7))
axis.scatter(coordinates[:, 0], coordinates[:, 1], color="#2a6f97", s=55)
for point, token_id in zip(coordinates, visible_ids.tolist()):
    axis.annotate(itos[token_id], point, xytext=(5, 5), textcoords="offset points")
axis.axhline(0, color="#9ca3af", linewidth=0.7)
axis.axvline(0, color="#9ca3af", linewidth=0.7)
axis.set(
    title="Frequent embedding rows projected from 12D to 2D",
    xlabel="projection axis 1",
    ylabel="projection axis 2",
)
axis.grid(alpha=0.15)
plt.tight_layout()
plt.show()

# %% 10 - Flip the arrow: Skip-gram predicts neighbours from the center
skipgram_inputs = targets.repeat_interleave(contexts.shape[1])
skipgram_targets = contexts.reshape(-1)
skipgram_dataloader = make_dataloader(skipgram_inputs, skipgram_targets)

print("CBOW lessons:", len(contexts))
print("Skip-gram lessons:", len(skipgram_inputs))
print(
    "First Skip-gram lesson:",
    itos[skipgram_inputs[0]],
    "->",
    itos[skipgram_targets[0]],
)

assert len(skipgram_inputs) == len(contexts) * contexts.shape[1]
assert len(skipgram_inputs) == len(skipgram_targets)

# %% The model swap is smaller than the objective swap
class SkipGram(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        self.hidden = nn.Linear(embedding_width, embedding_width // 2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(embedding_width // 2, vocabulary_size)

    def forward(self, center_ids, target_ids=None):
        center_vectors = self.embedding(center_ids)
        hidden_features = self.activation(self.hidden(center_vectors))
        return self.output(hidden_features)


torch.manual_seed(SEED)
skipgram_model = SkipGram(vocabulary_size, EMBEDDING_WIDTH)
skipgram_optimizer = torch.optim.Adam(skipgram_model.parameters(), lr=0.03)
skipgram_losses = skipgram_model.fit_language_model(
    skipgram_dataloader,
    criterion,
    skipgram_optimizer,
    epochs=220,
)

with torch.inference_mode():
    probe_center = targets[:1]
    neighbour_logits = skipgram_model(probe_center)
    predicted_ids = neighbour_logits.topk(contexts.shape[1], dim=1).indices[0]

print(f"Skip-gram loss: {skipgram_losses[0]:.4f} -> {skipgram_losses[-1]:.4f}")
print("Center:", itos[probe_center.item()])
print("Top predicted neighbours:", [itos[index] for index in predicted_ids.tolist()])

assert skipgram_losses[-1] < skipgram_losses[0]
assert skipgram_losses[-1] < math.log(vocabulary_size)
assert neighbour_logits.shape == (1, vocabulary_size)

# %% 11 - Both static objectives still miss token roles
model.eval()
ordered_context = contexts[:1]
permuted_context = ordered_context[:, torch.tensor([3, 1, 0, 2])]

with torch.no_grad():
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

# %% 12 - One target token is not a target sequence
BOS_ID = vocabulary_size
EOS_ID = vocabulary_size + 1
sequence_itos = itos + ["<bos>", "<eos>"]
sequence_vocabulary_size = len(sequence_itos)

window_ids = torch.tensor(windows, dtype=torch.long)
source_sequences = window_ids[:, :3]
continuations = window_ids[:, 3:]
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
print("Source:", [sequence_itos[index] for index in source_sequences[0].tolist()])
print("Target:", [sequence_itos[index] for index in sequence_targets[0].tolist()])

assert source_sequences.shape == (len(windows), 3)
assert sequence_targets.shape == (len(windows), 4)

# %% A destination preview, not a second embedding lesson
class EncoderDecoderLM(TrainableLanguageModel):
    def __init__(self, vocabulary_size, embedding_width, hidden_width):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        self.encoder = nn.GRU(embedding_width, hidden_width, batch_first=True)
        self.decoder = nn.GRU(embedding_width, hidden_width, batch_first=True)
        self.output = nn.Linear(hidden_width, vocabulary_size)

    def encode(self, source_ids):
        _, hidden = self.encoder(self.embedding(source_ids))
        return hidden

    def decode(self, decoder_input_ids, hidden):
        decoder_states, hidden = self.decoder(
            self.embedding(decoder_input_ids),
            hidden,
        )
        return self.output(decoder_states), hidden

    def forward(self, source_ids, target_ids=None):
        if target_ids is None:
            raise ValueError("Training requires target IDs for teacher forcing.")
        hidden = self.encode(source_ids)
        logits, _ = self.decode(target_ids[:, :-1], hidden)
        return logits

    def targets_for_loss(self, target_ids):
        return target_ids[:, 1:]


torch.manual_seed(SEED)
sequence_model = EncoderDecoderLM(
    sequence_vocabulary_size,
    embedding_width=EMBEDDING_WIDTH,
    hidden_width=24,
)
sequence_optimizer = torch.optim.Adam(sequence_model.parameters(), lr=0.03)
assert EncoderDecoderLM.fit_language_model is TrainableLanguageModel.fit_language_model
sequence_losses = sequence_model.fit_language_model(
    sequence_dataloader,
    criterion,
    sequence_optimizer,
    epochs=400,
)

with torch.inference_mode():
    teacher_forced_logits = sequence_model(source_sequences, sequence_targets)
    teacher_forced_predictions = teacher_forced_logits.argmax(dim=-1)
    expected_sequence_ids = sequence_targets[:, 1:]
    teacher_forced_accuracy = (
        teacher_forced_predictions == expected_sequence_ids
    ).float().mean().item()

print(f"Seq2Seq loss: {sequence_losses[0]:.4f} -> {sequence_losses[-1]:.4f}")
print("Teacher-forced token accuracy:", f"{teacher_forced_accuracy:.0%}")
print("Sequence logits:", tuple(teacher_forced_logits.shape))

assert sequence_losses[-1] < sequence_losses[0]
assert sequence_losses[-1] < math.log(sequence_vocabulary_size)
assert teacher_forced_accuracy >= 0.9

# %% Teacher forcing worked - can the decoder use its own answer?
def greedy_decode(model, source_ids, bos_id, eos_id, max_steps):
    model.eval()
    hidden = model.encode(source_ids)
    current_ids = torch.full(
        (len(source_ids), 1),
        bos_id,
        dtype=torch.long,
    )
    generated = []

    with torch.inference_mode():
        for _ in range(max_steps):
            logits, hidden = model.decode(current_ids, hidden)
            next_ids = logits[:, -1].argmax(dim=-1)
            generated.append(next_ids)
            current_ids = next_ids.unsqueeze(1)
            if bool((next_ids == eos_id).all()):
                break

    return torch.stack(generated, dim=1)


generated_ids = greedy_decode(
    sequence_model,
    source_sequences[:1],
    BOS_ID,
    EOS_ID,
    max_steps=sequence_targets.shape[1] - 1,
)

with torch.inference_mode():
    ordered_hidden = sequence_model.encode(source_sequences[:1])
    reversed_hidden = sequence_model.encode(source_sequences[:1].flip(dims=[1]))

expected_words = [
    sequence_itos[index]
    for index in sequence_targets[0, 1:].tolist()
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
