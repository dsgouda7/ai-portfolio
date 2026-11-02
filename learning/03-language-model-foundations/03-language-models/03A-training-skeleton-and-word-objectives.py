"""Code companion for the focused word-embedding lesson; larger comparisons run in their own optional lab."""

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

    # If no document is long enough, return an empty array.
    if not per_document:
        return np.empty((0, window_width), dtype=np.int64)

    # Concatenate windows from each document without crossing boundaries.
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
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset

class WindowDataLoader(ABC):
    # Inherits ABC to require pairing hooks while owning window checks and seeded batching.
    """Split windows once; subclasses choose which words are inputs and answers."""

    @staticmethod
    def validate_window_ids(window_ids: torch.Tensor) -> None:
        if not isinstance(window_ids, torch.Tensor):
            raise TypeError("Word windows must be a tensor of IDs.")
        if window_ids.ndim != 2 or window_ids.shape[1] < 3 or window_ids.shape[1] % 2 == 0:
            raise ValueError("Word windows must have an odd width of at least three.")
        if window_ids.dtype != torch.long:
            raise TypeError("Word windows must use torch.long IDs.")
        if len(window_ids) == 0 or bool((window_ids < 0).any()):
            raise ValueError("Word windows must be nonempty and contain nonnegative IDs.")

    @staticmethod
    def batch_tensors(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        batch_size: int = 512,
        *,
        seed: int = 17,
    ) -> DataLoader:
        if not isinstance(input_ids, torch.Tensor) or not isinstance(target_ids, torch.Tensor):
            raise TypeError("Inputs and targets must be tensors of word IDs.")
        if input_ids.ndim < 1 or target_ids.ndim < 1:
            raise ValueError("Inputs and targets need an example dimension.")
        if len(input_ids) != len(target_ids) or len(input_ids) == 0:
            raise ValueError("Inputs and targets must contain the same nonzero number of examples.")
        if input_ids.dtype != torch.long or target_ids.dtype != torch.long:
            raise TypeError("Word inputs and targets must use torch.long IDs.")
        return DataLoader(
            TensorDataset(input_ids, target_ids),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    def __init__(self, window_ids: torch.Tensor, *, batch_size: int = 512, seed: int = 17):
        self.validate_window_ids(window_ids)
        self.batch_size = batch_size
        self.seed = seed
        center = window_ids.shape[1] // 2
        self.center_ids = window_ids[:, center]
        self.neighbour_ids = torch.cat(
            [window_ids[:, :center], window_ids[:, center + 1:]], dim=1,
        )
        self._examples = self._prepare_examples(self.center_ids, self.neighbour_ids)

    @staticmethod
    @abstractmethod
    def _prepare_examples(
        center_ids: torch.Tensor, neighbour_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("A word objective must define its input/target pairs.")

    def prepare_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._examples

    def make_dataloader(self) -> DataLoader:
        return self.batch_tensors(
            *self.prepare_data(), batch_size=self.batch_size, seed=self.seed,
        )

# %% 2.3 - Remove the center answer from the clues
class CBOWDataLoader(WindowDataLoader):
    # Inherits shared checks and batching; uses neighbours as inputs and the center as the answer.
    @staticmethod
    def _prepare_examples(
        center_ids: torch.Tensor, neighbour_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return neighbour_ids, center_ids

cbow_data = CBOWDataLoader(window_ids, seed=SEED)
cbow_contexts, cbow_targets = cbow_data.prepare_data()
cbow_dataloader = cbow_data.make_dataloader()
cbow_batch_contexts, cbow_batch_targets = next(iter(cbow_dataloader))

print("CBOW: neighbours -> middle word")
print(
    "Illustrated cat-document pair:",
    [itos[index] for index in cbow_contexts[-3].tolist()],
    "->", itos[cbow_targets[-3]],
)
print("Questions:", len(cbow_contexts))
print("Batch inputs:", tuple(cbow_batch_contexts.shape))
print("Batch answers:", tuple(cbow_batch_targets.shape))

# All later models use these same reserved stories and fixed word IDs.
from real_text_embeddings import prepare_validation_windows
VALIDATION_STORIES = 200
validation_windows = prepare_validation_windows(
    stoi, INTRO_STORIES, reserved_stories=VALIDATION_STORIES, window_width=window_width,
)

# %% 3 - Define the shared training contract
from collections.abc import Callable
from dataclasses import dataclass
from torch.optim import Optimizer

@dataclass
class TrainingRun:
    # A plain result record keeps measured losses and time separate from the model's weights.
    losses: list[float]
    seconds: float


class TrainableLanguageModel(nn.Module, ABC):
    # Inherits PyTorch parameters/modes and requires forward; the base owns training-step order.
    """Share target alignment and optimization across language objectives."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("A language model must define its prediction route.")

    def targets_for_loss(self, target_ids):
        # Which answers should the guesses be marked against?
        # CBOW and Skip-gram: the loader's answers already line up one-to-one
        # with the guesses, so hand them back unchanged.
        # A subclass overrides this when guesses and answers are offset
        # (the encoder-decoder drops its <bos> start marker).
        return target_ids

    def training_loss(self, input_ids, target_ids, loss_function):
        # 1. Make the guesses. self(...) runs the subclass's forward().
        #    target_ids rides along because the encoder-decoder needs it for
        #    teacher forcing; CBOW and Skip-gram simply ignore it.
        #      CBOW / Skip-gram scores: [B, V]     one guess per lesson
        #      Encoder-decoder scores:  [B, T, V]  one guess per output step
        logits = self(input_ids, target_ids)

        # 2. Pick the answers those guesses should be checked against.
        expected_ids = self.targets_for_loss(target_ids)

        # 3. Calculate the flat versions of logits and targets for the loss function.
        # logits.reshape(-1, logits.shape[-1]) flattens the batch and time dimensions
        # into one dimension, keeping the vocabulary dimension intact. This is needed
        # for the loss function which expects [N, V] for logits and [N] for targets.
        # for example, if logits is [2, 3, 5] (B=2, T=3, V=5), flat_logits
        # will be [6, 5] and flat_targets will be [6]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = expected_ids.reshape(-1)

        # 4. One number: on average, how surprised was the model by the right answers?
        return loss_function(flat_logits, flat_targets)

    def configure_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def configure_optimizer(self, learning_rate: float) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def train_model(
        self,
        dataloader: DataLoader,
        epochs: int,
        learning_rate: float = 0.008,
        *,
        on_epoch_end: Callable[[int, float], None] | None = None,
    ) -> TrainingRun:
        if epochs < 1:
            raise ValueError("Training requires at least one epoch.")
        loss_function = self.configure_loss_function()
        optimizer = self.configure_optimizer(learning_rate)
        started = time.perf_counter()
        losses = self.fit_language_model(
            dataloader, loss_function, optimizer, epochs, on_epoch_end=on_epoch_end,
        )
        return TrainingRun(losses, time.perf_counter() - started)

    def fit_language_model(
        self, dataloader, loss_function, optimizer, epochs,
        *, on_epoch_end: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        if epochs < 1:
            raise ValueError("Training requires at least one epoch.")
        history = []

        for epoch in range(1, epochs + 1):
            # set the model to training mode
            self.train()
            total_loss = 0.0
            example_count = 0

            for input_batch, target_batch in dataloader:
                # initialize the gradients for this batch
                optimizer.zero_grad()
                # compute the loss for this batch
                loss = self.training_loss(input_batch, target_batch, loss_function)
                # compute the gradients for this batch
                loss.backward()
                # update the model parameters based on the gradients
                # (subtract the gradients scaled by the learning rate)
                optimizer.step()

                # loss is the AVERAGE over this batch. Times the batch size turns it
                # back into a batch TOTAL, so a short last batch (say 2 lessons
                # instead of 5) counts for 2 lessons, not as much as a full batch.
                # this math allows us to accumulate the total loss correctly even if the
                # last batch is smaller than the others.
                total_loss += loss.item() * len(input_batch)
                example_count += len(input_batch)

            if example_count == 0:
                raise ValueError("Training requires at least one example per epoch.")
            epoch_loss = total_loss / example_count
            history.append(epoch_loss)
            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_loss)

        return history

# %% 3 - Define the shared word-model route
class WordEmbeddingLanguageModel(TrainableLanguageModel):
    # Inherits training and shares word layers; subclasses supply the embedding-combination rule.
    """Share lookup and prediction layers; subclasses decide whether to pool words."""

    def __init__(self, vocabulary_size: int, embedding_width: int):
        super().__init__()
        # One trainable [V, D] table belongs to this model, not to each input.
        self.embedding = nn.Embedding(vocabulary_size, embedding_width)
        self.hidden = nn.Linear(embedding_width, embedding_width // 2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(embedding_width // 2, vocabulary_size)

    @abstractmethod
    def combine_embeddings(self, token_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("A word model must define how it combines its input rows.")

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        # IDs select rows; only CBOW combines several rows into one context.
        token_vectors = self.embedding(input_ids)
        features = self.combine_embeddings(token_vectors)
        hidden_features = self.activation(self.hidden(features))
        return self.output(hidden_features)

# %% 3 - Create one embedding table inside CBOW

EMBEDDING_WIDTH = 32


class CBOW(WordEmbeddingLanguageModel):
    # Inherits the shared word-model route; averages the fetched clue rows.
    def combine_embeddings(self, token_vectors):
        # [B, C, D] -> [B, D]: average clues within a question, not across the batch.
        return token_vectors.mean(dim=1)

# %% 3.1 - Create the untrained model and check score shapes
torch.manual_seed(SEED)

# initialize the model
model = CBOW(vocabulary_size, EMBEDDING_WIDTH)

print("Trainable embedding table [V, D]:", tuple(model.embedding.weight.shape))

# run the untrained model on the CBOW contexts
untrained_logits = model(cbow_batch_contexts)

print("Input cbow_contexts:", tuple(cbow_batch_contexts.shape))
print("Vocabulary logits:", tuple(untrained_logits.shape))

# %% 3.1 - Import prediction recording helpers
from language_validation import ModelProbeRecorder, word_probe

# %% 3.1 - Before training: inspect three familiar questions
cat_contexts, _ = CBOWDataLoader(cat_window_ids).prepare_data()
cbow_recorder = ModelProbeRecorder(
    model, cat_contexts, words=itos, expected_words=("cat", "sat", "on"), name="CBOW",
)
cbow_before_record = cbow_recorder.record_before(show=True)
embedding_before = cbow_before_record.embeddings
cat_contexts_before = cbow_before_record.input_ids
cat_before_logits = cbow_before_record.logits
cbow_before = cbow_before_record.probes

# %% 3.2 - Prepare validation without owning its implementation
from language_validation import CBOWValidator

cbow_validator = CBOWValidator(validation_windows, words=itos)

# %% 3.3 - Train this CBOW now and watch each epoch
validation_loss_before = cbow_validator.record_before(model).loss
cbow_learning_history = [{
    "epoch": 0, "train_loss": None, "heldout_loss": validation_loss_before,
    "cat_probability": cbow_before["cat"]["probability"],
    "cat_rank": cbow_before["cat"]["rank"], "cat_row_shift": 0.0,
}]
print("CBOW epoch | practice loss | reserved loss | P(cat) | cat rank | cat row change")
print(f"0 | not trained | {validation_loss_before:.4f} | "
      f"{cbow_before['cat']['probability']:.3%} | {cbow_before['cat']['rank']} | 0")


def record_cbow_epoch(epoch, epoch_loss):
    with torch.inference_mode():
        epoch_cat = word_probe(model(cat_contexts[:1])[0], "cat", words=itos)
        cat_row_shift = (
            model.embedding.weight[stoi["cat"]] - embedding_before[stoi["cat"]]
        ).norm().item()
    epoch_heldout = cbow_validator.evaluate(model).loss
    cbow_learning_history.append({
        "epoch": epoch, "train_loss": epoch_loss, "heldout_loss": epoch_heldout,
        "cat_probability": epoch_cat["probability"], "cat_rank": epoch_cat["rank"],
        "cat_row_shift": cat_row_shift,
    })
    print(f"{epoch} | {epoch_loss:.4f} | {epoch_heldout:.4f} | "
          f"{epoch_cat['probability']:.3%} | {epoch_cat['rank']} | {cat_row_shift:.4f}")


cbow_run = model.train_model(cbow_dataloader, epochs=10, on_epoch_end=record_cbow_epoch)
losses = cbow_run.losses
cbow_seconds = cbow_run.seconds

print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
print(f"CBOW: {cbow_seconds:.1f}s on one CPU thread (10 epochs, including progress checks)")
cbow_after_record = cbow_recorder.record_after()
cbow_recorder.report_probes()
embedding_after = cbow_after_record.after.embeddings
embedding_row_shifts = cbow_after_record.row_changes
cbow_recorder.report_embeddings(("the", "cat", "sat"), features=4)
print("The learned values changed; the integer input IDs did not.")

# %% 3.3 - Compare validation before and after this run
cbow_validation_comparison = cbow_validator.compare_after(model, epochs=len(cbow_run.losses))

# %% 3.4 - Inspect the learned table
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

# %% 4.1 - Build center-to-neighbour pairs
class SkipGramDataLoader(WindowDataLoader):
    # Inherits shared checks and batching; expands each center into its neighbour-training pairs.
    @staticmethod
    def _prepare_examples(
        center_ids: torch.Tensor, neighbour_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten neighbours row by row; repeat each center beside its own answers.
        # [30] and [10, 20, 40, 50] become four aligned center/answer pairs.
        inputs = center_ids.repeat_interleave(neighbour_ids.shape[1])
        return inputs, neighbour_ids.flatten()


skipgram_data = SkipGramDataLoader(window_ids, seed=SEED)
skipgram_inputs, skipgram_targets = skipgram_data.prepare_data()
skipgram_dataloader = skipgram_data.make_dataloader()

# Let's pick the last 5 centers to trace their transformation
num_centers_to_show = 5
print(f"--- BEFORE repeat_interleave (Last {num_centers_to_show} centers) ---")
print("Centers:", skipgram_data.center_ids[-num_centers_to_show:].tolist())

# Since we looked at 5 centers and each repeats 4 times,
# the transformed inputs for those same 5 centers will be the last 20 items (5 * 4 = 20)
num_inputs_to_show = num_centers_to_show * skipgram_data.neighbour_ids.shape[1]
print(f"\n--- AFTER repeat_interleave (Corresponding {num_inputs_to_show} inputs) ---")
print("Inputs (notice each center repeated 4 times):", skipgram_inputs[-num_inputs_to_show:].tolist())

# Print those exact same last 20 inputs and their targets side by side
print(f"\n{'Input (Center)':<16} | {'Target (Context)'}")
print("-" * 35)
for inp, tgt in zip(skipgram_inputs[-num_inputs_to_show:].tolist(), skipgram_targets[-num_inputs_to_show:].tolist()):
    print(f"{inp:<16} | {tgt}")

# %% 4.2 - Train and inspect Skip-gram
class SkipGram(WordEmbeddingLanguageModel):
    # Inherits the shared word-model route; keeps its single center row unchanged.
    def combine_embeddings(self, token_vectors):
        # [B, D] stays [B, D]: one center supplies one row, so there is no average.
        return token_vectors

# same training setup as CBOW
torch.manual_seed(SEED)
skipgram_model = SkipGram(vocabulary_size, EMBEDDING_WIDTH)
from language_validation import SkipGramValidator
skipgram_validator = SkipGramValidator(validation_windows, words=itos)
skipgram_validator.record_before(skipgram_model)
# each error is treated as given a word, making the neighbors more likely to be predicted
# which means cat will be associated with all of its neighbors.
# during inference we simply pick the top 4 most likely words given a single center word
skipgram_run = skipgram_model.train_model(skipgram_dataloader, epochs=4)
skipgram_losses = skipgram_run.losses
skipgram_seconds = skipgram_run.seconds

with torch.inference_mode():
    neighbour_logits = skipgram_model(torch.tensor([stoi["cat"]]))
    predicted_ids = neighbour_logits.topk(4, dim=1).indices[0]

print(f"Skip-gram loss: {skipgram_losses[0]:.4f} -> {skipgram_losses[-1]:.4f}; "
      f"{skipgram_seconds:.1f}s (4 epochs)")
print("cat ->", [itos[index] for index in predicted_ids.tolist()])

# %% 4.2 - Compare Skip-gram validation before and after this run
skipgram_validation_comparison = skipgram_validator.compare_after(
    skipgram_model, epochs=len(skipgram_run.losses),
)

# %% 5.1 - Open saved interactive controls without model setup
from pathlib import Path
from IPython.display import HTML, display

_saved_replay_name = "03a-encoder-decoder-state-flow.html"
_saved_replay_roots = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
    Path.cwd() / "learning" / "03-language-model-foundations" / "03-language-models",
)
_saved_replay_path = next(
    (root / "images" / _saved_replay_name for root in _saved_replay_roots
     if (root / "images" / _saved_replay_name).is_file()),
    None,
)
if _saved_replay_path is None:
    raise FileNotFoundError("Saved replay not found. Run from the repository root or this chapter directory.")
display(HTML(_saved_replay_path.read_text(encoding="utf-8")))

# %% 5.1 - Prepare source words and continuation answers
def prepare_sequence_data(
    window_ids: torch.Tensor, bos_id: int, eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    WindowDataLoader.validate_window_ids(window_ids)
    if bos_id < 0 or eos_id < 0 or bos_id == eos_id:
        raise ValueError("Start and stop markers need distinct nonnegative IDs.")

    # Split each window into source tokens and continuation tokens.
    source_width = window_ids.shape[1] // 2 + 1
    sources = window_ids[:, :source_width]
    continuations = window_ids[:, source_width:]

    # Add BOS before and EOS after each continuation.
    targets = torch.cat([
        torch.full((len(window_ids), 1), bos_id, dtype=torch.long, device=window_ids.device),
        continuations,
        torch.full((len(window_ids), 1), eos_id, dtype=torch.long, device=window_ids.device),
    ], dim=1)
    return sources, targets

BOS_ID = vocabulary_size
EOS_ID = vocabulary_size + 1
sequence_itos = itos + ["<bos>", "<eos>"] # add the <bos> and <eos> tokens to the vocab
sequence_vocabulary_size = len(sequence_itos)

# for each window of size 5, use the first 3 tokens as the source and the last 2 as the continuation.
source_sequences, sequence_targets = prepare_sequence_data(window_ids, BOS_ID, EOS_ID)
sequence_dataloader = WindowDataLoader.batch_tensors(source_sequences, sequence_targets, seed=SEED)

print("Source shape:", tuple(source_sequences.shape))
print("Target shape:", tuple(sequence_targets.shape))
print("Source:", [sequence_itos[index] for index in source_sequences[-3].tolist()])
print("Target:", [sequence_itos[index] for index in sequence_targets[-3].tolist()])


sequence_batch_inputs, sequence_batch_targets = next(iter(sequence_dataloader))
print("Sequence batch sources:", tuple(sequence_batch_inputs.shape))
print("Sequence batch full answers:", tuple(sequence_batch_targets.shape))
print("Writer sees:", [sequence_itos[index] for index in sequence_targets[-3, :-1].tolist()])
print("Loss checks:", [sequence_itos[index] for index in sequence_targets[-3, 1:].tolist()])

# %% 5.1 - Define the reader, writer, and shared embedding
class EncoderDecoderLM(TrainableLanguageModel):
    # Inherits shared training; recurrent reader/writer layers share one embedding table.
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

# %% 5.3 - Check the input-row teaching route on model copies
torch.manual_seed(SEED)
sequence_model = EncoderDecoderLM(
    sequence_vocabulary_size, embedding_width=EMBEDDING_WIDTH, hidden_width=48,
)
cat_sources = source_sequences[-3:]
cat_answers = sequence_targets[-3:]
from encoder_decoder_feedback import compare_feedback_routes
embedding_feedback = compare_feedback_routes(
    sequence_model, cat_sources[:1], cat_answers[:1], sequence_itos,
)

# %% 5.4 - Train the reader/writer; keep reporting separate
from language_validation import EncoderDecoderProbeRecorder, EncoderDecoderValidator
sequence_validator = EncoderDecoderValidator(validation_windows, words=sequence_itos)
sequence_validator.record_before(sequence_model)
sequence_recorder = EncoderDecoderProbeRecorder(
    sequence_model, cat_sources, cat_answers, sequence_itos,
)
sequence_before_record = sequence_recorder.record_before()
sequence_embedding_before = sequence_before_record.embeddings


def record_sequence_epoch(epoch, epoch_loss):
    sequence_recorder.record_epoch(epoch, epoch_loss)


sequence_run = sequence_model.train_model(
    sequence_dataloader, epochs=6, on_epoch_end=record_sequence_epoch,
)
sequence_losses = sequence_run.losses
sequence_seconds = sequence_run.seconds
sequence_learning_history = sequence_recorder.history
sequence_after_record = sequence_recorder.record_after()
sequence_embedding_after = sequence_after_record.after.embeddings
sequence_embedding_shifts = sequence_after_record.row_changes
sequence_readout = sequence_recorder.report_after_training(
    source_sequences, sequence_targets, sequence_losses, sequence_seconds,
)

# %% 5.4 - Check the same reserved stories before and after
sequence_validation_comparison = sequence_validator.compare_after(
    sequence_model, epochs=len(sequence_run.losses),
)

# %% 5.5 - Measure order and inspect one uncorrected continuation
from language_validation import compare_source_order, greedy_decode
order_check = compare_source_order(sequence_model, cat_sources[:1], BOS_ID)
print(f"Mean-row difference (rounding only): {order_check.mean_difference:.2e}")
print(f"Final-state difference after reversal: {order_check.hidden_difference:.6f}")
print(f"First word-score difference: {order_check.score_difference:.6f}")
generated_ids = greedy_decode(
    sequence_model, cat_sources[:1], BOS_ID, EOS_ID, max_steps=sequence_targets.shape[1] - 1,
)
generated_words = [sequence_itos[index] for index in generated_ids[0].tolist()]
print("Without correct previous words:", generated_words)
