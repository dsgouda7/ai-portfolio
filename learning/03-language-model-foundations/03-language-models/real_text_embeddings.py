"""Bounded real-text experiment for 03A; model definitions stay in the lesson."""

import copy
import hashlib
import itertools
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view


DATASET = "roneneldan/TinyStories"
# Pinned commit, so the text cannot change under the lesson.
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
CORPUS_FILE = "TinyStories-valid.txt"
CORPUS_URL = (
    f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{CORPUS_FILE}"
)
CORPUS_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
STORY_BUDGET = 8_000
WORD_GROUPS = {
    "family": ("mom", "dad", "brother", "sister"),
    "animals": ("dog", "cat", "bird", "bear"),
    "colors": ("red", "blue", "green", "yellow"),
    "feelings": ("happy", "sad", "scared", "excited"),
}
COLORS = ("#007f86", "#bf5700", "#7950a3", "#2766a3")


def tokenize(text):
    """Lowercase words, keeping apostrophes inside words such as don't."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def story_sentences(story, min_words=5):
    """Split one story at . ! ? and keep sentences long enough for a five-word window."""
    sentences = [tokenize(part) for part in re.split(r"[.!?]+", story)]
    return [sentence for sentence in sentences if len(sentence) >= min_words]


def read_stories(stories):
    """Return the first `stories` stories from the verified, cached TinyStories file."""
    cache = Path(__file__).resolve().parent / "data" / "tinystories"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / CORPUS_FILE
    if not local.exists():
        print(f"Downloading {DATASET} ({CORPUS_FILE}, 19 MB) from Hugging Face.", flush=True)
        with urllib.request.urlopen(CORPUS_URL, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != CORPUS_SHA256:
            raise ValueError("Corpus checksum mismatch; download was not cached.")
        temporary = local.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(local)
    payload = local.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CORPUS_SHA256:
        raise ValueError(f"Corrupt corpus cache: remove {local} and rerun.")

    # The file's own story order, not hand-picked stories.
    return [story for story in payload.decode("utf-8").split("<|endoftext|>")
            if story.strip()][:stories]


def read_story_sentences(stories):
    """Sentences from the first `stories` stories; each sentence is its own document."""
    return list(itertools.chain.from_iterable(map(story_sentences, read_stories(stories))))


def make_document_windows(id_documents, window_width):
    """Keep complete windows in document order; never bridge document boundaries."""
    if window_width < 1:
        raise ValueError("window_width must be at least one.")
    per_document = [
        sliding_window_view(np.asarray(ids, dtype=np.int64), window_width)
        for ids in id_documents
        if len(ids) >= window_width
    ]
    if not per_document:
        return np.empty((0, window_width), dtype=np.int64)
    return np.concatenate(per_document)


def load_real_text(stories=STORY_BUDGET, train_windows=40_000, valid_windows=4_000):
    """Split whole stories before making any windows."""
    texts = read_stories(stories)
    sentences_of = story_sentences
    split = int(0.8 * len(texts))
    train_sentences = list(itertools.chain.from_iterable(map(sentences_of, texts[:split])))
    valid_sentences = list(itertools.chain.from_iterable(map(sentences_of, texts[split:])))
    counts = Counter(itertools.chain.from_iterable(train_sentences))
    words = ["<unk>"] + [word for word, _ in counts.most_common(1799)]
    word_ids = {word: index for index, word in enumerate(words)}
    generator = torch.Generator().manual_seed(71)

    def windows_for(sentences, limit):
        id_documents = [[word_ids.get(word, 0) for word in sentence] for sentence in sentences]
        windows = torch.tensor(make_document_windows(id_documents, 5), dtype=torch.long)
        # [all sentence-local windows, 5] -> [at most limit, 5]: sample whole rows,
        # never concatenate sentence ends with the next sentence's beginning.
        return windows[torch.randperm(len(windows), generator=generator)[:limit]]

    train = windows_for(train_sentences, train_windows)
    valid = windows_for(valid_sentences, valid_windows)
    probes = list(itertools.chain.from_iterable(WORD_GROUPS.values()))
    missing = [word for word in probes if word not in word_ids]
    if missing:
        raise ValueError(f"Probe words missing from the training vocabulary: {missing}")
    print(f"Dataset: {DATASET} ({CORPUS_FILE} @ {REVISION[:7]}); "
          f"{len(texts):,} stories, split 80/20 by story")
    print(f"Training text: {sum(map(len, train_sentences)):,} tokens; "
          f"held-out text: {sum(map(len, valid_sentences)):,} tokens")
    print(f"Vocabulary: {len(words):,}; windows: {len(train):,} train / {len(valid):,} held out")
    print(f"Unknown slots: train {train.eq(0).float().mean():.1%}, "
          f"held out {valid.eq(0).float().mean():.1%}")
    print("Probe counts (chosen before training):",
          {word: counts[word] for word in probes})
    return {"train": train, "valid": valid, "words": words, "word_ids": word_ids}


def objective_pairs(windows, objective, bos_id):
    """Choose the question and answer for this objective, not how they are batched."""
    # [N, 5] -> [N, 4] clues and [N] answers: remove the middle word, not a row.
    context = windows[:, [0, 1, 3, 4]]
    center = windows[:, 2]
    if objective == "CBOW":
        return context, center
    if objective == "Skip-gram":
        # [N] -> [4N] centers; [N, 4] -> [4N] answers in matching row-major order.
        # Repeating a center per neighbour changes the objective, not story weighting.
        return center.repeat_interleave(4), context.reshape(-1)
    if objective == "Encoder-decoder":
        # A fixed two-word continuation: no padding or easy EOS target in this comparison.
        bos = torch.full((len(windows), 1), bos_id, dtype=torch.long)
        # [N, 5] -> [N, 3] source; join [N, 1] start + [N, 2] continuation -> [N, 3].
        return windows[:, :3], torch.cat([bos, windows[:, 3:]], dim=1)
    raise ValueError(f"Unknown objective: {objective}")


def embedding_metrics(weights, probe_ids):
    normalized = torch.nn.functional.normalize(weights, dim=1)
    # [V, 48] -> [16, 48]: compare the named probe rows, not the full vocabulary.
    probe_vectors = normalized[probe_ids]
    # Transpose [16, 48] -> [48, 16]; [16, 48] @ [48, 16] compares every pair.
    similarities = probe_vectors @ probe_vectors.T
    labels = torch.arange(len(WORD_GROUPS)).repeat_interleave(4)
    # [16] -> [16, 1] and [1, 16]: broadcasting builds the pairwise group mask.
    same = labels[:, None] == labels[None, :]
    off_diagonal = ~torch.eye(len(probe_ids), dtype=torch.bool)
    within = similarities[same & off_diagonal].mean().item()
    across = similarities[~same].mean().item()
    neighbours = similarities.masked_fill(~off_diagonal, -float("inf")).topk(3).indices
    purity = (labels[neighbours] == labels[:, None]).float().mean().item()
    return within, across, purity


def train_comparison(corpus, cbow_class, skipgram_class, sequence_class, epochs=16,
                     seed=71, verbose=True):
    """Call the inherited fit loop one epoch at a time to observe its results."""
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    torch.manual_seed(seed)
    vocabulary = len(corpus["words"])
    width = 48
    initial_rows = torch.randn(vocabulary, width) * 0.1
    probe_words = list(itertools.chain.from_iterable(WORD_GROUPS.values()))
    probe_ids = torch.tensor([corpus["word_ids"][word] for word in probe_words])
    constructors = {
        "CBOW": lambda: cbow_class(vocabulary, width),
        "Skip-gram": lambda: skipgram_class(vocabulary, width),
        "Encoder-decoder": lambda: sequence_class(vocabulary + 1, width, 64),
    }
    results = {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for objective, constructor in constructors.items():
            torch.manual_seed(seed)
            model = constructor().cpu()
            with torch.no_grad():
                # [V+1, 48] -> first [V, 48]: leave the writer-only start row separate.
                model.embedding.weight[:vocabulary].copy_(initial_rows)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
            loss_function = nn.CrossEntropyLoss()
            train_x, train_y = objective_pairs(corpus["train"], objective, vocabulary)
            valid_x, valid_y = objective_pairs(corpus["valid"], objective, vocabulary)
            loader = DataLoader(
                TensorDataset(train_x, train_y), batch_size=256, shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            if verbose:
                print(f"\n{objective} data: {len(train_x):,} training pairs; "
                      f"inputs {tuple(train_x.shape)}, answers {tuple(train_y.shape)}; "
                      f"{len(valid_x):,} held-out pairs; batch size {loader.batch_size}",
                      flush=True)

            def evaluate(inputs, targets):
                total, count = 0.0, 0
                model.eval()
                with torch.inference_mode():
                    for start in range(0, len(inputs), 512):
                        # Aligned [N, ...] -> at most [512, ...] slices bound score memory.
                        x, y = inputs[start:start + 512], targets[start:start + 512]
                        expected = model.targets_for_loss(y)
                        loss = model.training_loss(x, y, loss_function)
                        total += loss.item() * expected.numel()
                        count += expected.numel()
                return total / count

            # Fixed train probe keeps the epoch-zero point comparable to later points.
            # Keep the same first at most 4096 pairs at every checkpoint for comparable loss.
            train_probe_x, train_probe_y = train_x[:4096], train_y[:4096]
            record = {
                "train_loss": [], "valid_loss": [], "snapshots": [],
                "within": [], "across": [], "purity": [], "epoch_fit_loss": [],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "targets_per_epoch": model.targets_for_loss(train_y).numel(),
            }
            started = time.perf_counter()
            for epoch in range(epochs + 1):
                if epoch:
                    loss = model.fit_language_model(loader, loss_function, optimizer, epochs=1)
                    record["epoch_fit_loss"].append(loss[0])
                # [V+1, 48] -> [V, 48] for the writer: exclude its extra <bos> row.
                weights = model.embedding.weight[:vocabulary].detach().clone()
                within, across, purity = embedding_metrics(weights, probe_ids)
                record["train_loss"].append(evaluate(train_probe_x, train_probe_y))
                record["valid_loss"].append(evaluate(valid_x, valid_y))
                record["snapshots"].append(weights)
                record["within"].append(within)
                record["across"].append(across)
                record["purity"].append(purity)
                if record["valid_loss"][-1] == min(record["valid_loss"]):
                    record["best_model"] = copy.deepcopy(model).eval()
                if verbose:
                    print(f"{objective:15s} epoch {epoch:2d}: "
                          f"train {record['train_loss'][-1]:.3f}; "
                          f"held out {record['valid_loss'][-1]:.3f}; "
                          f"group gap {within - across:+.3f}", flush=True)
            record["seconds"] = time.perf_counter() - started
            results[objective] = record
    finally:
        torch.set_num_threads(previous_threads)
    for name, record in results.items():
        assert record["train_loss"][-1] < record["train_loss"][0], name
        assert record["valid_loss"][-1] < record["valid_loss"][0], name
        assert torch.equal(record["snapshots"][0], initial_rows), name
        assert torch.isfinite(record["snapshots"][-1]).all(), name
        if verbose:
            print(f"{name}: {record['seconds']:.1f}s; {record['parameters']:,} parameters; "
                  f"{record['targets_per_epoch']:,} target tokens/epoch")
    return results


def _blank_chances(model, inputs):
    model.eval()
    with torch.no_grad():
        chances = model(inputs).softmax(dim=-1)
    chances[:, 0] = 0  # hide <unk> from the shortlist
    return chances


def blank_shortlist(model, corpus, left, right, top_k=5):
    """CBOW's top answers for `left ___ right`, given two words on each side."""
    words = left.split() + right.split()
    if len(words) != 4:
        raise ValueError("Give exactly two words on each side of the blank.")
    unknown = [word for word in words if word not in corpus["word_ids"]]
    if unknown:
        raise ValueError(f"Not in the vocabulary: {unknown}")
    ids = torch.tensor([[corpus["word_ids"][word] for word in words]])
    top = _blank_chances(model, ids)[0].topk(top_k)
    return [(corpus["words"][index], chance)
            for index, chance in zip(top.indices.tolist(), top.values.tolist())]


def shortlist_summary(model, corpus, top_k=5, seed=3):
    """Score every held-out CBOW blank; compare shortlist rows with random words' rows."""
    vocabulary = len(corpus["words"])
    inputs, answers = objective_pairs(corpus["valid"], "CBOW", vocabulary)
    top = _blank_chances(model, inputs).topk(top_k, dim=1)
    known = answers != 0
    rows = nn.functional.normalize(model.embedding.weight[:vocabulary].detach(), dim=1)
    others = ~torch.eye(top_k, dtype=torch.bool)

    def row_similarity(ids):
        # [V, 48] indexed by [B, K] -> [B, K, 48]: one shortlist's rows per example.
        vectors = rows[ids]
        # [B, K, 48] @ [B, 48, K] -> [B, K, K]; keep off-diagonal pairs per shortlist.
        return (vectors @ vectors.transpose(1, 2))[:, others].mean().item()

    random_ids = torch.randint(1, vocabulary, top.indices.shape,
                               generator=torch.Generator().manual_seed(seed))
    return {
        "blanks": int(known.sum()),
        "first_right": (top.indices[:, 0] == answers)[known].float().mean().item(),
        # [B] -> [B, 1]: compare each correct answer with its own K guesses, not other rows.
        "shortlist_right": (top.indices == answers[:, None]).any(dim=1)[known].float().mean().item(),
        "first_share": top.values[:, 0][known].mean().item(),
        "shortlist_similarity": row_similarity(top.indices[known]),
        "random_similarity": row_similarity(random_ids[known]),
    }


def best_versus_final(results):
    """Same recorded run, read at two moments: best held-out epoch and the last epoch."""
    summary = {}
    for name, record in results.items():
        best = int(np.argmin(record["valid_loss"]))
        gaps = np.array(record["within"]) - np.array(record["across"])
        summary[name] = {
            "best_epoch": best, "best_gap": gaps[best], "final_gap": gaps[-1],
            "best_purity": record["purity"][best], "final_purity": record["purity"][-1],
        }
        print(f"{name:15s} best held-out epoch {best:2d}: gap {gaps[best]:+.3f}, "
              f"purity {record['purity'][best]:.0%}  |  last epoch: gap {gaps[-1]:+.3f}, "
              f"purity {record['purity'][-1]:.0%}")
    return summary


def plot_losses_and_geometry(results):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, (name, record) in enumerate(results.items()):
        axis = axes[0, column]
        axis.plot(record["train_loss"], label="Fixed training probe", color="#007f86")
        axis.plot(record["valid_loss"], label="Held-out text", color="#bf5700")
        best = int(np.argmin(record["valid_loss"]))
        axis.scatter(best, record["valid_loss"][best], color="#bf5700", marker="*", s=100)
        axis.annotate(f"Best held out: {best}", (best, record["valid_loss"][best]),
                      xytext=(5, 12), textcoords="offset points", fontsize=8)
        axis.set(title=name, xlabel="Epoch (0 = random)", ylabel="Loss per target word")
        axis.legend(frameon=False, fontsize=8)
        axis = axes[1, column]
        axis.plot(record["within"], label="Within named groups", color="#007f86")
        axis.plot(record["across"], label="Across those groups", color="#7950a3")
        axis.set(xlabel="Epoch", ylabel="Mean cosine in all 48 dimensions", ylim=(-0.2, 1))
        axis.legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Same text and starting rows; three different practice questions\n"
                 "Compare each loss with its own start, not with another objective")
    fig.tight_layout()
    return fig


def plot_similarity_heatmaps(corpus, results):
    """Show full-dimensional cosine, with identical scales and probe ordering."""
    words = list(itertools.chain.from_iterable(WORD_GROUPS.values()))
    ids = torch.tensor([corpus["word_ids"][word] for word in words])
    first = next(iter(results.values()))["snapshots"][0]
    panels = {"Same random start": first}
    panels.update({name: record["snapshots"][-1] for name, record in results.items()})
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), layout="constrained")
    for axis, (name, weights) in zip(axes, panels.items()):
        vectors = torch.nn.functional.normalize(weights[ids], dim=1)
        # [16, 48] @ transposed [48, 16] -> [16, 16]: all full-row cosine pairs.
        similarities = vectors @ vectors.T
        similarities.fill_diagonal_(float("nan"))
        image = axis.imshow(similarities.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        axis.set(title=name, xticks=range(len(words)), yticks=range(len(words)))
        axis.set_xticklabels(words, rotation=90, fontsize=8)
        axis.set_yticklabels(words, fontsize=8)
        for boundary in (3.5, 7.5, 11.5):
            axis.axhline(boundary, color="#222222", linewidth=0.6)
            axis.axvline(boundary, color="#222222", linewidth=0.6)
    fig.colorbar(image, ax=list(axes), shrink=0.55, label="Cosine in all 48 dimensions")
    fig.suptitle("Do the same-color words really draw together?\n"
                 "Look for warmer within-group blocks, not a uniformly warm whole grid.")
    return fig


def project_snapshots(corpus, results):
    probe_words = list(itertools.chain.from_iterable(WORD_GROUPS.values()))
    ids = torch.tensor([corpus["word_ids"][word] for word in probe_words])
    trajectories = []
    for record in results.values():
        # Epoch list of [16, 48] -> [epochs+1, 16, 48]: preserve the measured trajectory.
        trajectories.append(torch.stack([
            torch.nn.functional.normalize(weights[ids], dim=1)
            for weights in record["snapshots"]
        ]).numpy())
    # Three [E, 16, 48] trajectories -> [3, E, 16, 48]: keep objective and time axes.
    values = np.stack(trajectories)
    # [3, E, 16, 48] -> [3*E*16, 48]: fit ONE camera to every recorded point.
    all_points = values.reshape(-1, values.shape[-1])
    center = all_points.mean(axis=0)
    _, singular_values, directions = np.linalg.svd(all_points - center, full_matrices=False)
    # [2, 48] -> [48, 2]: project each 48-number row to two axes.
    # Leading axes survive: [3, E, 16, 48] -> [3, E, 16, 2].
    coordinates = (values - center) @ directions[:2].T
    explained = (singular_values[:2] ** 2).sum() / (singular_values ** 2).sum()
    print(f"Shared 2D PCA keeps {explained:.1%} of this probe trajectory's variance.")
    print("The cosine diagnostics use all 48 dimensions; the map is only a sketch.")
    return probe_words, coordinates, explained


def draw_map(axis, points, words, title, limits):
    for group, color in zip(WORD_GROUPS, COLORS):
        indices = [words.index(word) for word in WORD_GROUPS[group]]
        axis.scatter(points[indices, 0], points[indices, 1], c=color, label=group, s=35)
        for index in indices:
            axis.annotate(words[index], points[index], xytext=(3, 3),
                          textcoords="offset points", fontsize=8)
    axis.set(title=title, xlim=limits[0], ylim=limits[1],
             xlabel="Shared PCA direction 1", ylabel="Shared PCA direction 2")
    axis.grid(alpha=0.15)


def plot_word_maps(words, coordinates, results):
    limits = [(coordinates[..., dim].min() - 0.12, coordinates[..., dim].max() + 0.12)
              for dim in range(2)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for column, name in enumerate(results):
        for row, epoch in enumerate((0, coordinates.shape[1] - 1)):
            draw_map(axes[row, column], coordinates[column, epoch], words,
                     f"{name}: epoch {epoch}", limits)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Starting rows above; trained rows below - one frozen PCA view\n"
                 "Colors are human-picked word groups, not labels supplied to training")
    fig.tight_layout()
    return fig


def animate_word_maps(words, coordinates, results):
    """Animate only recorded states; the static before/after remains the fallback."""
    limits = [(coordinates[..., dim].min() - 0.12, coordinates[..., dim].max() + 0.12)
              for dim in range(2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    def update(epoch):
        for column, (axis, name) in enumerate(zip(axes, results)):
            axis.clear()
            draw_map(axis, coordinates[column, epoch], words, f"{name}: epoch {epoch}", limits)
        fig.tight_layout()

    animation = FuncAnimation(
        fig, update, frames=coordinates.shape[1], interval=500, repeat=False,
    )
    plt.close(fig)
    return animation


def report_neighbours(corpus, results):
    words, word_ids = corpus["words"], corpus["word_ids"]
    probe_words = list(itertools.chain.from_iterable(WORD_GROUPS.values()))
    probe_ids = torch.tensor([word_ids[word] for word in probe_words])
    print("Full-vocabulary neighbours (not restricted to the colored words):")
    for name, record in results.items():
        print(f"\n{name}: group-neighbour purity "
              f"{record['purity'][0]:.1%} -> {record['purity'][-1]:.1%} "
              "(random among these probes: 20%)")
        best = int(np.argmin(record["valid_loss"]))
        print(f"  Best held-out epoch: {best}; "
              f"loss {record['valid_loss'][best]:.3f}; "
              f"final {record['valid_loss'][-1]:.3f}. "
              "Final maps show the fixed training budget, not a selected checkpoint.")
        for word in ("dad", "cat", "blue", "sad"):
            rows = []
            for weights in (record["snapshots"][0], record["snapshots"][-1]):
                normalized = torch.nn.functional.normalize(weights, dim=1)
                scores = normalized @ normalized[word_ids[word]]
                scores[[0, word_ids[word]]] = -float("inf")
                rows.append(", ".join(words[index] for index in scores.topk(5).indices))
            print(f"  {word:8s} before: {rows[0]}; after: {rows[1]}")
        _, _, purity = embedding_metrics(record["snapshots"][-1], probe_ids)
        gap = record["within"][-1] - record["across"][-1]
        if gap <= 0 or purity <= 0.2:
            print("  These probes did not show clear group separation. Don't claim convergence.")
    print("\nNearby rows can share context without sharing meaning. "
          "This is one corpus and one seed, not a semantic-quality benchmark.")
