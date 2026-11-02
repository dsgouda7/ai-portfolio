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
from matplotlib.patches import FancyBboxPatch
from torch import nn
from numpy.lib.stride_tricks import sliding_window_view

from language_data import CBOWDataLoader, SkipGramDataLoader, WindowDataLoader
from language_validation import ModelValidator

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


def prepare_validation_windows(
    stoi: dict[str, int], training_stories: int, *,
    reserved_stories: int = 200, window_width: int = 5, max_windows: int = 2_048,
) -> torch.Tensor:
    if training_stories < 1 or reserved_stories < 1 or max_windows < 1:
        raise ValueError("Validation story and window budgets must be positive.")
    if "<unk>" not in stoi:
        raise ValueError("Validation needs the training vocabulary's unknown-word ID.")
    stories = read_stories(training_stories + reserved_stories)[training_stories:]
    if len(stories) != reserved_stories:
        raise ValueError("The corpus does not contain the requested reserved stories.")
    documents = [sentence for story in stories for sentence in story_sentences(story)]
    ids = [[stoi.get(word, stoi["<unk>"]) for word in document] for document in documents]
    windows = torch.tensor(make_document_windows(ids, window_width)[:max_windows], dtype=torch.long)
    if len(windows) == 0:
        raise ValueError("Reserved stories must supply at least one complete validation window.")
    print(f"Reserved stories: {training_stories + 1} through "
          f"{training_stories + reserved_stories}; windows: {len(windows)}")
    print(f"Reserved unknown slots: {windows.eq(stoi['<unk>']).float().mean():.1%}")
    return windows


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
    if objective == "CBOW":
        return CBOWDataLoader(windows).prepare_data()
    if objective == "Skip-gram":
        return SkipGramDataLoader(windows).prepare_data()
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
            loader = WindowDataLoader.batch_tensors(train_x, train_y, batch_size=256, seed=seed)
            if verbose:
                print(f"\n{objective} data: {len(train_x):,} training pairs; "
                      f"inputs {tuple(train_x.shape)}, answers {tuple(train_y.shape)}; "
                      f"{len(valid_x):,} held-out pairs; batch size {loader.batch_size}",
                      flush=True)

            # Fixed train probe keeps the epoch-zero point comparable to later points.
            # Keep the same first at most 4096 pairs at every checkpoint for comparable loss.
            train_probe_x, train_probe_y = train_x[:4096], train_y[:4096]
            train_validator = ModelValidator(train_probe_x, train_probe_y)
            valid_validator = ModelValidator(valid_x, valid_y)
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
                record["train_loss"].append(train_validator.evaluate(model).loss)
                record["valid_loss"].append(valid_validator.evaluate(model).loss)
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


def record_embedding_routing(model, contexts, targets, learning_rate=0.1):
    """Observe the actual CBOW graph on a copy, including mean and lookup gradients."""
    if contexts.ndim != 2 or contexts.shape[0] != 1 or contexts.shape[1] == 0:
        raise ValueError("Routing visualization requires one nonempty context window.")
    if targets.shape != (1,):
        raise ValueError("Routing visualization requires one checking answer.")
    if learning_rate <= 0:
        raise ValueError("Routing learning rate must be positive.")
    candidate = copy.deepcopy(model).cpu()
    contexts = contexts.detach().cpu().clone()
    targets = targets.detach().cpu().clone()
    optimizer = torch.optim.SGD(candidate.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    activations = {}

    def remember(name, tensor):
        tensor.retain_grad()
        activations[name] = tensor

    def capture_output(name):
        def capture(module, inputs, output):
            remember(name, output)
        return capture

    def capture_mean(module, inputs):
        remember("mean", inputs[0])

    hooks = [
        candidate.embedding.register_forward_hook(capture_output("lookup")),
        candidate.hidden.register_forward_pre_hook(capture_mean),
        candidate.hidden.register_forward_hook(capture_output("linear")),
        candidate.activation.register_forward_hook(capture_output("hidden")),
        candidate.output.register_forward_hook(capture_output("logits")),
    ]
    weights = [candidate.embedding.weight.detach().clone()]
    try:
        logits = candidate(contexts)
        weights.append(candidate.embedding.weight.detach().clone())
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        weights.append(candidate.embedding.weight.detach().clone())
        gradient = candidate.embedding.weight.grad.detach().clone()
        head_gradient = candidate.output.bias.grad.detach().clone()
        if any(value.grad is None for value in activations.values()):
            raise RuntimeError("The recorded CBOW graph did not connect every activation to loss.")
        values = {name: value.detach().clone() for name, value in activations.items()}
        gradients = {name: value.grad.detach().clone() for name, value in activations.items()}
        probabilities = values["logits"].softmax(dim=-1)
        expected_logit_gradient = probabilities.clone()
        expected_logit_gradient[0, targets[0]] -= 1
        torch.testing.assert_close(gradients["logits"], expected_logit_gradient)
        torch.testing.assert_close(
            gradients["linear"], gradients["hidden"] * values["linear"].gt(0),
        )
        torch.testing.assert_close(
            gradients["mean"], gradients["linear"] @ candidate.hidden.weight.detach(),
        )
        optimizer.step()
        weights.append(candidate.embedding.weight.detach().clone())
    finally:
        for hook in hooks:
            hook.remove()

    count = contexts.shape[1]
    expected_occurrences = gradients["mean"][:, None, :].expand_as(values["lookup"]) / count
    torch.testing.assert_close(gradients["lookup"], expected_occurrences)
    reconstructed = torch.zeros_like(gradient)
    reconstructed.index_add_(
        0, contexts.flatten(), gradients["lookup"].reshape(-1, gradient.shape[1]),
    )
    torch.testing.assert_close(reconstructed, gradient)
    assert torch.equal(weights[0], weights[1]) and torch.equal(weights[0], weights[2])
    selected = set(contexts.flatten().tolist())
    assert set(gradient.ne(0).any(dim=1).nonzero().flatten().tolist()) <= selected
    assert set(weights[3].ne(weights[0]).any(dim=1).nonzero().flatten().tolist()) <= selected
    return {
        "contexts": contexts, "targets": targets, "saved_ids": contexts.clone(),
        "mean_count": count, "weights": weights, "gradient": gradient,
        "reconstructed_gradient": reconstructed, "head_gradient": head_gradient,
        "activations": values, "activation_gradients": gradients,
        "probabilities": probabilities, "loss": loss.item(), "learning_rate": learning_rate,
    }


def _routing_visible_ids(trace):
    visible = list(dict.fromkeys(
        trace["contexts"].flatten().tolist() + trace["targets"].tolist()
    ))
    unused = [index for index in range(len(trace["weights"][0])) if index not in visible]
    return visible + unused[:1]


def embedding_routing_steps(trace):
    """Keep every displayed input/output tied to an actual recorded tensor."""
    values, gradients = trace["activations"], trace["activation_gradients"]
    loss = torch.tensor(trace["loss"])
    return (
        ("Pick the word rows", 0, "forward", trace["saved_ids"], values["lookup"]),
        (f"Average the {trace['mean_count']} vectors", 1, "forward", values["lookup"], values["mean"]),
        ("Mix the features", 2, "forward", values["mean"], values["linear"]),
        ("ReLU: negatives become zero", 3, "forward", values["linear"], values["hidden"]),
        ("Score every vocabulary word", 4, "forward", values["hidden"], values["logits"]),
        ("Check the answer: cross entropy", 5, "forward", values["logits"], loss),
        ("Loss becomes score gradients", 5, "backward", loss, gradients["logits"]),
        ("Back through the vocabulary head", 4, "backward", gradients["logits"], gradients["hidden"]),
        ("Back through ReLU", 3, "backward", gradients["hidden"], gradients["linear"]),
        ("Back through the first linear layer", 2, "backward", gradients["linear"], gradients["mean"]),
        (f"Split the gradient {trace['mean_count']} ways", 1, "backward", gradients["mean"], gradients["lookup"]),
        ("Saved IDs send gradients home", 0, "backward", gradients["lookup"], trace["gradient"]),
        ("SGD: now the weights change", 0, "update", trace["weights"][0], trace["weights"][3]),
    )


_ROUTING_COLORS = {
    "background": "#0b1220", "panel": "#142238", "text": "#e6edf3",
    "muted": "#91a4ba", "dim": "#293b52", "forward": "#38d9e8",
    "backward": "#ffad5c", "update": "#69e6ab", "target": "#ffd166",
}


def _routing_vector(values, precision=3):
    return "[" + ", ".join(f"{value:.{precision}g}" for value in values.flatten()[:3].tolist()) + "]"


def draw_embedding_routing(axis, trace, words, stage=11, progress=1.0):
    """One pipeline, a persistent vocabulary table, and one large active-layer readout."""
    steps = embedding_routing_steps(trace)
    if not 0 <= stage < len(steps) or not 0 <= progress <= 1:
        raise ValueError("Routing stage or animation progress is outside its valid range.")
    title, node, direction, incoming, outgoing = steps[stage]
    palette = _ROUTING_COLORS
    color = palette[direction]
    axis.clear()
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_axis_off()
    axis.figure.set_facecolor(palette["background"])

    def text(x, y, value, size=14, shade=None, **kwargs):
        return axis.text(
            x, y, value, fontsize=size, color=shade or palette["text"],
            transform=axis.transAxes, **kwargs,
        )

    def box(x, y, width, height, edge, fill=None, linewidth=1.5):
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.008",
            facecolor=fill or palette["panel"], edgecolor=edge, linewidth=linewidth,
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        return patch

    ids = trace["saved_ids"][0].tolist()
    selected = set(ids)
    visible = _routing_visible_ids(trace)
    target = trace["targets"][0].item()
    values, gradients = trace["activations"], trace["activation_gradients"]
    vocabulary, width = trace["weights"][0].shape
    hidden_width = values["hidden"].shape[-1]
    clues = [words[index] for index in ids]
    gap = len(clues) // 2
    text(0.025, 0.955, " ".join(clues[:gap] + ["___"] + clues[gap:]), 25, weight="bold")
    text(0.975, 0.958, f"answer: {words[target]} [{target}]", 17,
         palette["target"], ha="right")
    centers = (0.09, 0.255, 0.42, 0.575, 0.74, 0.91)
    labels = ("LOOKUP", "MEAN", "LINEAR", "ReLU", "LOGITS", "CE LOSS")
    shapes = (
        f"{vocabulary} x {width}", f"{trace['mean_count']} vectors -> 1",
        f"{width} -> {hidden_width}", f"{hidden_width} -> {hidden_width}",
        f"{hidden_width} -> {vocabulary}", "scores + answer",
    )
    for index, center in enumerate(centers):
        active = index == node
        pulse = np.sin(np.pi * progress) if active else 0
        block_width = 0.12 + (0.012 + 0.008 * pulse if active else 0)
        block_height = 0.1 + (0.018 if active else 0)
        box(center - block_width / 2, 0.79 - block_height / 2, block_width,
            block_height, color if active else palette["dim"],
            linewidth=3 + pulse if active else 1.5)
        text(center, 0.803, labels[index], 14 if active else 12,
             color if active else palette["muted"], ha="center", weight="bold")
        text(center, 0.765, shapes[index], 11, palette["muted"], ha="center")
        if index:
            axis.annotate(
                "", xy=(center - 0.072, 0.79), xytext=(centers[index - 1] + 0.072, 0.79),
                arrowprops={"arrowstyle": "->", "color": palette["dim"], "lw": 2},
            )
    if stage not in (0, 12):
        previous_node = steps[stage - 1][1]
        start, end = centers[previous_node], centers[node]
        if start == end:
            packet_x, packet_y = end, 0.9
        else:
            packet_x = start + (end - start) * progress
            packet_y = 0.9
        if incoming.shape == values["logits"].shape:
            sample = incoming[0, target].item()
        else:
            sample = incoming.item() if incoming.numel() == 1 else incoming.flatten()[0].item()
        text(packet_x, packet_y, f"{sample:.4g}", 15, palette["background"], ha="center",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": color})
        if start != end:
            axis.annotate(
                "", xy=(end, 0.865), xytext=(start, 0.865),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 3},
            )
    if node == 5:
        axis.annotate(
            "", xy=(centers[5], 0.85), xytext=(0.965, 0.935),
            arrowprops={"arrowstyle": "->", "color": palette["target"], "lw": 2},
        )

    text(0.025, 0.685, f"{direction.upper()}  {stage + 1}/{len(steps)}", 14, color, weight="bold")
    text(0.38, 0.685, title, 22, color, weight="bold")
    box(0.025, 0.115, 0.305, 0.505, palette["dim"])
    box(0.38, 0.115, 0.595, 0.505, color, linewidth=2)
    table_mode = "GRADIENTS" if stage == 11 else "WEIGHTS"
    text(0.04, 0.58, f"EMBEDDING {table_mode}", 13, palette["muted"], weight="bold")
    table_values = trace["gradient"] if stage == 11 else trace["weights"][3 if stage == 12 else 0]
    row_y = {}
    for index, row in enumerate(visible):
        y = 0.525 - index * 0.058
        row_y[row] = y
        highlighted = row in selected
        if stage == 0:
            highlighted = row in set(ids[:min(len(ids), 1 + int(progress * len(ids)))])
        row_color = color if highlighted else palette["muted"]
        if highlighted:
            box(0.035, y - 0.017, 0.285, 0.039, row_color,
                fill=palette["panel"], linewidth=2)
        text(0.045, y, f"{words[row]} [{row}]", 11, row_color, va="center")
        formatted = "[" + ", ".join(f"{value:.6f}" for value in table_values[row, :3].tolist()) + "]"
        text(0.157, y, formatted, 10, row_color, va="center", family="DejaVu Sans Mono")
    unread = vocabulary - len(selected)
    text(0.04, 0.14, f"+ {unread} unread rows", 12, palette["muted"])

    logit_ids = list(dict.fromkeys([target] + values["logits"][0].topk(min(3, vocabulary)).indices.tolist()))
    precision = 6 if direction == "update" else 3

    def readout(x, label, tensor, kind):
        text(x, 0.565, label, 13, palette["muted"], weight="bold")
        if kind == "loss":
            text(x, 0.39, f"{tensor.item():.4f}", 38, color, family="DejaVu Sans Mono")
        elif kind == "words":
            for index, word_id in enumerate(logit_ids):
                y = 0.485 - index * 0.076
                shade = palette["target"] if word_id == target else palette["text"]
                text(x, y, f"{words[word_id]} [{word_id}]", 12, shade)
                text(x + 0.12, y, f"{tensor[0, word_id].item():+.5f}", 18,
                     shade, family="DejaVu Sans Mono")
        elif kind == "ids":
            for index, row in enumerate(ids):
                text(x, 0.48 - index * 0.076, f"{words[row]}  [{row}]", 18, color)
        elif kind in {"occurrences", "rows"}:
            rows = ids if kind == "occurrences" else list(dict.fromkeys(ids))
            tensors = tensor[0] if kind == "occurrences" else tensor[rows]
            for index, (row, vector) in enumerate(zip(rows, tensors)):
                y = 0.49 - index * 0.083
                text(x, y, f"{words[row]} [{row}]", 10, palette["muted"])
                text(x, y - 0.034, _routing_vector(vector, precision), 14, color,
                     family="DejaVu Sans Mono")
        else:
            for index, value in enumerate(tensor.flatten()[:3]):
                text(x, 0.47 - index * 0.095, f"f{index}  {value.item():+.6f}",
                     21, color, family="DejaVu Sans Mono")

    input_kind, output_kind = "features", "features"
    if stage == 0:
        input_kind, output_kind = "ids", "occurrences"
    elif stage == 1:
        input_kind = "occurrences"
    elif stage == 4:
        output_kind = "words"
    elif stage == 5:
        input_kind, output_kind = "words", "loss"
    elif stage == 6:
        input_kind, output_kind = "loss", "words"
    elif stage == 7:
        input_kind = "words"
    elif stage == 10:
        output_kind = "occurrences"
    elif stage == 11:
        input_kind, output_kind = "occurrences", "rows"
    elif stage == 12:
        input_kind = output_kind = "rows"
    in_label, out_label = ("IN", "OUT") if direction == "forward" else ("GRADIENT IN", "GRADIENT OUT")
    if stage == 12:
        in_label, out_label = "BEFORE", "AFTER"
    readout(0.405, in_label, incoming, input_kind)
    readout(0.72, out_label, outgoing, output_kind)
    text(0.68, 0.39, ">", 28, color, ha="center", weight="bold")
    if stage == 5:
        probability = trace["probabilities"][0, target].item()
        text(0.405, 0.15, f"p({words[target]}) = {probability:.6f}    loss = -log p({words[target]})", 13, color)
    elif stage == 6:
        text(0.405, 0.15, "cross-entropy gradient = probability - target", 13, color)
    elif stage == 10:
        text(0.405, 0.15, f"each occurrence receives gradient / {trace['mean_count']}", 14, color)
    elif stage == 11:
        text(0.405, 0.15, "saved addresses route the gradient; weights are still unchanged", 12, color)
    elif stage == 12:
        text(0.405, 0.15, f"one SGD step: weight - {trace['learning_rate']:g} x gradient", 14, color)
    elif stage in (3, 8):
        text(0.405, 0.15, "negative forward inputs block this route", 13, palette["muted"])
    text(0.025, 0.065, "SAVED IDS", 11, palette["muted"])
    for index, row in enumerate(ids):
        x = 0.12 + index * 0.105
        text(x, 0.065, str(row), 14, color if stage in (0, 11) else palette["muted"],
             bbox={"boxstyle": "round,pad=0.25", "facecolor": palette["panel"],
                   "edgecolor": color if stage in (0, 11) else palette["dim"]})
    text(0.975, 0.065, "3 features / selected scores shown; full tensors used",
         11, palette["muted"], ha="right")
    if stage in (0, 11):
        row = ids[min(len(ids) - 1, int(progress * len(ids)))]
        if row in row_y:
            axis.annotate(
                "", xy=(0.325, row_y[row]), xytext=(0.37, 0.36),
                arrowprops={"arrowstyle": "->" if stage == 11 else "<-",
                            "color": color, "lw": 3},
            )
    axis.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)


def plot_embedding_routing(trace, words):
    """A compact static checkpoint of the same pipeline at lookup backward."""
    figure, axis = plt.subplots(figsize=(18, 8))
    draw_embedding_routing(axis, trace, words)
    return figure


def animate_embedding_routing(trace, words):
    """Move recorded values forward, then gradients backward, through one cat-probe pass."""
    figure, axis = plt.subplots(figsize=(18, 8))
    frames_per_step = 5

    def update(frame):
        draw_embedding_routing(
            axis, trace, words, stage=frame // frames_per_step,
            progress=(frame % frames_per_step) / (frames_per_step - 1),
        )

    animation = FuncAnimation(
        figure, update, frames=len(embedding_routing_steps(trace)) * frames_per_step,
        interval=500, repeat=False,
    )
    plt.close(figure)
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
