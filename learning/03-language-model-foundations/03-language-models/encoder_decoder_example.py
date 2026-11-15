"""Frozen cat-family sources, aligned practice batches and bounded generation."""

import hashlib
import json
from pathlib import Path
import random

import torch

from language_data import CAT_FAMILY_WORDS, CAT_WORDS


VOCABULARY = ("<pad>", *CAT_FAMILY_WORDS, "<bos>", "<eos>")
PAD, BOS, EOS = 0, len(VOCABULARY) - 2, len(VOCABULARY) - 1
_WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
_CORPUS_SHA256 = "b76c7b13b19a48420be4dc2d23ad016d3c404ea58d4d37d8ba18c81d749074f5"


def encode_words(words) -> list[int]:
    return [_WORD_TO_ID[word] for word in words]


def decode_ids(ids) -> list[str]:
    return [VOCABULARY[int(index)] for index in ids]


def teacher_inputs(target_ids: torch.Tensor) -> torch.Tensor:
    """Prepend BOS and remove the last answer; never feed a current target to itself."""
    if not isinstance(target_ids, torch.Tensor) or target_ids.dtype != torch.long:
        raise TypeError("Answers must be a torch.long tensor.")
    if target_ids.ndim != 2 or not target_ids.shape[0] or not target_ids.shape[1]:
        raise ValueError("Answers must be a nonempty [batch, answer time] tensor.")
    return torch.cat(
        [target_ids.new_full((len(target_ids), 1), BOS), target_ids[:, :-1]], dim=1,
    )


def generate(model, words=CAT_WORDS):
    """Keep the former main-notebook generation example available without target inputs."""
    source = torch.tensor([encode_words(words)])
    chosen = []
    model.eval()
    with torch.no_grad():
        state = model.encode(source)
        previous = source.new_full((1, 1), BOS)
        for _ in range(len(words) + 1):
            logits, state = model.decode(previous, state)
            previous = logits[:, -1].argmax(-1, keepdim=True)
            chosen.append(previous.item())
            if chosen[-1] == EOS:
                break
    return decode_ids(chosen)


def load_cat_corpus() -> dict[str, dict[int, torch.Tensor]]:
    """Load only the original seed-31 split, with no runtime source generation."""
    path = Path(__file__).with_name("data") / "03c-cat-reversal-corpus.json"
    payload = json.loads(path.read_bytes())
    canonical = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != _CORPUS_SHA256:
        raise ValueError("The fixed cat-family corpus fingerprint has changed.")
    if set(payload) != {"train", "test"}:
        raise ValueError("The corpus must contain train and test splits.")
    corpus = {name: {} for name in payload}
    for name, split in payload.items():
        if set(split) != {str(length) for length in range(2, 9)}:
            raise ValueError("Each split must contain source lengths two through eight.")
        for key, rows in split.items():
            length = int(key)
            expected = 40 if name == "test" else min(1200, len(CAT_FAMILY_WORDS) ** length - 40)
            tensor = torch.tensor(rows, dtype=torch.long)
            if tensor.shape != (expected, length) or not bool(((tensor > PAD) & (tensor < BOS)).all()):
                raise ValueError("Sources must contain only word IDs in the fixed length buckets.")
            if len({tuple(row) for row in rows}) != expected:
                raise ValueError("Source orders must be unique within each split.")
            corpus[name][length] = tensor
    for length in corpus["train"]:
        training = {tuple(row) for row in corpus["train"][length].tolist()}
        held_out = {tuple(row) for row in corpus["test"][length].tolist()}
        if not training.isdisjoint(held_out):
            raise ValueError("A held-out source occurs in the training split.")
    return corpus


def training_batches(corpus, updates=1000):
    """Replay the original seed-99 length schedule and 64-source index batches."""
    if not isinstance(updates, int) or updates < 0:
        raise ValueError("Updates must be a nonnegative integer.")
    schedule = random.Random(99)
    batch_rng = torch.Generator().manual_seed(99)
    for _ in range(updates):
        length = schedule.randint(2, 8)
        rows = corpus["train"][length]
        source = rows[torch.randint(len(rows), (64,), generator=batch_rng)]
        target = torch.cat([source.flip(1), source.new_full((len(source), 1), EOS)], dim=1)
        yield source, teacher_inputs(target), target
