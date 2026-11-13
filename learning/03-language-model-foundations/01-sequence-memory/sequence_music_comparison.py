"""Original, offline next-note learning for the three generic 01A readers.

The three class definitions need only ``from torch import nn`` when copied into
a notebook. They accept feature vectors, not music-specific token IDs. The
application below supplies one-hot note features and next-note targets.

Notebook application (repeat with the inline LSTM and GRU classes):

    from IPython.display import HTML, display
    from sequence_music_comparison import (
        build_music_dataset, run_music_experiment,
        print_music_report, music_comparison_html,
    )
    music_data = build_music_dataset()
    rnn_music = run_music_experiment(RNNSequenceModel, music_data, kind="RNN")
    print_music_report(rnn_music)
    display(HTML(music_comparison_html(rnn_music)))

Passing the inline class, rather than a model-name lookup, makes reader edits
affect the actual fit. No downloads, plotting packages, or audio packages are
used. PyTorch is the only non-standard-library dependency.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from html import escape
from io import BytesIO
import itertools
import json
import math
import random
import struct
import time
import wave

import torch
from torch import nn
from torch.nn import functional as F


class RNNSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # One hidden state is rewritten each step, without gates.
        self.recurrent = nn.RNN(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        hidden_path, state = self.recurrent(x, state)
        note_scores = self.head(hidden_path)
        return note_scores, state


class LSTMSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # State is (hidden, cell); gates control what to keep, write, and read.
        self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        hidden_path, state = self.recurrent(x, state)
        note_scores = self.head(hidden_path)
        return note_scores, state


class GRUSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # One hidden state; reset/update gates control old and new content.
        self.recurrent = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, state=None):
        hidden_path, state = self.recurrent(x, state)
        note_scores = self.head(hidden_path)
        return note_scores, state


# Like 01C, these are authored minor-key chord tones, not a borrowed tune.
CHORD_ROOTS = (50, 46, 41, 48)  # D minor, B-flat major, F major, C major
CHORD_INTERVALS = (
    (0, 3, 7, 12, 15, 19),
    (0, 4, 7, 12, 16, 19),
    (0, 4, 7, 12, 16, 19),
    (0, 4, 7, 12, 16, 19),
)
PROGRESSIONS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (0, 1, 3, 0))
CONTOURS = (
    (0, 2, 3, 4, 5, 4, 3, 2),
    (5, 4, 3, 2, 1, 2, 3, 4),
    (0, 3, 2, 4, 1, 3, 2, 0),
    (0, 1, 2, 3, 2, 1, 2, 0),
    (3, 2, 4, 3, 5, 4, 2, 3),
    (0, 2, 1, 3, 2, 4, 3, 2),
)
VOCABULARY = (None,) + tuple(sorted({
    root + interval + register
    for root, intervals in zip(CHORD_ROOTS, CHORD_INTERVALS)
    for interval in intervals
    for register in (0, 12)
}))


@dataclass(frozen=True)
class MusicDataset:
    """Whole phrases, split before any input/target pairs are constructed."""

    vocabulary: tuple[int | None, ...]
    train_phrases: tuple[tuple[int, ...], ...]
    heldout_phrases: tuple[tuple[int, ...], ...]
    split_seed: int

    @property
    def fingerprint(self) -> str:
        content = (self.vocabulary, self.train_phrases, self.heldout_phrases)
        return hashlib.sha256(json.dumps(content).encode("utf-8")).hexdigest()


def build_music_dataset(*, split_seed: int = 17) -> MusicDataset:
    """Compose 384 unique 32-event phrases; hold out 20% of complete phrases.

    Variation: four chord progressions, six melodic contours, four contour
    rotations, two registers, and either continuous notes or bar-ending rests.
    Templates and chord tones are shared across splits on purpose; complete
    phrases are not. This tests unseen combinations, not unseen musical styles.
    The fixed vocabulary is specified by the composition rules, not inferred
    from held-out examples. Each event lasts the same amount of time.
    """
    token_for = {pitch: token for token, pitch in enumerate(VOCABULARY)}
    phrases = set()
    for progression, contour, rotation, register, rests in itertools.product(
        PROGRESSIONS, CONTOURS, (0, 2, 4, 6), (0, 12), (False, True)
    ):
        phrase = []
        pattern = contour[rotation:] + contour[:rotation]
        for chord in progression:
            pitches = [
                CHORD_ROOTS[chord] + interval + register
                for interval in CHORD_INTERVALS[chord]
            ]
            phrase.extend(
                token_for[None if rests and beat == 7 else pitches[index]]
                for beat, index in enumerate(pattern)
            )
        phrases.add(tuple(phrase))
    ordered = sorted(phrases)
    random.Random(split_seed).shuffle(ordered)
    heldout_count = max(1, round(len(ordered) * 0.2))
    return MusicDataset(
        VOCABULARY, tuple(ordered[heldout_count:]),
        tuple(ordered[:heldout_count]), split_seed,
    )


def make_next_note_batch(
    phrases: Sequence[Sequence[int]], vocabulary_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x[:, t] is note t; y[:, t] is note t+1, never the current note."""
    tokens = torch.tensor(phrases, dtype=torch.long, device="cpu")
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] < 2:
        raise ValueError("Provide nonempty, equal-length phrases of at least two events.")
    if bool(((tokens < 0) | (tokens >= vocabulary_size)).any()):
        raise ValueError("A phrase contains a token outside the fixed vocabulary.")
    x = F.one_hot(tokens[:, :-1], num_classes=vocabulary_size).float()
    return x, tokens[:, 1:]


def _validate_dataset(dataset: MusicDataset) -> None:
    if not dataset.train_phrases or not dataset.heldout_phrases:
        raise ValueError("Both the train and held-out phrase splits must be nonempty.")
    if set(dataset.train_phrases) & set(dataset.heldout_phrases):
        raise ValueError("Train and held-out phrases must be disjoint.")
    lengths = {len(phrase) for phrase in (*dataset.train_phrases, *dataset.heldout_phrases)}
    if len(lengths) != 1:
        raise ValueError("Use the same phrase length throughout the corpus.")
    if len(set(dataset.vocabulary)) != len(dataset.vocabulary):
        raise ValueError("Vocabulary entries must be unique.")


def _score(model: nn.Module, x: torch.Tensor, targets: torch.Tensor) -> dict:
    model.eval()
    with torch.inference_mode():
        scores, _ = model(x)
        loss = F.cross_entropy(scores.flatten(0, 1), targets.flatten()).item()
        accuracy = (scores.argmax(-1) == targets).float().mean().item()
    return {"loss": loss, "accuracy": accuracy}


def _continue(
    model: nn.Module, prompt: Sequence[int], steps: int, vocabulary_size: int,
) -> list[int]:
    """Feed the prompt once, then carry state and feed only our own last guess."""
    model.eval()
    generated = []
    with torch.inference_mode():
        x = F.one_hot(torch.tensor([prompt]), num_classes=vocabulary_size).float()
        scores, state = model(x)
        for step in range(steps):
            token = int(scores[0, -1].argmax().item())
            generated.append(token)
            if step + 1 < steps:
                x = F.one_hot(torch.tensor([[token]]), num_classes=vocabulary_size).float()
                scores, state = model(x, state)
    return generated


def _example_predictions(model, x, prompt, steps, vocabulary_size):
    with torch.inference_mode():
        scores, _ = model(x)
        teacher_fed = scores[0].argmax(-1).tolist()
    return {
        "teacher_fed_next_tokens": teacher_fed,
        "generated_continuation": _continue(model, prompt, steps, vocabulary_size),
    }


def run_music_experiment(
    model_factory: Callable[..., nn.Module],
    dataset: MusicDataset,
    *,
    kind: str,
    seed: int = 17,
    hidden_size: int = 48,
    updates: int = 320,
    batch_size: int = 48,
    learning_rate: float = 0.012,
    prompt_notes: int = 16,
) -> dict:
    """Fit the supplied class on CPU and return JSON-serializable evidence.

    Every model gets the same data, minibatch schedule, hidden width, Adam
    settings, next-note cross-entropy, and update budget. Different gate counts
    mean different parameter counts: this is not a pure architecture ranking.
    The final update is reported, not the best held-out checkpoint. Held-out
    notes never enter the optimizer; held-out metrics feed the true past, whereas
    greedy generation feeds guesses after a fixed prompt. The example is always
    heldout_phrases[0], selected before looking at any model's results.

    CPU RNG and thread settings are restored even if the caller's factory fails.
    Python's global RNG is never used. Returned reports do not contain a model
    or tensors; edits to the inline factory require rerunning the application.
    """
    _validate_dataset(dataset)
    if updates < 1 or hidden_size < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Use positive updates, hidden size, batch size, and learning rate.")
    if kind not in ("RNN", "LSTM", "GRU"):
        raise ValueError("kind must label RNN, LSTM, or GRU.")
    reference = list(dataset.heldout_phrases[0])
    if not 0 < prompt_notes < len(reference):
        raise ValueError("The prompt must be shorter than a phrase and contain at least one note.")
    prompt, steps = reference[:prompt_notes], len(reference) - prompt_notes
    vocabulary_size = len(dataset.vocabulary)
    train_x, train_y = make_next_note_batch(dataset.train_phrases, vocabulary_size)
    heldout_x, heldout_y = make_next_note_batch(dataset.heldout_phrases, vocabulary_size)
    started = time.perf_counter()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = model_factory(
                input_size=vocabulary_size, hidden_size=hidden_size,
                output_size=vocabulary_size,
            ).to(device="cpu", dtype=torch.float32)
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            before = {
                "train": _score(model, train_x, train_y),
                "heldout": _score(model, heldout_x, heldout_y),
                **_example_predictions(model, heldout_x[:1], prompt, steps, vocabulary_size),
            }
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            batch_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
            training_losses = []
            model.train()
            for _ in range(updates):
                indices = torch.randint(len(train_x), (batch_size,), generator=batch_rng)
                scores, _ = model(train_x[indices])
                loss = F.cross_entropy(scores.flatten(0, 1), train_y[indices].flatten())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                training_losses.append(loss.item())
            after = {
                "train": _score(model, train_x, train_y),
                "heldout": _score(model, heldout_x, heldout_y),
                **_example_predictions(model, heldout_x[:1], prompt, steps, vocabulary_size),
            }
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "kind": kind,
        "parameter_count": parameter_count,
        "config": {
            "seed": seed, "hidden_size": hidden_size, "updates": updates,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "optimizer": "Adam", "loss": "next-note cross-entropy",
            "clip_norm": 1.0, "generation": "greedy argmax",
            "cpu_threads": 1,
        },
        "dataset": {
            "fingerprint": dataset.fingerprint, "split_seed": dataset.split_seed,
            "train_phrases": len(dataset.train_phrases),
            "heldout_phrases": len(dataset.heldout_phrases),
            "events_per_phrase": len(reference),
            "vocabulary": list(dataset.vocabulary),
            "target_pairs_per_phrase": len(reference) - 1,
            "example_selection": "first held-out phrase, fixed before fitting",
        },
        "example": {
            "heldout_index": 0, "prompt_tokens": prompt, "reference_tokens": reference,
            "reference_continuation": reference[prompt_notes:],
        },
        "before": before,
        "after": after,
        "training_losses": training_losses,
        "elapsed_seconds": time.perf_counter() - started,
    }


def note_names(tokens: Sequence[int], vocabulary: Sequence[int | None]) -> list[str]:
    names = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    return [
        "rest" if vocabulary[token] is None
        else f"{names[vocabulary[token] % 12]}{vocabulary[token] // 12 - 1}"
        for token in tokens
    ]


def print_music_report(report: dict) -> None:
    """Compact actual metrics and the common prompt, without a winner claim."""
    data, config = report["dataset"], report["config"]
    print(
        f'{report["kind"]}: {report["parameter_count"]:,} parameters; '
        f'{data["train_phrases"]} train / {data["heldout_phrases"]} held-out phrases; '
        f'{config["updates"]} updates; {report["elapsed_seconds"]:.2f}s'
    )
    for split, label in (("train", "Train"), ("heldout", "Held-out")):
        before, after = report["before"][split], report["after"][split]
        print(
            f'{label} teacher-fed loss {before["loss"]:.3f} -> {after["loss"]:.3f}; '
            f'accuracy {before["accuracy"]:.1%} -> {after["accuracy"]:.1%}'
        )
    vocabulary = data["vocabulary"]
    example = report["example"]
    print("Same held-out seed:", " ".join(note_names(example["prompt_tokens"], vocabulary)))
    print("Reference continuation:", " ".join(note_names(example["reference_continuation"], vocabulary)))
    for key, label in (("before", "Initial"), ("after", "Trained")):
        continuation = report[key]["generated_continuation"]
        print(f"{label} free-running:", " ".join(note_names(continuation, vocabulary)))
    print("Teacher-fed scores use true past notes; free-running audio feeds back guesses.")
    print("Same hidden width, different parameter counts; synthetic melody, not a music benchmark.")


def melody_wav(
    midi_notes: Sequence[int | None], *,
    seconds_per_note: float = 0.16, sample_rate: int = 16000,
) -> bytes:
    """Deterministic mono PCM: fixed rhythm, a decaying tone, and silent rests."""
    if not midi_notes or seconds_per_note <= 0 or sample_rate < 1000:
        raise ValueError("Use a nonempty melody, positive duration, and sample rate >= 1000.")
    frames_per_note = round(seconds_per_note * sample_rate)
    if frames_per_note < 1:
        raise ValueError("Each note must occupy at least one audio frame.")
    samples = bytearray()
    cache = {}
    for midi in midi_notes:
        if midi is not None and (not isinstance(midi, int) or not 0 <= midi <= 127):
            raise ValueError("Pitches must be integer MIDI notes 0..127, or None for a rest.")
        if midi not in cache:
            frames = []
            frequency = 0 if midi is None else 440 * 2 ** ((midi - 69) / 12)
            for frame in range(frames_per_note):
                t = frame / sample_rate
                phase = 2 * math.pi * frequency * t
                attack = min(1.0, t / 0.008)
                release = min(1.0, (frames_per_note - 1 - frame) / (sample_rate * 0.025))
                envelope = attack * release * math.exp(-5.5 * t)
                tone = math.sin(phase) + 0.25 * math.sin(2 * phase) + 0.1 * math.sin(3 * phase)
                frames.append(round(15000 * envelope * tone / 1.35))
            cache[midi] = struct.pack(f"<{len(frames)}h", *frames)
        samples.extend(cache[midi])
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples)
    return buffer.getvalue()


def music_comparison_html(report: dict) -> str:
    """Export a self-contained HTML fragment with three actual WAV players.

    Display with IPython HTML, save as .html, or persist the fragment as notebook
    Markdown for playback without fitting again. There are no scripts, external
    styles, network requests, or runtime plotting dependencies. Players are not
    autoplayed. Each complete clip starts with exactly the same held-out seed.
    """
    vocabulary = report["dataset"]["vocabulary"]
    prompt = report["example"]["prompt_tokens"]
    rows = []
    for split, label in (("train", "Train"), ("heldout", "Held-out")):
        before, after = report["before"][split], report["after"][split]
        rows.append(
            f'<tr><th scope="row">{label}</th>'
            f'<td>{before["loss"]:.3f}</td><td>{after["loss"]:.3f}</td>'
            f'<td>{before["accuracy"]:.1%}</td><td>{after["accuracy"]:.1%}</td></tr>'
        )
    clips = []
    for label, tokens in (
        ("Authored reference", report["example"]["reference_tokens"]),
        ("Initial reader: seed + its guesses", prompt + report["before"]["generated_continuation"]),
        ("Trained reader: seed + its guesses", prompt + report["after"]["generated_continuation"]),
    ):
        midi = [vocabulary[token] for token in tokens]
        encoded = base64.b64encode(melody_wav(midi)).decode("ascii")
        seed_text = escape(" ".join(note_names(tokens[:len(prompt)], vocabulary)))
        continuation = escape(" ".join(note_names(tokens[len(prompt):], vocabulary)))
        clips.append(
            '<div style="padding:10px 0;border-top:1px solid #d5dce1">'
            f'<strong>{escape(label)}</strong><br>'
            f'<audio controls preload="none" style="width:100%;max-width:620px" '
            f'aria-label="{escape(label)}" src="data:audio/wav;base64,{encoded}"></audio>'
            f'<div style="font:13px monospace;overflow-wrap:anywhere">'
            f'<span style="color:#586876">Seed: {seed_text}</span><br>'
            f'Continuation: {continuation}</div></div>'
        )
    data, config = report["dataset"], report["config"]
    title = escape(report["kind"])
    return (
        '<section style="background:#fffaf1;color:#17324d;padding:18px;'
        'border:1px solid #bdc9ce;border-radius:10px;font:15px/1.5 system-ui,sans-serif;'
        'max-width:1000px">'
        f'<h3 style="margin:0 0 8px">{title}: learn an original melody vocabulary</h3>'
        f'<p>{data["train_phrases"]} train / {data["heldout_phrases"]} held-out whole phrases, '
        f'{data["events_per_phrase"]} events each; {len(vocabulary)} fixed note/rest tokens. '
        f'{config["updates"]} Adam updates, hidden width {config["hidden_size"]}, '
        f'{report["parameter_count"]:,} parameters.</p>'
        '<table style="text-align:right;border-spacing:12px 3px">'
        '<caption style="text-align:left"><strong>Teacher-fed next-note scoring</strong> '
        '(the reader receives the true past, not its guesses)</caption>'
        '<thead><tr><th>Split</th><th>Initial loss</th><th>Trained loss</th>'
        '<th>Initial accuracy</th><th>Trained accuracy</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'<p><strong>Free-running listening:</strong> the first {len(prompt)} events '
        'are the same held-out seed in every clip. After the seed, each reader '
        'receives only its own previous guess. The reference is one authored '
        'continuation, not the only musically possible ending.</p>'
        f'{"".join(clips)}'
        '<p style="font-size:13px">No selected best run or held-out checkpoint. '
        'Train and held-out phrases share composition rules, but no complete phrase. '
        'Same hidden width means different parameter counts, not a pure architecture ranking. '
        'These are original synthetic arpeggios with fixed timing and simple synthesized tones, '
        'not professional music or a music-quality benchmark.</p></section>'
    )
