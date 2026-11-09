"""Measured failure-first bridges from recurrent gradients to gated memory."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from torch import Tensor, nn

from backprop_memory_flow import (
    NOTEBOOK, NOTES, NOTE_HEAD_WEIGHTS, _inputs_and_weights, compact_animation_frames,
    record_trace, source_fingerprint as reader_fingerprint,
)
from lstm_state_roles import (
    _arrow, _canvas, _figure, _label, _panel, trace_cell_memory,
)
from memory_gate_flows import responsive_animation_html
from sequence_studio import (
    BACKGROUND, CORAL, CYAN, GOLD, MUTED, TEXT, VIOLET, PLAYER_STYLE,
    _instrument, _keyboard, draw_studio, studio_figure,
)

HERE = Path(__file__).resolve().parent
REPLAY = "01a-gradient-sensitivity-replay"
IMAGE_FILES = (
    "01a-tanh-saturation.png", REPLAY + ".png",
    "01a-unsquashed-total.png", "01a-protected-cell-feedback.png",
    "01a-gru-protected-update.png",
    "01a-rnn-notecard-closeup.png",
)


@dataclass(frozen=True)
class NotecardProbe:
    current_input: Tensor
    old_hidden: Tensor
    input_weights: Tensor
    memory_weights: Tensor
    input_part: Tensor
    memory_part: Tensor
    combined: Tensor
    new_hidden: Tensor
    scores: Tensor
    prediction: str


@dataclass(frozen=True)
class FeedbackProbe:
    states: tuple[tuple[float, ...], ...]
    feedback: tuple[float, ...]


@dataclass(frozen=True)
class MemoryProbe:
    stored: tuple[float, ...]
    exposed: tuple[float, ...]
    stored_feedback: float
    exposed_feedback: float


@dataclass(frozen=True)
class ClippingProbe:
    raw_gradient: float
    clipped_gradient: float
    raw_update: float
    clipped_update: float


def notecard_probe(*, erase_incoming: bool = False, path: Path = NOTEBOOK) -> NotecardProbe:
    """Open the recorded D4 step without changing the note or learned parameters."""
    if not isinstance(erase_incoming, bool):
        raise ValueError("erase_incoming must be a boolean.")
    inputs, input_weights, memory_weights = _inputs_and_weights(path)
    trace = record_trace(path)
    old_hidden = torch.zeros_like(trace.states[2]) if erase_incoming else trace.states[2]
    input_part, memory_part = input_weights @ inputs[2], memory_weights @ old_hidden
    combined = input_part + memory_part
    new_hidden = torch.tanh(combined)
    scores = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=torch.float64) @ new_hidden
    if not erase_incoming:
        torch.testing.assert_close(new_hidden, trace.states[3], rtol=0, atol=0)
        torch.testing.assert_close(scores, trace.scores[2], rtol=0, atol=0)
    return NotecardProbe(
        inputs[2], old_hidden, input_weights, memory_weights, input_part, memory_part,
        combined, new_hidden, scores, NOTES[int(scores.argmax())],
    )


def _delay(delay: int) -> None:
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        raise ValueError("The delay must be a positive integer.")


def _finite(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")


def _fraction(**values: float) -> None:
    _finite(**values)
    for name, value in values.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one.")


def _feedback_probe(states: list[Tensor], readout: Tensor) -> FeedbackProbe:
    gradients = torch.autograd.grad(readout, states)
    return FeedbackProbe(
        tuple(tuple(state.detach().reshape(-1).tolist()) for state in states),
        tuple(float(gradient.norm()) for gradient in gradients),
    )


def recurrent_probe(gain: float = 0.8, *, opening: float = 0.8,
                    drive: float = 0.0, delay: int = 20) -> FeedbackProbe:
    """Differentiate an actual scalar tanh RNN, not a fixed retention surrogate."""
    _delay(delay)
    _finite(gain=gain, opening=opening, drive=drive)
    states = [torch.tensor(opening, dtype=torch.float64, requires_grad=True)]
    for _ in range(delay):
        states.append(torch.tanh(gain * states[-1] + drive))
    return _feedback_probe(states, states[-1])


def piano_probe(delay: int = 20, path: Path = NOTEBOOK) -> FeedbackProbe:
    """Reuse the lesson's two-feature reader and D3 head; A3 fills the delay."""
    _delay(delay)
    inputs, input_weights, memory_weights = _inputs_and_weights(path)
    opening = torch.tanh(input_weights @ inputs[0]).detach().requires_grad_()
    states = [opening]
    for _ in range(delay):
        states.append(torch.tanh(input_weights @ inputs[1] + memory_weights @ states[-1]))
    head = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=torch.float64)
    return _feedback_probe(states, (head @ states[-1])[0])


def clipping_probe(gain: float, *, opening: float, delay: int = 20,
                   max_norm: float = 1.0) -> ClippingProbe:
    """Clip the gradient of a first-note writing weight, not a carried activation."""
    _delay(delay)
    _finite(gain=gain, opening=opening, max_norm=max_norm)
    if max_norm <= 0:
        raise ValueError("The clipping cap must be positive.")
    writer = nn.Parameter(torch.tensor(opening, dtype=torch.float64))
    value = writer * 1.0
    for _ in range(delay):
        value = torch.tanh(gain * value)
    (0.5 * (value - 1.0).square()).backward()
    if writer.grad is None:
        raise RuntimeError("The error did not reach the first-note writing weight.")
    raw_gradient = writer.grad.item()
    unbounded = nn.Parameter(writer.detach().clone())
    unbounded.grad = writer.grad.detach().clone()
    torch.optim.SGD((unbounded,), lr=0.01).step()
    torch.nn.utils.clip_grad_norm_((writer,), max_norm, error_if_nonfinite=True)
    clipped_gradient = writer.grad.item()
    torch.optim.SGD((writer,), lr=0.01).step()
    return ClippingProbe(
        raw_gradient, clipped_gradient,
        unbounded.item() - opening, writer.item() - opening,
    )


def _memory_probe(states: list[Tensor], exposed: list[Tensor]) -> MemoryProbe:
    stored_gradient = torch.autograd.grad(states[-1], states[0], retain_graph=True)[0]
    exposed_gradient = torch.autograd.grad(exposed[-1], states[0])[0]
    stored_values = tuple(value.detach().item() for value in states)
    exposed_values = stored_values if states is exposed else tuple(value.detach().item() for value in exposed)
    return MemoryProbe(
        stored_values, exposed_values,
        stored_gradient.item(), exposed_gradient.item(),
    )


def unsquashed_total(opening: str) -> MemoryProbe:
    history = trace_cell_memory(opening)
    states = [torch.tensor(history[0].proposal, dtype=torch.float64, requires_grad=True)]
    for step in history[1:]:
        states.append(states[-1] + step.proposal)
    return _memory_probe(states, states)


def lstm_carry(keep: float = 0.99, *, opening: float = 0.8,
               write: float = 0.0, proposal: float = 0.8, delay: int = 20) -> MemoryProbe:
    """Constant controls isolate c's carry path and a single final tanh readout."""
    _delay(delay)
    _fraction(keep=keep, write=write)
    _finite(opening=opening, proposal=proposal)
    if not -1 <= proposal <= 1:
        raise ValueError("The chosen LSTM candidate must be between -1 and +1.")
    states = [torch.tensor(opening, dtype=torch.float64, requires_grad=True)]
    for _ in range(delay):
        states.append(keep * states[-1] + write * proposal)
    return _memory_probe(states, [torch.tanh(value) for value in states])


def gru_carry(keep: float = 0.99, *, opening: float = 0.8,
              reset: float = 0.0, drive: float = 4.0, delay: int = 20) -> MemoryProbe:
    """PyTorch's z keeps old h; only the proposal passes through tanh."""
    _delay(delay)
    _fraction(keep=keep, reset=reset)
    _finite(opening=opening, drive=drive)
    if not -1 <= opening <= 1:
        raise ValueError("Start this bounded-GRU demonstration with h between -1 and +1.")
    states = [torch.tensor(opening, dtype=torch.float64, requires_grad=True)]
    for _ in range(delay):
        proposal = torch.tanh(drive + reset * 0.6 * states[-1])
        states.append(keep * states[-1] + (1 - keep) * proposal)
    return _memory_probe(states, states)


def native_gated_probe(kind: str, proposal_gain: float, *, keep: float = 0.99,
                       delay: int = 20) -> MemoryProbe:
    """Stress a real scalar cell near zero, keeping all other gate weights fixed."""
    _delay(delay)
    _fraction(keep=keep)
    _finite(proposal_gain=proposal_gain)
    if kind not in ("lstm", "gru") or not 0 < keep < 1:
        raise ValueError("Choose lstm or gru and a finite sigmoid KEEP strictly between zero and one.")
    with torch.random.fork_rng(devices=[]):
        cell = (nn.LSTMCell(1, 1) if kind == "lstm" else nn.GRUCell(1, 1)).double()
        with torch.no_grad():
            for parameter in cell.parameters():
                parameter.zero_()
            if cell.bias_ih is None:
                raise RuntimeError("The native gate probe requires the default biased cell.")
            cell.bias_ih[1] = math.log(keep / (1 - keep))
            cell.weight_hh[2, 0] = proposal_gain
        cell.requires_grad_(False)
    opening = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    inputs = torch.zeros_like(opening)
    if isinstance(cell, nn.LSTMCell):
        c, h = opening, 0.5 * torch.tanh(opening)
        stored, exposed = [c], [h]
        for _ in range(delay):
            h, c = cell(inputs, (h, c))
            stored.append(c)
            exposed.append(h)
        return _memory_probe(stored, exposed)
    states = [opening]
    for _ in range(delay):
        states.append(cell(inputs, states[-1]))
    return _memory_probe(states, states)


def source_fingerprint(path: Path = NOTEBOOK) -> str:
    return hashlib.sha256(
        Path(__file__).read_bytes() + (HERE / "lstm_state_roles.py").read_bytes()
        + reader_fingerprint(path).encode()
    ).hexdigest()


def draw_notecard(axis, probe: NotecardProbe) -> None:
    _canvas(axis)
    vector = lambda values: "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"
    matrix = lambda values: "\n".join(
        "[" + ", ".join(f"{value:+.1f}" for value in row) + "]" for row in values
    )
    _label(axis, 9, 10.35, "Open the notecard: D4 + old h2 makes h3", 27, weight="bold")
    _label(axis, 9, 9.55, "Current input, shared weights, temporary state: three different objects.", 18, MUTED)
    _label(axis, 2.35, 8.65, "Current note: D4", 19, GOLD)
    _keyboard(axis, 1.52, 7.8, "D4", visible=True)
    _panel(axis, 0.6, 6.35, 3.5, 1.05, "input x3\n" + vector(probe.current_input), GOLD, 17)
    _panel(axis, 0.6, 3.75, 3.5, 1.5,
           "old h2 after D3 A3\n" + vector(probe.old_hidden), CYAN, 17)
    _panel(axis, 4.95, 6.05, 3.5, 2.0,
           "input weights W_x\n" + matrix(probe.input_weights), GOLD, 18)
    _panel(axis, 4.95, 3.45, 3.5, 2.0,
           "memory weights W_h\n" + matrix(probe.memory_weights), CYAN, 18)
    _arrow(axis, (4.2, 6.9), (4.8, 6.9), GOLD)
    _arrow(axis, (4.2, 4.5), (4.8, 4.5), CYAN)
    _panel(axis, 9.3, 6.35, 3.4, 1.4,
           "input contribution\n" + vector(probe.input_part), GOLD, 17)
    _panel(axis, 9.3, 3.75, 3.4, 1.4,
           "past contribution\n" + vector(probe.memory_part), CYAN, 17)
    _arrow(axis, (8.55, 7.05), (9.15, 7.05), GOLD)
    _arrow(axis, (8.55, 4.45), (9.15, 4.45), CYAN)
    _label(axis, 13.6, 5.8, "+", 25)
    _arrow(axis, (12.8, 7.05), (13.45, 6.1), GOLD)
    _arrow(axis, (12.8, 4.45), (13.45, 5.45), CYAN)
    _panel(axis, 14.1, 5.15, 3.3, 1.3,
           "combined features\n" + vector(probe.combined), TEXT, 17)
    _arrow(axis, (13.83, 5.8), (14.0, 5.8), TEXT)
    _instrument(axis, 15.75, 3.95, "tanh", CYAN, gid="notecard-squash")
    _arrow(axis, (15.75, 5.05), (15.75, 4.6), CYAN)
    _panel(axis, 14.1, 1.85, 3.3, 1.3,
           "new notecard h3\n" + vector(probe.new_hidden), CYAN, 18)
    _arrow(axis, (15.75, 3.25), (15.75, 3.2), CYAN)
    _arrow(axis, (17.5, 2.5), (17.95, 2.5), CYAN, gid="notecard-carry")
    _label(axis, 15.75, 1.35, "carry h3 to the next read", 15, CYAN)
    _instrument(axis, 10.85, 2.5, "head", VIOLET, gid="notecard-head")
    _label(axis, 10.85, 3.3, "shared W_out", 16, VIOLET)
    _arrow(axis, (14.0, 2.5), (11.55, 2.5), VIOLET, gid="notecard-expose")
    _arrow(axis, (10.15, 2.5), (9.8, 2.5), VIOLET)
    _panel(axis, 5.05, 1.65, 4.6, 1.45,
           f"Next-note scores\nD3: {probe.scores[0]:.4f}   A3: {probe.scores[1]:.4f}\n"
           f"D4: {probe.scores[2]:.4f}", VIOLET, 16)
    axis.texts[-1].set_gid("notecard-score-label")
    axis.patches[-1].set_gid("notecard-score-card")
    _label(axis, 7.35, 1.1, f"Largest score: {probe.prediction}, the next-note guess", 15, VIOLET)
    _label(axis, 2.4, 2.4, "The two h numbers are\ncomputed features, not\nwritten notes or named facts.", 16, MUTED)
    _label(axis, 9, 0.4, "Input D4 is unchanged. Its new context comes from old h2. Hand-picked weights, not learned music.", 15, MUTED)


def draw_saturation(axis) -> None:
    _canvas(axis)
    _label(axis, 9, 10.35, "Same tiny nudge. Different output changes.", 28, weight="bold")
    _label(axis, 9, 9.55, "Nudge one internal feature by +0.0001. This is not a predicted note versus a target.", 17, MUTED)
    _label(axis, 3.8, 8.65, "Internal value BEFORE tanh", 19, GOLD)
    _label(axis, 13.65, 8.65, "Card value AFTER tanh", 19, CYAN)
    nudge = 0.0001
    for start, y, description in ((0.0, 6.25, "Near the centre"), (3.0, 3.25, "Near the flat edge")):
        before, after = math.tanh(start), math.tanh(start + nudge)
        _label(axis, 3.8, y + 1.85, description, 18, MUTED)
        _panel(axis, 1.0, y, 5.6, 1.3, f"before: {start:.4f}\nafter nudge: {start + nudge:.4f}", GOLD, 21)
        _arrow(axis, (6.75, y + 0.65), (10.15, y + 0.65), CYAN)
        _label(axis, 8.45, y + 1.25, "tanh", 18, CYAN)
        _panel(axis, 10.3, y, 6.7, 1.3, f"before: {before:.10f}\nafter nudge: {after:.10f}", CYAN, 21)
        _label(axis, 3.8, y - 0.5, "Input change: +0.0001000000", 17, GOLD)
        _label(axis, 13.65, y - 0.5, f"Output change: +{after - before:.10f}", 17, CYAN)
    _label(axis, 9, 1.75, f"Local backward factor: 1.000000 near 0; {1 - math.tanh(3)**2:.6f} near 3.", 20)
    _label(axis, 9, 1.05, "Near the flat edge, a small forward change and backward feedback both get weakened.", 17)
    _label(axis, 9, 0.4, "Near zero, small changes pass almost unchanged. Tanh's shrinking is not equally strong everywhere.", 15, MUTED)


def feedback_figure(frame: int = 20):
    if not 0 <= frame <= 20:
        raise ValueError("Choose one of the 21 recorded backward-inspection frames.")
    probes = (
        ("Scalar RNN: gain 0.8", recurrent_probe(), CYAN),
        ("Scalar RNN: gain 1.2, zero-centred", recurrent_probe(1.2, opening=0), CORAL),
        ("Actual two-feature piano reader", piano_probe(), GOLD),
    )
    figure, axis = plt.subplots(figsize=(16, 9), facecolor=BACKGROUND)
    figure.subplots_adjust(left=0.085, right=0.97, top=0.79, bottom=0.25)
    draw_feedback_chart(figure, axis, probes, frame)
    return figure, axis, probes


def draw_feedback_chart(figure, axis, probes, frame: int) -> None:
    axis.clear()
    axis.set_facecolor("#14283e")
    index = 20 - frame
    for label, probe, color in probes:
        sizes = np.array(probe.feedback) / probe.feedback[-1]
        axis.semilogy(np.arange(index, 21), sizes[index:], color=color, lw=2.5, label=label)
    axis.set(xlim=(-0.4, 20.4), ylim=(1e-12, 1e3), xticks=(0, 5, 10, 15, 20))
    axis.set_xlabel("0: opening card                 Forward reads                 20: final card", color=TEXT, fontsize=17)
    axis.set_ylabel("Sensitivity relative to final card (= 1)", color=TEXT, fontsize=17)
    axis.tick_params(colors=TEXT, labelsize=14)
    axis.grid(color=MUTED, alpha=0.15)
    for spine in axis.spines.values():
        spine.set_color(MUTED)
    axis.axvline(index, color=TEXT, alpha=0.4, linestyle="--")
    axis.legend(facecolor=BACKGROUND, labelcolor=TEXT, edgecolor=MUTED, fontsize=14, loc="lower right")
    for text in list(figure.texts):
        text.remove()
    figure.text(0.5, 0.94, "Bounded states. Fading or surging feedback.", ha="center",
                fontsize=28, color=TEXT, fontfamily="DejaVu Serif", weight="bold")
    figure.text(0.5, 0.875, "Read RIGHT to LEFT for backward feedback. We scale final sensitivity to 1, not the state value.",
                ha="center", fontsize=17, color=MUTED)
    for x, (_, probe, color) in zip((0.24, 0.5, 0.78), probes):
        ratio = probe.feedback[index] / probe.feedback[-1]
        figure.text(x, 0.12, f"Card {index} sensitivity ratio: {ratio:.6f}",
                    ha="center", fontsize=16, color=color)
    figure.text(0.5, 0.055, "Below 1: weaker response. Above 1: amplified response. These are not memory amounts or note probabilities.",
                ha="center", fontsize=15, color=MUTED)


def draw_unsquashed_total(axis) -> None:
    _canvas(axis)
    _label(axis, 9, 10.35, "Remove the squash. Still lose the order.", 28, weight="bold")
    _label(axis, 9, 9.55, "Add every note's chosen proposal: no squash, and no decision about what to protect.", 17, MUTED)
    for opening, y in (("D3", 6.1), ("D4", 2.9)):
        history, probe = trace_cell_memory(opening), unsquashed_total(opening)
        _label(axis, 1.3, y + 0.55, f"{opening}\nopening", 19)
        for index, (x, step, value) in enumerate(zip((3.0, 6.4, 9.8, 13.2), history, probe.stored)):
            _label(axis, x + 1.3, y + 2.15, step.note, 18, GOLD)
            _keyboard(axis, x + 0.48, y + 1.35, step.note, visible=True)
            _panel(axis, x, y, 2.6, 1.1, f"[{value:+.2f}]", GOLD, 24)
            if index < 3:
                _arrow(axis, (x + 2.7, y + 0.55), (x + 3.3, y + 0.55), GOLD)
    _label(axis, 9, 2.1, "Different openings, same final A3, same total: +0.40.", 20)
    _label(axis, 9, 1.3, "Feedback to the first written number is 1.0. Perfect carry alone did not preserve order.", 17)
    _label(axis, 9, 0.55, "Unsquashed does not mean raw, complete or recoverable. We still need a useful representation.", 16, MUTED)


def draw_cell_feedback(axis) -> None:
    _canvas(axis)
    _label(axis, 9, 10.35, "Keep the memory. Squash only the view.", 28, weight="bold")
    _label(axis, 9, 9.55, "Start c at +0.8; twenty A3 distractions; WRITE closed, SHOW open. Chosen controls.", 17, MUTED)
    for x, keep in ((0.65, 1.0), (6.45, 0.99), (12.25, 0.9)):
        probe = lstm_carry(keep)
        center = x + 2.5
        _label(axis, center, 8.7, f"KEEP = {keep:.2f}", 22, GOLD)
        _panel(axis, x, 6.9, 5, 1.2,
               f"stored c20: {probe.stored[-1]:+.4f}\nOpening credit: {probe.stored_feedback:.4f}", GOLD, 19)
        _instrument(axis, center, 5.3, "tanh", CYAN, gid=f"readout-{keep}")
        _arrow(axis, (center, 6.85), (center, 6.0), CYAN)
        _arrow(axis, (center, 4.65), (center, 4.4), CYAN)
        _panel(axis, x, 3.1, 5, 1.2, f"exposed h20: {probe.exposed[-1]:+.4f}", CYAN, 22)
        _label(axis, center, 2.55, f"Readout credit: {probe.exposed_feedback:.4f}", 17, CYAN)
    _label(axis, 9, 1.45, "c avoids repeated squashing; the final h readout can still limit feedback.", 19)
    _label(axis, 9, 0.6, "Learned controls add other derivative routes. This isolates the direct carry lane, not architecture quality.", 15, MUTED)


def draw_gru_feedback(axis) -> None:
    _canvas(axis)
    probe = gru_carry()
    one_step = gru_carry(delay=1)
    _label(axis, 9, 10.35, "Protect old h outside the proposal's squash.", 27, weight="bold")
    _label(axis, 9, 9.55, "A GRU blends retained h with a bounded proposal; it does not squash the whole blend again.", 17, MUTED)
    _panel(axis, 0.8, 7.1, 4.6, 1.2, "old h: +0.8", CYAN, 23)
    _arrow(axis, (5.55, 7.7), (12.15, 7.7), CYAN)
    _label(axis, 8.7, 8.45, "KEEP z = 0.99", 20, CYAN)
    _panel(axis, 12.3, 7.1, 4.6, 1.2, f"new h: {one_step.stored[-1]:+.4f}", CYAN, 21)
    _arrow(axis, (17.05, 7.7), (17.65, 7.7), CYAN)
    _keyboard(axis, 1.05, 4.7, "A3", visible=True)
    _label(axis, 1.9, 5.7, "input A3", 19, GOLD)
    _arrow(axis, (2.9, 5.0), (6.15, 5.0), GOLD)
    _panel(axis, 6.3, 4.4, 4.6, 1.2, f"proposal: tanh(4)\n{math.tanh(4):+.4f}", VIOLET, 20)
    _arrow(axis, (11.05, 5.0), (12.15, 7.25), VIOLET)
    _label(axis, 12.7, 5.2, "WRITE 1-z", 16, VIOLET)
    _label(axis, 8.6, 3.8, "RESET = 0: old h stays out of this proposal.", 16, MUTED)
    _arrow(axis, (14.6, 7.0), (14.6, 4.2), VIOLET)
    _instrument(axis, 14.6, 3.45, "head", VIOLET, gid="gru-head")
    _label(axis, 16.9, 6.15, "carry\nsame h", 16, CYAN)
    _label(axis, 14.6, 2.6, "note scores", 16, VIOLET)
    _label(axis, 9, 1.6, f"Opening feedback after 20 transitions: {probe.stored_feedback:.6f}, not twenty tanh derivatives.", 18)
    _label(axis, 9, 0.65, "Bounded h is compatible with protected feedback. Learned gates and proposal routes still affect the full gradient.", 15, MUTED)


def export(path: Path = NOTEBOOK, directory: Path = HERE / "images") -> None:
    directory.mkdir(exist_ok=True)
    metadata = {
        "SourceFingerprint": source_fingerprint(path), "Generator": "measured local gradient-memory studio",
        "ProbeDefaults": json.dumps({
            "piano_feedback": piano_probe(path=path).feedback[0],
            "cell_feedback": lstm_carry().stored_feedback,
            "gru_feedback": gru_carry().stored_feedback,
        }),
    }
    for filename, draw in (
        (IMAGE_FILES[0], draw_saturation), (IMAGE_FILES[2], draw_unsquashed_total),
        (IMAGE_FILES[3], draw_cell_feedback), (IMAGE_FILES[4], draw_gru_feedback),
        (IMAGE_FILES[5], lambda axis: draw_notecard(axis, notecard_probe(path=path))),
    ):
        figure, axis = _figure()
        draw(axis)
        figure.savefig(directory / filename, dpi=100, facecolor=BACKGROUND, metadata=metadata)
        plt.close(figure)
    figure, axis, probes = feedback_figure()
    figure.savefig(directory / IMAGE_FILES[1], dpi=100, facecolor=BACKGROUND, metadata=metadata)
    figure.set_dpi(80)
    animation = FuncAnimation(
        figure, lambda frame: draw_feedback_chart(figure, axis, probes, frame),
        frames=21, interval=400, repeat=False,
    )
    plt.close(figure)
    (directory / (REPLAY + ".html")).write_text(
        PLAYER_STYLE + compact_animation_frames(responsive_animation_html(animation)), encoding="utf-8",
    )
    animation.save(directory / (REPLAY + ".gif"), writer="pillow", fps=2.5, dpi=80)
    figure, axis = studio_figure()
    draw_studio(axis, record_trace(path), stage=2, overview=True,
                caption="One shared reader, three saved cards, a next-note guess at every read.")
    target = directory / "01a-backprop-saved-cards.png"
    figure.savefig(target, dpi=100, facecolor=BACKGROUND, metadata=metadata)
    plt.close(figure)
    from PIL import Image
    with Image.open(target) as image:
        image.resize((1280, 720), Image.Resampling.LANCZOS).save(
            directory / "01a-backprop-saved-cards-preview.png", optimize=True,
        )
    print("Saved six measured figures, a 21-frame feedback replay, and the quieter section-2 storyboard.")


def refresh_notebook_captures(path: Path = NOTEBOOK, directory: Path = HERE / "images") -> None:
    from notebook_replays import NOTEBOOK_SIZE_LIMIT, embed_notebook_images, refresh_embedded_replays

    raw = path.read_bytes()
    book = json.loads(raw)
    ending = "\n" if raw.endswith(b"\n") else ""
    if (json.dumps(book, ensure_ascii=False, indent=1) + ending).encode() != raw:
        raise ValueError("Notebook serialization changed; refresh with a notebook-aware editor.")
    sources = ["".join(cell["source"]) for cell in book["cells"]]
    owners = {}
    for filename in IMAGE_FILES + (REPLAY + ".gif", "01a-backprop-saved-cards.png"):
        matches = [cell for cell in book["cells"] if f"attachment:{filename}" in "".join(cell["source"])]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one notebook owner for {filename}.")
        owners[matches[0]["id"]] = matches[0]
    embed_notebook_images({"cells": list(owners.values())}, images_directory=directory, refresh_existing=True)
    players = [cell for cell in book["cells"]
               if cell.get("metadata", {}).get("saved_replay", {}).get("file") == REPLAY + ".html"]
    if len(players) != 1:
        raise ValueError("Expected exactly one marked gradient-sensitivity player.")
    refresh_embedded_replays({"cells": players}, images_directory=directory)
    if sources != ["".join(cell["source"]) for cell in book["cells"]] or path.read_bytes() != raw:
        raise RuntimeError("Notebook sources changed during gradient capture refresh.")
    encoded = (json.dumps(book, ensure_ascii=False, indent=1) + ending).encode()
    if len(encoded) >= NOTEBOOK_SIZE_LIMIT:
        raise ValueError("Gradient captures exceed the approved notebook size budget.")
    path.write_bytes(encoded)


if __name__ == "__main__":
    export()
    refresh_notebook_captures()
