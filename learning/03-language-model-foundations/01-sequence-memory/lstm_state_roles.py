"""Measure the different jobs of LSTM c and h without changing the live cell."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from lstm_layer_flow import LSTMLayerStep, record_lstm_layer_step
from sequence_studio import (
    BACKGROUND, CORAL, CYAN, GOLD, MUTED, TEXT, VIOLET,
    _card as studio_card, _flow as studio_flow, _instrument as studio_instrument,
    _keyboard as studio_keyboard,
)


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01A-sequence-memory-theory.ipynb"
CONTROLS = ("input_gate", "forget_gate", "candidate", "output_gate")
PAPER, PAPER_INK, INK = "#ecdfc6", "#51391e", TEXT
CELL, HIDDEN = GOLD, CYAN


@dataclass(frozen=True)
class StateRoleProbe:
    baseline: LSTMLayerStep
    no_old_cell: LSTMLayerStep
    no_old_hidden: LSTMLayerStep
    show_closed: LSTMLayerStep


@dataclass(frozen=True)
class CellMemoryStep:
    note: str
    previous: float
    keep: float
    write: float
    proposal: float
    value: float


def trace_cell_memory(opening: str, *, overwrite_last: bool = False) -> tuple[CellMemoryStep, ...]:
    """A hand-chosen single-coordinate demonstration, not a trained LSTM."""
    if opening not in ("D3", "D4"):
        raise ValueError("Choose D3 or D4 for this opening-cue demonstration.")
    notes = ("D3", "A3", "D4", "A3") if opening == "D3" else ("D4", "A3", "D3", "A3")
    proposals = {"D3": 0.8, "A3": 0.2, "D4": -0.8}
    value = 0.0
    history = []
    for index, note in enumerate(notes):
        keep, write = (0.0, 1.0) if index == 0 else (1.0, 0.0)
        if overwrite_last and index == len(notes) - 1:
            keep, write = 0.0, 1.0
        previous = value
        value = keep * previous + write * proposals[note]
        history.append(CellMemoryStep(note, previous, keep, write, proposals[note], value))
    return tuple(history)


def record_state_roles(cell: nn.LSTMCell, inputs: Tensor, old_h: Tensor, old_c: Tensor) -> StateRoleProbe:
    if not isinstance(cell, nn.LSTMCell):
        raise TypeError("The state-role probe requires an LSTMCell.")
    baseline = record_lstm_layer_step(cell, inputs, old_h, old_c)
    no_old_cell = record_lstm_layer_step(cell, inputs, old_h, torch.zeros_like(old_c))
    no_old_hidden = record_lstm_layer_step(cell, inputs, torch.zeros_like(old_h), old_c)
    closed = deepcopy(cell)
    if closed.bias_ih is None or closed.bias_hh is None:
        raise ValueError("Closing SHOW requires the lesson's biased LSTMCell.")
    output_rows = slice(3 * cell.hidden_size, 4 * cell.hidden_size)
    with torch.no_grad():
        closed.weight_ih[output_rows].zero_()
        closed.weight_hh[output_rows].zero_()
        closed.bias_ih[output_rows].fill_(-1e9)
        closed.bias_hh[output_rows].zero_()
    show_closed = record_lstm_layer_step(closed, inputs, old_h, old_c)
    torch.testing.assert_close(show_closed.new_cell, baseline.new_cell, rtol=0, atol=0)
    if torch.count_nonzero(show_closed.new_hidden):
        raise RuntimeError("The SHOW intervention did not close the exposed readout.")
    return StateRoleProbe(baseline, no_old_cell, no_old_hidden, show_closed)


def load_state_roles(path: Path = NOTEBOOK) -> StateRoleProbe:
    book = json.loads(path.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in book["cells"]}
    setup = ast.parse("".join(cells["eb7df241"]["source"]))
    melody = [
        node for node in setup.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "melody" for target in node.targets)
    ]
    if len(melody) != 1:
        raise ValueError("The notebook must define one running melody.")
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=melody, type_ignores=[]), "running_melody", "exec"), namespace)
    with redirect_stdout(StringIO()):
        exec(compile("".join(cells["lstm-real-cell-parity"]["source"]), "native_lstm", "exec"), namespace)
    cell, rows, trace = (namespace[name] for name in ("lstm_real_cell", "lstm_piano_rows", "lstm_real_trace"))
    if not isinstance(cell, nn.LSTMCell) or not isinstance(rows, Tensor) or not isinstance(trace, list):
        raise TypeError("The native LSTM example did not produce its cell, input rows and trace.")
    previous = trace[1]
    return record_state_roles(cell, rows[1:2], previous["old_h"], previous["old_c"])


def source_fingerprint(path: Path = NOTEBOOK) -> str:
    book = json.loads(path.read_text(encoding="utf-8"))
    cell = next(cell for cell in book["cells"] if cell["id"] == "lstm-real-cell-parity")
    return hashlib.sha256(
        "".join(cell["source"]).encode("utf-8") + Path(__file__).read_bytes()
        + (HERE / "lstm_layer_flow.py").read_bytes() + (HERE / "sequence_studio.py").read_bytes()
    ).hexdigest()


def _figure():
    figure, axis = plt.subplots(figsize=(16, 9), facecolor=BACKGROUND)
    figure.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.035)
    return figure, axis


def _canvas(axis):
    axis.clear()
    axis.set(xlim=(0, 18), ylim=(0, 11))
    axis.axis("off")
    axis.figure.set_facecolor(BACKGROUND)
    axis.add_patch(Ellipse((9, 5.9), 22, 16, facecolor="#102137", edgecolor="none", zorder=-10))
    axis.add_patch(Ellipse((9, 6.3), 16, 12, facecolor="#14283e", edgecolor="none", alpha=0.45, zorder=-9))


def _label(axis, x, y, text, size=20, color=INK, *, weight="normal"):
    return axis.text(x, y, text, ha="center", va="center", fontsize=size,
                     color=color, weight=weight, linespacing=1.3, zorder=10,
                     fontfamily="DejaVu Serif" if size >= 26 and weight == "bold" else "DejaVu Sans")


def _panel(axis, x, y, width, height, text, color=INK, size=20):
    studio_card(axis, x, y, width, height, color)
    card = axis.patches[-1]
    card.set_facecolor(PAPER if color == CELL else "#142b42")
    card.set_linewidth(1.4)
    card.set_path_effects([
        path_effects.SimplePatchShadow(offset=(2, -3), alpha=0.3),
        path_effects.Normal(),
    ])
    _label(axis, x + width / 2, y + height / 2, text, size, PAPER_INK if color == CELL else color)


def _arrow(axis, start, end, color=INK, *, gid="state-flow"):
    distance = end[0] - start[0]
    rise = end[1] - start[1]
    studio_flow(axis, (
        start, (start[0] + distance / 3, start[1] + rise / 6),
        (end[0] - distance / 3, end[1] - rise / 6), end,
    ), color, gid)
    arrow = axis.patches[-1]
    arrow.set_mutation_scale(12)
    arrow.set_zorder(8)


def _vector(values: Tensor) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


def draw_cell_memory(axis) -> None:
    _canvas(axis)
    _label(axis, 9, 10.35, "Cell state: a numeric workspace", 28, weight="bold")
    _label(axis, 9, 9.65, "One chosen coordinate of c. We pick its meaning and controls for this demonstration.", 17, MUTED)
    _label(axis, 9, 9.05, "Write the opening cue once. Then KEEP = 1 and WRITE = 0.", 20, CELL)
    positions = (3.0, 6.4, 9.8, 13.2)
    for opening, y in (("D3", 6.1), ("D4", 2.9)):
        history = trace_cell_memory(opening)
        _label(axis, 1.3, y + 0.5, f"{opening} opening\nstart: c0 = 0", 19)
        for index, (x, step) in enumerate(zip(positions, history)):
            label = _label(axis, x + 1.3, y + 2.15, f"read {step.note}", 18, CELL)
            label.set_gid(f"memory-input-{opening}-{index}")
            studio_keyboard(axis, x + 0.48, y + 1.35, step.note, visible=True)
            _panel(axis, x, y, 2.6, 1.1, f"c{index + 1}: [{step.value:+.2f}]", CELL, 24)
            if index < 3:
                _arrow(axis, (x + 2.7, y + 0.55), (positions[index + 1] - 0.1, y + 0.55),
                       CELL, gid=f"memory-carry-{opening}-{index}")
    _label(axis, 9, 2.2, "Same final A3 input, different retained numbers: the opening still matters here.", 18)
    _label(axis, 9, 1.55, "c1, c2, c3, c4 are versions of ONE current coordinate, not four carried memory slots.", 16, MUTED)
    overwritten = trace_cell_memory("D3", overwrite_last=True)[-1].value
    _label(axis, 9, 0.65,
           f"At the last A3, drop the old value and write +0.20: both finish at [{overwritten:+.2f}].\n"
           "That loses the opening distinction in this coordinate, not necessarily in the whole LSTM.", 16)


def draw_two_state_step(axis, probe: StateRoleProbe) -> None:
    _canvas(axis)
    step = probe.baseline
    _label(axis, 9, 10.3, "Edit the memory c; expose a view h", 28, weight="bold")
    _label(axis, 9, 9.55, "Actual initialized LSTM: the A3 step after D3. Both c and h contain numbers.", 18, MUTED)
    _panel(axis, 0.6, 7.25, 4.0, 1.35, f"old c1\n{_vector(step.old_cell)}", CELL, 21)
    _panel(axis, 7.0, 7.25, 4.0, 1.35, f"new c2\n{_vector(step.new_cell)}", CELL, 21)
    _arrow(axis, (4.7, 7.9), (6.9, 7.9), CELL)
    _label(axis, 5.8, 8.65, "KEEP old\n+ WRITE proposal", 16, CELL)
    _label(axis, 2.6, 5.35, "input A3 + old h1", 19)
    studio_keyboard(axis, 0.9, 4.15, "A3", visible=True)
    _label(axis, 2.85, 4.45, "+", 22, MUTED)
    _panel(axis, 3.15, 4.1, 1.1, 0.7, "h1", HIDDEN, 17)
    _label(axis, 2.6, 3.55, "choose controls + new proposal", 16, MUTED)
    _arrow(axis, (4.7, 4.6), (6.9, 7.4), MUTED)
    _panel(axis, 7.0, 3.95, 4.0, 1.35, f"new h2\n{_vector(step.new_hidden)}", HIDDEN, 21)
    _arrow(axis, (9, 7.15), (9, 6.8), HIDDEN)
    studio_instrument(axis, 9, 6.2, "SHOW", HIDDEN, gid="show-control")
    _label(axis, 10.75, 6.2, "after tanh", 16, MUTED)
    _arrow(axis, (9, 5.62), (9, 5.34), HIDDEN)
    _panel(axis, 13.3, 5.75, 4.0, 2.85, "", INK)
    _label(axis, 15.3, 8.0, "Next read: D4\ngets BOTH (h2, c2)", 18)
    _label(axis, 15.3, 7.1, "c2: memory contents", 16, CELL)
    _label(axis, 15.3, 6.35, "h2: controls/proposal", 16, HIDDEN)
    _arrow(axis, (11.1, 7.9), (13.2, 7.5), CELL, gid="carry-c2")
    _label(axis, 12.15, 8.55, "carry c2", 16, CELL)
    _arrow(axis, (11.1, 4.6), (13.2, 6.45), HIDDEN, gid="carry-h2")
    _label(axis, 12.5, 4.9, "carry h2", 16, HIDDEN)
    studio_instrument(axis, 14.0, 2.8, "head", VIOLET, gid="prediction-head")
    _arrow(axis, (11.1, 4.4), (13.3, 2.8), VIOLET, gid="expose-h2")
    _arrow(axis, (14.72, 2.8), (15.0, 2.8), CELL, gid="note-scores")
    _label(axis, 16.15, 2.8, "Scores for\nnote 3", 19, CELL)
    _label(axis, 15.2, 1.85, "The note head reads h2, not c2.", 15, VIOLET)
    _label(axis, 9, 1.0,
           f"First coordinate of c2: kept {step.retained[0]:+.4f} + written {step.written[0]:+.4f}"
           f" = {step.new_cell[0]:+.4f}", 19)
    _label(axis, 9, 0.4, "These are computed feature values, not pitches, note IDs or probabilities.", 16, MUTED)


def draw_state_roles(axis, probe: StateRoleProbe) -> None:
    if probe.baseline.new_cell.numel() != 2:
        raise ValueError("This close-up requires the lesson's two-feature LSTM.")
    _canvas(axis)
    _label(axis, 9, 10.25, "Hide the readout. Keep the memory.", 27, weight="bold")
    _label(axis, 9, 9.55, "A3 after D3: change only SHOW in this current step.", 19)
    for x, title, step in (
        (1.0, "Original SHOW", probe.baseline),
        (10.0, "SHOW closed", probe.show_closed),
    ):
        center = x + 3.5
        _label(axis, center, 8.7, title, 23, HIDDEN)
        _panel(axis, x, 6.05, 7.0, 1.65, f"c2: memory contents\n{_vector(step.new_cell)}", CELL, 24)
        _arrow(axis, (center, 5.95), (center, 5.45), HIDDEN)
        studio_instrument(axis, center, 4.85, "SHOW" if step is probe.baseline else "0",
                          HIDDEN if step is probe.baseline else CORAL, gid=f"show-control-{title}")
        _label(axis, center - 1.9, 4.85, "readout\ncontrol", 16, MUTED)
        _label(axis, center + 1.9, 4.85, "after tanh", 16, MUTED)
        _arrow(axis, (center, 4.25), (center, 3.86), HIDDEN)
        _panel(axis, x, 2.15, 7.0, 1.65, f"h2: exposed readout\n{_vector(step.new_hidden)}", HIDDEN, 24)
        _label(axis, center, 1.55, "head reads h2; next step carries BOTH (h2, c2)", 15)
    _label(axis, 9, 0.65, "Zero h does not imply zero c. Hidden readout is not erased memory.", 19)


def export_state_roles(path: Path = NOTEBOOK, directory: Path = HERE / "images") -> None:
    probe = load_state_roles(path)
    figure, axis = _figure()
    draw_state_roles(axis, probe)
    directory.mkdir(exist_ok=True)
    target = directory / "01a-lstm-state-roles-trace.png"
    figure.savefig(target, dpi=100, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(path),
        "OriginalCell": json.dumps(probe.baseline.new_cell.tolist()),
        "ClosedCell": json.dumps(probe.show_closed.new_cell.tolist()),
        "OriginalHidden": json.dumps(probe.baseline.new_hidden.tolist()),
        "ClosedHidden": json.dumps(probe.show_closed.new_hidden.tolist()),
        "TraceOrigin": "actual A3-after-D3 LSTMCell; only current SHOW was closed",
    })
    plt.close(figure)
    print(f"Saved {target.name}: same c, different h; both are carried.")


def export_memory_diagrams(path: Path = NOTEBOOK, directory: Path = HERE / "images") -> None:
    probe = load_state_roles(path)
    directory.mkdir(exist_ok=True)
    for name, draw in (
        ("01a-lstm-cell-vs-hidden.png", lambda axis: draw_cell_memory(axis)),
        ("01a-lstm-two-state-step.png", lambda axis: draw_two_state_step(axis, probe)),
    ):
        figure, axis = _figure()
        draw(axis)
        figure.savefig(directory / name, dpi=100, facecolor=BACKGROUND, metadata={
            "SourceFingerprint": source_fingerprint(path), "Generator": "local music-studio diagram",
        })
        plt.close(figure)
        print(f"Saved {name}")


if __name__ == "__main__":
    export_memory_diagrams()
    export_state_roles()
