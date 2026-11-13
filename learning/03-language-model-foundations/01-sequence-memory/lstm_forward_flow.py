"""A native two-state LSTM forward pass with explicit h-only note predictions."""

import ast
import base64
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
import hashlib
from html import escape
from io import BytesIO
import json
from math import hypot
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from PIL import Image, ImageOps
import torch
from torch import Tensor, nn

from backprop_memory_flow import NOTES, NOTE_HEAD_WEIGHTS, _inputs_and_weights
from lstm_layer_flow import record_lstm_layer_step
from lstm_state_roles import record_state_roles


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01A1-sequence-memory-worked-details.ipynb"
PLAYER = HERE / "images" / "01a-lstm-forward-walkthrough.html"
STORYBOARD = PLAYER.with_suffix(".png")
GIF = PLAYER.with_suffix(".gif")
ART = HERE / "images" / "01a-lstm-state-objects.png"
ART_MANIFEST = HERE / "images" / "01a-lstm-forward-prompts.json"
COLORS = {
    "background": "#0d1b2e", "panel": "#20364f", "text": "#edf3ff",
    "muted": "#b0bed3", "cell": "#ffd172", "hidden": "#77dce5",
    "control": "#c5a3ff", "prediction": "#c5a3ff",
}
PHASES = (
    "Bring in the current note, old h and old c.",
    "The note and old h choose the controls. Old c takes a separate lane.",
    "KEEP retains selected old contents from c.",
    "WRITE admits selected proposed content.",
    "Add those two contributions to make new c.",
    "SHOW a bounded view of new c to make new h.",
    "Only new h enters the note head; c has no direct prediction arrow.",
    "Carry BOTH new c and new h into the next read.",
)
RECORDING = re.compile(
    r'<script id="recorded-lstm" type="application/json">\s*(.*?)\s*</script>', re.DOTALL,
)


@dataclass(frozen=True)
class Readout:
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    prediction: str


@dataclass(frozen=True)
class LSTMRead:
    index: int
    note: str
    next_note: str
    inputs: tuple[float, ...]
    old_hidden: tuple[float, ...]
    old_cell: tuple[float, ...]
    keep: tuple[float, ...]
    write: tuple[float, ...]
    proposal: tuple[float, ...]
    show: tuple[float, ...]
    retained: tuple[float, ...]
    written: tuple[float, ...]
    new_cell: tuple[float, ...]
    new_hidden: tuple[float, ...]
    readout: Readout
    closed_show_cell: tuple[float, ...]
    closed_show_hidden: tuple[float, ...]
    closed_show_readout: Readout


def load_cell_and_inputs(path: Path = NOTEBOOK) -> tuple[nn.LSTMCell, Tensor]:
    book = json.loads(path.read_bytes())
    source = "".join(next(cell for cell in book["cells"] if cell["id"] == "lstm-real-cell-parity")["source"])
    setup = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.With) and any(
            isinstance(child, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "lstm_real_cell" for target in child.targets
            ) for child in ast.walk(node)
        )
    ]
    if len(setup) != 1:
        raise ValueError("Expected one isolated native LSTM initialization in the lesson.")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=setup, type_ignores=[]), "lesson_lstm_initialization", "exec"), namespace)
    cell = namespace["lstm_real_cell"]
    if not isinstance(cell, nn.LSTMCell) or cell.input_size != 2 or cell.hidden_size != 2:
        raise TypeError("The walkthrough requires the lesson's two-input, two-feature LSTMCell.")
    inputs, _, _ = _inputs_and_weights(path)
    return cell, inputs


def note_readout(hidden: Tensor) -> Readout:
    if hidden.shape != (2,) or not torch.isfinite(hidden).all():
        raise ValueError("The note head reads one finite two-feature hidden state.")
    with torch.no_grad():
        weights = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=hidden.dtype, device=hidden.device)
        scores = weights @ hidden
        probabilities = scores.softmax(0)
    return Readout(tuple(scores.tolist()), tuple(probabilities.tolist()), NOTES[int(scores.argmax())])


def record_reads(path: Path = NOTEBOOK) -> tuple[LSTMRead, ...]:
    cell, inputs = load_cell_and_inputs(path)
    hidden = torch.zeros((1, 2), dtype=torch.float64)
    memory = torch.zeros_like(hidden)
    result = []
    for offset, (note, row) in enumerate(zip(NOTES, inputs, strict=True)):
        step = record_lstm_layer_step(cell, row[None], hidden, memory)
        closed = record_state_roles(cell, row[None], hidden, memory).show_closed
        result.append(LSTMRead(
            offset + 1, note, NOTES[(offset + 1) % len(NOTES)], tuple(step.input_row.tolist()),
            tuple(step.old_hidden.tolist()), tuple(step.old_cell.tolist()),
            tuple(step.forget_gate.tolist()), tuple(step.input_gate.tolist()),
            tuple(step.candidate.tolist()), tuple(step.output_gate.tolist()),
            tuple(step.retained.tolist()), tuple(step.written.tolist()),
            tuple(step.carried_cell.tolist()), tuple(step.carried_hidden.tolist()), note_readout(step.carried_hidden),
            tuple(closed.carried_cell.tolist()), tuple(closed.carried_hidden.tolist()), note_readout(closed.carried_hidden),
        ))
        hidden, memory = step.carried_hidden[None], step.carried_cell[None]
    return tuple(result)


def role_images() -> dict[str, Image.Image]:
    manifest = json.loads(ART_MANIFEST.read_text(encoding="utf-8"))[0]
    result = {}
    with Image.open(ART) as original:
        for role, bounds in manifest["crop_boxes"].items():
            left, top, right, bottom = bounds
            if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
                raise ValueError(f"The {role} artwork crop is outside the MAI image.")
            image = ImageOps.contain(original.crop(bounds).convert("RGB"), (240, 180), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (240, 180), COLORS["background"])
            canvas.paste(image, ((240 - image.width) // 2, (180 - image.height) // 2))
            result[role] = canvas
    if set(result) != {"cell", "hidden"}:
        raise ValueError("The MAI manifest must define cell and hidden artwork.")
    return result


def vector(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(str(int(value)) if value == int(value) else f"{value:.2f}" for value in values) + "]"


@dataclass(frozen=True)
class Label:
    x: float
    y: float
    text: str
    size: int = 20
    color: str = "text"
    anchor: str = "middle"
    mono: bool = False


@dataclass(frozen=True)
class Box:
    name: str
    bounds: tuple[float, float, float, float]
    role: str
    phases: tuple[int, ...]
    labels: tuple[Label, ...]
    art: str | None = None
    ready_progress: float = 1.


@dataclass(frozen=True)
class Wire:
    name: str
    source: str
    target: str
    points: tuple[tuple[float, float], ...]
    role: str
    phases: tuple[int, ...]


def prediction_head(
    readout: Readout, index: int, head_bounds: tuple[float, float, float, float],
    result_bounds: tuple[float, float, float, float], phase: int, *, bias: bool,
) -> tuple[Box, Box]:
    x, y, width, _ = head_bounds
    head = Box("head", head_bounds, "prediction", (phase,), (
        Label(x + width / 2, y + 28, "Linear / Dense head", 22, "prediction"),
        Label(x + width / 2, y + 60, f"scores = W_out h{index}" + (" + b" if bias else ""), 18),
        Label(x + width / 2, y + 88, "2 features -> 3 note scores", 16, "muted"),
        Label(x + width / 2, y + 122, vector(readout.scores), 16, "prediction", mono=True),
    ), ready_progress=.5)
    x, y, width, _ = result_bounds
    choice = ("Uniform: no preferred note" if len(set(readout.scores)) == 1
              else f"Next note: {readout.prediction} (max)")
    result = Box("prediction", result_bounds, "prediction", (phase,), (
        Label(x + width / 2, y + 30, "softmax(scores)", 22, "prediction"),
        Label(x + width / 2, y + 62, choice, 18),
        *(Label(x + 16, y + 95 + i * 31, note, 20, "prediction", "start")
          for i, note in enumerate(NOTES)),
        *(Label(x + width - 16, y + 95 + i * 31, f"{value:.4f}", 20, "prediction", "end", True)
          for i, value in enumerate(readout.probabilities)),
    ))
    return head, result


def scene(read: LSTMRead) -> tuple[list[Box], list[Wire], list[Label]]:
    t, old = read.index, read.index - 1
    boxes = [
        Box("old-cell", (20, 40, 200, 85), "cell", (0, 2), (
            Label(120, 67, f"c{old}: old workspace", 20, "cell"),
            Label(120, 102, vector(read.old_cell), 22, "cell", mono=True),
        )),
        Box("input", (20, 250, 200, 90), "text", (0, 1), (
            Label(120, 278, f"{read.note}: current note", 20),
            Label(120, 315, f"x{t} = {vector(read.inputs)}", 22, mono=True),
        )),
        Box("old-hidden", (20, 420, 200, 85), "hidden", (0, 1), (
            Label(120, 448, f"h{old}: old readout", 20, "hidden"),
            Label(120, 483, vector(read.old_hidden), 22, "hidden", mono=True),
        )),
        Box("controls", (260, 235, 280, 270), "control", (1,), (
            Label(400, 262, "Compute controls", 22),
            Label(400, 294, "Shared weights: note + old h", 18, "muted"),
            *(Label(276, y, name, 19, role, "start") for name, y, role in (
                ("KEEP", 327, "control"), ("WRITE", 368, "control"),
                ("NEW", 409, "cell"), ("SHOW", 450, "control"),
            )),
            *(Label(524, y, vector(values), 20, role, "end", True) for values, y, role in (
                (read.keep, 327, "control"), (read.write, 368, "control"),
                (read.proposal, 409, "cell"), (read.show, 450, "control"),
            )),
            Label(400, 487, "Old c is not mixed here", 17, "muted"),
        )),
        Box("kept", (280, 40, 180, 85), "cell", (2,), (
            Label(370, 66, "KEEP old", 22),
            Label(370, 102, vector(read.retained), 20, "cell", mono=True),
        )),
        Box("written", (560, 235, 180, 110), "cell", (3,), (
            Label(650, 262, "WRITE new", 22),
            Label(650, 294, vector(read.written), 20, "cell", mono=True),
            Label(650, 325, "WRITE x proposal", 16, "muted"),
        )),
        Box("new-cell", (750, 40, 330, 100), "cell", (4, 5, 7), (
            Label(972, 74, f"c{t}: new workspace", 22, "cell"),
            Label(972, 112, vector(read.new_cell), 22, "cell", mono=True),
        ), "cell"),
        Box("show", (760, 255, 320, 90), "hidden", (5,), (
            Label(920, 283, "Expose a bounded view", 22),
            Label(920, 319, f"SHOW x tanh(c{t})", 20, "hidden"),
        )),
        Box("new-hidden", (750, 415, 330, 100), "hidden", (5, 6, 7), (
            Label(972, 449, f"h{t}: new readout", 22, "hidden"),
            Label(972, 487, vector(read.new_hidden), 22, "hidden", mono=True),
        ), "hidden"),
        Box("prediction", (760, 560, 320, 174), "prediction", (6,), (
            Label(920, 590, f"Head reads h{t}, not c{t}", 22, "prediction"),
            Label(920, 622, "Uniform: no preferred note" if len(set(read.readout.scores)) == 1
                  else f"guess {read.readout.prediction}; actual next {read.next_note}", 18, "muted"),
            *(Label(779, y, note, 20, "text" if note == read.next_note else "prediction", "start")
              for note, y in zip(NOTES, (655, 686, 717))),
            *(Label(1064, y, f"{probability:.4f}", 20, "prediction", "end", True)
              for probability, y in zip(read.readout.probabilities, (655, 686, 717))),
        )),
        Box("next-cell", (20, 560, 300, 80), "cell", (7,), (
            Label(170, 589, f"Next old c = c{t}", 22, "cell"),
            Label(170, 625, vector(read.new_cell), 22, "cell", mono=True),
        )),
        Box("next-hidden", (365, 560, 300, 80), "hidden", (7,), (
            Label(515, 589, f"Next old h = h{t}", 22, "hidden"),
            Label(515, 625, vector(read.new_hidden), 22, "hidden", mono=True),
        )),
    ]
    wires = [
        Wire("cell-to-keep", "old-cell", "kept", ((226, 82), (274, 82)), "cell", (0, 2)),
        Wire("input-to-controls", "input", "controls", ((226, 295), (254, 295)), "text", (0, 1)),
        Wire("hidden-to-controls", "old-hidden", "controls", ((226, 462), (254, 462)), "hidden", (0, 1)),
        Wire("keep-control", "controls", "kept", ((370, 229), (370, 131)), "control", (1, 2)),
        Wire("write-control", "controls", "written", ((546, 368), (550, 368), (550, 275), (554, 275)), "control", (1, 3)),
        Wire("proposal-to-write", "controls", "written", ((546, 409), (650, 409), (650, 351)), "cell", (1, 3)),
        Wire("kept-to-add", "kept", "add", ((466, 82), (636, 82)), "cell", (2, 4)),
        Wire("written-to-add", "written", "add", ((650, 229), (650, 180), (660, 180), (660, 106)), "cell", (3, 4)),
        Wire("add-to-cell", "add", "new-cell", ((684, 82), (744, 82)), "cell", (4,)),
        Wire("cell-to-show", "new-cell", "show", ((970, 146), (970, 249)), "cell", (4, 5)),
        Wire("show-control", "controls", "show", ((546, 450), (744, 450), (744, 300), (754, 300)), "control", (1, 5)),
        Wire("show-to-hidden", "show", "new-hidden", ((970, 351), (970, 409)), "hidden", (5,)),
        Wire("hidden-to-head", "new-hidden", "prediction", ((970, 521), (970, 554)), "hidden", (6,)),
        Wire("carry-cell", "new-cell", "next-cell",
             ((1086, 90), (1098, 90), (1098, 762), (170, 762), (170, 646)), "cell", (7,)),
        Wire("carry-hidden", "new-hidden", "next-hidden",
             ((1086, 465), (1088, 465), (1088, 750), (515, 750), (515, 646)), "hidden", (7,)),
    ]
    labels = [
        Label(660, 91, "+", 29),
        Label(345, 540, "Carry BOTH to the next read", 22),
        Label(345, 685, "The next note + h choose controls;", 18, "hidden"),
        Label(345, 716, "c supplies retained contents.", 18, "cell"),
    ]
    return boxes, wires, labels


def gif_scene(read: LSTMRead) -> tuple[list[Box], list[Wire], list[Label]]:
    boxes, wires, _ = scene(read)
    positions = {
        "old-cell": (20, 40, 170, 85), "input": (20, 290, 170, 90),
        "old-hidden": (20, 470, 170, 85), "controls": (300, 245, 240, 300),
        "kept": (330, 40, 190, 85), "written": (620, 245, 170, 110),
        "new-cell": (870, 40, 210, 100), "show": (870, 300, 210, 110),
        "new-hidden": (870, 550, 210, 100),
        "next-cell": (20, 770, 300, 80), "next-hidden": (430, 770, 300, 80),
    }
    spaced = []
    titles = {
        "old-cell": f"old c{read.index - 1}", "old-hidden": f"old h{read.index - 1}", "input": f"Input: {read.note}",
        "controls": "Make gates + NEW", "kept": "KEEP x old c", "written": "WRITE x NEW",
        "new-cell": f"c{read.index}: stored", "show": "SHOW x tanh(c)",
        "new-hidden": f"h{read.index}: working view",
    }
    for box in boxes:
        if box.name == "prediction":
            continue
        x, y, width, height = positions[box.name]
        _, old_y, _, old_height = box.bounds
        labels = []
        for offset, label in enumerate(box.labels):
            if box.name in {"show", "written"} and offset > (0 if box.name == "show" else 1):
                continue
            label_x = x + (width / 2 if label.anchor == "middle"
                           else 16 if label.anchor == "start" else width - 16)
            label_y = y + (label.y - old_y) * height / old_height
            size = label.size
            text = label.text
            if offset == 0:
                text = titles.get(box.name, text)
            if box.name == "controls":
                if offset == 1:
                    text = "From note + old h"
                if label.mono:
                    size = 18
            elif box.name in {"old-cell", "old-hidden"} and label.mono:
                size = 20
            elif box.name == "written" and label.mono:
                size = 18
            elif box.name == "show":
                label_y = y + height / 2 + 6
            labels.append(replace(label, x=label_x, y=label_y, text=text, size=size))
        spaced.append(replace(box, bounds=(x, y, width, height), labels=tuple(labels), art=None))
    spaced.extend(prediction_head(read.readout, read.index, (840, 770, 240, 140),
                                  (840, 1030, 240, 174), 6, bias=False))
    controls = next(box for box in spaced if box.name == "controls")
    rows = {label.text: label.y for label in controls.labels}
    routes = {
        "cell-to-keep": ((196, 82), (324, 82)),
        "input-to-controls": ((196, 335), (294, 335)),
        "hidden-to-controls": ((196, 512), (294, 512)),
        "keep-control": ((425, 239), (425, 131)),
        "write-control": ((546, rows["WRITE"]), (580, rows["WRITE"]), (580, 285), (614, 285)),
        "proposal-to-write": ((546, rows["NEW"]), (705, rows["NEW"]), (705, 361)),
        "kept-to-add": ((526, 82), (656, 82)),
        "written-to-add": ((705, 239), (705, 185), (680, 185), (680, 106)),
        "add-to-cell": ((704, 82), (864, 82)),
        "cell-to-show": ((975, 146), (975, 294)),
        "show-control": ((546, rows["SHOW"]), (830, rows["SHOW"]), (830, 355), (864, 355)),
        "show-to-hidden": ((975, 416), (975, 544)),
        "hidden-to-head": ((975, 656), (975, 764)),
        "carry-cell": ((1086, 90), (1115, 90), (1115, 1235), (170, 1235), (170, 856)),
        "carry-hidden": ((1086, 600), (1100, 600), (1100, 1220), (580, 1220), (580, 856)),
    }
    labels = [
        Label(680, 91, "+", 29),
        Label(375, 735, "Carry BOTH to the next read", 22),
        Label(375, 895, "The next note + h choose controls;", 18, "hidden"),
        Label(375, 930, "c supplies retained contents.", 18, "cell"),
    ]
    spaced_wires = [
        replace(wire, points=routes[wire.name],
                target="head" if wire.name == "hidden-to-head" else wire.target)
        for wire in wires
    ]
    spaced_wires.append(Wire("head-to-prediction", "head", "prediction",
                            ((975, 916), (975, 1024)), "prediction", (6,)))
    return spaced, spaced_wires, labels


def svg_scene(read: LSTMRead, urls: dict[str, str]) -> str:
    boxes, wires, labels = scene(read)
    parts = ['<svg class="scene" viewBox="0 0 1110 790" role="group" aria-label="LSTM forward pass with separate cell, hidden and prediction routes">']
    parts.append("<title>Actual two-feature LSTM values. Only new h enters the prediction head; both states continue.</title>")
    parts.append("<defs>")
    for role, color in COLORS.items():
        parts.append(f'<marker id="arrow-{role}" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="{color}"/></marker>')
    parts.append("</defs>")
    for wire in wires:
        points = " ".join(f"{x},{y}" for x, y in wire.points)
        phases = " ".join(map(str, wire.phases))
        signal_kind = "control-signal" if wire.role == "control" else "content-signal"
        parts.append(
            f'<polyline id="{wire.name}" class="wire {signal_kind}" points="{points}" data-from="{wire.source}" '
            f'data-to="{wire.target}" data-phases="{phases}" stroke="{COLORS[wire.role]}" '
            f'marker-end="url(#arrow-{wire.role})"/>'
        )
        start_x, start_y = wire.points[0]
        motion = "M0,0 " + " ".join(
            f"L{x-start_x},{y-start_y}" for x, y in wire.points[1:]
        )
        parts.append(
            f'<circle class="signal {signal_kind}" data-phases="{phases}" aria-hidden="true" '
            f'cx="{start_x}" cy="{start_y}" r="{4 if wire.role == "control" else 6}" '
            f'fill="{COLORS[wire.role]}"><animateMotion dur="1.4s" begin="indefinite" '
            f'repeatCount="indefinite" path="{motion}"/></circle>'
        )
    parts.append('<circle id="add" cx="660" cy="82" r="18" fill="#20364f" stroke="#ffd172" stroke-width="2" data-phases="4"/>')

    def label_svg(label: Label) -> str:
        classes = ["numeric"] if label.mono else []
        if label.text.startswith(("guess ", "Uniform:")):
            classes.append("prediction-result")
        family = f' class="{" ".join(classes)}"' if classes else ""
        return (f'<text x="{label.x}" y="{label.y}" font-size="{label.size}" fill="{COLORS[label.color]}" '
                f'text-anchor="{label.anchor}"{family}>{escape(label.text)}</text>')

    for box in boxes:
        x, y, width, height = box.bounds
        phases = " ".join(map(str, box.phases))
        parts.append(f'<g id="{box.name}" class="node" data-role="{box.role}" data-phases="{phases}">')
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="#20364f" stroke="{COLORS[box.role]}" stroke-width="2"/>')
        if box.art:
            parts.append(f'<image class="role-art" data-art="{box.art}" href="{urls[box.art]}" x="{x + 12}" y="{y + 13}" width="98" height="74"/>')
        parts.extend(label_svg(label) for label in box.labels)
        if box.name == "prediction":
            for i, probability in enumerate(read.readout.probabilities):
                y = 643 + i * 31
                parts.append(f'<rect class="probability-track" x="828" y="{y}" width="150" height="12" rx="4" fill="#34455e"/>')
                parts.append(f'<rect class="probability" data-note="{NOTES[i]}" data-value="{probability}" x="828" y="{y}" width="{150 * probability}" height="12" rx="4" fill="#c5a3ff"/>')
        parts.append("</g>")
    parts.extend(label_svg(label) for label in labels)
    parts.append("</svg>")
    return "".join(parts)


def closed_readout(read: LSTMRead) -> LSTMRead:
    return replace(read, show=(0., 0.), new_hidden=read.closed_show_hidden,
                   readout=read.closed_show_readout)


def operation_readouts(read: LSTMRead) -> list[list[str]]:
    v = lambda values: "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"
    per_number = lambda left, right, result: [
        f"Feature {index}: {left[index]:+.4f} * {right[index]:+.4f} = {result[index]:+.4f}"
        for index in range(2)
    ]
    return [
        [f"{read.note} input: {v(read.inputs)}", f"Incoming h: {v(read.old_hidden)}",
         f"Incoming c: {v(read.old_cell)}"],
        [f"KEEP {v(read.keep)}; WRITE {v(read.write)}",
         f"Proposed content {v(read.proposal)}; SHOW {v(read.show)}",
         "Only the note and old h choose these controls; old c supplies contents."],
        ["KEEP * old c", *per_number(read.keep, read.old_cell, read.retained)],
        ["WRITE * proposed content", *per_number(read.write, read.proposal, read.written)],
        [f"Kept {v(read.retained)} + written {v(read.written)}",
         f"New stored c{read.index}: {v(read.new_cell)}",
         "This stored result has its own carry route, before making the h view."],
        [f"SHOW {v(read.show)} * tanh(c{read.index})", *[
            f"Feature {index}: {read.show[index]:.4f} * tanh({read.new_cell[index]:+.4f})"
            f" = {read.new_hidden[index]:+.4f}" for index in range(2)
        ]],
        [f"Stored c{read.index}: {v(read.new_cell)} (not sent directly to the head)",
         f"LSTM feature output h{read.index}: {v(read.new_hidden)}",
         f"Head scores: {v(read.readout.scores)}",
         f"softmax scores: {v(read.readout.probabilities)}; "
         + ("no preferred note (all scores tie)" if len(set(read.readout.scores)) == 1
            else f"guess note {read.index+1}: {read.readout.prediction}")],
        [f"Next read's old c = c{read.index}: {v(read.new_cell)}",
         f"Next read's old h = h{read.index}: {v(read.new_hidden)}",
         "The stored c is NOT replaced by its h view. Both states continue."],
    ]


def _recording(reads: tuple[LSTMRead, ...]) -> dict:
    return {
        "version": 1, "notes": NOTES, "state_width": 2, "phases": PHASES,
        "head_weights": NOTE_HEAD_WEIGHTS, "steps": [asdict(read) for read in reads],
        "readouts": [operation_readouts(read) for read in reads],
        "closed_steps": [asdict(closed_readout(read)) for read in reads],
        "closed_readouts": [operation_readouts(closed_readout(read)) for read in reads],
        "origin": "same native LSTMCell initialization and note codes as the lesson; initialized, not trained music",
    }


def recorded_data(path: Path = NOTEBOOK) -> dict:
    return _recording(record_reads(path))


PLAYER_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LSTM: two carried states, one prediction reader</title>
<style>
:root{font-family:system-ui,sans-serif;color:#edf3ff;background:#0d1b2e;color-scheme:dark;--cell:#ffd172;--hidden:#77dce5;--prediction:#c5a3ff}
*{box-sizing:border-box}body{margin:0}main{max-width:1120px;padding:14px;margin:auto}
h1{font-size:22px;margin:0 0 8px}p{line-height:1.35;margin:6px 0}.small{font-size:12px}.muted{color:#b0bed3}
.controls,.panel{background:#172941;border-radius:12px;padding:12px;margin-top:10px}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
button{font:600 13px system-ui;cursor:pointer;background:#20364f;color:#edf3ff;border:1px solid #7186a4;border-radius:8px;padding:7px 12px}
button[aria-pressed=true]{border-color:#ffd172;color:#ffd172}button:disabled{opacity:.55;cursor:default}
button:focus-visible{outline:3px solid #ffd172;outline-offset:3px}
.legend{font-size:13px;display:flex;gap:20px;flex-wrap:wrap}.cell-color{color:var(--cell)}.hidden-color{color:var(--hidden)}.prediction-color{color:var(--prediction)}
#phase{min-height:34px;font-size:13px;margin-top:8px;color:#edf3ff}
.viewport{overflow-x:auto}.scene{width:100%;height:auto;display:block}.scene text{font-family:system-ui,sans-serif}.scene text.numeric{font-family:ui-monospace,monospace}
.wire{fill:none;stroke-width:2;opacity:.38}.wire.control-signal{stroke-dasharray:6 4}
.wire.active{opacity:1;stroke-width:3}.wire.active.control-signal{animation:flow 1.2s linear infinite}
.signal{visibility:hidden}.signal.active{visibility:visible}.wire.future{opacity:.08}
.node.future{opacity:.32}.node.future .numeric,.node.future .prediction-result,.node.future .probability,.node.future .probability-track{visibility:hidden}
main[data-playing=false] .wire.active{animation-play-state:paused}
.node.active>rect,#add.active{stroke-width:4}.node>rect{transition:stroke-width .15s}
main[data-phase=all] .wire{opacity:1;animation:none;stroke-dasharray:none}
#active-lines{font-size:18px;line-height:1.55;overflow-wrap:anywhere}
#active-lines p{margin:4px 0}.outputs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.output-label{font-size:13px;color:#b0bed3}.output-value{display:block;font:18px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}
#preview-note{color:#ffd172;font-size:14px;line-height:1.5}
@keyframes flow{to{stroke-dashoffset:-18}}
details{font-size:12px;margin-top:8px}summary{cursor:pointer;color:#b0bed3}code{display:block;font:12px ui-monospace,monospace;line-height:1.5;overflow-wrap:anywhere;padding:7px;background:#0d1b2e;border-radius:6px;margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:6px;border-bottom:1px solid #344d68}td{font-family:ui-monospace,monospace;overflow-wrap:anywhere}
#problem{color:#ffac9e}
@media(max-width:850px){.outputs{grid-template-columns:repeat(2,minmax(0,1fr))}.guess-box{grid-column:1/-1}}
@media(max-width:600px){main{padding:10px}.panel,.controls{padding:10px}h1{font-size:20px}}
@media(prefers-reduced-motion:reduce){.wire.active{animation:none}.node>rect{transition:none}}
</style></head><body>
<main data-player="lstm-forward-walkthrough" data-phase="0" data-step="1" data-playing="false">
<h1>Read a note. Keep c. Show h. Guess the next note.</h1>
<p class="small muted">Press Play or Next stage. Follow the moving signals before reading the result; each read reuses the same weights.</p>
<p id="motion-note" class="small muted" hidden>Reduced motion: this view starts paused. Play phrase explicitly enables moving signals; Next stage stays static.</p>
<section class="controls" aria-label="LSTM walkthrough controls">
 <div class="row"><div id="note-buttons" role="group" aria-label="Choose a note read">
  <button type="button" data-note="0">D3</button><button type="button" data-note="1">A3</button><button type="button" data-note="2">D4</button>
 </div><div><button id="back" type="button">Back</button> <button id="play" type="button">Play phrase</button> <button id="pause" type="button">Resume</button> <button id="step" type="button">Next stage</button> <button id="reset" type="button">Reset</button></div></div>
 <div class="legend"><span class="cell-color">solid gold: stored/new content</span><span class="hidden-color">solid cyan: h readout</span><span class="prediction-color">dashed purple: gate controls</span><span class="muted">NEW is proposed content, not a fourth gate</span></div>
 <p id="phase" role="status"></p>
 <p id="progress" class="small muted"></p>
</section>
<p id="problem" role="alert" hidden></p>
<section class="panel" aria-label="Active operation, with full-size recorded arithmetic">
 <div id="active-lines" aria-live="polite"></div>
 <button id="preview" type="button" aria-pressed="false" disabled>Preview this read with SHOW = 0</button>
 <p id="preview-note" hidden>One-read comparison only: stored c is unchanged, h is zero and the head loses its context. Next returns to the normal recording before continuing.</p>
</section>
<section class="panel" aria-label="Measured LSTM forward diagram"><div id="scene" class="viewport"></div></section>
<section class="panel outputs" aria-label="Stored state, LSTM output and next-note prediction">
 <div><span class="output-label">Stored c, carried to the next read</span><strong id="cell-output" class="output-value cell-color"></strong></div>
 <div><span class="output-label">LSTM output h, also carried onward</span><strong id="hidden-output" class="output-value hidden-color"></strong></div>
 <div class="guess-box"><span class="output-label">Separate head: next-note output</span><strong id="prediction-output" class="output-value prediction-color"></strong></div>
</section>
<section class="panel" aria-label="LSTM role checks">
 <p class="small"><strong>Carry both states onward.</strong> Only h branches into the prediction head. These are two-feature states, not stored note names.</p>
 <details><summary>Open the arithmetic for this read</summary>
  <code id="cell-formula"></code><code id="hidden-formula"></code><code id="head-formula"></code>
  <p class="muted">KEEP/WRITE/SHOW and the proposal are computed by the same native LSTM weights at every note. In this standard cell, old c does not enter the gate projections.</p>
 </details>
 <details><summary>Prove that c is not fed directly to the note head</summary>
  <p class="muted">Isolated check: close only SHOW for this read, with its incoming values fixed.</p>
  <table><thead><tr><th></th><th>Normal</th><th>SHOW closed</th></tr></thead><tbody id="proof"></tbody></table>
  <p class="muted">c is unchanged, but h is zero. This bias-free head then gives equal note scores. The content is still stored; it is not exposed to the head.</p>
 </details>
</section>
<p class="small muted">Rounded native LSTMCell values and real head predictions. These weights are initialized, not trained music. MAI role art; measured labels remain separate.</p>
</main>
<script id="recorded-lstm" type="application/json">__DATA__</script>
<script>
(() => {
 const data=JSON.parse(document.getElementById("recorded-lstm").textContent);
 const root=document.querySelector("main"),scene=document.getElementById("scene");
 const playButton=document.getElementById("play"),pauseButton=document.getElementById("pause"),stepButton=document.getElementById("step");
 const backButton=document.getElementById("back"),previewButton=document.getElementById("preview");
 if(data.version!==1||data.state_width!==2||!Array.isArray(data.steps)||data.steps.length!==3
  ||!Array.isArray(data.scenes)||data.scenes.length!==3||!Array.isArray(data.closed_scenes)||data.closed_scenes.length!==3
  ||!Array.isArray(data.phases)||data.phases.length!==8
  ||!Array.isArray(data.readouts)||data.readouts.length!==3||data.readouts.some(rows=>!Array.isArray(rows)||rows.length!==8)
  ||!Array.isArray(data.closed_steps)||data.closed_steps.length!==3
  ||!Array.isArray(data.closed_readouts)||data.closed_readouts.length!==3||data.closed_readouts.some(rows=>!Array.isArray(rows)||rows.length!==8)){
  const alert=document.getElementById("problem");alert.hidden=false;alert.textContent="The recorded LSTM walkthrough is invalid. Regenerate it.";
  document.querySelectorAll("button").forEach(button=>button.disabled=true);throw new Error("Invalid LSTM forward recording.");
 }
 const reduced=window.matchMedia("(prefers-reduced-motion: reduce)").matches;
 const f=value=>Number.isInteger(value)?String(value):value.toFixed(4),v=values=>"["+values.map(f).join(", ")+"]";
 let selected=0,cursor=1,timer=null,currentPhase=0,preview=false,motionRequested=false;
 const total=data.steps.length*data.phases.length;
 function stop(){clearTimeout(timer);timer=null;root.dataset.playing="false";pauseButton.textContent=reduced&&!motionRequested?"Reduced motion":"Resume";
  const svg=scene.querySelector("svg");if(svg)svg.pauseAnimations();}
 function tableRow(label,before,after){const row=document.createElement("tr");for(const value of[label,v(before),v(after)]){const cell=document.createElement("td");cell.textContent=value;row.append(cell);}return row;}
 function render(index,phase){
  selected=index;currentPhase=phase;const read=(preview?data.closed_steps:data.steps)[index],t=read.index;
  const numericPhase=phase==="all"?7:phase;
  root.dataset.step=String(t);root.dataset.phase=String(phase);
  root.dataset.preview=String(preview);
  scene.innerHTML=(preview?data.closed_scenes:data.scenes)[index];
  scene.querySelectorAll("[data-phases]").forEach(node=>{
   const phases=node.dataset.phases.split(" ").map(Number);
   node.classList.toggle("active",phase!=="all"&&phases.includes(Number(phase)));
   node.classList.toggle("future",phase!=="all"&&Math.min(...phases)>numericPhase);
  });
  const svg=scene.querySelector("svg");
  if(root.dataset.playing==="true"&&(!reduced||motionRequested)){svg.unpauseAnimations();scene.querySelectorAll(".signal.active animateMotion").forEach(animation=>animation.beginElement());}
  else svg.pauseAnimations();
  document.querySelectorAll("#note-buttons button[data-note]").forEach(button=>button.setAttribute("aria-pressed",String(Number(button.dataset.note)===index)));
  document.getElementById("phase").textContent=phase==="all"?"After "+read.note+": c"+t+" and h"+t+" are ready; the head reads h"+t+".":data.phases[phase];
  document.getElementById("progress").textContent="Read "+t+" / 3 | operation "+(numericPhase+1)+" / 8 | no weights updated";
  const lines=(preview?data.closed_readouts:data.readouts)[index][numericPhase];
  const active=document.getElementById("active-lines");active.replaceChildren();
  lines.forEach(value=>{const row=document.createElement("p");row.textContent=value;active.append(row);});
  document.getElementById("cell-output").textContent=numericPhase>=4?v(read.new_cell):"Not computed yet";
  document.getElementById("hidden-output").textContent=numericPhase>=5?v(read.new_hidden):"Not computed yet";
  document.getElementById("prediction-output").textContent=numericPhase>=6
   ?(preview?"Uniform scores: no preferred note":"After "+read.note+", guess note "+(t+1)+": "+read.readout.prediction)
   :"Wait for the head to read h";
  previewButton.disabled=!(numericPhase===5||numericPhase===6||phase==="all");
  previewButton.setAttribute("aria-pressed",String(preview));
  document.getElementById("preview-note").hidden=!preview;
  document.getElementById("cell-formula").textContent="c"+t+" = kept + written = "+v(read.retained)+" + "+v(read.written)+" = "+v(read.new_cell);
  document.getElementById("hidden-formula").textContent="h"+t+" = SHOW * tanh(c"+t+") = "+v(read.show)+" * tanh("+v(read.new_cell)+") = "+v(read.new_hidden);
  document.getElementById("head-formula").textContent="scores = W_out * h"+t+" = "+v(read.readout.scores)+"; softmax[D3,A3,D4] = "+v(read.readout.probabilities);
  document.getElementById("proof").replaceChildren(
   tableRow("new c",read.new_cell,read.closed_show_cell),tableRow("new h",read.new_hidden,read.closed_show_hidden),
   tableRow("note probabilities",read.readout.probabilities,read.closed_show_readout.probabilities));
  root.dataset.cell=JSON.stringify(read.new_cell);root.dataset.hidden=JSON.stringify(read.new_hidden);
  root.dataset.probabilities=JSON.stringify(read.readout.probabilities);
  stepButton.disabled=cursor>=total;
  backButton.disabled=cursor<=1;
 }
 function finish(){stop();preview=false;cursor=total;render(data.steps.length-1,"all");pauseButton.textContent="Replay";}
 function advance(){if(cursor>=total){finish();return;}const position=cursor++;preview=false;render(Math.floor(position/data.phases.length),position%data.phases.length);}
 function tick(){if(cursor>=total){finish();return;}advance();timer=setTimeout(tick,2800);}
 function play(){stop();cursor=0;preview=false;motionRequested=true;pauseButton.disabled=false;root.dataset.playing="true";pauseButton.textContent="Pause";tick();}
 document.querySelectorAll("#note-buttons button[data-note]").forEach(button=>button.addEventListener("click",()=>{stop();selected=Number(button.dataset.note);preview=false;cursor=selected*data.phases.length+1;render(selected,0);}));
 playButton.addEventListener("click",play);
 pauseButton.addEventListener("click",()=>{if(timer!==null){stop();return;}if(cursor>=total){play();return;}root.dataset.playing="true";pauseButton.textContent="Pause";tick();});
 stepButton.addEventListener("click",()=>{stop();advance();});
 backButton.addEventListener("click",()=>{stop();const position=Math.max(0,cursor-2);cursor=position+1;preview=false;render(Math.floor(position/data.phases.length),position%data.phases.length);});
 document.getElementById("reset").addEventListener("click",()=>{stop();cursor=1;preview=false;render(0,0);});
 previewButton.addEventListener("click",()=>{stop();preview=!preview;render(selected,currentPhase);});
 document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
 window.addEventListener("pagehide",stop);
 render(0,0);
 if(reduced){pauseButton.disabled=true;pauseButton.textContent="Reduced motion";document.getElementById("motion-note").hidden=false;}
})();
</script></body></html>"""


def render_player(path: Path = NOTEBOOK) -> str:
    reads = record_reads(path)
    data = _recording(reads)
    images = role_images()
    urls = {}
    for role, image in images.items():
        buffer = BytesIO()
        image.save(buffer, format="WEBP", lossless=True, method=6)
        urls[role] = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    data["scenes"] = [svg_scene(read, urls) for read in reads]
    data["closed_scenes"] = [svg_scene(closed_readout(read), urls) for read in reads]
    return PLAYER_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False)) + "\n"


def read_recording(path: Path = PLAYER) -> dict:
    matches = list(RECORDING.finditer(path.read_text(encoding="utf-8")))
    if len(matches) != 1:
        raise ValueError("Expected one recorded-lstm data block.")
    return json.loads(matches[0][1])


def export_player(path: Path = PLAYER) -> None:
    raw = path.read_bytes() if path.exists() else None
    encoded = render_player().encode("utf-8")
    if raw == encoded:
        return
    with NamedTemporaryFile(dir=path.parent, prefix=".lstm-forward-", suffix=".tmp", delete=False) as staged:
        staged_path = Path(staged.name)
        staged.write(encoded)
    try:
        if (path.read_bytes() if path.exists() else None) != raw:
            raise RuntimeError("The LSTM walkthrough changed while it was being regenerated.")
        staged_path.replace(path)
    finally:
        staged_path.unlink(missing_ok=True)


def flow_signals(phase: int, progress: float) -> tuple[tuple[str, float], ...]:
    if not 0 <= phase < len(PHASES) or not 0 <= progress <= 1:
        raise ValueError("Choose an LSTM operation and a signal progress between zero and one.")
    routes = {
        0: ("input-to-controls", "hidden-to-controls"),
        1: (),
        2: ("cell-to-keep", "keep-control"),
        3: ("write-control", "proposal-to-write"),
        7: ("carry-cell", "carry-hidden"),
    }
    if phase in (4, 5, 6):
        first, second = {
            4: (("kept-to-add", "written-to-add"), ("add-to-cell",)),
            5: (("cell-to-show", "show-control"), ("show-to-hidden",)),
            6: (("hidden-to-head",), ("head-to-prediction",)),
        }[phase]
        return tuple((name, progress * 2 if progress <= .5 else (progress - .5) * 2)
                     for name in (first if progress <= .5 else second))
    return tuple((name, progress) for name in routes[phase])


def _signal_position(wire: Wire, progress: float) -> tuple[float, float]:
    lengths = [hypot(end[0] - start[0], end[1] - start[1])
               for start, end in zip(wire.points, wire.points[1:])]
    distance = progress * sum(lengths)
    for start, end, length in zip(wire.points, wire.points[1:], lengths):
        if distance <= length:
            fraction = distance / length
            return (start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]))
        distance -= length
    return wire.points[-1]


def draw_forward_scene(
    axis, boxes: list[Box], wires: list[Wire], labels: list[Label], *,
    phase: int | None, progress: float, completed_phase: int,
    signals_for: Callable[[int, float], tuple[tuple[str, float], ...]],
    probabilities: tuple[float, ...], images: dict[str, Image.Image] | None = None,
) -> None:
    signals = dict(signals_for(phase, progress)) if phase is not None else {}
    visited = set(signals)
    if phase is not None:
        for previous in range(max(completed_phase, phase - 1) + 1):
            for position in (0., 1.):
                visited.update(name for name, _ in signals_for(previous, position))
        if progress >= .5:
            visited.update(name for name, _ in signals_for(phase, 0.))
    for wire in wires:
        alpha = 1. if phase is None or wire.name in visited else .18
        line_width = 1.4 if phase is None else 2.2
        for start, end in zip(wire.points[:-2], wire.points[1:-1]):
            axis.plot((start[0], end[0]), (start[1], end[1]), color=COLORS[wire.role], linewidth=line_width,
                      linestyle="--" if wire.role == "control" else "-", alpha=alpha,
                      gid=f"wire-{wire.name}")
        arrowprops = {"arrowstyle": "->", "color": COLORS[wire.role], "lw": line_width,
                      "linestyle": "--" if wire.role == "control" else "-", "alpha": alpha,
                      "gid": f"wire-{wire.name}"}
        if phase is not None:
            arrowprops["mutation_scale"] = 14
        axis.annotate("", wire.points[-1], wire.points[-2],
                      arrowprops=arrowprops)
        if wire.name in signals:
            axis.add_patch(plt.Circle(_signal_position(wire, signals[wire.name]),
                                      5 if wire.role == "control" else 7,
                                      color=COLORS[wire.role], zorder=5))
    font_scale = .82 if phase is None else .70
    for box in boxes:
        x, y, width, height = box.bounds
        ready = -1 if box.name in {"input", "old-cell", "old-hidden"} else min(box.phases)
        computed = phase is None or phase > ready or (phase == ready and progress >= box.ready_progress)
        alpha = 1. if phase is None or min(box.phases) <= max(phase, completed_phase) else .42
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=8",
                                     facecolor=COLORS["panel"], edgecolor=COLORS[box.role],
                                     linewidth=1.5, alpha=alpha, gid=f"node-{box.name}"))
        if box.art and images is not None:
            axis.imshow(images[box.art], extent=(x + 12, x + 110, y + 87, y + 13), aspect="auto", zorder=2)
        elif box.art and computed:
            axis.text(x + 62, y + 60, "STORE" if box.art == "cell" else "VIEW",
                      ha="center", color=COLORS[box.role], fontsize=16)
        for offset, label in enumerate(box.labels):
            if not computed and offset:
                continue
            axis.text(label.x, label.y, label.text, color=COLORS[label.color], fontsize=label.size * font_scale,
                      ha={"middle": "center", "start": "left", "end": "right"}[label.anchor],
                      family="monospace" if label.mono else "sans-serif", alpha=alpha)
    addition = next(label for label in labels if label.text == "+")
    axis.add_patch(plt.Circle((addition.x, addition.y - 9), 18, facecolor=COLORS["panel"],
                              edgecolor=COLORS["cell"], linewidth=1.5))
    carry_phase = min(next(box for box in boxes if box.name == "next-hidden").phases)
    for label in labels:
        if phase is None or max(phase, completed_phase) >= carry_phase or label.text == "+":
            axis.text(label.x, label.y, label.text, color=COLORS[label.color], fontsize=label.size * font_scale,
                      ha={"middle": "center", "start": "left", "end": "right"}[label.anchor])
    prediction = next(box for box in boxes if box.name == "prediction")
    head_phase = min(prediction.phases)
    if phase is None or phase > head_phase or (phase == head_phase and progress == 1.):
        x, y, _, _ = prediction.bounds
        for i, probability in enumerate(probabilities):
            axis.add_patch(FancyBboxPatch((x + (68 if phase is None else 60), y + 83 + i * 31),
                                         (150 if phase is None else 75) * probability, 12,
                                         boxstyle="round,pad=0,rounding_size=3",
                                         facecolor=COLORS["prediction"], edgecolor="none"))


def _draw_diagram(axis, read: LSTMRead, images: dict[str, Image.Image] | None, *,
                  phase: int | None = None, progress: float = 1., completed_phase: int = -1) -> None:
    boxes, wires, labels = scene(read) if phase is None else gif_scene(read)
    axis.set(xlim=(0, 1110 if phase is None else 1130), ylim=(790 if phase is None else 1250, 0))
    axis.axis("off")
    draw_forward_scene(axis, boxes, wires, labels, phase=phase, progress=progress,
                       completed_phase=completed_phase, signals_for=flow_signals,
                       probabilities=read.readout.probabilities, images=images)


def export_storyboard(path: Path = STORYBOARD) -> None:
    read = record_reads()[1]
    figure, axis = plt.subplots(figsize=(13, 9.1), facecolor=COLORS["background"])
    figure.subplots_adjust(left=0.015, right=0.985, top=0.925, bottom=0.035)
    _draw_diagram(axis, read, role_images())
    figure.suptitle("A3 + old h choose HOW. Old c supplies WHAT. Only new h feeds the note head.",
                    color=COLORS["text"], fontsize=15)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [label for label in axis.texts if label.get_text()]
    for offset, label in enumerate(texts):
        bounds = label.get_window_extent(renderer)
        for other in texts[offset + 1:]:
            if bounds.overlaps(other.get_window_extent(renderer)):
                plt.close(figure)
                raise RuntimeError(f"LSTM labels overlap: {label.get_text()} / {other.get_text()}")
    for image in axis.images:
        left, right, bottom, top = image.get_extent()
        points = axis.transData.transform(((left, bottom), (right, top)))
        bounds = Bbox.from_extents(min(points[:, 0]), min(points[:, 1]), max(points[:, 0]), max(points[:, 1]))
        for label in axis.texts:
            if label.get_window_extent(renderer).overlaps(bounds):
                plt.close(figure)
                raise RuntimeError(f"LSTM text overlaps MAI artwork: {label.get_text()}")
    figure.savefig(path, dpi=120, facecolor=COLORS["background"], metadata={
        "TraceOrigin": "native two-feature LSTMCell; A3 after D3; note head reads new h only",
        "HeadInput": "new_hidden", "StateWidth": "2",
    })
    plt.close(figure)


@dataclass(frozen=True)
class ForwardFrame:
    read_index: int
    phase: int
    progress: float
    duration_ms: int
    close_show: bool = False
    completed_phase: int = -1


def gif_frames() -> tuple[ForwardFrame, ...]:
    stages = [(2, phase, False) for phase in range(7)]
    stages += [(2, 5, True), (2, 6, True), (2, 7, False), (3, 0, False), (3, 1, False)]
    frames = []
    completed = -1
    for index, phase, closed in stages:
        for progress in (0., .25, .5, .75, 1.):
            frames.append(ForwardFrame(index, phase, progress, 1600 if progress == 1. else 120,
                                       closed, completed))
        completed = max(completed, phase)
    return tuple(frames)


GIF_CAPTIONS = (
    "A new note arrives. Bring the previous stored contents c and working view h.",
    "The note + old h choose THREE fractions and one proposed write. The gates do not feed one another.",
    "KEEP: how much of each OLD stored amount should survive?",
    "WRITE: how much of each proposed NEW amount may enter?",
    "Add the kept and admitted amounts. This is the new stored contents c.",
    "SHOW: expose a bounded view of c as h. Making a view does NOT replace the store.",
    "A learned Linear head makes note scores. Softmax + the largest probability gives the next note.",
    "Carry BOTH c and h. The next input uses these states, not the predicted note.",
)


def draw_forward_header(figure: Figure, title: str, caption: str, detail: str, footer: str) -> None:
    figure.text(.5, .967, title, ha="center", va="top", fontsize=21, color=COLORS["cell"])
    words = caption.split()
    lines, line = [], ""
    for word in words:
        if len(line) + len(word) + 1 > 93:
            lines.append(line)
            line = ""
        line += (" " if line else "") + word
    lines.append(line)
    figure.text(.5, .925, "\n".join(lines), ha="center", va="top", fontsize=13.5,
                color=COLORS["text"], linespacing=1.4)
    figure.text(.5, .845, detail,
                ha="center", va="center", fontsize=12, color=COLORS["hidden"])
    figure.text(.5, .012, footer,
                ha="center", fontsize=10, color=COLORS["muted"])


def gif_figure(read: LSTMRead, frame: ForwardFrame):
    shown = closed_readout(read) if frame.close_show else read
    figure, axis = plt.subplots(figsize=(11.3, 14.85), dpi=100, facecolor=COLORS["background"])
    figure.subplots_adjust(left=.015, right=.985, top=.79, bottom=.035)
    _draw_diagram(axis, shown, None, phase=frame.phase, progress=frame.progress,
                  completed_phase=frame.completed_phase)
    title = f"Read {read.index}: {read.note} | {frame.phase + 1}/8"
    if frame.close_show:
        title += " | WHAT IF SHOW = 0?"
    elif frame.phase == 7:
        title += " | restore the normal read"
    caption = ("Same stored c. Close SHOW: h becomes zero; the head has no preferred note."
               if frame.close_show else GIF_CAPTIONS[frame.phase])
    readouts = operation_readouts(shown)[frame.phase]
    detail = readouts[0] if frame.phase not in (2, 3, 5, 6) else readouts[1]
    if frame.phase == 6:
        detail = f"head(h) scores {vector(shown.readout.scores)}; softmax = {vector(shown.readout.probabilities)}"
    draw_forward_header(
        figure, title, caption,
        detail if frame.progress == 1. else "Follow the moving contents and dashed controls.",
        "Gold: stored contents | Cyan: working h | Dashed purple: gate controls | Initialized, not trained.",
    )
    axis.plot((25, 25 + 1060 * frame.progress), (-20, -20), color=COLORS["cell"],
              linewidth=3, clip_on=False)
    return figure


def write_forward_gif(path: Path, figures: Iterable[Figure], durations: list[int], metadata: dict) -> None:
    palette = Image.new("P", (1, 1))
    colors = []
    background = tuple(int(COLORS["background"][i:i + 2], 16) for i in (1, 3, 5))
    for color in COLORS.values():
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        for level in range(24):
            colors.extend(round(old + (new - old) * level / 23) for old, new in zip(background, rgb))
    palette.putpalette(colors + [0] * (768 - len(colors)))
    images = []
    for figure in figures:
        try:
            figure.canvas.draw()
            image = Image.frombuffer("RGBA", figure.canvas.get_width_height(),
                                     figure.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
            images.append(image.quantize(palette=palette, dither=Image.Dither.NONE))
        finally:
            plt.close(figure)
    images[0].save(
        path, save_all=True, append_images=images[1:], loop=0, disposal=1, optimize=True,
        duration=durations,
        comment=json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8"),
    )


def export_gif(path: Path = GIF) -> None:
    reads = record_reads()
    specifications = gif_frames()
    metadata = {
        "version": 1, "head_input": "new_hidden", "state_width": 2,
        "steps": [asdict(read) for read in reads],
        "frames": [asdict(frame) for frame in specifications],
        "origin": "Native LSTMCell; A3 after D3, then D4; SHOW=0 is an isolated comparison.",
    }
    write_forward_gif(
        path, (gif_figure(reads[frame.read_index - 1], frame) for frame in specifications),
        [frame.duration_ms for frame in specifications], metadata,
    )


if __name__ == "__main__":
    export_player()
    export_storyboard()
    export_gif()
    print("Exported the native LSTM player, inline gate-by-gate GIF and static role map.")
