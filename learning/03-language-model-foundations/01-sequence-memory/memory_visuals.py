"""Spacious, recorded-state motion graphics shared by the 01A memory replays."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import fill
from typing import TYPE_CHECKING, Literal

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from torch import Tensor

if TYPE_CHECKING:
    from gru_layer_flow import LayerStep
    from lstm_layer_flow import LSTMLayerStep
    from memory_gate_flows import MemoryStep


MOTION_FRAMES = 6
INTERVAL_MS = 180
BACKGROUND, TEXT = "#101b2f", "#f0f5ff"
BLUE, AMBER, PURPLE, GREEN = "#79c9ff", "#ffc184", "#c8b0ff", "#9aefc4"
NATIVE_LABELS = ("Read", "Mix", "Controls", "Keep/write", "Combine", "Expose", "Carry")
GRU_LABELS = ("Read", "Mix", "Controls", "Proposal", "Keep/write", "Combine", "Carry")
SCALAR_LABELS = ("Read", "Controls", "Keep/write", "Combine", "Carry")


@dataclass(frozen=True)
class MemoryScene:
    kind: Literal["lstm", "gru"]
    note: str
    old: tuple[float, ...]
    keep: tuple[float, ...]
    write: tuple[float, ...]
    proposal: tuple[float, ...]
    retained: tuple[float, ...]
    written: tuple[float, ...]
    hidden: tuple[float, ...]
    cell: tuple[float, ...] | None = None
    show: tuple[float, ...] | None = None
    reset: tuple[float, ...] | None = None
    inputs: tuple[float, ...] | None = None
    old_hidden: tuple[float, ...] | None = None
    input_mix: tuple[float, ...] | None = None
    state_mix: tuple[float, ...] | None = None


def _values(tensor: Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.detach().reshape(-1).tolist())


def lstm_scene(step: LSTMLayerStep, note="A3") -> MemoryScene:
    return MemoryScene(
        "lstm", note, _values(step.old_cell), _values(step.forget_gate),
        _values(step.input_gate), _values(step.candidate), _values(step.retained),
        _values(step.written), _values(step.output), cell=_values(step.carried_cell),
        show=_values(step.output_gate), inputs=_values(step.input_row),
        old_hidden=_values(step.old_hidden), input_mix=_values(step.input_mix),
        state_mix=_values(step.state_mix),
    )


def gru_scene(step: LayerStep, note="A3") -> MemoryScene:
    keep = _values(step.update_gate)
    return MemoryScene(
        "gru", note, _values(step.old_state), keep, tuple(1 - value for value in keep),
        _values(step.candidate), _values(step.retained), _values(step.written),
        _values(step.output), reset=_values(step.reset_gate), inputs=_values(step.input_row),
        old_hidden=_values(step.old_state), input_mix=_values(step.input_mix),
        state_mix=_values(step.state_mix),
    )


def scalar_scene(kind: str, step: MemoryStep) -> MemoryScene:
    if kind not in ("lstm", "gru"):
        raise ValueError("A memory scene must be LSTM or GRU.")
    if kind == "lstm" and step.show is None:
        raise ValueError("The recorded LSTM scene needs its SHOW control.")
    if kind == "gru" and step.reset is None:
        raise ValueError("The recorded GRU scene needs its RESET control.")
    return MemoryScene(
        "lstm" if kind == "lstm" else "gru", step.note, (step.old,), (step.keep,), (step.write,), (step.proposal,),
        (step.retained,), (step.written,), (step.exposed,),
        cell=(step.carried,) if kind == "lstm" else None,
        show=(step.show,) if step.show is not None else None,
        reset=(step.reset,) if step.reset is not None else None,
    )


def _number(values: tuple[float, ...]) -> str:
    sample = ", ".join(f"{value:+.4f}" for value in values[:2])
    return sample if len(values) == 1 else "[" + sample + (", ...]" if len(values) > 2 else "]")


def draw_scene(axis, scene: MemoryScene, phase: int | None, *, native: bool,
               progress=0.0, note_position: str = "") -> None:
    labels = (NATIVE_LABELS if scene.kind == "lstm" else GRU_LABELS) if native else SCALAR_LABELS
    if scene.kind not in ("lstm", "gru") or (phase is not None and not 0 <= phase < len(labels)):
        raise ValueError("Select a valid recorded memory phase.")
    if scene.kind == "lstm" and (scene.cell is None or scene.show is None):
        raise ValueError("LSTM graphics require both the carried cell and its SHOW controls.")
    if scene.kind == "gru" and scene.reset is None:
        raise ValueError("GRU graphics require RESET controls.")
    axis.clear()
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")
    gradient = np.linspace(0, 1, 180)[:, None]
    bottom, top = np.array([16, 27, 47]), np.array([29, 47, 75])
    pixels = (bottom + gradient[..., None] * (top - bottom)) / 255
    axis.imshow(pixels, extent=(0, 1, 0, 1), aspect="auto", zorder=-10)
    title = f"{scene.kind.upper()}  /  {scene.note}  /  {'one real layer' if native else 'one memory coordinate'}"
    axis.text(0.04, 0.955, title, color=TEXT, fontsize=26, weight="bold", va="top", gid="heading")
    origin = "Initialized cell, not fitted music" if native else "Chosen scalar controls, not trained gates"
    axis.text(0.04, 0.895, origin + ("  |  " + note_position if note_position else ""),
              color=BLUE, fontsize=18, va="top", gid="origin")
    for index, label in enumerate(labels):
        x = 0.04 + index * 0.92 / len(labels)
        active = phase == index
        axis.add_patch(FancyBboxPatch(
            (x, 0.80), 0.92 / len(labels) - 0.012, 0.044,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor=GREEN if active else "#293c59", zorder=0,
        ))
        axis.text(x + 0.004, 0.822, label, va="center", color=BACKGROUND if active else TEXT,
                  fontsize=14, weight="bold" if active else "normal", gid=f"navigation-{index}")
        if active:
            axis.add_patch(Circle((x + 0.008 + progress * (0.92 / len(labels) - 0.028), 0.797),
                                 0.004, color=GREEN, zorder=5))

    serial = 0

    def panel(x, y, w, h, title, values=None, detail="", color=BLUE, gate=False):
        nonlocal serial
        serial += 1
        key = f"card-{serial}"
        axis.add_patch(FancyBboxPatch((x + 0.006, y - 0.009), w, h,
                                     boxstyle="round,pad=0.008", facecolor="#070e1c",
                                     edgecolor="none", alpha=0.45, zorder=1))
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                            facecolor="#20314c", edgecolor=color, linewidth=1.7,
                            zorder=2, gid=key)
        axis.add_patch(box)
        axis.text(x + 0.018, y + h - 0.023, title, va="top", color=TEXT,
                  fontsize=18, weight="bold", zorder=3, gid=key + ":title")
        if values is not None:
            if gate:
                for row, value in enumerate(values[:2]):
                    if not 0 <= value <= 1:
                        raise ValueError("A displayed gate must be in [0,1].")
                    yy = y + h - 0.150 - row * 0.095
                    axis.add_patch(FancyBboxPatch(
                        (x + 0.018, yy), w - 0.036, 0.025, boxstyle="round,pad=0.002",
                        facecolor="#101b2f", edgecolor="none", zorder=3,
                    ))
                    if value > 0:
                        axis.add_patch(FancyBboxPatch(
                            (x + 0.018, yy), (w - 0.036) * value, 0.025,
                            boxstyle="round,pad=0.002", facecolor=color, edgecolor="none", alpha=0.7, zorder=4,
                        ))
                    label = f"feature {row}: {value:.4f}" if len(values) > 1 else f"{value:.4f}"
                    axis.text(x + 0.018, yy + 0.071, label, color=color, fontsize=18,
                              va="top", zorder=5, gid=key + f":gate-{row}")
            else:
                axis.text(x + 0.018, y + h - 0.082, _number(values), va="top",
                          color=color, fontsize=20, zorder=3, gid=key + ":value")
        if detail:
            width = max(16, int(w * 92))
            axis.text(x + 0.018, y + 0.026, fill(detail, width), va="bottom",
                      color=TEXT, fontsize=18, zorder=3, gid=key + ":detail")
        return box

    def link(start, end, color):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                       color=color, linewidth=2, zorder=0, gid="connector"))
        t = progress * progress * (3 - 2 * progress)
        x, y = (start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1]))
        axis.add_patch(Circle((x, y), 0.011, color=color, alpha=0.15, zorder=1))
        axis.add_patch(Circle((x, y), 0.0045, color=color, zorder=1))

    result = scene.cell if scene.kind == "lstm" else scene.hidden
    state_name = "cell c" if scene.kind == "lstm" else "hidden h"
    event = labels[phase] if phase is not None else "Overview"
    hint = ""
    if phase is None:
        panel(0.05, 0.51, 0.42, 0.23, "KEPT old memory", scene.retained, "KEEP * old " + state_name)
        panel(0.53, 0.51, 0.42, 0.23, "WRITTEN proposal", scene.written, "WRITE * proposal", AMBER)
        panel(0.20, 0.20, 0.60, 0.22, "New " + state_name, result,
              ("h shown: " + _number(scene.hidden)) if scene.kind == "lstm" else "Output now = h carried onward",
              GREEN)
        link((0.26, 0.49), (0.40, 0.44), BLUE)
        link((0.74, 0.49), (0.60, 0.44), AMBER)
        hint = "Follow the two contributions, not a maze of wiring. The replay opens each operation."
    elif event == "Read":
        if native:
            if scene.inputs is None or scene.old_hidden is None:
                raise ValueError("A native step needs its input and previous hidden.")
            panel(0.05, 0.37, 0.28, 0.30, "Input " + scene.note, scene.inputs, "Current input profile", AMBER)
            panel(0.36, 0.37, 0.28, 0.30, "Previous h", scene.old_hidden, "Exposed history")
            if scene.kind == "lstm":
                panel(0.67, 0.37, 0.28, 0.30, "Previous c", scene.old, "Carried memory", PURPLE)
            hint = "The token arrives with previous state. These numbers are features, not generated notes."
        else:
            panel(0.05, 0.37, 0.42, 0.30, "Old " + state_name, scene.old, "What arrived from the last note")
            panel(0.53, 0.37, 0.42, 0.30, "Proposed content", scene.proposal,
                  "Chosen teaching value" if scene.kind == "lstm" else "Calculated using input and reset-controlled old history",
                  AMBER)
            hint = "One familiar note, one recorded coordinate. The controls were chosen for inspection."
    elif event == "Mix":
        if scene.inputs is None or scene.input_mix is None or scene.state_mix is None:
            raise ValueError("A projection view needs actual recorded Linear outputs.")
        chunks = 4 if scene.kind == "lstm" else 3
        hidden_width = len(scene.hidden)
        panel(0.05, 0.36, 0.42, 0.33, f"Linear: {len(scene.inputs)} -> {chunks * hidden_width}",
              scene.input_mix, "Input ingredients; " + ("i / f / g / o" if chunks == 4 else "r / z / n"), AMBER)
        panel(0.53, 0.36, 0.42, 0.33, f"Linear: {hidden_width} -> {chunks * hidden_width}",
              scene.state_mix, "Old-h ingredients; includes bias")
        hint = "The two learned mixes supply gate and proposal ingredients. LSTM old c bypasses them." if chunks == 4 else "Two learned mixes, reused at every token. RESET later acts on projected old history."
    elif event == "Controls":
        panel(0.05, 0.42, 0.28, 0.30, "KEEP f" if scene.kind == "lstm" else "KEEP z",
              scene.keep, color=BLUE, gate=True)
        panel(0.36, 0.42, 0.28, 0.30, "WRITE i" if scene.kind == "lstm" else "WRITE 1-z",
              scene.write, color=AMBER, gate=True)
        panel(0.67, 0.42, 0.28, 0.30, "SHOW o" if scene.kind == "lstm" else "RESET r",
              scene.show if scene.kind == "lstm" else scene.reset, color=PURPLE, gate=True)
        if scene.kind == "lstm" or not native:
            panel(0.20, 0.20, 0.60, 0.15, "Proposal (tanh)", scene.proposal, color=AMBER)
        hint = "Dimmers are per feature. The proposal is signed content, not a fourth gate." if scene.kind == "lstm" else "RESET controls proposal history, not the kept branch. WRITE is the complement of KEEP."
    elif event == "Proposal":
        if scene.input_mix is None or scene.state_mix is None or scene.reset is None:
            raise ValueError("The native proposal view requires its recorded projections.")
        width = len(scene.hidden)
        input_part = scene.input_mix[-width:]
        old_part = tuple(reset * value for reset, value in zip(scene.reset, scene.state_mix[-width:]))
        panel(0.05, 0.51, 0.42, 0.21, "Input ingredients", input_part, color=AMBER)
        panel(0.53, 0.51, 0.42, 0.21, "RESET * projected old h", old_part, color=BLUE)
        panel(0.20, 0.20, 0.60, 0.22, "Proposed content n", scene.proposal, "tanh(input part + old part)", AMBER)
        link((0.26, 0.49), (0.40, 0.44), AMBER)
        link((0.74, 0.49), (0.60, 0.44), BLUE)
        hint = "RESET acts AFTER the old-state projection. The separate retained-old route is untouched."
    elif event == "Keep/write":
        panel(0.05, 0.37, 0.42, 0.30, "KEPT old memory", scene.retained,
              "KEEP * old " + state_name, BLUE)
        panel(0.53, 0.37, 0.42, 0.30, "WRITTEN proposal", scene.written,
              "WRITE * proposal", AMBER)
        hint = "These two contributions are calculated separately. Neither is the final state by itself."
    elif event == "Combine":
        panel(0.05, 0.51, 0.42, 0.21, "KEPT", scene.retained, color=BLUE)
        panel(0.53, 0.51, 0.42, 0.21, "WRITTEN", scene.written, color=AMBER)
        panel(0.20, 0.20, 0.60, 0.22, "New " + state_name, result, "KEPT + WRITTEN", GREEN)
        link((0.26, 0.49), (0.40, 0.44), BLUE)
        link((0.74, 0.49), (0.60, 0.44), AMBER)
        hint = "An additive carry route, not a rewrite of the whole transcript."
    elif event == "Expose":
        if scene.kind == "lstm":
            panel(0.05, 0.37, 0.28, 0.31, "Squash c", tuple(float(np.tanh(value)) for value in scene.cell),
                  "tanh(new c)", BLUE)
            panel(0.36, 0.37, 0.28, 0.31, "SHOW o", scene.show, color=PURPLE, gate=True)
            panel(0.67, 0.37, 0.28, 0.31, "Exposed h", scene.hidden, "SHOW * tanh(c)", GREEN)
            link((0.34, 0.52), (0.35, 0.52), PURPLE)
            link((0.65, 0.52), (0.66, 0.52), GREEN)
            hint = "SHOW chooses the exposed view. Hiding c does not delete its carried value."
        else:
            panel(0.20, 0.35, 0.60, 0.31, "Exposed hidden h", scene.hidden, "Same new h; no separate SHOW gate", GREEN)
            hint = "The GRU's one h is both exposed and carried. There is no second c lane."
    else:
        panel(0.05, 0.37, 0.42, 0.30, "Output now: h", scene.hidden, "To caller / next layer", GREEN)
        panel(0.53, 0.37, 0.42, 0.30, "Next token: (h, c)" if scene.kind == "lstm" else "Next token: h",
              scene.hidden, "c = " + _number(scene.cell) if scene.cell is not None else "Same h values, another job", BLUE)
        hint = "The next LSTM token needs BOTH h and c. A separate head scores notes." if scene.kind == "lstm" else "Two routes, the SAME h values. These features are not note scores."
    axis.text(0.04, 0.075, fill(hint, 83), color=GREEN, fontsize=18,
              va="center", linespacing=1.45, gid="takeaway")
