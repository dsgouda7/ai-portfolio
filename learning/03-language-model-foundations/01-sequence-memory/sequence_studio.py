"""Musical-studio graphics for a recurrent state with carry and prediction branches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path
import numpy as np

from musical_memory import MIDI

if TYPE_CHECKING:
    from backprop_memory_flow import BackpropTrace


BACKGROUND = "#081426"
TEXT, MUTED = "#f4f1e8", "#a9bad1"
CYAN, VIOLET, GOLD, CORAL = "#75e1e3", "#c5a6ff", "#ffcd76", "#ff9c8d"
POSITIONS = (0.45, 6.2, 11.95)
PLAYER_STYLE = (
    "<style>body{background:#081426;color:#f4f1e8}"
    ".animation .anim-buttons button{background:#14283e;color:#f4f1e8;"
    "border:1px solid #53667f;border-radius:7px;padding:5px 9px}"
    ".animation input{accent-color:#75e1e3}"
    ".animation .anim-controls{color:#f4f1e8}</style>"
)


def studio_figure():
    figure, axis = plt.subplots(figsize=(16, 10), facecolor=BACKGROUND)
    figure.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
    return figure, axis


def _text(axis, x, y, text, size=18, color=TEXT, *, weight="normal", gid=None):
    return axis.text(
        x, y, text, ha="center", va="center", fontsize=size, color=color,
        weight=weight, linespacing=1.25, zorder=10, gid=gid,
    )


def _flow(axis, points, color, gid, *, alpha=1.0):
    path = Path(points, [Path.MOVETO] + [Path.CURVE4] * (len(points) - 1))
    for width, opacity in ((13, 0.045), (7, 0.10)):
        axis.add_patch(PathPatch(
            path, facecolor="none", edgecolor=color, linewidth=width,
            alpha=opacity * alpha, zorder=2,
        ))
    axis.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=15, linewidth=2,
        color=color, alpha=alpha, zorder=3, gid=gid,
    ))


def _card(axis, x, y, width, height, color, *, gid=None, alpha=1.0):
    fold = min(0.25, width / 5)
    vertices = (
        (x, y), (x + width, y), (x + width, y + height - fold),
        (x + width - fold, y + height), (x, y + height), (x, y),
    )
    path = Path(vertices, [Path.MOVETO] + [Path.LINETO] * 4 + [Path.CLOSEPOLY])
    axis.add_patch(PathPatch(
        path, facecolor="#142b42", edgecolor=color, linewidth=2,
        alpha=alpha, zorder=5, gid=gid,
    ))
    axis.plot(
        (x + width - fold, x + width - fold, x + width),
        (y + height, y + height - fold, y + height - fold),
        color=color, alpha=alpha * 0.6, linewidth=1, zorder=6,
    )


def _keyboard(axis, x, y, note, *, visible):
    white = [pitch for pitch in range(48, 72) if pitch % 12 not in (1, 3, 6, 8, 10)]
    width = 1.65 / len(white)
    active = MIDI[note] if visible else None
    for index, pitch in enumerate(white):
        axis.add_patch(Rectangle(
            (x + index * width, y), width * 0.94, 0.6,
            facecolor=GOLD if pitch == active else "#e2e8ec",
            edgecolor="#324050", linewidth=0.5, alpha=1 if visible else 0.25, zorder=5,
        ))
        if pitch % 12 in (0, 2, 5, 7, 9):
            axis.add_patch(Rectangle(
                (x + (index + 0.72) * width, y + 0.29), width * 0.52, 0.31,
                facecolor="#07101b", edgecolor="#67768b", linewidth=0.3, zorder=6,
            ))


def _instrument(axis, x, y, text, color, *, gid, visible=True):
    for radius, alpha in ((0.79, 0.04), (0.68, 0.10)):
        axis.add_patch(Circle((x, y), radius, color=color, alpha=alpha, zorder=1))
    axis.add_patch(Circle(
        (x, y), 0.56, facecolor="#14283e", edgecolor=color,
        linewidth=2, alpha=1 if visible else 0.25, zorder=5, gid=gid,
    ))
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        axis.plot(
            (x + 0.63 * np.cos(angle), x + 0.69 * np.cos(angle)),
            (y + 0.63 * np.sin(angle), y + 0.69 * np.sin(angle)),
            color=color, alpha=0.55, linewidth=1,
        )
    _text(axis, x, y, text, size=18, color=color)


def draw_studio(axis, trace: BackpropTrace, *, stage: int, caption: str,
                progress=0.0, truncated_trace: BackpropTrace | None = None, overview=False):
    axis.clear()
    axis.set(xlim=(0, 18), ylim=(0, 11))
    axis.axis("off")
    axis.figure.set_facecolor(BACKGROUND)
    yy, xx = np.mgrid[0:1:100j, 0:1:160j]
    light = np.exp(-((xx - 0.5) ** 2 / 0.28 + (yy - 0.7) ** 2 / 0.55))
    base = np.array([8, 20, 38])
    pixels = (base + light[..., None] * np.array([17, 26, 35])) / 255
    axis.imshow(pixels, extent=(0, 18, 0, 11), aspect="auto", zorder=-10)
    for row in range(5):
        axis.plot((0.2, 17.8), (2.1 + row * 0.10,) * 2,
                  color=CYAN, alpha=0.055, linewidth=0.8)
    _text(axis, 9, 10.45, "One card. Two useful paths.", 27, weight="bold")
    _text(axis, 9, 9.84, caption, 18)
    for x, name, color in ((3, "CARRY MEMORY", CYAN), (9, "EXPOSE h TO THE HEAD", VIOLET),
                           (15, "GUESS THE NEXT NOTE", GOLD)):
        axis.scatter((x - 1.75,), (9.22,), s=24, color=color, zorder=6)
        _text(axis, x, 9.22, name, 13, color)

    shown = 3 if overview else min(stage + 1, 3)
    active = 6 - stage if 4 <= stage <= 6 else None
    view = truncated_trace if stage == 8 and truncated_trace is not None else trace
    for index, (x, note) in enumerate(zip(POSITIONS, ("D3", "A3", "D4"))):
        visible = index < shown
        opacity = 1 if visible else 0.25
        state_color = CORAL if active == index else CYAN
        _text(axis, x + 1.65, 8.53, f"Input: {note} + h{index}", 19, gid=f"input-pair-{index}")
        _text(axis, x + 0.7, 7.98, f"Input\n{note}", 16, GOLD)
        _text(axis, x + 2.6, 7.98, f"Incoming\nh{index}", 16, CYAN)
        _keyboard(axis, x - 0.12, 6.92, note, visible=visible)
        _text(axis, x + 1.77, 7.25, "+", 22, MUTED)
        _card(axis, x + 2.08, 6.92, 1.05, 0.65, CYAN, alpha=opacity)
        for line in (7.12, 7.29):
            axis.plot((x + 2.23, x + 2.83), (line, line), color=CYAN, alpha=0.4)
        _flow(axis, ((x + 0.7, 6.87), (x + 0.7, 6.65), (x + 1.1, 6.55), (x + 1.3, 6.35)),
              GOLD, f"note-input-{index}", alpha=opacity)
        _flow(axis, ((x + 2.6, 6.87), (x + 2.6, 6.65), (x + 2.2, 6.55), (x + 2.0, 6.35)),
              CYAN, f"state-input-{index}", alpha=opacity)
        _instrument(axis, x + 1.65, 5.78, "RNN", CYAN, gid=f"reader-{index}", visible=visible)
        _text(axis, x + 1.65, 4.93, "shared weights", 12, MUTED)
        _flow(axis, ((x + 1.65, 5.13), (x + 1.65, 4.95), (x + 1.65, 4.7), (x + 1.65, 4.55)),
              state_color, f"cell-result-{index}", alpha=opacity)
        _card(axis, x + 0.15, 3.32, 3.0, 1.17, state_color, gid=f"state-card-{index}", alpha=opacity)
        state = trace.states[index + 1]
        role = "New" if stage < 4 else "Saved"
        detail = f"after {note}" if overview else (
            f"[{state[0]:+.3f}, {state[1]:+.3f}]" if visible else "not read yet"
        )
        _text(axis, x + 1.65, 3.91, f"{role} h{index + 1}\n{detail}", 17, state_color,
              gid=f"state-label-{index}")

        _flow(axis, ((x + 3.2, 3.95), (x + 3.8, 4.15), (x + 3.7, 5.1), (x + 4.08, 5.32)),
              VIOLET, f"expose-output-{index}", alpha=opacity)
        _text(axis, x + 3.08, 4.88, f"expose h{index + 1}", 13, VIOLET)
        _instrument(axis, x + 4.35, 5.78, "head", VIOLET, gid=f"head-{index}", visible=visible)
        _text(axis, x + 4.35, 6.61, "same note head", 12, VIOLET)
        _flow(axis, ((x + 4.35, 5.12), (x + 4.35, 4.95), (x + 4.35, 4.8), (x + 4.35, 4.65)),
              GOLD, f"prediction-output-{index}", alpha=opacity)
        for line in range(5):
            axis.plot((x + 3.42, x + 5.28), (4.11 + line * 0.1,) * 2,
                      color=GOLD, alpha=0.18)
        axis.add_patch(Ellipse((x + 4.18, 4.31), 0.30, 0.17, angle=22,
                              facecolor=GOLD, alpha=opacity, zorder=7))
        axis.plot((x + 4.31, x + 4.31), (4.32, 4.84), color=GOLD, alpha=opacity, linewidth=2)
        prediction = trace.predictions[index] if visible else "?"
        _text(axis, x + 4.35, 3.69, f"guess n{index + 2}: {prediction}", 18, GOLD,
              gid=f"prediction-label-{index}")
        if visible:
            confidence = float(trace.scores[index].softmax(0).max())
            _text(axis, x + 4.35, 3.13, f"p = {confidence:.2f}", 12, MUTED)

        cx = x + 1.65
        if index < 2:
            gap = x + 5.38
            destination = POSITIONS[index + 1] + 3.2
            points = (
                (cx, 3.25), (cx, 2.4), (cx + 0.8, 1.9), (cx + 1.6, 1.9),
                (gap, 1.9), (gap, 2.0), (gap, 3.2),
                (gap, 8.93), (gap, 8.93), (destination + 0.3, 8.93),
                (destination + 0.3, 8.6), (destination + 0.3, 7.24), (destination, 7.24),
            )
        else:
            points = ((cx, 3.25), (cx, 1.9), (x + 3.3, 1.9), (x + 5.35, 1.9))
        _flow(axis, points, CYAN, f"carry-state-{index}",
              alpha=opacity * (0.40 if stage >= 4 else 0.85))
        _text(axis, x + 3.35, 1.65, f"carry h{index + 1}" + (" onward" if index == 2 else ""), 13, CYAN)
        if stage >= 4 and not overview:
            signal = view.feedback[index]
            feedback = "next backward checks" if active is not None and index < active else (
                "feedback stopped" if signal is None else f"feedback size: {signal.norm():.4f}"
            )
            _text(axis, cx, 2.82, feedback, 12, CORAL if signal is None else MUTED)

    if stage in (3, 4) and not overview:
        x = POSITIONS[-1]
        _flow(axis, ((x + 4.35, 4.65), (x + 4.35, 4.8), (x + 4.35, 4.95), (x + 4.35, 5.12)),
              CORAL, "prediction-error")
        _flow(axis, ((x + 4.08, 5.32), (x + 3.7, 5.1), (x + 3.8, 4.15), (x + 3.2, 3.95)),
              CORAL, "head-backward")
    if active is not None and active > 0:
        start, end = POSITIONS[active] + 1.65, POSITIONS[active - 1] + 1.65
        _flow(axis, ((start, 3.2), (start, 1.1), (end, 1.1), (end, 3.2)),
              CORAL, "backward-feedback")
        marker_x = start + (end - start) * (3 * progress ** 2 - 2 * progress ** 3)
        marker_y = ((1 - progress) ** 3 + progress ** 3) * 3.2 + 3 * progress * (1 - progress) * 1.1
        axis.add_patch(Circle((marker_x, marker_y), 0.08,
                              facecolor=CORAL, zorder=8, gid="feedback-marker"))
    if stage == 8:
        axis.plot((5.95, 5.95), (1.0, 8.65), color=CORAL, linestyle="--", linewidth=1.7)
    if overview or stage < 3:
        message = "Carry the card into the next read. Expose the SAME card to the note head."
    elif stage == 3:
        message = f"Note 4: guess {trace.guess}; target D3. The other guesses are not graded in this trace."
    elif stage == 7:
        message = f"Collect feedback, then update ONCE: loss {trace.loss_before:.4f} -> {trace.loss_after:.4f}."
    elif stage == 8:
        message = "Detach after h1: its values still enter h2; backward feedback stops at the boundary."
    else:
        message = "Inspect the saved input/state pair. Neither the old card nor the input is rewritten."
    _text(axis, 9, 0.73, message, 15)
    _text(axis, 9, 0.28, "Initialized reader and shared head. Actual cards and guesses, not learned musical skill.", 11, MUTED)
