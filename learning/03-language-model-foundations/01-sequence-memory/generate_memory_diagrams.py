"""Rebuild the chapter's static intuition sketches locally, without a model or network."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.switch_backend("Agg")
DIRECTORY = Path(__file__).resolve().parent / "images"
INK = "#17324d"
TEAL = "#dff2ef"
ORANGE = "#ffe0b5"
PURPLE = "#e9e1f4"
GRAY = "#e8e9ea"
PAPER = "#fffaf1"


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=PAPER)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax.set(xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    label(ax, 8, 8.5, title, size=25, weight="bold")
    label(ax, 8, 7.95, subtitle, size=15)
    return fig, ax


def label(ax, x, y, text, size=16, weight="normal", color=INK, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, fontweight=weight,
            color=color, linespacing=1.45)


def card(ax, x, y, width, height, text, color=TEAL, size=17):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=color, edgecolor=INK, linewidth=1.3,
    )
    ax.add_patch(patch)
    label(ax, x + width / 2, y + height / 2, text, size=size)


def arrow(ax, start, end, text=None, color=INK):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2, color=color,
    ))
    if text:
        label(ax, (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.3,
              text, size=13, color=color)


def save(fig, name):
    DIRECTORY.mkdir(exist_ok=True)
    fig.savefig(DIRECTORY / name, dpi=110, facecolor=PAPER)
    plt.close(fig)
    print(name)


def order_story():
    fig, ax = canvas(
        "Same notes. Different tune.",
        "Counting notes throws away a distinction the listener can hear.",
    )
    for y, title, notes in [
        (6.1, "Phrase A", ["D3", "A3", "D4"]),
        (4.0, "Phrase B", ["D4", "A3", "D3"]),
    ]:
        label(ax, 1.25, y + 0.45, title)
        for index, note in enumerate(notes):
            card(ax, 2.6 + index * 1.6, y, 1.2, 0.9, note)
        arrow(ax, (7.5, y + 0.45), (9, y + 0.45))
        card(ax, 9.2, y - 0.1, 5.4, 1.1, "D3: once   A3: once   D4: once", GRAY, size=16)
    label(ax, 8, 2.85, "The count cards match. The ordered rows do not.", weight="bold")
    card(ax, 1.1, 1.1, 13.8, 1.1,
         "Keep a seat for each step: one phrase, three positions, one note at each position.", PURPLE)
    label(ax, 8, 0.45, "A fixed ordered window can keep order too. Recurrence adds a reusable update, not order by magic.", size=13)
    save(fig, "01a-01-order-is-not-a-bag.png")


def notecard_story():
    fig, ax = canvas(
        "One reader. A changing notecard.",
        "The three boxes below are three uses of the SAME update rule.",
    )
    positions = [1.1, 6.0, 10.9]
    notes = ["D3", "A3", "D4"]
    for index, (x, note) in enumerate(zip(positions, notes)):
        card(ax, x + 0.7, 6.25, 2.4, 0.8, f"Read {note}", ORANGE)
        arrow(ax, (x + 1.9, 6.15), (x + 1.9, 5.55))
        card(ax, x, 4.25, 3.8, 1.2, "Same update rule\nsame weights", PURPLE)
        arrow(ax, (x + 1.9, 4.15), (x + 1.9, 3.55))
        card(ax, x + 0.15, 2.45, 3.5, 0.95, f"Notecard after {note}")
        if index < 2:
            arrow(ax, (x + 3.7, 2.95), (positions[index + 1] - 0.1, 4.65))
    label(ax, 0.45, 4.85, "blank\ncard", size=12)
    arrow(ax, (0.5, 4.8), (0.95, 4.8))
    label(ax, 8, 1.45, "Current note + old card -> fresh card. Then pass that card onward.", size=20, weight="bold")
    label(ax, 8, 0.55, "The card is a short row of numbers, not stored English. New phrases start with a fresh blank card.", size=14)
    save(fig, "01a-02-one-reader-changing-card.png")


def feedback_story():
    fig, ax = canvas(
        "Reading goes forward. Teaching comes back.",
        "Imagine the final prediction is wrong. Which earlier uses of the reader helped cause it?",
    )
    for x, text in [(1.0, "Read D3\nupdate card"), (5.7, "Read A3\nupdate card"), (10.4, "Read D4\nupdate card")]:
        card(ax, x, 4.6, 3.4, 1.3, text, TEAL)
    arrow(ax, (4.5, 5.25), (5.5, 5.25), "carry")
    arrow(ax, (9.2, 5.25), (10.2, 5.25), "carry")
    card(ax, 10.4, 2.0, 3.4, 1.1, "Check final guess", ORANGE)
    arrow(ax, (12.1, 4.45), (12.1, 3.2))
    arrow(ax, (13.2, 3.2), (13.2, 4.45), color="#a7501b")
    arrow(ax, (10.3, 2.55), (7.4, 2.55), "feedback", "#a7501b")
    arrow(ax, (7.2, 2.55), (2.7, 2.55), "feedback crosses another update", "#a7501b")
    arrow(ax, (2.7, 2.7), (2.7, 4.45), color="#a7501b")
    arrow(ax, (7.4, 2.7), (7.4, 4.45), color="#a7501b")
    label(ax, 8, 6.85, "Forward: what should the next card contain?", size=18)
    label(ax, 8, 1.1, "A longer gap means more transitions for feedback to cross.", size=20, weight="bold")
    label(ax, 8, 0.4, "Backpropagation calculates shared-weight gradients. The optimizer changes weights, not the input notes.", size=14)
    save(fig, "01a-03-forward-memory-backward-feedback.png")


def gate_story():
    fig, ax = canvas(
        "Keep some. Write some. Show some.",
        "An LSTM has three controls and a proposed update, not four separate memories.",
    )
    card(ax, 0.4, 5.45, 2.4, 1.1, "Old carried\nmemory")
    card(ax, 3.7, 5.45, 3.1, 1.1, "Forget control\nhow much to KEEP", ORANGE, size=16)
    arrow(ax, (2.95, 6.0), (3.55, 6.0))
    card(ax, 0.4, 2.65, 2.4, 1.1, "Proposed\nnew content", PURPLE)
    card(ax, 3.7, 2.65, 3.1, 1.1, "Input control\nhow much to WRITE", ORANGE, size=16)
    arrow(ax, (2.95, 3.2), (3.55, 3.2))
    label(ax, 7.65, 4.6, "+", size=32, weight="bold")
    arrow(ax, (6.95, 5.8), (7.7, 4.95))
    arrow(ax, (6.95, 3.4), (7.7, 4.25))
    card(ax, 8.7, 4.0, 2.4, 1.2, "New carried\nmemory")
    arrow(ax, (8.0, 4.6), (8.55, 4.6))
    arrow(ax, (9.9, 5.35), (9.9, 6.7))
    label(ax, 11.7, 7.0, "Carry onward", size=16)
    card(ax, 12.0, 3.95, 3.1, 1.3, "Output control\nSHOW after tanh", ORANGE, size=16)
    arrow(ax, (11.25, 4.6), (11.85, 4.6))
    arrow(ax, (13.55, 3.8), (13.55, 2.85))
    card(ax, 12.0, 1.65, 3.1, 1.05, "Exposed hidden\nstate", PURPLE)
    label(ax, 5.5, 1.75, "Close WRITE: ignore a distraction.\nClose SHOW: hide memory without erasing it.", size=17)
    label(ax, 8, 0.5, "Real gates are learned from today's input and the previous hidden state. The next code chooses them by hand.", size=14)
    save(fig, "01a-04-keep-write-show.png")


def forecast_story():
    fig, ax = canvas(
        "Slide the question. Keep the answer in the future.",
        "One question contains 24 observed values; its answer is the next value.",
    )
    for y, start, end, target in [(6.1, 0, 23, 24), (4.65, 1, 24, 25)]:
        card(ax, 1.0, y, 8.9, 0.85, f"Observed: time {start}, {start + 1}, ... , {end}")
        arrow(ax, (10.1, y + 0.42), (11.3, y + 0.42))
        card(ax, 11.5, y, 3.4, 0.85, f"Answer: time {target}", ORANGE)
    card(ax, 0.9, 2.15, 6.6, 1.25, "Training answers\nat times 24 through 545", TEAL)
    card(ax, 8.5, 2.15, 6.6, 1.25, "Validation answers\nat times 546 through 719", PURPLE)
    label(ax, 8, 1.2, "The first validation question may use earlier observations. Its answer is still later.", size=17)
    label(ax, 8, 0.45, "A random window split mixes neighboring histories and changes the question from forecasting to interpolation.", size=14)
    save(fig, "01b-01-window-and-later-answer.png")


def shift_story():
    fig, ax = canvas(
        "The answer row is one note ahead.",
        "The final note A4 is a training answer, not an input at the F4 position.",
    )
    label(ax, 1.25, 6.55, "Original")
    for index, note in enumerate(["D3", "A3", "D4", "F4", "A4"]):
        card(ax, 3.0 + 2.1 * index, 6.1, 1.5, 0.9, note, GRAY)
    label(ax, 1.25, 4.6, "Inputs")
    label(ax, 1.25, 2.5, "Answers")
    for index, (note, answer) in enumerate(zip(["D3", "A3", "D4", "F4"], ["A3", "D4", "F4", "A4"])):
        x = 3.0 + 2.1 * index
        card(ax, x, 4.15, 1.5, 0.9, note)
        card(ax, x, 2.05, 1.5, 0.9, answer, ORANGE)
        arrow(ax, (x + 0.75, 4.0), (x + 0.75, 3.1))
    label(ax, 13.65, 3.5, "Score every\npossible note;\ncheck this answer.", size=15)
    label(ax, 8, 1.05, "At D4, the reader may use D3, A3, D4. It must guess F4 without reading F4 yet.", size=18, weight="bold")
    label(ax, 8, 0.4, "The arrows pair predictions with answers; they do not feed the answer backward into the reader.", size=14)
    save(fig, "01b-02-shift-the-answer-not-the-question.png")


def packing_story():
    fig, ax = canvas(
        "Stopping the reader is not the same as marking the test.",
        "The shortest real input is D3 A3. Three blank slots make it as wide as the longest row.",
    )
    for y, title, packed in [(5.9, "Without packing", False), (3.9, "With packing", True)]:
        label(ax, 1.75, y + 0.4, title, size=16)
        for index, note in enumerate(["D3", "A3", "pad", "pad", "pad"]):
            x = 3.4 + index * 1.5
            card(ax, x, y, 1.1, 0.8, note, GRAY if index >= 2 else TEAL, size=16)
            if index < 4 and (not packed or index == 0):
                arrow(ax, (x + 1.18, y + 0.4), (x + 1.42, y + 0.4))
            if packed and index >= 2:
                ax.plot([x + 0.08, x + 1.02], [y + 0.08, y + 0.72], color="#9a5540", linewidth=2)
        if packed:
            ax.plot([6.15, 6.15], [y - 0.1, y + 0.95], color="#9a5540", linewidth=3)
            label(ax, 6.15, y + 1.15, "STOP", size=13, color="#9a5540")
            label(ax, 8.7, y - 0.45, "Blank seats aren't read.", size=13)
            label(ax, 13.0, y + 0.4, "Stop after A3.\nKeep its final card.", size=16)
        else:
            label(ax, 13.0, y + 0.4, "Three more updates.\nFinal card can drift.", size=16)
    card(ax, 0.9, 1.6, 6.7, 1.15, "PACKING\nStop state updates at each real length.", PURPLE, size=16)
    card(ax, 8.4, 1.6, 6.7, 1.15, "LOSS MASK\nGrade real answers; skip blank targets.", ORANGE, size=16)
    label(ax, 8, 0.7, "Unpacking restores blank output slots. The prediction head can still score them, so loss must skip them.", size=15)
    save(fig, "01b-03-stop-reading-skip-grading.png")


def causal_story():
    fig, ax = canvas(
        "Change a later note. Earlier guesses must stay put.",
        "Two sequences share the same first three input notes. Only position 3 changes.",
    )
    for y, title, notes in [
        (6.0, "Original", ["D3", "A3", "D4", "F4"]),
        (4.1, "Edited", ["D3", "A3", "D4", "C4"]),
    ]:
        label(ax, 1.25, y + 0.4, title)
        for index, note in enumerate(notes):
            card(ax, 3.1 + 2.5 * index, y, 1.7, 0.9, note, TEAL if index < 3 else ORANGE)
    card(ax, 2.9, 1.8, 6.9, 1.15, "At D3, A3, D4:\nidentical prefixes -> identical score rows", TEAL, size=17)
    card(ax, 10.5, 1.8, 4.5, 1.15, "At the edited position:\nscores may change", ORANGE, size=17)
    label(ax, 8, 0.75, "An earlier guess need not be correct. This test checks access to future inputs, not prediction skill.", size=16)
    save(fig, "01b-04-future-edit-same-prefix.png")


def piano_story():
    fig, ax = canvas(
        "Same clock. Different key. The opening cue must survive.",
        "Two versions of the same authored phrase differ by three semitones.",
    )
    for y, title, cue, ending in [
        (5.8, "D-minor version", "D3 cue", "D4 ending"),
        (3.7, "F-minor version", "F3 cue", "F4 ending"),
    ]:
        label(ax, 1.65, y + 0.5, title, size=16)
        card(ax, 3.5, y, 2.4, 1.0, cue, ORANGE)
        card(ax, 6.7, y, 4.2, 1.0, "Same repeating clock\nno more pitch cues", TEAL, size=16)
        card(ax, 11.7, y, 3.2, 1.0, ending, PURPLE)
        arrow(ax, (6.0, y + 0.5), (6.55, y + 0.5))
        arrow(ax, (11.0, y + 0.5), (11.55, y + 0.5))
    label(ax, 8, 2.2, "A clock-only model sees identical inputs but is asked for different endings.", size=19, weight="bold")
    label(ax, 8, 1.3, "Train on both versions. Then swap only the opening cue and measure the ending's pitch change.", size=17)
    label(ax, 8, 0.5, "Success is evidence on these two practiced phrases, not proof of new-key generalization or composing skill.", size=15)
    save(fig, "01c-01-same-clock-different-key.png")


if __name__ == "__main__":
    for draw in (
        order_story, notecard_story, feedback_story, gate_story, forecast_story,
        shift_story, packing_story, causal_story, piano_story,
    ):
        draw()
