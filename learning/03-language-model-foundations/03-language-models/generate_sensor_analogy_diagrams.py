"""Rebuild the registry, sensor-profile and observatory diagrams locally."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
TOKEN_IMAGES = HERE.parent / "02-tokenization-and-embeddings" / "images"
NAVY = "#172b4d"
TEAL = "#007f86"
PURPLE = "#7950a3"
ORANGE = "#bf5700"


def canvas(title: str):
    figure, axis = plt.subplots(figsize=(14, 7.875))
    figure.patch.set_facecolor("#fffaf1")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(0.5, 0.95, title, ha="center", color=NAVY, fontsize=22, weight="bold")
    return figure, axis


def card(axis: Axes, x: float, y: float, width: float, height: float,
         text: str, color: str = TEAL):
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012",
        edgecolor=color, facecolor="white", linewidth=2,
    ))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              color=NAVY, fontsize=13, linespacing=1.7)


def arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = NAVY):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops=dict(arrowstyle="->", color=color, lw=2))


def save(figure, path: Path):
    figure.savefig(path, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Saved {path.name}")


def registry():
    figure, axis = canvas("An ID identifies. A sensor profile supplies features.")
    card(axis, 0.03, 0.57, 0.20, 0.20, "Token: ' cat'\nRegistry ID: 3797")
    card(axis, 0.33, 0.57, 0.28, 0.20,
         "Stored profile\n[e1, e2, ..., eD]\nOne reading per channel")
    card(axis, 0.72, 0.57, 0.24, 0.20,
         "Later processing\nuses the profile\nwith sentence context", PURPLE)
    arrow(axis, (0.25, 0.67), (0.31, 0.67))
    arrow(axis, (0.63, 0.67), (0.70, 0.67))
    axis.text(0.5, 0.47, "GPT-2's tokenizer: 50,257 registry entries; ID size is not meaning",
              ha="center", color=NAVY, fontsize=13)
    card(axis, 0.03, 0.18, 0.28, 0.17, "the loud bark\nwoke the cat")
    card(axis, 0.38, 0.18, 0.24, 0.17, "Same bark ID\nSame starting profile")
    card(axis, 0.69, 0.18, 0.28, 0.17, "the rough bark\ncovered the tree")
    arrow(axis, (0.33, 0.265), (0.36, 0.265))
    arrow(axis, (0.67, 0.265), (0.64, 0.265))
    axis.text(0.5, 0.07, "Abstract learned channels, not literal temperature or mass readings.",
              ha="center", color=ORANGE, fontsize=13)
    save(figure, TOKEN_IMAGES / "02-03-id-selects-a-sensor-profile.png")


def numeric_trace():
    table = np.array([[2., 0.], [1., 3.], [0., 4.]])
    weights = np.array([[0., 1.], [1., 0.], [1., 1.]])
    profile = table[1]
    scores = weights @ profile
    assert profile.tolist() == [1., 3.] and scores.tolist() == [3., 1., 4.]
    figure, axis = canvas("One token. Two layers. Different numerical objects.")
    card(axis, 0.02, 0.45, 0.16, 0.29, "cat\nRegistry ID 1")
    card(axis, 0.27, 0.45, 0.24, 0.29,
         "Layer 1: lookup\nSensor profile [1, 3]\nTHIS is the embedding")
    card(axis, 0.60, 0.45, 0.17, 0.29, "Layer 2\nSeparate linear\npredictor", PURPLE)
    card(axis, 0.85, 0.45, 0.13, 0.29, "Scores\n"
         + "\n".join(f"{word}: {score:g}" for word, score in zip(
             ("the", "cat", "sat"), scores)), PURPLE)
    for start, end in ((0.20, 0.25), (0.53, 0.58), (0.79, 0.83)):
        arrow(axis, (start, 0.595), (end, 0.595))
    axis.text(0.5, 0.29, "Local registry: the=0 -> [2, 0]    cat=1 -> [1, 3]    sat=2 -> [0, 4]",
              ha="center", color=NAVY, fontsize=13)
    axis.text(0.5, 0.18, "Two sensor channels, not two named meanings. Scores are not embeddings.",
              ha="center", color=TEAL, fontsize=13)
    axis.text(0.5, 0.08, "Hand-picked teaching numbers, not a trained result or a physical measurement.",
              ha="center", color=ORANGE, fontsize=13)
    save(figure, TOKEN_IMAGES / "02-07-word-id-to-embedding-to-scores.png")


def learning():
    figure, axis = canvas("Practice adjusts the profile, not the registry number.")
    card(axis, 0.03, 0.59, 0.25, 0.19, "Fixed cat ID\nCat's four-channel profile")
    card(axis, 0.39, 0.59, 0.23, 0.19, "Separate predictor\nWord scores", PURPLE)
    card(axis, 0.73, 0.59, 0.23, 0.19, "Check against sat\nLoss", ORANGE)
    arrow(axis, (0.30, 0.685), (0.37, 0.685))
    arrow(axis, (0.64, 0.685), (0.71, 0.685))
    card(axis, 0.12, 0.23, 0.29, 0.18, "Update cat's profile\nKeep its ID", TEAL)
    card(axis, 0.59, 0.23, 0.29, 0.18, "Update predictor weights", PURPLE)
    arrow(axis, (0.84, 0.56), (0.265, 0.43), ORANGE)
    arrow(axis, (0.84, 0.56), (0.735, 0.43), ORANGE)
    axis.text(0.5, 0.10, "Isolated plain-SGD probe: unused dog and bird input profiles stay unchanged.",
              ha="center", color=NAVY, fontsize=13)
    save(figure, TOKEN_IMAGES / "02-05-practice-nudges-rows.png")


def observatory():
    figure, axis = canvas("Encoder-decoder: read a system, then write a likely report.")
    card(axis, 0.02, 0.47, 0.28, 0.34, "", TEAL)
    axis.add_patch(Ellipse((0.16, 0.68), 0.20, 0.10, fill=False, edgecolor=TEAL))
    for x, y in ((0.07, 0.69), (0.16, 0.72), (0.25, 0.66)):
        axis.add_patch(Circle((x, y), 0.012, color=TEAL))
    axis.text(0.16, 0.54, "Ordered observing log\nProfiles read one at a time",
              ha="center", color=NAVY, fontsize=12)
    card(axis, 0.39, 0.47, 0.22, 0.34,
         "Encoder briefing h\nA compact learned state\nNot a raw-profile average", PURPLE)
    card(axis, 0.70, 0.47, 0.28, 0.34,
         "Decoder's likely report\nAbout a newly observed object\nOne word, then the next", ORANGE)
    arrow(axis, (0.32, 0.64), (0.37, 0.64))
    arrow(axis, (0.63, 0.64), (0.68, 0.64))
    axis.text(0.16, 0.34, "Actual source:\nthe -> sleepy -> cat",
              ha="center", color=TEAL, fontsize=13)
    axis.text(0.50, 0.34, "Actual handoff:\nfinal encoder state",
              ha="center", color=PURPLE, fontsize=13)
    axis.text(0.84, 0.34, "Training answer:\nsat -> on -> <eos>",
              ha="center", color=ORANGE, fontsize=13)
    axis.text(0.5, 0.16, "The galaxy is the analogy. This model predicts vocabulary tokens, not physical properties.",
              ha="center", color=NAVY, fontsize=12)
    axis.text(0.5, 0.06, "Only the final briefing reaches the writer here. Attention will later let it revisit the log.",
              ha="center", color=NAVY, fontsize=12)
    save(figure, HERE / "images" / "03a-encoder-decoder-observatory.png")


if __name__ == "__main__":
    registry()
    numeric_trace()
    learning()
    observatory()
