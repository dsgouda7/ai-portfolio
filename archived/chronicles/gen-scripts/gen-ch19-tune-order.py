"""Generate Reference/img/ch19-tune-order.png — tuning priority cascade.

A single-panel "dial order" chart: ten hyperparameter dials stacked by
tuning priority (top = highest leverage / cheapest to adjust first),
each with a sensitivity bar, a first-pass default, and a note on what
the dial actually trades off (accuracy / time / memory / generalisation).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

DARK   = "#2C3E50"
GOLD   = "#F39C12"
BLUE   = "#2E86C1"
GREEN  = "#27AE60"
PURPLE = "#8E44AD"
ORANGE = "#E67E22"
RED    = "#E74C3C"
GREY   = "#7F8C8D"
NAVY   = "#1F3A5F"

# (rank, dial, sensitivity 1-5, default, trade-off note, accent)
DIALS = [
    (1,  "Learning rate  (η)",      5, "1e-3 Adam / 1e-2 SGD", "accuracy · time",                RED),
    (2,  "Batch size",               3, "64–256",               "step time · memory · noise",     ORANGE),
    (3,  "Optimiser",                3, "AdamW",                "convergence speed · memory ×3",  GOLD),
    (4,  "Weight init",              2, "He (ReLU) / Xavier",   "startup stability (set + forget)", BLUE),
    (5,  "Depth · width",            4, "2–4 layers, 64–256 u", "capacity · memory · time",       PURPLE),
    (6,  "Dropout  (p)",             3, "0.3 on FC layers",     "generalisation gap",             GREEN),
    (7,  "Weight decay  (λ)",        3, "1e-4 (AdamW)",         "generalisation gap",             GREEN),
    (8,  "LR schedule / warmup",     2, "cosine + 5–10% warmup","free +1–3% test acc",            ORANGE),
    (9,  "Loss function",            0, "match target type",    "this is a choice, not a tune",   GREY),
    (10, "More data",                5, "halve-then-double test","variance ⇓ (strongest lever)",   NAVY),
]

COST_LABELS = ["none", "tiny", "low", "medium", "high", "very high"]

fig, ax = plt.subplots(figsize=(13.6, 8.4), facecolor="white")
ax.set_xlim(0, 100); ax.set_ylim(0, len(DIALS) + 1.2)
ax.invert_yaxis()
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)

# Header band
ax.add_patch(mpatches.Rectangle((0, 0.05), 100, 0.75,
    facecolor=DARK, alpha=0.92, edgecolor="none"))
ax.text(2,  0.42, "RANK", color="white", fontsize=9.5, va="center",
        fontweight="bold")
ax.text(8,  0.42, "DIAL", color="white", fontsize=9.5, va="center",
        fontweight="bold")
ax.text(35, 0.42, "SENSITIVITY", color="white", fontsize=9.5, va="center",
        fontweight="bold")
ax.text(60, 0.42, "FIRST-PASS DEFAULT", color="white", fontsize=9.5,
        va="center", fontweight="bold")
ax.text(85, 0.42, "TRADES OFF", color="white", fontsize=9.5,
        va="center", fontweight="bold")

# Rows
for k, (rank, name, sens, default, tradeoff, accent) in enumerate(DIALS):
    y = k + 1.2
    row_bg = "#F8F9FB" if k % 2 == 0 else "white"
    ax.add_patch(mpatches.Rectangle((0, y - 0.42), 100, 0.82,
        facecolor=row_bg, edgecolor="none"))

    # Left accent bar
    ax.add_patch(mpatches.Rectangle((0, y - 0.42), 1.2, 0.82,
        facecolor=accent, edgecolor="none"))

    # Rank circle
    ax.add_patch(mpatches.Circle((3.2, y), 0.30,
        facecolor=accent, edgecolor=accent, linewidth=0))
    ax.text(3.2, y, str(rank), ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white")

    # Dial name
    ax.text(6.5, y, name, va="center", fontsize=10.5,
            fontweight="bold", color=DARK)

    # Sensitivity bar
    bar_x, bar_w, bar_h = 35, 15, 0.30
    ax.add_patch(mpatches.Rectangle((bar_x, y - bar_h/2), bar_w, bar_h,
        facecolor="#E6E9EE", edgecolor="none"))
    fill_w = (sens / 5.0) * bar_w
    ax.add_patch(mpatches.Rectangle((bar_x, y - bar_h/2), fill_w, bar_h,
        facecolor=accent, edgecolor="none"))
    ax.text(bar_x + bar_w + 1.0, y,
            f"{sens}/5" if sens > 0 else "n/a  (choose)",
            va="center", fontsize=9, color=DARK)

    # Default
    ax.text(60, y, default, va="center", fontsize=9.5,
            color=DARK, family="monospace")

    # Trade-off
    ax.text(85, y, tradeoff, va="center", fontsize=9, color=GREY,
            style="italic")

# Left-edge cascade arrow showing order
arrow_x = -1.0
ax.annotate("", xy=(arrow_x, len(DIALS) + 0.9),
            xytext=(arrow_x, 1.1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6",
                            color=GOLD, lw=2.2),
            annotation_clip=False)
ax.text(arrow_x - 1.4, (len(DIALS) + 1) / 2,
        "T U N E   I N   T H I S   O R D E R",
        rotation=90, ha="center", va="center",
        fontsize=11, color=GOLD, fontweight="bold",
        clip_on=False)

# Title + footer
ax.text(50, -0.45,
        "Hyperparameter Tuning — cheapest, highest-leverage dial first",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=DARK)

foot = ("Rule: change ONE dial, re-run on the validation set, attribute "
        "the delta, repeat.  Two dials at once = no signal.")
ax.text(50, len(DIALS) + 0.8, foot,
        ha="center", va="center", fontsize=9.5, color=DARK,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8E1",
                  edgecolor=GOLD, linewidth=1.1))

out = Path(__file__).resolve().parent / "ch19-tune-order.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
