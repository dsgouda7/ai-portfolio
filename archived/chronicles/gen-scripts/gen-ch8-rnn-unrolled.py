"""Generate Reference/img/ch8-rnn-unrolled.png — plain RNN unrolled through time.

One input x_t per timestep, one hidden state h_t recurring, shared weights
W_x, W_h, W_y across timesteps.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BLUE = "#2E86C1"
GREEN = "#27AE60"
ORANGE = "#E67E22"
DARK = "#2C3E50"
GREY = "#7F8C8D"

fig, ax = plt.subplots(figsize=(13, 5.5), facecolor="white")
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("RNN unrolled through time — one cell reused across t = 1 … T",
             fontsize=13, fontweight="bold", color=DARK, pad=10)

timesteps = [1, 2, 3, "…", "T"]
xs = [1.5, 4.25, 7.0, 9.75, 12.5]

# initial hidden state
ax.add_patch(plt.Circle((0.2, 3), 0.28, facecolor="white",
                        edgecolor=DARK, lw=1.5))
ax.text(0.2, 3, r"$h_0$", ha="center", va="center", fontsize=11, color=DARK)

prev = (0.2 + 0.28, 3)
for i, (t, cx) in enumerate(zip(timesteps, xs)):
    # timestep label
    label = f"t = {t}" if t != "…" else "…"
    ax.text(cx, 5.6, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=DARK)

    if t == "…":
        # ellipsis cell
        ax.text(cx, 3, "…", ha="center", va="center",
                fontsize=28, color=GREY)
        # pass-through arrow for hidden state
        ax.annotate("", xy=(cx + 0.8, 3), xytext=(prev[0], prev[1]),
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5))
        prev = (cx + 0.8, 3)
        continue

    # cell box
    ax.add_patch(mpatches.FancyBboxPatch((cx - 0.95, 2.3), 1.9, 1.4,
                                         boxstyle="round,pad=0.04,rounding_size=0.15",
                                         facecolor="#EBF5FB",
                                         edgecolor=DARK, lw=1.3))
    ax.text(cx, 3, "RNN\ncell", ha="center", va="center",
            fontsize=10, fontweight="bold", color=DARK)

    # hidden state arrow from previous
    ax.annotate("", xy=(cx - 0.95, 3), xytext=prev,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))
    # label
    if i == 0:
        ax.text((prev[0] + cx - 0.95) / 2, 3.25, r"$h_0$",
                ha="center", va="bottom", fontsize=9, color=DARK)
    else:
        prev_idx = i - 1
        t_prev = timesteps[prev_idx]
        if t_prev != "…":
            ax.text((prev[0] + cx - 0.95) / 2, 3.25,
                    fr"$h_{{{t_prev}}}$",
                    ha="center", va="bottom", fontsize=9, color=DARK)

    # input arrow up into the cell
    ax.annotate("", xy=(cx, 2.3), xytext=(cx, 1.0),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
    ax.text(cx, 0.75, fr"$x_{{{t}}}$",
            ha="center", va="top", fontsize=11, color=BLUE, fontweight="bold")

    # output arrow up out of the cell
    ax.annotate("", xy=(cx, 5.1), xytext=(cx, 3.7),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax.text(cx, 5.25, fr"$\hat y_{{{t}}}$",
            ha="center", va="bottom", fontsize=11, color=GREEN, fontweight="bold")

    prev = (cx + 0.95, 3)

# final hidden arrow leaving the last cell
ax.annotate("", xy=(13.7, 3), xytext=prev,
            arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6))
ax.text(13.7, 3.25, r"$h_T$", ha="left", va="bottom", fontsize=10, color=DARK)

# weight-sharing note
ax.text(7, 0.15,
        r"Shared weights across time: $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$,   $\hat y_t = W_y h_t + c$",
        ha="center", va="center", fontsize=10, color="#555",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FDF2E9",
                  edgecolor=ORANGE, lw=0.9))

out = Path(__file__).resolve().parent / "ch8-rnn-unrolled.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
