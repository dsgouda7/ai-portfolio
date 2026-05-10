from pathlib import Path
"""Generate ch8 rnn.png — RNN unrolled + LSTM cell diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(16, 10), facecolor="white")
fig.suptitle("RNNs, LSTMs & GRUs", fontsize=20, fontweight="bold", y=0.97)

# ── Layout: 2 rows ──────────────────────────────────────────────────────────
# Row 1: RNN unrolled (top)
# Row 2: left = LSTM cell internals, right = GRU vs LSTM comparison table
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                      left=0.05, right=0.97, top=0.90, bottom=0.07)

ax_rnn  = fig.add_subplot(gs[0, :])   # full-width top
ax_lstm = fig.add_subplot(gs[1, 0])   # bottom-left
ax_gru  = fig.add_subplot(gs[1, 1])   # bottom-right

for ax in [ax_rnn, ax_lstm, ax_gru]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ════════════════════════════════════════════════════════════════════
# PANEL 1 — RNN Unrolled
# ════════════════════════════════════════════════════════════════════
ax_rnn.set_title("Unrolled RNN  ( hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + b) )",
                 fontsize=13, fontweight="bold", pad=8)

steps = ["t-2", "t-1", "t", "t+1"]
xs = [0.12, 0.35, 0.58, 0.81]
cell_w, cell_h = 0.13, 0.22
cy = 0.52  # cell centre y

CELL_COLOR  = "#AED6F1"  # blue
ARROW_PROPS = dict(arrowstyle="-|>", color="#2C3E50", lw=1.8,
                   mutation_scale=14)

for i, (t, cx) in enumerate(zip(steps, xs)):
    # cell box
    rect = FancyBboxPatch((cx - cell_w/2, cy - cell_h/2), cell_w, cell_h,
                          boxstyle="round,pad=0.02",
                          facecolor=CELL_COLOR, edgecolor="#1A5276", lw=1.5,
                          transform=ax_rnn.transAxes, zorder=3)
    ax_rnn.add_patch(rect)
    ax_rnn.text(cx, cy, f"hₜ\n({t})", ha="center", va="center",
                fontsize=10, fontweight="bold", transform=ax_rnn.transAxes, zorder=4)

    # input arrow (xₜ ↑ cell)
    ax_rnn.annotate("", xy=(cx, cy - cell_h/2),
                    xytext=(cx, cy - cell_h/2 - 0.18),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=ARROW_PROPS, zorder=3)
    ax_rnn.text(cx, cy - cell_h/2 - 0.21, f"x({t})",
                ha="center", va="top", fontsize=9,
                transform=ax_rnn.transAxes)

    # output arrow (cell → yₜ)
    ax_rnn.annotate("", xy=(cx, cy + cell_h/2 + 0.17),
                    xytext=(cx, cy + cell_h/2),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=ARROW_PROPS, zorder=3)
    label = "ŷ" if t == "t" else ""
    ax_rnn.text(cx, cy + cell_h/2 + 0.19, f"h({t})" + (f"\n→ ŷ" if t == "t" else ""),
                ha="center", va="bottom", fontsize=8,
                transform=ax_rnn.transAxes)

    # hidden-state arrow to next cell
    if i < len(xs) - 1:
        nx = xs[i + 1]
        ax_rnn.annotate("", xy=(nx - cell_w/2 - 0.005, cy),
                        xytext=(cx + cell_w/2 + 0.005, cy),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=ARROW_PROPS, zorder=3)

# "Shared W" brace label
ax_rnn.annotate("", xy=(0.89, 0.78), xytext=(0.11, 0.78),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=1.5,
                                mutation_scale=12))
ax_rnn.text(0.5, 0.80, "Shared weights W  (same parameters at every time step)",
            ha="center", va="bottom", fontsize=9, color="#E74C3C",
            transform=ax_rnn.transAxes)

# Vanishing gradient note
ax_rnn.text(0.5, 0.08,
            "⚠  Vanishing gradient: gradients shrink exponentially over long sequences → "
            "LSTMs & GRUs add gating to preserve long-range dependencies.",
            ha="center", va="center", fontsize=8.5, color="#7D6608",
            style="italic", transform=ax_rnn.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7",
                      edgecolor="#F39C12", lw=1))

# ════════════════════════════════════════════════════════════════════
# PANEL 2 — LSTM Cell internals
# ════════════════════════════════════════════════════════════════════
ax_lstm.set_title("LSTM Cell — Four Gates", fontsize=12, fontweight="bold", pad=6)

# Cell state horizontal line
ax_lstm.annotate("", xy=(0.95, 0.82), xytext=(0.05, 0.82),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#117A65", lw=2.5,
                                 mutation_scale=14))
ax_lstm.text(0.5, 0.88, "Cell state  cₜ  (long-term memory highway)",
             ha="center", va="center", fontsize=9, color="#117A65",
             fontweight="bold", transform=ax_lstm.transAxes)

gates = [
    ("f",  0.18, "#E74C3C",   "Forget gate\nσ(Wf·[hₜ₋₁,xₜ]+bf)\nErase old memory"),
    ("i",  0.38, "#2E86C1",   "Input gate\nσ(Wi·[hₜ₋₁,xₜ]+bi)\nHow much to write"),
    ("g",  0.58, "#1E8449",   "Cell gate\ntanh(Wg·[hₜ₋₁,xₜ]+bg)\nNew candidate"),
    ("o",  0.78, "#7D3C98",   "Output gate\nσ(Wo·[hₜ₋₁,xₜ]+bo)\nWhat to expose"),
]

for sym, gx, color, desc in gates:
    circle = plt.Circle((gx, 0.58), 0.08, color=color, zorder=3,
                         transform=ax_lstm.transAxes)
    ax_lstm.add_patch(circle)
    ax_lstm.text(gx, 0.58, sym, ha="center", va="center",
                 fontsize=14, color="white", fontweight="bold",
                 transform=ax_lstm.transAxes, zorder=4)
    ax_lstm.text(gx, 0.38, desc, ha="center", va="top",
                 fontsize=7, color="#2C3E50",
                 transform=ax_lstm.transAxes,
                 bbox=dict(boxstyle="round,pad=0.2",
                           facecolor=color+"22", edgecolor=color, lw=0.8))
    ax_lstm.annotate("", xy=(gx, 0.82 - 0.02), xytext=(gx, 0.66 + 0.02),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                     mutation_scale=10))

# hₜ output
ax_lstm.annotate("", xy=(0.5, 0.13), xytext=(0.5, 0.50),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#7D3C98", lw=1.5,
                                 mutation_scale=10))
ax_lstm.text(0.5, 0.10, "hₜ  (hidden state / output)",
             ha="center", va="top", fontsize=8.5, color="#7D3C98",
             transform=ax_lstm.transAxes)

# ════════════════════════════════════════════════════════════════════
# PANEL 3 — GRU vs LSTM comparison
# ════════════════════════════════════════════════════════════════════
ax_gru.set_title("GRU vs LSTM — Quick Comparison", fontsize=12, fontweight="bold", pad=6)

rows = [
    ("", "LSTM", "GRU"),
    ("Gates", "4  (f, i, g, o)", "2  (reset r, update z)"),
    ("Cell state", "Separate cₜ & hₜ", "Single hₜ"),
    ("Parameters", "More (larger)", "Fewer (faster)"),
    ("Long-range", "Excellent", "Good"),
    ("Train speed", "Slower", "Faster"),
    ("When to use", "Long sequences,\ncomplex tasks", "Shorter seq.,\nresource-limited"),
]

col_x  = [0.05, 0.37, 0.70]
row_ys = np.linspace(0.90, 0.10, len(rows))

header_colors = ["#FDFEFE", "#2E86C1", "#1E8449"]
for ci, (hc, lbl) in enumerate(zip(header_colors, ["", "LSTM", "GRU"])):
    if ci > 0:
        rect = FancyBboxPatch((col_x[ci] - 0.02, row_ys[0] - 0.06),
                              0.32, 0.12,
                              boxstyle="round,pad=0.01",
                              facecolor=hc, edgecolor="none",
                              transform=ax_gru.transAxes)
        ax_gru.add_patch(rect)
    ax_gru.text(col_x[ci], row_ys[0], lbl, ha="left", va="center",
                fontsize=10, fontweight="bold",
                color="white" if ci > 0 else "#2C3E50",
                transform=ax_gru.transAxes)

for ri, (label, lstm_val, gru_val) in enumerate(rows[1:], start=1):
    bg = "#F2F3F4" if ri % 2 == 0 else "white"
    rect = FancyBboxPatch((0.0, row_ys[ri] - 0.055), 1.0, 0.11,
                          boxstyle="square,pad=0",
                          facecolor=bg, edgecolor="none",
                          transform=ax_gru.transAxes, zorder=1)
    ax_gru.add_patch(rect)
    for ci, txt in enumerate([label, lstm_val, gru_val]):
        ax_gru.text(col_x[ci], row_ys[ri], txt,
                    ha="left", va="center", fontsize=8,
                    color="#2C3E50", transform=ax_gru.transAxes, zorder=2)

# horizontal dividers
for ri in range(1, len(rows)):
    ax_gru.axhline(row_ys[ri] + 0.055, color="#BFC9CA", lw=0.5,
                   xmin=0, xmax=1)

# ━━━ shared caption ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig.text(0.5, 0.025,
         "RNNs process sequences via recurrent hidden states; "
         "LSTMs add gated cell state to capture long-range dependencies; "
         "GRUs simplify with two gates and a single state vector.",
         ha="center", va="center", fontsize=9, color="#555",
         style="italic",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9F9",
                   edgecolor="#BDC3C7", lw=0.8))

out = str(Path(__file__).resolve().parent.parent / "img" / "ch8 rnn.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
