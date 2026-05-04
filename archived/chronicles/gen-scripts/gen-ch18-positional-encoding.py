"""Generate Reference/img/ch18-positional-encoding.png — sinusoidal PE.

Two-panel figure:
  (left)   Heatmap of the (pos, dim) PE matrix for T=64, d_model=128
           showing the sin/cos bands at different frequencies.
  (right)  Line plot of four selected dimensions across positions,
           illustrating that low dims oscillate slowly (long-range) and
           high dims oscillate fast (fine-grained).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DARK  = "#2C3E50"
ATTN2 = "#3730A3"
BLUE  = "#2E86C1"
GREEN = "#27AE60"
ORANGE= "#E67E22"
RED   = "#E74C3C"
GREY  = "#7F8C8D"

T = 64
d_model = 128

pos = np.arange(T)[:, None]                 # (T, 1)
dim = np.arange(d_model)[None, :]           # (1, d_model)
even_mask = (dim % 2 == 0)
# div_term per pair index i = dim // 2
i = dim // 2
denom = np.power(10000.0, (2 * i) / d_model)
angles = pos / denom                        # (T, d_model)
PE = np.where(even_mask, np.sin(angles), np.cos(angles))   # (T, d_model)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), facecolor="white",
                         gridspec_kw={"width_ratios": [1.2, 1.0]})
fig.suptitle("Sinusoidal Positional Encoding — one unique signature per position",
             fontsize=13, fontweight="bold", color=DARK, y=0.995)

# ── LEFT: heatmap of PE matrix ────────────────────────────────────────────
ax = axes[0]
im = ax.imshow(PE, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1,
               interpolation="nearest")
ax.set_xlabel(r"dimension index  $2i$ / $2i+1$", fontsize=10, color=DARK)
ax.set_ylabel(r"position  $pos$", fontsize=10, color=DARK)
ax.set_title(r"PE[$pos$, $j$] matrix   ($T$=64, $d_{\mathrm{model}}$=128)",
             fontsize=11, fontweight="bold", color=ATTN2, loc="left", pad=6)
ax.invert_yaxis()  # pos=0 at top
cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
cbar.set_label("value", fontsize=9, color=DARK)

# Annotate dim bands
ax.annotate("high-frequency\n(fine-grained)", xy=(3, 55), xytext=(18, 58),
            fontsize=8.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF8E1",
                      edgecolor="#E67E22", linewidth=0.9),
            arrowprops=dict(arrowstyle="->", color="#E67E22", lw=0.9))
ax.annotate("low-frequency\n(long-range)", xy=(115, 55), xytext=(76, 58),
            fontsize=8.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#E3F2FD",
                      edgecolor=BLUE, linewidth=0.9),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))

# ── RIGHT: four selected dimensions across positions ──────────────────────
ax = axes[1]
chosen = [(2, BLUE,   r"$2i$=0   (fastest sin)"),
          (10, GREEN,  r"$2i$=10"),
          (40, ORANGE, r"$2i$=40"),
          (100, RED,   r"$2i$=100 (slowest sin)")]
for j, col, lab in chosen:
    ax.plot(np.arange(T), PE[:, j], color=col, linewidth=1.8, label=lab)

ax.axhline(0, color=GREY, linewidth=0.7, alpha=0.5)
ax.set_xlabel("position  $pos$", fontsize=10, color=DARK)
ax.set_ylabel(r"PE[$pos$, $j$]", fontsize=10, color=DARK)
ax.set_title("Four dimensions — different frequencies",
             fontsize=11, fontweight="bold", color=ATTN2, loc="left", pad=6)
ax.set_xlim(0, T - 1); ax.set_ylim(-1.15, 1.15)
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

ax.text(T / 2, -1.45,
        r"$\mathrm{PE}_{(pos,\,2i)} = \sin(pos / 10000^{2i/d})$   •   "
        r"$\mathrm{PE}_{(pos,\,2i+1)} = \cos(pos / 10000^{2i/d})$",
        ha="center", va="top", fontsize=9.5, color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EDE9FE",
                  edgecolor=ATTN2, linewidth=1.0))

fig.tight_layout(rect=(0, 0.04, 1, 0.95))
out = Path(__file__).resolve().parent / "ch18-positional-encoding.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
