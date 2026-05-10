"""
Gen script: ch03-permutation-loop.png
Conceptual 4-panel (2×2) static diagram of the permutation importance shuffle loop.
Panel 0: Baseline test set (MAE = $55k).
Panel 1: MedInc column shuffled (model weights unchanged).
Panel 2: MAE rises to $73k — permutation importance = 0.334.
Panel 3: Final ranking bar chart after 30 repeats across all features.
Output: ../img/ch03-permutation-loop.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
OUT  = HERE.parent / "img" / "ch03-permutation-loop.png"

BG        = "#1a1a2e"
PANEL_BG  = "#12122a"
LABEL_CLR = "#e2e8f0"
AMBER     = "#d97706"
GREEN     = "#16a34a"
BLUE      = "#2563eb"
RED       = "#dc2626"

# Mock test data (5 rows × 4 features)
mock_features = ["MedInc", "Latitude", "HouseAge", "AveRooms"]
mock_base = np.array([
    [4.2, 37.8, 25, 5.1],
    [1.8, 34.1, 40, 4.3],
    [8.5, 38.2, 15, 6.2],
    [2.1, 36.5, 52, 3.8],
    [5.3, 33.9, 30, 5.7],
])

rng = np.random.default_rng(7)
mock_shuffled = mock_base.copy()
mock_shuffled[:, 0] = rng.permutation(mock_base[:, 0])


def draw_mini_table(ax, data, headers, title, highlight_col=None,
                    highlight_color=BLUE):
    """Draw a mini data table on a matplotlib Axes (no real axes)."""
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.set_title(title, color=LABEL_CLR, fontsize=9, pad=8, fontweight="bold")

    n_rows, n_cols = data.shape
    col_w = 1.0 / n_cols
    row_h = 0.13

    # Header
    for j, hdr in enumerate(headers):
        fc = highlight_color if j == highlight_col else "#2a3670"
        ax.add_patch(mpatches.FancyBboxPatch(
            (j * col_w, 1 - row_h), col_w - 0.01, row_h - 0.01,
            boxstyle="round,pad=0.01", fc=fc, ec="#2d2d4e", zorder=2,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(j * col_w + col_w / 2, 1 - row_h / 2, hdr,
                ha="center", va="center", color=LABEL_CLR,
                fontsize=7.5, fontweight="bold", transform=ax.transAxes)

    # Data rows
    for i in range(n_rows):
        for j in range(n_cols):
            fc = highlight_color if j == highlight_col else "#1e2040"
            ax.add_patch(mpatches.FancyBboxPatch(
                (j * col_w, 1 - row_h * (i + 2)),
                col_w - 0.01, row_h - 0.01,
                boxstyle="round,pad=0.01", fc=fc, ec="#2d2d4e", zorder=2,
                transform=ax.transAxes, clip_on=False,
            ))
            ax.text(j * col_w + col_w / 2, 1 - row_h * (i + 1.5),
                    f"{data[i, j]:.1f}",
                    ha="center", va="center", color=LABEL_CLR,
                    fontsize=7, transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
fig.suptitle(
    "Permutation Importance — The Shuffle Loop",
    color=LABEL_CLR, fontsize=13, fontweight="bold", y=0.98,
)

# ── Panel 0: Baseline ────────────────────────────────────────────────────────
ax0 = axes[0, 0]
draw_mini_table(ax0, mock_base, mock_features,
                "Step 0 — Baseline  (MAE = $55k)",
                highlight_col=0, highlight_color=BLUE)
ax0.text(0.5, 0.04, "MAE = $55k  ← baseline",
         ha="center", va="bottom", color=AMBER, fontsize=9.5,
         fontweight="bold", transform=ax0.transAxes)

# ── Panel 1: Shuffled ────────────────────────────────────────────────────────
ax1 = axes[0, 1]
draw_mini_table(ax1, mock_shuffled, mock_features,
                "Step 1 — Shuffle MedInc column",
                highlight_col=0, highlight_color=RED)
ax1.text(0.5, 0.04,
         "Model weights UNCHANGED\nOnly MedInc values scrambled",
         ha="center", va="bottom", color=LABEL_CLR, fontsize=7.5,
         transform=ax1.transAxes)

# ── Panel 2: MAE rise ────────────────────────────────────────────────────────
ax2 = axes[1, 0]
ax2.set_facecolor(PANEL_BG)
ax2.axis("off")
ax2.set_title("Step 2 — Recompute MAE",
              color=LABEL_CLR, fontsize=9, pad=8, fontweight="bold")
ax2.text(0.5, 0.75, "$55k  →  $73k",
         ha="center", va="center", color=AMBER, fontsize=22,
         fontweight="bold", transform=ax2.transAxes)
ax2.text(0.5, 0.57, "Δ MAE = +$18k",
         ha="center", va="center", color=RED, fontsize=14,
         transform=ax2.transAxes)
ax2.text(0.5, 0.40,
         "Permutation importance  =  Δ MAE / MAE_baseline",
         ha="center", va="center", color=LABEL_CLR, fontsize=8.5,
         transform=ax2.transAxes)
ax2.text(0.5, 0.26, "= $18k / $55k  ≈  0.334",
         ha="center", va="center", color=GREEN, fontsize=12,
         transform=ax2.transAxes)
ax2.text(0.5, 0.10, "(MedInc is the most irreplaceable feature)",
         ha="center", va="center", color=LABEL_CLR, fontsize=8,
         transform=ax2.transAxes)

# ── Panel 3: Final ranking bar chart ─────────────────────────────────────────
ax3 = axes[1, 1]
ax3.set_facecolor(PANEL_BG)
ax3.set_title("Step 3 — Repeat × 30 shuffles, average",
              color=LABEL_CLR, fontsize=9, pad=8, fontweight="bold")

feat_all = ["MedInc", "Latitude", "Longitude", "AveOccup",
            "HouseAge", "AveRooms", "AveBedrms", "Population"]
perm_all = [0.334, 0.165, 0.133, 0.058, 0.029, 0.016, 0.005, 0.002]
accent   = [BLUE, GREEN, GREEN, AMBER, AMBER,
            "#9333ea", "#9333ea", "#6b7280"]

y_pos = np.arange(len(feat_all))
bars  = ax3.barh(y_pos, perm_all, color=accent, height=0.6)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(feat_all, color=LABEL_CLR, fontsize=8)
ax3.set_xlabel("Permutation importance", color=LABEL_CLR, fontsize=8)
ax3.tick_params(colors=LABEL_CLR)
for spine in ax3.spines.values():
    spine.set_edgecolor("#2d2d4e")

for bar, val in zip(bars, perm_all):
    ax3.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=7, color=LABEL_CLR)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
