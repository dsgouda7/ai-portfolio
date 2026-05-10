"""
Gen script: ch03-correlation-heatmap.png
8×8 annotated correlation heatmap for California Housing features.
Uses pure matplotlib (no seaborn dependency).
Highlights AveRooms/AveBedrms and Lat/Lon collinear pairs with bounding boxes.
Output: ../img/ch03-correlation-heatmap.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import fetch_california_housing

HERE = Path(__file__).parent
OUT  = HERE.parent / "img" / "ch03-correlation-heatmap.png"

# Use full dataset for a stable correlation matrix
# (train-only gives nearly identical results)
data = fetch_california_housing()
df   = pd.DataFrame(data.data, columns=data.feature_names)
corr = df.corr()
feat_names = list(corr.columns)
n = len(feat_names)

BG        = "#1a1a2e"
LABEL_CLR = "#e2e8f0"
cmap      = plt.get_cmap("coolwarm")

fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
ax.set_facecolor(BG)

corr_arr = corr.values

# Draw cells
for i in range(n):
    for j in range(n):
        val   = corr_arr[i, j]
        norm  = (val + 1) / 2.0            # map [-1,1] → [0,1]
        color = cmap(norm)
        rect  = plt.Rectangle([j, n - i - 1], 1, 1, fc=color, ec="#2d2d4e", lw=0.5)
        ax.add_patch(rect)
        # Text color: dark on light cells, light on dark cells
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        txt_color = "#111111" if lum > 0.5 else LABEL_CLR
        ax.text(j + 0.5, n - i - 0.5, f"{val:.2f}",
                ha="center", va="center", fontsize=7.5,
                color=txt_color, fontweight="bold" if abs(val) > 0.5 else "normal")

# Bounding boxes for collinear pairs
# AveRooms (col/row 2) ↔ AveBedrms (col/row 3) — amber
ax.add_patch(plt.Rectangle((2, n - 4), 2, 2,
             fill=False, edgecolor="#d97706", lw=2.5, zorder=5))
# Latitude (col/row 6) ↔ Longitude (col/row 7) — red
ax.add_patch(plt.Rectangle((6, n - 8), 2, 2,
             fill=False, edgecolor="#dc2626", lw=2.5, zorder=5))

# Axis labels
ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_xticks(np.arange(n) + 0.5)
ax.set_yticks(np.arange(n) + 0.5)
ax.set_xticklabels(feat_names, rotation=30, ha="right", color=LABEL_CLR, fontsize=9)
ax.set_yticklabels(reversed(feat_names), color=LABEL_CLR, fontsize=9)
ax.tick_params(length=0)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color=LABEL_CLR)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LABEL_CLR)

ax.set_title(
    "California Housing — Feature × Feature Correlation\n"
    "⚡ Amber = AveRooms/AveBedrms (VIF risk)  ·  Red = Lat/Lon",
    color=LABEL_CLR, fontsize=11, pad=14,
)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
