"""
Gen script: ch03-importance-comparison.png
Side-by-side grouped bar chart: Univariate R² vs Permutation Importance for top-6 features.
Output: ../img/ch03-importance-comparison.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent
OUT  = HERE.parent / "img" / "ch03-importance-comparison.png"

# ── Data (hardcoded from Ch.3 California Housing model results) ──────────────
features = ["MedInc", "Latitude", "Longitude", "AveOccup", "HouseAge", "AveRooms"]
uni_r2   = [0.473,    0.021,      0.002,       0.001,      0.001,      0.023]
perm_imp = [0.334,    0.165,      0.133,       0.058,      0.029,      0.016]

BG        = "#1a1a2e"
LABEL_CLR = "#e2e8f0"
ACCENT    = ["#2563eb", "#16a34a"]

x = np.arange(len(features))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor(BG)

bars1 = ax.bar(x - w / 2, uni_r2,   width=w, color=ACCENT[0], label="Univariate R²")
bars2 = ax.bar(x + w / 2, perm_imp, width=w, color=ACCENT[1], label="Permutation Importance")

# Annotate bar values
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=LABEL_CLR)

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=LABEL_CLR)

# Annotation pointing to Latitude bar (low univariate but high permutation)
ax.annotate(
    "← Low univariate but high permutation\n   = jointly irreplaceable feature",
    xy=(x[1] + w / 2, perm_imp[1]),
    xytext=(2.4, 0.22),
    fontsize=7.5, color="#d97706",
    arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.2),
)

ax.set_xticks(x)
ax.set_xticklabels(features, rotation=30, ha="right", color=LABEL_CLR, fontsize=10)
ax.set_ylabel("Importance score", color=LABEL_CLR, fontsize=10)
ax.set_title(
    "Importance Rankings: Univariate R² vs Permutation\n"
    "(Ch.3 → Ch.2 model, California Housing)",
    color=LABEL_CLR, fontsize=11, pad=12,
)
ax.tick_params(colors=LABEL_CLR)
for spine in ax.spines.values():
    spine.set_edgecolor("#2d2d4e")

ax.legend(
    handles=[
        mpatches.Patch(color=ACCENT[0], label="Univariate R²"),
        mpatches.Patch(color=ACCENT[1], label="Permutation Importance"),
    ],
    loc="upper right",
    facecolor="#2d2d4e",
    labelcolor=LABEL_CLR,
    fontsize=9,
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
