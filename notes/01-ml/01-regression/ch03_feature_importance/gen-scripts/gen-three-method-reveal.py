"""
gen_three_method_reveal.py

Animated reveal showing all 3 importance methods for all 8 California Housing
features, with an explicit caption explaining *how ŷ is determined* for each.

Stages:
  1. Method 1 only  — ŷ = 8 separate single-feature models (isolation)
  2. + Method 2     — ŷ = one full joint model (all features, partial effect)
  3. + Method 3     — ŷ = same full model, one column shuffled (never retrained)
  4. Annotations    — convergence patterns highlighted for key features

Output: ../img/three-method-reveal.gif

Run:  python gen_three_method_reveal.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).parent
OUT = HERE.parent / "img" / "three-method-reveal.gif"

# ── Palette ───────────────────────────────────────────────────────────────
BG    = "#1a1a2e"
DARK  = "#0d0d1f"
TEXT  = "#e2e8f0"
SUBTEXT = "#94a3b8"
GREY  = "#374151"
BLUE  = "#2563eb"   # M1 Univariate R²
AMBER = "#d97706"   # M2 Standardised Weights
GREEN = "#16a34a"   # M3 Permutation Importance
RED   = "#dc2626"

# ── Data (California Housing, Ch.2 model) ─────────────────────────────────
FEATURES = [
    "MedInc", "Latitude", "Longitude", "AveOccup",
    "HouseAge", "AveRooms", "AveBedrms", "Population",
]
N = len(FEATURES)

UNI_R2 = np.array([0.473, 0.021, 0.002, 0.001, 0.001, 0.023, 0.002, 0.001])
STD_W  = np.array([0.830, 0.890, 0.870, 0.030, 0.060, 0.120, 0.100, 0.010])
PERM   = np.array([0.334, 0.165, 0.133, 0.058, 0.029, 0.016, 0.005, 0.002])

# Normalize each method to its own maximum for visual comparison
m1 = UNI_R2 / UNI_R2.max()   # max = 0.473 (MedInc)
m2 = STD_W  / STD_W.max()    # max = 0.890 (Latitude)
m3 = PERM   / PERM.max()     # max = 0.334 (MedInc)

# ── Animation parameters ──────────────────────────────────────────────────
FPS   = 10
HOLD  = 22   # hold frames per stage (~2.2 s)
TRANS = 12   # transition frames between stages (~1.2 s)

# Frame boundaries
T0 = HOLD                      # end of stage-0 hold → start M2 growth
T1 = T0 + TRANS                # M2 fully grown → start stage-1 hold
T2 = T1 + HOLD                 # end of stage-1 hold → start M3 growth
T3 = T2 + TRANS                # M3 fully grown → start stage-2 hold
T4 = T3 + HOLD                 # end of stage-2 hold → annotations snap in
T5 = T4 + HOLD + HOLD          # final hold (extra long)
TOTAL_FRAMES = T5


def _ease(t: float) -> float:
    """Smooth cubic ease-in-out."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def frame_state(f: int) -> tuple[int, float, float, float, bool]:
    """
    Return (stage, s1, s2, s3, show_annot) for frame f.
    s1/s2/s3 = current scale for M1/M2/M3 bar heights (0 → 1 during growth).
    show_annot = whether convergence annotation boxes are visible.
    """
    if f < T0:
        return 0, 1.0, 0.0, 0.0, False
    elif f < T1:
        p = _ease((f - T0) / TRANS)
        return 0, 1.0, p, 0.0, False
    elif f < T2:
        return 1, 1.0, 1.0, 0.0, False
    elif f < T3:
        p = _ease((f - T2) / TRANS)
        return 1, 1.0, 1.0, p, False
    elif f < T4:
        return 2, 1.0, 1.0, 1.0, False
    else:
        return 3, 1.0, 1.0, 1.0, True


# ── Caption text per stage ────────────────────────────────────────────────
CAPTIONS = [
    (
        "Method 1 — Univariate R²",
        "ŷ source: 8 separate single-feature models, each fitted in complete isolation.\n"
        "  MedInc R²=0.473  |  Latitude R²=0.021  |  Longitude R²=0.002  |  Population R²≈0.001\n"
        "  MedInc towers. Latitude and Longitude look nearly flat — misleadingly so.",
    ),
    (
        "Method 2 — Standardised Weights  (partial contribution)",
        "ŷ source: one full joint model with all 8 features simultaneously.\n"
        "  Each |w_std| = marginal shift in ŷ for a 1-σ move in that feature, all others held fixed.\n"
        "  Latitude (0.89) and Longitude (0.87) leap to #1 and #2.  MedInc (0.83) drops to #3.",
    ),
    (
        "Method 3 — Permutation Importance",
        "ŷ source: same full model as M2, never retrained.  One feature's column is randomly shuffled;\n"
        "  rise in MAE reveals how badly the existing weights are handicapped without that feature's ordering.\n"
        "  MedInc ($18.4k Δ) reclaims #1.  Latitude + Longitude hold #2/#3.  Population ≈ 0 on all three.",
    ),
    (
        "Convergence = confidence.  Divergence = diagnostic story.",
        "  MedInc:  all 3 agree → independent, irreplaceable signal\n"
        "  Latitude / Longitude:  M1 ≈ 0, M2+M3 high → jointly irreplaceable (geography only emerges in combination)\n"
        "  Population:  all 3 ≈ 0 → genuinely uninformative, safe to drop",
    ),
]

# ── Figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8.0), facecolor=BG)

# Bar chart area
ax = fig.add_axes([0.07, 0.28, 0.90, 0.64])
ax.set_facecolor(DARK)

x = np.arange(N)
W = 0.24
offsets = np.array([-W, 0.0, W])

# Initialize zero-height bars
bars_m1 = ax.bar(x + offsets[0], np.zeros(N), width=W, color=BLUE,  alpha=0.88, zorder=3, label="M1")
bars_m2 = ax.bar(x + offsets[1], np.zeros(N), width=W, color=AMBER, alpha=0.88, zorder=3, label="M2")
bars_m3 = ax.bar(x + offsets[2], np.zeros(N), width=W, color=GREEN, alpha=0.88, zorder=3, label="M3")

ax.set_xlim(-0.55, N - 0.45)
ax.set_ylim(0, 1.30)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, color=TEXT, fontsize=11, rotation=30, ha="right")
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], color=SUBTEXT, fontsize=9)
ax.set_ylabel("Normalized importance\n(1.0 = that method's maximum)", color=TEXT, fontsize=10)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values():
    spine.set_edgecolor(GREY)

# Subtle vertical separators between feature groups
for xi in np.arange(0.5, N - 0.5):
    ax.axvline(xi, color=GREY, linewidth=0.5, alpha=0.35, zorder=1)
ax.axhline(0, color=GREY, linewidth=0.8, zorder=2)

ax.set_title(
    "Three Methods — Same 8 Features, Three Different ŷ Sources",
    color=TEXT, fontsize=13, fontweight="bold", pad=10,
)

# Legend
legend_handles = [
    mpatches.Patch(color=BLUE,  label="M1 · Univariate R²            ŷ = single-feature model (isolation)"),
    mpatches.Patch(color=AMBER, label="M2 · Std Weights               ŷ = full joint model (all features in)"),
    mpatches.Patch(color=GREEN, label="M3 · Permutation Importance   ŷ = full model, column shuffled (never retrained)"),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    facecolor="#1e1e3a",
    labelcolor=TEXT,
    fontsize=8.0,
    framealpha=0.92,
    edgecolor=GREY,
    handlelength=1.4,
)

# Convergence annotation boxes (hidden until stage 3)
_annot_kw = dict(ha="center", va="bottom", fontsize=8, zorder=6, visible=False)

annot_medinc = ax.text(
    x[0], 1.07,
    "All 3 agree\n→ independent,\n   irreplaceable",
    color=GREEN, **_annot_kw,
    bbox=dict(boxstyle="round,pad=0.3", fc="#052e16", ec=GREEN, lw=1.5, alpha=0.92),
)

# Place between Latitude (1) and Longitude (2)
annot_latlon = ax.text(
    (x[1] + x[2]) / 2.0, 1.07,
    "M1≈0 but M2+M3 high\n→ jointly irreplaceable\n   (geography in combo)",
    color=BLUE, **_annot_kw,
    bbox=dict(boxstyle="round,pad=0.3", fc="#0c1a3a", ec=BLUE, lw=1.5, alpha=0.92),
)

annot_pop = ax.text(
    x[7], 1.07,
    "All 3 ≈ 0\n→ genuinely\n   uninformative",
    color=RED, **_annot_kw,
    bbox=dict(boxstyle="round,pad=0.3", fc="#2d0b0b", ec=RED, lw=1.5, alpha=0.92),
)

# ── Caption area ──────────────────────────────────────────────────────────
cap_ax = fig.add_axes([0.05, 0.01, 0.90, 0.24])
cap_ax.set_facecolor("#0d0d20")
cap_ax.axis("off")
for spine in cap_ax.spines.values():
    spine.set_edgecolor(GREY)

# Stage label (bold, colored)
stage_label = cap_ax.text(
    0.5, 0.82, "",
    ha="center", va="top",
    fontsize=10.5, fontweight="bold", color=AMBER,
    transform=cap_ax.transAxes,
)

# Detail text
stage_detail = cap_ax.text(
    0.5, 0.58, "",
    ha="center", va="top",
    fontsize=8.8, color=TEXT,
    transform=cap_ax.transAxes,
    linespacing=1.6,
)


# ── Update function ───────────────────────────────────────────────────────
def update(frame: int):
    stage, s1, s2, s3, show_annot = frame_state(frame)

    # Update bar heights
    for i, bar in enumerate(bars_m1):
        bar.set_height(m1[i] * s1)
    for i, bar in enumerate(bars_m2):
        bar.set_height(m2[i] * s2)
    for i, bar in enumerate(bars_m3):
        bar.set_height(m3[i] * s3)

    # Update captions
    label, detail = CAPTIONS[stage]
    stage_label.set_text(label)
    stage_detail.set_text(detail)

    # Show/hide annotations
    for annot in (annot_medinc, annot_latlon, annot_pop):
        annot.set_visible(show_annot)

    return (
        list(bars_m1) + list(bars_m2) + list(bars_m3)
        + [stage_label, stage_detail, annot_medinc, annot_latlon, annot_pop]
    )


# ── Render ────────────────────────────────────────────────────────────────
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000 / FPS, blit=False)
ani.save(OUT, writer=PillowWriter(fps=FPS))
plt.close()
print(f"Saved → {OUT}")
