"""Generate ch04 SVM visuals.

Produces:
  img/ch04-svm-needle.gif   — recall needle climbing from RF→Linear SVM→RBF SVM
  img/ch04-svm-progress-check.png — FaceAI constraint scorecard after Ch.4

Run from anywhere; paths are resolved relative to this file.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Wedge
import numpy as np
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
IMG_DIR  = THIS_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── palette (dark background, gold-standard colours) ──────────────────────────
BG       = "#1a1a2e"
PRIMARY  = "#1e3a8a"
SUCCESS  = "#15803d"
CAUTION  = "#b45309"
DANGER   = "#b91c1c"
INFO     = "#1d4ed8"
WHITE    = "#f1f5f9"
GOLD     = "#fbbf24"
GREY     = "#64748b"

rng = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1 · NEEDLE GIF — recall climbing RF → Linear → RBF
# ══════════════════════════════════════════════════════════════════════════════

def _needle_frame(ax, recall_pct: float, stage_label: str, stage_color: str):
    """Draw one speedometer-style needle frame on ax."""
    ax.set_facecolor(BG)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── arc background (180° semicircle) ─────────────────────────────────────
    theta = np.linspace(0, np.pi, 300)
    arc_x = np.cos(theta)
    arc_y = np.sin(theta)
    ax.fill_between(arc_x, arc_y, alpha=0.08, color=WHITE)

    # ── coloured zones: red 0–85%, amber 85–90%, green 90–100% ───────────────
    def arc_wedge(pct_start, pct_end, colour, alpha=0.35):
        ang_start = 180 - pct_start * 1.8   # map 0–100% to 180°–0°
        ang_end   = 180 - pct_end   * 1.8
        w = Wedge((0, 0), 1.0, ang_end, ang_start,
                  width=0.28, facecolor=colour, alpha=alpha, zorder=2)
        ax.add_patch(w)

    arc_wedge(0,  85, DANGER)
    arc_wedge(85, 90, CAUTION)
    arc_wedge(90, 100, SUCCESS)

    # ── tick marks ────────────────────────────────────────────────────────────
    for pct in range(0, 101, 10):
        ang = np.radians(180 - pct * 1.8)
        r0, r1 = 0.72, 0.85 if pct % 20 == 0 else 0.78
        ax.plot([r0 * np.cos(ang), r1 * np.cos(ang)],
                [r0 * np.sin(ang), r1 * np.sin(ang)],
                color=WHITE, lw=1.2 if pct % 20 == 0 else 0.7, alpha=0.7, zorder=3)
        if pct % 20 == 0:
            rx = 0.94 * np.cos(ang)
            ry = 0.94 * np.sin(ang)
            ax.text(rx, ry, f"{pct}%", ha="center", va="center",
                    fontsize=7, color=WHITE, alpha=0.8, zorder=4)

    # target line at 90%
    ang_target = np.radians(180 - 90 * 1.8)
    ax.plot([0, 1.08 * np.cos(ang_target)],
            [0, 1.08 * np.sin(ang_target)],
            color=GOLD, lw=1.6, ls="--", alpha=0.9, zorder=5)
    ax.text(1.14 * np.cos(ang_target) + 0.02,
            1.14 * np.sin(ang_target),
            "TARGET", fontsize=6.5, color=GOLD, ha="left", va="center", zorder=6)

    # ── needle ────────────────────────────────────────────────────────────────
    ang_needle = np.radians(180 - recall_pct * 1.8)
    nx = 0.82 * np.cos(ang_needle)
    ny = 0.82 * np.sin(ang_needle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=stage_color,
                                lw=2.6, mutation_scale=16),
                zorder=7)
    # hub
    ax.add_patch(plt.Circle((0, 0), 0.055, color=stage_color, zorder=8))

    # ── text ──────────────────────────────────────────────────────────────────
    ax.text(0, -0.28, f"{recall_pct:.0f}%",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=stage_color, zorder=9)
    ax.text(0, -0.42, "Eyeglasses Recall",
            ha="center", va="center", fontsize=9, color=WHITE, alpha=0.85, zorder=9)
    ax.text(0, 1.14, stage_label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=stage_color, zorder=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                      edgecolor=stage_color, linewidth=1.2))


# Keyframes: (recall_pct, label, colour)
KEYFRAMES = [
    (85.0, "Random Forest baseline",       DANGER),
    (86.0, "Random Forest baseline",       DANGER),
    (87.0, "Linear SVM  (kernel='linear')", CAUTION),
    (87.5, "Linear SVM  (kernel='linear')", CAUTION),
    (88.0, "Linear SVM  (kernel='linear')", CAUTION),
    (89.0, "RBF SVM  (kernel='rbf')",       INFO),
    (89.5, "RBF SVM  (kernel='rbf')",       INFO),
    (90.0, "RBF SVM  C=10, g=0.01  [TARGET MET]",     SUCCESS),
    (90.0, "RBF SVM  C=10, g=0.01  [TARGET MET]",     SUCCESS),
    (90.0, "RBF SVM  C=10, g=0.01  [TARGET MET]",     SUCCESS),
]

frames = []
for recall, label, colour in KEYFRAMES:
    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
    _needle_frame(ax, recall, label, colour)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)[:, :, :3]  # drop alpha channel
    frames.append(buf)
    plt.close(fig)

# Save as GIF using only stdlib + matplotlib (no Pillow needed if unavailable)
try:
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    durations = [400] * (len(imgs) - 3) + [800, 800, 1200]
    imgs[0].save(
        IMG_DIR / "ch04-svm-needle.gif",
        save_all=True, append_images=imgs[1:],
        duration=durations, loop=0,
    )
    print(f"wrote {IMG_DIR / 'ch04-svm-needle.gif'}  ({len(imgs)} frames)")
except ImportError:
    # Fallback: save final frame as PNG if Pillow is not installed
    from PIL import Image  # noqa – re-raise with helpful message
    raise SystemExit(
        "Pillow is required for GIF export.  pip install Pillow"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2 · PROGRESS CHECK PNG — FaceAI constraint scorecard
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")

ax.text(0.5, 0.97, "FaceAI — Progress Check after Ch.4 SVM",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=13, fontweight="bold", color=WHITE)

constraints = [
    ("#1 ACCURACY",
     "RF: 91% avg, 85% Eyeglasses recall",
     "RBF SVM: 91% avg, 90% recall  [DONE]",
     SUCCESS),
    ("#2 GENERALIZATION",
     "RF generalised well",
     "Max-margin: wider boundary -> better on unseen faces",
     SUCCESS),
    ("#3 MULTI-LABEL <200ms",
     "40 RF classifiers in pipeline",
     "40 SVM classifiers; ~140ms/face  [DONE]",
     SUCCESS),
    ("#4 INTERPRETABILITY",
     "RF feature importance",
     "w = sum(ai*yi*xi) -> HOG region weights overlay",
     SUCCESS),
    ("#5 PRODUCTION",
     "RF sklearn pipeline",
     "make_pipeline(StandardScaler(), SVC(...))  [DONE]",
     SUCCESS),
]

row_h = 0.145
y0    = 0.86
col_x = [0.01, 0.19, 0.60]

# header
for x, txt in zip(col_x, ["Constraint", "Before Ch.4", "After Ch.4"]):
    ax.text(x, y0 + 0.04, txt, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=GOLD, va="center")

for i, (cname, before, after, colour) in enumerate(constraints):
    y = y0 - (i + 0.5) * row_h
    # row background
    rect = mpatches.FancyBboxPatch(
        (0.005, y - row_h * 0.44), 0.99, row_h * 0.88,
        boxstyle="round,pad=0.008", linewidth=0,
        facecolor="#ffffff08", transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(rect)
    ax.text(col_x[0], y, cname, transform=ax.transAxes,
            fontsize=7.8, color=WHITE, va="center", fontweight="bold")
    ax.text(col_x[1], y, before, transform=ax.transAxes,
            fontsize=7.2, color=GREY, va="center")
    ax.text(col_x[2], y, after, transform=ax.transAxes,
            fontsize=7.2, color=colour, va="center", fontweight="bold")

# recall bar chart at the bottom
ax_bar = fig.add_axes([0.06, 0.04, 0.88, 0.12], facecolor=BG)
ax_bar.set_facecolor(BG)
models   = ["RF\nbaseline", "Linear SVM\n(kernel='linear')", "RBF SVM\n(kernel='rbf')"]
recalls  = [85, 87, 90]
colours  = [DANGER, CAUTION, SUCCESS]
bars = ax_bar.barh(models, recalls, color=colours, height=0.55, zorder=3)
ax_bar.axvline(90, color=GOLD, lw=1.4, ls="--", zorder=4)
ax_bar.text(90.5, 2.45, "TARGET 90%", color=GOLD, fontsize=7, va="top")
ax_bar.set_xlim(80, 95)
ax_bar.set_xlabel("Eyeglasses Recall (%)", fontsize=8, color=WHITE)
ax_bar.tick_params(colors=WHITE, labelsize=7.5)
for spine in ax_bar.spines.values():
    spine.set_edgecolor(GREY)
for bar, val in zip(bars, recalls):
    ax_bar.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=8, color=WHITE, fontweight="bold")

fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.16)
out_png = IMG_DIR / "ch04-svm-progress-check.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"wrote {out_png}")
