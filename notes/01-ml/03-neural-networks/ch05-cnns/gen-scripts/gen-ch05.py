"""
gen_ch05.py — generate visual assets for Ch.5 CNNs
────────────────────────────────────────────────────
Produces:
  img/ch05-cnns-needle.gif        CelebA Smiling accuracy: HOG+logistic 88% → CNN 93%
  img/ch05-cnns-progress-check.png  UnifiedAI constraint dashboard

Usage:
  python gen_scripts/gen_ch05.py
from the chapter root  (notes/01-ml/03_neural_networks/ch05_cnns/).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── colour palette (matches authoring-guide conventions) ──────────────────────
BG         = "#1a1a2e"
FG         = "#e2e8f0"
PRIMARY    = "#1e3a8a"
SUCCESS    = "#15803d"
CAUTION    = "#b45309"
DANGER     = "#b91c1c"
INFO       = "#1d4ed8"
NEEDLE_CLR = "#38bdf8"
ACC_CLR    = "#34d399"   # accuracy line colour
BASE_CLR   = "#f87171"   # baseline reference colour

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NEEDLE GIF — CelebA Smiling accuracy: HOG+logistic 88% → CNN 93%
# ══════════════════════════════════════════════════════════════════════════════

STAGES = [
    {
        "label": "HOG + Logistic Regression",
        "accuracy": 88.2,
        "caption": "Hand-crafted HOG features (1,764-dim).\nOrientation histograms in 8×8 cells — no learning.",
    },
    {
        "label": "+ 1 Conv Block (32 filters)",
        "accuracy": 89.8,
        "caption": "One shared 3×3 filter bank detects edges.\nWeight sharing: 320 params vs 524k for dense.",
    },
    {
        "label": "+ Pooling + 2 Conv Blocks",
        "accuracy": 91.4,
        "caption": "MaxPool adds local translation invariance.\nSecond conv block detects textures and corners.",
    },
    {
        "label": "CNN (3 conv blocks + dense head)",
        "accuracy": 93.0,
        "caption": "Hierarchical features: edges → textures → face parts.\n93% accuracy — beats all HOG baselines. ✅",
    },
]

NEEDLE_MIN = 75.0    # gauge minimum (%)
NEEDLE_MAX = 100.0   # gauge maximum (%)
N_TRANSITION = 20    # frames per stage transition
N_HOLD = 10          # frames to hold at each stage


def _acc_to_angle(acc: float) -> float:
    """Map accuracy value to needle angle in degrees (180°=left/low, 0°=right/high)."""
    frac = (acc - NEEDLE_MIN) / (NEEDLE_MAX - NEEDLE_MIN)
    frac = np.clip(frac, 0.0, 1.0)
    return 180.0 - frac * 180.0


def _draw_gauge(ax, accuracy: float, label: str, caption: str):
    """Draw a semicircular accuracy gauge with a single needle."""
    ax.set_facecolor(BG)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.6, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    theta = np.linspace(0, np.pi, 300)

    # --- arc colour zones (low = danger, mid = caution, high = success) ------
    danger_theta  = np.linspace(np.pi,       np.pi * 2 / 3, 100)
    caution_theta = np.linspace(np.pi * 2/3, np.pi / 3,     100)
    success_theta = np.linspace(np.pi / 3,   0,             100)

    for th, col in [
        (danger_theta,  DANGER),
        (caution_theta, CAUTION),
        (success_theta, SUCCESS),
    ]:
        ax.fill_between(np.cos(th), np.zeros_like(th), np.sin(th) * 0.9,
                        color=col, alpha=0.22)

    # outer arc
    ax.plot(np.cos(theta), np.sin(theta), color=FG, lw=2, alpha=0.55)

    # tick marks and labels
    tick_vals = [75, 80, 85, 88, 90, 93, 95, 100]
    for v in tick_vals:
        angle_deg = _acc_to_angle(v)
        angle_rad = np.deg2rad(angle_deg)
        inner, outer = 0.82, 0.97
        ax.plot(
            [inner * np.cos(angle_rad), outer * np.cos(angle_rad)],
            [inner * np.sin(angle_rad), outer * np.sin(angle_rad)],
            color=FG, lw=1.1, alpha=0.6,
        )
        ax.text(
            1.12 * np.cos(angle_rad), 1.12 * np.sin(angle_rad),
            f"{v}%", ha="center", va="center", fontsize=7,
            color=FG, alpha=0.7,
        )

    # --- main accuracy needle ------------------------------------------------
    angle_rad = np.deg2rad(_acc_to_angle(accuracy))
    needle_color = SUCCESS if accuracy >= 92.0 else (CAUTION if accuracy >= 88.5 else BASE_CLR)
    ax.annotate(
        "", xy=(0.72 * np.cos(angle_rad), 0.72 * np.sin(angle_rad)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=needle_color, lw=3.5,
                        mutation_scale=18),
    )

    # --- HOG baseline marker (dashed reference line at 88.2%) ----------------
    ref_angle = np.deg2rad(_acc_to_angle(88.2))
    ax.plot(
        [0, 0.68 * np.cos(ref_angle)],
        [0, 0.68 * np.sin(ref_angle)],
        color=BASE_CLR, lw=1.5, linestyle="--", alpha=0.6,
    )
    ax.text(
        0.78 * np.cos(ref_angle), 0.78 * np.sin(ref_angle) + 0.04,
        "HOG\nbaseline", ha="center", va="center", fontsize=6.5,
        color=BASE_CLR, alpha=0.8,
    )

    # hub dot
    ax.add_patch(plt.Circle((0, 0), 0.07, color=FG, zorder=5))

    # --- accuracy readout ----------------------------------------------------
    acc_color = SUCCESS if accuracy >= 92.0 else (CAUTION if accuracy >= 88.5 else BASE_CLR)
    ax.text(0, -0.18, f"{accuracy:.1f}%",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=acc_color)
    ax.text(0, -0.34, "CelebA Smiling Accuracy",
            ha="center", va="center", fontsize=8.5, color=FG, alpha=0.75)

    # --- stage label ---------------------------------------------------------
    ax.text(0, 1.42, label, ha="center", va="top", fontsize=10,
            fontweight="bold", color=FG, wrap=True)

    # --- caption -------------------------------------------------------------
    ax.text(0, -0.50, caption, ha="center", va="top", fontsize=8,
            color=FG, alpha=0.8, linespacing=1.45)


def build_needle_gif():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor(BG)

    # build frame list: hold → transition → hold ...
    frames = []
    for si, stage in enumerate(STAGES):
        if si == 0:
            for _ in range(N_HOLD):
                frames.append((stage["accuracy"], si))
        else:
            prev = STAGES[si - 1]
            for f in range(N_TRANSITION):
                t = f / N_TRANSITION
                t_smooth = t * t * (3 - 2 * t)   # smoothstep
                acc = prev["accuracy"] + (stage["accuracy"] - prev["accuracy"]) * t_smooth
                frames.append((acc, si))
            for _ in range(N_HOLD):
                frames.append((stage["accuracy"], si))

    def update(frame_idx):
        ax.clear()
        acc, si = frames[frame_idx]
        stage = STAGES[si]
        _draw_gauge(ax, acc, stage["label"], stage["caption"])
        return (ax,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=55, blit=False)
    out_path = IMG_DIR / "ch05-cnns-needle.gif"
    anim.save(str(out_path), writer=PillowWriter(fps=18), dpi=120)
    plt.close(fig)
    print(f"  ✓ saved {out_path.relative_to(IMG_DIR.parent.parent.parent.parent.parent)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROGRESS CHECK PNG — UnifiedAI constraint dashboard
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    {
        "num": "#1",
        "name": "ACCURACY",
        "target": "≤$28k MAE + ≥95% acc",
        "status": "⚡ close",
        "value": "CNN 93% Smiling",
        "color": CAUTION,
        "pct": 0.75,
    },
    {
        "num": "#2",
        "name": "GENERALIZATION",
        "target": "Unseen faces + districts",
        "status": "✅ improved",
        "value": "Weight sharing helps",
        "color": SUCCESS,
        "pct": 0.80,
    },
    {
        "num": "#3",
        "name": "MULTI-TASK",
        "target": "Value + 40 Attributes",
        "status": "⚡ partial",
        "value": "Multi-label head possible",
        "color": CAUTION,
        "pct": 0.45,
    },
    {
        "num": "#4",
        "name": "INTERPRETABILITY",
        "target": "Explainable predictions",
        "status": "❌ blocked",
        "value": "Ch.11 SHAP / Ch.18",
        "color": DANGER,
        "pct": 0.10,
    },
    {
        "num": "#5",
        "name": "PRODUCTION",
        "target": "Scale + Monitor",
        "status": "⚡ partial",
        "value": "Fast inference; no monitor",
        "color": CAUTION,
        "pct": 0.40,
    },
]

# CelebA Smiling accuracy at each method stage
ACC_HISTORY = [
    ("Dense\nMLP\n(pixels)", 81.4),
    ("HOG +\nLogistic", 88.2),
    ("HOG +\nSVM", 89.1),
    ("CNN\n3 blocks", 93.0),
]

# Parameter count comparison (log scale)
PARAM_COMPARISON = [
    ("Dense\nlayer 1", 503_578_624),
    ("Conv1\n(32 filt)", 320),
    ("Conv2\n(64 filt)", 18_496),
    ("Conv3\n(128 filt)", 73_856),
]


def build_progress_check():
    fig = plt.figure(figsize=(13, 7.5), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.40,
                          left=0.06, right=0.97, top=0.88, bottom=0.10)

    # ── top-left: UnifiedAI constraint progress bars ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    ax1.set_title("UnifiedAI Constraint Progress", color=FG, fontsize=10,
                  fontweight="bold", pad=8)

    bar_h = 0.52
    for i, c in enumerate(CONSTRAINTS):
        y = len(CONSTRAINTS) - 1 - i
        ax1.barh(y, 1.0, height=bar_h, color="#1e2a3a", left=0)
        if c["pct"] > 0:
            ax1.barh(y, c["pct"], height=bar_h, color=c["color"], left=0, alpha=0.85)
        label = f"{c['num']} {c['name']}  ({c['value']})"
        ax1.text(0.02, y, label, va="center", fontsize=7.5, color=FG, fontweight="bold")

    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(-0.6, len(CONSTRAINTS) - 0.4)
    ax1.axis("off")

    # ── top-centre: CelebA Smiling accuracy bar chart ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    ax2.set_title("CelebA Smiling Accuracy (%)", color=FG, fontsize=10,
                  fontweight="bold", pad=8)

    labels, accs = zip(*ACC_HISTORY)
    xs = np.arange(len(labels))
    bar_colors = [DANGER, CAUTION, CAUTION, SUCCESS]
    bars = ax2.bar(xs, accs, color=bar_colors, alpha=0.85, width=0.55, edgecolor=BG)
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, color=FG)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8, color=FG)
    ax2.set_ylim(70, 100)
    ax2.axhline(95, color=SUCCESS, lw=1.5, linestyle="--", alpha=0.7)
    ax2.text(3.3, 95.3, "Target 95%", color=SUCCESS, fontsize=7.5, va="bottom")
    ax2.tick_params(colors=FG)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2d3748")
    ax2.yaxis.label.set_color(FG)
    ax2.set_ylabel("Accuracy (%)", color=FG, fontsize=8)

    # ── top-right: parameter count (log scale) ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(BG)
    ax3.set_title("Parameters per Layer (log scale)", color=FG, fontsize=10,
                  fontweight="bold", pad=8)

    param_labels, param_vals = zip(*PARAM_COMPARISON)
    xs3 = np.arange(len(param_labels))
    pcols = [DANGER, SUCCESS, SUCCESS, SUCCESS]
    bars3 = ax3.bar(xs3, param_vals, color=pcols, alpha=0.85, width=0.55,
                    edgecolor=BG, log=True)
    for bar, pv in zip(bars3, param_vals):
        if pv >= 1_000_000:
            lbl = f"{pv/1e6:.0f}M"
        elif pv >= 1_000:
            lbl = f"{pv/1e3:.1f}k"
        else:
            lbl = str(pv)
        ax3.text(bar.get_x() + bar.get_width() / 2, pv * 1.8,
                 lbl, ha="center", va="bottom", fontsize=8.5, color=FG)
    ax3.set_xticks(xs3)
    ax3.set_xticklabels(param_labels, fontsize=7.5, color=FG)
    ax3.tick_params(colors=FG)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#2d3748")
    ax3.yaxis.label.set_color(FG)
    ax3.set_ylabel("Parameters", color=FG, fontsize=8)
    ax3.set_ylim(100, 2e9)

    # ── bottom row: receptive field growth + feature hierarchy ────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(BG)
    ax4.set_title("Receptive Field vs Depth (3×3 convs)", color=FG, fontsize=10,
                  fontweight="bold", pad=8)

    layers = [0, 1, 2, 3]
    rf = [1, 3, 5, 7]
    ax4.plot(layers, rf, "o-", color=INFO, lw=2.5, markersize=9)
    for l, r in zip(layers, rf):
        ax4.text(l, r + 0.2, f"{r}×{r}", ha="center", va="bottom",
                 fontsize=9, color=FG)
    ax4.set_xticks(layers)
    ax4.set_xticklabels(["Input", "After\nConv1", "After\nConv2", "After\nConv3"],
                        fontsize=8, color=FG)
    ax4.set_ylabel("Receptive field (px)", color=FG, fontsize=8)
    ax4.set_ylim(0, 9)
    ax4.tick_params(colors=FG)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#2d3748")

    # ── bottom-centre: spatial dimension reduction through the network ─────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(BG)
    ax5.set_title("Spatial Dimensions Through CNN", color=FG, fontsize=10,
                  fontweight="bold", pad=8)

    layer_names = ["Input", "Conv1", "Pool1", "Conv2", "Pool2", "Conv3", "Pool3"]
    heights = [64, 62, 31, 29, 14, 12, 6]
    channels = [1, 32, 32, 64, 64, 128, 128]

    xs5 = np.arange(len(layer_names))
    ax5_twin = ax5.twinx()
    ax5.bar(xs5, heights, color=INFO, alpha=0.7, width=0.4, label="H×W (px)")
    ax5_twin.plot(xs5, channels, "s--", color=CAUTION, lw=2.0, markersize=8,
                  label="Channels")

    ax5.set_xticks(xs5)
    ax5.set_xticklabels(layer_names, fontsize=7.5, color=FG, rotation=30, ha="right")
    ax5.set_ylabel("Spatial size (px)", color=FG, fontsize=8)
    ax5_twin.set_ylabel("Channels", color=CAUTION, fontsize=8)
    ax5.tick_params(colors=FG)
    ax5_twin.tick_params(colors=CAUTION)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#2d3748")
    ax5.set_ylim(0, 80)
    ax5_twin.set_ylim(0, 160)

    lines1 = [mpatches.Patch(color=INFO, alpha=0.7, label="Spatial size")]
    lines2 = [mpatches.Patch(color=CAUTION, label="Channels")]
    ax5.legend(handles=lines1 + lines2, loc="upper right",
               fontsize=7, facecolor=BG, labelcolor=FG, edgecolor="#2d3748")

    # ── bottom-right: feature hierarchy ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG)
    ax6.set_title("Feature Hierarchy (CelebA)", color=FG, fontsize=10,
                  fontweight="bold", pad=8)
    ax6.axis("off")

    levels = [
        ("Layer 1", "Edges, gradients, colour blobs", SUCCESS),
        ("Layer 2", "Corners, stripes, curved edges", INFO),
        ("Layer 3", "Eye shapes, lip contours, hair texture", CAUTION),
        ("Layer 4+", "Smiling, glasses, young, blonde…", PRIMARY),
        ("Output", "Binary attribute probability", SUCCESS),
    ]
    for i, (lname, desc, col) in enumerate(levels):
        y = 0.88 - i * 0.20
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.07), 0.96, 0.13,
            boxstyle="round,pad=0.01", facecolor=col, alpha=0.25, edgecolor=col,
        ))
        ax6.text(0.08, y, lname, va="center", fontsize=9,
                 fontweight="bold", color=col)
        ax6.text(0.35, y, desc, va="center", fontsize=7.5, color=FG, alpha=0.9)
        if i < len(levels) - 1:
            ax6.annotate("", xy=(0.5, y - 0.09), xytext=(0.5, y - 0.06),
                         arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.0, alpha=0.5))

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)

    # ── figure title ──────────────────────────────────────────────────────────
    fig.suptitle("Ch.5 · CNNs — UnifiedAI Progress Dashboard",
                 color=FG, fontsize=13, fontweight="bold", y=0.96)

    out_path = IMG_DIR / "ch05-cnns-progress-check.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ saved {out_path.relative_to(IMG_DIR.parent.parent.parent.parent.parent)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Ch.5 CNNs — generating visual assets …")
    build_needle_gif()
    build_progress_check()
    print("Done.")
