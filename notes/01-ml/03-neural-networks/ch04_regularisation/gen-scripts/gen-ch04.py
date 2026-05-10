"""
gen_ch04.py — generate visual assets for Ch.4 Regularisation
─────────────────────────────────────────────────────────────
Produces:
  img/ch04-regularisation-needle.gif        train-val gap narrowing $36k → $18k
  img/ch04-regularisation-progress-check.png  UnifiedAI constraint dashboard

Usage:
  python gen_scripts/gen_ch04.py
from the chapter root  (notes/01-ml/03_neural_networks/ch04_regularisation/).
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
BG = "#1a1a2e"
FG = "#e2e8f0"
PRIMARY = "#1e3a8a"
SUCCESS = "#15803d"
CAUTION = "#b45309"
DANGER = "#b91c1c"
INFO = "#1d4ed8"
NEEDLE_COLOR = "#38bdf8"
TRAIN_COLOR = "#34d399"
VAL_COLOR = "#f87171"
RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NEEDLE GIF — train-val gap narrowing
# ══════════════════════════════════════════════════════════════════════════════

STAGES = [
    {
        "label": "No regularisation",
        "train_mae": 32.0,
        "val_mae": 68.0,
        "caption": "Model memorises training districts.\nWeights grow large to fit every quirk.",
    },
    {
        "label": "+ L2  (λ=0.001)",
        "train_mae": 34.0,
        "val_mae": 58.0,
        "caption": "Weight decay shrinks large coefficients.\nGap narrows by $10k.",
    },
    {
        "label": "+ Dropout  (p=0.3)",
        "train_mae": 36.0,
        "val_mae": 54.0,
        "caption": "Random neuron silencing forces redundancy.\nGap falls to $18k.",
    },
    {
        "label": "+ BatchNorm + Early Stop",
        "train_mae": 38.0,
        "val_mae": 56.0,
        "caption": "BatchNorm stabilises; early stop recovers\nbest checkpoint.  Gap: $18k ✅",
    },
]

NEEDLE_MIN = 10.0   # gauge minimum ($k)
NEEDLE_MAX = 90.0   # gauge maximum ($k)
N_TRANSITION = 18   # frames per stage transition
N_HOLD = 8          # frames to hold at each stage
TOTAL_STAGES = len(STAGES)


def _mae_to_angle(mae: float) -> float:
    """Map MAE value to needle angle in degrees (180°=left/high, 0°=right/low)."""
    frac = (mae - NEEDLE_MIN) / (NEEDLE_MAX - NEEDLE_MIN)
    frac = np.clip(frac, 0.0, 1.0)
    return 180.0 - frac * 180.0


def _draw_gauge(ax, train_mae: float, val_mae: float, label: str, caption: str):
    """Draw a semicircular gauge showing train and val MAE needles."""
    ax.set_facecolor(BG)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- arc background zones ------------------------------------------------
    theta = np.linspace(0, np.pi, 300)
    # danger zone (high MAE): 180°–120°
    danger_theta = np.linspace(np.pi, np.pi * 2 / 3, 100)
    # caution zone: 120°–60°
    caution_theta = np.linspace(np.pi * 2 / 3, np.pi / 3, 100)
    # success zone: 60°–0°
    success_theta = np.linspace(np.pi / 3, 0, 100)

    for th, col, alpha in [
        (danger_theta, DANGER, 0.25),
        (caution_theta, CAUTION, 0.25),
        (success_theta, SUCCESS, 0.25),
    ]:
        ax.fill_between(np.cos(th), np.zeros_like(th), np.sin(th) * 0.9,
                        color=col, alpha=alpha)

    # outer arc
    ax.plot(np.cos(theta), np.sin(theta), color=FG, lw=2, alpha=0.6)

    # tick marks and labels
    tick_vals = [20, 30, 40, 50, 60, 70, 80]
    for v in tick_vals:
        angle_deg = _mae_to_angle(v)
        angle_rad = np.deg2rad(angle_deg)
        inner, outer = 0.80, 0.95
        ax.plot(
            [inner * np.cos(angle_rad), outer * np.cos(angle_rad)],
            [inner * np.sin(angle_rad), outer * np.sin(angle_rad)],
            color=FG, lw=1.2, alpha=0.6,
        )
        ax.text(
            1.12 * np.cos(angle_rad), 1.12 * np.sin(angle_rad),
            f"${v}k", ha="center", va="center", fontsize=7,
            color=FG, alpha=0.7,
        )

    # --- needles -------------------------------------------------------------
    for mae, color, lbl, lw in [
        (train_mae, TRAIN_COLOR, "Train", 3.5),
        (val_mae, VAL_COLOR, "Val", 3.5),
    ]:
        angle_rad = np.deg2rad(_mae_to_angle(mae))
        ax.annotate(
            "", xy=(0.72 * np.cos(angle_rad), 0.72 * np.sin(angle_rad)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                            mutation_scale=16),
        )

    # hub
    ax.add_patch(plt.Circle((0, 0), 0.07, color=FG, zorder=5))

    # --- gap bar -------------------------------------------------------------
    gap = val_mae - train_mae
    gap_color = SUCCESS if gap <= 20 else (CAUTION if gap <= 30 else DANGER)
    ax.text(0, -0.22, f"Gap  ${gap:.0f}k",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=gap_color)

    # --- MAE readouts --------------------------------------------------------
    ax.text(-0.6, -0.38, f"Train\n${train_mae:.0f}k",
            ha="center", va="center", fontsize=10, color=TRAIN_COLOR)
    ax.text(0.6, -0.38, f"Val\n${val_mae:.0f}k",
            ha="center", va="center", fontsize=10, color=VAL_COLOR)

    # --- stage label ---------------------------------------------------------
    ax.text(0, 1.35, label, ha="center", va="top", fontsize=11,
            fontweight="bold", color=FG)

    # --- caption -------------------------------------------------------------
    ax.text(0, -0.52, caption, ha="center", va="top", fontsize=8.5,
            color=FG, alpha=0.8, linespacing=1.4)

    # --- legend dots ---------------------------------------------------------
    for x, col, lbl in [(-0.5, TRAIN_COLOR, "Train MAE"),
                         (0.5, VAL_COLOR, "Val MAE")]:
        ax.plot(x, 1.25, "o", color=col, markersize=7)
        ax.text(x + 0.04, 1.25, lbl, va="center", fontsize=7.5, color=FG)


def build_needle_gif():
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.patch.set_facecolor(BG)

    # build frame list: interpolate between stages
    frames = []  # each entry: (train_mae, val_mae, stage_idx)
    for si, stage in enumerate(STAGES):
        if si == 0:
            # hold on first stage
            for _ in range(N_HOLD):
                frames.append((stage["train_mae"], stage["val_mae"], si))
        else:
            prev = STAGES[si - 1]
            for f in range(N_TRANSITION):
                t = f / N_TRANSITION
                t_smooth = t * t * (3 - 2 * t)  # smoothstep
                tr = prev["train_mae"] + (stage["train_mae"] - prev["train_mae"]) * t_smooth
                vl = prev["val_mae"] + (stage["val_mae"] - prev["val_mae"]) * t_smooth
                frames.append((tr, vl, si))
            for _ in range(N_HOLD):
                frames.append((stage["train_mae"], stage["val_mae"], si))

    def update(frame_idx):
        ax.clear()
        tr, vl, si = frames[frame_idx]
        stage = STAGES[si]
        _draw_gauge(ax, tr, vl, stage["label"], stage["caption"])
        return (ax,)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    out_path = IMG_DIR / "ch04-regularisation-needle.gif"
    anim.save(str(out_path), writer=PillowWriter(fps=16), dpi=120)
    plt.close(fig)
    print(f"  ✓ saved {out_path.relative_to(SCRIPT_DIR.parent.parent.parent.parent.parent)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROGRESS CHECK PNG — constraint dashboard
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    {
        "num": "#1",
        "name": "ACCURACY",
        "target": "≤$28k MAE",
        "status": "⚠️ close",
        "value": "Train $38k",
        "color": CAUTION,
        "pct": 0.70,
    },
    {
        "num": "#2",
        "name": "GENERALIZATION",
        "target": "Gap < $20k",
        "status": "✅ ACHIEVED",
        "value": "Gap $18k",
        "color": SUCCESS,
        "pct": 1.00,
    },
    {
        "num": "#3",
        "name": "MULTI-TASK",
        "target": "Value + Segment",
        "status": "❌ blocked",
        "value": "Ch.5+",
        "color": DANGER,
        "pct": 0.0,
    },
    {
        "num": "#4",
        "name": "INTERPRETABILITY",
        "target": "Explainable",
        "status": "❌ blocked",
        "value": "Ch.8+",
        "color": DANGER,
        "pct": 0.0,
    },
    {
        "num": "#5",
        "name": "PRODUCTION",
        "target": "Scale + Monitor",
        "status": "❌ blocked",
        "value": "Ch.10+",
        "color": DANGER,
        "pct": 0.0,
    },
]

GAP_HISTORY = [
    ("Baseline\n(no reg)", 36.0),
    ("+L2\n(λ=0.001)", 24.0),
    ("+Dropout\n(p=0.3)", 18.0),
    ("+BN+\nEarlyStop", 18.0),
]


def build_progress_check():
    fig = plt.figure(figsize=(11, 6.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.88, bottom=0.10)

    # ── top-left: constraint progress bars ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    ax1.set_title("UnifiedAI Constraint Progress", color=FG, fontsize=11,
                  fontweight="bold", pad=8)

    bar_h = 0.55
    for i, c in enumerate(CONSTRAINTS):
        y = len(CONSTRAINTS) - 1 - i
        # background
        ax1.barh(y, 1.0, height=bar_h, color="#1e2a3a", left=0)
        # progress
        if c["pct"] > 0:
            ax1.barh(y, c["pct"], height=bar_h, color=c["color"], left=0, alpha=0.85)
        # label
        label = f"{c['num']} {c['name']}  ({c['value']})"
        ax1.text(0.02, y, label, va="center", fontsize=8.5, color=FG, fontweight="bold")

    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(-0.6, len(CONSTRAINTS) - 0.4)
    ax1.axis("off")

    # ── top-right: gap reduction bar chart ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    ax2.set_title("Train-Val Gap Reduction  ($k)", color=FG, fontsize=11,
                  fontweight="bold", pad=8)

    labels, gaps = zip(*GAP_HISTORY)
    xs = np.arange(len(labels))
    bar_colors = [DANGER, CAUTION, SUCCESS, SUCCESS]
    bars = ax2.bar(xs, gaps, color=bar_colors, alpha=0.85, width=0.55, edgecolor=BG)
    for bar, gap in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"${gap:.0f}k", ha="center", va="bottom", fontsize=9, color=FG)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8, color=FG)
    ax2.set_ylim(0, 44)
    ax2.axhline(20, color=SUCCESS, lw=1.5, linestyle="--", alpha=0.8)
    ax2.text(3.3, 21, "Target <$20k", color=SUCCESS, fontsize=8, va="bottom")
    ax2.tick_params(colors=FG)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2d3748")
    ax2.set_facecolor(BG)
    ax2.yaxis.label.set_color(FG)

    # ── bottom-left: train vs val MAE progression ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    ax3.set_title("MAE Before → After Regularisation", color=FG, fontsize=11,
                  fontweight="bold", pad=8)

    x = np.array([0, 1])
    train_before, train_after = 32.0, 38.0
    val_before, val_after = 68.0, 56.0

    ax3.plot(x, [train_before, train_after], "o-", color=TRAIN_COLOR, lw=2.5,
             markersize=8, label="Train MAE")
    ax3.plot(x, [val_before, val_after], "s-", color=VAL_COLOR, lw=2.5,
             markersize=8, label="Val MAE")
    ax3.fill_between(x, [train_before, train_after], [val_before, val_after],
                     alpha=0.12, color=CAUTION)

    for xv, tv, vv, tag in [(0, train_before, val_before, "Before"),
                             (1, train_after, val_after, "After")]:
        ax3.text(xv, tv - 2.5, f"${tv:.0f}k", ha="center", va="top",
                 fontsize=9, color=TRAIN_COLOR)
        ax3.text(xv, vv + 1.5, f"${vv:.0f}k", ha="center", va="bottom",
                 fontsize=9, color=VAL_COLOR)

    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["No\nreg", "Full\nstack"], fontsize=9, color=FG)
    ax3.set_ylim(20, 80)
    ax3.tick_params(colors=FG)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#2d3748")
    ax3.legend(fontsize=8.5, facecolor="#1e2a3a", edgecolor="#2d3748",
               labelcolor=FG, loc="upper right")

    # ── bottom-right: technique contribution breakdown ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    ax4.set_title("Gap Reduction by Technique", color=FG, fontsize=11,
                  fontweight="bold", pad=8)

    techniques = ["L2\nweight decay", "Dropout\n(p=0.3)", "BatchNorm\n+Early Stop"]
    contributions = [12.0, 6.0, 0.0]   # $k gap reduction each step
    tech_colors = [INFO, PRIMARY, "#7c3aed"]

    ax4.barh(np.arange(len(techniques)), contributions, color=tech_colors,
             height=0.55, alpha=0.85, edgecolor=BG)
    for i, (v, t) in enumerate(zip(contributions, techniques)):
        if v > 0:
            ax4.text(v + 0.3, i, f"−${v:.0f}k gap", va="center",
                     fontsize=9, color=FG)
        else:
            ax4.text(0.3, i, "stabilise training", va="center",
                     fontsize=9, color=FG, alpha=0.7)
    ax4.set_yticks(np.arange(len(techniques)))
    ax4.set_yticklabels(techniques, fontsize=8.5, color=FG)
    ax4.set_xlim(0, 18)
    ax4.tick_params(colors=FG)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#2d3748")

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Ch.4 — Regularisation  |  Constraint #2 GENERALIZATION ✅  (gap $36k → $18k)",
        fontsize=12, fontweight="bold", color=FG, y=0.96,
    )

    out_path = IMG_DIR / "ch04-regularisation-progress-check.png"
    fig.savefig(str(out_path), dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved {out_path.relative_to(SCRIPT_DIR.parent.parent.parent.parent.parent)}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Ch.4 Regularisation visual assets …")
    build_needle_gif()
    build_progress_check()
    print("Done.")
