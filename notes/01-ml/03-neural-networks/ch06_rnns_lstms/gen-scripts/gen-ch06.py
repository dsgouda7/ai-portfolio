"""
gen_ch06.py — generate visual assets for Ch.6 RNNs / LSTMs
────────────────────────────────────────────────────────────
Produces
  img/ch06-rnns-lstms-needle.gif          prediction accuracy needle for housing time series
  img/ch06-rnns-lstms-progress-check.png  UnifiedAI constraint dashboard

Usage (from chapter root):
  python gen_scripts/gen_ch06.py
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

# ── colour palette ─────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
FG        = "#e2e8f0"
PRIMARY   = "#1e3a8a"
SUCCESS   = "#15803d"
CAUTION   = "#b45309"
DANGER    = "#b91c1c"
INFO      = "#1d4ed8"
NEEDLE_C  = "#38bdf8"
TRAIN_C   = "#34d399"
VAL_C     = "#f87171"

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NEEDLE GIF — prediction accuracy improving with LSTM memory
# ══════════════════════════════════════════════════════════════════════════════

STAGES = [
    {
        "label": "Dense baseline\n(memoryless)",
        "value": 60.0,
        "caption": "MLP treats 12 months as independent features.\nIgnores temporal order — MAPE ~9%.",
    },
    {
        "label": "Vanilla RNN\n(short memory)",
        "value": 74.0,
        "caption": "RNN hidden state threads through time.\nCaptures recent trend — MAPE ~5.5%.",
    },
    {
        "label": "LSTM\n(gated memory)",
        "value": 87.0,
        "caption": "LSTM cell state retains full seasonal cycle.\nForget gate erases stale signals — MAPE ~3%.",
    },
    {
        "label": "LSTM + dropout\n+ grad clip",
        "value": 93.0,
        "caption": "Regularised LSTM generalises to unseen months.\nProduction-ready — MAPE ~2.2%",
    },
]

GAUGE_MIN   = 40.0
GAUGE_MAX   = 100.0
N_TRANSIT   = 20      # frames per stage transition
N_HOLD      = 10      # frames to hold each stage


def _val_to_angle(val: float) -> float:
    """Map accuracy value to needle angle: 180°=left/low, 0°=right/high."""
    frac = (val - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)
    return 180.0 - frac * 180.0


def _draw_gauge(ax, angle: float, label: str, caption: str, val: float) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.30, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── arc background ─────────────────────────────────────────────────────────
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#374151", linewidth=14, solid_capstyle="butt")

    # ── coloured progress arc ─────────────────────────────────────────────────
    frac  = (val - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)
    theta2 = np.linspace(np.pi, np.pi - frac * np.pi, 300)
    col = DANGER if val < 65 else (CAUTION if val < 80 else SUCCESS)
    ax.plot(np.cos(theta2), np.sin(theta2), color=col, linewidth=14, solid_capstyle="butt")

    # ── tick marks ─────────────────────────────────────────────────────────────
    for tick_val in [40, 50, 60, 70, 80, 90, 100]:
        tick_frac = (tick_val - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)
        tick_angle = np.pi - tick_frac * np.pi
        for r_in, r_out in [(0.82, 0.92)]:
            ax.plot([r_in * np.cos(tick_angle), r_out * np.cos(tick_angle)],
                    [r_in * np.sin(tick_angle), r_out * np.sin(tick_angle)],
                    color=FG, linewidth=1.2, alpha=0.6)
        ax.text(1.06 * np.cos(tick_angle), 1.06 * np.sin(tick_angle),
                str(tick_val), ha="center", va="center", fontsize=6,
                color=FG, alpha=0.7)

    # ── needle ────────────────────────────────────────────────────────────────
    rad = np.deg2rad(angle)
    ax.annotate("", xy=(0.78 * np.cos(rad), 0.78 * np.sin(rad)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=NEEDLE_C, lw=2.0,
                                mutation_scale=14))
    ax.add_patch(plt.Circle((0, 0), 0.06, color=NEEDLE_C, zorder=5))

    # ── value text ─────────────────────────────────────────────────────────────
    ax.text(0, -0.08, f"{val:.1f}%", ha="center", va="top",
            fontsize=20, fontweight="bold", color=FG)

    # ── stage label ───────────────────────────────────────────────────────────
    ax.text(0, 1.12, label, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=FG, linespacing=1.4)

    # ── caption ───────────────────────────────────────────────────────────────
    ax.text(0, -0.20, caption, ha="center", va="top", fontsize=7.5,
            color=FG, alpha=0.85, linespacing=1.4)

    # ── x-axis label ─────────────────────────────────────────────────────────
    ax.text(0, -0.28, "Sequence Prediction Accuracy (%)", ha="center", va="top",
            fontsize=7, color=FG, alpha=0.6, family="monospace")


def build_needle_gif() -> None:
    fig, ax = plt.subplots(figsize=(5, 3.8))
    fig.patch.set_facecolor(BG)

    frames_list: list[tuple[float, str, str, float]] = []
    prev_val = STAGES[0]["value"]
    prev_ang = _val_to_angle(prev_val)

    for stage in STAGES:
        tgt_val  = float(stage["value"])
        tgt_ang  = _val_to_angle(tgt_val)
        label    = stage["label"]
        caption  = stage["caption"]

        # transition frames
        for k in range(N_TRANSIT):
            t   = k / (N_TRANSIT - 1)
            val = prev_val + (tgt_val - prev_val) * t
            ang = prev_ang + (tgt_ang - prev_ang) * t
            frames_list.append((ang, label, caption, val))

        # hold frames
        for _ in range(N_HOLD):
            frames_list.append((tgt_ang, label, caption, tgt_val))

        prev_val = tgt_val
        prev_ang = tgt_ang

    def _update(i: int):
        ax.clear()
        ang, lbl, cap, val = frames_list[i]
        _draw_gauge(ax, ang, lbl, cap, val)

    anim = FuncAnimation(fig, _update, frames=len(frames_list), interval=60)
    out  = IMG_DIR / "ch06-rnns-lstms-needle.gif"
    anim.save(str(out), writer=PillowWriter(fps=18))
    plt.close(fig)
    print(f"  [OK]  {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROGRESS CHECK PNG — UnifiedAI constraint dashboard
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    {"label": "#1  ACCURACY\n≤$28k MAE + ≥95% acc",  "pct": 55, "note": "LSTM: 2–4% MAPE\non housing index"},
    {"label": "#2  GENERALIZATION\nUnseen districts",   "pct": 60, "note": "Chronological split\nno future leakage"},
    {"label": "#3  MULTI-TASK\nRegression + class",    "pct": 70, "note": "Same LSTM cell\nswap output head"},
    {"label": "#4  INTERPRETABILITY\nExplainable preds","pct": 10, "note": "Gates are internal\n→ need Ch.9 attention"},
    {"label": "#5  PRODUCTION\n<100ms inference",       "pct": 45, "note": "<100ms for T≤52\nno monitoring yet"},
]


def build_progress_check() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.2))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Ch.6 — RNNs / LSTMs · UnifiedAI Constraint Dashboard",
                 fontsize=12, fontweight="bold", color=FG, y=1.02)

    for ax, c in zip(axes, CONSTRAINTS):
        pct  = c["pct"]
        col  = SUCCESS if pct >= 70 else (CAUTION if pct >= 40 else DANGER)

        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # background bar
        bar_h = 0.28
        bar_y = 0.32
        ax.add_patch(mpatches.FancyBboxPatch((0.1, bar_y), 0.8, bar_h,
                                              boxstyle="round,pad=0.02",
                                              linewidth=0, color="#374151"))

        # filled progress
        fill_w = 0.8 * pct / 100
        if fill_w > 0.005:
            ax.add_patch(mpatches.FancyBboxPatch((0.1, bar_y), fill_w, bar_h,
                                                  boxstyle="round,pad=0.02",
                                                  linewidth=0, color=col))

        # percentage label
        ax.text(0.5, bar_y + bar_h / 2, f"{pct}%",
                ha="center", va="center", fontsize=13,
                fontweight="bold", color=FG)

        # constraint label
        ax.text(0.5, 0.78, c["label"], ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=FG, linespacing=1.5)

        # note
        ax.text(0.5, 0.13, c["note"], ha="center", va="center",
                fontsize=6.8, color=FG, alpha=0.75, linespacing=1.4)

        # status label
        icon = "OK" if pct >= 70 else ("~" if pct >= 40 else "X")
        icon_col = SUCCESS if pct >= 70 else (CAUTION if pct >= 40 else DANGER)
        ax.text(0.5, bar_y - 0.12, icon, ha="center", va="center",
                fontsize=12, fontweight="bold", color=icon_col)

    plt.tight_layout()
    out = IMG_DIR / "ch06-rnns-lstms-progress-check.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [OK]  {out}")


# ══════════════════════════════════════════════════════════════════════════════
# entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Ch.6 visual assets …")
    build_needle_gif()
    build_progress_check()
    print("Done.")
