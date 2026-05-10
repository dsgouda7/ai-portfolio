"""
gen_ch05.py — Generate visual assets for Ch.5 Ensemble Anomaly Detection.

Produces:
  img/ch05-ensemble-anomaly-needle.gif   — recall needle 78% → 82% ✅ over target
  img/ch05-ensemble-anomaly-progress-check.png — FraudShield constraint status

Run from repo root:
    python notes/01-ml/05_anomaly_detection/ch05_ensemble_anomaly/gen_scripts/gen_ch05.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
IMG_DIR = HERE.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

# ── canonical colours (authoring-guide) ──────────────────────────────────────
BG = "#1a1a2e"
PRIMARY = "#1e3a8a"
SUCCESS = "#15803d"
CAUTION = "#b45309"
DANGER = "#b91c1c"
INFO = "#1d4ed8"
WHITE = "#e2e8f0"
MUTED = "#64748b"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  NEEDLE GIF  —  recall 78 % → 82 % ✅
# ─────────────────────────────────────────────────────────────────────────────

STAGES = [
    {"label": "Best single\n(OCSVM)", "value": 0.78, "color": CAUTION},
    {"label": "Score avg\n(IF+AE+OCSVM)", "value": 0.80, "color": INFO},
    {"label": "Weighted\n(AUC weights)", "value": 0.81, "color": INFO},
    {"label": "Stacking LR\n(meta-learner)", "value": 0.82, "color": SUCCESS},
]
TARGET = 0.80
METRIC = "Recall @ 0.5% FPR"
TITLE = "Ch.5 — Ensemble: 78% → 82% ✅"

FRAMES_PER_STAGE = 18
PAUSE_FRAMES = 6


def _draw_gauge(ax: plt.Axes, value: float, target: float, color: str, label: str) -> None:
    """Draw a semicircular gauge with needle at `value`."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)

    # background arc (0..100%)
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), lw=14, color="#2d2d4e", solid_capstyle="round", zorder=1)

    # coloured fill arc up to value
    frac = np.clip(value, 0, 1)
    theta_fill = np.linspace(np.pi, np.pi * (1 - frac), 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), lw=14, color=color, solid_capstyle="round", zorder=2)

    # target marker
    target_angle = np.pi * (1 - target)
    ax.plot(np.cos(target_angle), np.sin(target_angle), "o", ms=10, color="#f59e0b", zorder=5)
    ax.text(
        np.cos(target_angle) * 1.18,
        np.sin(target_angle) * 1.18,
        f"target\n{target:.0%}",
        ha="center", va="center", fontsize=7, color="#f59e0b",
    )

    # needle
    needle_angle = np.pi * (1 - frac)
    ax.annotate(
        "", xy=(np.cos(needle_angle) * 0.82, np.sin(needle_angle) * 0.82),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=2, mutation_scale=16),
        zorder=6,
    )
    ax.plot(0, 0, "o", ms=8, color=WHITE, zorder=7)

    # centre text
    check = " ✅" if value >= target else ""
    ax.text(0, -0.15, f"{value:.0%}{check}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0, 0.38, label, ha="center", va="center", fontsize=9, color=WHITE, linespacing=1.4)


def build_needle_gif() -> Path:
    out = IMG_DIR / "ch05-ensemble-anomaly-needle.gif"

    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor=BG)
    ax.set_facecolor(BG)
    fig.suptitle(TITLE, color=WHITE, fontsize=11, y=0.98)
    gauge_ax = fig.add_axes([0.15, 0.05, 0.7, 0.85], facecolor=BG)

    frames: list[dict] = []
    for i, stage in enumerate(STAGES):
        prev_value = STAGES[i - 1]["value"] if i > 0 else STAGES[0]["value"]
        cur_value = stage["value"]
        for f in range(FRAMES_PER_STAGE):
            t = f / max(FRAMES_PER_STAGE - 1, 1)
            val = prev_value + t * (cur_value - prev_value)
            frames.append({"value": val, "color": stage["color"], "label": stage["label"]})
        for _ in range(PAUSE_FRAMES):
            frames.append({"value": cur_value, "color": stage["color"], "label": stage["label"]})

    def update(frame: dict) -> list:
        gauge_ax.cla()
        gauge_ax.set_facecolor(BG)
        _draw_gauge(gauge_ax, frame["value"], TARGET, frame["color"], frame["label"])
        # subtitle
        gauge_ax.text(
            0, -0.27, METRIC, ha="center", va="center", fontsize=8, color=MUTED,
        )
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=60, blit=False, repeat=True,
    )
    ani.save(str(out), writer="pillow", fps=16)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROGRESS CHECK PNG  —  FraudShield constraint status
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINTS = [
    ("#1 DETECTION",     "82% recall @ 0.5% FPR",          True,  SUCCESS),
    ("#2 PRECISION",     "FPR ≤ 0.5%",                      True,  SUCCESS),
    ("#3 REAL-TIME",     "~60ms (3 detectors)",             None,  CAUTION),  # partial
    ("#4 ADAPTABILITY",  "Static model — no drift detect",  False, DANGER),
    ("#5 EXPLAINABILITY","Ensemble scores ≠ explanations",  False, DANGER),
]

DETECTOR_PROGRESS = [
    ("Z-score (Ch.1)",        0.45, DANGER),
    ("Isolation Forest (Ch.2)", 0.65, CAUTION),
    ("Autoencoder (Ch.3)",     0.75, CAUTION),
    ("One-Class SVM (Ch.4)",   0.78, CAUTION),
    ("Ensemble (Ch.5)",        0.82, SUCCESS),
]


def build_progress_check() -> Path:
    out = IMG_DIR / "ch05-ensemble-anomaly-progress-check.png"

    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=(13, 5.5), facecolor=BG,
        gridspec_kw={"width_ratios": [1, 1.3]},
    )
    fig.suptitle(
        "FraudShield — Ch.5 Progress Check", color=WHITE, fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Left: recall progression bar chart ─────────────────────────────────
    ax_bar.set_facecolor(BG)
    labels = [d[0] for d in DETECTOR_PROGRESS]
    values = [d[1] for d in DETECTOR_PROGRESS]
    colors = [d[2] for d in DETECTOR_PROGRESS]
    y_pos = np.arange(len(labels))

    bars = ax_bar.barh(y_pos, values, color=colors, height=0.55, zorder=3)
    ax_bar.axvline(0.80, color="#f59e0b", lw=2, ls="--", label="80% target", zorder=4)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_bar.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color=WHITE, fontsize=9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color=WHITE, fontsize=9)
    ax_bar.set_xlabel("Recall @ 0.5% FPR", color=WHITE, fontsize=10)
    ax_bar.set_title("Recall Progression", color=WHITE, fontsize=11, pad=8)
    ax_bar.tick_params(axis="both", colors=WHITE)
    ax_bar.spines[:].set_color(MUTED)
    ax_bar.set_facecolor(BG)
    ax_bar.grid(axis="x", color="#2d2d4e", zorder=1)

    # value labels
    for bar, val in zip(bars, values):
        ax_bar.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}",
            va="center", ha="left", fontsize=9, color=WHITE, fontweight="bold",
        )

    ax_bar.legend(loc="lower right", facecolor="#2d2d4e", edgecolor=MUTED,
                  labelcolor=WHITE, fontsize=8)

    # ── Right: constraint status table ─────────────────────────────────────
    ax_table.set_facecolor(BG)
    ax_table.axis("off")
    ax_table.set_title("FraudShield Constraint Status", color=WHITE, fontsize=11, pad=8)

    row_h = 0.16
    col_x = [0.01, 0.26, 0.71, 0.91]
    headers = ["Constraint", "Status", "Met?"]

    # header row
    for j, (hdr, x) in enumerate(zip(headers, col_x[:-1])):
        ax_table.text(x, 0.95, hdr, color=WHITE, fontsize=9,
                      fontweight="bold", transform=ax_table.transAxes, va="top")

    for i, (name, note, met, color) in enumerate(CONSTRAINTS):
        y = 0.82 - i * row_h
        bg_color = "#1a2a1a" if met else ("#2a1a1a" if met is False else "#1a1a2a")
        rect = mpatches.FancyBboxPatch(
            (0.0, y - 0.025), 1.0, row_h - 0.01,
            boxstyle="round,pad=0.005",
            facecolor=bg_color, edgecolor=color, lw=1.0,
            transform=ax_table.transAxes, zorder=2,
        )
        ax_table.add_patch(rect)
        ax_table.text(col_x[0], y, name, color=color, fontsize=8.5,
                      fontweight="bold", transform=ax_table.transAxes, va="center")
        ax_table.text(col_x[1], y, note, color=WHITE, fontsize=7.5,
                      transform=ax_table.transAxes, va="center")
        icon = "✅" if met is True else ("⚡" if met is None else "❌")
        ax_table.text(col_x[2], y, icon, fontsize=12,
                      transform=ax_table.transAxes, va="center", ha="center")

    # footnote
    ax_table.text(
        0.0, 0.02,
        "Constraints #4 and #5 addressed in Ch.6 — Production",
        color=MUTED, fontsize=8, transform=ax_table.transAxes, va="bottom",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_needle_gif()
    build_progress_check()
    print("All assets generated.")
