"""
gen_ch03.py — Generate visual assets for Ch.3 Q-Learning.

Produces:
  img/ch03-q-learning-needle.gif   Agent capability: random → Q-table → optimal GridWorld
  img/ch03-q-learning-progress-check.png   AgentAI 5-constraint progress dashboard

Run from the chapter root:
    python gen_scripts/gen_ch03.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHAPTER_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Needle GIF  (avg episode return: random → Q-table learning → optimal)
# ---------------------------------------------------------------------------

STAGES = [
    {
        "label": "0 eps (12)",
        "value": 12.0,
        "threshold": 0.82,
        "threshold_note": "Q ≈ all zeros",
        "caption": (
            "With all Q-values at zero, action selection is essentially random. "
            "The agent stumbles around the grid, rarely reaching the goal, "
            "collecting only scattered rewards."
        ),
    },
    {
        "label": "100 eps (40)",
        "value": 40.0,
        "threshold": 0.67,
        "threshold_note": "goal-adjacent cells learned",
        "caption": (
            "Early TD updates reveal which actions near the goal are good. "
            "Q(s14, →) and Q(s11, ↓) have turned positive and the return "
            "climbs sharply as the agent exploits these local signals."
        ),
    },
    {
        "label": "500 eps (66)",
        "value": 66.0,
        "threshold": 0.55,
        "threshold_note": "ε shrinking — more exploitation",
        "caption": (
            "Repeated visits stabilize most state-action values. "
            "The agent reliably avoids the hole and navigates a near-optimal "
            "path from most start states. ε has decayed below 0.2."
        ),
    },
    {
        "label": "2k eps (84)",
        "value": 84.0,
        "threshold": 0.48,
        "threshold_note": "greedy ≈ optimal π*",
        "caption": (
            "The Q-table has converged. The greedy policy matches value "
            "iteration output from Ch.2, reaching the goal in minimum steps "
            "from every start state while avoiding the hole."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. Progress-check PNG  (AgentAI 5-constraint dashboard)
# ---------------------------------------------------------------------------


def _make_progress_check(chapter_dir: Path) -> None:
    """
    Render the AgentAI constraint progress bar for Ch.3 Q-Learning and save to
    img/ch03-q-learning-progress-check.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    rng = np.random.default_rng(42)
    _ = rng  # seed held for reproducibility

    DARK_BG = "#1a1a2e"
    SUCCESS = "#15803d"
    CAUTION = "#b45309"
    DANGER  = "#b91c1c"
    TEXT    = "#e2e8f0"

    # (label, current_pct, target_pct, color, note)
    constraints = [
        ("#1 OPTIMALITY",      100, 100, SUCCESS, "Converges to Q* (Watkins 1992)"),
        ("#2 EFFICIENCY",       70, 100, CAUTION, "Online — no model, but slow for large envs"),
        ("#3 SCALABILITY",       0, 100, DANGER,  "Blocked — Q-table fails for continuous states"),
        ("#4 STABILITY",        65, 100, CAUTION, "Sensitive to α, ε — needs tuning"),
        ("#5 GENERALIZATION",    0, 100, DANGER,  "Blocked — tabular, no cross-state transfer"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    bar_height = 0.55

    for i, (label, current, target, color, note) in enumerate(constraints):
        # Background (full target width)
        ax.barh(i, 100, height=bar_height, color="#2a2a4a", zorder=1)
        # Progress bar
        ax.barh(i, current, height=bar_height, color=color, zorder=2, alpha=0.9)
        # Label on the left
        ax.text(-2, i, label, va="center", ha="right", color=TEXT,
                fontsize=9, fontweight="bold")
        # Note on the right of the bar
        x_note = current + 1.5 if current > 5 else 3
        ax.text(x_note, i, f"{current}% — {note}", va="center",
                ha="left", color=TEXT, fontsize=8, alpha=0.85)

    ax.set_xlim(-38, 145)
    ax.set_ylim(-0.6, len(constraints) - 0.4)
    ax.set_xlabel("% Achieved", color=TEXT, fontsize=10)
    ax.set_title(
        "AgentAI — Ch.3 Q-Learning: Constraint Progress",
        color=TEXT, fontsize=12, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[:].set_color("#3a3a5a")
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 65, 70, 100])
    for tick in ax.get_xticklabels():
        tick.set_color(TEXT)

    # Annotations
    ax.annotate(
        "GridWorld ✅\nQ-table matches V*",
        xy=(100, 0),
        xytext=(108, 1.3),
        arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.4),
        color=SUCCESS, fontsize=8, fontweight="bold",
    )
    ax.annotate(
        "CartPole ❌\nNeed DQN (Ch.4)",
        xy=(0, 2),
        xytext=(18, 3.4),
        arrowprops=dict(arrowstyle="->", color=DANGER, lw=1.4),
        color=DANGER, fontsize=8, fontweight="bold",
    )

    # Legend
    patches = [
        mpatches.Patch(color=SUCCESS, label="✅ Achieved"),
        mpatches.Patch(color=CAUTION, label="⚡ Partial"),
        mpatches.Patch(color=DANGER,  label="❌ Blocked"),
    ]
    ax.legend(
        handles=patches, loc="lower right",
        facecolor=DARK_BG, edgecolor="#3a3a5a",
        labelcolor=TEXT, fontsize=9,
    )

    out_path = chapter_dir / "img" / "ch03-q-learning-progress-check.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"[OK] Progress check PNG -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Needle GIF
    render_metric_story(
        CHAPTER_DIR,
        "ch03-q-learning-needle",
        "Ch.3 — Q-learning lifts avg return from 12 to 84",
        "Avg Episode Return",
        STAGES,
        better="higher",
        style="threshold",
    )

    # 2. Progress-check PNG
    _make_progress_check(CHAPTER_DIR)
