"""
gen_ch05.py
───────────
Generates two visual assets for Ch.5 — Policy Gradients:

  img/ch05-policy-gradients-needle.gif
      Animated needle: DQN ~150 → REINFORCE ~170 → PPO ~190
      (CartPole average episode return progression)

  img/ch05-policy-gradients-progress-check.png
      Static bar chart: AgentAI CartPole progress Ch.3 → Ch.5

Usage:
    python gen_scripts/gen_ch05.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from shared.animation_renderer import render_metric_story

# ── Needle animation stages ─────────────────────────────────────────────────
# Maps to chapter narrative: DQN (Ch.4) → REINFORCE → PPO
STAGES = [
    {
        "label": "DQN (Ch.4)",
        "value": 150.0,
        "m": 0.52,
        "b": 0.22,
        "bend": -0.01,
        "margin": 0.38,
        "caption": "DQN (~150/200): argmax over Q-values — discrete only, deterministic policy.",
    },
    {
        "label": "REINFORCE",
        "value": 170.0,
        "m": 0.63,
        "b": 0.16,
        "bend": -0.04,
        "margin": 0.50,
        "caption": "REINFORCE (~170/200): stochastic policy, built-in exploration — but high variance.",
    },
    {
        "label": "PPO",
        "value": 190.0,
        "m": 0.74,
        "b": 0.10,
        "bend": -0.07,
        "margin": 0.62,
        "caption": "PPO (~190/200): clipped objective prevents catastrophic updates. ≥195 target in Ch.6.",
    },
]


def _make_progress_check(out_dir: Path) -> None:
    """Bar chart: AgentAI CartPole scores per chapter (Ch.3 → Ch.5)."""
    rng = np.random.default_rng(42)

    algorithms = [
        "Random\npolicy",
        "Q-learning\n(Ch.3)",
        "DQN\n(Ch.4)",
        "REINFORCE\n(Ch.5)",
        "PPO\n(Ch.5)",
    ]
    scores = [22, 60, 150, 170, 190]
    colors = [
        "#374151",  # random — grey
        "#1e3a8a",  # q-learning — dark blue
        "#1d4ed8",  # dqn — blue
        "#b45309",  # reinforce — amber
        "#15803d",  # ppo — green
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(
        range(len(algorithms)),
        scores,
        color=colors,
        edgecolor="#e5e7eb",
        linewidth=0.8,
        alpha=0.9,
    )

    # Target line
    ax.axhline(
        195,
        color="#b91c1c",
        linewidth=2.5,
        linestyle="--",
        label="Target ≥195/200",
        zorder=5,
    )

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{score}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#f9fafb",
        )

    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=10, color="#e5e7eb")
    ax.set_ylabel("Avg Episode Return (/ 200)", fontsize=11, color="#e5e7eb")
    ax.set_ylim(0, 215)
    ax.set_title(
        "AgentAI — CartPole Progress (Ch.3 → Ch.5)",
        fontsize=13,
        fontweight="bold",
        color="#f9fafb",
        pad=14,
    )
    ax.tick_params(colors="#e5e7eb")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4b5563")
    ax.yaxis.grid(True, alpha=0.2, color="#6b7280")
    ax.legend(
        fontsize=10,
        facecolor="#1f2937",
        edgecolor="#4b5563",
        labelcolor="#f9fafb",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ch05-policy-gradients-progress-check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    img_dir = Path(__file__).parents[1] / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ch05-policy-gradients-needle.gif ...")
    render_metric_story(
        img_dir,
        "ch05-policy-gradients-needle",
        "Ch.5 — Policy Gradients: CartPole episode return",
        "Episode Return",
        STAGES,
        better="higher",
        style="classification",
    )

    print("Generating ch05-policy-gradients-progress-check.png ...")
    _make_progress_check(img_dir)

    print("Done.")
