"""
gen_ch06_modern_rl.py
─────────────────────────────────────────────────────────────────────────────
Generates two visual assets for Ch.6 — Modern RL: SAC, Rainbow DQN & HER

  img/ch06-modern-rl-needle.gif
      Animated needle: PPO ~190 → SAC ~195 → Rainbow ~198
      (CartPole average episode return — AgentAI target ≥195 ✅ achieved)

  img/ch06-modern-rl-progress-check.png
      Bar chart showing AgentAI CartPole scores across the complete RL track
      (Ch.3 Q-learning through Ch.6 Rainbow)

Usage:
    python gen_scripts/gen_ch06_modern_rl.py

Outputs are written to ../img/ (relative to this script).
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

# ── Needle animation ─────────────────────────────────────────────────────────
# Three stages map onto the chapter's narrative arc:
#   Stage 1: PPO (Ch.5 baseline)  ~190/200 — close but not ≥195
#   Stage 2: SAC                  ~195/200 — entropy bonus ✅ target crossed
#   Stage 3: Rainbow DQN          ~198/200 — 6 improvements + prioritized replay

STAGES = [
    {
        "label": "PPO (Ch.5)",
        "value": 190.0,
        "m": 0.70,
        "b": 0.12,
        "bend": -0.06,
        "margin": 0.58,
        "caption": (
            "PPO (~190/200): clipped objective prevents catastrophic updates "
            "— close but not ≥195. Policy overconfident near balance point."
        ),
    },
    {
        "label": "SAC",
        "value": 195.0,
        "m": 0.78,
        "b": 0.10,
        "bend": -0.08,
        "margin": 0.66,
        "caption": (
            "SAC (~195/200): entropy bonus forces exploration near balance point. "
            "AgentAI target ≥195/200 ✅ achieved. Off-policy replay reuses all experience."
        ),
    },
    {
        "label": "Rainbow DQN",
        "value": 198.0,
        "m": 0.84,
        "b": 0.08,
        "bend": -0.10,
        "margin": 0.72,
        "caption": (
            "Rainbow (~198/200): 6 DQN improvements combined — prioritized replay "
            "masters rare failure states. 3× DQN on Atari benchmark."
        ),
    },
]


def _make_progress_check(out_dir: Path) -> None:
    """Bar chart: AgentAI CartPole scores across the complete RL track (Ch.3–Ch.6)."""
    rng = np.random.default_rng(42)  # noqa: F841 — seed kept for reproducibility

    bar_items = [
        ("Random\npolicy", 22, "#374151"),        # grey
        ("Q-learning\n(Ch.3)", 60, "#1e3a8a"),    # dark blue
        ("DQN\n(Ch.4)", 150, "#1d4ed8"),           # blue
        ("REINFORCE\n(Ch.5)", 170, "#b45309"),      # amber
        ("PPO\n(Ch.5)", 190, "#b45309"),            # amber
        ("SAC\n(Ch.6)", 195, "#15803d"),            # green
        ("Rainbow\n(Ch.6)", 198, "#22c55e"),        # bright green
    ]

    labels = [item[0] for item in bar_items]
    scores = [item[1] for item in bar_items]
    colors = [item[2] for item in bar_items]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    bars = ax.bar(
        range(len(bar_items)),
        scores,
        color=colors,
        edgecolor="#e5e7eb",
        linewidth=0.8,
        alpha=0.9,
    )

    # AgentAI target line
    ax.axhline(
        195,
        color="#b91c1c",
        linewidth=2.5,
        linestyle="--",
        label="AgentAI Target ≥195/200",
        zorder=5,
    )

    # Score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{score}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#f9fafb",
        )

    ax.set_xticks(range(len(bar_items)))
    ax.set_xticklabels(labels, fontsize=9, color="#e5e7eb")
    ax.set_ylabel("Avg Episode Return (/ 200)", fontsize=11, color="#e5e7eb")
    ax.set_ylim(0, 215)
    ax.set_title(
        "AgentAI — CartPole Progress: Complete RL Track (Ch.3 → Ch.6) ✅",
        fontsize=13,
        fontweight="bold",
        color="#f9fafb",
        pad=14,
    )
    ax.tick_params(colors="#e5e7eb")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4b5563")
    ax.yaxis.grid(True, alpha=0.2, color="#6b7280")
    ax.legend(fontsize=10, facecolor="#1f2937", edgecolor="#4b5563", labelcolor="#f9fafb")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ch06-modern-rl-progress-check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    img_dir = Path(__file__).parents[1] / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ch06-modern-rl-needle.gif ...")
    render_metric_story(
        img_dir,
        "ch06-modern-rl-needle",
        "AgentAI CartPole: PPO 190 → SAC 195 → Rainbow 198",
        "CartPole Avg Episode Return (/ 200)",
        STAGES,
        better="higher",
        style="classification",
    )

    print("Generating ch06-modern-rl-progress-check.png ...")
    _make_progress_check(img_dir)

    print("Done. Assets written to img/")
