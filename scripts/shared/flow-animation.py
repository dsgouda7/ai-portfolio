"""Shared flow animation renderer for MultimodalAI chapter diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


DEFAULT_FPS = 6
DEFAULT_FRAMES_PER_STAGE = 12
DEFAULT_SEED = 17
BOX_COLORS = ["#0f766e", "#0ea5e9", "#ea580c", "#16a34a", "#8b5cf6", "#ef4444", "#a16207"]


@dataclass(frozen=True)
class FlowStage:
    title: str
    subtitle: str


def _draw_background(ax: plt.Axes, rng: np.random.Generator) -> None:
    gradient = np.linspace(0.93, 1.0, 256)
    bg = np.outer(np.ones(256), gradient)
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", cmap="YlGnBu", alpha=0.28, zorder=0)
    points = rng.uniform(0.02, 0.98, size=(35, 2))
    ax.scatter(points[:, 0], points[:, 1], s=10, c="#0c4a6e", alpha=0.1, zorder=0)


def render_flow_animation(
    chapter_title: str,
    flow_caption: str,
    stages: list[FlowStage],
    output_stem: str,
    chapter_dir: Path,
    fps: int = DEFAULT_FPS,
    frames_per_stage: int = DEFAULT_FRAMES_PER_STAGE,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    if len(stages) < 3:
        raise ValueError("Flow animation needs at least three stages.")

    chapter_dir = Path(chapter_dir)
    out_dir = chapter_dir / "img"
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_path = out_dir / f"{output_stem}.gif"
    png_path = out_dir / f"{output_stem}.png"

    stage_count = len(stages)
    total_frames = stage_count * frames_per_stage
    x_positions = np.linspace(0.11, 0.89, stage_count)
    y_stage = 0.58

    fig, ax = plt.subplots(figsize=(12.6, 5.8), dpi=140)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rng = np.random.default_rng(seed)

    def draw_scene(frame_idx: int) -> None:
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        _draw_background(ax, rng)

        active_stage = min(frame_idx // frames_per_stage, stage_count - 1)
        local_frame = frame_idx % frames_per_stage
        fade_span = max(2, frames_per_stage // 4)
        travel_start = fade_span
        travel_end = max(travel_start + 1, frames_per_stage - fade_span)

        ax.text(0.5, 0.93, chapter_title, ha="center", va="center", fontsize=17, fontweight="bold", color="#082f49")
        ax.text(0.5, 0.86, flow_caption, ha="center", va="center", fontsize=11.5, color="#1f2937")

        for i, stage in enumerate(stages):
            left = x_positions[i] - 0.095
            box_color = BOX_COLORS[i % len(BOX_COLORS)]
            alpha = 0.96 if i == active_stage else 0.48
            edge_width = 2.8 if i == active_stage else 1.4

            rect = plt.Rectangle(
                (left, y_stage - 0.11),
                0.19,
                0.22,
                facecolor=box_color,
                edgecolor="#0f172a",
                linewidth=edge_width,
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(x_positions[i], y_stage + 0.02, stage.title, ha="center", va="center", fontsize=11, color="white", fontweight="bold", zorder=3)
            ax.text(x_positions[i], y_stage - 0.055, stage.subtitle, ha="center", va="center", fontsize=8.6, color="white", zorder=3)

            if i < stage_count - 1:
                ax.annotate(
                    "",
                    xy=(x_positions[i + 1] - 0.105, y_stage),
                    xytext=(x_positions[i] + 0.105, y_stage),
                    arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0),
                    zorder=1,
                )

        if active_stage < stage_count - 1:
            if local_frame <= travel_start:
                progress = 0.0
            elif local_frame >= travel_end:
                progress = 1.0
            else:
                progress = (local_frame - travel_start) / (travel_end - travel_start)
            x0 = x_positions[active_stage] + 0.11
            x1 = x_positions[active_stage + 1] - 0.11
            packet_x = x0 + (x1 - x0) * progress
            packet_y = y_stage + 0.035 * np.sin(progress * np.pi)
            ax.scatter([packet_x], [packet_y], s=130, c="#facc15", edgecolors="#854d0e", linewidths=1.2, zorder=4)

        if local_frame < fade_span:
            caption_alpha = (local_frame + 1) / fade_span
        elif local_frame > (frames_per_stage - fade_span):
            caption_alpha = max(0.0, (frames_per_stage - local_frame) / fade_span)
        else:
            caption_alpha = 1.0

        ax.add_patch(
            plt.Rectangle(
                (0.14, 0.085),
                0.72,
                0.1,
                facecolor="#e2e8f0",
                edgecolor="#94a3b8",
                linewidth=1.1,
                alpha=0.85,
                zorder=1,
            )
        )
        ax.text(
            0.5,
            0.135,
            f"Step {active_stage + 1}/{stage_count}: {stages[active_stage].title}",
            ha="center",
            va="center",
            fontsize=11,
            color="#0f172a",
            fontweight="bold",
            alpha=caption_alpha,
            zorder=2,
        )

    anim = animation.FuncAnimation(fig, draw_scene, frames=total_frames, interval=1000 // fps)
    writer = animation.PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)

    draw_scene(total_frames - 1)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return gif_path, png_path
