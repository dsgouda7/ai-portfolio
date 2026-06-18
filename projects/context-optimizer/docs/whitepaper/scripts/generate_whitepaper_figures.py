from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            linewidth=1.8,
            edgecolor="#1f2937",
            facecolor=color,
        )
        ax.add_patch(rect)
        title_wrapped = "\n".join(textwrap.wrap(title, width=max(14, int(w * 7))))
        body_wrapped = "\n".join(
            textwrap.fill(line, width=max(20, int(w * 8))) for line in body.split("\n")
        )
        ax.text(x + 0.25, y + h - 0.45, title_wrapped, fontsize=11, fontweight="bold", color="#0f172a", va="top")
        ax.text(x + 0.25, y + h - 1.15, body_wrapped, fontsize=10, color="#111827", va="top")

    # Pipeline bands
    box(
        0.4,
        5.5,
        3.0,
        1.8,
        "Input Streams",
        "Chat turns\nAudio transcripts\nVideo scene captions\nMultimodal metadata",
        "#dbeafe",
    )
    box(
        3.9,
        5.5,
        3.1,
        1.8,
        "Stage A: Compression",
        "Small LLM / RNN / lightweight encoder\nOutputs structured latent brief",
        "#dcfce7",
    )
    box(
        7.5,
        5.5,
        3.0,
        1.8,
        "Stage B: Retrieval Router",
        "Memory + retrieval + tool planner\nBudgeted context assembly",
        "#fef3c7",
    )
    box(
        11.0,
        5.5,
        2.7,
        1.8,
        "Stage C: Reasoning",
        "Final answer / action plan\nBounded token window",
        "#fee2e2",
    )

    # Bottom control plane
    box(
        2.2,
        2.2,
        9.4,
        2.2,
        "Control + Evaluation Plane",
        "Confidence gating | fallback policy | retrieval diagnostics | safety filters | cost monitor",
        "#ede9fe",
    )

    # Arrows main path
    arrow_style = dict(arrowstyle="-|>", linewidth=2.1, color="#1f2937")
    ax.annotate("", xy=(3.8, 6.4), xytext=(3.4, 6.4), arrowprops=arrow_style)
    ax.annotate("", xy=(7.4, 6.4), xytext=(7.0, 6.4), arrowprops=arrow_style)
    ax.annotate("", xy=(10.9, 6.4), xytext=(10.5, 6.4), arrowprops=arrow_style)

    # Links to control plane
    for x in [5.3, 9.0, 12.2]:
        ax.annotate(
            "",
            xy=(x, 4.4),
            xytext=(x, 5.5),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="#4b5563", linestyle="--"),
        )

    ax.text(
        0.4,
        0.8,
        "Figure 1. Proposed tri-stage compression-retrieval-reasoning architecture with modality-agnostic extension path.",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.4,
        0.35,
        "Note: This figure encodes a proposed design pattern and should not be interpreted as validated production performance.",
        fontsize=9,
        color="#334155",
    )

    _save(fig, "figure1_system_overview.png")


def figure_token_scaling_hypothesis() -> None:
    # Values are in thousands of tokens and mirror the documented scale behavior.
    x = np.array([1, 10, 50, 100], dtype=float)
    naive = np.array([44.1, 438.5, 2192.4, 4385.0])
    proposed = np.array([1.7, 1.74, 1.74, 1.74])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x, naive, marker="o", linewidth=2.8, color="#dc2626", label="Naive full-context pipeline")
    ax.plot(x, proposed, marker="o", linewidth=2.8, color="#0f766e", label="Proposed staged architecture")

    ax.fill_between(x, proposed, naive, color="#fde68a", alpha=0.45)

    ax.set_title("Token Budget Hypothesis vs Corpus Growth", fontsize=16, fontweight="bold")
    ax.set_xlabel("Relative corpus scale (x)", fontsize=12)
    ax.set_ylabel("Prompt tokens to reasoning model (thousands)", fontsize=12)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)

    ax.text(
        1.2,
        90,
        "Hypothesis only: staged pipeline keeps reasoning input near-constant\nvia intermediate compression and selective retrieval.",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecfeff", edgecolor="#155e75"),
    )

    _save(fig, "figure2_token_scaling_hypothesis.png")


def figure_modality_transfer_map() -> None:
    modalities = ["Text Chat", "Long Audio", "Long Video", "Multimodal Session"]
    compressors = [
        "Small LLM summarizer",
        "ASR + recurrent semantic compressor",
        "Scene graph + temporal RNN",
        "Cross-modal projector + low-cost LLM",
    ]
    retrievers = [
        "Memory + doc retrieval",
        "Transcript segment retrieval",
        "Frame/scene + transcript retrieval",
        "Unified event and embedding retrieval",
    ]
    outputs = [
        "Answer / action",
        "Answer + cited segments",
        "Answer + cited scenes",
        "Agentic plan + trace",
    ]

    n = len(modalities)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(-0.8, n - 0.2)
    ax.invert_yaxis()
    ax.axis("off")

    ax.text(0.5, -0.45, "Input Modality", fontsize=12, fontweight="bold")
    ax.text(3.1, -0.45, "Low-Cost Compression", fontsize=12, fontweight="bold")
    ax.text(6.4, -0.45, "Targeted Retrieval", fontsize=12, fontweight="bold")
    ax.text(9.2, -0.45, "Final Output", fontsize=12, fontweight="bold")

    for i in range(n):
        ax.add_patch(patches.FancyBboxPatch((0.2, i - 0.28), 2.6, 0.56, boxstyle="round,pad=0.03", facecolor="#e2e8f0", edgecolor="#334155"))
        ax.add_patch(patches.FancyBboxPatch((3.2, i - 0.28), 3.1, 0.56, boxstyle="round,pad=0.03", facecolor="#dcfce7", edgecolor="#166534"))
        ax.add_patch(patches.FancyBboxPatch((6.7, i - 0.28), 2.4, 0.56, boxstyle="round,pad=0.03", facecolor="#fef3c7", edgecolor="#92400e"))
        ax.add_patch(patches.FancyBboxPatch((9.4, i - 0.28), 1.8, 0.56, boxstyle="round,pad=0.03", facecolor="#fee2e2", edgecolor="#991b1b"))

        ax.text(0.35, i, modalities[i], fontsize=10, va="center")
        ax.text(3.3, i, compressors[i], fontsize=9.5, va="center")
        ax.text(6.8, i, retrievers[i], fontsize=9.5, va="center")
        ax.text(9.45, i, outputs[i], fontsize=9.2, va="center")

        ax.annotate("", xy=(3.15, i), xytext=(2.8, i), arrowprops=dict(arrowstyle="-|>", color="#1f2937", linewidth=1.5))
        ax.annotate("", xy=(6.65, i), xytext=(6.3, i), arrowprops=dict(arrowstyle="-|>", color="#1f2937", linewidth=1.5))
        ax.annotate("", xy=(9.35, i), xytext=(9.1, i), arrowprops=dict(arrowstyle="-|>", color="#1f2937", linewidth=1.5))

    ax.text(
        0.2,
        n - 0.02,
        "Figure 3. Modality transfer map: same architectural decomposition reused across text, audio, video, and multimodal chat.",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
    )

    _save(fig, "figure3_modality_transfer_map.png")


def figure_mcp_pull_architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, title, lines, color, edge="#1f2937"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            linewidth=1.8, edgecolor=edge, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + 0.2, y + h - 0.45, title, fontsize=11, fontweight="bold", color="#0f172a", va="top")
        for i, line in enumerate(lines):
            ax.text(x + 0.22, y + h - 0.85 - i * 0.38, line, fontsize=9.5, color="#1e293b", va="top")

    def arrow(x1, y1, x2, y2, label="", style="-|>", color="#1f2937", lw=2.0, ls="solid"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, linewidth=lw, color=color,
                                   linestyle=ls, connectionstyle="arc3,rad=0.0"))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.06, my, label, fontsize=8.5, color="#475569")

    # ── Structured Shell ──────────────────────────────────────────────────────
    box(0.4, 7.2, 4.2, 2.5, "Structured Shell  (~1.7k tokens, fixed)",
        ["persona | instructions | safety constraints",
         "tools: retrieve_context(topic, depth, format)",
         "       write_to_cache(content)",
         "       get_session_memory(session_id)",
         "task:  <compressed anchor>  ← last line"],
        "#dbeafe", edge="#2563eb")

    # ── Reasoning Model ───────────────────────────────────────────────────────
    box(5.4, 7.2, 3.2, 2.5, "Reasoning Model",
        ["Reads structured shell",
         "Issues typed MCP tool calls",
         "Integrates retrieved context",
         "Produces final answer"],
        "#dcfce7", edge="#16a34a")

    # ── Context Budget Manager ────────────────────────────────────────────────
    box(9.4, 7.2, 3.2, 2.5, "Context Budget Manager",
        ["Tracks cumulative tool-response",
         "tokens this turn",
         "Soft-stop at configurable ceiling",
         "Signals partial-answer flag"],
        "#fef3c7", edge="#d97706")

    # ── MCP Server ────────────────────────────────────────────────────────────
    box(5.4, 3.8, 3.2, 2.8, "MCP Server",
        ["Typed tool contracts",
         "Routes to: vector DB |",
         "  session cache | live tools",
         "Returns pre-compressed chunks",
         "Enforces per-call token cap"],
        "#ede9fe", edge="#7c3aed")

    # ── Vector DB (pre-compressed) ────────────────────────────────────────────
    box(0.4, 1.0, 3.2, 2.5, "Vector DB",
        ["Embeddings of compressed text",
         "Indexed at write time",
         "Cheap LLM summarizer on ingest",
         "Keyword + semantic retrieval"],
        "#e0f2fe", edge="#0284c7")

    # ── Session Semantic Cache ────────────────────────────────────────────────
    box(5.4, 1.0, 3.2, 2.5, "Session Semantic Cache",
        ["Per-session, persisted",
         "Compressed before storage",
         "Loaded on session restore",
         "⚠ Invalidation: open problem"],
        "#fce7f3", edge="#be185d")

    # ── Feedback / Re-compression ─────────────────────────────────────────────
    box(9.4, 3.8, 3.2, 2.8, "Re-compression Gate",
        ["Monitors accumulated tokens",
         "If > threshold: route through",
         "  compression pipeline",
         "Produces schema-consistent",
         "  output (not a truncation)"],
        "#fef9c3", edge="#ca8a04")

    # ── Pass-through path ─────────────────────────────────────────────────────
    box(9.4, 1.0, 3.2, 2.2, "Pass-through Policy",
        ["If anchor ≤ budget:",
         "  skip MCP, send direct",
         "MCP is an optimisation,",
         "  not a mandatory stage"],
        "#f0fdf4", edge="#15803d")

    # ── Arrows ────────────────────────────────────────────────────────────────
    arrow(4.6, 8.45, 5.4, 8.45)
    arrow(8.6, 8.45, 9.4, 8.45)
    arrow(7.0, 7.2, 7.0, 6.6, "tool call")
    arrow(7.0, 3.8, 7.0, 3.0, "query")
    arrow(3.6, 3.5, 5.4, 3.0, "retrieve")
    arrow(8.6, 3.0, 9.4, 4.2, "> threshold?", ls="dashed", color="#b45309")
    arrow(9.4, 8.45, 8.6, 8.45, color="#b45309", ls="dashed")
    # re-compression feeds back to vector DB
    ax.annotate("", xy=(1.95, 3.5), xytext=(9.4, 4.8),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.6, color="#b45309",
                                linestyle="dashed", connectionstyle="arc3,rad=-0.35"))
    ax.text(5.0, 3.55, "re-compress → re-index", fontsize=8.5, color="#92400e")

    ax.text(0.4, 0.55,
            "Figure 4. MCP pull architecture: reasoning model as retrieval orchestrator with pre-compressed store, "
            "session cache, feedback-driven re-compression, and pass-through policy.",
            fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(0.4, 0.22,
            "Note: cache invalidation strategy is an open design problem and is not resolved in this proposal.",
            fontsize=9, color="#475569")

    _save(fig, "figure4_mcp_pull_architecture.png")


def main() -> None:
    figure_system_overview()
    figure_token_scaling_hypothesis()
    figure_modality_transfer_map()
    figure_mcp_pull_architecture()
    print(f"Generated 4 figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
