from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "proposed-whitepaper.md"
PDF_PATH = ROOT / "proposed-whitepaper.pdf"
FIG_DIR = ROOT / "figures"


def _read_markdown() -> list[str]:
    text = MD_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Convert display-math fences into plain text lines for robust PDF rendering.
    normalized: list[str] = []
    in_math_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == "$$":
            in_math_block = not in_math_block
            continue
        if in_math_block:
            normalized.append(f"[Equation] {line}")
        else:
            normalized.append(line)

    return normalized


def _section_pages(lines: list[str], chars_per_line: int = 106, lines_per_page: int = 43) -> list[list[str]]:
    wrapped: list[str] = []
    for line in lines:
        if line.startswith("#"):
            wrapped.append("")
            wrapped.append(line)
            wrapped.append("")
            continue
        if not line.strip():
            wrapped.append("")
            continue

        # Preserve bullets and equations in readable wrapped form.
        indent = ""
        if line.lstrip().startswith(("- ", "1. ", "2. ", "3. ", "4. ", "5. ")):
            indent = "  "

        segments = wrap(line, width=chars_per_line, break_long_words=False, break_on_hyphens=False)
        if not segments:
            wrapped.append("")
        else:
            wrapped.extend([segments[0]] + [indent + s for s in segments[1:]])

    pages: list[list[str]] = []
    for i in range(0, len(wrapped), lines_per_page):
        pages.append(wrapped[i : i + lines_per_page])
    return pages


def _render_text_page(pdf: PdfPages, lines: list[str], page_num: int) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.96
    for line in lines:
        if line.startswith("# "):
            ax.text(0.08, y, line[2:], fontsize=15, fontweight="bold", va="top", ha="left")
            y -= 0.035
            continue
        if line.startswith("## "):
            ax.text(0.08, y, line[3:], fontsize=12.5, fontweight="bold", va="top", ha="left")
            y -= 0.03
            continue
        if line.startswith("### "):
            ax.text(0.08, y, line[4:], fontsize=11.5, fontweight="bold", va="top", ha="left")
            y -= 0.026
            continue

        ax.text(0.08, y, line, fontsize=9.6, va="top", ha="left")
        y -= 0.021

    ax.text(0.92, 0.03, f"{page_num}", fontsize=9, ha="right", va="bottom", color="#334155")
    pdf.savefig(fig)
    plt.close(fig)


def _render_figure_page(pdf: PdfPages, image_path: Path, title: str, caption: str, page_num: int) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.08, 0.96, title, fontsize=13, fontweight="bold", va="top", ha="left")

    img = mpimg.imread(image_path)
    img_ax = fig.add_axes([0.08, 0.22, 0.84, 0.66])
    img_ax.imshow(img)
    img_ax.axis("off")

    ax.text(0.08, 0.16, caption, fontsize=9.5, va="top", ha="left")
    ax.text(0.92, 0.03, f"{page_num}", fontsize=9, ha="right", va="bottom", color="#334155")

    pdf.savefig(fig)
    plt.close(fig)


def build_pdf() -> None:
    lines = _read_markdown()
    text_pages = _section_pages(lines)

    figures = [
        (
            FIG_DIR / "figure1_system_overview.png",
            "Figure 1. System Overview",
            "Proposed tri-stage context decomposition architecture with control and evaluation plane.",
        ),
        (
            FIG_DIR / "figure2_token_scaling_hypothesis.png",
            "Figure 2. Token Scaling Hypothesis",
            "Hypothetical scaling behavior comparing monolithic prompting vs staged decomposition.",
        ),
        (
            FIG_DIR / "figure3_modality_transfer_map.png",
            "Figure 3. Modality Transfer Map",
            "Transfer pattern from text chat to long audio, long video, and multimodal sessions.",
        ),
    ]

    page_num = 1
    with PdfPages(PDF_PATH) as pdf:
        for page_lines in text_pages:
            _render_text_page(pdf, page_lines, page_num)
            page_num += 1

        for fig_path, title, caption in figures:
            if fig_path.exists():
                _render_figure_page(pdf, fig_path, title, caption, page_num)
                page_num += 1

    print(f"PDF generated: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
