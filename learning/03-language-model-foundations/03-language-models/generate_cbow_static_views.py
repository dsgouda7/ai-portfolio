"""Rebuild the saved core-CBOW profile comparison and window storyboard."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "03A-training-skeleton-and-word-objectives.ipynb"
PROFILE_IMAGE = HERE / "images" / "03a-cbow-word-profiles-before-after.png"
WINDOW_IMAGE = HERE / "images" / "03a-cbow-window-storyboard.png"
BACKGROUND = "#1a1a2e"
FOREGROUND = "#e7edf7"
TEAL = "#58d7d1"
ORANGE = "#ffbb78"


def core_sources() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell["id"] == "ch03a-extensions":
            return sources
        if cell["cell_type"] == "code":
            sources.append("".join(cell["source"]))
    raise ValueError("The notebook needs a boundary after the core CBOW lesson.")


def source_fingerprint(core: list[str] | None = None) -> str:
    sources = (core_sources() if core is None else core) + [
        path.read_text(encoding="utf-8") for path in (
            HERE / "real_text_embeddings.py", HERE / "language_model.py",
            HERE / "language_data.py", HERE / "language_validation.py", Path(__file__),
        )
    ]
    syntax = "\n".join(ast.dump(ast.parse(source)) for source in sources)
    return hashlib.sha256(syntax.encode("utf-8")).hexdigest()


def save(figure, path: Path, metadata: dict[str, str]) -> None:
    figure.savefig(
        path, dpi=140, facecolor=figure.get_facecolor(), bbox_inches="tight", metadata=metadata,
    )
    plt.close(figure)
    print(f"Saved {path.name}")


def profile_capture(namespace: dict[str, object], fingerprint: str) -> None:
    before = namespace["embedding_before"]
    after = namespace["embedding_weights"]
    words = namespace["itos"]
    losses = namespace["losses"]
    seed = namespace["SEED"]
    if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
        raise TypeError("The lesson must record its actual before/after embedding tensors.")
    if before.ndim != 2 or before.shape != after.shape:
        raise ValueError("Before and after must describe the same embedding table.")
    if not isinstance(words, list) or not isinstance(losses, list) or not isinstance(seed, int):
        raise TypeError("The lesson must retain its vocabulary, loss history, and seed.")
    visible_words = ["the", "sleepy", "cat", "sat", "on", "mat", "dog"]
    missing = [word for word in visible_words if word not in words]
    if missing:
        raise ValueError(f"Profile capture words are absent from the training vocabulary: {missing}")
    ids = torch.tensor([words.index(word) for word in visible_words])
    before_rows = torch.nn.functional.normalize(before.detach().cpu()[ids], dim=1)
    after_rows = torch.nn.functional.normalize(after.detach().cpu()[ids], dim=1)
    recorded = torch.stack([before_rows, after_rows]).numpy()
    all_points = recorded.reshape(-1, recorded.shape[-1])
    center = all_points.mean(axis=0)
    _, singular_values, directions = np.linalg.svd(all_points - center, full_matrices=False)
    coordinates = (recorded - center) @ directions[:2].T
    explained = float((singular_values[:2] ** 2).sum() / (singular_values ** 2).sum())
    cat, dog = visible_words.index("cat"), visible_words.index("dog")
    similarity_before = float((before_rows[cat] * before_rows[dog]).sum())
    similarity_after = float((after_rows[cat] * after_rows[dog]).sum())
    dimensions = before.shape[1]
    x_min, x_max = coordinates[..., 0].min(), coordinates[..., 0].max()
    y_min, y_max = coordinates[..., 1].min(), coordinates[..., 1].max()
    x_span, y_span = float(x_max - x_min), float(y_max - y_min)
    limits = ((x_min - 0.15 * x_span, x_max + 0.75 * x_span),
              (y_min - 0.20 * y_span, y_max + 0.20 * y_span))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.3))
    figure.patch.set_facecolor(BACKGROUND)
    for stage, axis in enumerate(axes):
        axis.set_facecolor(BACKGROUND)
        points = coordinates[stage]
        colors = [TEAL if word in {"cat", "dog"} else FOREGROUND for word in visible_words]
        axis.scatter(points[:, 0], points[:, 1], s=65, c=colors, zorder=3)
        axis.plot(points[[cat, dog], 0], points[[cat, dog], 1],
                  color=ORANGE, linestyle="--", linewidth=1.3, alpha=0.8)
        label_y = np.linspace(y_min + 0.90 * y_span, y_min + 0.10 * y_span, len(visible_words))
        for index, y in zip(np.argsort(-points[:, 1]), label_y):
            axis.annotate(
                visible_words[index], xy=points[index], xytext=(x_max + 0.20 * x_span, y),
                fontsize=12, color=colors[index], ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=colors[index], lw=0.8, alpha=0.6),
            )
        axis.set(
            xlim=limits[0], ylim=limits[1],
            title="Before training" if stage == 0 else f"After {len(losses)} CBOW epochs",
            xlabel="Shared view direction 1", ylabel="Shared view direction 2",
        )
        axis.title.set_color(FOREGROUND)
        axis.xaxis.label.set_color(FOREGROUND)
        axis.yaxis.label.set_color(FOREGROUND)
        axis.tick_params(colors=FOREGROUND)
        axis.grid(color="#64748b", alpha=0.2)
        for spine in axis.spines.values():
            spine.set_color("#64748b")
    figure.suptitle("The same learned word profiles, viewed through one fixed camera",
                   color=FOREGROUND, fontsize=17)
    figure.text(
        0.5, 0.075,
        f"All {dimensions} features compared: cat/dog similarity "
        f"{similarity_before:.3f} -> {similarity_after:.3f} (higher means more alike)",
        ha="center", color=ORANGE, fontsize=12,
    )
    figure.text(
        0.5, 0.025,
        f"The 2D sketch retains {explained:.0%} of these recorded profiles' variation. "
        "Labels connect to measured dots; the full-profile check is above.",
        ha="center", color=FOREGROUND, fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.93))
    save(figure, PROFILE_IMAGE, {
        "SourceFingerprint": fingerprint, "Seed": str(seed), "Epochs": str(len(losses)),
        "Dimensions": str(dimensions), "CosineBefore": repr(similarity_before),
        "CosineAfter": repr(similarity_after), "RetainedVariation": repr(explained),
        "Capture": "actual core CBOW before/after tensors; one shared projection",
    })
    print(f"Recorded cat/dog full-profile similarity: {similarity_before:.6f} -> {similarity_after:.6f}")


def window_capture(namespace: dict[str, object], fingerprint: str) -> None:
    windows = namespace["cat_window_ids"]
    documents = namespace["id_documents"]
    words = namespace["itos"]
    width = namespace["window_width"]
    radius = namespace["CONTEXT_RADIUS"]
    if not isinstance(windows, torch.Tensor) or not isinstance(documents, list) or not isinstance(words, list):
        raise TypeError("The lesson must retain its actual window IDs, documents, and vocabulary.")
    if not isinstance(width, int) or not isinstance(radius, int):
        raise TypeError("Window width and context radius must be integers.")
    story_ids = documents[-1]
    rows = windows.tolist()
    expected = [story_ids[start:start + width] for start in range(len(story_ids) - width + 1)]
    if rows != expected:
        raise ValueError("The captured probe windows do not match the original sentence.")
    story_words = [words[index] for index in story_ids]
    answers = [words[row[radius]] for row in rows]
    figure, axes = plt.subplots(len(rows), 1, figsize=(13, 8))
    figure.patch.set_facecolor(BACKGROUND)
    for start, axis in enumerate(np.atleast_1d(axes)):
        axis.set(xlim=(-0.6, len(story_words) - 0.4), ylim=(-0.6, 2.0))
        axis.axis("off")
        axis.set_title(f"Window starts at position {start}", color=FOREGROUND, fontsize=13)
        for position, word in enumerate(story_words):
            active = start <= position < start + width
            axis.text(
                position, 1.35, word, ha="center", va="center", fontsize=13,
                color="#172b4d" if active else "#94a3b8",
                bbox=dict(boxstyle="round,pad=0.35", edgecolor=TEAL if active else "#64748b",
                          facecolor="#dbeafe" if active else BACKGROUND),
            )
        row_words = [words[index] for index in rows[start]]
        clues = row_words[:radius] + ["___"] + row_words[radius + 1:]
        axis.text(3, 0.45, "CBOW sees: " + " ".join(clues),
                  ha="center", color=TEAL, fontsize=13)
        axis.text(3, -0.25, "Checked separately: " + answers[start],
                  ha="center", color=ORANGE, fontsize=13, weight="bold")
    figure.suptitle(
        f"One {len(story_words)}-word sentence supplies {len(rows)} complete {width}-word windows",
        color=FOREGROUND, fontsize=17,
    )
    figure.text(0.5, 0.02, "The raw text is above each question. The center word is not a CBOW input.",
                ha="center", color=FOREGROUND, fontsize=11)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(figure, WINDOW_IMAGE, {
        "SourceFingerprint": fingerprint, "WindowCount": str(len(rows)),
        "WindowWidth": str(width), "Answers": ", ".join(answers),
        "Capture": "actual sentence-local cat-probe window IDs",
    })


def main() -> None:
    cache = HERE / "data" / "tinystories" / "TinyStories-valid.txt"
    if not cache.is_file():
        raise FileNotFoundError("Run the lesson once to obtain its pinned TinyStories cache.")
    sources = core_sources()
    fingerprint = source_fingerprint(sources)
    namespace: dict[str, object] = {
        "__name__": "__main__", "__file__": str(NOTEBOOK.with_suffix(".py")),
    }
    for index, source in enumerate(sources, 1):
        exec(compile(source, f"{NOTEBOOK.name}: code cell {index}", "exec"), namespace)
    profile_capture(namespace, fingerprint)
    window_capture(namespace, fingerprint)


if __name__ == "__main__":
    main()
