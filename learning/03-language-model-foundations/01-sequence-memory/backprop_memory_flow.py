"""Record one real backward pass and replay inspection of the saved notecards."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from PIL import Image, ImageSequence
import torch
from torch import Tensor
from torch.nn import functional as F

from generate_memory_diagrams import PAPER, ORANGE, PURPLE, TEAL, arrow, canvas, card, label
from memory_gate_flows import responsive_animation_html
from notebook_replays import NOTEBOOK_SIZE_LIMIT


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01A-sequence-memory-theory.ipynb"
PREFIX = "01a-backprop-saved-cards"
NOTES = ("D3", "A3", "D4")
FRAMES_PER_STAGE = 2
INTERVAL_MS = 650
STAGES = (
    "Read D3. Save h1.",
    "Read A3. Save h2.",
    "Read D4. Save h3.",
    "Predict note 4 from h3. Check the answer.",
    "Inspect saved h3 and the h2 that helped make it.",
    "Inspect saved h2 and the h1 that helped make it.",
    "Inspect saved h1. Reach the initial blank card.",
    "Collect the corrections. Update the shared weights ONCE.",
    "Keep the card. Choose where backward feedback stops.",
)


@dataclass(frozen=True)
class BackpropTrace:
    states: tuple[Tensor, ...]
    feedback: tuple[Tensor | None, ...]
    memory_contributions: tuple[Tensor, ...]
    memory_gradient: Tensor
    guess: str
    loss_before: float
    loss_after: float
    truncated: bool


def _inputs_and_weights(path: Path) -> tuple[Tensor, Tensor, Tensor]:
    book = json.loads(path.read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in book["cells"] if cell["id"] == "ac9aca71")
    nodes = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"x_steps", "W_x", "W_h"}
            for target in node.targets
        )
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook_recurrence", "exec"), namespace)
    values = tuple(namespace[name].detach().double().clone() for name in ("x_steps", "W_x", "W_h"))
    if tuple(value.shape for value in values) != ((3, 2), (2, 2), (2, 2)):
        raise ValueError("This close-up needs the notebook's three two-value inputs and two-by-two weights.")
    return values


def record_trace(path: Path = NOTEBOOK, *, truncated: bool = False) -> BackpropTrace:
    inputs, input_weights, memory_weights = _inputs_and_weights(path)
    input_weights.requires_grad_()
    memory_weights.requires_grad_()
    head = torch.tensor(
        [[1.0, -0.5], [-0.5, 1.0], [0.25, 0.25]],
        dtype=torch.float64, requires_grad=True,
    )
    optimizer = torch.optim.SGD((input_weights, memory_weights, head), lr=0.02)
    states = [torch.zeros(2, dtype=torch.float64)]
    for index, note in enumerate(inputs):
        old = states[-1].detach() if truncated and index == 1 else states[-1]
        states.append(torch.tanh(input_weights @ note + memory_weights @ old))
    logits = head @ states[-1]
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))
    feedback = torch.autograd.grad(loss, states[1:], retain_graph=True, allow_unused=True)
    contributions = tuple(
        torch.zeros_like(memory_weights) if signal is None else
        torch.outer(signal * (1 - state.square()), old).detach()
        for state, old, signal in zip(states[1:], states[:-1], feedback)
    )
    loss.backward()
    if memory_weights.grad is None:
        raise RuntimeError("The final loss did not reach the shared memory weights.")
    memory_gradient = memory_weights.grad.detach().clone()
    torch.testing.assert_close(torch.stack(contributions).sum(0), memory_gradient, rtol=0, atol=1e-12)
    optimizer.step()
    with torch.no_grad():
        after = torch.zeros(2, dtype=torch.float64)
        for note in inputs:
            after = torch.tanh(input_weights @ note + memory_weights @ after)
        loss_after = F.cross_entropy((head @ after).unsqueeze(0), torch.tensor([0])).item()
    return BackpropTrace(
        tuple(state.detach().clone() for state in states),
        tuple(None if signal is None else signal.detach().clone() for signal in feedback),
        contributions, memory_gradient, NOTES[int(logits.argmax())],
        loss.detach().item(), loss_after, truncated,
    )


def source_fingerprint(path: Path = NOTEBOOK) -> str:
    inputs, input_weights, memory_weights = _inputs_and_weights(path)
    payload = {
        "values": [value.tolist() for value in (inputs, input_weights, memory_weights)],
        "source": Path(__file__).read_text(encoding="utf-8"),
    }
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def draw_stage(axis, trace: BackpropTrace, truncated: BackpropTrace, stage: int, progress=0.0) -> None:
    if not 0 <= stage < len(STAGES) or not 0 <= progress <= 1:
        raise ValueError("Choose a recorded stage and marker progress between zero and one.")
    axis.clear()
    axis.set(xlim=(0, 16), ylim=(0, 9))
    axis.axis("off")
    label(axis, 8, 8.3, "Training can inspect the earlier cards.", size=25, weight="bold")
    label(axis, 8, 7.65, STAGES[stage], size=19)
    positions = (1.0, 5.7, 10.4)
    active = 6 - stage if 4 <= stage <= 6 else None
    shown = min(stage + 1, 3)
    view = truncated if stage == 8 else trace
    for index, (x, note) in enumerate(zip(positions, NOTES)):
        exists = index < shown
        label(axis, x + 1.7, 6.6, f"Read {note}", size=22)
        color = ORANGE if active == index else TEAL if exists else "#e8e9ea"
        state = trace.states[index + 1]
        values = f"[{state[0]:+.3f}, {state[1]:+.3f}]" if exists else "not read yet"
        card(axis, x, 4.65, 3.4, 1.4, f"h{index + 1}\n{values}", color, size=21)
        if stage >= 4:
            signal = view.feedback[index]
            if active is not None and index < active:
                text = "next backward checks"
            else:
                text = "feedback stopped" if signal is None else f"feedback size: {signal.norm():.4f}"
            label(axis, x + 1.7, 4.05, text, size=17)
        if index < 2:
            if stage < 4:
                arrow(axis, (x + 3.5, 5.35), (positions[index + 1] - 0.15, 5.35))
            elif stage != 8 or index == 1:
                arrow(axis, (positions[index + 1] - 0.15, 5.35), (x + 3.5, 5.35), color="#a7501b")
    if active is not None and active > 0:
        start = positions[active] - 0.15
        end = positions[active - 1] + 3.5
        axis.add_patch(Circle((start + (end - start) * progress, 5.35), 0.11, color="#a7501b"))
    if stage < 3:
        message = "Save each card and its calculation. The weights have not changed."
    elif stage == 3:
        message = f"Initialized guess: {trace.guess}. Checking answer: D3. Do not update weights yet."
    elif stage == 4:
        message = "Saved h2 and note D4 enter the correction for the reader that made h3."
    elif stage == 5:
        message = "Saved h1 and note A3 enter the correction for the reader that made h2."
    elif stage == 6:
        message = "Inspect h1's calculation too. Stop at the initial blank card."
    elif stage == 7:
        message = f"One actual optimizer update: loss {trace.loss_before:.4f} -> {trace.loss_after:.4f}."
    else:
        axis.plot((4.95, 4.95), (3.8, 6.25), color="#a7501b", linestyle="--", linewidth=2)
        message = "Detach after h1: its values still enter h2, but feedback cannot travel into h1."
    label(axis, 8, 3.15, message, size=17)
    card(axis, 1.0, 1.5, 12.8, 0.95,
         "Collect contributions first. Update the SAME shared recipe once." if stage == 7 else
         "The SAME card-writing recipe is reused at all three steps.", PURPLE, size=19)
    label(axis, 8, 0.65,
          "Recorded cards. Measured feedback. One actual weight update.", size=15)


def render_replay(path: Path = NOTEBOOK, directory: Path = HERE / "images") -> None:
    full, truncated = record_trace(path), record_trace(path, truncated=True)
    directory.mkdir(exist_ok=True)
    prefix = directory / PREFIX
    with Image.open(prefix.with_suffix(".png")) as storyboard:
        storyboard.resize((1280, 720), Image.Resampling.LANCZOS).save(
            prefix.with_name(PREFIX + "-preview.png"), optimize=True,
        )
    figure, axis = canvas("Training can inspect the earlier cards.", "")
    figure.set_dpi(80)
    draw_stage(axis, full, truncated, 5)
    metadata = {
        "SourceFingerprint": source_fingerprint(path),
        "TraceOrigin": "actual notebook recurrence; hand-picked prediction head; one measured optimizer update",
    }
    figure.savefig(prefix.with_name(PREFIX + "-trace.png"), dpi=80, facecolor=PAPER, metadata=metadata)
    animation = FuncAnimation(
        figure,
        lambda frame: draw_stage(
            axis, full, truncated, frame // FRAMES_PER_STAGE,
            (frame % FRAMES_PER_STAGE) / (FRAMES_PER_STAGE - 1),
        ),
        frames=len(STAGES) * FRAMES_PER_STAGE, interval=INTERVAL_MS, repeat=False,
    )
    plt.close(figure)
    prefix.with_suffix(".html").write_text(responsive_animation_html(animation), encoding="utf-8")
    animation.save(prefix.with_suffix(".gif"), writer="pillow", fps=1000 / INTERVAL_MS, dpi=80)
    with Image.open(prefix.with_suffix(".gif")) as image:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(image)]
        frames = [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(image)]
    frames[0].save(
        prefix.with_suffix(".gif"), save_all=True, append_images=frames[1:], duration=durations, loop=0,
        comment=json.dumps({**metadata, "Stages": STAGES, "LossBefore": full.loss_before,
                            "LossAfter": full.loss_after, "TruncatedOpeningFeedback": None}).encode("utf-8"),
    )
    print(f"Saved {PREFIX}: inspect h3, h2, h1; collect gradients; update once; demonstrate a detach boundary.")


def refresh_notebook_captures(path: Path = NOTEBOOK, directory: Path = HERE / "images") -> None:
    import base64
    from notebook_replays import refresh_embedded_replays

    before = path.read_bytes()
    book = json.loads(before)
    ending = "\n" if before.endswith(b"\n") else ""
    if (json.dumps(book, ensure_ascii=False, indent=1) + ending).encode("utf-8") != before:
        raise ValueError("Notebook serialization changed; use a notebook-aware editor to refresh its captures.")
    prose = next(cell for cell in book["cells"] if cell["id"] == "968f14e7")
    replay = next(cell for cell in book["cells"] if cell["id"] == "backprop-saved-card-replay")
    sources = [cell["source"] for cell in book["cells"]]
    for filename, mime in (
        (PREFIX + ".png", "image/png"),
        (PREFIX + "-trace.png", "image/png"),
        (PREFIX + ".gif", "image/gif"),
    ):
        prose.setdefault("attachments", {})[filename] = {
            mime: base64.b64encode((directory / filename).read_bytes()).decode("ascii"),
        }
    refresh_embedded_replays({"cells": [replay]}, images_directory=directory)
    if [cell["source"] for cell in book["cells"]] != sources or path.read_bytes() != before:
        raise RuntimeError("Notebook sources changed during capture refresh.")
    encoded = (json.dumps(book, ensure_ascii=False, indent=1) + ending).encode("utf-8")
    if len(encoded) >= NOTEBOOK_SIZE_LIMIT:
        raise ValueError(f"Embedded captures would exceed the notebook size budget: {len(encoded):,} bytes.")
    path.write_bytes(encoded)


if __name__ == "__main__":
    render_replay()
    refresh_notebook_captures()
