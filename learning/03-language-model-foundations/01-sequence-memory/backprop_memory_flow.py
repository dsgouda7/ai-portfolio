"""Record one real backward pass and replay inspection of the saved notecards."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageSequence
import torch
from torch import Tensor
from torch.nn import functional as F

from memory_gate_flows import responsive_animation_html
from notebook_replays import NOTEBOOK_SIZE_LIMIT
from sequence_studio import BACKGROUND, PLAYER_STYLE, draw_studio, studio_figure


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01A-sequence-memory-theory.ipynb"
PREFIX = "01a-backprop-saved-cards"
NOTES = ("D3", "A3", "D4")
NOTE_HEAD_WEIGHTS = ((1.0, -0.5), (-0.5, 1.0), (0.25, 0.25))
FRAMES_PER_STAGE = 2
INTERVAL_MS = 650
STAGES = (
    "Read D3 + h0. Carry h1 and predict note 2.",
    "Read A3 + h1. Carry h2 and predict note 3.",
    "Read D4 + h2. Carry h3 and predict note 4.",
    "Outputs exist at every step. Grade note 4 here.",
    "Inspect the saved step: D4 + h2 -> h3.",
    "Inspect the saved step: A3 + h1 -> h2.",
    "Inspect the saved step: D3 + h0 -> h1.",
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
    outputs: tuple[Tensor, ...]
    scores: tuple[Tensor, ...]
    predictions: tuple[str, ...]


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
        NOTE_HEAD_WEIGHTS,
        dtype=torch.float64, requires_grad=True,
    )
    optimizer = torch.optim.SGD((input_weights, memory_weights, head), lr=0.02)
    states = [torch.zeros(2, dtype=torch.float64)]
    scores: list[Tensor] = []
    for index, note in enumerate(inputs):
        old = states[-1].detach() if truncated and index == 1 else states[-1]
        states.append(torch.tanh(input_weights @ note + memory_weights @ old))
        scores.append(head @ states[-1])
    loss = F.cross_entropy(scores[-1].unsqueeze(0), torch.tensor([0]))
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
    saved_states = tuple(state.detach().clone() for state in states)
    predictions = tuple(NOTES[int(score.argmax())] for score in scores)
    return BackpropTrace(
        states=saved_states,
        feedback=tuple(None if signal is None else signal.detach().clone() for signal in feedback),
        memory_contributions=contributions, memory_gradient=memory_gradient,
        guess=predictions[-1], loss_before=loss.detach().item(), loss_after=loss_after,
        truncated=truncated, outputs=saved_states[1:],
        scores=tuple(score.detach().clone() for score in scores), predictions=predictions,
    )


def source_fingerprint(path: Path = NOTEBOOK) -> str:
    inputs, input_weights, memory_weights = _inputs_and_weights(path)
    payload = {
        "values": [value.tolist() for value in (inputs, input_weights, memory_weights)],
        "source": Path(__file__).read_text(encoding="utf-8"),
        "studio": (HERE / "sequence_studio.py").read_text(encoding="utf-8"),
    }
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def draw_stage(axis, trace: BackpropTrace, truncated: BackpropTrace, stage: int, progress=0.0) -> None:
    if not 0 <= stage < len(STAGES) or not 0 <= progress <= 1:
        raise ValueError("Choose a recorded stage and marker progress between zero and one.")
    draw_studio(axis, trace, stage=stage, caption=STAGES[stage], progress=progress,
                truncated_trace=truncated)


def compact_animation_frames(html: str) -> str:
    """Keep every timed frame, but reference earlier identical image payloads."""
    seen = {}

    def compact(match):
        index, image = match.groups()
        key = image.replace("\\\r\n", "").replace("\\\n", "")
        if key in seen:
            return f"frames[{index}] = frames[{seen[key]}];"
        seen[key] = index
        return match[0]

    packed = re.sub(r'frames\[(\d+)\]\s*=\s*"([^"]+)"', compact, html)
    if not seen:
        raise ValueError("The animation export contains no image-frame assignments.")
    return packed


def render_replay(path: Path = NOTEBOOK, directory: Path = HERE / "images") -> None:
    full, truncated = record_trace(path), record_trace(path, truncated=True)
    directory.mkdir(exist_ok=True)
    prefix = directory / PREFIX
    with Image.open(prefix.with_suffix(".png")) as storyboard:
        storyboard.resize((1280, 720), Image.Resampling.LANCZOS).save(
            prefix.with_name(PREFIX + "-preview.png"), optimize=True,
        )
    figure, axis = studio_figure()
    figure.set_dpi(80)
    draw_stage(axis, full, truncated, 5)
    metadata = {
        "SourceFingerprint": source_fingerprint(path),
        "TraceOrigin": "actual recurrence and shared head at every step; only note-four loss; one optimizer update",
        "StepPredictions": json.dumps(full.predictions),
    }
    figure.savefig(prefix.with_name(PREFIX + "-trace.png"), dpi=80, facecolor=BACKGROUND, metadata=metadata)
    animation = FuncAnimation(
        figure,
        lambda frame: draw_stage(
            axis, full, truncated, frame // FRAMES_PER_STAGE,
            (frame % FRAMES_PER_STAGE) / (FRAMES_PER_STAGE - 1),
        ),
        frames=len(STAGES) * FRAMES_PER_STAGE, interval=INTERVAL_MS, repeat=False,
    )
    plt.close(figure)
    prefix.with_suffix(".html").write_text(
        PLAYER_STYLE + compact_animation_frames(responsive_animation_html(animation)), encoding="utf-8",
    )
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
