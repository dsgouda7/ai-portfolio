"""Replay recorded LSTM/GRU arithmetic from 01A without inventing state values."""

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

from gru_layer_flow import LAYER_PHASES, record_layer_step
from memory_visuals import (
    MOTION_FRAMES, INTERVAL_MS, draw_scene, gru_scene, scalar_scene,
)


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01A-sequence-memory-theory.ipynb"
RECORDING_CELLS = ("eb7df241", "f6985b8c", "gru-controlled-recording")
BACKGROUND = "#1a1a2e"
TEXT = "#e7edf7"
MEMORY = "#76c6ff"
WRITE = "#ffbb78"
CONTROL = "#c5a3ff"
RESULT = "#8cdb9b"
PHASES = ("read old memory", "inspect the controls", "keep and write", "combine", "carry onward")


def responsive_animation_html(animation):
    html = animation.to_jshtml(default_mode="once")
    html = re.sub(r'<link\b[^>]*font-awesome[^>]*>\s*', "", html)
    html = re.sub(
        r'(<button\b[^>]*title="([^"]+)"[^>]*>)\s*<i\b[^>]*></i>',
        lambda match: match[1] + match[2], html,
    )
    first = re.search(r'frames\[0\] = "([^"]+)"', html)
    if first is None:
        raise ValueError("The animation export contains no first frame to display without scripts.")
    source = first[1].replace("\\\n", "")
    html = re.sub(
        r'(<img\b[^>]*)(>)',
        lambda match: match[1] + f' src="{source}" alt="Recorded memory-state animation"' + match[2],
        html, count=1,
    )
    return (
        "<style>.animation img{max-width:100%;height:auto}"
        ".animation .anim-buttons{display:flex;flex-wrap:wrap;justify-content:center;gap:4px}"
        ".animation .anim-buttons button{width:auto;height:auto;min-height:32px}"
        "</style>" + html
    )


def refresh_image_attachments():
    before = NOTEBOOK.read_bytes()
    notebook = json.loads(before)
    ending = "\n" if before.endswith(b"\n") else ""
    if (json.dumps(notebook, ensure_ascii=False, indent=1) + ending).encode("utf-8") != before:
        raise ValueError("Notebook formatting changed; refresh its image attachments with a notebook-aware editor.")
    import base64
    for cell in notebook["cells"]:
        for name, attachment in cell.get("attachments", {}).items():
            for mime in ("image/png", "image/gif"):
                if mime in attachment:
                    attachment[mime] = base64.b64encode((HERE / "images" / name).read_bytes()).decode("ascii")
    from notebook_replays import refresh_embedded_replays
    refresh_embedded_replays(notebook, images_directory=HERE / "images")
    if NOTEBOOK.read_bytes() != before:
        raise RuntimeError("The notebook changed while its image attachments were being refreshed.")
    NOTEBOOK.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + ending).encode("utf-8"))


@dataclass(frozen=True)
class MemoryStep:
    # A recorded scalar coordinate makes carry, replacement, and exposure separately inspectable.
    note: str
    old: float
    keep: float
    write: float
    proposal: float
    retained: float
    written: float
    carried: float
    exposed: float
    show: float | None = None
    reset: float | None = None
    input_proposal: float | None = None
    memory_proposal: float | None = None


def fingerprint(notebook: dict) -> str:
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    sources = [
        "".join(cells[cell_id]["source"])
        for cell_id in (*RECORDING_CELLS, "9df9ac84", "be857f77")
    ]
    sources.append(Path(__file__).read_text(encoding="utf-8"))
    sources.append((HERE / "gru_layer_flow.py").read_text(encoding="utf-8"))
    sources.append((HERE / "memory_visuals.py").read_text(encoding="utf-8"))
    syntax = "\n".join(ast.dump(ast.parse(source)) for source in sources)
    return hashlib.sha256(syntax.encode("utf-8")).hexdigest()


def recorded_steps(notebook: dict) -> dict[str, list[MemoryStep]]:
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    namespace = {"__name__": "memory_gate_recording"}
    for cell_id in RECORDING_CELLS:
        exec(compile("".join(cells[cell_id]["source"]), cell_id, "exec"), namespace)
    lstm, gru = [], []
    previous = 0.0
    for index, note in enumerate(namespace["melody"]):
        keep = float(namespace["forget"][index])
        write = float(namespace["input_gate"][index])
        proposal = float(namespace["candidate"][index])
        carried = float(namespace["cell_history"][index])
        lstm.append(MemoryStep(
            note, previous, keep, write, proposal, keep * previous, write * proposal,
            carried, float(namespace["hidden_history"][index]),
            show=float(namespace["output"][index]),
        ))
        previous = carried
        trace = namespace["gru_history"][index]
        keep = float(namespace["gru_update"][index])
        gru.append(MemoryStep(
            note, trace["old"], keep, 1 - keep, trace["proposal"],
            trace["retained"], trace["written"], trace["carried"], trace["carried"],
            reset=float(namespace["gru_reset"][index]),
            input_proposal=float(namespace["gru_input_proposal"][index]),
            memory_proposal=trace["memory_proposal"],
        ))
    return {"lstm": lstm, "gru": gru}


def draw_gate_flow(axis, kind: str, steps: list[MemoryStep], frame: int, *, progress=0.0) -> None:
    if kind not in {"lstm", "gru"} or not 0 <= frame < len(steps) * len(PHASES):
        raise ValueError("Choose a recorded LSTM/GRU flow frame.")
    index, phase = divmod(frame, len(PHASES))
    draw_scene(axis, scalar_scene(kind, steps[index]), phase, native=False, progress=progress,
               note_position=f"note {index + 1}/{len(steps)}")


def render_flows(notebook: dict) -> None:
    recordings = recorded_steps(notebook)
    source_hash = fingerprint(notebook)
    for kind, steps in recordings.items():
        figure, axis = plt.subplots(figsize=(12.8, 8))
        figure.patch.set_facecolor(BACKGROUND)
        figure.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.025)
        draw_scene(axis, scalar_scene(kind, steps[2]), None, native=False, note_position="note 3/6")
        prefix = HERE / "images" / f"01a-{kind}-gate-flow"
        figure.savefig(prefix.with_suffix(".png"), dpi=120, facecolor=BACKGROUND, metadata={
            "SourceFingerprint": source_hash, "Kind": kind.upper(),
            "TraceOrigin": "chosen controls; actual notebook state arithmetic; not fitted architecture scores",
        })
        figure.savefig(prefix.with_suffix(".svg"), facecolor=BACKGROUND)
        animation = FuncAnimation(
            figure, lambda frame, kind=kind, steps=steps, axis=axis: draw_gate_flow(
                axis, kind, steps, frame // MOTION_FRAMES,
                progress=(frame % MOTION_FRAMES) / (MOTION_FRAMES - 1),
            ),
            frames=len(steps) * len(PHASES) * MOTION_FRAMES, interval=INTERVAL_MS, repeat=False,
        )
        plt.close(figure)
        prefix.with_suffix(".html").write_text(responsive_animation_html(animation), encoding="utf-8")
        animation.save(prefix.with_suffix(".gif"), writer="pillow", fps=1000 / INTERVAL_MS)
        metadata = {
            "SourceFingerprint": source_hash, "Kind": kind.upper(), "Notes": [step.note for step in steps],
            "Phases": list(PHASES), "StateCount": 2 if kind == "lstm" else 1,
            "FinalCarried": steps[-1].carried, "FinalExposed": steps[-1].exposed,
            "MotionFramesPerPhase": MOTION_FRAMES,
            "TraceOrigin": "chosen controls; actual notebook arithmetic; not trained gates",
        }
        with Image.open(prefix.with_suffix(".gif")) as image:
            durations = [frame.info["duration"] for frame in ImageSequence.Iterator(image)]
            frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]
        frames[0].save(
            prefix.with_suffix(".gif"), save_all=True, append_images=frames[1:],
            duration=durations, loop=0, comment=json.dumps(metadata).encode("utf-8"),
        )
        print(f"Saved {prefix.name}: {len(frames)} flow stages, final carried {steps[-1].carried:.6f}.")


def recorded_evidence(notebook: dict) -> dict:
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    namespace = {"__name__": "memory_evidence_recording"}
    for cell_id in ("eb7df241", "9df9ac84", "f6985b8c", "be857f77"):
        exec(compile("".join(cells[cell_id]["source"]), cell_id, "exec"), namespace)
    return namespace


def render_evidence_charts(notebook: dict) -> None:
    values = recorded_evidence(notebook)
    source_hash = fingerprint(notebook)
    distances, gradient = values["distances"], values["vanilla_gradient"]

    def save(figure, name, measured):
        figure.savefig(
            HERE / "images" / f"01a-{name}.png", dpi=120,
            metadata={"SourceFingerprint": source_hash, "MeasuredValues": json.dumps(measured)},
        )
        plt.close(figure)

    with plt.rc_context({
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
        "axes.titlepad": 14, "axes.labelpad": 10,
        "figure.facecolor": "#fffaf1", "axes.facecolor": "#fffaf1", "text.color": "#17324d",
        "axes.labelcolor": "#17324d", "xtick.color": "#17324d", "ytick.color": "#17324d",
    }):
        figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
        axis.plot(distances, gradient, color="#f97316", linewidth=3)
        axis.axhline(0.01, color="#ef4444", linestyle="--", label="1% retained")
        axis.set(yscale="log", xlabel="Distance from the loss", ylabel="Gradient reaching the cue",
                 title="Chosen scalar route: feedback decays with distance")
        axis.legend()
        save(figure, "feedback-decay", {
            "distances": distances.tolist(), "gradient": gradient.tolist(),
        })

        steps = range(len(values["cell_history"]))
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
        for name, label, color in (
            ("forget", "keep / forget", "#60a5fa"),
            ("input_gate", "write / input", "#f59e0b"),
            ("output", "show / output", "#34d399"),
        ):
            axes[0].plot(steps, values[name], label=label, color=color, marker="o")
        axes[0].set_ylabel("Chosen gate value")
        axes[0].legend(ncol=3)
        axes[1].plot(steps, values["cell_history"], label="cell state", color="#a78bfa", marker="o")
        axes[1].plot(steps, values["hidden_history"], label="visible hidden", color="#f472b6", marker="o")
        axes[1].set(xlabel="Note position", ylabel="State value")
        axes[1].set_xticks(list(steps), values["melody"])
        axes[1].legend()
        save(figure, "lstm-state-trace", {
            key: values[key].tolist() if hasattr(values[key], "tolist") else values[key]
            for key in ("forget", "input_gate", "output", "cell_history", "hidden_history")
        })

        figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
        axis.semilogy(distances, gradient, label="vanilla route (0.65)", linewidth=3)
        axis.semilogy(distances, values["lstm_gradient"], label="LSTM cell route (0.95)", linewidth=3)
        axis.set(xlabel="Distance from loss", ylabel="Gradient reaching cue",
                 title="Chosen carry routes, not trained-architecture scores")
        axis.legend()
        save(figure, "feedback-retention-comparison", {
            "distances": distances.tolist(), "vanilla": gradient.tolist(),
            "lstm": values["lstm_gradient"].tolist(),
        })


def recorded_foundation_layer(notebook: dict):
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    namespace = {"__name__": "foundation_layer_recording"}
    for cell_id in ("eb7df241", "gru-real-cell-parity"):
        exec(compile("".join(cells[cell_id]["source"]), cell_id, "exec"), namespace)
    step = record_layer_step(
        namespace["gru_real_cell"], namespace["gru_piano_rows"][1:2],
        namespace["gru_real_trace"][1]["old"],
    )
    torch.testing.assert_close(step.output, namespace["gru_real_trace"][1]["actual"].reshape(-1),
                               rtol=0, atol=1e-12)
    return step


def render_foundation_layer(notebook: dict) -> None:
    step = recorded_foundation_layer(notebook)
    figure, axis = plt.subplots(figsize=(12.8, 8))
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.025)
    scene = gru_scene(step)

    def draw(frame):
        draw_scene(axis, scene, frame // MOTION_FRAMES, native=True,
                   progress=(frame % MOTION_FRAMES) / (MOTION_FRAMES - 1))

    prefix = HERE / "images" / "01a-gru-token-layer-flow"
    draw_scene(axis, scene, None, native=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=120, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": fingerprint(notebook), "Input": "A3", "InputWidth": "2",
        "HiddenWidth": "2", "OutputEqualsCarried": str(torch.equal(step.output, step.carried)),
        "Output": json.dumps(step.output.tolist()), "Carried": json.dumps(step.carried.tolist()),
    })
    figure.savefig(prefix.with_suffix(".svg"), facecolor=BACKGROUND)
    animation = FuncAnimation(figure, draw, frames=len(LAYER_PHASES) * MOTION_FRAMES,
                              interval=INTERVAL_MS, repeat=False)
    plt.close(figure)
    prefix.with_suffix(".html").write_text(responsive_animation_html(animation), encoding="utf-8")
    animation.save(prefix.with_suffix(".gif"), writer="pillow", fps=1000 / INTERVAL_MS)
    print(f"Saved {prefix.name}: {len(LAYER_PHASES)} stages; output and next-token state are identical.")


if __name__ == "__main__":
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    render_flows(notebook)
    render_evidence_charts(notebook)
    render_foundation_layer(notebook)
    refresh_image_attachments()
