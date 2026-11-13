"""A one-state GRU forward GIF using the optional board's native-cell recording."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from gated_sequence_storyboards import NOTES, record_story
from lstm_forward_flow import (
    Box, COLORS, Label, Readout, Wire, draw_forward_header, draw_forward_scene,
    prediction_head, vector, write_forward_gif,
)


HERE = Path(__file__).parent
GIF = HERE / "images" / "01a-gru-forward-walkthrough.gif"
STORYBOARD = GIF.with_suffix(".png")
PHASES = (
    "Bring in the current note and the ONE old h.",
    "Choose RESET and KEEP. WRITE is 1-KEEP, not a third learned gate.",
    "KEEP retains old amounts directly, bypassing the proposal.",
    "RESET filters only the old-h context used by the proposal.",
    "The note context + reset-filtered past make NEW proposed content.",
    "WRITE admits the complementary share of NEW.",
    "Add kept + written to get new h. No final tanh wraps this blend.",
    "A learned Linear head makes note scores. Softmax + the largest probability gives the next note.",
    "Carry the SAME h into the next input. There is no separate c or SHOW.",
)


@dataclass(frozen=True)
class GRURead:
    index: int
    note: str
    next_note: str
    inputs: tuple[float, ...]
    old_hidden: tuple[float, ...]
    reset: tuple[float, ...]
    keep: tuple[float, ...]
    write: tuple[float, ...]
    input_context: tuple[float, ...]
    old_context: tuple[float, ...]
    reset_context: tuple[float, ...]
    proposal: tuple[float, ...]
    retained: tuple[float, ...]
    written: tuple[float, ...]
    new_hidden: tuple[float, ...]
    readout: Readout


def record_reads(*, close_reset: bool = False) -> tuple[GRURead, ...]:
    story = record_story("gru")
    head = story["head_parameters"]
    result = []
    for step in story["steps"]:
        input_context = tuple(step["input_projection"][-2:])
        reset, reset_context = tuple(step["reset"]), tuple(step["reset_branch"])
        proposal, written, hidden = tuple(step["proposal"]), tuple(step["written"]), tuple(step["h"])
        readout = Readout(tuple(step["logits"]), tuple(step["probabilities"]), step["guess"])
        if close_reset:
            reset, reset_context = (0., 0.), (0., 0.)
            proposal = tuple(torch.tensor(input_context, dtype=torch.float64).tanh().tolist())
            written = tuple(gate * value for gate, value in zip(step["write"], proposal, strict=True))
            hidden = tuple(old + new for old, new in zip(step["retained"], written, strict=True))
            scores = F.linear(torch.tensor(hidden, dtype=torch.float64),
                              torch.tensor(head["weight"], dtype=torch.float64),
                              torch.tensor(head["bias"], dtype=torch.float64))
            readout = Readout(tuple(scores.tolist()), tuple(scores.softmax(0).tolist()),
                              NOTES[int(scores.argmax())])
        result.append(GRURead(
            step["index"] + 1, step["note"], NOTES[(step["index"] + 1) % len(NOTES)],
            tuple(step["x"]), tuple(step["old_h"]), reset, tuple(step["keep"]), tuple(step["write"]),
            input_context, tuple(step["proposal_hidden_projection"]), reset_context, proposal,
            tuple(step["retained"]), written, hidden, readout,
        ))
    return tuple(result)


def scene(read: GRURead) -> tuple[list[Box], list[Wire], list[Label]]:
    t = read.index
    head, result = prediction_head(read.readout, t, (840, 300, 240, 140),
                                   (840, 560, 240, 174), 7, bias=True)

    def node(name, bounds, role, phase, title, values, size=20):
        x, y, width, _ = bounds
        return Box(name, bounds, role, (phase,), (
            Label(x + width / 2, y + 27, title, 21, role),
            Label(x + width / 2, y + 65, vector(values), size, role, mono=True),
        ))

    boxes = [
        node("old-hidden", (20, 40, 170, 85), "hidden", 0, f"old h{t-1}", read.old_hidden),
        node("input", (20, 290, 170, 90), "text", 0, f"Input: {read.note}", read.inputs),
        Box("controls", (300, 245, 240, 220), "control", (1,), (
            Label(420, 275, "RESET + KEEP", 22),
            Label(420, 307, "WRITE = 1 - KEEP", 18, "muted"),
            *(Label(316, y, name, 19, "control", "start")
              for name, y in (("RESET", 350), ("KEEP", 390), ("WRITE", 430))),
            *(Label(524, y, vector(values), 18, "control", "end", True)
              for values, y in ((read.reset, 350), (read.keep, 390), (read.write, 430))),
        )),
        node("kept", (330, 40, 190, 85), "hidden", 2, "KEEP x old h", read.retained),
        node("reset", (300, 600, 240, 110), "cell", 3, "RESET x old context", read.reset_context, 18),
        node("proposal", (620, 600, 170, 110), "cell", 4, "NEW: tanh(...)", read.proposal, 18),
        node("written", (620, 245, 170, 110), "cell", 5, "WRITE x NEW", read.written, 18),
        node("new-hidden", (870, 40, 210, 100), "hidden", 6, f"h{t}: new state", read.new_hidden, 22),
        head, result,
        node("next-hidden", (840, 770, 240, 90), "hidden", 8, f"Next old h = h{t}", read.new_hidden),
    ]
    wires = [
        Wire("hidden-to-keep", "old-hidden", "kept", ((196, 82), (324, 82)), "hidden", (2,)),
        Wire("input-to-controls", "input", "controls", ((196, 335), (294, 335)), "text", (0,)),
        Wire("hidden-to-controls", "old-hidden", "controls",
             ((196, 82), (270, 82), (270, 200), (350, 200), (350, 239)), "hidden", (0,)),
        Wire("keep-control", "controls", "kept", ((425, 239), (425, 131)), "control", (2,)),
        Wire("hidden-to-reset", "old-hidden", "reset",
             ((14, 82), (8, 82), (8, 655), (294, 655)), "hidden", (3,)),
        Wire("reset-control", "controls", "reset", ((420, 471), (420, 594)), "control", (3,)),
        Wire("reset-to-proposal", "reset", "proposal", ((546, 655), (614, 655)), "cell", (4,)),
        Wire("input-to-proposal", "input", "proposal",
             ((196, 335), (280, 335), (280, 550), (705, 550), (705, 594)), "text", (4,)),
        Wire("proposal-to-written", "proposal", "written", ((755, 594), (755, 361)), "cell", (5,)),
        Wire("write-control", "controls", "written",
             ((546, 430), (580, 430), (580, 300), (614, 300)), "control", (5,)),
        Wire("kept-to-add", "kept", "add", ((526, 82), (656, 82)), "hidden", (6,)),
        Wire("written-to-add", "written", "add",
             ((705, 239), (705, 185), (680, 185), (680, 106)), "cell", (6,)),
        Wire("add-to-hidden", "add", "new-hidden", ((704, 82), (864, 82)), "hidden", (6,)),
        Wire("hidden-to-head", "new-hidden", "head", ((975, 146), (975, 294)), "hidden", (7,)),
        Wire("head-to-prediction", "head", "prediction", ((975, 446), (975, 554)), "prediction", (7,)),
        Wire("carry-hidden", "new-hidden", "next-hidden",
             ((1086, 90), (1115, 90), (1115, 920), (960, 920), (960, 866)), "hidden", (8,)),
    ]
    return boxes, wires, [
        Label(680, 91, "+", 29),
        Label(460, 855, "ONE h: output and next memory", 22, "hidden"),
        Label(460, 895, "No separate c. No final tanh after the blend.", 18, "muted"),
    ]


def flow_signals(phase: int, progress: float) -> tuple[tuple[str, float], ...]:
    if not 0 <= phase < len(PHASES) or not 0 <= progress <= 1:
        raise ValueError("Choose a GRU operation and signal progress between zero and one.")
    routes = {
        0: ("input-to-controls", "hidden-to-controls"), 1: (),
        2: ("hidden-to-keep", "keep-control"), 3: ("hidden-to-reset", "reset-control"),
        4: ("input-to-proposal", "reset-to-proposal"), 5: ("proposal-to-written", "write-control"),
        8: ("carry-hidden",),
    }
    if phase in (6, 7):
        first, second = {
            6: (("kept-to-add", "written-to-add"), ("add-to-hidden",)),
            7: (("hidden-to-head",), ("head-to-prediction",)),
        }[phase]
        names = first if progress <= .5 else second
        position = progress * 2 if progress <= .5 else (progress - .5) * 2
        return tuple((name, position) for name in names)
    return tuple((name, progress) for name in routes[phase])


@dataclass(frozen=True)
class GRUFrame:
    read_index: int
    phase: int
    progress: float
    duration_ms: int
    close_reset: bool = False
    completed_phase: int = -1


def gif_frames() -> tuple[GRUFrame, ...]:
    stages = [(2, phase, False) for phase in range(8)]
    stages += [(2, 7, True), (2, 8, False), (3, 0, False), (3, 1, False)]
    frames, completed = [], -1
    for index, phase, closed in stages:
        for progress in (0., .25, .5, .75, 1.):
            frames.append(GRUFrame(index, phase, progress, 1600 if progress == 1 else 120, closed, completed))
        completed = max(completed, phase)
    return tuple(frames)


def gif_figure(read: GRURead, frame: GRUFrame):
    figure, axis = plt.subplots(figsize=(11.3, 12.15), dpi=100, facecolor=COLORS["background"])
    figure.subplots_adjust(left=.015, right=.985, top=.79, bottom=.035)
    axis.set(xlim=(0, 1130), ylim=(980, 0))
    axis.axis("off")
    draw_forward_scene(
        axis, *scene(read), phase=frame.phase, progress=frame.progress,
        completed_phase=frame.completed_phase, signals_for=flow_signals,
        probabilities=read.readout.probabilities,
    )
    title = f"Read {read.index}: {read.note} | {frame.phase + 1}/9"
    if frame.close_reset:
        title += " | WHAT IF RESET = 0?"
    elif frame.phase == 8:
        title += " | restore the normal read"
    caption = ("Close RESET: the kept old amount is UNCHANGED. Only the proposed write changes."
               if frame.close_reset else PHASES[frame.phase])
    details = (
        f"old h = {vector(read.old_hidden)}; input = {vector(read.inputs)}",
        f"RESET {vector(read.reset)}; KEEP {vector(read.keep)}; WRITE {vector(read.write)}",
        f"Kept {vector(read.retained)} = KEEP * old h",
        f"RESET {vector(read.reset)} * projected old context {vector(read.old_context)}",
        f"NEW = tanh(note context + reset context) = {vector(read.proposal)}",
        f"Written {vector(read.written)} = (1-KEEP) * NEW",
        f"Kept {vector(read.retained)} + written {vector(read.written)} = h {vector(read.new_hidden)}",
        f"head(h) scores {vector(read.readout.scores)}; softmax = {vector(read.readout.probabilities)}",
        f"Next old h = {vector(read.new_hidden)}; the head reads these SAME values.",
    )
    detail = (f"Kept {vector(read.retained)} (same); NEW {vector(read.proposal)}; h {vector(read.new_hidden)}"
              if frame.close_reset else details[frame.phase])
    draw_forward_header(
        figure, title, caption, detail if frame.progress == 1 else "Follow the separate keep and proposal routes.",
        "Cyan: carried h | Gold: proposed/written contents | Dashed purple: controls | Hand-picked, not trained.",
    )
    axis.plot((25, 25 + 1060 * frame.progress), (-20, -20), color=COLORS["cell"], linewidth=3, clip_on=False)
    return figure


def export_gif(path: Path = GIF, poster: Path = STORYBOARD) -> None:
    reads, closed = record_reads(), record_reads(close_reset=True)
    specifications = gif_frames()
    metadata = {
        "version": 1, "state_count": 1, "state_width": 2, "head_input": "new_hidden",
        "steps": [asdict(read) for read in reads], "closed_steps": [asdict(read) for read in closed],
        "frames": [asdict(frame) for frame in specifications],
        "origin": "Native GRUCell from the optional board; RESET=0 is one isolated read; no training.",
    }
    snapshot = next(frame for frame in specifications if frame.phase == 8 and frame.progress == 1)
    figure = gif_figure(reads[1], snapshot)
    try:
        figure.savefig(poster, dpi=100, facecolor=COLORS["background"],
                       metadata={"HeadInput": "new_hidden", "StateCount": "1",
                                 "RecordedRead": json.dumps(asdict(reads[1]))})
    finally:
        plt.close(figure)
    write_forward_gif(
        path, (gif_figure((closed if frame.close_reset else reads)[frame.read_index - 1], frame)
               for frame in specifications),
        [frame.duration_ms for frame in specifications], metadata,
    )


if __name__ == "__main__":
    export_gif()
    print("Exported the native one-state GRU GIF and static fallback.")
