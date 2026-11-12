"""A native two-state LSTM forward pass with explicit h-only note predictions."""

import ast
import base64
from dataclasses import asdict, dataclass
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
from PIL import Image, ImageOps
import torch
from torch import Tensor, nn

from backprop_memory_flow import NOTES, NOTE_HEAD_WEIGHTS, _inputs_and_weights
from lstm_layer_flow import record_lstm_layer_step
from lstm_state_roles import record_state_roles


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01A-sequence-memory-theory.ipynb"
PLAYER = HERE / "images" / "01a-lstm-forward-walkthrough.html"
STORYBOARD = PLAYER.with_suffix(".png")
ART = HERE / "images" / "01a-lstm-state-objects.png"
ART_MANIFEST = HERE / "images" / "01a-lstm-forward-prompts.json"
COLORS = {
    "background": "#0d1b2e", "panel": "#20364f", "text": "#edf3ff",
    "muted": "#b0bed3", "cell": "#ffd172", "hidden": "#77dce5",
    "control": "#c5a3ff", "prediction": "#c5a3ff",
}
PHASES = (
    "Bring in the current note, old h and old c.",
    "The note and old h choose the controls. Old c takes a separate lane.",
    "KEEP old contents. WRITE selected new content.",
    "Add those two contributions to make new c.",
    "SHOW a bounded view of new c to make new h.",
    "Only new h enters the note head; c has no direct prediction arrow.",
    "Carry BOTH new c and new h into the next read.",
)
RECORDING = re.compile(
    r'<script id="recorded-lstm" type="application/json">\s*(.*?)\s*</script>', re.DOTALL,
)


@dataclass(frozen=True)
class Readout:
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    prediction: str


@dataclass(frozen=True)
class LSTMRead:
    index: int
    note: str
    next_note: str
    inputs: tuple[float, ...]
    old_hidden: tuple[float, ...]
    old_cell: tuple[float, ...]
    keep: tuple[float, ...]
    write: tuple[float, ...]
    proposal: tuple[float, ...]
    show: tuple[float, ...]
    retained: tuple[float, ...]
    written: tuple[float, ...]
    new_cell: tuple[float, ...]
    new_hidden: tuple[float, ...]
    readout: Readout
    closed_show_cell: tuple[float, ...]
    closed_show_hidden: tuple[float, ...]
    closed_show_readout: Readout


def load_cell_and_inputs(path: Path = NOTEBOOK) -> tuple[nn.LSTMCell, Tensor]:
    book = json.loads(path.read_bytes())
    source = "".join(next(cell for cell in book["cells"] if cell["id"] == "lstm-real-cell-parity")["source"])
    setup = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.With) and any(
            isinstance(child, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "lstm_real_cell" for target in child.targets
            ) for child in ast.walk(node)
        )
    ]
    if len(setup) != 1:
        raise ValueError("Expected one isolated native LSTM initialization in the lesson.")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=setup, type_ignores=[]), "lesson_lstm_initialization", "exec"), namespace)
    cell = namespace["lstm_real_cell"]
    if not isinstance(cell, nn.LSTMCell) or cell.input_size != 2 or cell.hidden_size != 2:
        raise TypeError("The walkthrough requires the lesson's two-input, two-feature LSTMCell.")
    inputs, _, _ = _inputs_and_weights(path)
    return cell, inputs


def note_readout(hidden: Tensor) -> Readout:
    if hidden.shape != (2,) or not torch.isfinite(hidden).all():
        raise ValueError("The note head reads one finite two-feature hidden state.")
    with torch.no_grad():
        weights = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=hidden.dtype, device=hidden.device)
        scores = weights @ hidden
        probabilities = scores.softmax(0)
    return Readout(tuple(scores.tolist()), tuple(probabilities.tolist()), NOTES[int(scores.argmax())])


def record_reads(path: Path = NOTEBOOK) -> tuple[LSTMRead, ...]:
    cell, inputs = load_cell_and_inputs(path)
    hidden = torch.zeros((1, 2), dtype=torch.float64)
    memory = torch.zeros_like(hidden)
    result = []
    for offset, (note, row) in enumerate(zip(NOTES, inputs, strict=True)):
        step = record_lstm_layer_step(cell, row[None], hidden, memory)
        closed = record_state_roles(cell, row[None], hidden, memory).show_closed
        result.append(LSTMRead(
            offset + 1, note, NOTES[(offset + 1) % len(NOTES)], tuple(step.input_row.tolist()),
            tuple(step.old_hidden.tolist()), tuple(step.old_cell.tolist()),
            tuple(step.forget_gate.tolist()), tuple(step.input_gate.tolist()),
            tuple(step.candidate.tolist()), tuple(step.output_gate.tolist()),
            tuple(step.retained.tolist()), tuple(step.written.tolist()),
            tuple(step.carried_cell.tolist()), tuple(step.carried_hidden.tolist()), note_readout(step.carried_hidden),
            tuple(closed.carried_cell.tolist()), tuple(closed.carried_hidden.tolist()), note_readout(closed.carried_hidden),
        ))
        hidden, memory = step.carried_hidden[None], step.carried_cell[None]
    return tuple(result)


def role_images() -> dict[str, Image.Image]:
    manifest = json.loads(ART_MANIFEST.read_text(encoding="utf-8"))[0]
    result = {}
    with Image.open(ART) as original:
        for role, bounds in manifest["crop_boxes"].items():
            left, top, right, bottom = bounds
            if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
                raise ValueError(f"The {role} artwork crop is outside the MAI image.")
            image = ImageOps.contain(original.crop(bounds).convert("RGB"), (240, 180), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (240, 180), COLORS["background"])
            canvas.paste(image, ((240 - image.width) // 2, (180 - image.height) // 2))
            result[role] = canvas
    if set(result) != {"cell", "hidden"}:
        raise ValueError("The MAI manifest must define cell and hidden artwork.")
    return result


def vector(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(str(int(value)) if value == int(value) else f"{value:.2f}" for value in values) + "]"


@dataclass(frozen=True)
class Label:
    x: float
    y: float
    text: str
    size: int = 20
    color: str = "text"
    anchor: str = "middle"
    mono: bool = False


@dataclass(frozen=True)
class Box:
    name: str
    bounds: tuple[float, float, float, float]
    role: str
    phases: tuple[int, ...]
    labels: tuple[Label, ...]
    art: str | None = None


@dataclass(frozen=True)
class Wire:
    name: str
    source: str
    target: str
    points: tuple[tuple[float, float], ...]
    role: str
    phases: tuple[int, ...]


def scene(read: LSTMRead) -> tuple[list[Box], list[Wire], list[Label]]:
    t, old = read.index, read.index - 1
    boxes = [
        Box("old-cell", (20, 40, 200, 85), "cell", (0, 2), (
            Label(120, 67, f"c{old}: old workspace", 20, "cell"),
            Label(120, 102, vector(read.old_cell), 22, "cell", mono=True),
        )),
        Box("input", (20, 250, 200, 90), "text", (0, 1), (
            Label(120, 278, f"{read.note}: current note", 20),
            Label(120, 315, f"x{t} = {vector(read.inputs)}", 22, mono=True),
        )),
        Box("old-hidden", (20, 420, 200, 85), "hidden", (0, 1), (
            Label(120, 448, f"h{old}: old readout", 20, "hidden"),
            Label(120, 483, vector(read.old_hidden), 22, "hidden", mono=True),
        )),
        Box("controls", (260, 235, 280, 270), "control", (1,), (
            Label(400, 262, "Compute controls", 22),
            Label(400, 294, "Shared weights: note + old h", 18, "muted"),
            *(Label(276, y, name, 19, "control", "start") for name, y in (
                ("KEEP", 327), ("WRITE", 368), ("NEW", 409), ("SHOW", 450),
            )),
            *(Label(524, y, vector(values), 20, "control", "end", True) for values, y in (
                (read.keep, 327), (read.write, 368), (read.proposal, 409), (read.show, 450),
            )),
            Label(400, 487, "Old c is not mixed here", 17, "muted"),
        )),
        Box("kept", (280, 40, 180, 85), "cell", (2,), (
            Label(370, 66, "KEEP old", 22),
            Label(370, 102, vector(read.retained), 20, "cell", mono=True),
        )),
        Box("written", (560, 235, 180, 110), "cell", (2,), (
            Label(650, 262, "WRITE new", 22),
            Label(650, 294, vector(read.written), 20, "cell", mono=True),
            Label(650, 325, "WRITE x NEW", 16, "muted"),
        )),
        Box("new-cell", (750, 40, 330, 100), "cell", (3, 4, 6), (
            Label(972, 74, f"c{t}: new workspace", 22, "cell"),
            Label(972, 112, vector(read.new_cell), 22, "cell", mono=True),
        ), "cell"),
        Box("show", (760, 255, 320, 90), "hidden", (4,), (
            Label(920, 283, "Expose a bounded view", 22),
            Label(920, 319, f"SHOW x tanh(c{t})", 20, "hidden"),
        )),
        Box("new-hidden", (750, 415, 330, 100), "hidden", (4, 5, 6), (
            Label(972, 449, f"h{t}: new readout", 22, "hidden"),
            Label(972, 487, vector(read.new_hidden), 22, "hidden", mono=True),
        ), "hidden"),
        Box("prediction", (760, 560, 320, 174), "prediction", (5,), (
            Label(920, 590, f"Head reads h{t}, not c{t}", 22, "prediction"),
            Label(920, 622, f"guess {read.readout.prediction}; actual next {read.next_note}", 18, "muted"),
            *(Label(779, y, note, 20, "text" if note == read.next_note else "prediction", "start")
              for note, y in zip(NOTES, (655, 686, 717))),
            *(Label(1064, y, f"{probability:.4f}", 20, "prediction", "end", True)
              for probability, y in zip(read.readout.probabilities, (655, 686, 717))),
        )),
    ]
    wires = [
        Wire("cell-to-keep", "old-cell", "kept", ((226, 82), (274, 82)), "cell", (0, 2)),
        Wire("input-to-controls", "input", "controls", ((226, 295), (254, 295)), "text", (0, 1)),
        Wire("hidden-to-controls", "old-hidden", "controls", ((226, 462), (254, 462)), "hidden", (0, 1)),
        Wire("keep-control", "controls", "kept", ((370, 229), (370, 131)), "control", (1, 2)),
        Wire("write-control", "controls", "written", ((546, 388), (550, 388), (550, 290), (554, 290)), "control", (1, 2)),
        Wire("kept-to-add", "kept", "add", ((466, 82), (636, 82)), "cell", (2, 3)),
        Wire("written-to-add", "written", "add", ((650, 229), (650, 180), (660, 180), (660, 106)), "cell", (2, 3)),
        Wire("add-to-cell", "add", "new-cell", ((684, 82), (744, 82)), "cell", (3,)),
        Wire("cell-to-show", "new-cell", "show", ((970, 146), (970, 249)), "cell", (3, 4)),
        Wire("show-control", "controls", "show", ((546, 450), (744, 450), (744, 300), (754, 300)), "control", (1, 4)),
        Wire("show-to-hidden", "show", "new-hidden", ((970, 351), (970, 409)), "hidden", (4,)),
        Wire("hidden-to-head", "new-hidden", "prediction", ((970, 521), (970, 554)), "hidden", (5,)),
        Wire("carry-cell", "new-cell", "next-cell", ((1086, 90), (1098, 90)), "cell", (6,)),
        Wire("carry-hidden", "new-hidden", "next-hidden", ((1086, 465), (1098, 465)), "hidden", (6,)),
    ]
    labels = [
        Label(660, 91, "+", 29),
        Label(848, 170, f"c{t} continues", 18, "cell"),
        Label(848, 545, f"h{t} continues", 18, "hidden"),
        Label(400, 574, "c keeps the contents.", 22, "cell"),
        Label(400, 613, "h shows a controlled view.", 22, "hidden"),
        Label(400, 652, "The head predicts from h.", 22, "prediction"),
    ]
    return boxes, wires, labels


def svg_scene(read: LSTMRead, urls: dict[str, str]) -> str:
    boxes, wires, labels = scene(read)
    parts = ['<svg class="scene" viewBox="0 0 1110 740" role="group" aria-label="LSTM forward pass with separate cell, hidden and prediction routes">']
    parts.append("<title>Actual two-feature LSTM values. Only new h enters the prediction head; both states continue.</title>")
    parts.append("<defs>")
    for role, color in COLORS.items():
        parts.append(f'<marker id="arrow-{role}" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="8" markerHeight="8" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="{color}"/></marker>')
    parts.append("</defs>")
    for wire in wires:
        points = " ".join(f"{x},{y}" for x, y in wire.points)
        phases = " ".join(map(str, wire.phases))
        parts.append(
            f'<polyline id="{wire.name}" class="wire" points="{points}" data-from="{wire.source}" '
            f'data-to="{wire.target}" data-phases="{phases}" stroke="{COLORS[wire.role]}" '
            f'marker-end="url(#arrow-{wire.role})"/>'
        )
    parts.append('<circle id="add" cx="660" cy="82" r="18" fill="#20364f" stroke="#ffd172" stroke-width="2" data-phases="3"/>')

    def label_svg(label: Label) -> str:
        family = ' class="numeric"' if label.mono else ""
        return (f'<text x="{label.x}" y="{label.y}" font-size="{label.size}" fill="{COLORS[label.color]}" '
                f'text-anchor="{label.anchor}"{family}>{escape(label.text)}</text>')

    for box in boxes:
        x, y, width, height = box.bounds
        phases = " ".join(map(str, box.phases))
        parts.append(f'<g id="{box.name}" class="node" data-role="{box.role}" data-phases="{phases}">')
        parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="#20364f" stroke="{COLORS[box.role]}" stroke-width="2"/>')
        if box.art:
            parts.append(f'<image class="role-art" data-art="{box.art}" href="{urls[box.art]}" x="{x + 12}" y="{y + 13}" width="98" height="74"/>')
        parts.extend(label_svg(label) for label in box.labels)
        parts.append("</g>")
    for i, probability in enumerate(read.readout.probabilities):
        y = 643 + i * 31
        parts.append(f'<rect class="probability-track" x="828" y="{y}" width="150" height="12" rx="4" fill="#34455e"/>')
        parts.append(f'<rect class="probability" data-note="{NOTES[i]}" data-value="{probability}" x="828" y="{y}" width="{150 * probability}" height="12" rx="4" fill="#c5a3ff"/>')
    parts.extend(label_svg(label) for label in labels)
    parts.append("</svg>")
    return "".join(parts)


def _recording(reads: tuple[LSTMRead, ...]) -> dict:
    return {
        "version": 1, "notes": NOTES, "state_width": 2, "phases": PHASES,
        "head_weights": NOTE_HEAD_WEIGHTS, "steps": [asdict(read) for read in reads],
        "origin": "same native LSTMCell initialization and note codes as the lesson; initialized, not trained music",
    }


def recorded_data(path: Path = NOTEBOOK) -> dict:
    return _recording(record_reads(path))


PLAYER_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LSTM: two carried states, one prediction reader</title>
<style>
:root{font-family:system-ui,sans-serif;color:#edf3ff;background:#0d1b2e;color-scheme:dark;--cell:#ffd172;--hidden:#77dce5;--prediction:#c5a3ff}
*{box-sizing:border-box}body{margin:0}main{max-width:1120px;padding:14px;margin:auto}
h1{font-size:22px;margin:0 0 8px}p{line-height:1.35;margin:6px 0}.small{font-size:12px}.muted{color:#b0bed3}
.controls,.panel{background:#172941;border-radius:12px;padding:12px;margin-top:10px}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
button{font:600 13px system-ui;cursor:pointer;background:#20364f;color:#edf3ff;border:1px solid #7186a4;border-radius:8px;padding:7px 12px}
button[aria-pressed=true]{border-color:#ffd172;color:#ffd172}button:disabled{opacity:.55;cursor:default}
button:focus-visible{outline:3px solid #ffd172;outline-offset:3px}
.legend{font-size:13px;display:flex;gap:20px;flex-wrap:wrap}.cell-color{color:var(--cell)}.hidden-color{color:var(--hidden)}.prediction-color{color:var(--prediction)}
#phase{min-height:34px;font-size:13px;margin-top:8px;color:#edf3ff}
.viewport{overflow-x:auto}.scene{width:100%;height:auto;display:block}.scene text{font-family:system-ui,sans-serif}.scene text.numeric{font-family:ui-monospace,monospace}
.wire{fill:none;stroke-width:2;opacity:.38}.wire.active{opacity:1;stroke-width:3;stroke-dasharray:12 6;animation:flow .9s linear infinite}
main[data-playing=false] .wire.active{animation-play-state:paused}
.node.active>rect,#add.active{stroke-width:4}.node>rect{transition:stroke-width .15s}
main[data-phase=all] .wire{opacity:1;animation:none;stroke-dasharray:none}
@keyframes flow{to{stroke-dashoffset:-18}}
details{font-size:12px;margin-top:8px}summary{cursor:pointer;color:#b0bed3}code{display:block;font:12px ui-monospace,monospace;line-height:1.5;overflow-wrap:anywhere;padding:7px;background:#0d1b2e;border-radius:6px;margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:6px;border-bottom:1px solid #344d68}td{font-family:ui-monospace,monospace;overflow-wrap:anywhere}
#problem{color:#ffac9e}
@media(max-width:600px){main{padding:10px}.panel,.controls{padding:10px}.scene{min-width:780px}h1{font-size:20px}}
@media(prefers-reduced-motion:reduce){.wire.active{animation:none;stroke-dasharray:none}.node>rect{transition:none}}
</style></head><body>
<main data-player="lstm-forward-walkthrough" data-phase="all" data-step="2" data-playing="false">
<h1>Two carried states. One prediction reader.</h1>
<p class="small muted">Current note + old h choose the controls. Old c supplies the workspace. The note head reads new h, not c.</p>
<section class="controls" aria-label="LSTM walkthrough controls">
 <div class="row"><div id="note-buttons" role="group" aria-label="Choose a note read">
  <button type="button" data-note="0">D3</button><button type="button" data-note="1">A3</button><button type="button" data-note="2">D4</button>
 </div><div><button id="play" type="button">Play phrase</button> <button id="pause" type="button">Pause</button> <button id="step" type="button">Next stage</button></div></div>
 <div class="legend"><span class="cell-color">c: workspace</span><span class="hidden-color">h: readout + context</span><span class="prediction-color">head: note probabilities</span><span class="muted">NEW: proposed content</span></div>
 <p id="phase" role="status"></p>
</section>
<p id="problem" role="alert" hidden></p>
<section class="panel" aria-label="Measured LSTM forward diagram"><div id="scene" class="viewport"></div></section>
<section class="panel" aria-label="LSTM role checks">
 <p class="small"><strong>Carry both states onward.</strong> Only h branches into the prediction head. These are two-feature states, not stored note names.</p>
 <details><summary>Open the arithmetic for this read</summary>
  <code id="cell-formula"></code><code id="hidden-formula"></code><code id="head-formula"></code>
  <p class="muted">KEEP/WRITE/SHOW and the proposal are computed by the same native LSTM weights at every note. In this standard cell, old c does not enter the gate projections.</p>
 </details>
 <details><summary>Prove that c is not fed directly to the note head</summary>
  <p class="muted">Isolated check: close only SHOW for this read, with its incoming values fixed.</p>
  <table><thead><tr><th></th><th>Normal</th><th>SHOW closed</th></tr></thead><tbody id="proof"></tbody></table>
  <p class="muted">c is unchanged, but h is zero. This bias-free head then gives equal note scores. The content is still stored; it is not exposed to the head.</p>
 </details>
</section>
<p class="small muted">Rounded native LSTMCell values; exact arithmetic is available above. The head is hand-picked: initialized mechanics, not trained music. MAI role art; labels stay outside the images.</p>
</main>
<script id="recorded-lstm" type="application/json">__DATA__</script>
<script>
(() => {
 const data=JSON.parse(document.getElementById("recorded-lstm").textContent);
 const root=document.querySelector("main"),scene=document.getElementById("scene");
 const playButton=document.getElementById("play"),pauseButton=document.getElementById("pause"),stepButton=document.getElementById("step");
 if(data.version!==1||data.state_width!==2||!Array.isArray(data.steps)||data.steps.length!==3||!Array.isArray(data.scenes)||data.scenes.length!==3){
  const alert=document.getElementById("problem");alert.hidden=false;alert.textContent="The recorded LSTM walkthrough is invalid. Regenerate it.";
  document.querySelectorAll("button").forEach(button=>button.disabled=true);throw new Error("Invalid LSTM forward recording.");
 }
 const reduced=window.matchMedia("(prefers-reduced-motion: reduce)").matches;
 const f=value=>Number.isInteger(value)?String(value):value.toFixed(4),v=values=>"["+values.map(f).join(", ")+"]";
 let selected=1,cursor=7,timer=null;
 function stop(){clearTimeout(timer);timer=null;root.dataset.playing="false";pauseButton.textContent=reduced?"Reduced motion":"Resume";}
 function tableRow(label,before,after){const row=document.createElement("tr");for(const value of[label,v(before),v(after)]){const cell=document.createElement("td");cell.textContent=value;row.append(cell);}return row;}
 function render(index,phase){
  selected=index;const read=data.steps[index],t=read.index;
  root.dataset.step=String(t);root.dataset.phase=String(phase);
  scene.innerHTML=data.scenes[index];
  scene.querySelectorAll("[data-phases]").forEach(node=>node.classList.toggle("active",phase!=="all"&&node.dataset.phases.split(" ").includes(String(phase))));
  document.querySelectorAll("#note-buttons button[data-note]").forEach(button=>button.setAttribute("aria-pressed",String(Number(button.dataset.note)===index)));
  document.getElementById("phase").textContent=phase==="all"?"After "+read.note+": c"+t+" and h"+t+" are ready; the head reads h"+t+".":data.phases[phase];
  document.getElementById("cell-formula").textContent="c"+t+" = kept + written = "+v(read.retained)+" + "+v(read.written)+" = "+v(read.new_cell);
  document.getElementById("hidden-formula").textContent="h"+t+" = SHOW * tanh(c"+t+") = "+v(read.show)+" * tanh("+v(read.new_cell)+") = "+v(read.new_hidden);
  document.getElementById("head-formula").textContent="scores = W_out * h"+t+" = "+v(read.readout.scores)+"; softmax[D3,A3,D4] = "+v(read.readout.probabilities);
  document.getElementById("proof").replaceChildren(
   tableRow("new c",read.new_cell,read.closed_show_cell),tableRow("new h",read.new_hidden,read.closed_show_hidden),
   tableRow("note probabilities",read.readout.probabilities,read.closed_show_readout.probabilities));
  root.dataset.cell=JSON.stringify(read.new_cell);root.dataset.hidden=JSON.stringify(read.new_hidden);
  root.dataset.probabilities=JSON.stringify(read.readout.probabilities);
  stepButton.disabled=cursor>=data.steps.length*data.phases.length;
 }
 function finish(){stop();cursor=data.steps.length*data.phases.length;render(data.steps.length-1,"all");pauseButton.textContent="Replay";}
 function advance(){if(cursor>=data.steps.length*data.phases.length){finish();return;}const position=cursor++;render(Math.floor(position/data.phases.length),position%data.phases.length);}
 function tick(){if(cursor>=data.steps.length*data.phases.length){finish();return;}advance();timer=setTimeout(tick,650);}
 function play(){stop();cursor=0;if(reduced){render(selected,"all");return;}root.dataset.playing="true";pauseButton.textContent="Pause";tick();}
 document.querySelectorAll("#note-buttons button[data-note]").forEach(button=>button.addEventListener("click",()=>{stop();selected=Number(button.dataset.note);cursor=selected*data.phases.length;render(selected,"all");}));
 playButton.addEventListener("click",play);
 pauseButton.addEventListener("click",()=>{if(timer!==null){stop();return;}if(cursor>=data.steps.length*data.phases.length){play();return;}root.dataset.playing="true";pauseButton.textContent="Pause";tick();});
 stepButton.addEventListener("click",()=>{stop();advance();});
 document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
 window.addEventListener("pagehide",stop);
 render(selected,"all");
 if(reduced){playButton.disabled=true;pauseButton.disabled=true;pauseButton.textContent="Reduced motion";}else play();
})();
</script></body></html>"""


def render_player(path: Path = NOTEBOOK) -> str:
    reads = record_reads(path)
    data = _recording(reads)
    images = role_images()
    urls = {}
    for role, image in images.items():
        buffer = BytesIO()
        image.save(buffer, format="WEBP", lossless=True, method=6)
        urls[role] = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    data["scenes"] = [svg_scene(read, urls) for read in reads]
    return PLAYER_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False)) + "\n"


def read_recording(path: Path = PLAYER) -> dict:
    matches = list(RECORDING.finditer(path.read_text(encoding="utf-8")))
    if len(matches) != 1:
        raise ValueError("Expected one recorded-lstm data block.")
    return json.loads(matches[0][1])


def export_player(path: Path = PLAYER) -> None:
    raw = path.read_bytes() if path.exists() else None
    encoded = render_player().encode("utf-8")
    if raw == encoded:
        return
    with NamedTemporaryFile(dir=path.parent, prefix=".lstm-forward-", suffix=".tmp", delete=False) as staged:
        staged_path = Path(staged.name)
        staged.write(encoded)
    try:
        if (path.read_bytes() if path.exists() else None) != raw:
            raise RuntimeError("The LSTM walkthrough changed while it was being regenerated.")
        staged_path.replace(path)
    finally:
        staged_path.unlink(missing_ok=True)


def export_storyboard(path: Path = STORYBOARD) -> None:
    read = record_reads()[1]
    boxes, wires, labels = scene(read)
    images = role_images()
    figure, axis = plt.subplots(figsize=(13, 9.1), facecolor=COLORS["background"])
    figure.subplots_adjust(left=0.015, right=0.985, top=0.925, bottom=0.035)
    axis.set(xlim=(0, 1110), ylim=(740, 0))
    axis.axis("off")
    for wire in wires:
        for start, end in zip(wire.points[:-2], wire.points[1:-1]):
            axis.plot((start[0], end[0]), (start[1], end[1]), color=COLORS[wire.role], linewidth=1.4)
        axis.annotate("", wire.points[-1], wire.points[-2],
                      arrowprops={"arrowstyle": "->", "color": COLORS[wire.role], "lw": 1.4})
    for box in boxes:
        x, y, width, height = box.bounds
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=8",
                                     facecolor=COLORS["panel"], edgecolor=COLORS[box.role], linewidth=1.5))
        if box.art:
            axis.imshow(images[box.art], extent=(x + 12, x + 110, y + 87, y + 13), aspect="auto", zorder=2)
        for label in box.labels:
            axis.text(label.x, label.y, label.text, color=COLORS[label.color], fontsize=label.size * 0.82,
                      ha={"middle": "center", "start": "left", "end": "right"}[label.anchor],
                      family="monospace" if label.mono else "sans-serif")
    axis.add_patch(plt.Circle((660, 82), 18, facecolor=COLORS["panel"], edgecolor=COLORS["cell"], linewidth=1.5))
    for label in labels:
        axis.text(label.x, label.y, label.text, color=COLORS[label.color], fontsize=label.size * 0.82,
                  ha={"middle": "center", "start": "left", "end": "right"}[label.anchor])
    for i, probability in enumerate(read.readout.probabilities):
        axis.add_patch(FancyBboxPatch((828, 643 + i * 31), 150 * probability, 12,
                                     boxstyle="round,pad=0,rounding_size=3", facecolor=COLORS["prediction"], edgecolor="none"))
    figure.suptitle("A3 + old h choose HOW. Old c supplies WHAT. Only new h feeds the note head.",
                    color=COLORS["text"], fontsize=15)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [label for label in axis.texts if label.get_text()]
    for offset, label in enumerate(texts):
        bounds = label.get_window_extent(renderer)
        for other in texts[offset + 1:]:
            if bounds.overlaps(other.get_window_extent(renderer)):
                plt.close(figure)
                raise RuntimeError(f"LSTM labels overlap: {label.get_text()} / {other.get_text()}")
    for image in axis.images:
        left, right, bottom, top = image.get_extent()
        points = axis.transData.transform(((left, bottom), (right, top)))
        bounds = Bbox.from_extents(min(points[:, 0]), min(points[:, 1]), max(points[:, 0]), max(points[:, 1]))
        for label in axis.texts:
            if label.get_window_extent(renderer).overlaps(bounds):
                plt.close(figure)
                raise RuntimeError(f"LSTM text overlaps MAI artwork: {label.get_text()}")
    figure.savefig(path, dpi=120, facecolor=COLORS["background"], metadata={
        "TraceOrigin": "native two-feature LSTMCell; A3 after D3; note head reads new h only",
        "HeadInput": "new_hidden", "StateWidth": "2",
    })
    plt.close(figure)


if __name__ == "__main__":
    export_player()
    export_storyboard()
    print("Exported three native LSTM reads, c/h routing, h-only predictions and paired MAI role art.")
