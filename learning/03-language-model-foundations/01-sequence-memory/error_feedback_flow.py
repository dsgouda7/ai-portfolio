"""Record a named-note RNN, its chain-rule arithmetic, and one real SGD step."""

import base64
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox
from PIL import Image
import torch
from torch.nn import functional as F

from backprop_memory_flow import NOTES, NOTE_HEAD_WEIGHTS


HERE = Path(__file__).resolve().parent
PLAYER = HERE / "images" / "01a-error-feedback-tracker.html"
STORYBOARD = HERE / "images" / "01a-note-gradient-chain.png"
CARD_ART = HERE / "images" / "01a-gradient-memory-card.png"
NOTE_INPUTS = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
WRITING_WEIGHTS = (0.001, -0.0005)
PREDICTION_WEIGHTS = tuple(row[1] for row in NOTE_HEAD_WEIGHTS)
TARGET_NOTE = "D3"
LEARNING_RATE = 0.02
PROBABILITY_DECIMALS = 10
MEMORY_NUDGE = 0.00001
GAINS = (0.8, 1.2)
SCALE_GAINS = tuple(value / 100 for value in range(60, 131))
REPETITIONS = (1, 7)
RECORDING = re.compile(
    r'(<script id="recorded-cases" type="application/json">)\s*(.*?)\s*(</script>)',
    re.DOTALL,
)


@dataclass(frozen=True)
class Parameters:
    writing: tuple[float, ...]
    memory: float
    prediction: tuple[float, ...]


@dataclass(frozen=True)
class NoteStep:
    index: int
    note: str
    inputs: tuple[float, ...]
    previous: float
    input_part: float
    combined: float
    state: float
    slope: float
    factor: float
    feedback: float
    delta: float
    previous_feedback: float
    writing_contribution: tuple[float, ...]
    memory_contribution: float
    memory_part: float
    next_note: str
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    prediction: str
    local_loss: float
    graded: bool
    loss_if_memory_lower: float
    loss_if_memory_higher: float
    measured_response: float


@dataclass(frozen=True)
class ErrorFeedbackProbe:
    gain: float
    repeats: int
    steps: tuple[NoteStep, ...]
    weights: Parameters
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    score_errors: tuple[float, ...]
    prediction: str
    loss: float
    weight_gradients: Parameters
    updated_weights: Parameters
    after_states: tuple[float, ...]
    after_probabilities: tuple[float, ...]
    after_prediction: str
    after_loss: float


def record_probe(*, gain: float = 0.8, repeats: int = 1) -> ErrorFeedbackProbe:
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain):
        raise ValueError("The previous-memory multiplier must be a finite number.")
    if not 0.6 <= gain <= 1.3:
        raise ValueError("Choose a previous-memory multiplier between 0.6 and 1.3.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats not in REPETITIONS:
        raise ValueError("Repeat the three-note phrase either once or seven times.")
    inputs = torch.tensor(NOTE_INPUTS * repeats, dtype=torch.float64)
    writing = torch.tensor(WRITING_WEIGHTS, dtype=torch.float64, requires_grad=True)
    memory = torch.tensor(gain, dtype=torch.float64, requires_grad=True)
    head = torch.tensor(PREDICTION_WEIGHTS, dtype=torch.float64, requires_grad=True)
    states = [torch.tensor(0.0, dtype=torch.float64, requires_grad=True)]
    input_parts, combined = [], []
    for current in inputs:
        input_parts.append(writing @ current)
        combined.append(input_parts[-1] + memory * states[-1])
        states.append(torch.tanh(combined[-1]))
    step_scores = [head * state for state in states[1:]]
    step_probabilities = [values.softmax(0) for values in step_scores]
    next_indices = [(offset + 1) % len(NOTES) for offset in range(len(inputs))]
    step_losses = [
        F.cross_entropy(values.unsqueeze(0), torch.tensor([next_index]))
        for values, next_index in zip(step_scores, next_indices, strict=True)
    ]
    scores = step_scores[-1]
    target = torch.tensor([NOTES.index(TARGET_NOTE)])
    loss = step_losses[-1]
    probabilities = step_probabilities[-1]
    score_errors = probabilities.detach().clone()
    score_errors[target.item()] -= 1
    gradients = torch.autograd.grad(loss, (*states, writing, memory, head))
    feedback = gradients[:-3]
    steps = []
    for offset, (current, old, state, signal) in enumerate(
        zip(inputs, states[:-1], states[1:], feedback[1:], strict=True)
    ):
        slope = 1 - state.item() ** 2
        delta = signal.item() * slope
        nearby_losses = []
        with torch.no_grad():
            for change in (-MEMORY_NUDGE, MEMORY_NUDGE):
                carried = state.detach() + change
                for following in inputs[offset + 1:]:
                    carried = torch.tanh(writing @ following + memory * carried)
                nearby_losses.append(
                    F.cross_entropy((head * carried).unsqueeze(0), target).item()
                )
        lower_loss, higher_loss = nearby_losses
        steps.append(NoteStep(
            index=offset + 1, note=NOTES[offset % len(NOTES)], inputs=tuple(current.tolist()),
            previous=old.item(), input_part=input_parts[offset].item(), combined=combined[offset].item(),
            state=state.item(), slope=slope, factor=gain * slope, feedback=signal.item(),
            delta=delta, previous_feedback=feedback[offset].item(),
            writing_contribution=tuple((delta * current).tolist()), memory_contribution=delta * old.item(),
            memory_part=gain * old.item(), next_note=NOTES[next_indices[offset]],
            scores=tuple(step_scores[offset].detach().tolist()),
            probabilities=tuple(step_probabilities[offset].detach().tolist()),
            prediction=NOTES[int(step_scores[offset].argmax())],
            local_loss=step_losses[offset].item(), graded=offset == len(inputs) - 1,
            loss_if_memory_lower=lower_loss, loss_if_memory_higher=higher_loss,
            measured_response=(higher_loss - lower_loss) / (2 * MEMORY_NUDGE),
        ))
    parameter_gradients = Parameters(
        tuple(gradients[-3].tolist()), gradients[-2].item(), tuple(gradients[-1].tolist()),
    )
    optimizer = torch.optim.SGD((writing, memory, head), lr=LEARNING_RATE)
    for parameter, gradient in zip((writing, memory, head), gradients[-3:], strict=True):
        parameter.grad = gradient.detach().clone()
    optimizer.step()
    with torch.no_grad():
        after = torch.tensor(0.0, dtype=torch.float64)
        after_states = [after.item()]
        for current in inputs:
            after = torch.tanh(writing @ current + memory * after)
            after_states.append(after.item())
        after_scores = head * after
        after_probabilities = after_scores.softmax(0)
        after_loss = F.cross_entropy(after_scores.unsqueeze(0), target).item()
    if not all(math.isfinite(value) for value in (
        loss.item(), after_loss, *(signal.item() for signal in feedback),
        *(value for step in steps for value in (
            step.loss_if_memory_lower, step.loss_if_memory_higher, step.measured_response,
        )),
    )):
        raise RuntimeError("The recorded note model produced non-finite loss or gradients.")
    return ErrorFeedbackProbe(
        gain, repeats, tuple(steps), Parameters(WRITING_WEIGHTS, gain, PREDICTION_WEIGHTS),
        tuple(scores.detach().tolist()), tuple(probabilities.detach().tolist()),
        tuple(score_errors.tolist()), NOTES[int(scores.argmax())], loss.item(),
        parameter_gradients,
        Parameters(tuple(writing.tolist()), memory.item(), tuple(head.tolist())),
        tuple(after_states), tuple(after_probabilities.tolist()),
        NOTES[int(after_scores.argmax())], after_loss,
    )


def recorded_data() -> dict:
    return {
        "version": 6, "notes": NOTES, "target_note": TARGET_NOTE, "memory_nudge": MEMORY_NUDGE,
        "learning_rate": LEARNING_RATE, "probability_decimals": PROBABILITY_DECIMALS,
        "routes": [asdict(record_probe(gain=gain, repeats=repeats))
                   for repeats in REPETITIONS for gain in SCALE_GAINS],
    }


def read_recording(path: Path = PLAYER) -> dict:
    matches = list(RECORDING.finditer(path.read_text(encoding="utf-8")))
    if len(matches) != 1:
        raise ValueError("Expected one recorded-cases data block in the feedback player.")
    return json.loads(matches[0][2])


PLAYER_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One note error, a chain of gradients</title>
<style>
:root { color-scheme: dark; font-family: system-ui,sans-serif; color:#edf3ff; background:#0d1b2e; --fixed:#b0bed3; --gold:#ffd172; --cyan:#77dce5; --signal:#77dce5; }
* { box-sizing:border-box; }
body { margin:0; }
main { max-width:1100px; margin:auto; padding:14px; }
h1 { font-size:22px; margin:0 0 8px; }
h2 { font-size:16px; margin:0 0 8px; }
p { margin:6px 0; line-height:1.35; }
.small { font-size:12px; }
.muted,.fixed { color:var(--fixed); }
.computed { color:var(--cyan); }
.adjustable { color:var(--gold); }
.panel { background:#172941; border-radius:12px; padding:12px; margin-top:10px; }
.row { display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap; }
button,input { font:inherit; }
button { cursor:pointer; color:#edf3ff; background:#20364f; border:1px solid #536781; border-radius:8px; padding:6px 10px; }
button:focus-visible,input:focus-visible { outline:3px solid var(--gold); outline-offset:3px; }
#weight-value { color:var(--gold); font:700 20px ui-monospace,monospace; }
#weight-scale { width:100%; accent-color:var(--gold); cursor:ew-resize; margin:10px 0 4px; }
.scale-labels { display:flex; justify-content:space-between; font-size:12px; color:var(--fixed); }
#gradient-status { color:var(--signal); font-size:13px; font-weight:700; }
.legend { display:flex; gap:12px; font-size:12px; flex-wrap:wrap; }
.weights { display:flex; gap:8px 18px; flex-wrap:wrap; font-size:12px; margin-top:8px; }
code { font:13px ui-monospace,monospace; overflow-wrap:anywhere; }
.cells { overflow-x:auto; padding:4px 0; }
.trace { display:block; width:100%; max-width:950px; height:auto; margin:auto; }
.cells.long .trace { max-width:none; }
.trace text { font-family:system-ui,sans-serif; text-anchor:middle; fill:var(--fixed); }
.trace .note { font-size:22px; font-weight:700; fill:#edf3ff; }
.trace .input { font:17px ui-monospace,monospace; }
.cell { cursor:pointer; outline:none; }
.rnn-block { fill:#20364f; stroke:#7186a4; stroke-width:2; }
.cell[aria-pressed="true"] .rnn-block,.cell:focus-visible .rnn-block { stroke:var(--gold); stroke-width:4; }
main[data-phase="backward"] .cell[aria-pressed="true"] .rnn-block { stroke:var(--signal); }
.trace .block-title { font-size:20px; font-weight:700; fill:#edf3ff; }
.trace .block-label { font-size:13px; }
.trace .block-weights { font:16px ui-monospace,monospace; }
.trace .recipe,.trace .memory-product { font:700 20px ui-monospace,monospace; fill:var(--gold); }
.trace .operation { font-size:16px; }
.trace .branch-label { font-size:16px; fill:var(--cyan); }
.prediction-block { fill:#14263f; stroke:#7186a4; stroke-width:2; stroke-dasharray:5 4; }
.prediction-block.graded { stroke:var(--gold); stroke-width:3; stroke-dasharray:none; }
.trace .head-title { font-size:20px; font-weight:700; fill:#edf3ff; }
.trace .head-guess { font-size:16px; }
.trace .head-probability { font:18px ui-monospace,monospace; fill:var(--cyan); }
.trace .head-loss { font:18px ui-monospace,monospace; }
.trace .head-loss.graded,.trace .head-status.graded { fill:var(--gold); }
.trace .head-status { font-size:15px; }
main[data-phase="head"] .cell[aria-pressed="true"] .prediction-block { stroke:var(--cyan); stroke-width:4; }
main[data-phase="head"] .cell[aria-pressed="true"] .prediction-block.graded { stroke:var(--gold); }
.trace .state-name { font-size:19px; fill:var(--cyan); }
.trace .state-value { font:600 18px ui-monospace,monospace; fill:var(--cyan); }
.trace .fixed { fill:var(--fixed); }
.trace .gradient-name,.trace .gradient-context { font-size:14px; fill:var(--signal); }
.trace .gradient { font:700 20px ui-monospace,monospace; fill:var(--signal); }
.trace .factor { font:18px ui-monospace,monospace; fill:var(--signal); }
.backward-wire { fill:var(--signal); opacity:0.65; }
main[data-phase="backward"] .cell[aria-pressed="true"] .backward-wire { opacity:1; }
details { margin-top:6px; font-size:12px; }
summary { cursor:pointer; color:var(--fixed); }
.formula { display:block; padding:6px 8px; background:#0d1b2e; border-radius:6px; margin-top:5px; font:13px ui-monospace,monospace; overflow-wrap:anywhere; line-height:1.4; }
.mechanics { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.memory-probe { padding:8px; border:1px solid #536781; border-radius:8px; }
.memory-probe code { display:block; margin-top:4px; color:var(--cyan); }
.label { font-size:12px; color:var(--fixed); }
#backward-formula { border-left:3px solid var(--signal); }
#phase { color:var(--gold); font-size:12px; }
.ratio { font:600 15px ui-monospace,monospace; color:var(--signal); }
table { width:100%; border-collapse:collapse; font-size:12px; }
th,td { text-align:left; padding:6px; border-bottom:1px solid #344d68; }
td:not(:first-child) { font-family:ui-monospace,monospace; overflow-wrap:anywhere; }
#problem { color:#ffac9e; }
@media(max-width:600px) {
 main { padding:10px; } .panel { padding:10px; } h1 { font-size:20px; }
 .mechanics { grid-template-columns:1fr; }
 code,.formula { font-size:12px; } th,td { padding:5px 3px; font-size:11px; }
}
@media(max-width:500px) { .trace { min-width:600px; } }
</style>
</head>
<body>
<main data-player="error-feedback-tracker" data-phase="inspect">
  <h1>One prediction error. Trace its influence backward.</h1>
  <p class="small muted">D3 &rarr; A3 &rarr; D4 &rarr; D3. Each head predicts the next note; only the ending error drives backward.</p>
  <section class="panel" aria-label="Controlled shared weights">
    <div class="row">
      <label for="weight-scale">Previous-memory multiplier (recurrent weight) <output id="weight-value">0.80</output></label>
      <output id="gradient-status" aria-live="polite">Fading feedback</output>
    </div>
    <input id="weight-scale" type="range" min="0.6" max="1.3" step="0.01" value="0.8" aria-label="Previous-memory multiplier (recurrent weight)">
    <div class="scale-labels"><span>0.60</span><span>1.30</span></div>
    <div class="row">
      <label class="small"><input id="long-phrase" type="checkbox"> Repeat the phrase 7 times (21 notes)</label>
      <div><button id="replay" type="button">Replay</button> <button id="pause" type="button">Pause</button></div>
    </div>
    <details>
      <summary>Shared weights</summary>
      <div class="weights" id="model-block">
        <span class="fixed">Input multipliers: <code id="writing-weight">[0.001, -0.0005]</code></span>
        <span class="fixed">Prediction multipliers: <code id="prediction-weight">[-0.5, 1, 0.25]</code></span>
        <span class="adjustable">Recurrent weight m: <code id="memory-weight">0.80</code></span>
      </div>
      <p class="small muted">Chosen starting weights, not fresh random weights per note. Gold changes; grey stays fixed during the sweep.</p>
    </details>
  </section>
  <p id="problem" role="alert" hidden></p>
  <section class="panel" aria-label="Unrolled note cells">
    <h2>Read forward. Follow the prediction branches.</h2>
    <div id="cells" class="cells" role="group" aria-label="Inspect each note and saved state"></div>
    <p class="small muted" id="loss-scope"></p>
  </section>
  <section class="panel" aria-label="Live per-cell forward and backward arithmetic">
    <div class="row"><h2 id="selected-note"></h2><span id="phase" role="status"></span></div>
    <p class="small">Freeze the weights. Nudge only this saved memory, then rerun the remaining reads and ending prediction.</p>
    <div class="mechanics" aria-label="Two measured memory interventions">
      <div class="memory-probe"><span class="label">Slightly lower memory</span><code id="lower-memory"></code></div>
      <div class="memory-probe"><span class="label">Slightly higher memory</span><code id="higher-memory"></code></div>
    </div>
    <code class="formula">Error response = ending-error change / saved-memory change</code>
    <code class="formula" id="response-arithmetic"></code>
    <p class="small computed" id="memory-nudge"></p>
    <p class="small muted">This nearby slope estimates the gradient g. Backprop computes the tiny-change limit by chaining local rates, not trial reruns. It adjusts weights, not the saved h.</p>
    <code class="formula" id="backward-formula"></code>
    <code class="formula" id="factor-formula"></code>
    <p class="small" id="why"></p>
    <p class="small">Opening / final feedback: <output id="gradient-ratio" class="ratio"></output>&times;.
      <span id="distance-note"></span></p>
    <details>
      <summary>Full cell arithmetic</summary>
      <p class="small muted" id="response-comparison"></p>
      <code class="formula" id="forward-formula"></code>
      <code class="formula" id="state-formula"></code>
      <code class="formula" id="chain-formula"></code>
      <code class="formula" id="contribution-formula"></code>
      <code class="formula" id="head-formula"></code>
      <p class="small muted" id="head-scope"></p>
      <code class="formula" id="seed-formula"></code>
      <p class="small muted">Math shorthand: L = prediction error (cross-entropy loss);
        g = ending-error response to a tiny change in saved h. These are rates, not weight updates.
        h0 is fixed, not a learned weight.</p>
    </details>
  </section>
  <section class="panel" aria-label="One shared SGD update and new prediction">
    <h2>Sum the gradients. Update once.</h2>
    <p class="small computed" id="after-update"></p>
    <details>
      <summary>New weights and gradient sums</summary>
      <p class="small">New weight = old weight - <span id="learning-rate"></span> &times; summed gradient.</p>
      <table aria-label="Actual gradients and updated shared weights">
        <thead><tr><th>Shared weights</th><th>Before</th><th>Gradient</th><th>After one SGD step</th></tr></thead>
        <tbody id="update-rows"></tbody>
      </table>
      <code class="formula" id="sum-formula"></code>
      <code class="formula" id="update-formula"></code>
      <p class="small computed" id="after-probability"></p>
      <p class="small muted">One SGD preview, not continuous training. The after-result reruns the same phrase with new weights;
        saved cards stay unchanged. Moving the slider resets the trial.</p>
    </details>
  </section>
  <p class="small muted">Same two-number note codes as the earlier reader; one hidden feature instead of two.
    Only the final prediction is graded. The first prediction stays fixed because h0 = 0.
    Actual autograd values; MAI icons mark saved memory.</p>
  <noscript>Use the notebook's paired static figure if this viewer blocks scripts.</noscript>
</main>
<script id="recorded-cases" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("recorded-cases").textContent);
  const root = document.querySelector("main");
  const scale = document.getElementById("weight-scale");
  const extended = document.getElementById("long-phrase");
  const cells = document.getElementById("cells");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cardArt = "__CARD_ART__";
  const memoryNudge = data.memory_nudge;
  if (data.version !== 6 || !Array.isArray(data.routes) || data.routes.length !== 142 ||
      !Number.isFinite(memoryNudge) || memoryNudge <= 0 ||
      !Number.isInteger(data.probability_decimals) || data.probability_decimals < 10 || data.probability_decimals > 15) {
    const problem = document.getElementById("problem");
    problem.hidden = false;
    problem.textContent = "The note-gradient recording is invalid. Regenerate this player.";
    scale.disabled = true;
    extended.disabled = true;
    throw new Error("Invalid note-gradient recording.");
  }
  let route, selected = 2, timer = null, stages = [], position = 0;
  const f = value => Number.isInteger(value) ? String(value) : Math.abs(value) < 0.0001 ? value.toExponential(3) : value.toFixed(6);
  const vector = values => "[" + values.map(f).join(", ") + "]";
  const probability = value => value.toFixed(data.probability_decimals);
  const text = (id, value) => { document.getElementById(id).textContent = value; };
  function svgNode(tag, attrs, value) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, content] of Object.entries(attrs)) node.setAttribute(name, content);
    if (value !== undefined) node.textContent = value;
    return node;
  }
  function drawCells() {
    const width = route.steps.length * 270 + 105;
    const svg = svgNode("svg", {viewBox:"0 0 " + width + " 560", class:"trace", role:"group",
      "aria-label":"Weighted RNN blocks; memory flows right and ending-error responses flow left"});
    svg.append(svgNode("title", {}, "Text stays outside card images. Backward ribbon widths use a compressed logarithmic scale."));
    if (extended.checked) svg.style.width = (route.steps.length * 210 + 70) + "px";
    const defs = svgNode("defs", {});
    defs.append(svgNode("image", {id:"memory-card-art", href:cardArt, width:60, height:34}));
    for (const [id, color] of [["input-arrow", "#b0bed3"], ["memory-arrow", "#77dce5"], ["loss-arrow", "#ffd172"]]) {
      const marker = svgNode("marker", {id, viewBox:"0 0 8 8", refX:8, refY:4,
        markerWidth:8, markerHeight:8, orient:"auto", markerUnits:"userSpaceOnUse"});
      marker.append(svgNode("path", {d:"M0 0L8 4L0 8Z", fill:color}));
      defs.append(marker);
    }
    svg.append(defs);
    svg.append(svgNode("path", {d:"M6 160H84", stroke:"#b0bed3", "stroke-width":2, "marker-end":"url(#input-arrow)"}));
    svg.append(svgNode("text", {x:45, y:139, class:"state-name fixed"}, "h0"));
    svg.append(svgNode("text", {x:45, y:181, class:"state-value fixed"}, "0"));
    svg.append(svgNode("use", {href:"#memory-card-art", x:15, y:199, opacity:0.55, class:"memory-icon"}));
    const compactWeights = route.weights.writing.map(value => String(value).replace(/^(-?)0\./, "$1.")).join(",");
    for (const s of route.steps) {
      const cell = svgNode("g", {class:"cell", transform:"translate(" + (s.index - 1) * 270 + " 0)",
        role:"button", tabindex:0, "aria-pressed":"false",
        "aria-label":s.note + " read " + s.index + ", h" + (s.index - 1) + " to h" + s.index,
        "aria-describedby":"head-target-" + s.index + " head-probability-" + s.index +
          " head-loss-" + s.index + " head-status-" + s.index});
      cell.dataset.step = String(s.index);
      cell.dataset.feedback = String(s.feedback);
      cell.dataset.state = String(s.state);
      cell.dataset.factor = String(s.factor);
      cell.dataset.memoryPart = String(s.memory_part);
      cell.dataset.nextNote = s.next_note;
      cell.dataset.loss = String(s.local_loss);
      cell.dataset.probability = String(s.probabilities[data.notes.indexOf(s.next_note)]);
      cell.dataset.graded = String(s.graded);
      cell.append(
        svgNode("text", {x:180, y:24, class:"note"}, s.note),
        svgNode("text", {x:180, y:49, class:"input"}, "x" + s.index + "=" + vector(s.inputs)),
        svgNode("path", {d:"M180 58V86", stroke:"#b0bed3", "stroke-width":2, "marker-end":"url(#input-arrow)"}),
        svgNode("rect", {x:90, y:92, width:180, height:142, rx:10, class:"rnn-block"}),
        svgNode("text", {x:180, y:112, class:"block-title"}, "RNN"),
        svgNode("text", {x:180, y:132, class:"block-label"}, "Input multipliers"),
        svgNode("text", {x:180, y:151, class:"block-weights"}, "[" + compactWeights + "]"),
        svgNode("text", {x:180, y:174, class:"recipe"}, "h" + (s.index - 1) + " \u00d7 " + route.gain.toFixed(2)),
        svgNode("text", {x:180, y:199, class:"memory-product"}, "= " + f(s.memory_part)),
        svgNode("text", {x:180, y:226, class:"operation"}, "mix \u2192 tanh"),
        svgNode("path", {d:"M276 160H354", stroke:"#77dce5", "stroke-width":2, "marker-end":"url(#memory-arrow)"}),
        svgNode("text", {x:315, y:139, class:"state-name"}, "h" + s.index),
        svgNode("text", {x:315, y:181, class:"state-value"}, f(s.state)),
        svgNode("use", {href:"#memory-card-art", x:285, y:199, class:"memory-icon"}),
        svgNode("path", {d:"M180 240V258", stroke:"#77dce5", "stroke-width":2, "marker-end":"url(#memory-arrow)"}),
        svgNode("text", {x:204, y:253, class:"branch-label"}, "h" + s.index),
        svgNode("rect", {x:70, y:264, width:220, height:136, rx:10,
          class:"prediction-block" + (s.graded ? " graded" : "")}),
        svgNode("text", {x:180, y:290, class:"head-title", id:"head-target-" + s.index}, "Actual next: " + s.next_note),
        svgNode("text", {x:180, y:313, class:"head-guess"}, "guess " + s.prediction),
        svgNode("text", {x:180, y:339, class:"head-probability", id:"head-probability-" + s.index},
          "P(" + s.next_note + ")=" + probability(s.probabilities[data.notes.indexOf(s.next_note)])),
        svgNode("text", {x:180, y:364, class:"head-loss" + (s.graded ? " graded" : ""), id:"head-loss-" + s.index},
          "Error=" + probability(s.local_loss)),
        svgNode("text", {x:180, y:388, class:"head-status" + (s.graded ? " graded" : ""), id:"head-status-" + s.index},
          s.graded ? "backprop source" : "preview only"),
        svgNode("text", {x:315, y:438, class:"gradient-name"}, "Ending-error"),
        svgNode("text", {x:315, y:458, class:"gradient-context"}, "response at h" + s.index),
        svgNode("text", {x:315, y:484, class:"gradient"}, f(s.feedback)),
      );
      if (s.graded) cell.append(svgNode("path", {d:"M296 352H315V420", class:"loss-seed-wire",
        fill:"none", stroke:"#ffd172", "stroke-width":2, "marker-end":"url(#loss-arrow)"}));
      if (s.index > 1) {
        const from = 2 + 2 * Math.log1p(5 * Math.abs(s.feedback));
        const to = 2 + 2 * Math.log1p(5 * Math.abs(s.previous_feedback));
        cell.append(
          svgNode("path", {class:"backward-wire",
            d:"M315 " + (502 - from / 2) + "L55 " + (502 - to / 2) + "V" + (496 - to / 2) +
              "L45 502L55 " + (508 + to / 2) + "V" + (502 + to / 2) + "L315 " + (502 + from / 2) + "Z"}),
          svgNode("text", {x:180, y:534, class:"factor"}, "\u00d7 " + f(s.factor)),
        );
      }
      const select = () => { stop(); inspect(s.index - 1); };
      cell.addEventListener("click", select);
      cell.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); }
      });
      svg.append(cell);
    }
    cells.replaceChildren(svg);
  }
  function stop() {
    clearTimeout(timer);
    timer = null;
    text("pause", "Resume");
    root.dataset.playing = "false";
  }
  function inspect(index, phase = "inspect") {
    selected = index;
    const s = route.steps[index], t = s.index;
    root.dataset.step = String(t);
    root.dataset.phase = phase;
    cells.querySelectorAll(".cell").forEach((cell, offset) => cell.setAttribute("aria-pressed", String(offset === index)));
    text("selected-note", "Inside " + s.note + ": h" + (t - 1) + " \u2192 h" + t);
    const lowerMemory = s.state - memoryNudge, higherMemory = s.state + memoryNudge;
    text("lower-memory", "h" + t + " = " + lowerMemory.toFixed(8) +
      " \u2192 ending error " + s.loss_if_memory_lower.toFixed(12));
    text("higher-memory", "h" + t + " = " + higherMemory.toFixed(8) +
      " \u2192 ending error " + s.loss_if_memory_higher.toFixed(12));
    text("response-arithmetic", "(" + s.loss_if_memory_higher.toFixed(12) + " - " +
      s.loss_if_memory_lower.toFixed(12) + ") / (" + higherMemory.toFixed(8) + " - " +
      lowerMemory.toFixed(8) + ") \u2248 " + f(s.measured_response));
    text("memory-nudge", s.feedback > 0 ? "Positive response: raising this memory raises ending error; lowering it slightly would reduce error." :
      s.feedback < 0 ? "Negative response: raising this memory lowers ending error." :
        "Zero response: a tiny change here has no first-order effect on ending error.");
    document.getElementById("memory-nudge").dataset.measuredResponse = String(s.measured_response);
    document.getElementById("memory-nudge").dataset.nudge = String(memoryNudge);
    text("response-comparison", "Measured nearby slope: " + f(s.measured_response) +
      "; backprop's tiny-change limit: " + f(s.feedback) + ". Finite nudges can differ slightly.");
    text("phase", {forward:"Read forward", head:s.graded ? "Grade the ending" : "Prediction preview", backward:"Send feedback backward",
      update:"Sum, then update once", after:"Rerun with new weights", inspect:"Inspect this saved cell"}[phase]);
    text("forward-formula", "z" + t + " = " + vector(route.weights.writing) + " \u00b7 " + vector(s.inputs) +
      " + " + route.gain.toFixed(2) + " \u00d7 " + f(s.previous) + " = " + f(s.combined));
    text("state-formula", "h" + t + " = tanh(" + f(s.combined) + ") = " + f(s.state));
    text("backward-formula", "Error response, h" + t + " back to h" + (t - 1) + ": " + f(s.feedback) + " \u00d7 " + f(s.slope) +
      " \u00d7 " + route.gain.toFixed(2) + " = " + f(s.previous_feedback));
    text("factor-formula", "Handoff response = recurrent weight \u00d7 tanh responsiveness = " +
      route.gain.toFixed(2) + " \u00d7 " + f(s.slope) + " = " + f(s.factor));
    text("chain-formula", "g" + (t - 1) + " = g" + t + " \u00d7 (1 - h" + t + "\u00b2) \u00d7 m");
    text("head-formula", "Scores = " + vector(route.weights.prediction) + " \u00d7 h" + t + " = " + vector(s.scores) +
      "; softmax P[D3,A3,D4] = " + vector(s.probabilities) + "; L" + t + " = -log P(" + s.next_note +
      ") = " + probability(s.local_loss));
    text("head-scope", s.graded ? "This ending prediction error starts the backward pass." :
      "This prediction error is a preview only. All backward responses still come from the gold ending error.");
    text("why", s.factor < 0.999 ? "The ending-error response shrinks across this handoff." :
      s.factor > 1.001 ? "The ending-error response grows across this handoff." : "The ending-error response passes almost unchanged.");
    text("contribution-formula", "delta" + t + " = " + f(s.delta) + "; input-weight contribution = delta \u00d7 x = " +
      vector(s.writing_contribution) + "; memory-weight contribution = delta \u00d7 h" + (t - 1) + " = " + f(s.memory_contribution));
    root.dataset.incomingGradient = String(s.feedback);
    root.dataset.outgoingGradient = String(s.previous_feedback);
    root.dataset.localFactor = String(s.factor);
    if (extended.checked && phase !== "inspect") {
      const active = cells.querySelectorAll(".cell")[index].getBoundingClientRect();
      const viewport = cells.getBoundingClientRect();
      cells.scrollLeft = Math.max(0, cells.scrollLeft + active.left - viewport.left - viewport.width / 2 + active.width / 2);
    }
  }
  function row(label, key, id) {
    const tr = document.createElement("tr");
    tr.dataset.parameter = key;
    tr.dataset.before = JSON.stringify(route.weights[key]);
    tr.dataset.gradient = JSON.stringify(route.weight_gradients[key]);
    tr.dataset.after = JSON.stringify(route.updated_weights[key]);
    const format = value => Array.isArray(value) ? vector(value) : f(value);
    for (const [offset, value] of [label, format(route.weights[key]), format(route.weight_gradients[key]), format(route.updated_weights[key])].entries()) {
      const node = document.createElement(offset === 0 ? "th" : "td");
      node.textContent = value;
      if (offset === 2) { node.id = id; node.dataset.value = JSON.stringify(route.weight_gradients[key]); }
      tr.append(node);
    }
    return tr;
  }
  function refresh() {
    stop();
    const repeats = extended.checked ? 7 : 1;
    const index = Math.round((scale.valueAsNumber - 0.6) * 100);
    if (!Number.isInteger(index) || index < 0 || index > 70) throw new Error("Multiplier outside the recorded range.");
    route = data.routes[(repeats === 7 ? 71 : 0) + index];
    if (route.repeats !== repeats || Math.abs(route.gain - scale.valueAsNumber) > 1e-10) throw new Error("Recording does not match the selected trial.");
    const ratio = Math.abs(route.steps[0].feedback / route.steps.at(-1).feedback);
    const mode = ratio < 0.9 ? "vanishing" : ratio > 1.1 ? "exploding" : "balanced";
    root.dataset.mode = mode;
    root.dataset.ratio = String(ratio);
    root.dataset.repeats = String(repeats);
    root.style.setProperty("--signal", mode === "vanishing" ? "#77dce5" : mode === "exploding" ? "#ff997e" : "#ffd172");
    text("gradient-status", mode === "vanishing" ? "Fading feedback" : mode === "exploding" ? "Growing feedback" : "Feedback nearly unchanged");
    text("weight-value", route.gain.toFixed(2));
    text("memory-weight", route.gain.toFixed(2));
    text("gradient-ratio", f(ratio));
    text("distance-note", repeats === 1 ? "2 backward links." : "20 backward links.");
    cells.classList.toggle("long", repeats === 7);
    drawCells();
    const n = route.steps.length;
    text("loss-scope", "Error = prediction error (cross-entropy loss). The backward numbers measure how strongly " +
      "that ending error responds to a tiny change in each saved h, not the error itself or a weight update. " +
      "Only the gold ending error drives backward; dashed errors are previews. Click a block for a tiny-change example.");
    text("seed-formula", "Score errors = probabilities - [1, 0, 0] = " + vector(route.score_errors) +
      "; g" + n + " = " + vector(route.weights.prediction) + " \u00b7 " + vector(route.score_errors) + " = " + f(route.steps.at(-1).feedback));
    document.getElementById("update-rows").replaceChildren(
      row("Input multipliers (2 features)", "writing", "writing-gradient"),
      row("Recurrent weight", "memory", "memory-gradient"),
      row("Prediction multipliers [D3, A3, D4]", "prediction", "prediction-gradient"),
    );
    text("learning-rate", String(data.learning_rate));
    text("sum-formula", "dL/dm = sum of all cells' delta \u00d7 previous h = " +
      route.steps.map(s => f(s.memory_contribution)).join(" + ") + " = " + f(route.weight_gradients.memory));
    text("update-formula", "m_new = " + f(route.weights.memory) + " - " + data.learning_rate +
      " \u00d7 " + f(route.weight_gradients.memory) + " = " + f(route.updated_weights.memory));
    text("after-update", "One SGD step: prediction error " + f(route.loss) + " \u2192 " + f(route.after_loss) +
      "; guess " + route.prediction + " \u2192 " + route.after_prediction + ".");
    text("after-probability", "P(D3): " + f(route.probabilities[0]) + " \u2192 " + f(route.after_probabilities[0]));
    document.getElementById("after-update").dataset.loss = String(route.after_loss);
    document.getElementById("after-update").dataset.probability = String(route.after_probabilities[0]);
    selected = Math.min(selected, n - 1);
    inspect(selected);
  }
  function tick() {
    if (position >= stages.length) { stop(); text("pause", "Replay"); root.dataset.phase = "done"; return; }
    const [phase, index] = stages[position++];
    inspect(index, phase);
    timer = setTimeout(tick, extended.checked ? 350 : 850);
  }
  function play() {
    stop();
    const n = route.steps.length;
    stages = [
      ...route.steps.flatMap((_, i) => [["forward", i], ["head", i]]),
      ...route.steps.map((_, i) => ["backward", n - i - 1]),
      ["update", 0], ["after", n - 1],
    ];
    position = 0;
    if (reduced) {
      inspect(n - 1);
      root.dataset.phase = "done";
      document.getElementById("pause").disabled = true;
      text("pause", "Reduced motion");
      return;
    }
    root.dataset.playing = "true";
    text("pause", "Pause");
    tick();
  }
  scale.addEventListener("input", refresh);
  scale.addEventListener("change", () => { refresh(); play(); });
  extended.addEventListener("change", () => { refresh(); play(); });
  document.getElementById("replay").addEventListener("click", play);
  document.getElementById("pause").addEventListener("click", () => {
    if (timer !== null) { stop(); return; }
    if (position >= stages.length) { play(); return; }
    root.dataset.playing = "true";
    text("pause", "Pause");
    tick();
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  window.addEventListener("pagehide", stop);
  refresh();
  play();
})();
</script>
</body>
</html>"""


def render_player() -> str:
    with Image.open(CARD_ART) as image:
        buffer = BytesIO()
        image.resize((512, 288), Image.Resampling.LANCZOS).save(buffer, format="WEBP", lossless=True, method=6)
    result = PLAYER_TEMPLATE.replace("__DATA__", json.dumps(recorded_data(), separators=(",", ":"), allow_nan=False))
    result = result.replace("__CARD_ART__", "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    if any(marker in result for marker in ("__DATA__", "__CARD_ART__")):
        raise RuntimeError("The note-gradient template contains an unfilled marker.")
    return result + "\n"


def export_player(path: Path = PLAYER) -> None:
    raw = path.read_bytes() if path.exists() else None
    encoded = render_player().encode("utf-8")
    if encoded == raw:
        return
    with NamedTemporaryFile(dir=path.parent, prefix=".feedback-data-", suffix=".tmp", delete=False) as staged:
        staged_path = Path(staged.name)
        staged.write(encoded)
    try:
        if (path.read_bytes() if path.exists() else None) != raw:
            raise RuntimeError("The feedback player changed while it was being regenerated.")
        staged_path.replace(path)
    finally:
        staged_path.unlink(missing_ok=True)


def export_storyboard(path: Path = STORYBOARD) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 12), facecolor="#0d1b2e")
    head_labels = []
    with Image.open(CARD_ART) as source:
        card = source.copy()
    for axis, gain in zip(axes, GAINS, strict=True):
        probe = record_probe(gain=gain)
        color = "#77dce5" if gain < 1 else "#ff997e"
        axis.set(xlim=(0, 915), ylim=(560, 0), facecolor="#0d1b2e")
        axis.axis("off")
        axis.set_title(f"Recurrent weight = {gain:.2f} | {'fading' if gain < 1 else 'growing'} response",
                       loc="left", color="#ffd172", fontsize=13, pad=10)
        axis.annotate("", (84, 160), (6, 160),
                      arrowprops={"arrowstyle": "->", "color": "#b0bed3", "lw": 1.5})
        axis.text(45, 139, "h0", ha="center", color="#b0bed3", fontsize=13)
        axis.text(45, 181, "0", ha="center", color="#b0bed3", fontsize=13)
        axis.imshow(card, extent=(15, 75, 233, 199), aspect="auto", alpha=0.55)
        for i, step in enumerate(probe.steps):
            offset = i * 270
            x, output = offset + 180, offset + 315
            axis.text(x, 24, step.note, ha="center", color="#edf3ff", fontsize=14, weight="bold")
            axis.text(x, 49, f"x{i + 1}={[int(value) for value in step.inputs]}",
                      ha="center", color="#b0bed3", fontsize=11, family="monospace")
            axis.annotate("", (x, 86), (x, 58),
                          arrowprops={"arrowstyle": "->", "color": "#b0bed3", "lw": 1.5})
            axis.add_patch(FancyBboxPatch(
                (offset + 90, 92), 180, 142, boxstyle="round,pad=0,rounding_size=8",
                facecolor="#20364f", edgecolor="#7186a4", linewidth=1.5,
            ))
            for y, value, size, text_color in (
                (112, "RNN", 13, "#edf3ff"),
                (132, "Input multipliers", 9, "#b0bed3"),
                (151, "[.001,-.0005]", 10, "#b0bed3"),
                (177, f"h{i} x {gain:.2f}", 14, "#ffd172"),
                (203, f"= {step.memory_part:.6f}", 14, "#ffd172"),
                (226, "mix -> tanh", 11, "#b0bed3"),
            ):
                axis.text(x, y, value, ha="center", color=text_color, fontsize=size)
            axis.annotate("", (offset + 354, 160), (offset + 276, 160),
                          arrowprops={"arrowstyle": "->", "color": "#77dce5", "lw": 1.5})
            axis.text(output, 139, f"h{i + 1}", ha="center", color="#77dce5", fontsize=13)
            axis.text(output, 181, f"{step.state:.6f}", ha="center", color="#77dce5",
                      fontsize=12, family="monospace")
            axis.imshow(card, extent=(output - 30, output + 30, 233, 199), aspect="auto")
            axis.annotate("", (x, 258), (x, 240),
                          arrowprops={"arrowstyle": "->", "color": "#77dce5", "lw": 1.5})
            axis.text(x + 24, 253, f"h{i + 1}", ha="center", color="#77dce5", fontsize=10)
            axis.add_patch(FancyBboxPatch(
                (offset + 70, 264), 220, 136, boxstyle="round,pad=0,rounding_size=8",
                facecolor="#14263f", edgecolor="#ffd172" if step.graded else "#7186a4",
                linewidth=2 if step.graded else 1.5, linestyle="solid" if step.graded else "dashed",
            ))
            actual_probability = step.probabilities[NOTES.index(step.next_note)]
            labels = []
            for y, value, size, text_color in (
                (290, f"Actual next: {step.next_note}", 13, "#edf3ff"),
                (313, f"guess {step.prediction}", 11, "#b0bed3"),
                (339, f"P({step.next_note})={actual_probability:.{PROBABILITY_DECIMALS}f}", 13, "#77dce5"),
                (364, f"Error={step.local_loss:.{PROBABILITY_DECIMALS}f}", 13,
                 "#ffd172" if step.graded else "#b0bed3"),
                (388, "backprop source" if step.graded else "preview only", 11,
                 "#ffd172" if step.graded else "#b0bed3"),
            ):
                labels.append(axis.text(
                    x, y, value, ha="center", color=text_color, fontsize=size,
                    family="monospace" if y in (339, 364) else "sans-serif",
                ))
            head_labels.append(labels)
            if step.graded:
                axis.plot([offset + 296, output], [352, 352], color="#ffd172", linewidth=1.5)
                axis.annotate("", (output, 420), (output, 352),
                              arrowprops={"arrowstyle": "->", "color": "#ffd172", "lw": 1.5})
            axis.text(output, 438, "Ending-error", ha="center", color=color, fontsize=10)
            axis.text(output, 458, f"response at h{i + 1}", ha="center", color=color, fontsize=10)
            axis.text(output, 484, f"{step.feedback:.6f}", ha="center", color=color,
                      fontsize=13, family="monospace", weight="bold")
            if i:
                axis.annotate("", (offset + 45, 502), (output, 502),
                              arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
                axis.text(x, 534, f"x {step.factor:.6f}", ha="center", color=color, fontsize=12)
    figure.suptitle("Predict at every note. Backprop only from the gold ending error.", color="#edf3ff", fontsize=14)
    figure.text(0.5, 0.025, "Error response = tiny change in ending error / tiny change in saved memory (weights fixed).",
                ha="center", color="#b0bed3", fontsize=11)
    figure.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.07, hspace=0.28)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for labels in head_labels:
        for first, second in zip(labels, labels[1:]):
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                plt.close(figure)
                raise RuntimeError(f"Prediction-head labels overlap: {first.get_text()} / {second.get_text()}")
    for axis in axes:
        labels = [label for label in axis.texts if label.get_text()]
        for index, label in enumerate(labels):
            bounds = label.get_window_extent(renderer)
            if not (axis.bbox.contains(bounds.x0, bounds.y0) and axis.bbox.contains(bounds.x1, bounds.y1)):
                plt.close(figure)
                raise RuntimeError(f"Diagram label is clipped: {label.get_text()}")
            for other in labels[index + 1:]:
                if bounds.overlaps(other.get_window_extent(renderer)):
                    plt.close(figure)
                    raise RuntimeError(f"Diagram labels overlap: {label.get_text()} / {other.get_text()}")
        for image in axis.images:
            left, right, bottom, top = image.get_extent()
            points = axis.transData.transform(((left, bottom), (right, top)))
            art_bounds = Bbox.from_extents(
                min(points[:, 0]), min(points[:, 1]), max(points[:, 0]), max(points[:, 1]),
            )
            for label in axis.texts:
                if label.get_window_extent(renderer).overlaps(art_bounds):
                    plt.close(figure)
                    raise RuntimeError(f"Diagram text overlaps memory-card artwork: {label.get_text()}")
    figure.savefig(path, dpi=120, facecolor=figure.get_facecolor(),
                   metadata={
                       "TraceOrigin": "autograd; D3/A3/D4; one final cross-entropy loss; one hidden feature",
                       "DiagramLayout": "RNN blocks; per-note preview heads; final-loss backprop; all labels outside artwork",
                   })
    plt.close(figure)


if __name__ == "__main__":
    export_player()
    export_storyboard()
    print("Exported note-labelled cell mechanics, chain-rule gradients, static comparison and one real SGD update.")
