"""Export a weighted RNN forward pass and its actual backward gradients."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import torch


HERE = Path(__file__).resolve().parent
PLAYER = HERE / "images" / "01a-error-feedback-tracker.html"
TARGET = 0.5
REWRITES = 20
INPUT = 1.0
WRITING_WEIGHT = 0.001
PREDICTION_WEIGHT = 1.0
GAINS = (0.8, 1.2)
SCALE_GAINS = tuple(value / 100 for value in range(60, 131))
RECORDING = re.compile(
    r'(<script id="recorded-cases" type="application/json">)\s*(.*?)\s*(</script>)',
    re.DOTALL,
)


@dataclass(frozen=True)
class ErrorFeedbackProbe:
    gain: float
    prediction: float
    error: float
    loss: float
    states: tuple[float, ...]
    feedback: tuple[float, ...]
    writing_gradient: float
    memory_gradient: float
    prediction_gradient: float


def record_probe(*, gain: float = 0.8, target: float = TARGET) -> ErrorFeedbackProbe:
    for name, value in (("target", target), ("gain", gain)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if not 0.6 <= gain <= 1.3:
        raise ValueError("Choose a memory weight between 0.6 and 1.3.")
    writing = torch.tensor(WRITING_WEIGHT, dtype=torch.float64, requires_grad=True)
    memory = torch.tensor(gain, dtype=torch.float64, requires_grad=True)
    prediction_weight = torch.tensor(PREDICTION_WEIGHT, dtype=torch.float64, requires_grad=True)
    previous = torch.tensor(0.0, dtype=torch.float64)
    states = []
    for step in range(REWRITES + 1):
        current_input = INPUT if step == 0 else 0.0
        previous = torch.tanh(writing * current_input + memory * previous)
        states.append(previous)
    output = prediction_weight * states[-1]
    error = output - target
    loss = 0.5 * error.square()
    gradients = torch.autograd.grad(loss, (*states, writing, memory, prediction_weight))
    feedback = tuple(value.item() for value in gradients[:-3])
    if not all(math.isfinite(value) for value in (*feedback, loss.item())):
        raise ValueError("The toy penalty or feedback overflowed; use a smaller target.")
    return ErrorFeedbackProbe(
        gain, output.item(), error.item(), loss.item(),
        tuple(value.item() for value in states), feedback,
        gradients[-3].item(), gradients[-2].item(), gradients[-1].item(),
    )


def recorded_data() -> dict:
    routes = []
    for gain in SCALE_GAINS:
        probe = record_probe(gain=gain)
        routes.append({
            "gain": gain, "states": probe.states, "feedback": probe.feedback,
            "prediction": probe.prediction, "error": probe.error, "loss": probe.loss,
            "weight_gradients": {
                "writing": probe.writing_gradient, "memory": probe.memory_gradient,
                "prediction": probe.prediction_gradient,
            },
        })
    return {
        "version": 3, "input": INPUT, "target": TARGET, "rewrites": REWRITES,
        "writing_weight": WRITING_WEIGHT, "prediction_weight": PREDICTION_WEIGHT,
        "routes": routes,
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
<title>Drag the weight, watch the gradients</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; color: #edf3ff; background: #0d1b2e; }
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1000px; margin: auto; padding: 16px; }
h1 { font-size: 23px; margin: 0 0 12px; }
h2 { font-size: 17px; margin: 0 0 10px; }
p { line-height: 1.35; margin: 6px 0; }
.muted { color: #b0bed3; }
.controls { padding: 12px 14px; background: #172941; border-radius: 12px; margin-bottom: 12px; }
.control-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.presets { display: inline-flex; gap: 3px; padding: 3px; border-radius: 999px; background: #0d1b2e; }
button { font: 600 13px system-ui; border: 0; border-radius: 999px; padding: 7px 12px; cursor: pointer; background: transparent; color: #b0bed3; }
button[aria-pressed="true"] { color: #0d1b2e; background: var(--signal, #77dce5); }
input:focus-visible, button:focus-visible { outline: 3px solid #ffd172; outline-offset: 3px; }
#weight-value { color: #ffd172; font: 600 20px ui-monospace, monospace; }
#weight-scale { width: 100%; margin: 12px 0 4px; cursor: ew-resize; accent-color: #ffd172; height: 22px; }
.scale-labels { display: flex; justify-content: space-between; color: #b0bed3; font-size: 12px; }
.panels { display: grid; grid-template-columns: 270px minmax(0, 1fr); gap: 12px; }
.panel { padding: 12px; border-radius: 12px; background: #172941; }
.weights, .numbers { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
.weight, .number { padding: 8px; background: #20364f; border-radius: 8px; min-width: 0; }
.weight span, .number span { display: block; font-size: 12px; color: #b0bed3; }
.weight strong { display: block; margin-top: 4px; color: #ffd172; font: 600 16px ui-monospace, monospace; }
.numbers { grid-template-columns: 1fr 1fr; margin-top: 10px; }
.number output { font: 600 20px ui-monospace, monospace; display: block; margin-top: 4px; }
.small { font-size: 12px; }
.arrow { color: #b0bed3; text-align: center; margin: 5px 0; }
svg { width: 100%; height: auto; display: block; }
svg text { font-family: system-ui, sans-serif; fill: #b0bed3; }
.card { fill: #20364f; stroke: #77dce5; stroke-width: 1; }
.memory { fill: #77dce5; font: 600 20px ui-monospace, monospace; }
.feedback { font: 600 16px ui-monospace, monospace; }
.result { display: flex; justify-content: space-between; gap: 8px; align-items: center; padding: 9px; background: #20364f; border-radius: 8px; }
.result span { font-size: 12px; color: #b0bed3; }
.result output { font: 600 21px ui-monospace, monospace; color: var(--signal, #77dce5); }
#phase { color: var(--signal, #77dce5); font-size: 13px; }
#problem { color: #ffac9e; }
@media (max-width: 800px) {
  .panels { grid-template-columns: 1fr; }
  .forward-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: center; }
  .numbers { margin-top: 0; }
}
@media (max-width: 430px) { main { padding: 12px; } .forward-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main data-player="error-feedback-tracker" data-phase="done">
  <h1>Drag the weight. Watch the feedback.</h1>
  <div class="controls">
    <div class="control-row">
      <label for="weight-scale" title="The recurrent weight multiplies the previous state; it is not the stored state.">Recurrent weight <output id="weight-value">0.80</output></label>
      <div class="presets" role="group" aria-label="Gradient examples">
        <button id="preset-vanishing" type="button" aria-pressed="true">Vanishing</button>
        <button id="preset-exploding" type="button" aria-pressed="false">Exploding</button>
      </div>
    </div>
    <input id="weight-scale" type="range" min="0.6" max="1.3" step="0.01" value="0.8" aria-label="Recurrent weight scale">
    <div class="scale-labels"><span>0.60 &middot; weaker</span><span>1.00</span><span>stronger &middot; 1.30</span></div>
  </div>
  <p id="problem" role="alert" hidden></p>
  <div class="panels">
    <section id="forward-panel" class="panel" aria-label="Weights output and error">
      <h2>1. Weights &rarr; output &rarr; error</h2>
      <div class="forward-grid">
        <div>
          <div class="weights" aria-label="Model weights">
            <div class="weight"><span>Input weight</span><strong id="writing-weight">__WRITING__</strong></div>
            <div class="weight"><span>Recurrent weight</span><strong id="memory-weight">0.80</strong></div>
            <div class="weight"><span>Prediction weight</span><strong id="prediction-weight">__HEAD__</strong></div>
          </div>
          <p class="small muted">Input __INPUT__, then __REWRITES__ zero inputs.</p>
        </div>
        <div>
          <div class="numbers">
            <div class="number"><span>Output</span><output id="output-value">__OUTPUT__</output></div>
            <div class="number"><span>Error: output - target __TARGET__</span><output id="error-value">__ERROR__</output></div>
          </div>
        </div>
      </div>
    </section>
    <section id="backward-panel" class="panel" aria-label="Backward gradients">
      <h2>2. Feedback flows backward</h2>
      <svg viewBox="0 0 640 232" role="img" aria-label="Gradients travelling backward through recorded RNN memory values">
        <path class="card" d="M14 16H174L188 30V138H14Z M174 16V30H188"/>
        <path class="card" d="M232 16H392L406 30V138H232Z M392 16V30H406"/>
        <path class="card" d="M450 16H610L624 30V138H450Z M610 16V30H624"/>
        <text x="101" y="42" text-anchor="middle" font-size="16">First</text>
        <text x="319" y="42" text-anchor="middle" font-size="16">Middle</text>
        <text x="537" y="42" text-anchor="middle" font-size="16">Last</text>
        <text x="101" y="63" text-anchor="middle" font-size="13">stored state h</text>
        <text x="319" y="63" text-anchor="middle" font-size="13">stored state h</text>
        <text x="537" y="63" text-anchor="middle" font-size="13">stored state h</text>
        <text id="opening-state" class="memory" x="101" y="86" text-anchor="middle">__FIRST__</text>
        <text id="middle-state" class="memory" x="319" y="86" text-anchor="middle">__MIDDLE__</text>
        <text id="last-state" class="memory" x="537" y="86" text-anchor="middle">__LAST__</text>
        <text x="101" y="104" text-anchor="middle" font-size="12">gradient strength</text>
        <text x="319" y="104" text-anchor="middle" font-size="12">gradient strength</text>
        <text x="537" y="104" text-anchor="middle" font-size="12">gradient strength</text>
        <text id="opening-feedback" class="feedback" x="101" y="126" text-anchor="middle">__G0__</text>
        <text id="middle-feedback" class="feedback" x="319" y="126" text-anchor="middle">__G10__</text>
        <text id="last-feedback" class="feedback" x="537" y="126" text-anchor="middle">__G20__</text>
        <path d="M537 174H101l10 -6m-10 6l10 6" fill="none" stroke="#70849c" stroke-width="2"/>
        <g id="trail"></g>
        <circle id="pulse" cx="101" cy="174" r="2" fill="#77dce5"/>
        <text x="101" y="218" text-anchor="middle" font-size="14">20 rewrites back</text>
        <text x="319" y="218" text-anchor="middle" font-size="14">10</text>
        <text x="537" y="218" text-anchor="middle" font-size="14">0</text>
      </svg>
      <p id="phase">Drag the scale; release to replay.</p>
      <div class="result">
        <span>Opening weight's gradient</span>
        <output id="writing-gradient">__WRITING_GRADIENT__</output>
      </div>
    </section>
  </div>
  <p class="small muted">Numeric RNN &middot; target __TARGET__ &middot; recorded gradients &middot; compressed pulse sizes &middot; no weight updates.</p>
  <noscript>JavaScript enables dragging. The default numbers show the vanishing example.</noscript>
</main>
<script id="recorded-cases" type="application/json">
__DATA__
</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("recorded-cases").textContent);
  const root = document.querySelector("main");
  const scale = document.getElementById("weight-scale");
  const problem = document.getElementById("problem");
  if (data.version !== 3 || data.rewrites !== 20 || !Array.isArray(data.routes) || data.routes.length !== 71) {
    problem.hidden = false;
    problem.textContent = "The recorded model data is invalid. Regenerate this player.";
    scale.disabled = true;
    throw new Error("Invalid weight-driven gradient recording.");
  }
  const pulse = document.getElementById("pulse");
  const segments = [];
  for (let step = 1; step <= 20; step++) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", 537 - (step - 1) * 21.8);
    line.setAttribute("x2", 537 - step * 21.8);
    line.setAttribute("y1", 174);
    line.setAttribute("y2", 174);
    line.setAttribute("stroke-linecap", "round");
    document.getElementById("trail").append(line);
    segments.push(line);
  }
  let route = data.routes.find(item => item.gain === 0.8);
  let frame = 20;
  let playing = false;
  let animation = 0;
  let started = null;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function format(value) {
    if (value === 0) return "0.000000";
    return Math.abs(value) < 0.0001 ? value.toExponential(3) : value.toFixed(6);
  }
  function showForward() {
    const index = Math.round((scale.valueAsNumber - 0.6) * 100);
    if (!Number.isInteger(index) || index < 0 || index >= data.routes.length) throw new Error("Weight outside the recorded scale.");
    route = data.routes[index];
    const ratio = Math.abs(route.feedback[0] / route.feedback[20]);
    const colour = ratio < 0.9 ? "#77dce5" : ratio > 1.1 ? "#ff997e" : "#ffd172";
    root.style.setProperty("--signal", colour);
    root.dataset.mode = ratio < 0.9 ? "vanishing" : ratio > 1.1 ? "exploding" : "balanced";
    document.getElementById("weight-value").textContent = route.gain.toFixed(2);
    document.getElementById("memory-weight").textContent = route.gain.toFixed(2);
    document.getElementById("preset-vanishing").setAttribute("aria-pressed", String(route.gain === 0.8));
    document.getElementById("preset-exploding").setAttribute("aria-pressed", String(route.gain === 1.2));
    for (const [id, value] of [["output-value", route.prediction], ["error-value", route.error]]) {
      const node = document.getElementById(id);
      node.textContent = format(value);
      node.dataset.value = String(value);
    }
    for (const [name, index] of [["opening", 0], ["middle", 10], ["last", 20]]) {
      document.getElementById(name + "-state").textContent = format(route.states[index]);
    }
    segments.forEach((line, i) => {
      const gradient = route.feedback[19 - i];
      line.dataset.feedback = String(gradient);
      line.setAttribute("stroke", colour);
      line.setAttribute("stroke-width", 3 * Math.log1p(5 * Math.abs(gradient)));
    });
  }
  function showBackward(frame) {
    root.dataset.frame = String(frame);
    const colour = root.style.getPropertyValue("--signal");
    for (const [name, index] of [["opening", 0], ["middle", 10], ["last", 20]]) {
      const node = document.getElementById(name + "-feedback");
      node.textContent = format(Math.abs(route.feedback[index]));
      node.dataset.feedback = String(route.feedback[index]);
      node.style.fill = colour;
    }
    segments.forEach((line, i) => { line.style.visibility = i < frame ? "visible" : "hidden"; });
    pulse.style.visibility = frame < 0 ? "hidden" : "visible";
    pulse.setAttribute("cx", 537 - Math.max(0, frame) * 21.8);
    pulse.setAttribute("fill", colour);
    pulse.setAttribute("r", 4 * Math.log1p(5 * Math.abs(route.feedback[20 - Math.max(0, frame)])));
    const result = document.getElementById("writing-gradient");
    result.textContent = format(route.weight_gradients.writing);
    result.dataset.value = String(route.weight_gradients.writing);
    const ratio = Math.abs(route.feedback[0] / route.feedback[20]);
    document.getElementById("phase").textContent = ratio < 0.9 ? "Fading toward the opening" : ratio > 1.1 ? "Growing toward the opening" : "Almost unchanged";
  }
  function finish() {
    cancelAnimationFrame(animation);
    playing = false;
    root.dataset.phase = "done";
    frame = 20;
    showBackward(20);
  }
  function tick(time) {
    if (started === null) started = time;
    const elapsed = time - started;
    frame = Math.min(20, Math.floor(elapsed / 130));
    showBackward(frame);
    if (frame === 20) { finish(); return; }
    animation = requestAnimationFrame(tick);
  }
  function start() {
    cancelAnimationFrame(animation);
    showForward();
    started = null;
    if (reducedMotion) { finish(); return; }
    playing = true;
    root.dataset.phase = "backward";
    frame = 0;
    showBackward(0);
    animation = requestAnimationFrame(tick);
  }
  scale.addEventListener("input", () => { showForward(); showBackward(playing ? frame : 20); });
  scale.addEventListener("change", start);
  document.getElementById("preset-vanishing").addEventListener("click", () => { scale.value = "0.8"; start(); });
  document.getElementById("preset-exploding").addEventListener("click", () => { scale.value = "1.2"; start(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) finish(); });
  window.addEventListener("pagehide", () => cancelAnimationFrame(animation));
  start();
})();
</script>
</body>
</html>"""


def render_player() -> str:
    data = recorded_data()
    default = next(route for route in data["routes"] if route["gain"] == 0.8)
    replacements = {
        "__DATA__": json.dumps(data, separators=(",", ":")),
        "__INPUT__": f"{INPUT:g}", "__WRITING__": f"{WRITING_WEIGHT:g}",
        "__HEAD__": f"{PREDICTION_WEIGHT:g}", "__TARGET__": f"{TARGET:g}",
        "__REWRITES__": str(REWRITES), "__FIRST__": f'{default["states"][0]:.6f}',
        "__MIDDLE__": f'{default["states"][10]:.6f}', "__LAST__": f'{default["states"][20]:.8f}',
        "__OUTPUT__": f'{default["prediction"]:.8f}', "__ERROR__": f'{default["error"]:.6f}',
        "__G0__": f'{abs(default["feedback"][0]):.6f}',
        "__G10__": f'{abs(default["feedback"][10]):.6f}',
        "__G20__": f'{abs(default["feedback"][20]):.6f}',
        "__WRITING_GRADIENT__": f'{default["weight_gradients"]["writing"]:.6f}',
    }
    result = PLAYER_TEMPLATE
    for marker, value in replacements.items():
        assert marker in result
        result = result.replace(marker, value)
    assert "__" not in result
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
        if staged_path.exists():
            staged_path.unlink()


if __name__ == "__main__":
    export_player()
    print("Exported weights-to-output/error and vanishing/exploding backward passes with PyTorch autograd.")
