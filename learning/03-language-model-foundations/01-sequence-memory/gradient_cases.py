"""One row-aligned calculation across three measured RNN cases."""

from dataclasses import asdict
from html import escape
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from error_feedback_flow import LEARNING_RATE, record_probe
from handwritten_notes import FONT, PAPER, handwritten_math, paper_style


HERE = Path(__file__).resolve().parent
PLAYER = HERE / "images" / "01a-gradient-three-cases.html"
STORYBOARD = HERE / "images" / "01a-gradient-three-cases.png"
CASES = (("Stable", 1.0, "#77dce5"), ("Exploding", 1.2, "#ff997e"),
         ("Vanishing", 0.8, "#ffd172"))
EQUATIONS = (
    ("Memory", "h_t = tanh(W_x x_t + W_h h_(t-1))"),
    ("Prediction and loss", "p = softmax(W_out h_21); L = -log p[D3]"),
    ("Head gradient", "dL/dh_21 = sum W_out[k] * (p[k] - y[k])"),
    ("Backward handoffs",
     "dL/dh_(t-1) = dL/dh_t * W_h * (1 - h_t^2)\n"
     "dL/dh_1 = dL/dh_21 * prod_(t=2..21)(W_h * (1 - h_t^2))\n"
     "ratio = (dL/dh_1) / (dL/dh_21)"),
    ("Shared-weight gradient", "dL/dW_h = sum (dL/dh_t * (1 - h_t^2) * h_(t-1))"),
    ("Update", "W_h(new) = W_h - learning_rate * dL/dW_h; L(new) = -log p_new[D3]"),
)
QUESTIONS = (
    "What memory did the last note leave?",
    "How much confidence did we give the right note?",
    "Which way would changing that memory move the error?",
    "How much feedback makes it back to the opening?",
    "What did each use of the same weight contribute?",
    "Only now: change the weights and read again.",
)


def recorded_cases() -> list[dict]:
    cases = []
    for title, weight, color in CASES:
        probe = record_probe(gain=weight, repeats=7)
        ratio = probe.steps[0].feedback / probe.steps[-1].feedback
        last, previous = probe.steps[-1], probe.steps[-2]
        head = ", ".join(f"{weight:g}" for weight in probe.weights.prediction)
        rows = (
            f"h_21 = tanh({last.input_part:.6f}\n"
            f"       + {weight:.1f} * {last.previous:.6f})\n"
            f"     = {last.state:.6f}",
            f"p = softmax([{head}]\n"
            f"            * {last.state:.6f})\n"
            "  = [" + ", ".join(f"{value:.6f}" for value in probe.probabilities) + "]\n"
            f"L = -log({probe.probabilities[0]:.6f}) = {probe.loss:.6f}",
            f"({probe.weights.prediction[0]:g}) * ({probe.probabilities[0]:.6f} - 1)\n"
            f"+ {probe.weights.prediction[1]:g} * ({probe.probabilities[1]:.6f} - 0)\n"
            f"+ {probe.weights.prediction[2]:g} * ({probe.probabilities[2]:.6f} - 0)\n"
            f"= {last.feedback:.6f}",
            f"dL/dh_20 = {last.feedback:.6f}\n"
            f"  * {weight:.1f} * (1 - {last.state:.6f}^2)\n"
            f"  = {previous.feedback:.6f}\n"
            f"dL/dh_1 = {last.feedback:.6f} * {ratio:.6f}\n"
            f"  = {probe.steps[0].feedback:.6f}\n"
            f"ratio = {ratio:.6f}",
            f"t=21: {last.feedback:.6f} * (1 - {last.state:.6f}^2)\n"
            f"      * {last.previous:.6f} = {last.memory_contribution:.6f}\n"
            f"dL/dW_h = 0 + {probe.steps[1].memory_contribution:.6f}\n"
            f"          + ... + {last.memory_contribution:.6f}\n"
            f"        = {probe.weight_gradients.memory:.6f}",
            f"W_h(new) = {weight:.6f}\n"
            f"  - {LEARNING_RATE:.2f} * {probe.weight_gradients.memory:.6f}\n"
            f"  = {probe.updated_weights.memory:.6f}\n"
            f"L(new) = -log({probe.after_probabilities[0]:.6f})\n"
            f"       = {probe.after_loss:.6f}",
        )
        cases.append({
            "title": title, "color": color, "weight": weight, "ratio": ratio,
            "rows": rows, "probe": asdict(probe),
        })
    return cases


def source_fingerprint() -> str:
    return hashlib.sha256(
        Path(__file__).read_bytes()
        + json.dumps(recorded_cases(), sort_keys=True).encode()
    ).hexdigest()


def render_player() -> str:
    cases = recorded_cases()
    head = ", ".join(f"{weight:g}" for weight in cases[0]["probe"]["weights"]["prediction"])
    headers = "".join(
        f'<th scope="col" data-case-header="{case["title"]}">'
        f'{case["title"]}<br>W<sub>h</sub> = {case["weight"]:.1f}</th>'
        for case in cases
    )
    groups = []
    for index, (title, equation) in enumerate(EQUATIONS):
        values = "".join(
            f'<td data-case="{case["title"]}" data-weight="{case["weight"]:.1f}">'
            f'<div class="calculation">{handwritten_math(case["rows"][index])}</div>'
            f'<details class="exact"><summary>Exact text</summary>'
            f'<pre>{escape(case["rows"][index])}</pre></details></td>'
            for case in cases
        )
        groups.append(
            f'<tbody class="step" data-step="{index}"><tr class="formula">'
            f'<th colspan="3" scope="rowgroup"><span>{index + 1}. {QUESTIONS[index]}</span>'
            f'<div class="equation">{handwritten_math(equation)}</div>'
            f'<details class="exact"><summary>Formula as text</summary>'
            f'<pre>{escape(equation)}</pre></details></th></tr>'
            f'<tr class="substitutions">{values}</tr></tbody>'
        )
    traces = []
    for case in cases:
        trace = "".join(
            f'<tr><td>{s["index"]}</td><td>{s["note"]}</td><td>{s["state"]:.8f}</td>'
            f'<td>{s["feedback"]:.8f}</td><td>{s["memory_contribution"]:.8f}</td></tr>'
            for s in case["probe"]["steps"]
        )
        traces.append(
            f'<details><summary>{case["title"]}: all 21 recorded reads</summary><div class="trace"><table>'
            '<thead><tr><th>t</th><th>Note</th><th>h_t</th><th>dL/dh_t</th><th>W_h contribution</th></tr></thead>'
            f'<tbody>{trace}</tbody></table></div></details>'
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>One calculation, three gradient outcomes</title><style>
:root{color-scheme:light;font:15px system-ui;color:#23334a;background:#fff}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:22px}
h1{font-size:25px;margin:0 0 8px}p{line-height:1.6;margin:8px 0}
button{font:inherit;padding:8px 14px;
background:#fff;color:#234875;border:1px solid #8794a2;border-radius:7px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}button:focus-visible,summary:focus-visible{outline:3px solid #234875}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}
.scroll,.trace{max-width:100%;overflow-x:auto}
.scroll{padding:16px 8px 28px}
.comparison{width:100%;table-layout:fixed;border-collapse:collapse}
.comparison thead th{padding:12px;color:#234875;font-size:24px;font-weight:400}
.comparison td{width:33.333%;padding:24px 18px;vertical-align:top}
.formula th{text-align:left;padding:16px 18px;font-weight:400}
.formula span{display:block;font-size:25px;color:#23334a;margin-bottom:12px}
.equation{white-space:pre-wrap;overflow-wrap:anywhere;font:24px/1.8 LessonHand,cursive;color:#234875}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace;margin:0}
.calculation{white-space:pre-wrap;overflow-wrap:anywhere;color:#234875;font:23px/1.9 LessonHand,cursive}
.step[hidden]{display:none}.step.visible{animation:arrive .3s ease-out}
.muted{color:#596273;font-size:14px}.exact{font:12px system-ui;color:#596273;margin-top:12px}
.trace table{font:12px ui-monospace;border-collapse:collapse}
.trace td,.trace th{padding:7px;text-align:right;border-bottom:1px solid #c9cdd2}
summary{cursor:pointer;padding:10px 0}
@keyframes arrive{from{opacity:.25}to{opacity:1}}
@media(max-width:850px){.comparison thead{display:none}.comparison,.step,.step tr,.step th,.step td{display:block;width:100%}
.step[hidden]{display:none}.comparison td{padding:18px}.comparison td:before{content:attr(data-case) " — memory weight " attr(data-weight);
display:block;font-size:24px;color:#23334a;margin-bottom:10px}.formula th{padding:12px 18px}}
@media(prefers-reduced-motion:reduce){.step.visible{animation:none}}
</style><style>""" + paper_style() + """</style></head><body><main data-player="gradient-three-cases">
<h1>One calculation, three gradient outcomes</h1>
<p class="muted">D3 A3 D4 repeated 7 times; h_0 = 0; x_21 = [1, 1]; W_x = [0.001, -0.0005];
W_out = [""" + head + """]; y = [1, 0, 0]; learning_rate = 0.02.
Vector order: D3, A3, D4. Only initial W_h differs between columns.</p>
<div class="controls"><button id="back" type="button">Back</button><button id="play" type="button">Play steps</button>
<button id="next" type="button">Next step</button><button id="reset" type="button">Reset</button>
<span id="status" role="status" aria-live="polite"></span></div>
<div class="scroll paper" tabindex="0" role="region" aria-label="Handwritten measured gradient calculations">
<table class="comparison"><thead><tr>""" + headers + """</tr></thead>""" + "".join(groups) + """</table></div>
<p class="muted">One calculation at a time; compare all three cases. On narrow panes, read down the page.
Values are rounded; calculations use full precision. MAI paper; offline handwriting; no generated numbers.
SGD updates W_x, W_h and W_out together; the table displays W_h's update.</p>
""" + "".join(traces) + """
<noscript>All substitutions remain visible without JavaScript.</noscript>
<script>
let step=0,timer=null;
const total=6,play=document.getElementById("play"),next=document.getElementById("next"),back=document.getElementById("back");
function stop(){if(timer!==null){clearInterval(timer);timer=null;}play.textContent="Play steps";}
function render(){document.querySelectorAll(".step").forEach(group=>{
const visible=Number(group.dataset.step)===step;group.hidden=!visible;
group.classList.toggle("visible",visible);});
document.getElementById("status").textContent="Calculation step "+(step+1)+" of "+total;
next.disabled=step===total-1;back.disabled=step===0;}
function advance(){if(step<total-1){step++;render();}if(step===total-1)stop();}
play.addEventListener("click",()=>{if(timer!==null){stop();return;}
if(step===total-1){step=0;render();}
play.textContent="Pause";timer=setInterval(advance,6500);});
next.addEventListener("click",()=>{stop();advance();});
back.addEventListener("click",()=>{stop();step=Math.max(0,step-1);render();});
document.getElementById("reset").addEventListener("click",()=>{stop();step=0;render();});
render();
</script>
</main></body></html>"""


def markdown_comparison() -> str:
    cases = recorded_cases()
    headers = ["Formula"] + [f'{case["title"]}: W_h={case["weight"]:.1f}' for case in cases]
    lines = [
        "The weight update changes W_x, W_h and W_out together.",
        "The table shows only W_h's update; its new loss uses all three updated sets of weights.",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---|",
    ]
    for index, (title, equation) in enumerate(EQUATIONS):
        formula = escape(equation).replace("\n", "<br>")
        values = [escape(case["rows"][index]).replace("\n", "<br>") for case in cases]
        lines.append(
            f"| **{title}**<br><code>{formula}</code> | "
            + " | ".join(f"<code>{value}</code>" for value in values) + " |"
        )
    return "\n".join(lines)


def export(directory: Path = HERE / "images") -> None:
    directory.mkdir(exist_ok=True)
    html = render_player()
    (directory / PLAYER.name).write_text(html, encoding="utf-8", newline="\n")
    font = FontProperties(fname=FONT)
    figure, axes = plt.subplots(1, 3, figsize=(18, 9), facecolor="white")
    background = figure.add_axes((0, 0, 1, 1), zorder=-1)
    background.imshow(plt.imread(PAPER), aspect="auto")
    background.axis("off")
    figure.subplots_adjust(left=.035, right=.985, bottom=.12, top=.72, wspace=.10)
    figure.suptitle("How much feedback reaches the opening?", color="#23334a",
                   fontsize=30, fontproperties=font, y=.96)
    figure.text(.5, .86, "21 reads, 20 backward links. Only the starting memory weight changes.",
                color="#596273", ha="center", fontsize=19, fontproperties=font)
    figure.text(.5, .78, "Opening feedback = ending feedback × all the handoff responses",
                color="#234875", ha="center", fontsize=22, fontproperties=font)
    for axis, case in zip(axes, recorded_cases(), strict=True):
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
        axis.text(.05, .96, f'{case["title"]}: W_h = {case["weight"]:.1f}',
                  color="#23334a", fontsize=22, va="top", fontproperties=font)
        axis.text(.05, .77, case["rows"][3], color="#234875", fontsize=19, va="top",
                  linespacing=1.7, fontproperties=font)
    figure.text(.5, .06, "A ratio near 1 keeps the message; above 1 amplifies it; below 1 weakens it.",
                color="#23334a", ha="center", fontsize=21, fontproperties=font)
    figure.savefig(directory / STORYBOARD.name, dpi=150, facecolor=figure.get_facecolor(),
                   metadata={"SourceFingerprint": source_fingerprint(),
                             "MeasuredCases": json.dumps(recorded_cases())})
    plt.close(figure)
    print("Saved the three measured gradient panels and their step-by-step player.")


if __name__ == "__main__":
    export()
