"""Verify the three panels solve one real RNN calculation, not illustrative curves."""

from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from error_feedback_flow import LEARNING_RATE, record_probe
from gradient_cases import (
    CASES, EQUATIONS, PLAYER, STORYBOARD, export, markdown_comparison, recorded_cases, render_player,
)
from handwritten_notes import handwritten_math


class PanelParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.case_headers = []
        self.steps = []
        self.calculations = []
        self.external = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-case-header" in attrs:
            self.case_headers.append(attrs["data-case-header"])
        if "data-step" in attrs:
            self.steps.append(int(attrs["data-step"]))
        if tag == "td" and "data-case" in attrs:
            self.calculations.append(attrs["data-case"])
        for key in ("src", "href"):
            if attrs.get(key, "").startswith(("http:", "https:")):
                self.external.append(attrs[key])


class GradientCaseTests(unittest.TestCase):
    def test_all_cases_share_the_task_and_only_change_the_recurrent_weight(self):
        cases = recorded_cases()
        self.assertEqual([case["title"] for case in cases], ["Stable", "Exploding", "Vanishing"])
        baseline = cases[0]["probe"]
        for case in cases:
            probe = record_probe(gain=case["weight"], repeats=7)
            self.assertEqual(len(case["probe"]["steps"]), 21)
            self.assertEqual(case["probe"]["weights"]["writing"], baseline["weights"]["writing"])
            self.assertEqual(case["probe"]["weights"]["prediction"], baseline["weights"]["prediction"])
            self.assertEqual([step["inputs"] for step in case["probe"]["steps"]],
                             [step["inputs"] for step in baseline["steps"]])
            self.assertAlmostEqual(case["ratio"], math.prod(step.factor for step in probe.steps[1:]))
            self.assertAlmostEqual(sum(step.memory_contribution for step in probe.steps),
                                   probe.weight_gradients.memory)
            self.assertAlmostEqual(probe.updated_weights.memory,
                                   case["weight"] - LEARNING_RATE * probe.weight_gradients.memory)
        self.assertAlmostEqual(cases[0]["ratio"], 1, delta=.001)
        self.assertGreater(cases[1]["ratio"], 30)
        self.assertLess(cases[2]["ratio"], .02)

    def test_every_visible_calculation_uses_the_same_equations_and_measured_values(self):
        html = render_player()
        parser = PanelParser()
        parser.feed(html)
        self.assertEqual(parser.case_headers, ["Stable", "Exploding", "Vanishing"])
        self.assertEqual(parser.steps, list(range(6)))
        self.assertEqual(parser.calculations, ["Stable", "Exploding", "Vanishing"] * 6)
        self.assertEqual(parser.external, [])
        self.assertNotIn('type="range"', html)
        for case in recorded_cases():
            for row in case["rows"]:
                self.assertIn(escape(row), html)
            probe = case["probe"]
            self.assertIn(
                f'({probe["weights"]["prediction"][0]:g}) * ({probe["probabilities"][0]:.6f} - 1)',
                case["rows"][2],
            )
            last = probe["steps"][-1]
            self.assertIn(
                f'{last["feedback"]:.6f} * (1 - {last["state"]:.6f}^2)',
                case["rows"][4],
            )
        for _, equation in EQUATIONS:
            self.assertEqual(html.count(escape(equation)), 1)
        self.assertIn("clearInterval", html)
        self.assertIn("y = [1, 0, 0]", html)
        self.assertIn("prefers-reduced-motion", html)
        self.assertIn("<noscript>All substitutions remain visible", html)

    def test_exports_are_self_contained_and_embedded_in_the_notebook(self):
        with TemporaryDirectory() as directory:
            export(Path(directory))
            self.assertEqual((Path(directory) / PLAYER.name).read_text(encoding="utf-8"), render_player())
            with Image.open(Path(directory) / STORYBOARD.name) as image:
                cases = json.loads(image.info["MeasuredCases"])
                self.assertEqual([case["weight"] for case in cases], [value for _, value, _ in CASES])
                self.assertGreater(image.width, 2000)
        book = json.loads((PLAYER.parent.parent / "01A1-sequence-memory-worked-details.ipynb").read_bytes())
        self.assertFalse(any(STORYBOARD.name in cell.get("attachments", {}) for cell in book["cells"]))
        intro = next(cell for cell in book["cells"] if cell["id"] == "gradient-native-intro")
        self.assertIn(markdown_comparison(), "".join(intro["source"]))

    def test_handwritten_work_preserves_exact_values_and_has_no_external_resources(self):
        player = render_player()
        self.assertIn("data:font/ttf;base64,", player)
        self.assertIn("data:image/webp;base64,", player)
        self.assertIn('class="scroll paper"', player)
        self.assertIn("h<sub>21</sub>", player)
        self.assertIn("1 - h<sub>t</sub><sup>2</sup>", player)
        self.assertIn("∏<sub>t=2..21</sub>", player)
        self.assertIn("∑<sub>k</sub>", player)
        self.assertIn(".comparison td:before", player)
        self.assertNotIn("min-width:1080px", player)
        self.assertNotIn("<script src=", player)
        self.assertEqual(handwritten_math("h_21 = 0.000123 * W_h"),
                         "h<sub>21</sub> = 0.000123 × W<sub>h</sub>")
        self.assertEqual(handwritten_math("<unsafe>"), "&lt;unsafe&gt;")

    def test_step_controls_show_one_complete_comparison_at_a_time(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is not installed")
        script = re.search(r"<script>(.*?)</script>", render_player(), re.S).group(1)
        harness = r"""
const assert=require('assert/strict');
function element(){return {textContent:'',disabled:false,hidden:false,listeners:{},
 classList:{toggle:()=>{}},addEventListener:function(k,fn){this.listeners[k]=fn},
 click:function(){if(!this.disabled)this.listeners.click()}};}
const ids={};for(const id of ['play','next','back','reset','status'])ids[id]=element();
const groups=Array.from({length:6},(_,i)=>Object.assign(element(),{dataset:{step:String(i)}}));
const document={getElementById:id=>ids[id],querySelectorAll:()=>groups};
let tick=null;
function setInterval(fn){tick=fn;return 1}function clearInterval(){tick=null}
new Function('document','setInterval','clearInterval',SCRIPT)(document,setInterval,clearInterval);
function check(i){assert.deepEqual(groups.map(g=>!g.hidden),groups.map((_,j)=>i===j));}
check(0);assert(ids.back.disabled);
for(let i=1;i<6;i++){ids.next.click();check(i);}
assert(ids.next.disabled);ids.back.click();check(4);
ids.reset.click();check(0);ids.play.click();
let guard=0;while(tick&&guard++<10)tick();
check(5);assert(ids.next.disabled);assert.equal(tick,null);
"""
        result = subprocess.run(
            [node, "-e", harness.replace("SCRIPT", json.dumps(script))],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
