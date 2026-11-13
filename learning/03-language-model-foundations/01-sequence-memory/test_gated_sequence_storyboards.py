"""Independent equation, derivative, routing and standalone-playback checks."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from gated_sequence_storyboards import record_story, render_player


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_native_records_match_independent_numpy_gate_equations(kind):
    story = record_story(kind)
    assert story["initial_h"] == [0., 0.]
    assert story["initial_c"] == ([0., 0.] if kind == "lstm" else None)
    assert story["steps"][0]["retained"] == [0., 0.]
    assert story["input_size"] == 2
    assert [(s["note"], s["x"]) for s in story["steps"]] == [
        ("D3", [1., 0.]), ("A3", [0., 1.]), ("D4", [1., 1.]),
    ]
    weights = {k: np.array(v) for k, v in story["cell_parameters"].items()}
    assert weights["weight_ih"].shape[1] == 2
    assert np.array(story["input_gradients"]).shape == (3, 2)
    head = {k: np.array(v) for k, v in story["head_parameters"].items()}
    h = np.array(story["initial_h"])
    c = np.array(story["initial_c"]) if kind == "lstm" else None
    for step in story["steps"]:
        a = weights["weight_ih"] @ np.array(step["x"]) + weights["bias_ih"]
        b = weights["weight_hh"] @ h + weights["bias_hh"]
        if kind == "lstm":
            i, f, g, o = np.split(a + b, 4)
            keep, write, proposal, show = _sigmoid(f), _sigmoid(i), np.tanh(g), _sigmoid(o)
            retained, written = keep * c, write * proposal
            c = retained + written
            h = show * np.tanh(c)
            np.testing.assert_allclose(step["c"], c, atol=1e-14)
            np.testing.assert_array_equal(step["carry_c"], step["c"])
            np.testing.assert_allclose(step["show"], show, atol=1e-14)
        else:
            xr, xz, xn = np.split(a, 3)
            hr, hz, hn = np.split(b, 3)
            reset, keep = _sigmoid(xr + hr), _sigmoid(xz + hz)
            write, proposal = 1 - keep, np.tanh(xn + reset * hn)
            retained, written = keep * h, write * proposal
            h = retained + written
            np.testing.assert_allclose(step["reset"], reset, atol=1e-14)
            np.testing.assert_allclose(step["reset_branch"], reset * hn, atol=1e-14)
            assert not np.allclose(h, np.tanh(h), rtol=0, atol=1e-14)
        for key, expected in (("keep", keep), ("write", write), ("proposal", proposal),
                              ("retained", retained), ("written", written), ("h", h)):
            np.testing.assert_allclose(step[key], expected, atol=1e-14)
        assert not np.isclose(keep[0], keep[1], rtol=0, atol=1e-8)
        np.testing.assert_array_equal(step["carry_h"], step["head_input"])
        np.testing.assert_array_equal(step["head_input"], step["h"])
        logits = head["weight"] @ h + head["bias"]
        probability = np.exp(logits - max(logits))
        probability /= probability.sum()
        np.testing.assert_allclose(step["logits"], logits, atol=1e-14)
        np.testing.assert_allclose(step["probabilities"], probability, atol=1e-14)
    assert story["loss"] == pytest.approx(-np.log(probability[0]), abs=1e-14)


def _independent_graph(story, delta_h=0., delta_c=0.):
    p = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
         for k, v in story["cell_parameters"].items()}
    h0 = torch.tensor(story["initial_h"], dtype=torch.float64, requires_grad=True)
    h = h0 + torch.tensor([delta_h, 0.], dtype=torch.float64)
    two = story["kind"] == "lstm"
    c0 = torch.tensor(story["initial_c"], dtype=torch.float64, requires_grad=True) if two else None
    c = c0 + torch.tensor([delta_c, 0.], dtype=torch.float64) if two else None
    states = []
    for step in story["steps"]:
        x = torch.tensor(step["x"], dtype=torch.float64)
        a = p["weight_ih"] @ x + p["bias_ih"]
        b = p["weight_hh"] @ h + p["bias_hh"]
        if two:
            i, f, g, o = (a + b).chunk(4)
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = o.sigmoid() * c.tanh()
            c.retain_grad()
        else:
            ar, az, an = a.chunk(3)
            br, bz, bn = b.chunk(3)
            reset = (ar + br).sigmoid()
            keep = (az + bz).sigmoid()
            h = keep * h + (1 - keep) * (an + reset * bn).tanh()
        h.retain_grad()
        states.append((h, c))
    head = {k: torch.tensor(v, dtype=torch.float64) for k, v in story["head_parameters"].items()}
    logits = head["weight"] @ h + head["bias"]
    loss = torch.logsumexp(logits, 0) - logits[0]
    return loss, states, h0, c0, p


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_final_only_autograd_and_carry_routes_match_independent_graph(kind):
    story = record_story(kind)
    loss, states, h0, c0, params = _independent_graph(story)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(story["loss"], abs=1e-14)
    for name, value in params.items():
        np.testing.assert_allclose(story["parameter_gradients"][name], value.grad, atol=1e-14)
    for t, (step, (h, c)) in enumerate(zip(story["steps"], states)):
        feedback = step["feedback"]
        np.testing.assert_allclose(feedback["dh"], h.grad, atol=1e-14)
        if t < 2:
            assert feedback["dlogits"] == [0., 0., 0.]
            assert feedback["head_dh"] == [0., 0.]
            np.testing.assert_allclose(story["steps"][t + 1]["old_h"], step["h"], atol=0)
        else:
            expected_dlogits = np.array(step["probabilities"]) - np.array([1., 0., 0.])
            np.testing.assert_allclose(feedback["dlogits"], expected_dlogits, atol=1e-14)
            expected_dh = np.array(story["head_parameters"]["weight"]).T @ expected_dlogits
            np.testing.assert_allclose(feedback["head_dh"], expected_dh, atol=1e-14)
        previous_h = h0.grad if t == 0 else states[t - 1][0].grad
        np.testing.assert_allclose(feedback["to_previous_h"], previous_h, atol=1e-14)
        assert np.linalg.norm(feedback["to_previous_h"]) > 0
        if kind == "lstm":
            np.testing.assert_allclose(feedback["dc"], c.grad, atol=1e-14)
            joined = np.array(feedback["future_dc"]) + feedback["through_show_dc"]
            np.testing.assert_allclose(feedback["dc"], joined, atol=1e-14)
            np.testing.assert_allclose(feedback["to_previous_c"],
                                       np.array(step["keep"]) * joined, atol=1e-14)
            # Native c-output feedback excludes this step's internal SHOW branch.
            if t < 2:
                np.testing.assert_allclose(feedback["future_dc"],
                                           story["steps"][t + 1]["feedback"]["to_previous_c"], atol=1e-14)
            else:
                assert feedback["future_dc"] == [0., 0.]
                assert np.linalg.norm(feedback["dc"]) > 0
        else:
            joined = np.array(feedback["direct_h_carry"]) + feedback["proposal_and_gate_dh"]
            np.testing.assert_allclose(feedback["to_previous_h"], joined, atol=1e-14)
            assert np.linalg.norm(feedback["proposal_and_gate_dh"]) > .01
    np.testing.assert_allclose(story["initial_feedback"]["dh"], h0.grad, atol=1e-14)
    eps = 1e-6
    plus = float(_independent_graph(story, delta_h=eps)[0].detach())
    minus = float(_independent_graph(story, delta_h=-eps)[0].detach())
    assert story["initial_feedback"]["dh"][0] == pytest.approx((plus - minus) / (2 * eps), abs=2e-10)
    if kind == "lstm":
        np.testing.assert_allclose(story["initial_feedback"]["dc"], c0.grad, atol=1e-14)
        plus = float(_independent_graph(story, delta_c=eps)[0].detach())
        minus = float(_independent_graph(story, delta_c=-eps)[0].detach())
        assert story["initial_feedback"]["dc"][0] == pytest.approx((plus - minus) / (2 * eps), abs=2e-10)


def test_deterministic_records_do_not_change_rng_or_load_a_notebook():
    before = torch.random.get_rng_state().clone()
    for kind in ("lstm", "gru"):
        first = json.dumps(record_story(kind), sort_keys=True, allow_nan=False)
        assert first == json.dumps(record_story(kind), sort_keys=True, allow_nan=False)
        assert "Hand-picked" in first
    assert torch.equal(before, torch.random.get_rng_state())
    source = Path(__file__).with_name("gated_sequence_storyboards.py").read_text(encoding="utf-8")
    assert ".ipynb" not in source
    assert "exec(" not in source


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_fixed_three_column_board_is_self_contained_and_exported_from_same_records(kind):
    player = render_player(kind)
    svg = re.search(r"<svg\b.*?</svg>", player, re.S).group()
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    columns = root.findall("s:g[@class='note-column']", ns)
    assert len(columns) == 3
    assert [c.attrib["data-note"] for c in columns] == ["D3", "A3", "D4"]
    data = json.loads(re.search(r'<script id="story-data" type="application/json">(.*?)</script>',
                               player, re.S).group(1))
    assert data == record_story(kind)
    assert [f["column"] for f in data["frames"] if f["phase"] == "backward"] == [2, 1, 0]
    assert len([s for s in data["steps"] if s["graded"]]) == 1
    for i, col in enumerate(columns):
        labels = [e.text for e in col.findall(".//s:text", ns)]
        assert any(f"Saved h{i + 1}" in label for label in labels)
        if kind == "lstm":
            assert any(f"Saved c{i + 1}" in label for label in labels)
        assert col.find("s:g[@data-role='feedback']", ns) is not None
    assert not root.findall(".//s:image", ns)
    for prohibited in ('src=', 'fetch(', 'XMLHttpRequest', 'innerHTML', 'replaceChildren'):
        assert prohibited not in player
    assert "data:image/webp;base64," in player
    assert "data:font/ttf;base64," in player
    assert '<footer class="paper">' in player
    assert not re.search(r'(?:src|href)=["\']https?://', player)
    for control in ("play", "next", "back", "reset"):
        assert f'id="{control}"' in player
    assert len(player.encode()) < 1_000_000
    prefix = Path(__file__).parent / "images" / f"01a-{kind}-sequence-storyboard"
    if prefix.with_suffix(".html").exists():
        from PIL import Image
        assert prefix.with_suffix(".html").read_text(encoding="utf-8") == player
        with Image.open(prefix.with_suffix(".png")) as image:
            assert float(image.info["Loss"]) == data["loss"]
            assert image.info["StoryKind"] == kind
            assert json.loads(image.info["InitialFeedback"]) == data["initial_feedback"]
            assert image.width > image.height


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_compact_board_has_unscaled_readable_active_arithmetic(kind):
    player = render_player(kind)
    root = ET.fromstring(re.search(r"<svg\b.*?</svg>", player, re.S).group())
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    assert width <= 1200 and height < 700
    assert root.attrib["preserveAspectRatio"] == "xMidYMin meet"
    assert f'<html lang="en" data-player="{kind}-sequence-storyboard">' in player
    assert player.index("</svg>") < player.index('id="active-values"')
    assert 'id="active-values"' not in ET.tostring(root, encoding="unicode")
    # Saved values remain about 14 CSS pixels even in the actual 705px pane.
    ns = {"s": "http://www.w3.org/2000/svg"}
    for column in root.findall("s:g[@class='note-column']", ns):
        state_values = column.findall("s:g[@data-role='save']/s:text[@font-weight='bold']", ns)
        assert len(state_values) == (2 if kind == "lstm" else 1)
        assert all(float(v.attrib["font-size"]) * 705 / width >= 14 for v in state_values)
    style = re.search(r"<style>(.*?)</style>", player, re.S).group(1)
    active_style = re.search(r"\.active-value\{(.*?)\}", style, re.S).group(1)
    assert "22px" in active_style and "vw" not in active_style
    assert "LessonHand" in active_style
    assert "overflow:hidden" not in style
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in style
    visible = re.sub(r"<script\b.*?</script>", "", player, flags=re.S)
    assert "future_dc" not in visible and "dc joined" not in visible
    story = record_story(kind)
    assert all(len(frame["readout"]) == 4 for frame in story["frames"])
    blend = next(f for f in story["frames"] if f["focus"] == "blend")
    step = story["steps"][0]
    old = step["old_c" if kind == "lstm" else "old_h"]
    for n in (0, 1):
        expected = f"{step['keep'][n]:.3f} × {old[n]:.3f} = {step['retained'][n]:.3f}"
        assert blend["readout"][2 * n]["value"] == expected
    backward = next(f for f in story["frames"] if f["phase"] == "backward")
    assert backward["readout"][0]["label"] == "Feedback to h3"
    if kind == "lstm":
        assert backward["readout"][1]["label"] == "Feedback to c3"
        expected = "[" + ", ".join(f"{n:+.4f}" for n in story["steps"][2]["feedback"]["dc"]) + "]"
        assert backward["readout"][1]["value"] == expected


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_player_controls_execute_without_replacing_any_cards(kind):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed; structural HTML checks still run.")
    player = render_player(kind)
    svg = ET.fromstring(re.search(r"<svg\b.*?</svg>", player, re.S).group())
    ns = {"s": "http://www.w3.org/2000/svg"}
    groups = [g.attrib for g in svg.findall(".//s:g[@class='stage-group']", ns)]
    data = re.search(r'<script id="story-data" type="application/json">(.*?)</script>', player, re.S).group(1)
    script = re.findall(r"<script>(.*?)</script>", player, re.S)[0]
    harness = r"""
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const assert=require('assert/strict');
function element(){
 const classes=new Set();
 return {style:{}, dataset:{}, textContent:'', disabled:false, listeners:{}, attrs:{},
  classList:{toggle:(k,on)=>on?classes.add(k):classes.delete(k),contains:k=>classes.has(k)},
  addEventListener:function(k,fn){this.listeners[k]=fn},
  setAttribute:function(k,v){this.attrs[k]=v},
  click:function(){if(!this.disabled)this.listeners.click({});}};
}
const ids={};for(const id of ['phase','narration','play','next','back','reset','progress','story-data'])ids[id]=element();
for(let i=0;i<4;i++){ids['active-value-'+i]=element();ids['value-label-'+i]=element();}
ids['story-data'].textContent=input.data;
const groups=input.groups.map(a=>Object.assign(element(),{dataset:{reveal:a['data-reveal'],role:a['data-role'],column:a['data-column']}}));
const columns=[element(),element(),element()];const original=columns.slice();
const document={getElementById:id=>ids[id],querySelectorAll:q=>q==='.stage-group'?groups:columns,
 body:element(),addEventListener:()=>{}};
let tick=null;
function setInterval(fn){tick=fn;return 7}
function clearInterval(){tick=null}
new Function('document','setInterval','clearInterval',input.script)(document,setInterval,clearInterval);
const story=JSON.parse(input.data);
function checkReadout(step){
 story.frames[step].readout.forEach((v,i)=>{
  assert.equal(ids['active-value-'+i].textContent,v.value);
  assert.equal(ids['value-label-'+i].textContent,v.label);
  assert.equal(ids['active-value-'+i].attrs['data-tone'],v.tone);
 });
}
assert(ids.back.disabled);assert(ids.phase.textContent.startsWith('Start'));
checkReadout(0);
for(let i=0;i<16;i++){ids.next.click();checkReadout(i+1);}
assert(ids.phase.textContent.startsWith('Grade only'));
for(let i=0;i<3;i++){ids.next.click();assert(columns[2-i].classList.contains('active'));checkReadout(17+i);}
assert(groups.every(g=>Number(g.style.opacity)>0));
ids.next.click();assert(ids.next.disabled);assert(ids.phase.textContent.startsWith('Complete'));
ids.back.click();assert(ids.phase.textContent.startsWith('Backward'));
ids.reset.click();assert(ids.back.disabled);assert(ids.phase.textContent.startsWith('Start'));
checkReadout(0);
ids.play.click();assert.equal(ids.play.textContent,'Pause');assert(tick);
tick();assert(ids.phase.textContent.startsWith('Forward'));
ids.play.click();assert.equal(tick,null);assert.equal(ids.play.attrs['aria-pressed'],'false');
ids.play.click();let guard=0;while(tick&&guard++<30)tick();
assert(ids.next.disabled);assert.equal(ids.play.textContent,'Play');
assert.deepEqual(columns,original);console.log('controls and fixed cards passed');
"""
    result = subprocess.run([node, "-e", harness], input=json.dumps({
        "data": data, "script": script, "groups": groups,
    }), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "controls and fixed cards passed" in result.stdout
