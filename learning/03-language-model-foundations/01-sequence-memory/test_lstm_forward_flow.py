"""Verify that the visual LSTM states and prediction routes are real and distinct."""

import base64
from copy import deepcopy
from dataclasses import asdict
from html import unescape
from io import BytesIO
import json
from math import hypot
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree

import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageSequence
import torch

from backprop_memory_flow import NOTES, NOTE_HEAD_WEIGHTS, _inputs_and_weights
from lstm_forward_flow import (
    ART, ART_MANIFEST, GIF, NOTEBOOK, PHASES, PLAYER, STORYBOARD, ForwardFrame,
    export_player, export_storyboard, load_cell_and_inputs, note_readout,
    flow_signals, gif_figure, gif_frames, gif_scene, read_recording, record_reads, render_player, role_images, scene,
    vector, write_forward_steps,
)
from lstm_layer_flow import record_lstm_layer_step
from lstm_state_roles import load_state_roles
from notebook_replays import LSTM_FORWARD_FILE, refresh_embedded_replays, saved_replay


class LSTMForwardFlowTests(unittest.TestCase):
    def test_manual_steps_copy_completed_frames_without_replaying_the_next_read(self):
        for gif_path in (GIF, GIF.with_name("01a-gru-forward-walkthrough.gif")):
            with self.subTest(gif=gif_path.name), TemporaryDirectory() as directory:
                path = Path(directory) / "steps.html"
                write_forward_steps(path, gif_path)
                html = path.read_text(encoding="utf-8")
                stages = json.loads(re.search(
                    r'<script id="step-data" type="application/json">(.*?)</script>', html, re.S,
                )[1])
                with Image.open(gif_path) as image:
                    frames = json.loads(image.info["comment"])["frames"]
                    expected = [index for index, frame in enumerate(frames)
                                if frame["read_index"] == 2 and frame["progress"] == 1]
                    self.assertEqual(len(stages), len(expected))
                    for stage, index in zip(stages, expected, strict=True):
                        image.seek(index)
                        raw = base64.b64decode(stage["image"].split(",", 1)[1], validate=True)
                        with Image.open(BytesIO(raw)) as snapshot:
                            expected_frame = image.convert("RGB")
                            if "crop" in stage:
                                expected_frame = expected_frame.crop(stage["crop"])
                            self.assertIsNone(ImageChops.difference(
                                expected_frame, snapshot.convert("RGB"),
                            ).getbbox())
                self.assertIn("SHOW = 0" if gif_path == GIF else "RESET = 0",
                              " ".join(stage["label"] for stage in stages))

    def test_gru_manual_steps_wait_for_navigation_and_stop_at_each_boundary(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is not installed")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "steps.html"
            write_forward_steps(path, GIF.with_name("01a-gru-forward-walkthrough.gif"))
            html = path.read_text(encoding="utf-8")
        stages = json.loads(re.search(
            r'<script id="step-data" type="application/json">(.*?)</script>', html, re.S,
        )[1])
        script = re.findall(r"<script>\s*(.*?)</script>", html, re.S)[0]
        harness = r"""
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const assert=require('assert/strict'),vm=require('vm');
const elements={};
for(const id of ['step-data','snapshot','status','back','next','reset']){
 elements[id]={listeners:{},disabled:false,textContent:'',src:'',alt:'',
  addEventListener:function(event,fn){this.listeners[event]=fn;},
  click:function(){if(!this.disabled)this.listeners.click();}};
}
elements['step-data'].textContent=JSON.stringify(input.stages);
let keydown;
vm.runInNewContext(input.script,{
 document:{getElementById:id=>elements[id],addEventListener:(event,fn)=>{keydown=fn;}},
 setTimeout:()=>assert.fail('Manual walkthrough scheduled a timeout'),
 setInterval:()=>assert.fail('Manual walkthrough scheduled an interval')
});
assert.equal(elements.snapshot.src,input.stages[0].image);
assert.equal(elements.back.disabled,true);
assert.match(elements.status.textContent,/Step 1 \//);
elements.next.click();
assert.equal(elements.snapshot.src,input.stages[1].image);
elements.back.click();
assert.equal(elements.snapshot.src,input.stages[0].image);
keydown({key:'ArrowRight',preventDefault:()=>{}});
assert.equal(elements.snapshot.src,input.stages[1].image);
keydown({key:'ArrowLeft',preventDefault:()=>{}});
assert.equal(elements.snapshot.src,input.stages[0].image);
for(let i=1;i<input.stages.length;i++)elements.next.click();
assert.equal(elements.next.disabled,true);
const last=elements.snapshot.src;
elements.next.click();
assert.equal(elements.snapshot.src,last);
elements.reset.click();
assert.equal(elements.snapshot.src,input.stages[0].image);
assert.equal(elements.back.disabled,true);
assert.equal(elements.next.disabled,false);
console.log('manual forward/back, keyboard, reset and no autoplay verified');
"""
        result = subprocess.run([node, "-e", harness], input=json.dumps({
            "script": script, "stages": stages,
        }), text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lstm_playback_waits_for_play_and_supports_pause_timeline_and_replay(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is not installed")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "steps.html"
            write_forward_steps(path, GIF)
            html = path.read_text(encoding="utf-8")
        stages = json.loads(re.search(
            r'<script id="step-data" type="application/json">(.*?)</script>', html, re.S,
        )[1])
        script = re.findall(r"<script>\s*(.*?)</script>", html, re.S)[0]
        self.assertNotIn('id="next"', html)
        harness = r"""
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const assert=require('assert/strict'),vm=require('vm'),elements={};
for(const id of ['step-data','snapshot','status','caption','back','play','pause','reset','stage','pace','zoom']){
 elements[id]={listeners:{},disabled:false,textContent:'',src:'',alt:'',value:'',
  addEventListener:function(event,fn){this.listeners[event]=fn;},
  setAttribute:function(name,value){this[name]=value;},
  click:function(){if(!this.disabled)this.listeners.click();}};
}
elements['step-data'].textContent=JSON.stringify(input.stages);elements.pace.value='4000';
const root={dataset:{fit:'page',playing:'false'}},listeners={};
let timer=null,scheduled=0,hidden=false;
vm.runInNewContext(input.script,{
 document:{getElementById:id=>elements[id],querySelector:()=>root,
  addEventListener:(event,fn)=>{listeners[event]=fn;},get hidden(){return hidden;}},
 window:{addEventListener:()=>{}},
 setTimeout:(fn,delay)=>{assert.equal(delay,4000);scheduled++;timer=fn;return scheduled;},
 clearTimeout:()=>{timer=null;}
});
assert.equal(scheduled,0);assert.equal(root.dataset.index,'0');
elements.play.click();assert.equal(root.dataset.playing,'true');assert.equal(elements.play.disabled,true);
assert.equal(root.dataset.index,'0');timer();assert.equal(root.dataset.index,'1');
elements.pause.click();assert.equal(timer,null);assert.equal(root.dataset.playing,'false');
elements.stage.value='4';elements.stage.listeners.input();
assert.equal(elements.snapshot.src,input.stages[4].image);assert.equal(timer,null);
elements.back.click();assert.equal(root.dataset.index,'3');
elements.zoom.click();assert.equal(root.dataset.fit,'width');assert.equal(elements.zoom.textContent,'Fit page');
elements.zoom.click();assert.equal(root.dataset.fit,'page');
elements.play.click();hidden=true;listeners.visibilitychange();assert.equal(timer,null);
elements.reset.click();assert.equal(root.dataset.index,'0');assert.equal(root.dataset.playing,'false');
elements.play.click();for(let i=1;i<input.stages.length;i++)timer();
assert.equal(root.dataset.index,String(input.stages.length-1));assert.equal(timer,null);
assert.equal(elements.play.textContent,'Replay');assert.equal(elements.pause.disabled,true);
elements.play.click();assert.equal(root.dataset.index,'0');
console.log('LSTM play/pause, timeline, fit toggle, background pause and replay verified');
"""
        result = subprocess.run([node, "-e", harness], input=json.dumps({
            "script": script, "stages": stages,
        }), text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_values_and_both_carried_states_match_each_read(self):
        cell, inputs = load_cell_and_inputs()
        original_inputs, _, _ = _inputs_and_weights(NOTEBOOK)
        torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
        hidden = torch.zeros((1, 2), dtype=torch.float64)
        memory = torch.zeros_like(hidden)
        reads = record_reads()
        self.assertEqual(tuple(read.note for read in reads), NOTES)
        for offset, read in enumerate(reads):
            with torch.no_grad():
                actual_h, actual_c = cell(inputs[offset:offset + 1], (hidden, memory))
            self.assertEqual(read.inputs, tuple(inputs[offset].tolist()))
            self.assertEqual(read.old_hidden, tuple(hidden[0].tolist()))
            self.assertEqual(read.old_cell, tuple(memory[0].tolist()))
            self.assertEqual(read.new_hidden, tuple(actual_h[0].tolist()))
            self.assertEqual(read.new_cell, tuple(actual_c[0].tolist()))
            self.assertEqual(len(read.new_hidden), len(read.new_cell))
            self.assertEqual(len(read.new_cell), 2)
            self.assertEqual(read.next_note, NOTES[(offset + 1) % 3])
            if offset:
                self.assertEqual(read.old_hidden, reads[offset - 1].new_hidden)
                self.assertEqual(read.old_cell, reads[offset - 1].new_cell)
            hidden, memory = actual_h, actual_c

    def test_gate_arithmetic_and_head_read_the_correct_values(self):
        for read in record_reads():
            for index in range(2):
                self.assertAlmostEqual(read.retained[index], read.keep[index] * read.old_cell[index], places=14)
                self.assertAlmostEqual(read.written[index], read.write[index] * read.proposal[index], places=14)
                self.assertAlmostEqual(read.new_cell[index], read.retained[index] + read.written[index], places=14)
                expected_h = read.show[index] * torch.tanh(torch.tensor(read.new_cell[index], dtype=torch.float64)).item()
                self.assertAlmostEqual(read.new_hidden[index], expected_h, places=14)
            hidden = torch.tensor(read.new_hidden, dtype=torch.float64)
            weights = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=torch.float64)
            self.assertEqual(read.readout.scores, tuple((weights @ hidden).tolist()))
            self.assertEqual(read.readout.probabilities, tuple((weights @ hidden).softmax(0).tolist()))
            self.assertNotEqual(read.readout, note_readout(torch.tensor(read.new_cell, dtype=torch.float64)))

    def test_a3_snapshot_is_the_existing_lessons_native_step(self):
        actual = record_reads()[1]
        existing = load_state_roles().baseline
        self.assertEqual(actual.old_hidden, tuple(existing.old_hidden.tolist()))
        self.assertEqual(actual.old_cell, tuple(existing.old_cell.tolist()))
        self.assertEqual(actual.new_cell, tuple(existing.carried_cell.tolist()))
        self.assertEqual(actual.new_hidden, tuple(existing.carried_hidden.tolist()))

    def test_old_c_supplies_content_but_does_not_choose_standard_gates(self):
        cell, inputs = load_cell_and_inputs()
        read = record_reads()[1]
        hidden = torch.tensor([read.old_hidden], dtype=torch.float64)
        memory = torch.tensor([read.old_cell], dtype=torch.float64)
        baseline = record_lstm_layer_step(cell, inputs[1:2], hidden, memory)
        delta = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        altered = record_lstm_layer_step(cell, inputs[1:2], hidden, memory + delta)
        for name in ("input_gate", "forget_gate", "candidate", "output_gate"):
            torch.testing.assert_close(getattr(baseline, name), getattr(altered, name), rtol=0, atol=0)
        torch.testing.assert_close(
            altered.new_cell - baseline.new_cell, baseline.forget_gate * delta[0], rtol=1e-12, atol=1e-12,
        )
        self.assertFalse(torch.equal(altered.new_hidden, baseline.new_hidden))
        changed_h = record_lstm_layer_step(cell, inputs[1:2], hidden + delta, memory)
        self.assertEqual(tuple(changed_h.old_cell.tolist()), read.old_cell)
        self.assertFalse(torch.equal(changed_h.input_mix + changed_h.state_mix, baseline.input_mix + baseline.state_mix))

    def test_closing_show_preserves_c_but_removes_its_exposure_to_the_head(self):
        for read in record_reads():
            self.assertEqual(read.closed_show_cell, read.new_cell)
            self.assertNotEqual(read.new_cell, (0, 0))
            self.assertEqual(read.closed_show_hidden, (0, 0))
            self.assertEqual(read.closed_show_readout.scores, (0, 0, 0))
            self.assertEqual(read.closed_show_readout.probabilities, (1 / 3,) * 3)

    def test_diagram_dependencies_require_new_h_before_prediction(self):
        read = record_reads()[1]
        boxes, wires, _ = scene(read)
        destinations = {box.name: box for box in boxes}
        controls = next(box for box in boxes if box.name == "controls")
        self.assertEqual(controls.labels[0].text, "Compute controls")
        self.assertIn("Shared weights", controls.labels[1].text)
        incoming = {}
        for wire in wires:
            incoming.setdefault(wire.target, set()).add(wire.source)
        self.assertEqual(incoming["controls"], {"input", "old-hidden"})
        self.assertEqual(incoming["prediction"], {"new-hidden"})
        self.assertEqual(incoming["new-cell"], {"add"})
        self.assertEqual(incoming["add"], {"kept", "written"})
        self.assertEqual(incoming["new-hidden"], {"show"})
        self.assertEqual(incoming["show"], {"new-cell", "controls"})
        self.assertEqual(incoming["next-cell"], {"new-cell"})
        self.assertEqual(incoming["next-hidden"], {"new-hidden"})
        self.assertEqual(destinations["next-cell"].labels[-1].text, vector(read.new_cell))
        self.assertEqual(destinations["next-hidden"].labels[-1].text, vector(read.new_hidden))

    def test_recording_is_deterministic_without_global_rng_or_thread_changes(self):
        rng = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        self.assertEqual(record_reads(), record_reads())
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        data = read_recording()
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["state_width"], 2)
        self.assertEqual(data["head_weights"], [list(row) for row in NOTE_HEAD_WEIGHTS])
        self.assertEqual(data["steps"], json.loads(json.dumps([asdict(read) for read in record_reads()])))
        self.assertEqual(len(data["scenes"]), 3)
        self.assertEqual(len(PHASES), 8)
        for normal, closed, rows in zip(data["steps"], data["closed_steps"], data["readouts"], strict=True):
            self.assertEqual(len(rows), 8)
            self.assertEqual(normal["new_cell"], closed["new_cell"])
            self.assertEqual(closed["new_hidden"], [0., 0.])
            self.assertEqual(closed["show"], [0., 0.])
            self.assertIn("KEEP * old c", rows[2])
            self.assertIn("WRITE * proposed content", rows[3])
        for svg in data["scenes"]:
            root = ElementTree.fromstring(svg)
            self.assertEqual(root.attrib["viewBox"], "0 0 1110 790")
            self.assertEqual(len([node for node in root.iter("image")]), 2)
            self.assertEqual(len([node for node in root.iter("polyline") if node.attrib.get("data-to") == "prediction"]), 1)
            wires = {node.attrib["id"]: node for node in root.iter("polyline")}
            self.assertIn("control-signal", wires["write-control"].attrib["class"])
            self.assertIn("content-signal", wires["proposal-to-write"].attrib["class"])
            self.assertNotEqual(wires["write-control"].attrib["data-phases"],
                                wires["keep-control"].attrib["data-phases"])
            self.assertEqual(len(list(root.iter("animateMotion"))), len(wires))

    def test_player_steps_preview_and_playback_preserve_the_recorded_state_routes(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is not installed")
        player = render_player()
        data = read_recording()
        root = ElementTree.fromstring(data["scenes"][0])
        elements = [
            {"phases": element.attrib["data-phases"],
             "signal": "signal" in element.attrib.get("class", "").split()}
            for element in root.iter() if "data-phases" in element.attrib
        ]
        script = re.findall(r"<script>\s*(.*?)</script>", player, re.S)[0]
        harness = r"""
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const assert=require('assert/strict');
function run(reduced){
 function element(){
  const classes=new Set();
  return {dataset:{},children:[],textContent:'',disabled:false,hidden:false,listeners:{},
   classList:{toggle:(k,on)=>on?classes.add(k):classes.delete(k),contains:k=>classes.has(k)},
   addEventListener:function(k,fn){this.listeners[k]=fn},
   setAttribute:function(k,v){this[k]=v},append:function(v){this.children.push(v)},
   replaceChildren:function(...v){this.children=v},
   click:function(){if(!this.disabled)this.listeners.click({})}};
 }
 const ids={};for(const id of ['recorded-lstm','scene','play','pause','step','back','reset',
  'preview','preview-note','motion-note','problem','phase','progress','active-lines','cell-output',
  'hidden-output','prediction-output','cell-formula','hidden-formula','head-formula','proof'])ids[id]=element();
 const main=element();main.dataset.playing='false';
 const buttons=Array.from({length:3},(_,i)=>Object.assign(element(),{dataset:{note:String(i)}}));
 const elements=input.elements.map(v=>Object.assign(element(),{dataset:{phases:v.phases},signal:v.signal}));
 let motions=0,paused=0,tick=null;
 const svg={pauseAnimations:()=>paused++,unpauseAnimations:()=>{}};
 ids.scene.querySelector=()=>svg;
 ids.scene.querySelectorAll=q=>q==='[data-phases]'?elements:
  elements.filter(e=>e.signal&&e.classList.contains('active')).map(()=>({beginElement:()=>motions++}));
 ids['recorded-lstm'].textContent=JSON.stringify(input.data);
 const document={hidden:false,getElementById:id=>ids[id],querySelector:()=>main,
  querySelectorAll:()=>buttons,createElement:()=>element(),addEventListener:()=>{}};
 const window={matchMedia:()=>({matches:reduced}),addEventListener:()=>{}};
 new Function('document','window','setTimeout','clearTimeout',input.script)(
  document,window,fn=>{tick=fn;return 1},()=>{tick=null});
 function check(note,phase){
  assert.equal(main.dataset.step,String(note+1));assert.equal(main.dataset.phase,String(phase));
  assert.deepEqual(ids['active-lines'].children.map(e=>e.textContent),input.data.readouts[note][phase]);
 }
 check(0,0);assert(ids.back.disabled);assert.equal(ids['cell-output'].textContent,'Not computed yet');
 for(let phase=1;phase<=5;phase++){ids.step.click();check(0,phase);}
 assert.equal(ids.preview.disabled,false);
 ids.preview.click();assert.equal(main.dataset.preview,'true');
 assert.deepEqual(JSON.parse(main.dataset.cell),input.data.steps[0].new_cell);
 assert.deepEqual(JSON.parse(main.dataset.hidden),[0,0]);
 ids.step.click();check(0,6);assert.equal(main.dataset.preview,'false');
 ids.preview.click();assert.equal(ids['prediction-output'].textContent,'Uniform scores: no preferred note');
 assert.deepEqual(JSON.parse(main.dataset.probabilities),[1/3,1/3,1/3]);
 ids.step.click();check(0,7);assert.equal(main.dataset.preview,'false');
 ids.step.click();check(1,0);ids.back.click();check(0,7);
 ids.reset.click();check(0,0);buttons[1].click();check(1,0);
 if(reduced){
  assert(!ids.play.disabled);assert(ids.pause.disabled);assert.equal(motions,0);
  assert.equal(ids['motion-note'].hidden,false);
  ids.play.click();assert(motions>0);assert(tick);ids.pause.click();assert.equal(tick,null);
 }else {
  ids.play.click();check(0,0);assert(tick);assert(motions>0);
  ids.pause.click();assert.equal(tick,null);assert.equal(main.dataset.playing,'false');assert(paused>0);
  ids.pause.click();let guard=0;while(tick&&guard++<30)tick();
  assert(ids.step.disabled);assert.equal(main.dataset.phase,'all');
  assert.deepEqual(JSON.parse(main.dataset.cell),input.data.steps[2].new_cell);
  assert.deepEqual(JSON.parse(main.dataset.hidden),input.data.steps[2].new_hidden);
 }
}
run(false);run(true);console.log('LSTM controls, data routes and reduced motion passed');
"""
        result = subprocess.run([node, "-e", harness], input=json.dumps({
            "data": data, "script": script, "elements": elements,
        }), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LSTM controls, data routes and reduced motion passed", result.stdout)

    def test_exports_are_self_contained_and_reproducible(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / PLAYER.name
            export_player(path)
            self.assertEqual(path.read_bytes(), PLAYER.read_bytes())
            self.assertEqual(render_player(), PLAYER.read_text(encoding="utf-8"))
            png = path.with_suffix(".png")
            export_storyboard(png)
            with Image.open(STORYBOARD) as saved, Image.open(png) as rebuilt:
                self.assertEqual(saved.tobytes(), rebuilt.tobytes())
                self.assertEqual(saved.info["HeadInput"], "new_hidden")
                self.assertEqual(saved.info["StateWidth"], "2")
        for role, image in role_images().items():
            self.assertIn(role, ("cell", "hidden"))
            self.assertEqual(image.size, (240, 180))
        manifest = json.loads(ART_MANIFEST.read_text(encoding="utf-8"))[0]
        self.assertEqual(manifest["model"], "MAI-Image-2.6")
        self.assertEqual(manifest["filename"], ART.name)

    def test_gif_routes_wait_for_the_contributions_before_exposing_results(self):
        self.assertEqual({name for name, _ in flow_signals(4, .25)},
                         {"kept-to-add", "written-to-add"})
        self.assertEqual({name for name, _ in flow_signals(4, .75)}, {"add-to-cell"})
        self.assertEqual({name for name, _ in flow_signals(5, .25)},
                         {"cell-to-show", "show-control"})
        self.assertEqual({name for name, _ in flow_signals(5, .75)}, {"show-to-hidden"})
        read = record_reads()[1]
        for progress in (0., 1.):
            figure = gif_figure(read, ForwardFrame(2, 2, progress, 120))
            try:
                texts = [text.get_text() for text in figure.axes[0].texts]
                self.assertEqual(vector(read.retained) in texts, progress == 1.)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                labels = [text for text in figure.axes[0].texts if text.get_text()]
                for index, text in enumerate(labels):
                    bounds = text.get_window_extent(renderer)
                    self.assertTrue(figure.bbox.contains(bounds.x0, bounds.y0), text.get_text())
                    self.assertTrue(figure.bbox.contains(bounds.x1, bounds.y1), text.get_text())
                    for other in labels[index + 1:]:
                        self.assertFalse(bounds.overlaps(other.get_window_extent(renderer)),
                                         (text.get_text(), other.get_text()))
            finally:
                plt.close(figure)
        for phase, progress in ((-1, 0), (8, 0), (0, -1), (0, 1.1)):
            with self.assertRaises(ValueError):
                flow_signals(phase, progress)

    def test_gif_spacing_gives_links_clear_lanes_without_changing_dependencies(self):
        read = record_reads()[1]
        boxes, wires, _ = gif_scene(read)
        original_boxes, original_wires, _ = scene(read)
        self.assertEqual({box.name for box in boxes}, {box.name for box in original_boxes} | {"head"})
        self.assertEqual([(wire.name, wire.source, wire.target) for wire in wires],
                         [(wire.name, wire.source, "head" if wire.name == "hidden-to-head" else wire.target)
                          for wire in original_wires] + [("head-to-prediction", "head", "prediction")])
        bounds = {box.name: box.bounds for box in boxes}
        for left, right in (("input", "controls"), ("old-hidden", "controls"),
                            ("controls", "written"), ("written", "show")):
            x, _, width, _ = bounds[left]
            self.assertGreaterEqual(bounds[right][0] - x - width, 80)
        for top, bottom in (("new-cell", "show"), ("show", "new-hidden"),
                            ("new-hidden", "prediction")):
            _, y, _, height = bounds[top]
            self.assertGreaterEqual(bounds[bottom][1] - y - height, 120)
        for wire in wires:
            length = sum(hypot(end[0] - start[0], end[1] - start[1])
                         for start, end in zip(wire.points, wire.points[1:]))
            self.assertGreaterEqual(length, 90, wire.name)
            for start, end in zip(wire.points, wire.points[1:]):
                self.assertTrue(start[0] == end[0] or start[1] == end[1], wire.name)
                for box in boxes:
                    if box.name in (wire.source, wire.target):
                        continue
                    x, y, width, height = box.bounds
                    if start[1] == end[1]:
                        blocked = y - 10 < start[1] < y + height + 10 and (
                            max(min(start[0], end[0]), x - 10) < min(max(start[0], end[0]), x + width + 10))
                    else:
                        blocked = x - 10 < start[0] < x + width + 10 and (
                            max(min(start[1], end[1]), y - 10) < min(max(start[1], end[1]), y + height + 10))
                    self.assertFalse(blocked, (wire.name, box.name))
        figure = gif_figure(read, ForwardFrame(2, 6, 1., 1600))
        try:
            arrows = [text.arrow_patch for text in figure.axes[0].texts
                      if getattr(text, "arrow_patch", None) is not None]
            self.assertTrue(arrows)
            self.assertTrue(all(arrow.get_linewidth() >= 2.2 for arrow in arrows))
        finally:
            plt.close(figure)

    def test_gif_show_comparison_preserves_storage_and_restores_normal_next_read(self):
        read = record_reads()[1]
        figure = gif_figure(read, ForwardFrame(2, 6, 1., 1600, True))
        try:
            texts = [text.get_text() for text in figure.axes[0].texts]
            self.assertIn(vector(read.new_cell), texts)
            self.assertIn(vector(read.closed_show_hidden), texts)
            self.assertIn("Uniform: no preferred note", texts)
        finally:
            plt.close(figure)
        carried = [frame for frame in gif_frames() if frame.phase == 7]
        self.assertTrue(carried)
        self.assertTrue(all(not frame.close_show for frame in carried))
        next_read = record_reads()[2]
        self.assertEqual(next_read.old_cell, read.new_cell)
        self.assertEqual(next_read.old_hidden, read.new_hidden)

    def test_linear_head_scores_appear_before_softmax_and_note_choice(self):
        read = record_reads()[1]
        boxes, wires, _ = gif_scene(read)
        head = next(box for box in boxes if box.name == "head")
        self.assertEqual(head.ready_progress, .5)
        self.assertEqual({wire.source for wire in wires if wire.target == "head"}, {"new-hidden"})
        self.assertEqual({wire.source for wire in wires if wire.target == "prediction"}, {"head"})
        for progress in (.25, .5, 1.):
            figure = gif_figure(read, ForwardFrame(2, 6, progress, 120))
            try:
                texts = [text.get_text() for text in figure.axes[0].texts]
                self.assertEqual(vector(read.readout.scores) in texts, progress >= .5)
                self.assertEqual(f"Next note: {read.readout.prediction} (max)" in texts, progress == 1.)
                self.assertEqual("scores = W_out h2" in texts, progress >= .5)
            finally:
                plt.close(figure)
        self.assertEqual({name for name, _ in flow_signals(6, .25)}, {"hidden-to-head"})
        self.assertEqual({name for name, _ in flow_signals(6, .75)}, {"head-to-prediction"})

    def test_gif_visited_blocks_and_links_stay_bright_through_comparison_and_next_read(self):
        reads = record_reads()
        bright = set()
        for frame in gif_frames():
            if frame.progress != 1. and not (frame.phase in (4, 5) and frame.progress == .75):
                continue
            figure = gif_figure(reads[frame.read_index - 1], frame)
            try:
                axis = figure.axes[0]
                artists = list(axis.patches) + list(axis.lines)
                artists += [text.arrow_patch for text in axis.texts
                            if getattr(text, "arrow_patch", None) is not None]
                styles = {artist.get_gid(): artist.get_alpha() for artist in artists
                          if artist.get_gid() and artist.get_gid().startswith(("node-", "wire-"))}
                for name in bright:
                    self.assertEqual(styles[name], 1., (frame, name))
                bright.update(name for name, alpha in styles.items() if alpha == 1.)
                if frame.phase == 0 and frame.read_index == 2:
                    self.assertLess(styles["node-new-cell"], 1.)
                    self.assertLess(styles["wire-hidden-to-head"], 1.)
                if frame.close_show:
                    self.assertEqual(styles["node-prediction"], 1.)
                    self.assertEqual(styles["wire-hidden-to-head"], 1.)
            finally:
                plt.close(figure)
        boxes, wires, _ = gif_scene(reads[1])
        self.assertEqual(bright, {f"node-{box.name}" for box in boxes}
                         | {f"wire-{wire.name}" for wire in wires})

    def test_saved_gif_contains_real_stages_visible_motion_and_bounded_playback(self):
        specifications = gif_frames()
        with Image.open(GIF) as image:
            metadata = json.loads(image.info["comment"])
            self.assertEqual(metadata["steps"], json.loads(json.dumps([asdict(read) for read in record_reads()])))
            self.assertEqual(metadata["frames"], [asdict(frame) for frame in specifications])
            self.assertEqual(metadata["head_input"], "new_hidden")
            self.assertEqual(image.info["loop"], 0)
            self.assertEqual(image.n_frames, len(specifications))
            self.assertLessEqual(image.n_frames, 64)
            total = 0
            for frame, expected in zip(ImageSequence.Iterator(image), specifications, strict=True):
                self.assertEqual(frame.info["duration"], expected.duration_ms)
                total += frame.info["duration"]
            self.assertGreaterEqual(total, 20_000)
            self.assertLessEqual(total, 35_000)
            image.seek(0)
            first = image.convert("RGB")
            image.seek(3)
            self.assertIsNotNone(ImageChops.difference(first, image.convert("RGB")).getbbox())
        self.assertLess(GIF.stat().st_size, 4_000_000)

    def test_native_markdown_embed_has_no_code_box_and_protects_manual_edits(self):
        result = saved_replay(LSTM_FORWARD_FILE).data
        self.assertIn('sandbox="allow-scripts"', result)
        self.assertNotIn("allow-same-origin", result)
        document = unescape(re.search(r'srcdoc="([^"]+)"', result)[1])
        self.assertIn('data-player="lstm-forward-walkthrough"', document)
        self.assertIn('data-from=\\"new-hidden\\"', document)
        self.assertNotIn('src="http', document)
        self.assertNotIn('href="http', document)
        display = {"cell_type": "markdown", "source": [], "metadata": {"saved_replay": {"file": LSTM_FORWARD_FILE}}}
        book = {"cells": [display]}
        refresh_embedded_replays(book)
        self.assertNotIn("outputs", display)
        self.assertNotIn("execution_count", display)
        self.assertNotIn("\n", display["source"][0])
        snapshot = deepcopy(book)
        refresh_embedded_replays(book)
        self.assertEqual(book, snapshot)
        display["source"][0] += "manual change"
        with self.assertRaisesRegex(ValueError, "Markdown player changed"):
            refresh_embedded_replays(book)

    def test_invalid_head_inputs_are_reported(self):
        for hidden in (torch.zeros(3), torch.tensor([float("nan"), 0])):
            with self.assertRaises(ValueError):
                note_readout(hidden)


if __name__ == "__main__":
    unittest.main()
