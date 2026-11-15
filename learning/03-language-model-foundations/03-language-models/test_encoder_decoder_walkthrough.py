"""Verify that the teaching animation records the visible small model faithfully."""

import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import torch
from torch import nn

from encoder_decoder_example import BOS, EOS, VOCABULARY, decode_ids, encode_words, teacher_inputs
from encoder_decoder_walkthrough import BASE, PLAYER, TRACE, record_run, record_story
from language_data import CAT_WORDS


HERE = Path(__file__).parent


class RecordedReaderWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (HERE / "03C-recurrent-encoder-decoder-to-attention.py").read_text(encoding="utf-8")
        definition = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.ClassDef) and node.name == "EncoderDecoder")
        namespace = {"torch": torch, "nn": nn}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "<shown-small-model>", "exec"), namespace)
        cls.model_type = namespace["EncoderDecoder"]

    def model(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            return self.model_type(len(VOCABULARY))

    def test_recorded_teacher_inputs_follow_targets_but_generation_follows_actual_guesses(self):
        model = self.model()
        practice, generation = record_run(model, teacher=True), record_run(model, teacher=False)
        target = list(reversed(CAT_WORDS)) + ["<eos>"]
        for recording in (practice, generation):
            grades = [frame for frame in recording["frames"] if frame["stage"] == "grade"]
            inputs = [frame["input_word"] for frame in grades]
            self.assertEqual(inputs[0], "<bos>")
            self.assertEqual(inputs[1:], target[:-1] if recording["teacher"]
                             else [frame["guess"] for frame in grades[:-1]])
            for frame in grades:
                scores = torch.tensor(frame["scores"])
                answer = VOCABULARY.index(frame["expected"])
                self.assertEqual(frame["guess"], VOCABULARY[int(scores.argmax())])
                self.assertAlmostEqual(frame["correct_probability"], float(scores.softmax(-1)[answer]), places=6)
            if not recording["teacher"] and len(grades) < len(target):
                self.assertEqual(grades[-1]["guess"], "<eos>")
        self.assertNotEqual(practice["guesses"], target)
        teacher_frames = [frame for frame in practice["frames"] if frame["stage"] == "teacher"]
        self.assertEqual(teacher_frames[0]["next_input"], "mat")

    def test_whole_sequence_and_recorded_token_reads_have_the_same_state_and_scores(self):
        model = self.model()
        recording = record_run(model, teacher=True)
        source = torch.tensor([encode_words(CAT_WORDS)])
        target = torch.cat([source.flip(1), source.new_full((1, 1), EOS)], dim=1)
        reads = [frame for frame in recording["frames"] if frame["stage"] == "read"]
        grades = [frame for frame in recording["frames"] if frame["stage"] == "grade"]
        with torch.no_grad():
            state = model.encode(source)
            logits = model(source, teacher_inputs(target))
        torch.testing.assert_close(torch.tensor(reads[-1]["encoder"]), state.reshape(-1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.tensor([frame["scores"] for frame in grades]), logits[0],
                                   rtol=1e-5, atol=1e-6)

    def test_training_frames_show_real_feedback_without_changing_the_played_model(self):
        model = self.model()
        before = deepcopy(model.state_dict())
        story = record_story(model)
        training = story["training"]
        self.assertTrue(training["backward_preserved_weights"])
        self.assertTrue(all(value > 0 for value in training["gradient_norms"].values()))
        self.assertTrue(all(value > 0 for value in training["weight_changes"].values()))
        self.assertLess(training["loss_after"], training["loss_before"])
        self.assertEqual(story["background"]["model"], "MAI-Image-2.6")
        self.assertEqual(story["background"]["sha256"], sha256(BASE.read_bytes()).hexdigest())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_player_waits_for_input_and_navigation_has_safe_boundaries(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is not installed")
        player = PLAYER.read_text(encoding="utf-8")
        script = player.split("<script>", 1)[1].split("</script>", 1)[0]
        story = json.loads(TRACE.read_bytes())
        harness = r"""
const input=JSON.parse(require('fs').readFileSync(0,'utf8')),assert=require('assert/strict'),vm=require('vm');
const ids={};function element(){return{dataset:{},listeners:{},attributes:{},disabled:false,textContent:'',value:'',
 classList:{toggle:()=>{}},addEventListener:function(k,fn){this.listeners[k]=fn},
 setAttribute:function(k,v){this.attributes[k]=v},replaceChildren:()=>{},append:()=>{},
 click:function(){if(!this.disabled)this.listeners.click()}};}
for(const name of ['recorded-flow','run','next','back','play','pace','status','stage-title','caption',
 'encoder-read','encoder-state','handoff-label','decoder-input','decoder-guess','decoder-target',
 'decoder-probability','teacher-line','numbers','source-chips','reset'])ids[name]=element();
ids['recorded-flow'].textContent=JSON.stringify(input.story);
ids.run.value='practice';ids.run.selectedIndex=0;ids.run.options=[{text:'Practice'}];ids.pace.value='3000';
const root=element();let scheduled=0,tick=null,hidden=false;
const listeners={};const routes=['read','handoff','input','grade','teacher','own','backward','update'].map(name=>Object.assign(element(),{id:'route-'+name}));
ids['route-read']=routes[0];
const document={getElementById:name=>ids[name],querySelector:()=>root,querySelectorAll:()=>routes,
 createElementNS:()=>element(),addEventListener:(name,fn)=>{listeners[name]=fn},get hidden(){return hidden}};
vm.runInNewContext(input.script,{document,window:{addEventListener:()=>{}},
 setTimeout:fn=>{scheduled++;tick=fn;return scheduled},clearTimeout:()=>{tick=null}});
assert.equal(scheduled,0);assert.equal(root.dataset.index,'0');assert.equal(ids.back.disabled,true);
ids.next.click();assert.equal(root.dataset.index,'1');assert.equal(scheduled,0);
ids.back.click();assert.equal(root.dataset.index,'0');
ids.play.click();assert.equal(root.dataset.playing,'true');assert.equal(scheduled,1);
tick();assert.equal(root.dataset.index,'1');
hidden=true;listeners.visibilitychange();assert.equal(root.dataset.playing,'false');assert.equal(tick,null);
ids.reset.click();assert.equal(root.dataset.index,'0');
for(let i=1;i<input.story.runs.practice.frames.length;i++)ids.next.click();
assert.equal(ids.next.disabled,true);const last=root.dataset.index;ids.next.click();assert.equal(root.dataset.index,last);
ids.run.value='generation';ids.run.listeners.change();assert.equal(root.dataset.index,'0');
assert.equal(root.dataset.run,'generation');assert.equal(root.dataset.playing,'false');
console.log('manual-first playback, teacher/generation selection and boundary controls verified');
"""
        result = subprocess.run(
            [node, "-e", harness], input=json.dumps({"story": story, "script": script}),
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
