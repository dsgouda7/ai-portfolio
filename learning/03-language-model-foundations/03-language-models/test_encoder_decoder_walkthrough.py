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
from encoder_decoder_walkthrough import BASE, PLAYER, TRACE, notebook_embed, record_run, record_story
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
        self.assertEqual(story["background"]["sha256"], sha256(BASE.read_bytes()).hexdigest())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_notebook_embeds_the_current_player_without_snapshot_fallback(self):
        book = json.loads((HERE / "03C-recurrent-encoder-decoder-to-attention.ipynb").read_bytes())
        cell = next(cell for cell in book["cells"] if cell["id"] == "ch03c-end-to-end-walkthrough")
        source = "".join(cell["source"])
        self.assertIn(notebook_embed(), source)
        self.assertNotIn("attachments", cell)
        self.assertNotIn("without JavaScript", source)
        metadata = cell["metadata"]["recorded_walkthrough"]
        self.assertEqual(metadata["recording_sha256"], sha256(TRACE.read_bytes()).hexdigest())
        recorder = HERE / "encoder_decoder_walkthrough.py"
        self.assertEqual(metadata["source_sha256"], sha256(recorder.read_bytes()).hexdigest())

    def test_player_waits_for_input_and_navigation_has_safe_boundaries(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is not installed")
        player = PLAYER.read_text(encoding="utf-8")
        script = player.split("<script>", 1)[1].split("</script>", 1)[0]
        story = json.loads(TRACE.read_bytes())
        harness = r"""
const input=JSON.parse(require('fs').readFileSync(0,'utf8')),assert=require('assert/strict'),vm=require('vm');
const ids={};function element(){const item={dataset:{},listeners:{},attributes:{},children:[],disabled:false,textContent:'',value:'',
 addEventListener:function(k,fn){this.listeners[k]=fn},setAttribute:function(k,v){this.attributes[k]=v},
 replaceChildren:function(){this.children=[]},append:function(...items){this.children.push(...items)},
 click:function(){if(!this.disabled)this.listeners.click()}};
 item.classList={contains:name=>(item.attributes.class||'').split(' ').includes(name),
 toggle:(name,on)=>{const classes=new Set((item.attributes.class||'').split(' ').filter(Boolean));
 if(on)classes.add(name);else classes.delete(name);item.attributes.class=[...classes].join(' ')}};
 return item;}
for(const name of ['recorded-flow','run','next','back','play','pace','status','stage-title','caption',
 'encoder-read','encoder-state','handoff-label','decoder-input','decoder-guess','decoder-target',
 'decoder-probability','teacher-line','numbers','source-chips','decoder-input-chips','reset'])ids[name]=element();
ids['recorded-flow'].textContent=JSON.stringify(input.story);
ids.run.value='practice';ids.run.selectedIndex=0;ids.run.options=[{text:'Practice'}];ids.pace.value='3000';
const root=element();let scheduled=0,tick=null,hidden=false;
const listeners={};const routes=['read','handoff','input','grade','teacher','own','backward','update'].map(name=>Object.assign(element(),{id:'route-'+name}));
for(const route of routes)ids[route.id]=route;
const document={getElementById:name=>ids[name],querySelector:()=>root,querySelectorAll:()=>routes,
 createElementNS:()=>element(),addEventListener:(name,fn)=>{listeners[name]=fn},get hidden(){return hidden}};
vm.runInNewContext(input.script,{document,window:{addEventListener:()=>{}},
 setTimeout:fn=>{scheduled++;tick=fn;return scheduled},clearTimeout:()=>{tick=null}});
assert.equal(scheduled,0);assert.equal(root.dataset.index,'0');assert.equal(ids.back.disabled,true);
function checkActiveInput(){
 const frame=input.story.runs[root.dataset.run].frames[Number(root.dataset.index)];
 const active=id=>ids[id].children.filter(item=>item.classList.contains('active'));
 const source=active('source-chips'),decoder=active('decoder-input-chips');
 assert.equal(source.length,frame.stage==='read'?1:0,'source highlight must stop after encoding');
 if(source.length)assert.equal(source[0].attributes['data-position'],String(frame.source_position));
 const writing=['input','grade','teacher','own'].includes(frame.stage);
 assert.equal(decoder.length,writing?1:0,'only one decoder input may be active');
 if(writing){
  const supplying=['teacher','own'].includes(frame.stage);
  assert.equal(decoder[0].attributes['data-position'],String(frame.writer_step+(supplying?1:0)));
  assert.equal(decoder[0].attributes['data-word'],supplying?frame.next_input:frame.input_word);
 }
}
checkActiveInput();
ids.next.click();assert.equal(root.dataset.index,'1');assert.equal(scheduled,0);
ids.back.click();assert.equal(root.dataset.index,'0');
ids.play.click();assert.equal(root.dataset.playing,'true');assert.equal(scheduled,1);
tick();assert.equal(root.dataset.index,'1');
hidden=true;listeners.visibilitychange();assert.equal(root.dataset.playing,'false');assert.equal(tick,null);
ids.reset.click();assert.equal(root.dataset.index,'0');
for(let i=1;i<input.story.runs.practice.frames.length;i++){ids.next.click();checkActiveInput();}
assert.equal(ids.next.disabled,true);const last=root.dataset.index;ids.next.click();assert.equal(root.dataset.index,last);
ids.run.value='generation';ids.run.listeners.change();assert.equal(root.dataset.index,'0');
assert.equal(root.dataset.run,'generation');assert.equal(root.dataset.playing,'false');
checkActiveInput();
for(let i=1;i<input.story.runs.generation.frames.length;i++){ids.next.click();checkActiveInput();}
ids.run.value='trained';ids.run.listeners.change();checkActiveInput();
for(let i=1;i<input.story.runs.trained.frames.length;i++){ids.next.click();checkActiveInput();}
console.log('manual playback, exact active input at every stage, run selection and boundaries verified');
"""
        result = subprocess.run(
            [node, "-e", harness], input=json.dumps({"story": story, "script": script}),
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
