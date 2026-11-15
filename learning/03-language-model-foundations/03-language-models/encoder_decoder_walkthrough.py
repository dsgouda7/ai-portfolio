"""Record the small chapter model and replay its actual reading and writing steps."""

import ast
import base64
from copy import deepcopy
from hashlib import sha256
import html
from io import BytesIO
import json
from pathlib import Path
import runpy

from PIL import Image
import torch
from torch import nn

from encoder_decoder_example import BOS, EOS, VOCABULARY, decode_ids, encode_words, teacher_inputs
from language_data import CAT_WORDS


HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
BASE = IMAGES / "03c-encoder-decoder-animation-base.png"
PLAYER = IMAGES / "03c-encoder-decoder-animation.html"
TRACE = PLAYER.with_suffix(".json")


def _values(tensor):
    return tensor.detach().reshape(-1).tolist()


def _norm(module):
    return sum(float(parameter.grad.square().sum()) for parameter in module.parameters()
               if parameter.grad is not None) ** 0.5


def record_run(model, *, teacher):
    """Keep the input policy visible; reference answers never enter generation."""
    snapshot = deepcopy(model).eval()
    source = torch.tensor([encode_words(CAT_WORDS)])
    target = torch.cat([source.flip(1), source.new_full((1, 1), EOS)], dim=1)
    expected = decode_ids(target[0])
    frames, state, saved = [], None, []
    with torch.no_grad():
        for position, word in enumerate(CAT_WORDS):
            previous = state
            _, state = snapshot.encoder(snapshot.embedding(source[:, position:position + 1]), state)
            saved.append(state.detach().clone())
            frames.append({
                "stage": "read", "title": f"Read source word {position + 1}: {word}",
                "narration": f"The encoder reads {word}. Its new note depends on this word and the previous note. "
                             "No answer word is predicted by the encoder.",
                "source_position": position, "encoder": _values(state), "route": "read",
                "old_encoder": _values(previous) if previous is not None else [0.] * state.shape[-1],
            })
        torch.testing.assert_close(state, snapshot.encode(source), rtol=1e-5, atol=1e-6)
        frames.append({
            "stage": "handoff", "title": "Pass the final note, not a word or weights",
            "narration": "All six source words have been read. The final encoder note becomes the decoder's "
                         "starting state. The decoder has separate learned weights.",
            "source_position": len(CAT_WORDS) - 1, "encoder": _values(state),
            "decoder": _values(state), "route": "handoff",
        })
        previous_id, guesses, scores_by_step = BOS, [], []
        for step, answer_id in enumerate(target[0].tolist()):
            old_state = state
            previous = source.new_full((1, 1), previous_id)
            logits, state = snapshot.decode(previous, state)
            scores = logits[0, 0]
            probabilities = scores.softmax(-1)
            guess = int(scores.argmax())
            error = float(nn.functional.cross_entropy(scores.unsqueeze(0), target[:, step]))
            common = {
                "writer_step": step, "source_position": len(CAT_WORDS) - 1,
                "encoder": _values(saved[-1]), "old_decoder": _values(old_state),
                "decoder": _values(state), "input_id": previous_id,
                "input_word": VOCABULARY[previous_id], "expected": expected[step],
                "guess_id": guess, "guess": VOCABULARY[guess],
                "correct_probability": float(probabilities[answer_id]), "loss": error,
                "scores": _values(scores), "produced": [*guesses, VOCABULARY[guess]],
            }
            frames.append({
                **common, "stage": "input", "title": f"Writer step {step + 1}: input {common['input_word']}",
                "narration": f"The decoder receives {common['input_word']} and its carried writing state. "
                             "It computes the next writing state; the separate head then scores answer words.",
                "route": "input",
            })
            frames.append({
                **common, "stage": "grade", "title": f"Actual guess: {common['guess']}. Check against {expected[step]}.",
                "narration": f"The actual highest-scoring word is {common['guess']}. "
                             f"The current target is {expected[step]}; its probability is "
                             f"{common['correct_probability']:.1%}. Grading does not replace the prediction.",
                "route": "grade",
            })
            scores_by_step.append(logits)
            guesses.append(VOCABULARY[guess])
            if not teacher and guess == EOS:
                frames[-1]["narration"] += " The model chose end, so generation stops."
                break
            next_id = answer_id if teacher else guess
            if step < target.shape[1] - 1:
                frames.append({
                    **common, "stage": "teacher" if teacher else "own",
                    "title": ("Teacher supplies " if teacher else "Feed back the chosen word ") + VOCABULARY[next_id],
                    "narration": (
                        f"Keep the actual guess {common['guess']} and its error. Supply the correct previous "
                        f"answer, {VOCABULARY[next_id]}, as the NEXT input. The writing state is not reset."
                        if teacher else
                        f"There is no teacher. The actual guess {VOCABULARY[next_id]} becomes the NEXT input. "
                        "The source stays fixed and the writing state continues."
                    ),
                    "next_input": VOCABULARY[next_id],
                    "route": "teacher" if teacher else "own",
                })
            previous_id = next_id
        if teacher:
            batched = snapshot(source, teacher_inputs(target))
            torch.testing.assert_close(torch.cat(scores_by_step, dim=1), batched, rtol=1e-5, atol=1e-6)
    return {
        "teacher": teacher, "source_words": list(CAT_WORDS), "required_words": expected,
        "guesses": guesses, "frames": frames,
    }


def record_training(model):
    """One separate copied-example update: no weights change during the recorded reads."""
    trial = deepcopy(model)
    source = torch.tensor([encode_words(CAT_WORDS)])
    target = torch.cat([source.flip(1), source.new_full((1, 1), EOS)], dim=1)
    before = {name: parameter.detach().clone() for name, parameter in trial.named_parameters()}
    optimizer = torch.optim.Adam(trial.parameters(), lr=0.004)
    optimizer.zero_grad()
    scores = trial(source, teacher_inputs(target))
    loss = nn.functional.cross_entropy(scores.flatten(0, 1), target.flatten())
    loss.backward()
    norms = {name: _norm(getattr(trial, name)) for name in ("embedding", "encoder", "decoder", "head")}
    if any(not torch.equal(parameter, before[name]) for name, parameter in trial.named_parameters()):
        raise RuntimeError("Backward unexpectedly changed model weights.")
    optimizer.step()
    with torch.no_grad():
        changed_scores = trial(source, teacher_inputs(target))
        after = float(nn.functional.cross_entropy(changed_scores.flatten(0, 1), target.flatten()))
    updates = {
        prefix: max(float((parameter.detach() - before[name]).abs().max())
                    for name, parameter in trial.named_parameters() if name.startswith(prefix + "."))
        for prefix in norms
    }
    if not all(value > 0 for value in norms.values()) or not after < float(loss.detach()):
        raise RuntimeError("The initialized-example update did not establish the teaching claim.")
    return {
        "loss_before": float(loss.detach()), "loss_after": after,
        "gradient_norms": norms, "weight_changes": updates,
        "backward_preserved_weights": True,
    }


def record_story(model):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        initial = type(model)(len(VOCABULARY))
    practice = record_run(initial, teacher=True)
    training = record_training(initial)
    final = practice["frames"][-1]
    for title, narration, route in (
        ("Follow every answer error backward",
         "All current-word errors contribute. Feedback goes from the head through decoder states, "
         "across the source-note handoff, and through encoder reads to their learned weights.", "backward"),
        ("Backward prepares feedback; weights are still unchanged",
         "Both encoder and decoder receive measured weight gradients. The word table and head do too. "
         "Backward has not edited any stored prediction or parameter yet.", "backward"),
        ("Now the optimizer changes the learned rules",
         f"One Adam step on this separate copy lowers its example loss from {training['loss_before']:.4f} "
         f"to {training['loss_after']:.4f}. A new read will compute new notes. This is not a quality benchmark.",
         "update"),
    ):
        practice["frames"].append({
            **final, "stage": route, "title": title, "narration": narration,
            "route": route, "training": training,
        })
    return {
        "version": 1,
        "background": {"file": BASE.name, "sha256": sha256(BASE.read_bytes()).hexdigest()},
        "origin": "The exact small EncoderDecoder class shown in 03C; seed 31; all vectors and guesses recorded.",
        "model_source_sha256": model_source_fingerprint(),
        "runs": {
            "practice": practice, "generation": record_run(initial, teacher=False),
            "trained": record_run(model, teacher=False),
        },
        "training": training,
    }


def model_source_fingerprint():
    source = (HERE / "03C-recurrent-encoder-decoder-to-attention.py").read_text(encoding="utf-8")
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.ClassDef) and node.name == "EncoderDecoder")
    return sha256(ast.dump(definition).encode("utf-8")).hexdigest()


def render_player(story):
    buffer = BytesIO()
    with Image.open(BASE) as image:
        image.save(buffer, format="WEBP", lossless=True, method=6)
    background = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = json.dumps(story, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return PLAYER_TEMPLATE.replace("__BACKGROUND__", background).replace("__DATA__", payload)


PLAYER_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Encoder, decoder and teacher: follow one real example</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#fff9ed;color:#17324d;font:16px/1.5 system-ui,sans-serif}
main{max-width:1536px;margin:auto}nav{position:sticky;top:0;z-index:2;background:#fffdf7;padding:12px;border-bottom:1px solid #d5c8ae}
button,select{font:inherit;border:1px solid #667d8c;border-radius:7px;background:white;color:#17324d;padding:7px 12px;cursor:pointer}
button:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible{outline:3px solid #7950a3;outline-offset:2px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}p{margin:5px 0}
#status{color:#007f86;font-weight:650}#caption{font-size:18px;min-height:84px;padding:12px 18px;background:#fffdf7}
svg{width:100%;height:auto;display:block}.panel{fill:#fffdf7;fill-opacity:.97;stroke-width:3}
text{font-family:system-ui,sans-serif;fill:#17324d}.label{font-size:25px;font-weight:650}.value{font-size:24px}.small{font-size:20px}
.route{fill:none;stroke-width:5;opacity:.22}.route.active{opacity:1;stroke-width:7;stroke-dasharray:12 8}
main[data-playing=true] .route.active{animation:flow 1.8s linear infinite}
.chip{fill:#fffdf7;stroke:#bac3c7;stroke-width:2}.chip.used{fill:#eef0ef;stroke:#bac3c7}
.source-chip.active{fill:#ddf4ef;stroke:#007f86;stroke-width:4}
.decoder-chip.active{fill:#fce8d3;stroke:#ad5b15;stroke-width:4}
main[data-run=practice] .decoder-chip.active{fill:#eee3f7;stroke:#7950a3}
#details{padding:10px 18px;background:#fffdf7}.note{font-size:14px;color:#596b72}
@keyframes flow{to{stroke-dashoffset:-40}}
@media(prefers-reduced-motion:reduce){main[data-playing=true] .route.active{animation:none}}
</style></head><body>
<main data-player="encoder-decoder-teacher" data-playing="false">
<nav aria-label="Recorded encoder-decoder walkthrough">
<div class="row"><button id="back" type="button" disabled>Back</button><button id="next" type="button">Next stage</button>
<button id="play" type="button" aria-pressed="false">Play slowly</button><button id="reset" type="button">Reset</button>
<label>Recording <select id="run"><option value="practice">Before learning: teacher forcing</option>
<option value="generation">Same weights: own guesses</option><option value="trained">After corpus training: own guesses</option></select></label>
<label>Pace <select id="pace"><option value="3000">3 seconds / stage</option><option value="5000">5 seconds / stage</option></select></label></div>
<p id="status" role="status"></p><p class="note">Starts paused. Use Next/Back or arrow keys. No live model or network call runs during playback.</p>
</nav>
<svg viewBox="0 0 1536 720" role="img" aria-labelledby="diagram-title diagram-description">
<title id="diagram-title">One source read, a learned state handoff, separate writing and teacher inputs</title>
<desc id="diagram-description">The source stays fixed after encoding. One active decoder input advances through previous answer words. Every state and prediction comes from actual PyTorch records.</desc>
<image href="__BACKGROUND__" x="0" y="0" width="1536" height="864"/>
<rect x="930" y="300" width="55" height="75" fill="#fffdf7"/>
<rect x="38" y="15" width="1460" height="85" rx="14" class="panel" stroke="#d7c6a9"/>
<text id="stage-title" x="64" y="49" class="label"></text><text x="64" y="83" class="small">Read first. Carry context. Write one answer word. Grade it before choosing the NEXT input.</text>
<text x="40" y="185" class="small">Source</text><g id="source-chips"></g>
<text x="24" y="243" class="small">Writer input</text><g id="decoder-input-chips"></g>
<path id="route-read" class="route" stroke="#007f86" d="M470 210 V248 H308 V272"/>
<path id="route-handoff" class="route" stroke="#007f86" d="M538 407 H980"/>
<path id="route-input" class="route" stroke="#ad5b15" d="M1448 368 H1405"/>
<path id="route-grade" class="route" stroke="#ad5b15" d="M1212 397 V450"/>
<path id="route-teacher" class="route" stroke="#7950a3" d="M765 600 V566 H1480 V368 H1410"/>
<path id="route-own" class="route" stroke="#ad5b15" d="M1225 542 V567 H1480 V368 H1410"/>
<path id="route-backward" class="route" stroke="#b33c38" d="M1200 490 H950 V295 H310 V275"/>
<path id="route-update" class="route" stroke="#b33c38" d="M310 275 H1200"/>
<rect x="80" y="275" width="455" height="265" rx="18" class="panel" stroke="#007f86"/>
<text x="104" y="316" class="label" fill="#007f86">Encoder / reader</text>
<text id="encoder-read" x="104" y="365" class="value"></text>
<text x="104" y="414" class="small">Current note: 64 features</text><text id="encoder-state" x="104" y="452" class="small"></text>
<text x="104" y="500" class="small">Same reader weights at every source word</text>
<rect x="595" y="337" width="330" height="142" rx="18" class="panel" stroke="#007f86"/>
<text x="620" y="378" class="label">Source note</text><text id="handoff-label" x="620" y="419" class="small"></text>
<text x="620" y="456" class="small">Not words or copied weights</text>
<rect x="985" y="275" width="465" height="265" rx="18" class="panel" stroke="#ad5b15"/>
<text x="1010" y="316" class="label">Decoder + separate word head</text>
<text id="decoder-input" x="1010" y="365" class="value"></text><text id="decoder-guess" x="1010" y="414" class="value"></text>
<text id="decoder-target" x="1010" y="463" class="value"></text><text id="decoder-probability" x="1010" y="507" class="small"></text>
<rect x="75" y="600" width="1375" height="77" rx="15" class="panel" stroke="#7950a3"/>
<text id="teacher-line" x="97" y="632" class="value"></text>
<text x="97" y="663" class="small">The source boundary stays fixed. Only the writer's previous-answer input advances.</text>
</svg>
<p id="caption" aria-live="polite"></p>
<details id="details"><summary>Recorded states and training values</summary><pre id="numbers"></pre></details>
<p class="note">Recorded from the actual initialized/trained class shown in the notebook. The before-learning update is an isolated copy, not the corpus run.</p>
</main>
<script id="recorded-flow" type="application/json">__DATA__</script>
<script>
(() => {
 const story=JSON.parse(document.getElementById("recorded-flow").textContent),root=document.querySelector("main");
 const runs=document.getElementById("run"),next=document.getElementById("next"),back=document.getElementById("back");
 const play=document.getElementById("play"),pace=document.getElementById("pace");
 let index=0,timer=null;
 const frames=()=>story.runs[runs.value].frames;
 function stop(){clearTimeout(timer);timer=null;root.dataset.playing="false";play.textContent="Play slowly";play.setAttribute("aria-pressed","false");}
 function text(id,value){document.getElementById(id).textContent=value;}
 function chip(group,word,position,x,y,kind,active,used){
  const rect=document.createElementNS("http://www.w3.org/2000/svg","rect");
  for(const [name,value] of Object.entries({x,y,width:150,height:kind==="source"?55:42,rx:10,
   class:"chip "+kind+"-chip"+(active?" active":used?" used":""),"data-position":position,"data-word":word}))rect.setAttribute(name,String(value));
  const label=document.createElementNS("http://www.w3.org/2000/svg","text");
  label.setAttribute("x",String(x+18));label.setAttribute("y",String(y+(kind==="source"?37:29)));
  label.setAttribute("class",kind==="source"?"value":"small");label.textContent=word;
  group.append(rect,label);
 }
 function show(){
  const recording=story.runs[runs.value],frame=frames()[index],source=recording.source_words;
  const reading=frame.stage==="read",supplying=["teacher","own"].includes(frame.stage);
  const writing=["input","grade"].includes(frame.stage)||supplying;
  const inputStep=writing?frame.writer_step+(supplying?1:0):null;
  root.dataset.index=String(index);root.dataset.stage=frame.stage;root.dataset.run=runs.value;
  text("status","Stage "+(index+1)+" / "+frames().length+" | "+runs.options[runs.selectedIndex].text);
  text("stage-title",frame.title);text("caption",frame.narration);
  text("encoder-read",reading?"Read "+(frame.source_position+1)+" / "+source.length+": "+source[frame.source_position]:
    "Source fixed: "+source.length+" / "+source.length+" read");
  text("encoder-state","["+frame.encoder.slice(0,3).map(value=>value.toFixed(3)).join(", ")+", ...]");
  document.getElementById("route-read").setAttribute("d","M"+(245+190*frame.source_position)+" 205 V210 H150 V265 H308 V275");
  if(writing){
   const x=245+170*inputStep;
   document.getElementById("route-input").setAttribute("d","M"+x+" 256 V265 H1212 V275");
   for(const name of ["teacher","own"]){
    document.getElementById("route-"+name).setAttribute("d",
     (name==="teacher"?"M765 600":"M1225 542")+" V566 H1480 V263 H"+x+" V256");
   }
  }
  text("handoff-label",frame.stage==="read"?"Still reading source":"Initial writing context");
  text("decoder-input",(supplying?"Next input: ":"Previous input: ")+
    (supplying?frame.next_input:frame.input_word||"not supplied yet"));
  const showGuess=!["read","handoff","input"].includes(frame.stage);
  text("decoder-guess","Actual guess: "+(showGuess?frame.guess:"not computed yet"));
  text("decoder-target","Check against: "+(showGuess?frame.expected:"not graded yet"));
  text("decoder-probability",showGuess?"Correct-word probability: "+(100*frame.correct_probability).toFixed(1)+"%":"Head scores come after the writer update");
  text("teacher-line",frame.stage==="teacher"?"Teacher supplies NEXT input: "+frame.next_input:
    frame.stage==="own"?"Own guess supplies NEXT input: "+frame.next_input:
    frame.stage==="backward"?"Follow answer errors backward into BOTH learned components":
    frame.stage==="update"?"Optimizer step: recompute notes with updated weights":"Grade first. Then choose the following input.");
  text("numbers",JSON.stringify({encoder:frame.encoder,decoder:frame.decoder||null,training:frame.training||null},null,2));
  document.querySelectorAll(".route").forEach(route=>route.classList.toggle("active",route.id==="route-"+frame.route));
  const group=document.getElementById("source-chips");group.replaceChildren();
  source.forEach((word,position)=>{
   chip(group,word,position,170+position*190,150,"source",
    reading&&position===frame.source_position,!reading||position<frame.source_position);
  });
  const inputs=document.getElementById("decoder-input-chips");inputs.replaceChildren();
  const recordedInputs=frames().filter(item=>item.stage==="input");
  recordedInputs.forEach((item,position)=>{
   const visible=recording.teacher||position===0||inputStep!==null&&position<=inputStep||["backward","update"].includes(frame.stage);
   chip(inputs,visible?item.input_word:"?",position,170+position*170,214,"decoder",
    position===inputStep,writing?position<inputStep:["backward","update"].includes(frame.stage));
  });
  back.disabled=index===0;next.disabled=index===frames().length-1;
 }
 function tick(){if(index===frames().length-1){stop();return;}index++;show();timer=setTimeout(tick,Number(pace.value));}
 next.addEventListener("click",()=>{stop();index=Math.min(index+1,frames().length-1);show();});
 back.addEventListener("click",()=>{stop();index=Math.max(0,index-1);show();});
 document.getElementById("reset").addEventListener("click",()=>{stop();index=0;show();});
 runs.addEventListener("change",()=>{stop();index=0;show();});
 pace.addEventListener("change",()=>{if(timer!==null){stop();}});
 play.addEventListener("click",()=>{if(timer!==null){stop();return;}if(index===frames().length-1){index=0;show();}
  root.dataset.playing="true";play.textContent="Pause";play.setAttribute("aria-pressed","true");timer=setTimeout(tick,Number(pace.value));});
 document.addEventListener("keydown",event=>{if(event.target.tagName==="SELECT")return;
  if(event.key==="ArrowRight"){event.preventDefault();next.click();}if(event.key==="ArrowLeft"){event.preventDefault();back.click();}});
 document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
 window.addEventListener("pagehide",stop);
 show();
})();
</script></body></html>
"""


def notebook_embed():
    raw = PLAYER.read_text(encoding="utf-8")
    if 'data-player="encoder-decoder-teacher"' not in raw:
        raise ValueError("The saved encoder-decoder player is missing its marker.")
    return (
        '<iframe title="Recorded encoder-decoder and teacher-forcing walkthrough" '
        'sandbox="allow-scripts" '
        f'srcdoc="{html.escape(raw, quote=True)}" '
        'style="display:block;width:100%;height:1000px;border:0" '
        f'data-recorded-walkthrough="{PLAYER.name}"></iframe>'
    )


def save_story(story):
    TRACE.write_text(json.dumps(story, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    PLAYER.write_text(render_player(story), encoding="utf-8", newline="\n")
    print("Saved recorded encoder-decoder walkthrough and trace.", flush=True)


def export():
    namespace = runpy.run_path(str(HERE / "03C-recurrent-encoder-decoder-to-attention.py"))
    save_story(record_story(namespace["model"]))


if __name__ == "__main__":
    export()
