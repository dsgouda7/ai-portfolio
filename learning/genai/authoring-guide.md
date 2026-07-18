# GenAI Notebooks — Authoring Guide

> **Gold standard (mechanistic depth)**: [02-transformers/transformers.ipynb](02-transformers/transformers.ipynb)
> **Gold standard (narrative framing)**: [04-llm/01-llm-finetuning.ipynb](04-llm/01-llm-finetuning.ipynb)
> Every notebook under `learning/genai/` should be brought to the same pedagogical flow,
> intuition-building, and technical depth as these notebooks. This guide extracts the
> reusable patterns so they can be applied consistently across the folder. Section 8 covers
> the narrative/business-framing techniques specifically; Sections 1-7 cover the
> mechanistic "prove, don't assert" techniques common to both.

This is not a generic "notebook style guide." It is a distillation of *why* the gold
standard notebook works as a teaching artifact, written so the patterns can be copied
deliberately rather than imitated by vibes.

---

## 1 · Core Philosophy

The gold standard notebook treats every concept as something the reader should be made
to **want** before it is given to them. Nothing is asserted; almost everything is
**measured, shown, or proven** with code and a plot. The reader is walked through the
same sequence of curiosity → crude attempt → complaint → refinement → payoff that a
person discovering the idea for themselves would experience.

Three non-negotiable habits fall out of that philosophy:

1. **Don't assert — demonstrate.** Every claim ("residuals help gradient flow",
   "multi-head lets heads specialise", "√dₖ prevents saturation") is followed by a code
   cell that measures the effect and prints/plots the number that proves it.
2. **One running example, threaded throughout.** A single toy sentence
   (`"the cat sat on the mat"`) and a single 3D "semantic space" embedding scheme carries
   every concept from tokenisation to GPT-2 internals. New mechanisms are always shown
   operating *on the same tokens* the reader already knows.
3. **Toy first, real second, same code.** Every mechanism is built by hand in a tiny,
   visualisable space (3 dimensions, 6 tokens) *before* the notebook shows the identical
   mechanism running inside a real pretrained model (DistilGPT-2). The bridge cell is
   always explicit: "same Q/K/V, same softmax(QKᵀ/√dₖ) — just wider vectors."

If a candidate notebook skips straight to "here is the formula, here is the code," it is
not at parity with the gold standard, no matter how correct the code is.

---

## 2 · Structural Skeleton

Reuse this shape when restructuring a notebook. Not every notebook needs every part —
adapt to the topic — but the *skeleton logic* (motivate → build small → prove → scale →
summarise) should survive.

| Block | Purpose |
|---|---|
| **Title + roadmap table** | One Markdown cell: title, one-paragraph pitch, and a `\| Step \| Concept \| Key Idea \|` table listing every part of the notebook in order. This is the reader's map — written *last*, after the notebook is finished, so it's accurate. |
| **Setup cell(s)** | A dependency-install cell that only installs what's missing (`try: import X except ImportError: pip install`), then one imports+seeding cell. Deterministic seeds (`np.random.seed`, `tf.random.set_seed`) are set once, up top. |
| **Part 1 — establish the substrate** | Define the running example: vocabulary, embeddings, or dataset. Explain *why* the toy space has the dimensionality/shape it does (e.g. 3D axes each mean something human-readable). |
| **Parts 2..N — one mechanism per part** | Each part = one Markdown "why do we need this" cell → one or more code cells that build/visualise/prove it → a short reflection cell ("What just happened — and what's missing") that creates the crack the *next* part will pry open. |
| **Toy → real bridge** | A comparison table (toy dims vs. GPT-2/production dims) and then the same mechanism run against a real pretrained model. |
| **Exercises** | 🧪 "Your turn" cells scattered after the concepts they exercise, not bunched at the end. |
| **Summary** | A final Markdown cell: the roadmap table again but completed, plus a bulleted "Key insights to keep" list — each bullet is a one-line, quotable takeaway, not a restatement of the section title. |

---

## 3 · Markdown Cell Conventions

- **Section headers** use `## Part N — Title` for major sections, `### Na. Subtitle` for
  subsections (letters for sub-steps within a numbered part: `3a`, `3b`, `3d`…).
- **Horizontal rules (`---`)** open every new major Part.
- **Tables over prose** whenever comparing ≥2 things (RNN vs Transformer, toy vs GPT-2,
  Encoder vs Decoder vs Encoder-Decoder). Tables are skimmable; paragraphs are for the
  one idea that can't be tabulated.
- **Math is always followed by a plain-English gloss** within the same or next cell —
  never leave a LaTeX block to speak for itself. Example pattern:
  > $$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})\,V$$
  > "The √dₖ scaling prevents dot-products from growing so large that softmax saturates…"
- **Recurring emoji callouts** — reuse these exact markers, they are load-bearing
  signposts a returning reader will pattern-match on:
  - `🔮 Predict first` — pose a concrete, falsifiable question *before* the reveal cell.
    Always phrased so the reader can be wrong (e.g. give 2–3 candidate outcomes).
  - `🧪 Your turn — <topic>` — a change-one-variable-and-predict exercise, placed
    immediately after the concept it drills, with an inline `# 👉 CHANGE ...` comment
    in the code cell telling the reader exactly what knob to turn.
  - `#### What just happened` / `#### So they differ — but…` — a short reflective
    cell after a reveal that (a) names what was just shown and (b) plants the question
    the next part answers. This is the "complaint that forces the next step."
- **"Predict, then verify" cadence**: a 🔮 cell is never immediately followed by the
  answer in the same cell — the reader must run code to find out. Don't spoil it in the
  markdown.
- **Numbered comparison callouts** ("Problem 1 — …", "Problem 2 — …") when motivating why
  a design choice (e.g. cross-attention) beats a naïve alternative.

---

## 4 · Code Cell Conventions

- **Section-banner comments**: every code cell opens with
  `# ── Short Description ─────────────────────────────────────────` to visually chunk
  the notebook when skimming source. Sub-comments below explain *why*, not just *what*
  (`# The √dₖ scaling prevents softmax saturation`, not `# divide by sqrt`).
- **Print statements are part of the pedagogy, not debug noise.** Every demo cell ends
  with a few `print()` lines that state the takeaway in words, often with a `→` or `->`
  arrow leading into the conclusion:
  ```python
  print("  → A single head serves ONE relation well, never both.")
  print("  → Concatenating [Head P ; Head C] delivers BOTH targets in parallel.")
  ```
  A reader who only reads output (no plots) should still get the point.
  For example, the "predict then verify" exercises should print whether the actual
  outcome matched the prediction to give closure and check understanding.
- **Claims get measured.** If a paragraph says "residual connections keep gradients
  alive," the very next cell builds the smallest experiment that produces a number
  proving it (e.g. 24-layer stack, with/without skip, gradient norm at layer 1,
  log-scale plot). Avoid hand-waved assertions with no accompanying measurement.
- **Visualisation stack**: `matplotlib` + `seaborn` heatmaps (`sns.heatmap` with
  `annot=True`) for anything matrix-shaped (attention weights, masks); `plotly`
  `Scatter3d` for interactive 3D with a `HAS_PLOTLY` try/except **matplotlib fallback**
  so the notebook still runs without the optional dependency; `FuncAnimation` +
  `HTML(ani.to_jshtml(default_mode="loop"))` for anything that unfolds over steps
  (attention forming, RoPE rotating). Animations `plt.close(fig)` before returning the
  HTML to avoid a duplicate static frame.
- **Multi-panel figures** (`plt.subplots` or `plt.GridSpec`) for anything with a
  before/after or A/B/C comparison — never make the reader mentally diff two separate
  cells' output when they can be side by side.
- **Deterministic seeding** (`tf.random.set_seed(N)` / `np.random.seed(N)`) immediately
  before any cell whose numbers are quoted in the surrounding markdown, so re-running
  the notebook reproduces the exact prose.
- **Small, real classes, not black boxes.** `MultiHeadAttention`, `FeedForward`,
  `TransformerBlock`, `CrossAttention` are hand-written Keras layers a reader can read
  top-to-bottom in under a minute — never an opaque library call standing in for the
  concept being taught.
- **Naming mirrors the math.** Variables are `W_Q`, `Q`, `K`, `V`, `d_k`, `attn_w` — not
  `weight1`, `out`, `x2`. The code should be legible against the LaTeX above it.

---

## 5 · Pedagogical Techniques Worth Copying

1. **"Build it crude, then let the complaint drive the next version."** The RoPE
   animation section (3d) is the clearest example: Attempt 1 (one dial) is deliberately
   too primitive, the cell's own printed output states the complaint
   ("a real query vector has 3 pairs… I want more"), and the next cell fixes exactly
   that gap. Use this instead of dropping the reader into the polished final artifact.
2. **"Prove the claim, don't state it."** Multi-head specialisation, W_V as a relevance
   filter, residual connections enabling depth, √dₖ preventing saturation — every one of
   these is validated with a tiny constructed experiment and a printed/plotted number,
   not a sentence of assertion.
3. **Toy/real parity table.** Before touching a real pretrained model, show a table
   mapping every toy hyperparameter to its production equivalent (`d_model: 3 → 16 → 768`,
   `heads: — → 2 → 12`). This is what makes "if you understood the toy, you understand
   GPT-2" a credible sentence instead of a platitude.
4. **Same weights, one variable changed.** When comparing two behaviors (encoder vs.
   decoder mask, RoPE vs. no RoPE), reuse the *same* learned/random projections and only
   flip the one variable under test. Side-by-side heatmaps generated this way isolate
   the causal factor cleanly — comparisons across differently-seeded models are avoided
   because they conflate multiple differences.
5. **Reflection cells that plant the next question.** Almost every part ends with a
   short Markdown cell naming what's missing or what a sharp reader would object to,
   which becomes the hook for the next part's opening paragraph.
6. **Exercises are drills, not new content.** 🧪 cells never introduce a new concept —
   they let the reader turn a knob (`my_query`, `n_heads`, `DEPTH`, `USE_ENCODER_MASK`)
   on a mechanism already explained, with an explicit prediction prompt and a printed
   correctness check where feasible.
7. **A running "Summary" table that mirrors the intro table.** The notebook opens with a
   roadmap table and closes with the same table restated as a completed journey, plus a
   short list of quotable one-line insights (not restated headings).

---

## 6 · Checklist — Is a Notebook At Parity?

Use this when auditing or authoring a `plan.md` for a candidate notebook:

- [ ] Opens with a title + roadmap table pitching the whole notebook in one screen.
- [ ] Has a single running example/task threaded through every section (not a new
      dataset or sentence per part).
- [ ] Every non-trivial claim is followed by code that measures/proves it, not just
      states it.
- [ ] Uses 🔮 "Predict first" before any reveal that has a non-obvious answer.
- [ ] Has 🧪 "Your turn" exercises placed near the concept they drill, with a
      `# 👉 CHANGE ...` comment and (where possible) a printed correctness check.
- [ ] Reflection cells ("What just happened", "So — but…") close each part and open a
      crack for the next.
- [ ] If a mechanism has a toy and a real-world form, both are shown, bridged by an
      explicit parameter-mapping table.
- [ ] Math is never left un-glossed; every formula has a plain-English explanation
      within the same or the next cell.
- [ ] Visualisations favor heatmaps/multi-panel comparisons over single unremarkable
      plots; optional interactive (plotly) visuals degrade gracefully to matplotlib.
- [ ] Code cells use section-banner comments, math-mirroring variable names, and
      pedagogical print statements (not silent computation).
- [ ] Ends with a completed roadmap table + a short "Key insights to keep" bullet list.
- [ ] Seeds are set deterministically wherever quoted numbers appear in markdown.

---

## 7 · How This Guide Is Used

Each notebook under `learning/genai/` (other than the gold standard itself) should have
a companion `plan.md` in its own folder, written against this checklist, that:

1. Summarises the notebook's current state (topic, structure, what pedagogy already
   exists).
2. Scores/flags it against Section 6's checklist.
3. Lists concrete, ordered changes (add a 🔮 cell here, add a toy→real bridge there,
   replace an assertion with a proof, restructure into numbered Parts, etc.) needed to
   reach parity with [02-transformers/transformers.ipynb](02-transformers/transformers.ipynb)
   and, where the notebook has a real-world use case to motivate, [04-llm/01-llm-finetuning.ipynb](04-llm/01-llm-finetuning.ipynb).
4. Is scoped to that notebook only — it should not require changes to other notebooks.

---

## 8 · Narrative & Business-Stakes Framing (from `04-llm/01-llm-finetuning.ipynb`)

Sections 1-7 describe how to build *intuition* for a mechanism. This section describes a
complementary technique: giving the reader a reason to *care* which technique wins, by binding
the entire notebook to one concrete, named scenario with real constraints instead of a neutral
tour of options. `01-llm-finetuning.ipynb` is the reference example — a small publishing firm,
"Riverside House," wants an in-house editing assistant and knowledge base trained on its own
unpublished manuscripts, on a laptop CPU, with no data allowed to leave the building.

### 8.1 One brief, one set of constraints, threaded through every section

The opening Markdown cell states a **named client, a concrete ask, and a hard constraint**
(confidential data that can never leave the building; a laptop CPU, not a GPU cluster) instead of
a generic "here's what fine-tuning is." Every technique introduced later is framed as **answering
one of that client's questions**, not as an entry in a taxonomy:

- Continued pretraining → "Does it even know our characters and world exist?"
- Instruction tuning → "Does it follow a request instead of rambling?"
- Preference alignment → "Does it write the way our editors actually prefer?"
- Partial freezing / LoRA → "IT gave us one laptop — what can we actually afford to train?"

Each major Markdown section opens with a one-line **"Riverside's question for this section"**
callout before any code or theory — the reader always knows *why* they're about to read this part
before they read it.

### 8.2 Real numbers, never illustrative ones

Where Section 5 says "prove the claim, don't state it," this notebook takes that a step further:
**every chart is generated from the actual model trained earlier in the same notebook**, never a
hand-drawn "typical" curve. Loss curves come from `trainer.state.log_history` of the real
`Trainer` run above; per-block weight deltas come from diffing `real_model.named_parameters()`
against the untouched base checkpoint; the LoRA "before/after" activation comparison is captured
with real forward hooks on the actual PEFT-wrapped layer, not a re-derivation of the matrix math.
Code comments say this explicitly ("real numbers, not a toy example") so the reader never mistakes
a real result for a staged one.

### 8.3 "Crack it open" — verify a training claim on the real trained weights

After every training run, a follow-up cell measures whether the run actually did what its config
claimed, using the same untouched base model as a reference point: diff every trained parameter
against its value in the untouched base checkpoint, per transformer block, and plot the norm of
that difference. For partial freezing this directly confirms the frozen blocks measured a real
delta of effectively zero — not "should be frozen," but "measured frozen." Reuse this pattern
anywhere a notebook makes a claim about *which* weights change: don't just show the config, measure
the actual diff against an untouched reference copy of the model.

### 8.4 Honest results, including the failed one

When the DPO run's numbers don't clearly improve (30 synthetic preference pairs turned out to be
too little signal), the notebook says so directly instead of quietly moving on or cherry-picking a
better seed: it states plainly that with only 30 preference pairs and one pass through them, there
isn't enough signal for DPO to reliably converge, and that this is a real, useful result rather than
a notebook bug. This is deliberate and load-bearing: a notebook that only ever shows clean successes
teaches the reader to expect clean successes, which is not how real fine-tuning runs behave. Branch
the printed interpretation on the actual recorded numbers (an `if margin_improved and loss_improved`
style check) so the "lesson" text is always true of the run that just happened, not aspirational.

### 8.5 Per-technique "Common Pitfalls" + runnable health checks

Every technique section ends with a **Common Pitfalls** Markdown cell (Bad/Good pairs, each with a
one-line "why it matters") followed by a **Quick Health Check** — a short list of prompts/tests a
practitioner would actually run to sanity-check the result — and then a code cell that *runs* those
exact checks rather than leaving them as a hypothetical snippet. This turns "here's what could go
wrong" from an abstract warning into a rehearsed debugging habit.

### 8.6 Ablation studies framed as "what if we cut this corner"

Rather than a neutral "here's what happens if you reorder these steps," the ablation section is
framed as a deadline-pressure scenario ("the launch date got moved up — which corner is safe to
cut?"), and each experiment ends with a **Verdict** that ties the technical result back to what it
would mean for the stated client, not just an abstract observation. Where a hypothetical experiment
is followed by an actual runnable version of it later in the notebook ("Putting Experiment N to the
Test"), make sure the earlier section actually defines that experiment number — a dangling
forward-reference to an experiment that was never introduced is a real, easy-to-miss break in flow
that a slow-reading learner will stumble on trying to find.

### 8.7 Close with a scorecard and a decision, not just a recap

The final section is not a generic "what we learned" list — it's a **decision** framed against the
opening brief: a table scoring every trained checkpoint against the criteria the client actually
cares about (held-out perplexity, instruction-following, preference match, retraining cost), followed
by a concrete "here's what we'd actually ship, and why" recommendation that explicitly revisits every
open question raised earlier (including the one technique that didn't work). A pure summary of
"what this notebook covered" belongs *before* this decision, not as the notebook's last word — ending
on a taxonomy recap after a narrative build-up undercuts the payoff.

### 8.8 Checklist addendum — narrative framing

- [ ] Opens with a named scenario, a concrete ask, and a real constraint that shapes every later
      technical choice (compute budget, privacy, latency, etc.), not just "let's learn fine-tuning."
- [ ] Every major section states, in one sentence, which of the scenario's questions it answers,
      before the theory/code.
- [ ] Charts and numbers come from the actual objects trained earlier in the notebook's own kernel,
      never a fabricated "typical" curve — say so in a comment when this matters.
- [ ] Any claim about *which* parameters/weights changed is verified by diffing real trained weights
      against an untouched reference copy, not asserted from the config alone.
- [ ] When a real run's result is ambiguous, mixed, or a failure, the notebook says so plainly and
      explains why, with the printed interpretation branching on the actual recorded numbers.
- [ ] Every "what could go wrong" pitfalls list is followed by a runnable health-check cell, not a
      hypothetical code snippet.
- [ ] Any experiment referenced by number later in the notebook ("Experiment N") is actually defined
      earlier with that number — no dangling forward references.
- [ ] The notebook ends with a decision/recommendation scored against the opening scenario's stated
      needs, with any pure recap/summary content placed *before* that closing decision, not after it.

---

## 9 · Code Walkthrough Cells (from `04-llm/01-llm-finetuning.ipynb` iteration 2)

Section 8 covers *why* the narrative framing works. This section documents four additional patterns
extracted from the second major iteration of `01-llm-finetuning.ipynb`: code walkthrough markdown
cells, completion-only generation helpers, multi-axis technique comparison grids, and `FuncAnimation`
token-position visualisations.

### 9.1 Code walkthrough markdown cells — after any dense code block

**When to add one:** any code cell longer than ~30 lines that combines multiple library calls, custom
functions, or framework patterns the reader hasn't seen yet. The walkthrough cell goes *after* the
code cell (never before — the reader should run first, then understand).

**Format:**

```markdown
### Code Walkthrough: [Cell Purpose]

**What just ran — [N] [description] combined into one cell:**

---

**[Step A / 1. `function_or_call()`] — [one-line summary]**

[2–4 sentences explaining WHY this call is needed, what problem it solves, and how
it connects to the surrounding concept.]

---

**[Step B / 2. `next_call()`] — [summary]**

[Explanation with a code snippet showing the key line(s) and what they do:]

```python
key_line = does_something_important   # why this matters
```

[Optional: a `| Param | Value | What it controls |` table for any hyperparameter
block the reader is likely to tune.]
```

**Rules:**
- Use `---` separators between each step — they act as visual paragraph breaks for dense content.
- Number steps `Step A / Step B /…` (or `1 / 2 / 3`) — makes it easy to refer back.
- End each step with *why* it matters or what breaks if you skip it, not just what it does.
- Add a `> **PyTorch / HuggingFace shape note:**` blockquote at the bottom of the cell whenever
  the code involves tensor shapes, batch dimensions, or slice indexing — these are the exact places
  where readers who are new to PyTorch get lost.

**Example (from the instruction-tuning walkthrough cell):**

```markdown
### Code Walkthrough: Instruction Tuning Cell

**What just ran — four conceptual steps combined into one cell:**

---

**Step B: `tokenize_instruction()` — the prompt-mask pattern**

This is the core difference from continued pretraining's `tokenize_causal()`:

```
Continued pretraining:  [-100 for padding only,  real labels for everything else]
Instruction tuning:     [-100 for prompt + pad,  real labels for completion only]
```

| Param | Value | What it controls |
|---|---|---|
| `r=8` | Rank | Bottleneck dimension — 8 basis vectors to express the update |
| `lora_alpha=16` | Scaling | Effective LR multiplier = `alpha/r` = 2.0 |
```

### 9.2 Completion-only generation helpers

Any helper function that calls `model.generate()` and is used for demo/test output throughout the
notebook **must return only the newly generated tokens**, not the full prompt + completion sequence.

**Pattern:**
```python
def generate(model, prompt, max_new_tokens=60):
    """Returns only the newly generated tokens (prompt is stripped).

    Parameters
    ----------
    model          : any HuggingFace causal-LM model (base, LoRA adapter, DPO policy…)
    prompt         : input text — the function strips it from the output automatically
    max_new_tokens : hard cap on new tokens; model can stop earlier at EOS
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]          # track where the prompt ends
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,      # stochastic → varied output
            top_p=0.9,           # nucleus sampling: top 90% mass
            temperature=0.8,     # soften distribution slightly
            pad_token_id=tokenizer.pad_token_id,
        )
    # Slice from prompt_len onward — returns ONLY the model's continuation
    completion = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return completion if completion else "[model stopped immediately — sampled EOS as first token]"
```

**Why this matters:** if `generate()` returns the full sequence (prompt + completion), every test
print shows the prompt echoed back, which masks how much (or how little) the model actually
generated. Returning only `out[0][prompt_len:]` makes comparison cells clean and immediately legible.

**Corollary:** any downstream cell that previously stripped the prompt from `generate()` output
(e.g. `output[len(test_prompt):]`) should remove that stripping logic — it becomes a no-op and a
source of confusion.

### 9.3 Multi-axis technique comparison grids

When a notebook covers two orthogonal axes of technique choice (e.g. data objective × parameter
strategy, or architecture × training objective), add a **combination grid** after the held-out
evaluation section.

**Structure:**

1. A **Markdown table** listing all M×N combinations with ✅ (trained in this run) / ❌ (not trained):

```markdown
|                         | **Axis B — Option 1** | **Axis B — Option 2** | **Axis B — Option 3** |
|-------------------------|----------------------|----------------------|----------------------|
| **Axis A — Option 1**   | ✅ `checkpoint_a1b1` | ✅ `checkpoint_a1b2` | ❌ not trained       |
| **Axis A — Option 2**   | ❌ not trained       | ❌ not trained       | ✅ `checkpoint_a2b3` |
```

2. A **code cell** that renders two side-by-side heatmaps — one for a quality metric (lower = better)
   and one for a cost metric (lower = cheaper) — using `np.full((M, N), np.nan)` with a `trained_mask`
   array. Use `cmap.set_bad(color="#d3d3d3")` to grey out untrained cells:

```python
ppl_grid     = np.full((M, N), np.nan)
trained_mask = np.zeros((M, N), dtype=bool)

for (r, c), ckpt_name in checkpoint_map.items():
    if ckpt_name in results:
        ppl_grid[r, c]     = results[ckpt_name]["perplexity"]
        trained_mask[r, c] = True

cmap = plt.get_cmap("YlOrRd_r").copy()
cmap.set_bad(color="#d3d3d3")   # grey = untrained

im = ax.imshow(np.where(trained_mask, ppl_grid, np.nan), cmap=cmap, ...)
```

3. A **textual summary** that prints 2–3 numbered observations from the actual grid values:

```python
print("  Observation 1: Within [axis A row 0], quality degrades as [axis B] shrinks...")
print("  Observation 2: Models trained for [objective] show HIGHER [metric] on [eval]...")
print("  Observation 3: The grey cells represent real engineering options...")
```

### 9.4 `FuncAnimation` for token-position visualisations

Replace any static heatmap whose axes are (token position × dimension) or (step × something) with a
`FuncAnimation` where each frame advances by one position/step. This prevents the "cluttered heatmap"
anti-pattern while conveying the same information in a more legible, interactive form.

**Pattern:**

```python
# ── Static panel: snapshot at last (richest) position ────────────────────────
fig_static, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
# ... ax1: line plot comparing two series at last_pos ...
# ... ax2: bar chart of delta at last_pos ...
plt.tight_layout(); plt.show()

# ── Animated panel: evolve position-by-position ──────────────────────────────
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

n_frames  = data_arr.shape[0]             # one frame per token position / step
delta_max = np.abs(data_arr).max() * 1.15 or 1e-6

fig_anim, ax_anim = plt.subplots(figsize=(12, 4))
bars = ax_anim.bar(np.arange(n_dims), data_arr[0])
ax_anim.set_ylim(-delta_max, delta_max)
title_obj = ax_anim.set_title("")

def _update(frame):
    vals = data_arr[frame]
    for bar, v in zip(bars, vals):
        bar.set_height(v)
        bar.set_color("mediumseagreen" if v >= 0 else "coral")
    tok = tokens_decoded[frame] if frame < len(tokens_decoded) else "?"
    title_obj.set_text(f"Frame {frame}/{n_frames - 1}  token='{tok}'")
    return list(bars) + [title_obj]

anim = FuncAnimation(fig_anim, _update, frames=n_frames, interval=160, blit=False)
plt.close(fig_anim)    # prevent duplicate static frame in Jupyter

# Explain what to look for BEFORE displaying the animation
print("Animation: one frame per position. Green = positive, red = negative.")
print("Early positions show almost no signal; late positions fire harder (richer context).\n")
display(HTML(anim.to_jshtml(fps=6)))
```

**Rules:**
- Always call `plt.close(fig_anim)` before `display(HTML(...))` — otherwise Jupyter renders a
  redundant static frame alongside the widget.
- Print an explanation of what to watch for *before* the animation renders, not after.
- Use `to_jshtml(fps=N)` (not `to_html5_video`) — it works without `ffmpeg` installed.
- Decode the token labels using `tokenizer.convert_ids_to_tokens(input_ids[0].tolist())` and show
  the current token in the frame title — it transforms an abstract "frame 12" into "token 'Voss'."

### 9.5 Checklist addendum — code clarity patterns

- [ ] Any code cell longer than ~30 lines that combines multiple API calls or custom functions has a
      **Code Walkthrough** markdown cell immediately after it.
- [ ] Every `generate()`-style helper returns **only the completion tokens** (`out[0][prompt_len:]`),
      not the full prompt+completion sequence; has a docstring listing parameters and return value.
- [ ] When two orthogonal technique axes are compared, a **combination grid** (markdown table + dual
      heatmap code cell + printed observations) appears after the evaluation section.
- [ ] Any heatmap with a "token position" or "step" axis is replaced with a `FuncAnimation` using
      `to_jshtml(fps=N)`, with `plt.close(fig)` before `display(HTML(...))` and a print explaining
      the animation before it renders.
