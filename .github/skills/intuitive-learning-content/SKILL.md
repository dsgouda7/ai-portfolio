---
name: intuitive-learning-content
description: 'Create or substantially enhance any chapter under learning/ in this AI portfolio. Use for technical notebooks, theory notes, plain-English rewrites, diagrams, animations, complaint-chain explanations, toy-to-real bridges, coverage audits, and cross-chapter continuity. Enforces concrete examples before terminology, intuition before math, short code and result walkthroughs, measured rather than asserted outcomes, honest toy boundaries, abstractly titled concept recaps, notebook safety, and executable validation.'
argument-hint: 'Describe the learning topic, target notebook or chapter, audience, and desired endpoint.'
user-invocable: true
disable-model-invocation: false
---

# Intuitive Learning Content

Create technical learning material that makes each mechanism feel necessary before naming it. Optimize for durable intuition, executable evidence, and a clean path from a small visible example to the real system.
This explanation style applies to **every chapter under `learning/`**, not only language-model
or evaluation chapters. Apply it when creating content and when revising existing content.

## Canonical References

Read only the references needed for the task:

- Repository standard: [AUTHORING_GUIDE.md](../../../AUTHORING_GUIDE.md)
- Plain-English teaching voice: [what a score can really tell us](../../../learning/08-llm-evaluation/01-llm-evaluation-metrics-and-benchmarks.ipynb)
- Historical GenAI notebook patterns: [archived authoring guide](../../../archive/learning-migrations/genai-authoring-guide.md)
- Mechanistic gold standard: [attention, position, and RoPE](../../../learning/04-transformers/02-attention-and-position.ipynb)
- Complete-block continuation: [the complete Transformer block](../../../learning/04-transformers/03-complete-transformer-block.ipynb)
- Modern architecture continuation: [modern decoder-only LLM](../../../learning/04-transformers/06-modern-decoder-only-llm.ipynb)
- Data-lineage continuation: [pretraining data pipeline](../../../learning/05-llm-training/01-pretraining-data-pipeline.ipynb)
- Training-lifecycle continuation: [pretrain a base model](../../../learning/05-llm-training/03-pretrain-a-base-model-lab.ipynb)
- Theory-note style: [Transformer block theory notes](../../../learning/04-transformers/03-complete-transformer-block-theory.md)

The root guide wins when references conflict.

## When to Use

Use this skill for:

- creating a new educational notebook or chapter;
- substantially restructuring or deepening an existing notebook;
- adding intuitive diagrams, animations, exercises, or toy-to-real bridges;
- writing concise `-theory.md` companions and concept recaps;
- auditing topic coverage before authoring;
- fixing pedagogy that states mechanisms without demonstrating them.

Do not use it for a narrow runtime bug fix, a single factual answer, or production application code with no learning-content goal.

## Governing Pedagogy

Use this order for every major concept:

```text
visible problem or falsifiable question
-> static mental model
-> animation when state changes over time or position
-> measured experiment
-> readable implementation
-> optional compact notation only if ambiguity remains
```

Rules:

1. **Intuition and visuals are primary.** Do not open with a formula, definition dump, or API tour.
2. **Math is a last resort.** Avoid derivations and notation inventories. If one expression is essential, explain the idea first and gloss it immediately.
3. **Failure creates demand.** Show what breaks without the mechanism before presenting the fix.
4. **Prove, do not assert.** Follow every non-trivial claim with measured code, a diagnostic, or an independently checkable result.
5. **One running example.** Keep the same domain, entities, and data across sections so only the mechanism changes.
6. **Toy first, real second.** Build an inspectable version, then map it explicitly to production dimensions, configuration, or artifacts.
7. **Honest boundaries.** A toy result demonstrates mechanics, not production quality, reasoning, or readiness.
8. **Plain English must be easy to follow.** Use familiar words, concrete examples, and short explanations grounded in the reader's prerequisites. Keep the technical precision; remove avoidable difficulty from the wording.
9. **Explain before naming.** Describe the problem, what the proposed mechanism does, and why that helps in ordinary language before introducing its technical name. Jargon labels an idea the reader has already met; it does not replace the explanation. Introduce notation only after that intuition, when the math adds clarity.

## Plain-English Chapter Style

Use the evaluation lesson linked above as a voice reference, not a template to copy word for word.

1. **Start with something the reader can judge.** Show a concrete input, two plausible outputs,
   or one attempt that visibly fails. Explain what you wanted and what went wrong before naming
   the technique. Historical background is optional, not the opening requirement.
2. **Make each new noun earn its place.** Explain roles such as reference answer, claim,
   confidence, batch, or checkpoint in ordinary language when they first matter. Do not front-load
   an inventory of jargon or use headings such as "canonical owner" in place of a learner's question.
3. **Keep one familiar example in view.** Reuse it through the small calculation, code, result
   and next limitation. When the data or scale changes, say what changed and why.
4. **Guide the reader through non-obvious code.** Before the block, state the question or prediction.
   After it, explain the key inputs, operation and output in a few sentences. Prefer a short
   prose walkthrough to comments on every obvious line. A block shorter than 30 lines may still
   need a reading guide.
5. **Read an actual result before summarizing.** Point to one real row, answer or failure.
   Explain which quantity moved and why that supports only the stated claim. A passing control,
   unchanged result or worse result is valid evidence; never invent improvement for the narrative.
6. **Compare close variants in small tables.** Say what changes, what that buys and what remains
   missing. Implement a representative when variants share the same mechanism. Do not pretend
   every tool is a historical successor or that newer means universally better.
7. **Separate what a label promises from what the code does.** Word overlap named "factuality"
   is still word overlap. A hand-coded judge is not a live model judge. A declared fixture is
   not a collected human study. Put the boundary beside the result it qualifies.
8. **Make failures explicit.** No feasible policy, zero observations, missing evidence and
   undefined metrics must remain visible. Do not turn them into zero-error success or quietly
   relax a requirement. Check that the suggested exercise works with the actual code and guards.
9. **Keep the main reading path conversational.** Use optional details for lengthy setup choices,
   notation or reference material. Retain scope and provenance without opening with bureaucracy.
10. **Use abstract recap titles.** Prefer "Concept recap" or "Summary", not a handwriting-specific
    title. Pair concise explanations with short code sketches when useful, marking sketches
    versus executable recipes. Let the learner choose how to record the notes.

Before finishing, answer these without relying on terminology:
**What failed? What did we change? Why might that help? What happened? What is still missing?**
Then verify that the prose, figures, numbers and exercises tell the same story.

## The Golden-Version Essence

The historical Transformer gold standard worked because the learner experienced discovery in one continuous room. Preserve that feeling even when content is split across files.

### 1. Build a complaint chain, not a feature tour

Every major mechanism must be forced by the visible limitation immediately before it:

```text
crude attempt
-> measured or visible failure
-> plain-language complaint
-> smallest refinement
-> measured payoff
-> new complaint that opens the next mechanism
```

Name the complaint in the prose. Good headings include:

- `#### That worked, but it still cannot see order`
- `#### A nagging question before we move on`
- `#### So the heads differ - could one table carry both?`

Do not replace this chain with a roadmap that merely lists finished mechanisms.

### 2. Publish useful failed attempts

When an intuition visual or implementation becomes clear only through refinement, keep the intermediate attempts:

```text
Attempt 1: one dial
complaint: one vector contains several pairs
Attempt 2: stacked dials
complaint: one position is not a sequence
Attempt 3: one tower per position
payoff: relative motion becomes visible
```

Each attempt must be runnable or visible, short, and followed by the exact complaint it exposes. Do not show failed attempts that add no new understanding.

### 3. Preserve one-room continuity across files

Splitting a long notebook must not reset the learner's mental state.

- Keep the same running example, token names, shapes, colors, and variable roles until a real task boundary requires a change.
- End each chapter with the unresolved question that opens the next chapter.
- Open the next chapter by resuming that question and reusing one familiar artifact or tensor.
- Keep fresh-kernel setup compact. A recap re-establishes state; it must not replay the lesson or introduce a new narrative.
- When the example must change, add an explicit bridge explaining why the old example is no longer sufficient.

### 4. Preserve the consequence, not only the rule

Do not teach only the final prescription:

- not just `divide by sqrt(d)` - show false softmax confidence and the gradient consequence;
- not just `add residuals` - compare the gradient route with and without the skip path;
- not just `use RoPE` - expose the limitation of the previous position placement, then verify norm and equal-gap invariants;
- not just `use multiple heads` - construct two incompatible routing jobs and measure the one-table compromise.

If space is limited, keep the consequence experiment and shorten the derivation.

### 5. Use an evidence ladder

For a durable concept, prefer this sequence:

1. physical or everyday mental model;
2. static visual that survives GitHub and export;
3. animation when motion or accumulation matters;
4. one-variable measured comparison;
5. readable implementation;
6. real model, artifact, or configuration bridge.

Static evidence is mandatory when an animation carries essential intuition. The notebook must still teach when JavaScript output is unavailable.

### 6. Keep the instructional voice alive

Use precise, conversational second-person prose. Let the reader hear the objection a careful engineer would raise.

- Prefer `You fixed token identity. You still have no order.`
- Prefer `That plot looks convincing. It also hides a problem.`
- Avoid repeated institutional phrases such as `This section introduces`, `We now explore`, or `The mechanism is as follows`.
- Use brief dry humor only when it sharpens the complaint.
- Let reflections carry momentum; do not end a mechanism with a detached summary paragraph.

## Workflow

### 1. Inspect Before Editing

- Read the target notebook or chapter, neighboring content, README, and local setup files.
- Identify the controlling learning gap, not merely the next topic name.
- Check current Git status and preserve unrelated user changes.
- For notebooks, record the source-level baseline before edits.

### 2. Enumerate the Topic Space

Before writing, list what a complete treatment would include. Assign every item to one tier:

| Tier | Meaning |
|---|---|
| Built and measured | Runnable implementation with verified evidence |
| Explained or illustrated | Real intuition and boundaries, but not fully implemented |
| Named and out of scope | Acknowledged with a reason and forward link |

Use the enumeration to prevent accidental omissions. Put a compact scope map near the opening and the full ledger near the close.

### 3. Define the Learning Arc

For each part, write the dependency chain:

```text
current capability
-> concrete failure
-> minimal new mechanism
-> measured unlock
-> residual gap that motivates the next part
```

Prefer 5-10 coherent parts over many tiny sections. The notebook endpoint must be explicit.

For a multi-notebook series, also write the cross-file continuity chain:

```text
chapter N measured unlock
-> unresolved question at chapter N close
-> same question at chapter N+1 open
-> familiar example/tensor reused
```

### 4. Build the Opening

Open with:

- a title and one guiding question;
- where the learner is in the arc;
- the specific blocker;
- a roadmap table;
- prerequisites and missing-artifact behavior when applicable;
- scope boundaries.

For foundation content, use a physical or mechanistic problem. For applied content, use one named system with real constraints.

### 5. Author Each Mechanism

Use this repeated pattern:

1. `**Predict:**` Ask a concrete question with plausible alternatives.
2. Show a diagram or side-by-side mental model.
3. Build the smallest readable implementation.
4. Measure the exact claim under test while changing one variable.
5. Print the takeaway in words.
6. Add `**Your turn:**` to change one known variable, never to introduce a new concept.
7. Close with a reflection naming what was fixed and what remains missing.

When a polished visual hides the discovery process, add one or two explicit `Attempt` cells before it. Each attempt ends with a printed or written complaint that the next attempt answers.

Code that teaches a mechanism should be readable top-to-bottom in about a minute. Add a walkthrough immediately after code cells longer than roughly 30 lines.

### 6. Design Visual Evidence

Choose the medium by purpose:

- **Copilot/generated static image:** stable intuition before code; no fabricated numbers.
- **Mermaid:** short orientation or lifecycle with at most about eight nodes.
- **Matplotlib/Seaborn:** measured comparisons, matrices, distributions, and before/after evidence.
- **FuncAnimation:** token position, training step, packing, routing, gradient, or checkpoint evolution.
- **ASCII shape trace:** tensor dimensions or small matrix flow.

Visuals are evidence, not decoration:

- show labels from the running example, not anonymous row numbers;
- isolate one changed variable across side-by-side panels;
- put before/after states in one figure so the reader does not mentally diff separate outputs;
- annotate the exact failure or measured quantity;
- pair an animation with a static storyboard when the motion is conceptually load-bearing.

For animations:

- derive every frame from real recorded state;
- keep frame counts bounded;
- print what to watch before display;
- call `plt.close(fig)`;
- render with `to_jshtml` so FFmpeg is unnecessary.

Generated-image prompts must include target filename, semantic anchor, aspect ratio, prompt, acceptance criteria, prohibited content, and alt text. Never reference an image file until it exists.

### 7. Bridge Toy to Real

Include an explicit mapping table:

| Teaching mechanism | Production counterpart | What changes | What does not change |
|---|---|---|---|

Use real configuration fields or inspected artifacts where practical. Avoid network downloads when a representative local config teaches the same contract.

### 8. Close With Evidence

End with:

- a scorecard answering the opening questions with measured results;
- concise, quotable takeaways;
- the three-tier coverage ledger;
- practical failure modes;
- the exact next chapter or boundary.

## Theory Companions and Concept Recaps

Create `[notebook-name]-theory.md` when the chapter introduces durable concepts.

Theory notes are concise discovery narratives that a learner can copy into their own notes.
Use a topic-based title rather than labeling the learner's note-taking medium.
They are not compressed API documentation or a section-by-section notebook recap.

Use the same complaint chain as the notebook:

```text
what the learner can do
-> what still fails
-> memorable mental model
-> mechanism in plain language
-> measured result worth remembering
-> boundary or next question
```

Theory notes should:

- target roughly 850-1,150 words unless the chapter spans several distinct lifecycles;
- use numbered sections and one memorable mental model per mechanism;
- use short process traces, comparison tables, and concrete distinctions;
- avoid code, derivations, and unnecessary formulas;
- explain what each mechanism does, what it does not do, and why it exists;
- preserve the notebook's running example and instructional voice;
- include only numbers the notebook actually measured or stable shape facts;
- retain one plain-language complaint between consecutive mechanisms;
- finish with practical failure modes and one durable summary sentence.

Link the theory companion beside the notebook in local and track-level READMEs.

A notebook's closing **Concept recap** can include short, clearly labeled code sketches with
plain-English explanations. Keep setup and download boilerplate out of that recap; preserve
the distinction between the mechanism and a complete runnable program.

## Notebook Safety

Notebook files require source-aware handling:

1. Prefer notebook-aware editing tools for cell content.
2. Do not use broad text replacement against raw `.ipynb` JSON.
3. Immediately verify every notebook edit from disk.
4. Treat display cell identifiers as different from raw notebook IDs.
5. Compare cell sources before and after; cell-count equality cannot detect replacement corruption.
6. Inspect every source-level `replace` operation manually.
7. Keep existing outputs and metadata unless the task explicitly changes them.
8. Validate in a temporary executed copy, never by overwriting the source notebook.
9. Confirm the clean notebook and executed copy have identical cell sources.
10. Clear outputs and execution counts in committed learning notebooks.
11. Run whole-worktree status after automated or parallel notebook work to catch collateral changes.
12. Never mention internal cell IDs to the user; use visible cell numbers or headings.

When notebook-aware tools cannot reliably edit a large notebook, use a deterministic JSON script only after verifying the original serialization round-trips byte-for-byte. Back up first, target cells by unique source text or verified raw ID, and delete the temporary script and backup after validation.

## Validation

Before completion:

- parse notebook JSON;
- parse all Python cells after stripping notebook magics;
- check unique cell IDs and required metadata;
- execute from a fresh kernel into a temporary notebook;
- confirm every code cell ran and no error output exists;
- compare clean and executed cell sources for equality;
- verify output/manifest/checkpoint contracts from disk;
- check local links and image paths;
- run relevant repository tests or validators;
- run `git diff --check` and inspect full Git status;
- confirm unrelated files and gold-standard references were not changed.

## Anti-Patterns

Do not:

- lead with formalism before the learner feels the problem;
- add math for completeness;
- change datasets or narratives every section;
- use opaque framework calls to hide the mechanism being taught;
- fabricate smooth training curves, metrics, or image annotations;
- treat attention maps, embedding neighbors, or parameter deltas as complete explanations;
- claim a tiny teaching run acquired broad knowledge or reasoning;
- leave techniques mentioned without assigning a coverage tier;
- create decorative diagrams that teach nothing;
- replace a sequence of useful discovery attempts with one polished final diagram;
- split a notebook without carrying its example, unresolved question, and visual language forward;
- keep the formula but remove the experiment that made the formula necessary;
- turn `Your turn` into a prose suggestion when a nearby runnable one-variable drill is practical;
- commit temporary notebooks, runtime artifacts, generated checkpoints, or repair backups.
