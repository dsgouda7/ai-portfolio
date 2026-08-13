---
name: intuitive-learning-content
description: 'Create or substantially enhance intuitive technical learning notebooks and handwritten theory notes in this AI portfolio. Use when asked to author GenAI, Transformer, ML, agentic-AI, infrastructure, or other educational chapters; add diagrams or animations; build failure-first explanations; create toy-to-real bridges; audit topic coverage; or bring content to the Transformer gold-standard quality. Enforces visual-first pedagogy, minimal necessary math, prove-dont-assert experiments, theory companions, notebook safety, and executable validation.'
argument-hint: 'Describe the learning topic, target notebook or chapter, audience, and desired endpoint.'
user-invocable: true
disable-model-invocation: false
---

# Intuitive Learning Content

Create technical learning material that makes each mechanism feel necessary before naming it. Optimize for durable intuition, executable evidence, and a clean path from a small visible example to the real system.

## Canonical References

Read only the references needed for the task:

- Repository standard: [AUTHORING_GUIDE.md](../../../AUTHORING_GUIDE.md)
- GenAI notebook patterns: [learning/genai/authoring-guide.md](../../../learning/genai/authoring-guide.md)
- Mechanistic gold standard: [attention and Transformer blocks](../../../learning/genai/01-transformers/01-attention-and-transformer-blocks.ipynb)
- Modern architecture continuation: [modern decoder-only LLM](../../../learning/genai/01-transformers/04-modern-decoder-only-llm.ipynb)
- Data-lineage continuation: [pretraining data pipeline](../../../learning/genai/01-transformers/05-pretraining-data-pipeline.ipynb)
- Training-lifecycle continuation: [pretrain a base model](../../../learning/genai/01-transformers/06-pretrain-a-base-model.ipynb)
- Theory-note style: [Transformer theory notes](../../../learning/genai/01-transformers/01-attention-and-transformer-blocks-theory.md)

The root guide wins when references conflict.

## When to Use

Use this skill for:

- creating a new educational notebook or chapter;
- substantially restructuring or deepening an existing notebook;
- adding intuitive diagrams, animations, exercises, or toy-to-real bridges;
- writing concise `-theory.md` handwritten-note companions;
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

Code that teaches a mechanism should be readable top-to-bottom in about a minute. Add a walkthrough immediately after code cells longer than roughly 30 lines.

### 6. Design Visual Evidence

Choose the medium by purpose:

- **Copilot/generated static image:** stable intuition before code; no fabricated numbers.
- **Mermaid:** short orientation or lifecycle with at most about eight nodes.
- **Matplotlib/Seaborn:** measured comparisons, matrices, distributions, and before/after evidence.
- **FuncAnimation:** token position, training step, packing, routing, gradient, or checkpoint evolution.
- **ASCII shape trace:** tensor dimensions or small matrix flow.

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

## Handwritten Theory Companions

Create `[notebook-name]-theory.md` when the chapter introduces durable concepts.

Theory notes are concise intuition distillations that can be copied by hand:

- target roughly 850-1,150 words unless the chapter spans several distinct lifecycles;
- use numbered sections and one memorable mental model per mechanism;
- use short process traces, comparison tables, and concrete distinctions;
- avoid code, derivations, and unnecessary formulas;
- explain what each mechanism does, what it does not do, and why it exists;
- finish with practical failure modes and one durable summary sentence.

Link the theory companion beside the notebook in local and track-level READMEs.

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
- commit temporary notebooks, runtime artifacts, generated checkpoints, or repair backups.
