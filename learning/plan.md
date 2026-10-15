# Additive Learning Flow and Ownership Plan

## 1. Scope, preservation, and delivery

This is an implementation plan, not a notebook rewrite. Pedagogy drives the route;
historical evolution explains useful design pressures without dictating teaching order.

- Preserve every existing learning chapter, explanation, experiment, and optional branch.
- Preserve the canonical [Transformer track](04-transformers/README.md) as a
  self-contained, one-stop-shop route, including its redundant prerequisite buildup.
  No culling, abbreviation, replacement, or relocation out of that route.
- Prefer new notebooks and short additive bridges. Existing deep dives remain available
  even when a returning-reader route points past material already understood.
- Introduce concepts when the current model/task needs them, not in a prerequisite dump.
- Keep plain-English intuition ahead of terminology, minimal code, and optional depth.
  This is a hard requirement; see Section 10.
- Back every picture-dependent intuition with a graphical aid. You generate the images
  manually from Copilot prompts left at the point of use (Section 10.2).
- Give every learning directory a one-line-per-chapter `CONTENTS.md` (Section 10.3).
- Apply the same standard across all twelve stages, without claiming that a sampled
  planning review is a completed all-chapter audit.
- Current delivery: this plan only. Notebook creation and curriculum edits are future
  implementation work; proposed filenames below do not exist yet.

Before this file was created, all existing changes were committed and pushed to
`origin/main` as `e94a33f31f593e2329c0811226bb7cbbfcf0008c`. That commit is the
preservation baseline.

## 2. What the inspection actually found

Parallel read-only investigations covered Stage 02-03 foundation ownership, Stage 04
preservation, and README/headings plus selected material in the remaining stages.

An important correction to the earlier discussion: encoder-decoder foundations are
**already present**, not absent. The existing
[02C notebook](03-language-model-foundations/02-tokenization-and-embeddings/02C-context-windows-to-static-embeddings.ipynb)
and its [theory companion](03-language-model-foundations/02-tokenization-and-embeddings/02C-context-windows-to-static-embeddings-theory.md)
cover CBOW, skip-gram, recurrent encoder-decoder, teacher forcing, and the fixed-summary
limitation. The problem is concept convergence and visibility of ownership, not a need
to replace that material.

| Observed surface | Flow issue or strength | Additive response |
|---|---|---|
| [Sequence memory](03-language-model-foundations/01-sequence-memory/README.md) | Numeric recurrence precedes text, appropriately; the language lab uses token IDs, embeddings, shifts, and padding before their formal chapter | Add tiny first-use explanations at those transitions; retain the formal chapter and existing preview contracts |
| [Tokenization and embeddings](03-language-model-foundations/02-tokenization-and-embeddings/README.md) | 02C changes from representation learning to sequence generation within one notebook | Add an objective/shape map and an explicit task-boundary bridge; keep every existing section |
| [Stage 03 route](03-language-model-foundations/README.md) | Count-based next-token modeling, neural fixed-window LM, and recurrent attention lack a clearly named runnable progression | Add bounded comparison notebooks and a reader route, not a new compulsory historical survey |
| [Attention and position](04-transformers/02-attention-and-position.ipynb) | RoPE's visual construction precedes the dedicated learned Q/K/V lesson | Explain the preview versus actual computation order; retain all attempts, visuals, and experiments |
| [Transformer source-to-target theory](04-transformers/05a-encoder-decoder-and-cross-attention-theory.ipynb) | A pooling failure can be mistaken for the historical recurrent seq2seq bottleneck | Add a distinction between this controlled pooling experiment and learned recurrent compression |
| [Later learning stages](README.md) | Many already have good artifact gates and complaint chains; chapter-level consistency still needs exhaustive inspection | Preserve good examples and audit missing handoffs rather than rewrite every chapter |

## 3. Separate architecture evolution from Transformer teaching

Encoder-decoder describes **roles**, not a particular internal layer:
"read a source; use it to write a separate target." Recurrence, convolution, or
attention can implement those roles. Seq2seq is the task family, not a synonym for
Transformer or attention.

Use this pedagogical complaint chain:

```text
Fixed-summary recurrent encoder-decoder:
the reader compresses the source into a fixed-size state
-> the writer has no direct access to each source position

Attention-augmented recurrent encoder-decoder:
keep the source's per-position states available for each writing step
-> source retrieval improves, but recurrent state updates remain sequential

Transformer encoder-decoder:
use self-attention to build representations and cross-attention to retrieve source memory
-> sequence-position computation can be parallelized during teacher-forced training
-> autoregressive generation still proceeds token by token
```

LSTM addresses a different complaint: carrying information and teaching signals through
recurrent steps. It does not eliminate the fixed-summary interface. A fixed-size
summary is a capacity/access constraint, not proof that every short example must fail.

Attention predates Transformers. The original Transformer was encoder-decoder;
teaching decoder-only first in Stage 04 is a dependency choice, not a historical claim.
Neither recurrent attention nor the original Transformer makes recurrence's alternatives
or encoder-decoder obsolete.

## 4. Model map: core route, adjacent branch, and context only

This enumerates the relevant model families and roles, not every historical variant.
The rows are a teaching map, not publication chronology or an assertion that each model
descended directly from the preceding row.

| Model/family | Complaint it makes visible | Treatment and owner |
|---|---|---|
| Unigram, bigram, bounded n-gram LM | Counts predict continuations, but longer contexts become sparse and fixed | Compact runnable baseline in the new Stage 03 model-comparison notebook; smoothing only if the example requires it |
| Neural fixed-window LM | Learn reusable features instead of separate counts, but context still has a fixed boundary | Same new comparison notebook; distinguish ordered context from a bag |
| Vanilla RNN | Reuse a state over variable sequence lengths, but information/gradients can fade | Existing sequence-memory lesson remains owner; new text comparison connects that lesson to LM |
| LSTM | A controlled memory route helps with long dependencies; a single exported summary still limits access | Existing theory/lab remain owner; reuse rather than reteach gates |
| GRU | Another gated recurrent design, not a necessary step toward attention | Optional comparison or named-with-reason; never a main-route prerequisite |
| Fixed-summary recurrent encoder-decoder | Separate source reading from target writing; discover the compressed handoff | Existing 02C owns the introduction; new seq2seq notebook gives it room and a controlled comparison |
| Attention-augmented recurrent encoder-decoder | Let each target step retrieve from all valid source states | New additive seq2seq notebook; the immediate bridge to Transformer attention |
| Transformer encoder-decoder | Replace recurrent representation updates with self-attention while retaining separate source retrieval | Existing Stage 04 Parts 2-3 and 5 own mechanisms and full architecture |
| Encoder-only and decoder-only Transformers | Different visibility/objective choices, not successive historical upgrades | Preserve the Stage 04 architecture comparison and decoder-only route |
| Modern decoder-only Transformer | Refine block internals and inference trade-offs without changing the core objective | Preserve Stage 04 Part 6 |
| CBOW / skip-gram | Learn static word representations from context/center prediction | Existing 02C; adjacent objective branch, not a required evolutionary predecessor of RNN |
| Count/co-occurrence vectors and GloVe | Alternative routes to static representation geometry | Existing optional embedding-history branch; not a detour required to understand attention |
| Bidirectional recurrent encoders | An available source may use left and right context | Brief illustration when source memory first needs it; target generation remains causal |
| Convolutional sequence models | Another route to sequence representations and parallel computation | Name with a reason in the evolution map; no new required CNN-seq2seq deep dive |

Do not turn the table into one notebook per model. Necessary understanding, not family
coverage, determines how much runnable treatment each receives.

## 5. First-use map: introduce only what the task now needs

Distinguish four moments: **first encounter**, **minimal explanation**, **formal owner**,
and **later reuse**. Formal ownership does not forbid a useful local explanation.

| Concept | First actual need | Minimal introduction there | Formal ownership / later reuse |
|---|---|---|---|
| Token boundaries and vocabulary | First text/count example, not numeric forecasting | "Choose the pieces we count or predict"; start with characters or whitespace pieces | Stage 03 tokenization chapter; preserve Stage 04 recap |
| Integer token IDs | First tensor implementation of discrete text | "An ID is an address, not a numerical meaning" | Stage 03 tokenization; count models can operate on strings |
| Learned embeddings | First lookup-based neural text model | "Each ID retrieves a trainable row"; contrast with one-hot/features if useful | Stage 03 embeddings; existing sequence lab gets an immediate local bridge |
| BPE/subword strategy | When the toy vocabulary cannot represent useful new text | Show the word-level failure, then reusable pieces | Stage 03 tokenization; not a prerequisite for every RNN or seq2seq task |
| Context/target pairs | First next-token baseline | "These observed tokens are the clue; this next token is the answer" | New comparison notebook and existing Stage 03 objective lessons |
| Sequence target shifting | First many-position next-token lab or target decoder | Show input and expected-output rows aligned by one position | Existing Stage 03 labs/02C; Stage 04 retains its complete treatment |
| Padding and lengths | First variable-length batch | "Make rectangles without treating empty slots as content" | Stage 02 batching mechanics; Stage 03 text semantics and packing |
| Loss versus attention masks | When padding is graded or positions retrieve context | "Do not grade blanks" differs from "do not read blanks" | Stage 03 mask distinctions; Stage 04 causal/attention contracts |
| Encoder and decoder roles | First task with separate source and target | "One component reads; another writes" | Existing 02C, expanded by the new seq2seq comparison |
| Teacher forcing | First decoder trained with known earlier target tokens | "During practice, supply the correct previous token; in generation, use your own" | Existing 02C plus minimal new demonstration; Stage 04 repeats intentionally |
| Attention / source padding mask | The fixed-summary writer needs a source detail again | "Keep the reader's notes and look back at the useful position" | New recurrent-attention bridge; Stage 04 owns detailed attention mechanics |
| Positional information | Attention must distinguish order or repeated source tokens | Show the ambiguity before adding position | Stage 04; recurrent processing already introduces order through its state updates |
| KV caches and decoding policies | Correct generation already works and repeated computation matters | Explain the cost complaint, with a forward link if not needed yet | Preserve all existing Stage 04 cache and decoding material |

An n-gram baseline needs a next-token relationship, not necessarily a shifted tensor.
Numeric RNN/LSTM tasks need neither tokenizers nor embedding tables. Text implementations
may use them, but they are design choices rather than definitions of recurrence.

For current notebooks that encounter a concept before its formal owner, add a short
explanation immediately before first use plus a link to the owner. Do not force readers
to complete a large representation chapter just to understand one tensor operation.

## 6. New notebooks and the recommended additive route

Keep all current names and content. Add three self-contained notebooks in a proposed
unnumbered `03-language-model-foundations/model-evolution/` directory, with concise theory
companions and local navigation. The explicit route, not directory numbering, defines
their reading order.

| Proposed notebook | Bounded job | Stop before |
|---|---|---|
| `01-counts-and-fixed-window-language-models.ipynb` | One tiny next-token task: count baseline, ordered neural fixed window, and a dependency outside that window | Smoothing catalogues, tokenizer algorithm surveys, broad benchmark claims |
| `02-fixed-windows-to-recurrent-language-models.ipynb` | Reuse the same tokens/objective; show a vanilla-RNN text path and connect it to the existing LSTM evidence | Rebuilding the existing gates/backprop lesson or claiming recurrence has unlimited reliable memory |
| `03-recurrent-encoder-decoder-to-attention.ipynb` | Fixed-summary source-to-target model versus access to per-position source states; teacher-forced versus free-running behavior | Beam-search surveys, scheduled-sampling methods, cache taxonomy, Transformer implementation |

The main reader route becomes:

1. Existing Stage 02 implementation contracts.
2. Existing numeric sequence-memory route: recurrence without unnecessary text machinery.
3. First-use text primer at the language transition; existing tokenization/embedding
   material remains the formal reference.
4. New counts/fixed-window comparison, then recurrent text comparison, explicitly
   reconnecting to the already understood RNN/LSTM memory lesson.
5. Existing 02C representation-to-seq2seq material, with an added signpost distinguishing
   its objectives. Returning readers can follow its seq2seq anchor directly.
6. New recurrent seq2seq-to-attention notebook.
7. Unchanged canonical Stage 04 route, with additive bridges.

Thus the route need not teach RNNs twice: the numeric lesson owns recurrence; the new
text notebook compares model behavior using that established mechanism.
CBOW/skip-gram depth remains available, but understanding their full training details
is not a gate for attention.

At the source-to-target boundary, explicitly explain why next-token continuation is no
longer the right microscope. Reuse Stage 04's integer reversal task where practical:
newly introduced source/target roles are the only essential new moving parts.
Integer sequences avoid introducing a second tokenizer at the same time.

## 7. Plain-English seq2seq treatment and bounded evidence

Open each next model with its predecessor's unresolved complaint, rather than adding
a detached history section:

> You can remember an ordered stream. Now the whole input is available, but the answer
> is a different sequence. Let one component read it and another write the answer.

> The writer received one summary note. To recover a specific early detail, why not
> let it look back at the reader's notes for every source position?

Explain teacher forcing in one short diagram and a minimal input/target alignment
example before the training loop. It is a training method, not an encoder-decoder
architecture requirement, and it also applies to decoder-only language modeling.

Use a tiny, visibly traced example showing a wrong previous prediction changing the
next input. Name exposure bias only after that distinction is understood. Do not imply
that teacher forcing leaks the current answer or that an error must always cascade.

Required evidence in the new work:

- Identical fixed windows cannot distinguish examples that differ only outside them;
  distinguish this construction from a learned model's empirical accuracy.
- Compare vanilla-RNN/LSTM behavior with controlled data, objectives, and documented
  model capacity; do not attribute all differences to a single architectural factor.
- A fixed-summary model must really pass a learned recurrent summary, not substitute
  mean pooling and call that the historical architecture.
- Compare fixed-summary and attentive models on the same split and bounded task.
  Report sequence accuracy by source length and both evaluation conditions.
- If attention does not improve the measured task, say so; demonstrate direct source
  access separately rather than printing a predetermined success claim.
- Treat routing maps as diagnostics, not proof of complete causal explanation.

## 8. Additive Transformer flow improvements and focused copy

The canonical [Stage 04 route](04-transformers/README.md) remains authoritative and
self-contained. Add:

- A reader map distinguishing prerequisite recap, new mechanism, and deliberate reuse.
- Before RoPE's visual construction, a short query/key intuition and a preview of the
  actual path: states -> Q/K/V projections -> head split -> rotate Q/K -> scaled
  scores -> masks -> softmax -> weighted values. Values are not rotated.
  Preserve the current RoPE discovery sequence and later full Q/K/V lesson.
- At Part 5's opening, a link to the earlier encoder-decoder explanation:
  "You already know the reader/writer roles; here their internals become Transformer
  blocks." The canonical deep dive still follows for standalone readers.
- Near the pooling attempt, explain that it isolates lost source addresses, whereas
  recurrent fixed-summary models learn their compression and may encode order.
- A block comparison: encoder = self-attention then FFN; decoder-only = causal
  self-attention then FFN; encoder-decoder writer = causal self-attention,
  cross-attention, then FFN. Existing normalization/residual details remain intact.
- At repeated teacher-forcing/cache sections, identify what is familiar and what
  separate source memory changes. Do not merge or remove explanations.
- Keep Part 6's production-oriented RoPE revisit and all existing exercises.

Prefer explanatory additions over reordering. If a later dependency audit justifies
moving a section, preserve its entire content, publish an old-to-new heading map,
and verify fresh-kernel execution. No rearrangement is required by this plan.

Add a separate proposed `04-transformers/focused/` route, built in parallel from the
pushed baseline plus the agreed bridge text above. It is a derived, Transformer-focused copy of the relevant
notebook lessons, not merely a promise to skip cells and not a replacement:

- State assumed mastery of tokenization, embeddings, recurrence, basic seq2seq, and
  teacher forcing; include a readiness check with links back to canonical explanations.
- Keep Transformer-specific failure experiments, implementations, visuals, and
  prediction/generation contracts. Replace only inherited prerequisite buildup
  in this new copy with concise references.
- Copy only coherent lessons; do not duplicate the entire track without need.
  Record source notebook, source headings, and canonical revision for each copy.
- Make each notebook fresh-kernel runnable with explicit setup. Do not import or
  execute cells from the originals or assume another notebook's in-memory state.
- Update affected copies in the same future change as their canonical mechanisms,
  or explicitly mark them stale. Avoid a new synchronization framework.

## 9. Apply the pattern across every learning chapter

Keep a complete audit ledger in session/review artifacts, not a proliferation
of new planning files in the curriculum. Each parallel agent fills the rows for its own
scope; the integration pass merges them. Include every chapter, standalone notebook,
theory-only lesson, optional branch, refresher, and nested role chapter. Distinguish
asset-level inspection from chapter-level acceptance.

Record: path, required/optional role, concept owner, first use, prerequisites,
opening complaint, intuitive explanation, plain-English status, image needs,
contents-map line, runnable evidence where appropriate, closing question, next
consumer/artifact, runtime dependency, proposed addition, preservation check, and
validation status. Each item receives a disposition:
already sufficient, needs additive bridge, needs bounded new example, or blocked.

| Stage | Ownership to preserve | Specific additive audit focus |
|---|---|---|
| 01 ML foundations | Optimization, generalization, neural networks, backpropagation | Expand stage-level orientation only where missing; preserve existing required/optional routes |
| 02 PyTorch | Tensors, modules, autograd, data/training/checkpoint mechanics | Separate implementation previews from later domain explanations |
| 03 Language-model foundations | Recurrence, text representations, early seq2seq, recurrent attention | Apply Sections 3-7; expose task/objective boundaries |
| 04 Transformers | Attention, position, blocks, architecture families | Preserve canonical content and add Section 8 bridges |
| 05 LLM training | Data lineage, pretraining lifecycle, resumable artifacts | Make the Stage 04 architecture handoff explicit before the existing data/training route |
| 06 Fine-tuning | Adaptation objective, update location, comparison/provenance | Connect its existing four decisions to concrete complaints; preserve GPU prerequisite order |
| 07 RAG | Authorized retrieval and retrieval-versus-answer diagnosis | Preserve oracle-context diagnosis and distinguish retrieval embeddings from earlier token embeddings |
| 08 Evaluation | Evaluators, hallucination localization, safety/calibration | Preserve evidence distinctions and its evaluation-to-gateway ownership boundary |
| 09 Gateway | Routing, eligibility, retry/fallback, budgets, telemetry | Link policy decisions to Stage 08 evidence; preserve retry/fallback distinctions |
| 10 Agentic AI | Tool authority, state, approvals, side effects, recovery | Check every nested chapter against existing prerequisites and ownership maps |
| 11 Infrastructure | Hardware, inference, serving composition, operations | Add evidence-based prerequisite checks to shortcuts without removing entry options |
| 12 Role tracks | Synthesis, transfer, capstones, graduation evidence | Ensure targeted routes do not replace Stages 01-11; retain challenge-out evidence gates |

Add only what a chapter lacks: a short opening bridge, a first-use intuition,
a bounded evidence example, or an explicit closing handoff. Do not add identical
ceremonial sections to chapters that already teach this well.

Across chapter boundaries, carry the familiar example and unresolved question when
the task stays the same. When it changes, explain why and map the old artifact or
concept to its new role. Token embeddings, sentence/retrieval embeddings, and hidden
contextual states are related representations with different jobs, not interchangeable
uses of one word.

## 10. Hard authoring requirements

These apply to every new notebook, bridge, theory note, and contents map. They are
requirements, not style preferences.

### 10.1 Plain English beats formal language

A novice to the concept must be able to follow the text without a glossary or prior
exposure. Concise is good; compressed jargon is not.

- **Explain before naming.** Describe what happens in everyday words first, then give
  the technical term: "give the writer the correct previous word during practice —
  this is called *teacher forcing*."
- **One new idea per sentence.** Never stack two unexplained terms in one sentence.
- **Gloss every term at first use** in the same paragraph, even if another chapter
  formally owns it. A link to the owner supplements the gloss; it does not replace it.
- **Short, concrete sentences.** Prefer "the writer can't look back" to
  "the decoder lacks direct access to source-position representations."
- **Use the running example's actual tokens and numbers**, not abstract symbols, when
  explaining what a mechanism does.
- **One everyday mental model per mechanism** (notecard, reader and writer, looking
  back at notes) used consistently, with its limits stated when it stops being accurate.
- **Math is optional depth.** If an expression is genuinely needed, the plain-English
  explanation comes first and a one-line gloss follows immediately.
- **Banned register:** "it can be shown," "we now introduce," "the mechanism is as
  follows," undefined acronyms, and passive definition dumps.

Existing formal passages are not rewritten. Where an existing section opens in formal
language, insert a short plain-English lead-in cell or paragraph directly before it.

**Novice test** for every new or bridged section: a reader who has never heard the
term can state, in one sentence of their own words, what it does and why it was needed.
A reviewer checks this before the section is accepted.

### 10.2 Rich graphical aids with Copilot image prompts

Every mechanism whose intuition depends on a picture gets one. Choose the medium by job:

| Job | Medium |
|---|---|
| Stable intuition before code (mental models, roles, flows) | Generated image from a Copilot prompt (below) |
| Short orientation of at most about eight nodes | Mermaid |
| Measured comparison, matrix, distribution, before/after | Matplotlib from real computed values |
| State changing over steps or positions | `FuncAnimation` rendered with `to_jshtml`, plus a static storyboard |
| Tensor shapes | ASCII shape trace |

Generated images never carry fabricated measurements; numbers come from code.

**Image workflow (user generates the files manually):**

1. At the exact point of use, insert a markdown placeholder cell:

   ```markdown
   **Image pending:** `images/<stage>-<chapter>-<slug>.png`
   *What this will show:* one plain-English sentence, so the lesson still reads without it.

   > **Copilot image prompt**
   > - **Target file:** `images/<stage>-<chapter>-<slug>.png`
   > - **Teaches:** the single idea the reader must take away
   > - **Aspect ratio:** e.g. 16:9
   > - **Prompt:** full description — layout, labelled elements using the running
   >   example's real tokens, arrows/flow, style (clean hand-drawn whiteboard,
   >   consistent palette with the chapter)
   > - **Must show:** acceptance criteria a reviewer can check
   > - **Must not show:** invented numbers, unreadable text, decorative clutter,
   >   logos or copyrighted characters
   > - **Alt text:** one or two sentences
   ```

2. Record the same prompt in that directory's `images/PROMPTS.md`, the regeneration
   source of truth, with status `pending` or `generated`.
3. Do not add an image link until the file exists, which avoids broken references.
4. After you generate and save the file, a follow-up edit replaces only the placeholder
   cell's "Image pending" line with the embed and alt text, keeps the prompt in
   `PROMPTS.md`, and sets its status to `generated`.

**Planned image inventory (first pass; filenames are frozen in Phase A, and agents add rows for their own scope):**

| Location | Image idea |
|---|---|
| Stage 03 model-evolution map | One-page route: counts → fixed window → RNN → LSTM → fixed-summary reader/writer → reader/writer with look-back → Transformer, each labelled with its complaint |
| Counts and fixed-window notebook | A sliding window that cannot see the clue just outside it |
| Recurrent LM notebook | A notecard passed hand to hand, fading without gates |
| Seq2seq notebook | The reader squeezing a whole sentence onto one card for the writer |
| Seq2seq notebook | The writer looking back at the reader's per-word notes |
| Seq2seq notebook | Practice with the correct previous word versus generation with one's own guess |
| First-use concept map | Where tokens, IDs, embeddings, padding, shifting, and masks first become necessary |
| Stage 04 reader map | Recap versus new mechanism versus deliberate reuse across Parts 1–6 |
| Stage 04 Part 2 bridge | Where RoPE sits in the real path: states → Q/K/V → rotate Q and K → scores → softmax → weighted V |
| Stage 04 block comparison | Encoder, decoder-only, and encoder-decoder writer blocks side by side |

The existing `04-transformers/images/` convention is reused. New directories create
their own `images/` folder only when the first prompt is added.

### 10.3 A contents map in every directory

Add a `CONTENTS.md` to `learning/`, every stage directory, and every chapter or branch
directory containing more than one learning asset, including `optional/`, the proposed
`model-evolution/`, and the proposed `focused/` directories.

- One line per child chapter, notebook, theory note, or subdirectory, in reading order.
- Each line is a link plus a single plain-English sentence stating what the reader
  learns or can do, not the topic name restated:

  ```markdown
  - [01A Sequence Memory Theory](01-sequence-memory/01A-sequence-memory-theory.ipynb) — why a model needs a running memory, and why that memory fades over long sequences.
  ```

- Mark optional items as `(optional)` and returning-reader shortcuts as `(shortcut)`.
- Within a stage contents map, group entries under `Required route` and `Optional`
  without repeating the README's narrative.
- The existing README gains one additive line linking to `CONTENTS.md`; README content
  is otherwise unchanged.

The novice test from 10.1 applies to each line: a newcomer should understand what the
chapter offers without already knowing its terminology.

## 11. Implementation phases and acceptance gates

| Phase | Deliverable | Exit gate |
|---|---|---|
| A. Contract freeze | Frozen file paths, heading anchors, image filenames, and file-ownership table | Every planned file has exactly one owning agent; all later phases may start |
| B. Contents maps | `CONTENTS.md` in every qualifying directory, plus README links | Every child asset appears exactly once; every line passes the novice test |
| C. Foundation bridges | New Stage 03 notebooks/companions, additive navigation/first-use explanations, and image placeholders | Model roles, objective changes, and minimal teacher-forcing intuition are clear; fresh-kernel evidence is retained |
| D. Canonical Transformer additions | Short contextual bridges, plain-English lead-ins, cross-references, and image placeholders only | All original content survives; standalone route still works |
| E. Focused Transformer copy | Explicitly derived, independently runnable optional route | Prerequisites stated; canonical revision recorded; Transformer-specific evidence retained |
| F. Remaining curriculum | Stage-by-stage additive changes using the ledger | Every chapter has an appropriate complaint, owner, evidence boundary, and handoff or an explicit justified disposition |
| G. Image integration | Your generated files embedded at their placeholders | Every `PROMPTS.md` entry is `generated` or explicitly still pending; no broken image links |
| H. Integration and final verification | Ledger merge plus preservation, execution, link, anchor, plain-language, and artifact-contract checks | Every active asset has a ledger disposition; no omissions, unsupported success claims, broken anchors, or collateral changes |

### Parallel execution contract

Phases B-G run **simultaneously** with parallel agents. Instead of waiting on each
other, they depend on a short shared contract, frozen before agents start:

1. **Frozen names and anchors.** The exact paths of every proposed file (Section 6
   notebooks, `focused/` notebooks, `CONTENTS.md`, `images/PROMPTS.md`), image
   filenames from 10.2, and the heading text that cross-references will target.
   Agents link to these agreed destinations even before the files exist; Phase H
   verifies that they resolve.
2. **One owner per file.** Each agent writes only files in its own scope. Shared files
   have one designated owner: the `CONTENTS.md` agent owns every `CONTENTS.md` and the
   one-line README links; each directory's content agent owns that directory's
   `PROMPTS.md`. Agents needing a change elsewhere record it in the ledger for the
   owner instead of editing it.
3. **Fixed source baseline.** The focused copy (E) derives from the pushed baseline
   `e94a33f` plus the agreed bridge text in Section 8, not from D's in-progress edits.
   Phase H reconciles any drift.
4. **Phase A shrinks to contract-freeze.** Only the names, anchors, and ownership table
   are needed first. The full ledger is filled in by each agent for its own scope
   while it works.
5. **No shared runtime state.** Each agent validates in its own temporary notebook
   copies and kernels; no agent commits, pushes, or runs repository-wide formatting.
6. **Single integration pass (H).** After all agents finish, one pass checks links,
   anchors, contents maps against final files, preservation against `e94a33f`, and
   the full worktree, then commits once.

Phase G still waits only on you supplying image files and never blocks the others.

### Preservation and correctness checks

- Compare every existing asset against the baseline before editing. No deletions,
  replacement explanations, renamed runtime artifacts, or lost experiments.
- For additive notebook edits, original cell sources remain intact and in relative
  order; new cells are inserted between them. Any explicitly chosen rearrangement
  needs a content-preservation map and runtime validation.
- Preserve existing metadata, kernel contracts, fixtures, manifests, checkpoints,
  and output schemas. Existing outputs are already cleared; keep source notebooks
  clear and execute only temporary copies.
- Parse notebook JSON and Python cells, check IDs/metadata, execute changed/new
  notebooks in fresh kernels, and compare executed-copy sources with source files.
  Check exact measured claims, not merely successful execution.
- Keep static counterparts for essential animation-based intuition. Preserve useful
  failed attempts and the experiment that motivates each mechanism.
- Verify local links, images, companion links, required/optional navigation, and
  cross-stage handoffs. List future proposed files as proposals until they exist.
- Every image placeholder has a complete prompt mirrored in `images/PROMPTS.md`;
  no image link points at a file that does not exist.
- Every `CONTENTS.md` lists each child asset exactly once, in reading order, with
  working links and a one-sentence plain-English description.
- Every new or bridged section passes the novice test in 10.1; reviewers flag
  undefined terms, stacked jargon, and formula-first explanations.
- Preserve evidence labels and release holds. A toy task, simulation, or local
  fixture does not become production capability evidence through new prose.
- Run relevant existing validators, focused tests where applicable, and
  `git diff --check`; inspect the whole worktree for collateral edits.
- For this plan-only delivery, verify the pushed baseline and that the only new
  curriculum file is this plan. Do not claim future notebook execution has occurred.

The final pedagogical test is: **Can the reader explain why the next mechanism is
needed, using the previous example, before encountering its name or implementation?**
The preservation test is equally strict: **Can a standalone Transformer reader still
find every explanation and experiment that existed before this work?**
