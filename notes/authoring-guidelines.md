# Notes — Universal Authoring Guidelines

> **This is the master authoring reference for all tracks under `notes/`.**  
> Every chapter across every track — AI, AIInfrastructure, ML, MultiAgentAI, MultimodalAI, and InterviewGuides — follows the principles here.  
> Track-specific `AUTHORING_GUIDE.md` files adapt these principles to their domain, running example, and grand challenge.  
> When a track-specific guide conflicts with this document, the track guide wins within its scope.

<!-- STYLE-FINGERPRINT-V1
scope: all_notes_tracks
voice: second_person_practitioner
register: technical_but_conversational
pedagogy: failure_first
formula_rule: verbal_gloss_required_within_three_lines
numerical_walkthroughs: judicious_when_clarifying_never_decorative
callout_system: {insight:"💡", warning:"⚠️", constraint:"⚡", optional_depth:"📖", forward_pointer:"➡️"}
mermaid_color_palette: {primary:"#1e3a8a", success:"#15803d", caution:"#b45309", danger:"#b91c1c", info:"#1d4ed8"}
image_background: dark_facecolor_1a1a2e_for_generated_plots
section_order: [story_header, challenge_0, animation, core_idea, running_example, math, step_by_step, key_diagrams, failure_modes, progress_check, bridge]
red_lines: [no_formula_without_verbal_gloss, no_concept_without_running_example, no_section_without_forward_backward_context, no_math_without_numerical_example, no_callout_without_actionable_content, no_academic_register, no_fuzzy_metrics]
-->

---

## 1 · What These Notes Are For

These notes are **not textbooks**. They are not tutorials. They are not Wikipedia articles.

They are production engineering references for a smart engineer who:
- Has limited time and is preparing for a senior interview or building something real
- Gets annoyed by fluff, hedging, and unnecessary abstractions
- Learns by seeing things break before seeing them work
- Wants to know what to **do** and **why it matters**, not what something *is*

Every chapter must earn its place. Every sentence must deliver information. Every example must be tied to a production system with real stakes and real numbers.

---

## 2 · The Grand Challenge Pattern

Every track threads a single **production system** through all its chapters as the learning vehicle. The system must have:

1. **A named, business-grounded mission** — not "learn ML" but "ship SmartVal AI to production with <$40k MAE"
2. **A concrete stakeholder** — a CEO, lead engineer, or team with real pressure and a real deadline
3. **Measurable constraints** — not "good accuracy" but "<$40k MAE", "<3s p95 latency", ">25% conversion"
4. **Progressive unlocking** — each chapter advances at least one constraint measurably closer to the target
5. **A final outcome** — a fully operational system with all constraints met and an ROI calculation

This is not cosmetic. The grand challenge **is the pedagogical spine**. A reader who loses the thread of the grand challenge has lost the point of the chapter.

### The Grand Challenge Table (Quick Reference)

| Track | Grand Challenge | System | Key Target |
|-------|----------------|--------|-----------|
| **ML / 01-Regression** | SmartVal AI | California Housing | <$40k MAE |
| **ML / 02-Classification** | FaceAI | CelebA | >90% avg accuracy |
| **ML / 03-NeuralNetworks** | UnifiedAI | CA Housing + CelebA | ≤$28k MAE + ≥95% accuracy |
| **ML / 04-Recommender** | FlixAI | MovieLens 100k | >85% hit@10 |
| **ML / 05-AnomalyDetection** | FraudShield | Credit Card Fraud | 80% recall @ 0.5% FPR |
| **ML / 06-RL** | AgentAI | GridWorld + CartPole | CartPole ≥195/200 steps |
| **ML / 07-Unsupervised** | SegmentAI | UCI Wholesale | Silhouette >0.5 |
| **ML / 08-Ensemble** | EnsembleAI | California Housing | Beat single models by 5%+ |
| **AI** | Mamma Rosa's PizzaBot | Conversational AI system | >32% conversion, <$0.07/conv |
| **AIInfrastructure** | InferenceBase | Llama-3-8B self-hosting | <$15k/mo, ≤2s p95 |
| **MultiAgentAI** | OrderFlow | B2B PO automation | 1,000 POs/day, <4hr SLA |
| **MultimodalAI** | VisualForge Studio | Local diffusion pipeline | <30s/image, ≥4.0/5.0 quality |
| **InterviewGuides** | Interview-Ready Engineer | Technical interview prep | Land senior AI/ML role |

---

## 3 · Voice and Register

**The register is: technical-practitioner, second person, conversational within precision.**

The reader is a capable engineer who doesn't need flattery, gets impatient with abstract theory, and wants to know what to do and why it matters. Every sentence earns its place.

### Rules

**1. Second person is the default.**

The reader is placed inside the scenario at all times:
> *"You're the Lead AI Engineer at Mamma Rosa's Pizza."*  
> *"Your manager calls: the luxury coastal segment is haemorrhaging client trust."*  
> *"You just did gradient descent. Very slowly. And by feel."*

**2. Dry, brief humour appears at most once per major concept.** Never laboured. Never cute. "By feel." "The CEO is not amused." "That's it." — these illustrate the register.

**3. Contractions and em-dashes are correct.**
> *"That's it."*  
> *"MSE gives urgency — but it can panic over the wrong things."*  
> *"Full stop."*

**4. Academic register is forbidden.** Do not write:
- ❌ "In this section we demonstrate..."
- ❌ "It can be shown that..."
- ❌ "The reader may note..."
- ❌ "We present / We propose..."
- ❌ "Let us explore..."
- ❌ "This section introduces..."

---

## 4 · Story Header Pattern

Every chapter opens with three specific items **in a blockquote**, in this order:

**Item 1: The story** — who invented this, in what year, on what problem. Specific names, dates, papers. One paragraph. Closes with a sentence connecting the historical moment to the reader's daily engineering work.

**Item 2: Where you are in the curriculum** — one paragraph precisely describing what previous chapter(s) gave the reader and what gap this chapter fills. Must name specific constraint statuses or metric numbers from preceding chapters.

**Item 3: Notation in this chapter** — a single inline sentence with all symbols before the first section. Not a table — a sentence. Example from ML Ch.01:  
*"$x$ — input feature (`MedInc`); $y$ — true target (`MedHouseVal`); $\hat{y}=wx+b$ — model prediction; $w$ — weight; $b$ — bias; $N$ — samples; $L$ — MSE loss; $\eta$ — learning rate."*

---

## 5 · The Challenge Section — § 0

Every chapter opens with § 0. The format is fixed:

```markdown
## 0 · The Challenge — Where We Are

> 🎯 **The mission**: [Grand Challenge name] — [one-line constraint list]

**What we know so far:**
- ✅ [Previous chapter achievements with named metrics]
- ❌ **But we still can't [specific named gap]**

**What's blocking us:**
[2–4 sentences — concrete, named, with numbers.
 Not "non-linearity" but "income–value relationship curves at high incomes (diminishing returns)."
 Not "latency issue" but "6s p95 vs. <3s target → 40% conversation abandonment."]

**What this chapter unlocks:**
[Capability with expected metric improvement — e.g., "RAG: error rate 15% → <5%, conversion 15% → 18%"]
```

**Rules for § 0:**
- The gap is never "our model is not accurate enough" — it is "$55k MAE vs. $40k target — 38% overshoot"
- The blocker is never abstract — it is the specific named failure in the track's running example
- Numbers are always named. Fuzzy language is a red line violation.
- Constraint achievements are marked with ⚡ and a specific evidenced number

---

## 6 · The Failure-First Pedagogical Pattern

> **This is the most important rule.**

Concepts are never listed and explained — they are *discovered by exposing what breaks*.

**The pattern:**
```
Tool → Specific Failure → Minimal Fix → That Fix's Failure → Next Tool
```

If a section covers three methods, the section must show *what breaks* with method 1 before introducing method 2. Listing methods without demonstrating failure is the wrong pattern.

**Canonical example from ML Ch.01 (loss functions):**
- MAE: intuitive → show exactly where it breaks (luxury segment gradient vanishes near plateau)
- MSE: fixes gradient → show exactly where MSE breaks (outlier hijacking, units²)
- RMSE: not a new idea, just a unit converter — same flaw as MSE but readable
- Huber: show exactly how it fixes the MSE/MAE tension with a configurable δ

**The failure must be concrete with numbers.** Not "accuracy drops" but "$55k MAE vs. $40k target — we miss the constraint by 38%." Not "latency degrades" but "6s p95 → 40% conversation abandonment rate, measurable from session logs."

**This pattern applies to every domain:**
- AIInfrastructure: FP16 fills 20GB VRAM → batch=1 max → 3k req/day → quantize to INT4 → 8GB → batch=4 → 12k req/day ✅
- MultiAgentAI: single agent hits 8k context limit after 3 negotiations → multi-agent decomposition → context per agent stays <2k ✅
- MultimodalAI: unconditioned diffusion → random noise output → add CLIP text conditioning → coherent but prompt-agnostic → increase CFG scale to 7.5 → prompt-responsive ✅

---

## 7 · Mathematical Style

**Rule 1: Scalar form before vector form.** Every formula is shown first for one sample or one feature, then generalised. Never open with the matrix form.

> ML Ch.01: shows `ŷ = wx + b` first (one feature), then `ŷ = Wᵀx + b` (multiple features).

**Rule 2: Every formula has a verbal gloss immediately after.** Not in a notation table — in the prose directly below the LaTeX block. Three lines without a gloss = incomplete.

> *"The denominator is the total squared error of predicting the training-set mean ȳ for every district — the dumbest possible baseline. R² is the fraction of that baseline error your model eliminates."*

**Rule 3: The notation table lives in the story header.** Subsections don't introduce new symbols without immediate inline glosses.

**Rule 4: Optional depth goes in a callout box.** Full derivations that break the narrative flow go in `> 📖 **Optional:**` blocks. Always end with a cross-reference to MathUnderTheHood for rigorous treatment.

**Rule 5: ASCII matrix diagrams for matrix operations.** Aligned brackets, dimension annotations, actual numbers where possible. The dimension of every operand and result must be shown.

```
Xᵀ · e                                 (2×3) · (3×1) → (2×1)

  Xᵀ                     e
  ┌  0.5  1.5  2.0  ┐   ┌  -1.5  ┐
  └  1.0  0.0 -1.0  ┘ × │  -2.5  │
                         └  -4.0  ┘
```

**Rule 6: Intuition first, formalism second.** Explain the **why** before the **what**. A formula without motivation is decoration.

❌ **Wrong (formalism first):**  
> "The gradient is computed as ∇L = Xᵀ(ŷ - y). This gives us the direction to update weights."

✅ **Right (intuition first):**  
> "We need to know which direction makes loss smaller. If predictions are too high, reduce the weights. If too low, increase them. This 'which direction' question is answered by the gradient: ∇L = Xᵀ(ŷ - y)."

**Rule 7: Prioritize geometric intuition over algebraic manipulation.** Use diagrams, analogies, and plain-English explanations. Save the algebra for optional depth boxes.

---

## 8 · Using Numerical Examples Judiciously

Numerical examples are powerful pedagogical tools — but only when they **build intuition** rather than demonstrate arithmetic. Use them when concrete numbers make a concept clearer; skip them when they obscure the core idea.

**When to use numerical walkthroughs:**
- ✅ **Introducing a new algorithm** — show one complete iteration with explicit numbers to demystify the mechanics
- ✅ **Debugging a concept** — when readers commonly misunderstand (e.g., gradient accumulation, broadcasting rules, attention weight normalization)
- ✅ **Comparing alternatives** — show same example through two methods to highlight the difference (MAE vs MSE on same residuals)
- ✅ **Validating implementation** — provide "ground truth" numbers readers can reproduce to verify their code

**When to skip numerical walkthroughs:**
- ❌ **Concept is already intuitive** — don't calculate 0.7 × 3.2 = 2.24 if the pattern is obvious
- ❌ **Arithmetic obscures the idea** — if readers will focus on calculation details instead of the principle
- ❌ **Same pattern as previous example** — don't repeat arithmetic for every hyperparameter or every layer
- ❌ **Better shown visually** — use a plot or diagram instead of a table of numbers

**The judicious walkthrough structure (when you do use one):**
1. State the toy dataset as a markdown table with named columns (use the track's running example data, never purely synthetic)
2. State initial conditions: `w = [0, 0]`, `b = 0.0`, `α = 0.1`
3. Show **one complete iteration** with explicit arithmetic: `w₁ = 0.0 − 0.1 × (−8.333) = 0.833`
4. State the outcome with a metric: "MSE dropped from 8.167 → 1.233: an 85% reduction in one epoch"
5. **Close with what this demonstrates:** "This shows gradient descent finds the downhill direction — even with crude α = 0.1 — without trying every possible weight."

**Priority: Intuition over calculation.**  
If a reader finishes a section thinking "I can do the arithmetic" instead of "I understand when to use this," the section failed.

**Examples of intuition-building vs calculation-showing:**

| Concept | ❌ Calculation-Heavy | ✅ Intuition-Building |
|---------|---------------------|----------------------|
| **Gradient Descent** | Show 10 iterations of w₁, w₂, w₃... with full arithmetic | Show ONE iteration with numbers, then explain "this is why it converges — the gradient gets smaller as we approach the minimum" with a plot |
| **Batch Normalization** | Calculate μ, σ², normalized values for all 32 samples | Show one sample before/after normalization, explain "this centers activations so they don't explode or vanish," show distribution plot |
| **Attention Weights** | Compute softmax(QKᵀ/√d) for all 512 tokens | Show 3-token example, explain "softmax picks the most relevant context," show attention heatmap |
| **Learning Rate** | Show loss at α=0.001, 0.01, 0.1, 1.0... | Show one failure case (α too large → divergence), one success (α reasonable → convergence), explain the tradeoff |

**The test:** Can the reader explain the concept to a colleague WITHOUT referring to specific numbers? If not, the section taught arithmetic, not understanding.

---

## 9 · Forward and Backward Linking

Every new concept links to where it was first introduced and where it will matter again. This is not optional.

**Backward link pattern:**  
*"This is the same update rule from Ch.1 — the only difference is that Xᵀ now accumulates contributions from all d features."*

**Forward link pattern:**  
*"This is the conceptual foundation of neural network backpropagation. Every time you call `loss.backward()` in PyTorch, this matrix multiply is running — one per layer."*

**The `> ➡️` callout** plants seeds for concepts introduced in later chapters without derailing the current section.

**Cross-track links** to MathUnderTheHood are standard for rigorous derivations. Always reference the specific chapter:  
`[MathUnderTheHood ch06 — Gradient & Chain Rule](../math_under_the_hood/ch06_gradient_chain_rule)`

---

## 10 · Callout Box System

The meaning of every callout symbol is fixed. Do not improvise with new emoji or new meanings.

| Symbol | Meaning | When to use |
|--------|---------|-------------|
| `💡` | Key insight / conceptual payoff | After a result that reframes something the reader thought they understood |
| `⚠️` | Warning / common trap | Before or after a pattern that is often done wrong |
| `⚡` | Grand Challenge constraint connection | When content advances or validates one of the track's core constraints |
| `> 📖 **Optional:**` | Deeper derivation | Full proofs and math that break the narrative flow for practitioners |
| `> ➡️` | Forward pointer | When a concept needs to be planted before its formal treatment |

**Every callout ends with an actionable conclusion.** A Fix, a Rule, a What-to-do. No callout that just says "this is interesting" without consequence.

---

## 11 · Image and Animation Conventions

**Every image has a purpose.** No decorative diagrams. Every image must demonstrate something the prose cannot fully convey with text.

**Naming convention:**
- `chNN-[topic]-[type].png/.gif` — chapter-specific generated images
- `[concept]_generated.gif/.png` — algorithmically generated animations
- Store in `./img/` relative to the chapter's README

**Dark background requirement:** All generated plots use `facecolor="#1a1a2e"`. Light-background plots are not used in chapters.

**Alt-text is mandatory.** Descriptive, not "Figure 1":  
`![MSE(w) parabola (left) and its linear derivative dL/dw (right), making the residual-to-gradient link explicit](img/loss_parabola_generated.png)`

**Every chapter has a needle GIF** — the chapter-level animation showing which constraint needle moved. Appears immediately after § 0 under `## Animation`.

**Mermaid color palette — used exactly:**

| Role | Fill | Usage |
|------|------|-------|
| Primary / data flow | `#1e3a8a` | Input nodes, data pipeline |
| Success / achieved | `#15803d` | Constraint achieved, goal met |
| Caution / partial | `#b45309` | In-progress, partial unlock |
| Danger / blocked | `#b91c1c` | Failure state, blocker |
| Info / neutral | `#1d4ed8` | Intermediate nodes |

All Mermaid nodes: `stroke:#e2e8f0,stroke-width:2px,color:#ffffff`

---

## 12 · Code Style

**Minimal but complete.** Enough to run end-to-end with real output. Nothing extra.

**Comments explain WHY, not WHAT.**
- ❌ `# fit the scaler`
- ✅ `# fit on TRAIN statistics only — applying to test avoids leakage`

**The manual path alongside the library path.** When teaching a concept (gradient descent, attention, message passing, diffusion), show the manual implementation first, then the library version. Label clearly:
- `# Educational: [concept] from scratch`
- `# Production: [library] equivalent`

**Variable naming is consistent across all chapters of a track.** Declare the naming conventions in each track's AUTHORING_GUIDE. Readers should find the same variable names in Ch.1 and Ch.8.

**Track-specific defaults:**
- ML: `X_train`, `X_test`, `X_train_s` (scaled), `y_train`, `model`, `mae`, `alpha`
- AI: `messages`, `response`, `embedding`, `retrieved_docs`, `system_prompt`
- AIInfrastructure: `model_size_gb`, `vram_gb`, `batch_size`, `throughput_rps`, `p95_latency_s`
- MultiAgentAI: `agent_id`, `message`, `context`, `tool_call`, `result`
- MultimodalAI: `prompt`, `negative_prompt`, `guidance_scale`, `num_inference_steps`, `image`

---

## 13 · The Progress Check Section

The last substantive section of every chapter. Fixed format:

```markdown
## N · Progress Check — What We Can Solve Now

✅ **Unlocked capabilities:**
- [specific capability with named metric — e.g., "RAG: menu error rate 15% → 4.2% ✅"]
- [constraint achievement — e.g., "Constraint #2 ACCURACY ✅ ACHIEVED: <5% error rate"]

❌ **Still can't solve:**
- [named, specific gap with numbers — e.g., "6s p95 latency → 40% abandonment; target <3s"]
- [what's next and why it's blocked]

**Constraint status:**

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 [NAME] | ✅/❌/⚡ | [current metric vs. target] |
| ...        | ...    | ...           |

**[Mermaid LR flowchart showing all chapters, current one highlighted, metrics annotated]**
```

The Mermaid flowchart always shows the **full forward arc** — not just the current chapter. It anchors the reader in the overall narrative even when deep in one chapter's detail.

---

## 14 · Chapter Section Ordering

Every chapter follows this section sequence. Sections may be combined or have sub-sections but the order is fixed:

```
[Blockquote header: story, curriculum position, notation]

## 0 · The Challenge — Where We Are
## Animation                        ← needle GIF immediately after § 0
## 1 · Core Idea                    ← 2–3 sentences, plain English
## 2 · Running Example              ← tie the concept to the track's production scenario
## 3 · The Math                     ← formulas, annotated, scalar-first
## 4 · How It Works — Step by Step  ← numbered walkthrough or flow diagram
## 5 · Key Diagrams                 ← Mermaid / ASCII art minimum 1
## 6 · The Hyperparameter Dial      ← main tunable, effect, typical value
## 7 · What Can Go Wrong            ← 3–5 failure modes, one sentence each
## N-1 · Where This Reappears       ← forward links
## N · Progress Check               ← fixed format above
## N+1 · Bridge to Next Chapter     ← one clause what this established + one clause what next adds
```

Supplement documents (e.g., `notebook_supplement.ipynb_solution.ipynb` (reference) or `notebook_supplement.ipynb_exercise.ipynb` (practice)) skip § 0, § Animation, and Progress Check. They are deep technical dives for advanced readers.

---

## 15 · Red Lines — Never Do These

Absolute prohibitions. No exception, no track-level override:

| # | Red Line | Why It Matters |
|---|----------|---------------|
| 1 | **No formula without a verbal gloss** | Equations without explanation are decoration; the reader learns nothing |
| 2 | **No concept without grounding in the running example** | Abstract explanations disconnect learning from practice |
| 3 | **No section without forward/backward context** | Isolated concepts don't stick; only connected ones do |
| 4 | **No math derivation without intuition-building support** | Use numerical examples when they clarify; use diagrams, verbal glosses, or analogies when numbers obscure the concept |
| 5 | **No callout box without actionable content** | Every 💡⚠️⚡ ends with a Fix, Rule, or What-to-do |
| 6 | **No academic register** | "It can be shown that" and "In this section we" are banned |
| 7 | **No fuzzy metrics** | "Higher accuracy" is forbidden; "$55k MAE vs. $40k target" is required |
| 8 | **No forward reference without a ➡️ callout** | Plant every concept formally before using it downstream |
| 9 | **No supplement with a § 0 Challenge section** | Supplements are deep dives, not narrative chapters |
| 10 | **No chapter without a Progress Check** | Every chapter must close with the constraint status update |

---

## 16 · Completing a Chapter — Validation Checklist

Before marking a chapter complete:

- [ ] Story header: historical context → curriculum position → notation declaration ✅
- [ ] § 0 Challenge section with concrete failure scenario and named blocker ✅
- [ ] Needle GIF present under `## Animation` ✅
- [ ] Failure-first structure in every multi-option subsection ✅
- [ ] Every formula verbally glossed within 3 lines ✅
- [ ] At least one numerical walkthrough with explicit arithmetic ✅
- [ ] Forward/backward links on every new concept ✅
- [ ] Callout boxes use the standard symbols only ✅
- [ ] Images in `./img/` with dark background and descriptive alt-text ✅
- [ ] Code blocks labelled Educational vs Production ✅
- [ ] Progress Check with constraint status table and Mermaid arc ✅
- [ ] Bridge paragraph closes the chapter ✅
- [ ] No academic register, no fuzzy metrics, no decoration ✅

---

## 17 · See Also

Each track's `AUTHORING_GUIDE.md` extends this document with:
- The full chapter tracker and build status
- Track-specific dataset and running example details
- The specific constraint progression across chapters
- Track-specific code conventions and variable names
- Track-specific diagram requirements

| Track | AUTHORING_GUIDE |
|-------|----------------|
| **ML** | [notes/01-ml/authoring-guide.md](01-ml/authoring-guide.md) — canonical style reference (deepest documentation) |
| **AI** | [notes/03-ai/AUTHORING_GUIDE.md03-ai/authoring-guide.md) |
| **AIInfrastructure** | [notes/AIInfrastructure/AUTHORING_GUIDE.md](ai_infrastructure/authoring-guide.md) |
| **MultiAgentAI** | [notes/MultiAgentAI/AUTHORING_GUIDE.md](multi_agent_ai/authoring-guide.md) |
| **MultimodalAI** | [notes/MultimodalAI/AUTHORING_GUIDE.md](multimodal_ai/authoring-guide.md) |
| **InterviewGuides** | [notes/InterviewGuides/AUTHORING_GUIDE.md](interview_guides/authoring-guide.md) |
| **MathUnderTheHood** | [notes/MathUnderTheHood/AUTHORING_GUIDE.md](math_under_the_hood/authoring-guide.md) |

---

*Last updated: April 2026 — applies to all tracks under `notes/`*
