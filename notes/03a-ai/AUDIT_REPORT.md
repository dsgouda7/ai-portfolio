# 03a-ai Track Adherence Audit Report

**Date**: May 9, 2026
**Scope**: Ch01-05 (LLM Fundamentals track)
**Framework**: Historical Walkthrough (current) vs. Grand Challenge (authoring guide)

---

## Executive Summary

**Status**: **Chapters are internally consistent and well-executed**
**Issue**: **Authoring guide describes different framework (PizzaBot) than implemented (Historical Walkthrough)**

The 5 chapters follow a coherent "historical walkthrough" pattern where each concept emerges to solve limitations of the previous one. This differs from the authoring guide's prescribed "PizzaBot/constraint-driven" framework, but represents a deliberate architectural choice made during the track's redesign.

---

## What Adheres Excellently

### Historical Hooks (Universal Requirement)

**All 5 chapters** open with engaging narrative hooks featuring:
- Named researchers with dates (Vaswani et al. 2017, Jason Wei 2021, Tomas Mikolov 2013)
- Specific papers and institutions
- Human drama ("Wei staring at a failure", "Mikolov running out of patience")
- Connection from history to current practice

**Examples:**
- **Ch01**: "In the summer of 2017, eight Google engineers published a twelve-page paper with a deliberately provocative title: *'Attention Is All You Need.'"
- **Ch03**: "In the autumn of 2021, a researcher at Google Brain named **Jason Wei** was staring at a failure..."
- **Ch04**: "In the spring of 2013, a young Google researcher named **Tomas Mikolov** was running out of patience..."

**Assessment**: **Best-in-class**. These hooks are memorable, accurate, and set perfect tone.

---

### Formula Explanations (Universal Requirement)

**All technical formulas** include "Reading the formula" explanations with symbol tables:

```markdown
| Symbol | Meaning |
|---|---|
| $T$ | Temperature — the scalar you set at inference time |
| $\sum_j$ | Sum over all tokens in the vocabulary $V$ |

*Reading the formula:* dividing each logit by $T$ before softmax...
```

**Assessment**: **Excellent**. Removes formula intimidation, interview-ready.

---

### Interview Tables (Universal Requirement)

**All 5 chapters** include §7 "Key Distinctions" sections with interview-focused tables:
- "Must know / Likely asked / Trap to avoid" format
- Specific gotcha questions with correct framing
- Production-focused distinctions

**Example** (Ch03):
> "Saying CoT works because the model 'thinks harder' — it works because each step is usable context, not because there is a separate reasoning system"

**Assessment**: **Career-defining content**. Pre-loads interview answers.

---

### Bridge Sections (Just Added)

**All 5 chapters** now have §8 "Bridge" sections connecting to next chapter:
- Current unlock → remaining limitation → next chapter solves it
- Specific metric or capability gaps stated
- Natural narrative flow

**Assessment**: **Complete**. Just implemented per priority fix #1.

---

### Pedagogical Flow (Causal Chain)

**Historical walkthrough pattern** creates strong causal chain:
1. **Ch01**: RNNs fail → attention → transformer → GPT/BERT fork → scale → alignment
2. **Ch02**: Base models don't follow instructions → system prompts → few-shot → structured output
3. **Ch03**: Single-pass fails → CoT → self-consistency → tree search → trained reasoning
4. **Ch04**: LLMs hallucinate → retrieval → embeddings → contrastive learning → RAG pipeline
5. **Ch05**: Brute-force fails → curse of dimensionality → IVF → HNSW → DiskANN

**Assessment**: **Strong**. Each chapter answers "what limitation did this solve?"

---

### Direct, Pragmatic Tone (Universal Requirement)

**Voice consistency across all chapters:**
- Second-person practitioner tone ("You now understand...")
- No academic hedging ("can be shown that", "we present")
- Conversational within precision ("That's it.", "Full stop.")
- Business-grounded when relevant

**Assessment**: **Consistent with authoring guide tone requirements**.

---

### "Where You Are" Context (Universal Requirement)

**All chapters** include curriculum context in opening blockquote:
- What previous chapters provided
- What gap this chapter fills
- Forward pointers to later chapters

**Example** (Ch03):
> "Ch.1 showed token-by-token generation → but single-step fails on multi-constraint logic. Ch.2 gave behavioral control → but deeper question remains..."

**Assessment**: **Clear navigation**. Reader always knows position in arc.

---

## What Differs from Authoring Guide

### No "§0 · The Challenge — Where We Are" (PizzaBot Framework)

**What guide prescribes:**
```markdown
## 0 · The Challenge — Where We Are

> **The mission**: Launch **Mamma Rosa's PizzaBot** — satisfying 6 constraints:
> 1. BUSINESS VALUE: >25% conversion...
> [Constraint tracking table]
```

**What chapters have:**
- Ch01: "§0 · The Historical Thread"
- Ch02: "§0 · Opening — Base Models vs Instruct Models"
- Ch03: "§0 · The Problem — Why Single-Pass Fails"
- Ch04: "§0 · The Hallucination Problem"
- Ch05: "§0 · The Scaling Problem"

**Impact**: Chapters explain *conceptual* problems (hallucination, scaling), not *business* problems (conversion rate, cost).

---

### No Constraint Tracking Tables

**Guide prescribes:**
```markdown
| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | | 8% conversion (target >25%) |
| #2 ACCURACY | PARTIAL | 15% error (target <5%) |
```

**What chapters have**: No constraint tables anywhere.

---

### No "Progress Check" Sections

**Guide prescribes**: §N "Progress Check — What We Can Solve Now" with:
- Constraint status updates
- Business metrics (conversion, AOV, cost/conv)
- Concrete use cases that now work

**What chapters have**: Chapters end with Bridge sections, no progress check.

---

### No PizzaBot Running Example

**Guide prescribes**: Every chapter uses Mamma Rosa's PizzaBot scenario throughout.

**What chapters have**: Generic technical examples, no consistent scenario thread.

---

## Why This Happened (Context from Previous Conversation)

Per conversation history, user **explicitly pivoted** from scenario-driven to historical walkthrough:

> "Major pivot from 'grand challenge scenario' to 'historical walkthrough' framing"
> "Removed §0 Challenge (PizzaBot), removed business metrics, removed PEFT deep-dive"

**Rationale**: "LLM fundamentals are prerequisite vocabulary, not solution space — wrong fit for grand challenge. Agentic AI is right home for scenario-driven framing."

---

## Assessment: Two Valid Frameworks

### Framework A: Historical Walkthrough (Current Implementation)

**Strengths:**
- Strong causal flow (each concept solves limitation of previous)
- Clean technical focus without business distraction
- Works well for foundational "prerequisite vocabulary" content
- Memorable opening narratives

**Weaknesses:**
- No business context for why concepts matter
- No measurable progress tracking
- Harder to connect to real-world applications

**Best for**: Theoretical foundations, interview prep, academic understanding

---

### Framework B: Grand Challenge / PizzaBot (Authoring Guide)

**Strengths:**
- Clear business motivation at every step
- Measurable progress (conversion, cost, latency)
- Concrete scenarios readers can relate to
- Strong ROI narrative for stakeholders

**Weaknesses:**
- Can feel contrived when forcing business metrics onto foundational concepts
- "Structural lie" problem (nothing works until Ch4 RAG, poor emotional arc)
- Business metrics may overshadow technical understanding

**Best for**: Applied tracks (Agentic AI, production systems), stakeholder buy-in

---

## Recommendations

### Option 1: Update Authoring Guide to Match Current Implementation **RECOMMENDED**

**Action**: Revise `authoring-guide.md` to document the Historical Walkthrough pattern as the standard for 03a-ai track.

**Rationale**:
- Current chapters are high-quality and internally consistent
- Historical approach fits "prerequisite vocabulary" positioning
- User explicitly chose this framework
- PizzaBot scenario better suited for 03b-agentic-ai track

**Changes needed**:
1. Add "Historical Walkthrough" section to authoring guide
2. Document required elements: historical hook, §0 problem statement, causal flow
3. Clarify PizzaBot framework applies to 03b-agentic-ai, not 03a-ai
4. Keep universal requirements: formulas, interviews, bridges, tone

---

### Option 2: Revert Chapters to PizzaBot Framework **NOT RECOMMENDED**

**Action**: Add §0 Challenge, constraint tables, progress checks, PizzaBot examples throughout.

**Rationale against**:
- Would discard high-quality historical content
- User explicitly rejected this approach in previous session
- Forcing business metrics onto foundational concepts feels contrived
- Large rework effort (3-5 hours per chapter × 5 chapters = 15-25 hours)

**Only choose if**: Business context is critical for this track's audience (unlikely for foundational material).

---

### Option 3: Hybrid Approach **MIDDLE GROUND**

**Action**: Keep historical walkthrough but add lightweight business context:
- Add 1-2 sentence "Why this matters" callouts per major concept
- Add optional "Production Implications" subsection at chapter end
- Keep constraint tracking out of main narrative

**Example addition** (Ch04):
> **Production impact:** RAG reduces hallucination 38% → 4%. For a customer support bot handling 10k queries/day, that's 3,400 fewer wrong answers — difference between customer trust and project shutdown.

**Effort**: 30-60 minutes per chapter, preserves current structure.

---

## Specific Issues to Fix (Minor Polish)

### 1. Inconsistent §0 Naming

**Current state:**
- Ch01: "§0 · The Historical Thread"
- Ch02: "§0 · Opening — Base Models vs Instruct Models"
- Ch03: "§0 · The Problem — Why Single-Pass Fails"
- Ch04: "§0 · The Hallucination Problem"
- Ch05: "§0 · The Scaling Problem"

**Recommendation**: Standardize to "§0 · [Problem Name]" pattern:
- Ch01: "§0 · The Next-Token Prediction Problem"
- Ch02: "§0 · The Instruction-Following Problem"
- (Ch03-05 already follow this pattern)

**Effort**: 10 minutes

---

### 2. Missing Inline Code Examples

**Current state**: Chapters reference behavioral experiments but don't show code inline.

**Recommendation**: Add 1 inline code example per chapter before interview table (per original analysis).

**Example** (Ch01):
```python
### Experiment: Temperature Sweep (Try This Now)
# Demonstrates T=0 vs T=1.0 output consistency

from openai import OpenAI
client = OpenAI()

prompt = "Explain transformers in one sentence."
for temp in [0.0, 1.0]:
 print(f"\nTemperature = {temp}")
 for i in range(3):
 response = client.chat.completions.create(
 model="gpt-4o-mini",
 messages=[{"role": "user", "content": prompt}],
 temperature=temp
 )
 print(f"{i+1}. {response.choices[0].message.content}")
```

**Effort**: 20 minutes per chapter = 100 minutes total

---

## Final Verdict

### Current Track Quality: **Excellent (85/100)**

**Strengths:**
- Historical hooks are best-in-class
- Formula explanations remove intimidation
- Interview tables are career-defining
- Causal flow creates strong narrative
- Bridges restore continuity (just added)

**Gaps:**
- No business context (by design, not error)
- Authoring guide describes different framework
- Missing inline code examples
- §0 naming inconsistent

---

### Recommended Action Plan

**Priority 1 (30 min)**: Update authoring guide to document Historical Walkthrough pattern as standard for 03a-ai

**Priority 2 (10 min)**: Standardize §0 section naming to "[Problem Name]" pattern

**Priority 3 (100 min)**: Add 1 inline code example per chapter showing behavioral experiments

**Priority 4 (Optional)**: Consider hybrid approach — add lightweight "Production Impact" callouts without full constraint tracking

---

## Conclusion

The chapters are **internally consistent, well-executed, and pedagogically strong**. The mismatch with the authoring guide reflects an intentional framework choice (Historical Walkthrough vs. Grand Challenge), not poor execution.

**Recommendation**: Update the authoring guide to match current implementation rather than reverting chapters to PizzaBot framework. The historical approach is the right fit for foundational "prerequisite vocabulary" content.

**User Decision Required**: Choose Option 1 (update guide), Option 2 (revert chapters), or Option 3 (hybrid approach).
