# Ch.11 — Advanced Agentic Patterns: Reflection, Debate, and Orchestration

→ Interview prep: [Interview Guide](../../interview-guides/agentic-ai.md)

> **The story.** In 2022–2023, as GPT-3.5 powered the first wave of production agents, researchers discovered a pattern: single-pass LLM reasoning failed catastrophically on edge cases. Shunyu Yao (Princeton) published *ReAct* in October 2022, showing that reasoning + acting loops beat raw prompting. Two months later, Noah Shinn (Northeastern) released *Reflexion*, proving that agents could critique and revise their own outputs. Around the same time, Yujia Li's team at DeepMind found that multi-agent debate (propose → challenge → defend → vote) outperformed single-agent reasoning on complex tasks by 40%. These weren't theoretical exercises—they were survival strategies. Production agents at Stripe, Intercom, and Shopify were hallucinating, contradicting themselves, and failing on ambiguous inputs. The breakthrough wasn't better models; it was better *orchestration*. Trade tokens for reliability. Generate → critique → revise. Single-pass prediction < iterative refinement.
>
> **Where you are in the curriculum.** Chapters 1–10 taught you to build single-pass agents. Ch.1 (LLM Fundamentals) gave you the model. Ch.2 (Prompt Engineering) taught you to steer it. Ch.3 (Chain-of-Thought) added reasoning. Ch.4–5 (RAG + Vector DBs) grounded it in facts. Ch.6 (ReAct + Semantic Kernel) taught tool-calling orchestration. Ch.7 (Safety) taught guardrails. Ch.8 (Evaluation) taught measurement. Ch.9 (Cost/Latency) taught production constraints. Ch.10 (Fine-Tuning) taught task specialization. **But every pattern in Ch.1–10 assumes the agent gets one shot**. One prompt, one response, done. In this chapter, you'll learn four patterns that unlock iterative refinement: **Reflection** (self-critique), **Debate** (multi-agent reasoning), **Hierarchical Orchestration** (planner → workers → verifier), and **Tool Selection** (pick the right tool, retry when it fails). These patterns now power every major agent system: ChatGPT Code Interpreter (reflection + hierarchical), Claude artifacts (debate + verification), GitHub Copilot (hierarchical orchestration with fallback chains). This is where single-agent reasoning stops and agentic *systems* begin.
>
> **Notation in this chapter:**
> - **LLM(prompt)** → response from single-pass call
> - **Draft()** → initial answer generation
> - **Critique(draft)** → self-assessment of draft quality
> - **Revise(draft, critique)** → improved answer incorporating critique
> - **Propose(agent_i)** → agent i's solution proposal
> - **Vote(proposals)** → arbiter selecting best proposal
> - **Plan(task)** → decomposition into subtasks
> - **Execute(subtask)** → worker agent completing subtask
> - **Verify(result, plan)** → validation against original plan
> - **Tool(input)** → tool execution with input
> - **Fallback(tool_sequence)** → retry chain when tools fail
> - **T(pattern)** → token cost of pattern (relative to single-pass baseline T=1)

---

## 0 · The Challenge — Where We Are

> 🎯 **The goal**: Launch **PizzaBot v2.0** — an intelligent pizza ordering agent satisfying 6 production constraints:
> 1. **ACCURACY**: Handle 99%+ of customer orders without human escalation
> 2. **EDGE CASE ROBUSTNESS**: Handle contradictory inputs (gluten-free + extra cheese), pricing conflicts, menu ambiguities with <1% error rate
> 3. **COST EFFICIENCY**: ≤$0.25 per conversation (3× single-pass budget allowed for complex orders)
> 4. **LATENCY**: <15s for complex orders requiring refinement (vs. <5s for simple orders)
> 5. **TRANSPARENCY**: Show reasoning steps to customers ("Let me verify that pricing...")
> 6. **GRACEFUL DEGRADATION**: When uncertain, ask clarifying questions instead of hallucinating

**What we know so far:**

✅ **Ch.1–10 gave us single-pass agent capabilities:**
- Ch.1: GPT-4 can understand natural language orders
- Ch.2: Prompt engineering steers tone and format
- Ch.3: Chain-of-thought breaks down multi-item orders
- Ch.4–5: RAG + vector DB grounds menu lookups (no hallucinated items)
- Ch.6: ReAct orchestration calls tools (inventory check, pricing calculator)
- Ch.7: Safety guardrails prevent prompt injection
- Ch.8: Evaluation framework measures accuracy
- Ch.9: Cost/latency optimization hits $0.08/conversation, <5s latency
- Ch.10: Fine-tuning specializes model on pizza domain vocabulary

**This stack handles 92% of orders perfectly.** Simple cases work:

```
Customer: "I want a large pepperoni pizza for delivery."
PizzaBot v1.0: ✅ "Large pepperoni ($14.99) + delivery ($3.00) = $17.99.
                    Estimated delivery: 35 minutes. Confirm order?"
```

**But edge cases still break:**

❌ **Edge case #1: Contradictory constraints**
```
Customer: "I want a large pepperoni pizza, but make it gluten-free,
           dairy-free, and add extra cheese."
PizzaBot v1.0: ❌ "Error: dairy-free and extra cheese are incompatible."
                    [Customer abandons order]
```

❌ **Edge case #2: Pricing conflicts (overlapping discounts)**
```
Customer: "I have a 20% off coupon, a $5 loyalty reward, and the
           current 'Buy One Get One 50% Off' promo. Which applies?"
PizzaBot v1.0: ❌ "Only one discount can be applied per order."
                    [Wrong! Policy says: coupons + loyalty stack, but promo replaces both]
                    [Customer disputes charge]
```

❌ **Edge case #3: Complex catering orders**
```
Customer: "I need 15 pizzas for a company event: 5 at 12pm, 5 at 1pm,
           5 at 2pm. Budget is $200 total. What can you do?"
PizzaBot v1.0: ❌ "15 large pizzas = $224.85, over budget."
                    [Doesn't explore: medium pizzas, split toppings, promo codes]
                    [Misses $180 solution that exists]
```

**What's blocking us:**

Single-pass reasoning forces the model to get everything right in one attempt. No room for:
- Self-correction when initial answer contradicts constraints
- Exploring multiple solutions (discount strategies, pizza combinations)
- Decomposing complex orders into validated subtasks
- Recovering gracefully from tool failures (DB timeout → retry with cached estimate)

**Current performance on 1,000-order test set:**

| Metric | Ch.10 (single-pass) | Target | Gap |
|--------|---------------------|--------|-----|
| **Simple orders** (1-2 items, no edge cases) | 98% ✅ | 95% | +3% |
| **Edge cases** (contradictions, conflicts) | 92% ❌ | 99% | -7% |
| **Complex orders** (catering, multi-constraint) | 85% ❌ | 95% | -10% |
| **Overall accuracy** | 95% ❌ | 99% | -4% |
| **Customer escalation rate** | 8% ❌ | <1% | -7% |
| **Avg tokens/conversation** | 850 | 2,550 (3× budget) | OK |
| **Avg latency** | 4.2s | <15s | OK |

**What this chapter unlocks:**

Four patterns that trade tokens for reliability:

1. **Reflection** (self-critique): Generate → Critique → Revise loop fixes contradictions
   - Example: Detects "dairy-free + extra cheese" conflict → suggests vegan cheese
   - Cost: 3× tokens (draft + critique + revision)
   - Impact: Edge case accuracy 92% → 98%

2. **Debate & Consensus** (multi-agent): Multiple agents propose solutions, arbiter picks best
   - Example: Pricing conflict → Agent1 (generous) vs Agent2 (strict) vs Judge (policy)
   - Cost: N agents × tokens per round (typically 3 agents, 2 rounds = 6× baseline)
   - Impact: Pricing dispute accuracy 85% → 97%

3. **Hierarchical Orchestration** (planner → workers → verifier): Complex tasks decomposed
   - Example: Catering order → Planner splits into 3 batches → Workers execute → Verifier checks budget
   - Cost: 1 planner + N workers + 1 verifier (typically 1 + 3 + 1 = 5× baseline)
   - Impact: Complex order accuracy 85% → 96%

4. **Tool Selection** (smart retry chains): Try fast tool → if fails, escalate to expensive tool
   - Example: Inventory check: cached → DB → API → human escalation
   - Cost: Variable (1.1× for cache hit, 2.5× for full escalation)
   - Impact: Tool failure recovery 60% → 95%

**Expected outcomes after Ch.11:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Overall accuracy | 95% | 99.2% | +4.2% ✅ |
| Edge case handling | 92% | 98.5% | +6.5% ✅ |
| Complex order handling | 85% | 96% | +11% ✅ |
| Escalation rate | 8% | 0.8% | -7.2% ✅ |
| Avg cost/conversation | $0.08 | $0.18 | +$0.10 (still under $0.25 budget) ✅ |
| Avg latency | 4.2s | 11.3s | +7.1s (still under 15s limit) ✅ |

---

## 📽️ Animation Reference

> ⚠️ **Placeholder for animation assets** (to be generated by animation subagents):
>
> **Decision flow animations** (see `notes/03-ai/ch11_advanced_agentic_patterns/gen_scripts/`):
> - `gen_reflection_flow.py` → `img/reflection-flow.gif`: Draft → Critique → Revise loop with contradiction detection
> - `gen_debate_flow.py` → `img/debate-flow.gif`: Propose → Challenge → Defend → Vote loop with 3 agents
> - `gen_hierarchical_flow.py` → `img/hierarchical-flow.gif`: Planner → Workers → Verifier decomposition
> - `gen_tool_selection_flow.py` → `img/tool-selection-flow.gif`: Fallback chain with cost/success rates
>
> These will be referenced inline in §3–§6 below.

---

## 1 · The Core Idea

**Plain English:** Iterative refinement beats one-shot prediction for complex reasoning tasks. Trade tokens for reliability. When a single-pass agent would fail (contradictions, ambiguity, complexity), a refinement pattern succeeds:

- **Reflection**: The agent critiques its own output and revises it. Cost: 3× tokens. Gain: 6× error reduction on ambiguous inputs.
- **Debate**: Multiple agents propose solutions, challenge each other, and vote. Cost: N agents × tokens. Gain: 40% better decisions on high-stakes tasks (medical, legal, fraud).
- **Hierarchical**: Planner decomposes task → Workers execute subtasks → Verifier checks. Cost: 1 + N + 1 calls. Gain: 15× success rate on multi-step tasks.
- **Tool Selection**: Try fast/cheap tools first, escalate on failure. Cost: 1.1× (cache hit) to 2.5× (full escalation). Gain: 95% recovery from tool failures.

**Why it works:** LLMs are good at local reasoning (one-step inference) but struggle with self-consistency checks. Reflection adds an external critique step. Debate adds adversarial checks. Hierarchical adds decomposition. Tool selection adds runtime adaptability.

**The trade-off:** Every pattern costs more tokens. Single-pass: 850 tokens ($0.08). Reflection: 2,550 tokens ($0.24). Debate: 5,100 tokens ($0.48). The question isn't "which is best?" but "which pattern for which problem?"

**When to use each:**

| Pattern | When to use | Token cost | Error reduction | Example use case |
|---------|-------------|------------|-----------------|------------------|
| **Single-pass** | Simple, unambiguous inputs | 1× | Baseline | "Large pepperoni pizza" |
| **Reflection** | Ambiguous inputs, contradictions | 3× | 6× | "Gluten-free + extra cheese" |
| **Debate** | High-stakes, subjective decisions | 6× | 8× (on complex tasks) | Medical diagnosis, legal reasoning |
| **Hierarchical** | Multi-step, decomposable tasks | 5× | 15× (on catering-like tasks) | Catering orders, research tasks |
| **Tool Selection** | Multiple tools, failure-prone | 1.1×–2.5× | 2× recovery rate | Inventory lookup, API calls |

---

## 2 · Running Example: PizzaBot v2.0 — Edge Case Handling

**Scenario:** PizzaBot v1.0 (Ch.1–10) handles 92% of orders perfectly but fails on edge cases. We need 99%+ accuracy for production launch.

**Three canonical edge cases:**

### Edge Case #1: Contradictory Constraints (Reflection)

```
Customer: "I want a large pepperoni pizza, but make it gluten-free,
           dairy-free, and add extra cheese."
```

**PizzaBot v1.0 (single-pass):**
```
❌ Response: "Error: You requested dairy-free, but extra cheese contains dairy.
              Please clarify your order."
```
**Problem:** Technically correct, but unhelpful. Customer meant "vegan cheese." 8% of customers abandon at this point.

**PizzaBot v2.0 (with Reflection):**

**Step 1 — Draft (generate initial response):**
```python
draft = LLM("Customer ordered: gluten-free + dairy-free + extra cheese.
             Generate order confirmation.")
# Output: "Error: dairy-free and extra cheese are incompatible."
```

**Step 2 — Critique (self-assessment):**
```python
critique = LLM(f"Assess this draft: '{draft}'
                Customer context: They ordered dairy-free + extra cheese.
                Is there a better interpretation?")
# Output: "The draft is technically correct but unhelpful. Customer likely
#          wants vegan cheese. Check if gluten-free + vegan mozzarella is available."
```

**Step 3 — Revise (incorporate critique):**
```python
revised = LLM(f"Revise: '{draft}'
               Critique: '{critique}'
               Use menu API to check vegan cheese availability.")
# Calls menu API: vegan_mozzarella = available, +$2.50
# Output: "For dairy-free, we use vegan mozzarella. Would you like extra
#          vegan mozzarella (+$2.50) on your gluten-free pepperoni pizza?"
```

**Customer response:** ✅ "Yes, perfect!" (Conversion rate: 92% → 98%)

**Cost:** 850 tokens (draft) + 650 tokens (critique) + 950 tokens (revision) = **2,450 tokens** ($0.23) vs. $0.08 single-pass. **Worth it:** $0.15 extra cost prevents $25 abandoned order (1,667% ROI).

---

### Edge Case #2: Pricing Conflicts (Debate)

```
Customer: "I have a 20% off coupon (SAVE20), a $5 loyalty reward, and I see
           a 'Buy One Get One 50% Off' promo banner. What's my total?"
```

**Policy (stored in vector DB):**
- Coupons + loyalty rewards **stack** (both apply)
- Promotional discounts (BOGO) **replace** coupons + loyalty (pick best)
- Edge case: BOGO usually wins for 2+ pizzas, but not always (depends on base price)

**PizzaBot v1.0 (single-pass RAG):**
```
# RAG retrieves policy: "Only one promotional discount per order."
❌ Response: "Your total is $22.49 with the BOGO promo. The coupon and
              loyalty reward don't apply when using a promo."
# WRONG! Should have calculated both paths and picked cheaper one.
```

**Problem:** Policy is ambiguous ("pick best discount"). Model picked BOGO by default without comparing.

**PizzaBot v2.0 (with Debate & Consensus):**

**Step 1 — Propose (two pricing strategies):**
```python
# Agent 1 (Generous): Apply all discounts that don't conflict
proposal_1 = LLM("Customer has SAVE20 coupon, $5 loyalty, BOGO promo.
                  Policy: coupons + loyalty stack, promo replaces both.
                  Calculate: Coupon + loyalty total.")
# Output: "2 large pizzas ($29.98) → 20% off ($23.98) → $5 loyalty ($18.98)"

# Agent 2 (Strict): Apply promotional discount
proposal_2 = LLM("Customer has SAVE20 coupon, $5 loyalty, BOGO promo.
                  Policy: promo replaces coupons + loyalty.
                  Calculate: BOGO total.")
# Output: "2 large pizzas ($29.98) → 50% off 2nd pizza ($22.49)"
```

**Step 2 — Challenge (agents critique each other):**
```python
challenge_1_to_2 = LLM(f"Agent 2 says: {proposal_2}.
                        But policy says 'pick best discount'. Did you check
                        if coupon + loyalty is cheaper?")
# Agent 2: "No, I assumed BOGO is always better for 2 pizzas."

challenge_2_to_1 = LLM(f"Agent 1 says: {proposal_1}.
                        But promotional discount exists. Policy says it replaces
                        coupons + loyalty. Did you justify why coupon + loyalty
                        should apply instead?")
# Agent 1: "Yes. $18.98 (coupon + loyalty) < $22.49 (BOGO). Customer saves $3.51 more."
```

**Step 3 — Vote (arbiter picks best):**
```python
judge = LLM(f"Proposal 1: {proposal_1} = $18.98.
             Proposal 2: {proposal_2} = $22.49.
             Customer should pay whichever is cheaper.
             Which proposal is correct?")
# Output: "Proposal 1 ($18.98) is correct. Policy says 'pick best discount',
#          and coupon + loyalty is $3.51 cheaper than BOGO."
```

**Final response:** ✅ "Your total is **$18.98** (20% coupon + $5 loyalty reward). That's $3.51 cheaper than the BOGO promo, so we applied the coupon + loyalty instead."

**Customer response:** ✅ "Wow, thanks for checking that!" (Dispute rate: 15% → 3%)

**Cost:** 3 agents × 2 rounds = 6 LLM calls × 800 tokens = **4,800 tokens** ($0.45). **Worth it:** Prevents $25 refund dispute (5,555% ROI).

---

### Edge Case #3: Complex Catering Order (Hierarchical Orchestration)

```
Customer: "I need 15 pizzas for a company event. 5 delivered at 12pm, 5 at 1pm,
           5 at 2pm. Budget is $200 total. Mix of vegetarian and meat options.
           What can you do?"
```

**PizzaBot v1.0 (single-pass):**
```
# Chain-of-thought reasoning:
"15 large pizzas × $14.99 = $224.85. Over budget by $24.85."
❌ Response: "Your order total is $224.85, which exceeds your $200 budget.
              Would you like to reduce the quantity?"
```

**Problem:** Didn't explore solution space. Misses: medium pizzas ($11.99), half-topping splits, bulk discount (10+ pizzas = 15% off), promo code CATERING20.

**PizzaBot v2.0 (with Hierarchical Orchestration):**

**Step 1 — Plan (decompose task):**
```python
planner = LLM("Customer needs 15 pizzas, 3 delivery times, $200 budget.
               Break this into subtasks with constraints.")
# Output plan:
# Task 1: Optimize pizza sizes (large vs medium) to fit budget
# Task 2: Split toppings (5 vegetarian, 10 meat mix)
# Task 3: Apply bulk discount (10+ pizzas = 15% off)
# Task 4: Check promo code CATERING20 (20% off catering orders)
# Task 5: Schedule deliveries (12pm, 1pm, 2pm)
# Constraint: Total ≤ $200
```

**Step 2 — Execute (workers process subtasks):**
```python
# Worker 1: Optimize sizes
worker_1 = LLM("15 pizzas, $200 budget. Try: 15 large ($224.85), 15 medium ($179.85).
                Which fits?")
# Output: "15 medium pizzas = $179.85 (under budget by $20.15)"

# Worker 2: Split toppings
worker_2 = LLM("15 medium pizzas: 5 vegetarian (margherita), 10 meat (pepperoni,
                sausage, supreme mix). Pricing?")
# Output: "5 margherita ($11.99 each) + 10 meat ($12.49 avg) = $184.90"

# Worker 3: Apply discounts
worker_3 = LLM("10+ pizzas = 15% bulk discount. Also check CATERING20 promo (20% off).
                Which saves more?")
# Output: "Bulk discount: $184.90 × 0.85 = $157.17.
#          Promo CATERING20: $184.90 × 0.80 = $147.92.
#          Promo wins (saves $9.25 more)."

# Worker 4: Schedule deliveries
worker_4 = LLM("15 pizzas, 3 delivery windows (12pm, 1pm, 2pm). Split 5-5-5.
                Delivery fee?")
# Output: "5 pizzas/delivery × $3 fee × 3 deliveries = $9 total delivery."
```

**Step 3 — Verify (check constraints):**
```python
verifier = LLM(f"Plan: {plan}
                Results: {worker_1} + {worker_2} + {worker_3} + {worker_4}
                Total cost: $147.92 (pizzas) + $9 (delivery) = $156.92
                Budget: $200
                Constraint satisfied? Any conflicts?")
# Output: "✅ Total $156.92 < $200 budget (under by $43.08).
#          ✅ All 15 pizzas accounted for (5 veggie, 10 meat).
#          ✅ 3 delivery windows scheduled.
#          ✅ CATERING20 promo applied (saves $37).
#          No conflicts detected."
```

**Final response:** ✅ "I can do 15 medium pizzas (5 vegetarian, 10 meat) with the CATERING20 promo for **$156.92 total** (including delivery). That's $43 under budget! Deliveries at 12pm, 1pm, 2pm. Confirm?"

**Customer response:** ✅ "Perfect, let's do it!" (Complex order success: 85% → 96%)

**Cost:** 1 planner (1,200 tokens) + 4 workers (800 tokens each) + 1 verifier (950 tokens) = **5,350 tokens** ($0.50). **Worth it:** Captures $157 order that v1.0 would have lost (31,400% ROI).

---

## 3 · Pattern #1 — Reflection (Self-Critique)

### What It Is

**Reflection** is a three-step loop:
1. **Generate** an initial draft response
2. **Critique** the draft (self-assessment: is it correct? complete? helpful?)
3. **Revise** the draft based on critique

```
┌─────────────────────────────────────────────────────────────┐
│  REFLECTION LOOP                                            │
│                                                             │
│  User Input                                                 │
│      ↓                                                      │
│  ┌─────────────────┐                                       │
│  │  1. GENERATE    │  draft = LLM(prompt)                  │
│  │     (Draft)     │  "dairy-free + extra cheese = error"  │
│  └────────┬────────┘                                       │
│           ↓                                                 │
│  ┌─────────────────┐                                       │
│  │  2. CRITIQUE    │  critique = LLM(draft)                │
│  │  (Self-Assess)  │  "Unhelpful. Check vegan cheese."     │
│  └────────┬────────┘                                       │
│           ↓                                                 │
│  ┌─────────────────┐                                       │
│  │  3. REVISE      │  revised = LLM(draft + critique)      │
│  │  (Improve)      │  "Vegan mozzarella available +$2.50"  │
│  └────────┬────────┘                                       │
│           ↓                                                 │
│  Final Response                                             │
└─────────────────────────────────────────────────────────────┘
```

**Token cost:** 3× single-pass (850 → 2,550 tokens for PizzaBot)
**Latency:** 3 LLM calls (serial) ≈ 12s vs. 4s single-pass
**Error reduction:** 6× on ambiguous inputs (92% → 98.5% accuracy)

---

### When to Use Reflection

✅ **Use reflection when:**
- Input is ambiguous ("dairy-free + extra cheese")
- Output must be self-consistent (legal contracts, medical reports)
- High cost of error (financial transactions, safety-critical systems)
- You have 3× token budget and 3× latency budget

❌ **Don't use reflection when:**
- Input is unambiguous ("large pepperoni pizza")
- Speed matters more than accuracy (chatbot small talk)
- Single-pass accuracy is already >99% (simple classification tasks)

---

### How Reflection Works — Step by Step

**Example:** Customer orders "gluten-free + dairy-free + extra cheese"

#### Step 1: Generate Draft

```python
def generate_draft(user_input):
    """Generate initial response without reflection."""
    prompt = f"""
    Customer order: {user_input}

    Menu context (from RAG):
    - Gluten-free crust: available (+$2.00)
    - Dairy-free option: use vegan mozzarella
    - Extra cheese: +$2.50 (dairy mozzarella)

    Generate order confirmation or error message.
    """
    draft = LLM(prompt)
    return draft

# Input: "gluten-free + dairy-free + extra cheese"
draft = generate_draft(user_input)
# Output: "Error: dairy-free and extra cheese (dairy) are incompatible.
#          Please remove one option."
```

**Problem:** Draft is technically correct but misses the user's intent (vegan cheese).

---

#### Step 2: Critique Draft

```python
def critique_draft(draft, user_input):
    """Self-assess draft quality and suggest improvements."""
    critique_prompt = f"""
    You are a quality assurance agent. Assess this draft response:

    Draft: "{draft}"

    Original user input: "{user_input}"

    Evaluate:
    1. Is the response technically correct?
    2. Is it helpful to the customer?
    3. Are there alternative interpretations we missed?
    4. What would improve this response?

    Provide critique.
    """
    critique = LLM(critique_prompt)
    return critique

critique = critique_draft(draft, user_input)
# Output: "The draft is technically correct (dairy cheese conflicts with dairy-free).
#          However, it's unhelpful. The customer likely wants vegan cheese.
#          Improvement: Check if vegan mozzarella is available and offer it as
#          'extra vegan cheese' (+$2.50)."
```

---

#### Step 3: Revise Based on Critique

```python
def revise_draft(draft, critique, user_input):
    """Incorporate critique to improve draft."""
    revision_prompt = f"""
    Original draft: "{draft}"
    Critique: "{critique}"
    User input: "{user_input}"

    Tools available:
    - menu_api.check_vegan_cheese() → returns availability and price

    Generate improved response incorporating the critique.
    Use tools if needed.
    """
    # Tool call: menu_api.check_vegan_cheese()
    # Returns: {"available": true, "price": 2.50}

    revised = LLM(revision_prompt)
    return revised

revised = revise_draft(draft, critique, user_input)
# Output: "For dairy-free, we use vegan mozzarella (no dairy).
#          Would you like extra vegan mozzarella (+$2.50) on your
#          gluten-free pepperoni pizza? Total: $19.49."
```

**Customer response:** ✅ "Yes, perfect!"

---

### The Mental Model: Trade Tokens for Reliability

| Metric | Single-pass | Reflection | Improvement |
|--------|-------------|------------|-------------|
| **Contradictory orders** | 92% accuracy | 98.5% accuracy | +6.5% |
| **Tokens** | 850 | 2,550 | 3× cost |
| **Latency** | 4s | 12s | 3× slower |
| **Cost** | $0.08 | $0.24 | 3× cost |
| **Customer abandonment** | 8% | 1.5% | -6.5% |

**ROI calculation:**
- Extra cost: $0.16 per reflection
- Orders saved: 6.5% of 100 orders/day = 6.5 orders
- Revenue per order: $25 average
- Revenue saved: 6.5 × $25 = $162.50/day
- Extra cost: 6.5 × $0.16 = $1.04/day
- **Net gain: $161.46/day** (15,621% ROI)

---

### What Can Go Wrong: Reflection Pitfalls

#### Trap #1: Model Hallucinates That Its Hallucination Is Correct

```python
# Bad critique example
draft = "Your gluten-free pizza includes sourdough crust (naturally gluten-free)."
# ❌ WRONG! Sourdough contains gluten.

critique = critique_draft(draft, user_input)
# Output: "The response is correct. Sourdough is naturally gluten-free due to
#          fermentation breaking down gluten proteins."
# ❌ WRONG AGAIN! Model doubled down on hallucination.
```

**Fix:** Ground critique in external facts (RAG lookup, API call).

```python
def critique_with_grounding(draft, user_input):
    """Critique draft with external fact-checking."""
    # Step 1: Extract factual claims from draft
    claims = extract_claims(draft)
    # ["sourdough crust is gluten-free"]

    # Step 2: Verify each claim with RAG or API
    for claim in claims:
        fact_check = menu_api.verify_claim(claim)
        # Returns: {"claim": "sourdough is gluten-free", "verdict": "FALSE"}

        if fact_check["verdict"] == "FALSE":
            return f"Draft contains false claim: '{claim}'. Correct it."

    return "No factual errors detected."
```

---

#### Trap #2: Over-Iteration (Refinement Loop Never Converges)

```python
# Iteration 1
draft_1 = "Your total is $18.99."
critique_1 = "Consider mentioning delivery time."

# Iteration 2
draft_2 = "Your total is $18.99. Delivery in 35 minutes."
critique_2 = "Consider mentioning delivery fee."

# Iteration 3
draft_3 = "Your total is $21.99 (includes $3 delivery). Delivery in 35 minutes."
critique_3 = "Consider mentioning order tracking link."

# ... 10 iterations later ...
```

**Fix:** Set stopping criteria.

```python
def should_stop_iterating(draft, critique, iteration):
    """Decide if reflection loop should terminate."""
    # Stop after N iterations
    if iteration >= 3:
        return True

    # Stop if critique says "good enough"
    if any(phrase in critique.lower() for phrase in
           ["no improvements", "acceptable", "sufficient"]):
        return True

    # Stop if draft hasn't changed (convergence)
    if iteration > 0 and draft == previous_draft:
        return True

    return False
```

---

#### Trap #3: Reflection on Non-Ambiguous Inputs (Waste)

```python
# Simple, unambiguous input
user_input = "Large pepperoni pizza for delivery."

# Single-pass would work fine:
response = LLM(user_input)
# "Large pepperoni ($14.99) + delivery ($3.00) = $17.99. Confirm?"

# Reflection adds no value:
draft = generate_draft(user_input)     # Same response
critique = critique_draft(draft)       # "No improvements needed"
revised = revise_draft(draft, critique)  # Same response
# ❌ Wasted 2× extra LLM calls for identical output
```

**Fix:** Route simple inputs to single-pass, complex to reflection.

```python
def classify_input_complexity(user_input):
    """Decide if input needs reflection."""
    complexity_signals = [
        "contradictory keywords" if any(pair in user_input.lower() for pair in
            [("dairy-free", "cheese"), ("gluten-free", "extra cheese")]) else None,
        "multiple discounts" if user_input.lower().count("discount") +
            user_input.lower().count("coupon") + user_input.lower().count("promo") >= 2 else None,
        "complex constraints" if len(user_input.split()) > 30 else None,
    ]

    if any(complexity_signals):
        return "REFLECTION"  # Use 3-step loop
    else:
        return "SINGLE_PASS"  # Skip reflection

# Route inputs
if classify_input_complexity(user_input) == "REFLECTION":
    response = reflection_agent(user_input)
else:
    response = single_pass_agent(user_input)
```

---

### 💡 Insight: When Reflection Fails, Escalate to Debate

Reflection assumes the model can self-correct. But what if the model consistently critiques incorrectly (as in Trap #1)?

**Solution:** Escalate to multi-agent debate (§4) where adversarial agents challenge each other.

```python
def smart_routing(user_input):
    """Route to single-pass, reflection, or debate based on complexity."""
    complexity = classify_input_complexity(user_input)

    if complexity == "SIMPLE":
        return single_pass_agent(user_input)

    elif complexity == "AMBIGUOUS":
        # Try reflection first
        response, history = reflection_agent(user_input)

        # If reflection converged quickly, use it
        if len(history) <= 2:
            return response

        # If reflection didn't converge, escalate to debate
        else:
            return debate_agent(user_input)

    elif complexity == "HIGH_STAKES":
        # Skip reflection, go straight to debate (pricing disputes, medical)
        return debate_agent(user_input)
```

---

## 4 · Pattern #2 — Debate & Consensus (Multi-Agent Reasoning)

### What It Is

**Debate** is a multi-agent loop:
1. **Propose**: Multiple agents independently generate solutions
2. **Challenge**: Agents critique each other's proposals
3. **Defend**: Agents justify their proposals against challenges
4. **Vote**: An arbiter (judge) selects the best proposal

```
┌────────────────────────────────────────────────────────────────────┐
│  DEBATE & CONSENSUS LOOP                                           │
│                                                                    │
│  User Input                                                        │
│      ↓                                                             │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  1. PROPOSE (parallel)                                   │    │
│  │                                                           │    │
│  │  Agent 1 (Generous)    Agent 2 (Strict)    Agent 3 (...)│    │
│  │  "Apply all discounts"  "Apply promo only"  ...          │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  2. CHALLENGE (adversarial)                              │    │
│  │                                                           │    │
│  │  Agent 1 → Agent 2: "Did you check if coupon is cheaper?"│    │
│  │  Agent 2 → Agent 1: "Policy says promo replaces coupon"  │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  3. DEFEND (justify)                                     │    │
│  │                                                           │    │
│  │  Agent 1: "Coupon saves $3.51 more (policy: pick best)"  │    │
│  │  Agent 2: "I didn't compare. Agent 1 is correct."        │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     ↓                                              │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  4. VOTE (arbiter selects best)                          │    │
│  │                                                           │    │
│  │  Judge: "Agent 1's proposal ($18.98) is correct."        │    │
│  └──────────────────┬───────────────────────────────────────┘    │
│                     ↓                                              │
│  Final Response                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Token cost:** N agents × rounds × tokens (typically 3 agents × 2 rounds = 6× baseline)
**Latency:** 2 sequential rounds (propose → challenge+vote) ≈ 18s
**Error reduction:** 8× on high-stakes tasks, 40% better than single-agent on complex reasoning

---

### When to Use Debate

✅ **Use debate when:**
- Decision is high-stakes (medical diagnosis, legal reasoning, fraud detection)
- Multiple valid interpretations exist (policy ambiguity, subjective decisions)
- Single-agent reflection didn't converge (agent keeps changing answer)
- You need adversarial validation (avoid groupthink)

❌ **Don't use debate when:**
- Decision is objective with one correct answer ("2 + 2 = ?")
- Speed matters more than accuracy
- Token budget is tight (6× cost vs. single-pass)

---

### How Debate Works — Step by Step

**Example:** Pricing conflict (coupon + loyalty + promo)

#### Step 1: Propose (Parallel Agent Generation)

```python
def debate_propose(user_input, num_agents=3):
    """Multiple agents independently generate proposals."""
    proposals = []

    # Agent 1: Generous interpretation
    agent_1_prompt = f"""
    You are a customer-friendly pricing agent. Given this order:
    {user_input}

    Customer has: 20% coupon, $5 loyalty, BOGO promo.
    Policy: Coupons + loyalty stack. Promos replace coupons + loyalty if better.

    Calculate the LOWEST possible price (most customer-friendly interpretation).
    """
    proposal_1 = LLM(agent_1_prompt)
    proposals.append(("Agent 1 (Generous)", proposal_1))

    # Agent 2: Strict interpretation
    agent_2_prompt = f"""
    You are a policy-strict pricing agent. Given this order:
    {user_input}

    Customer has: 20% coupon, $5 loyalty, BOGO promo.
    Policy: Only one promotional discount applies.

    Calculate the price (strict policy interpretation).
    """
    proposal_2 = LLM(agent_2_prompt)
    proposals.append(("Agent 2 (Strict)", proposal_2))

    # Agent 3: Neutral (optional - can add more agents)
    # proposals.append(("Agent 3 (Neutral)", ...))

    return proposals

proposals = debate_propose(user_input)
# Agent 1: "$18.98 (coupon + loyalty stacked, no promo)"
# Agent 2: "$22.49 (BOGO promo applied, coupon + loyalty ignored)"
```

---

#### Step 2: Challenge (Adversarial Critique)

```python
def debate_challenge(proposals):
    """Each agent challenges the others' proposals."""
    challenges = []

    for i, (agent_i, proposal_i) in enumerate(proposals):
        for j, (agent_j, proposal_j) in enumerate(proposals):
            if i == j:
                continue  # Don't challenge self

            challenge_prompt = f"""
            You are {agent_i}. Your proposal: {proposal_i}.
            {agent_j} proposed: {proposal_j}.

            Critique their proposal: Did they miss anything? Misinterpret policy?
            """
            challenge = LLM(challenge_prompt)
            challenges.append((agent_i, agent_j, challenge))

    return challenges

challenges = debate_challenge(proposals)
# Agent 1 → Agent 2: "You applied BOGO ($22.49) but didn't check if
#                     coupon + loyalty ($18.98) is cheaper. Policy says
#                     'pick best discount'."
# Agent 2 → Agent 1: "You stacked coupon + loyalty, but policy says
#                     promotional discounts replace coupons. BOGO should apply."
```

---

#### Step 3: Defend (Justify Against Challenges)

```python
def debate_defend(proposals, challenges):
    """Agents defend their proposals against challenges."""
    defenses = []

    for agent_name, proposal in proposals:
        # Gather all challenges directed at this agent
        agent_challenges = [c for c in challenges if c[1] == agent_name]

        defense_prompt = f"""
        You are {agent_name}. Your proposal: {proposal}.

        Challenges against your proposal:
        {'\n'.join([f"{c[0]}: {c[2]}" for c in agent_challenges])}

        Defend your proposal or concede if the challenge is valid.
        """
        defense = LLM(defense_prompt)
        defenses.append((agent_name, defense))

    return defenses

defenses = debate_defend(proposals, challenges)
# Agent 1: "My proposal is correct. Policy says 'pick best discount'.
#           Coupon + loyalty ($18.98) < BOGO ($22.49), so coupon + loyalty applies."
# Agent 2: "I concede. I didn't compare both paths. Agent 1 is correct.
#           Customer should pay $18.98."
```

---

#### Step 4: Vote (Arbiter Selects Best)

```python
def debate_vote(proposals, challenges, defenses):
    """Judge evaluates proposals and selects best."""
    judge_prompt = f"""
    You are a neutral judge. Review these pricing proposals:

    {proposals}

    Challenges:
    {challenges}

    Defenses:
    {defenses}

    Which proposal is correct according to policy?
    Policy: Coupons + loyalty stack. Promotional discounts replace both
            unless coupon + loyalty is cheaper.

    Select the best proposal and explain why.
    """
    verdict = LLM(judge_prompt)
    return verdict

verdict = debate_vote(proposals, challenges, defenses)
# Output: "Agent 1's proposal ($18.98) is correct.
#          Coupon + loyalty stacked = $18.98.
#          BOGO promo = $22.49.
#          Policy: 'Pick best discount' → $18.98 wins.
#          Customer should pay $18.98."
```

---

### What Can Go Wrong: Debate Pitfalls

#### Trap #1: Groupthink (All Agents Agree on Wrong Answer)

```python
# Bad debate example
proposals = [
    ("Agent 1", "$22.49 (BOGO promo applied)"),
    ("Agent 2", "$22.49 (BOGO promo applied)"),
]
# Both agents picked BOGO without comparing to coupon + loyalty.

challenges = []
# No challenges (both agree).

verdict = debate_vote(proposals, challenges, [])
# Output: "$22.49 (BOGO promo applied)."
# ❌ WRONG! Should be $18.98 (coupon + loyalty is cheaper).
```

**Why this happens:** Agents have similar biases (all trained on same data), no adversarial diversity.

**Fix:** Assign explicit adversarial roles.

```python
def debate_propose_adversarial(user_input):
    """Agents assigned adversarial perspectives."""
    # Agent 1: Maximize customer savings
    agent_1_prompt = "Calculate LOWEST price (customer-friendly)."

    # Agent 2: Maximize revenue (business-friendly)
    agent_2_prompt = "Calculate HIGHEST valid price (business-friendly)."

    # Agent 3: Neutral (policy-strict)
    agent_3_prompt = "Calculate price (strict policy interpretation)."

    # Now agents are forced to disagree
```

---

#### Trap #2: Debate Deadlock (Agents Never Converge)

```python
# Round 1
Agent 1: "$18.98"
Agent 2: "$22.49"

# Round 2
Agent 1: "Still $18.98 (policy says pick best)"
Agent 2: "Still $22.49 (promo replaces coupon)"

# Round 3
Agent 1: "Still $18.98"
Agent 2: "Still $22.49"

# ... infinite loop
```

**Fix:** Add stopping criteria and tie-breaking.

```python
def should_stop_debate(defenses, round_num, max_rounds=2):
    """Decide if debate should terminate."""
    # Stop after max rounds
    if round_num >= max_rounds:
        return True

    # Stop if all agents converged (all defenses identical)
    if len(set(d[1] for d in defenses)) == 1:
        return True

    return False

def break_tie(proposals):
    """If agents don't converge, use tie-breaking rule."""
    # Rule: Pick most customer-friendly (lowest price)
    prices = extract_prices(proposals)
    return min(prices)
```

---

#### Trap #3: Over-Debating Simple Questions

```python
# Simple, objective question
user_input = "What's the price of a large pepperoni pizza?"

# Debate is overkill:
proposals = [
    ("Agent 1", "$14.99"),
    ("Agent 2", "$14.99"),
]
challenges = []  # No disagreement
verdict = "$14.99"

# ❌ Wasted 6× tokens for identical output
```

**Fix:** Route simple queries to single-pass, high-stakes to debate.

```python
def classify_query_stakes(user_input):
    """Decide if query needs debate."""
    high_stakes_signals = [
        "multiple discounts",
        "policy ambiguity",
        "medical diagnosis",
        "legal question",
        "fraud detection",
    ]

    if any(signal in user_input.lower() for signal in high_stakes_signals):
        return "DEBATE"
    else:
        return "SINGLE_PASS"
```

---

### 💡 Insight: Debate vs. Reflection

| Pattern | When to use | Agents | Cost | Best for |
|---------|-------------|--------|------|----------|
| **Reflection** | Ambiguous inputs | 1 (self-critique) | 3× | Contradictions, self-consistency |
| **Debate** | High-stakes decisions | 2–3 (adversarial) | 6× | Policy disputes, subjective judgments |

**Escalation path:** Single-pass → Reflection → Debate
- Try single-pass first (cheapest)
- If single-pass fails, try reflection (3× cost)
- If reflection doesn't converge, escalate to debate (6× cost)

---

## 5 · Pattern #3 — Hierarchical Orchestration (Planner → Workers → Verifier)

### What It Is

**Hierarchical orchestration** is a three-layer pattern:
1. **Planner**: Decomposes complex task into subtasks
2. **Workers**: Execute subtasks in parallel (or sequence)
3. **Verifier**: Checks results against original plan and constraints

```
┌──────────────────────────────────────────────────────────────────┐
│  HIERARCHICAL ORCHESTRATION                                      │
│                                                                  │
│  Complex Task                                                    │
│      ↓                                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. PLANNER (decompose)                                 │   │
│  │                                                          │   │
│  │  Plan: [Task1, Task2, Task3, Task4]                     │   │
│  │  Constraints: [Budget ≤ $200, 3 delivery times, ...]    │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     ↓                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2. WORKERS (execute in parallel)                       │   │
│  │                                                          │   │
│  │  Worker1: Optimize sizes    Worker2: Split toppings     │   │
│  │  Worker3: Apply discounts   Worker4: Schedule delivery  │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     ↓                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3. VERIFIER (validate)                                 │   │
│  │                                                          │   │
│  │  Check: Budget OK? All pizzas accounted? Conflicts?     │   │
│  │  ✅ All constraints satisfied.                           │   │
│  └──────────────────┬──────────────────────────────────────┘   │
│                     ↓                                            │
│  Final Response                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Token cost:** 1 planner + N workers + 1 verifier (typically 1 + 4 + 1 = 6× baseline)
**Latency:** 3 sequential steps (plan → workers parallel → verify) ≈ 15s
**Success rate:** 15× better on multi-step tasks (85% → 96%)

---

### When to Use Hierarchical Orchestration

✅ **Use hierarchical when:**
- Task is decomposable (catering orders, research tasks, data pipelines)
- Subtasks can run in parallel (faster than sequential)
- Need to enforce global constraints (budget, deadlines, dependencies)
- Workers might drift from plan (need verification)

❌ **Don't use hierarchical when:**
- Task is atomic (single-step, no decomposition)
- Sequential execution is required (each step depends on previous)
- Overhead of planner + verifier > benefit of parallelization

---

### How Hierarchical Orchestration Works — Step by Step

**Example:** Catering order (15 pizzas, 3 delivery times, $200 budget)

#### Step 1: Planner (Decompose Task)

```python
def hierarchical_plan(user_input):
    """Decompose complex task into subtasks with constraints."""
    planner_prompt = f"""
    Customer request: {user_input}

    Task: Break this into subtasks. For each subtask:
    1. What needs to be done?
    2. What are the constraints?
    3. What information does the worker need?

    Provide a structured plan.
    """
    plan = LLM(planner_prompt)
    return plan

user_input = """
I need 15 pizzas for a company event. 5 delivered at 12pm, 5 at 1pm, 5 at 2pm.
Budget is $200 total. Mix of vegetarian and meat options.
"""

plan = hierarchical_plan(user_input)
# Output:
# Task 1: Optimize pizza sizes (large vs medium) to fit $200 budget
#    - Constraint: Total cost ≤ $200 (including delivery)
#    - Input: 15 pizzas, menu pricing
#
# Task 2: Split toppings (vegetarian vs meat)
#    - Constraint: "Mix" means at least 20% vegetarian (3+ pizzas)
#    - Input: Customer preference ("mix of vegetarian and meat")
#
# Task 3: Apply bulk discounts or promo codes
#    - Constraint: Pick discount that saves most
#    - Input: 10+ pizzas = 15% bulk discount, CATERING20 = 20% off
#
# Task 4: Schedule deliveries
#    - Constraint: 3 delivery windows (12pm, 1pm, 2pm), 5 pizzas each
#    - Input: Delivery fee ($3/delivery), kitchen capacity
#
# Global constraint: Total cost ≤ $200
```

---

#### Step 2: Workers (Execute Subtasks in Parallel)

```python
def hierarchical_execute(plan):
    """Workers execute subtasks in parallel."""
    workers = []

    # Worker 1: Optimize sizes
    worker_1_prompt = f"""
    Subtask: {plan['Task 1']}

    Menu:
    - Large pizza: $14.99
    - Medium pizza: $11.99

    Calculate: 15 large vs 15 medium. Which fits $200 budget?
    """
    result_1 = LLM(worker_1_prompt)
    workers.append(("Worker 1", result_1))

    # Worker 2: Split toppings
    worker_2_prompt = f"""
    Subtask: {plan['Task 2']}

    Customer said "mix of vegetarian and meat". Split 15 pizzas.
    Constraint: At least 3 vegetarian (20%).
    """
    result_2 = LLM(worker_2_prompt)
    workers.append(("Worker 2", result_2))

    # Worker 3: Apply discounts
    worker_3_prompt = f"""
    Subtask: {plan['Task 3']}

    Options:
    - Bulk discount (10+ pizzas): 15% off
    - CATERING20 promo: 20% off

    Which saves more?
    """
    result_3 = LLM(worker_3_prompt)
    workers.append(("Worker 3", result_3))

    # Worker 4: Schedule deliveries
    worker_4_prompt = f"""
    Subtask: {plan['Task 4']}

    15 pizzas, 3 delivery windows (12pm, 1pm, 2pm).
    Split 5-5-5. Delivery fee: $3/delivery.
    """
    result_4 = LLM(worker_4_prompt)
    workers.append(("Worker 4", result_4))

    return workers

# Execute in parallel
workers = hierarchical_execute(plan)
# Worker 1: "15 medium pizzas = $179.85 (under budget by $20.15)"
# Worker 2: "5 vegetarian (margherita), 10 meat (pepperoni, sausage)"
# Worker 3: "CATERING20 (20% off) saves more than bulk discount (15% off)"
# Worker 4: "3 deliveries × $3 = $9 total delivery fee"
```

---

#### Step 3: Verifier (Validate Results)

```python
def hierarchical_verify(plan, workers):
    """Validate worker results against plan and constraints."""
    verifier_prompt = f"""
    Original plan: {plan}

    Worker results:
    {workers}

    Verify:
    1. Are all subtasks completed?
    2. Do results satisfy constraints?
    3. Are there any conflicts or missing pieces?

    Calculate final cost and check against $200 budget.
    """
    verification = LLM(verifier_prompt)
    return verification

verification = hierarchical_verify(plan, workers)
# Output:
# ✅ Task 1: 15 medium pizzas = $179.85
# ✅ Task 2: 5 vegetarian + 10 meat (20% vegetarian minimum satisfied)
# ✅ Task 3: CATERING20 applied → $179.85 × 0.80 = $143.88
# ✅ Task 4: 3 deliveries × $3 = $9
# ✅ Total: $143.88 + $9 = $152.88
# ✅ Budget: $152.88 < $200 ✅
# ✅ All constraints satisfied. No conflicts.
```

---

### What Can Go Wrong: Hierarchical Pitfalls

#### Trap #1: Workers Drift from Plan

```python
# Planner says:
plan = "Optimize sizes to fit $200 budget. Choose large OR medium, not both."

# Worker 1 drifts:
worker_1_result = "I chose 10 large pizzas ($149.90) and 5 medium ($59.95) = $209.85."
# ❌ WRONG! Exceeded budget. Worker ignored "choose large OR medium" constraint.
```

**Why this happens:** Workers don't see full context (only their subtask).

**Fix:** Include constraints in every worker prompt.

```python
def worker_execute_with_context(subtask, plan):
    """Execute subtask with full context."""
    worker_prompt = f"""
    Full plan: {plan}

    Your subtask: {subtask}

    Global constraints:
    - Budget ≤ $200
    - 15 pizzas total (no more, no less)
    - 3 delivery windows (12pm, 1pm, 2pm)

    Execute your subtask WITHOUT violating global constraints.
    """
    result = LLM(worker_prompt)
    return result
```

---

#### Trap #2: Verifier Misses Conflicts

```python
# Worker 1: "15 medium pizzas = $179.85"
# Worker 2: "5 vegetarian, 10 meat (15 pizzas total)"
# Worker 3: "CATERING20 applied → $143.88"
# Worker 4: "3 deliveries × $5 = $15" (WRONG! Fee is $3, not $5)

# Verifier fails to catch:
verification = LLM("Check if all constraints satisfied")
# Output: "✅ All constraints satisfied."
# ❌ WRONG! Delivery fee is $15, not $9. Total = $158.88, not $152.88.
```

**Fix:** Verifier must re-compute critical values.

```python
def hierarchical_verify_with_recompute(plan, workers):
    """Verifier re-computes to catch errors."""
    # Extract claims from worker results
    claims = {
        "pizza_cost": extract_value(workers, "Worker 1", "cost"),
        "delivery_fee": extract_value(workers, "Worker 4", "delivery fee"),
        "discount_rate": extract_value(workers, "Worker 3", "discount"),
    }

    # Re-compute total
    pizza_cost = 15 * 11.99  # 15 medium pizzas
    discount = 0.20 if "CATERING20" in workers else 0.15
    discounted_cost = pizza_cost * (1 - discount)
    delivery_fee = 3 * 3  # 3 deliveries × $3
    total = discounted_cost + delivery_fee

    # Compare to worker claims
    if abs(total - claims["pizza_cost"] - claims["delivery_fee"]) > 0.01:
        return f"❌ Verification failed. Re-computed total: ${total:.2f}"

    return f"✅ Verification passed. Total: ${total:.2f}"
```

---

#### Trap #3: Over-Decomposition (Too Many Workers)

```python
# Overly granular decomposition
plan = {
    "Task 1": "Check if large pizzas fit budget",
    "Task 2": "Check if medium pizzas fit budget",
    "Task 3": "Compare large vs medium",
    "Task 4": "Pick cheaper option",
    "Task 5": "Count vegetarian pizzas",
    "Task 6": "Count meat pizzas",
    "Task 7": "Check if vegetarian count ≥ 20%",
    # ... 15 subtasks total
}
# ❌ Overhead: 1 planner + 15 workers + 1 verifier = 17 LLM calls
# Could have been done in 1 planner + 4 workers + 1 verifier = 6 calls
```

**Fix:** Batch related subtasks.

```python
def hierarchical_plan_smart(user_input):
    """Planner batches related subtasks."""
    planner_prompt = f"""
    Customer request: {user_input}

    Break into 3-5 high-level subtasks (not 15 micro-tasks).
    Each subtask should be independently executable.
    Batch related operations (e.g., "Optimize sizes" includes large vs medium comparison).
    """
    plan = LLM(planner_prompt)
    return plan
```

---

### 💡 Insight: Hierarchical vs. Sequential Chain

| Pattern | Structure | When to use | Parallelization |
|---------|-----------|-------------|-----------------|
| **Sequential Chain** (Ch.6 ReAct) | Step1 → Step2 → Step3 | Dependencies between steps | No (sequential only) |
| **Hierarchical** (Ch.11) | Planner → [Workers parallel] → Verifier | Independent subtasks | Yes (workers parallel) |

**Example:**
- **Sequential**: "Book flight → Book hotel → Book rental car" (each depends on previous)
- **Hierarchical**: "Plan trip → [Book flight ‖ Book hotel ‖ Book car] → Verify itinerary" (subtasks parallel)

---

## 6 · Pattern #4 — Tool Selection Strategies

### What It Is

**Tool selection** is a decision pattern for choosing which tool to use when multiple options exist:
1. **Rule-based**: If-then rules (e.g., "if inventory check, try cache first")
2. **Cost-based**: Try cheapest tool first, escalate on failure
3. **LLM-based meta-agent**: Let LLM decide which tool to use

**Fallback chains**: When a tool fails, retry with next-best tool.

```
┌──────────────────────────────────────────────────────────────┐
│  TOOL SELECTION WITH FALLBACK CHAIN                          │
│                                                              │
│  Task: Check inventory for pepperoni                         │
│      ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Try Tool 1: CACHE (fast, 95% accuracy, $0.001)   │    │
│  │  ↓                                                  │    │
│  │  Success? → Return result                          │    │
│  │  Failure? ↓                                         │    │
│  └────────────────────────────────────────────────────┘    │
│      ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Try Tool 2: DATABASE (medium, 99% accuracy, $0.01)│    │
│  │  ↓                                                  │    │
│  │  Success? → Return result                          │    │
│  │  Failure? ↓                                         │    │
│  └────────────────────────────────────────────────────┘    │
│      ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Try Tool 3: API (slow, 100% accuracy, $0.05)     │    │
│  │  ↓                                                  │    │
│  │  Success? → Return result                          │    │
│  │  Failure? ↓                                         │    │
│  └────────────────────────────────────────────────────┘    │
│      ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Escalate to HUMAN (fallback of last resort)      │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Token cost:** Variable (1.1× for cache hit, 2.5× for full escalation)
**Success rate:** 95% recovery from tool failures (vs. 60% with no fallback)

---

### When to Use Tool Selection

✅ **Use tool selection when:**
- Multiple tools can solve the task (cache, DB, API)
- Tools have different cost/latency/accuracy trade-offs
- Tools can fail (network errors, timeouts, rate limits)
- Need graceful degradation (fallback to cheaper/slower tool)

❌ **Don't use tool selection when:**
- Only one tool available
- All tools have identical cost/latency/accuracy
- Tool failures are catastrophic (no fallback possible)

---

### Strategy #1: Rule-Based Tool Selection

**Approach:** Hard-coded if-then rules based on task type.

```python
def rule_based_tool_selection(task_type, context):
    """Select tool using if-then rules."""
    if task_type == "inventory_check":
        # Rule: Try fast cache first, fallback to DB
        return ["cache", "database", "api"]

    elif task_type == "pricing_calculation":
        # Rule: Pricing must be accurate → skip cache, go to DB
        return ["database", "api"]

    elif task_type == "menu_lookup":
        # Rule: Menu rarely changes → cache is sufficient
        return ["cache"]

    else:
        # Default: Try all tools in order (cheap → expensive)
        return ["cache", "database", "api", "human"]
```

**Pros:** Fast, deterministic, no LLM call needed
**Cons:** Can't adapt to runtime context (e.g., "cache is stale, skip it")

---

### Strategy #2: Cost-Based Tool Selection (Fallback Chain)

**Approach:** Try cheapest tool first, escalate on failure.

```python
def cost_based_fallback(tools, task):
    """Execute tools in order (cheapest → most expensive) until success."""
    for tool in tools:
        try:
            result = tool.execute(task)
            if result.success:
                return result
        except ToolFailure as e:
            # Log failure and try next tool
            logging.warning(f"{tool.name} failed: {e}. Trying next tool...")
            continue

    # All tools failed → escalate to human
    return escalate_to_human(task)

# Tool definitions
tools = [
    Tool(name="cache", cost=0.001, latency=0.1, accuracy=0.95),
    Tool(name="database", cost=0.01, latency=0.5, accuracy=0.99),
    Tool(name="api", cost=0.05, latency=2.0, accuracy=1.00),
]

# Example: Inventory check
task = "Check inventory for pepperoni"
result = cost_based_fallback(tools, task)

# Execution trace:
# 1. Try cache: ❌ Cache miss (pepperoni not in cache)
# 2. Try database: ✅ Success (pepperoni in stock: 50 units)
# Cost: $0.001 (cache attempt) + $0.01 (DB success) = $0.011
```

**Pros:** Optimizes for cost, automatic fallback
**Cons:** May waste time on cheap-but-failing tools

---

### Strategy #3: LLM-Based Meta-Agent Tool Selection

**Approach:** Let LLM decide which tool to use based on task context.

```python
def llm_based_tool_selection(task, context, available_tools):
    """LLM meta-agent selects best tool for task."""
    tool_descriptions = "\n".join([
        f"- {t.name}: Cost ${t.cost}, Latency {t.latency}s, Accuracy {t.accuracy*100}%"
        for t in available_tools
    ])

    meta_agent_prompt = f"""
    Task: {task}
    Context: {context}

    Available tools:
    {tool_descriptions}

    Which tool should we use? Consider:
    1. Task requirements (accuracy vs speed)
    2. Context (e.g., is cached data stale?)
    3. Cost-benefit trade-off

    Output: Tool name only.
    """
    selected_tool_name = LLM(meta_agent_prompt)
    selected_tool = [t for t in available_tools if t.name == selected_tool_name][0]

    return selected_tool

# Example: Inventory check with context
task = "Check inventory for pepperoni"
context = "Customer just ordered 20 pizzas. Cache was updated 5 minutes ago."

selected_tool = llm_based_tool_selection(task, context, tools)
# LLM output: "cache"
# Reasoning: "Cache is recent (5 min old) and customer needs fast response.
#             Use cache (95% accuracy is acceptable for inventory check)."

result = selected_tool.execute(task)
```

**Pros:** Adapts to context, can reason about trade-offs
**Cons:** Adds LLM call overhead ($0.02), may make suboptimal choices

---

### What Can Go Wrong: Tool Selection Pitfalls

#### Trap #1: Tool Thrashing (Retry Loop with No Convergence)

```python
# Fallback chain with no stopping condition
while True:
    result = try_cache()
    if result.success:
        return result

    result = try_database()
    if result.success:
        return result

    result = try_api()
    if result.success:
        return result

    # Loop forever if all tools fail
```

**Fix:** Add max retry limit and human escalation.

```python
def tool_selection_with_retry_limit(tools, task, max_retries=3):
    """Fallback chain with retry limit."""
    attempts = 0

    while attempts < max_retries:
        for tool in tools:
            try:
                result = tool.execute(task)
                return result
            except ToolFailure:
                continue

        attempts += 1
        logging.warning(f"All tools failed. Retry {attempts}/{max_retries}")

    # Max retries exceeded → escalate
    return escalate_to_human(task)
```

---

#### Trap #2: Ignoring Tool Cost (Always Use Expensive Tool)

```python
# LLM meta-agent always picks API (most accurate)
selected_tool = llm_based_tool_selection(task, context, tools)
# Output: "api" (100% accuracy, $0.05 cost)

# But cache would have worked (95% accuracy, $0.001 cost)
# ❌ Wasted $0.049 (49× more expensive)
```

**Fix:** Add cost-awareness to LLM prompt.

```python
def llm_based_tool_selection_cost_aware(task, context, tools, budget=0.02):
    """LLM meta-agent with cost constraint."""
    meta_agent_prompt = f"""
    Task: {task}
    Context: {context}
    Budget: ${budget}

    Available tools:
    {tool_descriptions}

    Pick the CHEAPEST tool that satisfies task requirements.
    Only use expensive tools if cheap tools can't meet requirements.
    """
    selected_tool_name = LLM(meta_agent_prompt)
    return selected_tool_name
```

---

#### Trap #3: No Fallback for Critical Tools

```python
# Payment processing has no fallback
def process_payment(card_info):
    result = payment_api.charge(card_info)
    if result.failure:
        return "Payment failed. Order cancelled."
    # ❌ No retry with backup payment processor
```

**Fix:** Always have fallback for critical operations.

```python
def process_payment_with_fallback(card_info):
    """Payment with fallback to backup processor."""
    primary_processors = [stripe_api, square_api, paypal_api]

    for processor in primary_processors:
        try:
            result = processor.charge(card_info)
            if result.success:
                return result
        except PaymentError:
            continue

    # All processors failed → escalate to manual processing
    return escalate_to_manual_payment(card_info)
```

---

## 7 · Mental Model: Trading Tokens for Reliability

Every agentic pattern is a trade-off: **tokens ↔ reliability**.

### The Cost-Reliability Curve

```
Reliability (%)
   100% ┤                                          ● Hierarchical + Debate
        │                                    ● Debate
    99% ┤                              ● Hierarchical
        │                        ● Reflection
    95% ┤                  ● Single-pass (Ch.1-10)
        │            ●
    90% ┤      ●
        │ ●
    85% ┤
        └────────────────────────────────────────────────────────────────
         1×      3×      5×      7×      9×      11×     13×     15×
                           Token Cost (relative to single-pass)
```

### Decision Tree: Which Pattern for Which Problem?

```
┌─ Is task simple & unambiguous?
│   ├─ YES → SINGLE-PASS (1×)
│   └─ NO → Continue ↓
│
├─ Does input have contradictions or ambiguity?
│   ├─ YES → REFLECTION (3×)
│   └─ NO → Continue ↓
│
├─ Is decision high-stakes (medical, legal, fraud)?
│   ├─ YES → DEBATE (6×)
│   └─ NO → Continue ↓
│
├─ Is task decomposable into subtasks?
│   ├─ YES → HIERARCHICAL (5–7×)
│   └─ NO → Continue ↓
│
├─ Do you have multiple tools with different cost/accuracy?
│   ├─ YES → TOOL SELECTION (1.1–2.5×)
│   └─ NO → SINGLE-PASS (1×)
```

### Pattern Comparison Table

| Pattern | When to use | Cost (tokens) | Latency | Error reduction | Example use case |
|---------|-------------|---------------|---------|-----------------|------------------|
| **Single-pass** | Simple, unambiguous inputs | 1× | 4s | Baseline | "Large pepperoni pizza" |
| **Reflection** | Ambiguous, contradictory inputs | 3× | 12s | 6× | "Dairy-free + extra cheese" |
| **Debate** | High-stakes, subjective decisions | 6× | 18s | 8× | Pricing disputes, medical diagnosis |
| **Hierarchical** | Complex, decomposable tasks | 5–7× | 15s | 15× | Catering orders, research tasks |
| **Tool Selection** | Multiple tools, failure-prone | 1.1–2.5× | Variable | 2× recovery | Inventory checks, API calls |

### Cost-Benefit Analysis

**Example:** PizzaBot edge case handling

| Scenario | Pattern | Cost | Revenue saved | ROI |
|----------|---------|------|---------------|-----|
| **Contradictory order** | Reflection | $0.24 | $25 (order saved) | 10,417% |
| **Pricing dispute** | Debate | $0.45 | $25 (refund avoided) | 5,556% |
| **Catering order** | Hierarchical | $0.50 | $157 (order captured) | 31,400% |
| **Inventory failure** | Tool Selection | $0.01 | $0 (no revenue impact, but prevents escalation) | N/A |

**Key insight:** Even expensive patterns (6× tokens) pay for themselves if they prevent order abandonment or disputes.

---

## 8 · What Can Go Wrong: Common Traps Across All Patterns

### Trap #1: Infinite Loops (Reflection, Debate)

**Symptom:** Agent iterates forever without converging.

```python
# Reflection loop
for i in range(100):  # ❌ No stopping condition
    critique = critique(response)
    response = revise(critique)
```

**Fix:** Add max iterations and convergence check.

```python
for i in range(3):  # Max 3 iterations
    critique = critique(response)
    if "no improvements" in critique:
        break  # Converged
    response = revise(critique)
```

---

### Trap #2: Hallucination Cascade (Reflection)

**Symptom:** Model critiques its own hallucination as correct.

```python
draft = "Sourdough crust is gluten-free."  # ❌ WRONG
critique = "Sourdough is naturally gluten-free."  # ❌ WRONG AGAIN
```

**Fix:** Ground critique in external facts (RAG, API).

```python
def critique_with_grounding(draft):
    claims = extract_claims(draft)
    for claim in claims:
        fact = rag.verify(claim)  # Check against knowledge base
        if not fact.verified:
            return f"False claim: {claim}"
```

---

### Trap #3: Groupthink (Debate)

**Symptom:** All agents agree on wrong answer.

```python
agent_1 = "$22.49 (BOGO promo)"
agent_2 = "$22.49 (BOGO promo)"  # No disagreement
```

**Fix:** Assign adversarial roles (generous vs. strict).

```python
agent_1_prompt = "Calculate LOWEST price (customer-friendly)"
agent_2_prompt = "Calculate price (policy-strict)"
```

---

### Trap #4: Worker Drift (Hierarchical)

**Symptom:** Workers ignore planner's constraints.

```python
plan = "Budget ≤ $200"
worker = "$209.85"  # ❌ Exceeded budget
```

**Fix:** Include constraints in every worker prompt.

```python
worker_prompt = f"""
Subtask: {subtask}
Global constraints: Budget ≤ $200
"""
```

---

### Trap #5: Tool Thrashing (Tool Selection)

**Symptom:** Retry loop with no convergence.

```python
while True:  # ❌ Infinite retry
    result = try_cache()
    result = try_db()
    result = try_api()
```

**Fix:** Add max retries and human escalation.

```python
for attempt in range(3):
    for tool in tools:
        result = tool.execute()
        if result.success:
            return result

escalate_to_human()
```

---

### Trap #6: Overusing Expensive Patterns

**Symptom:** Using debate for simple queries.

```python
user_input = "Large pepperoni pizza"
response = debate_agent(user_input)  # ❌ Wasted 6× tokens
```

**Fix:** Route simple queries to single-pass.

```python
if is_simple(user_input):
    return single_pass(user_input)  # 1× cost
else:
    return debate_agent(user_input)  # 6× cost
```

---

## 9 · Progress Check: Can You Pick the Right Pattern?

### Exercise 1: Pattern Selection

For each scenario, pick the best pattern and explain why.

**Scenario A:** Customer orders "Large pepperoni pizza with extra cheese."

<details>
<summary>Click for answer</summary>

**Pattern:** Single-pass (1×)

**Why:** Simple, unambiguous order. No contradictions, no edge cases. Single-pass is sufficient.

</details>

---

**Scenario B:** Customer orders "I'm allergic to dairy but I love cheese. Can you make it work?"

<details>
<summary>Click for answer</summary>

**Pattern:** Reflection (3×)

**Why:** Ambiguous input (allergic to dairy but wants cheese). Reflection can detect contradiction and suggest vegan cheese.

</details>

---

**Scenario C:** Customer has 3 overlapping discounts and asks "Which one should I use?"

<details>
<summary>Click for answer</summary>

**Pattern:** Debate (6×)

**Why:** High-stakes pricing decision. Multiple valid interpretations (stack discounts vs. pick best). Debate ensures correct answer.

</details>

---

**Scenario D:** Corporate client needs 50 pizzas delivered to 5 locations over 2 days, $500 budget.

<details>
<summary>Click for answer</summary>

**Pattern:** Hierarchical (5–7×)

**Why:** Complex, decomposable task (optimize sizes, split toppings, schedule deliveries, apply discounts). Planner → Workers → Verifier is ideal.

</details>

---

**Scenario E:** Inventory check for pepperoni (cache is 10 minutes old, customer is waiting).

<details>
<summary>Click for answer</summary>

**Pattern:** Tool Selection (1.1–2.5×)

**Why:** Multiple tools available (cache, DB, API). Cost-based fallback: try cache first (95% accuracy, fast), escalate to DB if needed.

</details>

---

### Exercise 2: PizzaBot Constraint Table

Show that all 6 constraints are now satisfied.

| Constraint | Ch.10 (single-pass) | Ch.11 (with patterns) | Status |
|------------|---------------------|----------------------|--------|
| **#1 ACCURACY** (99%+ orders handled) | 95% | 99.2% | ✅ |
| **#2 EDGE CASES** (<1% error rate) | 8% | 0.8% | ✅ |
| **#3 COST** (≤$0.25/conversation) | $0.08 | $0.18 avg | ✅ |
| **#4 LATENCY** (<15s for complex orders) | 4.2s | 11.3s avg | ✅ |
| **#5 TRANSPARENCY** (show reasoning) | ❌ No | ✅ Yes (multi-step explanations) | ✅ |
| **#6 GRACEFUL DEGRADATION** (ask clarifying questions) | ❌ No | ✅ Yes (reflection + debate) | ✅ |

---

### Exercise 3: Pattern Capability Bullets

Which capabilities does each pattern unlock?

**Single-pass (Ch.1-10):**
- ✅ Simple orders (1-2 items)
- ✅ Menu lookups (RAG grounding)
- ✅ Tool orchestration (ReAct)
- ❌ Contradictory inputs (no self-correction)
- ❌ Pricing disputes (no multi-agent reasoning)
- ❌ Complex catering (no decomposition)

**Reflection (Ch.11):**
- ✅ Simple orders
- ✅ Menu lookups
- ✅ Tool orchestration
- ✅ Contradictory inputs (self-critique)
- ❌ Pricing disputes (single-agent can't arbitrate)
- ❌ Complex catering (no decomposition)

**Debate (Ch.11):**
- ✅ Simple orders
- ✅ Menu lookups
- ✅ Tool orchestration
- ✅ Contradictory inputs
- ✅ Pricing disputes (multi-agent arbitration)
- ❌ Complex catering (no decomposition)

**Hierarchical (Ch.11):**
- ✅ Simple orders
- ✅ Menu lookups
- ✅ Tool orchestration
- ✅ Contradictory inputs (verifier catches)
- ✅ Pricing disputes (workers handle)
- ✅ Complex catering (planner decomposes)

**All patterns combined:**
- ✅ All 6 constraints satisfied
- ✅ Production-ready PizzaBot v2.0

---

## 10 · Bridge to Multi-Agent AI Track

**Where you are now:** You've learned four single-agent iterative patterns (reflection, debate, hierarchical, tool selection). These patterns trade tokens for reliability by adding loops, adversarial checks, and decomposition.

**What's next:** The **Multi-Agent AI track** (`notes/04-multi_agent_ai/`) extends these patterns to persistent multi-agent systems:

### Pattern Extensions: Single-Agent → Multi-Agent

| Ch.11 Pattern | Multi-Agent Extension | Track Chapter |
|---------------|----------------------|---------------|
| **Reflection** | Cross-agent critique (Agent A generates, Agent B critiques) | Multi-Agent Ch.3 |
| **Debate** | Persistent debate teams (agents specialize in roles: proposer, challenger, judge) | Multi-Agent Ch.4 |
| **Hierarchical** | Persistent orchestration (manager agents coordinate worker agents across tasks) | Multi-Agent Ch.5 |
| **Tool Selection** | Tool specialization (each agent owns a subset of tools, agents negotiate tool usage) | Multi-Agent Ch.6 |

### Key Differences: Single-Agent Patterns vs. Multi-Agent Systems

| Aspect | Ch.11 (Single-Agent Patterns) | Multi-Agent Track |
|--------|------------------------------|-------------------|
| **Agent persistence** | Agents are ephemeral (created per-conversation) | Agents are persistent (live across conversations) |
| **State management** | State lives in prompt context | State lives in agent memory (short-term + long-term) |
| **Communication** | Sequential (one agent finishes, next starts) | Concurrent (agents message each other asynchronously) |
| **Coordination** | Hard-coded (plan → execute → verify) | Emergent (agents negotiate, vote, escalate) |
| **Scaling** | 2-4 agents max (token overhead) | 10s-100s of agents (distributed systems) |

### Example: Reflection → Cross-Agent Critique

**Ch.11 Reflection (single-agent):**
```python
draft = Agent1("Generate response")
critique = Agent1("Critique your own draft")  # Same agent
revised = Agent1("Revise based on critique")
```

**Multi-Agent Ch.3 (cross-agent critique):**
```python
draft = AgentA("Generate response")
critique = AgentB("Critique AgentA's draft")  # Different agent (adversarial)
revised = AgentA("Revise based on AgentB's critique")
# AgentB persists, learns from past critiques, specializes in finding flaws
```

### Example: Hierarchical → Persistent Orchestration

**Ch.11 Hierarchical (ephemeral):**
```python
plan = Planner("Decompose task")
workers = [Worker1("Task 1"), Worker2("Task 2"), ...]  # Created per-task
verify = Verifier("Check results")
# Workers disappear after task completes
```

**Multi-Agent Ch.5 (persistent orchestration):**
```python
plan = ManagerAgent.decompose(task)
workers = [WorkerAgent1, WorkerAgent2, ...]  # Persistent, have memory of past tasks
results = await asyncio.gather(*[w.execute(subtask) for w in workers])
verify = ManagerAgent.verify(results)
# Workers persist, learn patterns (e.g., "Task 1 usually takes 5s"), optimize over time
```

### When to Escalate from Ch.11 to Multi-Agent Track

✅ **Escalate to multi-agent when:**
- Tasks span multiple conversations (agents need long-term memory)
- Agents specialize (each agent becomes expert in one domain)
- Coordination is dynamic (agents negotiate, not hard-coded plan)
- Scaling beyond 4 agents (distributed systems)

❌ **Stay with Ch.11 patterns when:**
- Tasks are scoped to single conversation
- Agents are disposable (no benefit to persistence)
- Coordination is simple (hard-coded plan works fine)
- Token budget is tight (multi-agent overhead > benefit)

### Recommended Reading Order

1. **Finish this chapter** (Ch.11 — Advanced Agentic Patterns)
2. **Multi-Agent Ch.1** — Introduction to Multi-Agent Systems (agent types, communication protocols)
3. **Multi-Agent Ch.2** — Agent Memory & State Management (short-term vs. long-term memory)
4. **Multi-Agent Ch.3** — Cross-Agent Critique & Debate (persistent adversarial teams)
5. **Multi-Agent Ch.4** — Hierarchical Multi-Agent Systems (manager-worker coordination)
6. **Multi-Agent Ch.5** — Tool Specialization & Negotiation (agents own tools, negotiate usage)
7. **Multi-Agent Ch.6** — Production Multi-Agent Systems (AutoGPT, MetaGPT, CrewAI)

---

## Summary

You've learned four agentic patterns that trade tokens for reliability:

1. **Reflection** (3×): Generate → Critique → Revise. Fixes contradictions, ambiguity. 6× error reduction on edge cases.

2. **Debate** (6×): Propose → Challenge → Defend → Vote. Multi-agent adversarial reasoning. 40% better on complex decisions.

3. **Hierarchical** (5–7×): Planner → Workers → Verifier. Decomposes complex tasks. 15× success rate on multi-step problems.

4. **Tool Selection** (1.1–2.5×): Try cheap tools first, escalate on failure. 95% recovery from tool failures.

**Key insight:** Every pattern is a trade-off. Single-pass is fast and cheap (1×) but fragile. Reflection is robust (3×) but slower. Debate is most accurate (6×) but expensive. Hierarchical is most powerful (5–7×) but complex. Tool selection is most efficient (1.1×) but only applies to tool-based tasks.

**Production rule:** Route simple queries to single-pass. Route edge cases to reflection. Route high-stakes decisions to debate. Route complex tasks to hierarchical. Route tool-heavy tasks to tool selection.

**PizzaBot v2.0 results:**
- Edge case accuracy: 92% → 98.5% (6.5% improvement)
- Complex order success: 85% → 96% (11% improvement)
- Escalation rate: 8% → 0.8% (7.2% reduction)
- Cost: $0.08 → $0.18 avg (still under $0.25 budget)
- All 6 constraints satisfied ✅

**Next stop:** Multi-Agent AI track (`notes/04-multi_agent_ai/`) extends these patterns to persistent, coordinated, multi-agent systems.

---

**You now have the complete agentic AI toolkit (Ch.1–11).** You can build production agents that handle 99%+ of real-world inputs with graceful degradation, transparent reasoning, and cost-efficient orchestration. 🎉
