# Chain-of-Thought Reasoning — How LLMs Think Out Loud

> **A brief history.** In the autumn of 2021, a researcher at Google Brain named **Jason Wei** was staring at a failure. GPT-3 could write poetry, translate French, and summarise documents — but ask it *"Roger has 5 apples, gives away 3, then buys 7 more. How many does he have?"* and the 175-billion-parameter model would confidently answer wrong. The problem was structural: the model was jumping from prompt to final answer in a single decoding step, with no room to catch its own reasoning errors.
>
> What happened next was almost accidental. Wei's team tried inserting *worked examples* into the prompt — showing each arithmetic step before asking the actual question. The model's accuracy corrected itself immediately. In **January 2022**, the paper *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"* made it official: worked-example prompts pushed accuracy on a standard math benchmark from under 20% to over 58%. The catch: the ability only emerged on models above roughly 62 billion parameters — below that threshold, the reasoning steps came out as incoherent noise.
>
> The field moved fast. Wang et al. (March 2022) noticed that different CoT chains reached different answers, so why not sample many and take the majority vote? **Self-Consistency** added another tier of accuracy with almost no engineering. By May 2023, **Tree of Thoughts** was branching the chain and backtracking from dead ends. Then in September 2024 came the conceptual leap: OpenAI's **o1** stopped prompting for CoT and instead *trained* models with reinforcement learning to generate long internal reasoning traces before answering — traces the user never sees. **DeepSeek-R1** replicated the recipe openly four months later. Every reasoning model today — o1, o3, R1 — is a descendant of Wei's embarrassingly simple trick.
>
> When PizzaBot decomposes "cheapest gluten-free pizza under 600 calories" into four sequential checks before answering, it's running the same mechanism Wei stumbled onto in 2021.
>
> **Where you are.** Ch.1 (LLM Fundamentals) taught tokenization and sampling → but raw GPT-3.5 gave 15% error rate. Ch.2 (Prompt Engineering) added system prompts and few-shot → 12% conversion, but multi-step queries failed completely ("cheapest gluten-free pizza under 600 cal" → wrong answer). This chapter unlocks step-by-step reasoning that connects language to tool execution.
>
> **Business context.** Current status: **12% conversion** (target: >25%, phone baseline: 22%), **~15% error rate** (target: <5%), **$0.002/conv** (target: <$0.08). CEO question: "Your bot just recommended gluten to a gluten-intolerant customer. That's a lawsuit waiting to happen. Can you make it reason through multi-step queries correctly?"
>
> **Notation.** $K$ — number of independent CoT chains sampled in self-consistency; $\hat{a}$ — majority-vote final answer; $d$ — draft depth in speculative decoding; $\alpha$ — acceptance rate of speculative tokens; $C_K = K \cdot \bar{t}$ — total token cost of self-consistency ($\bar{t}$ = mean tokens per chain).

## Running Example: Mamma Rosa's PizzaBot

> This note uses the PizzaBot system defined in [AIPrimer.md](../ai-primer.md) as its running example. All tool names and queries below come from that system.

```
User: "What's the cheapest gluten-free pizza under 600 calories that you can deliver to me by 7 pm?"
```

> **How does "predict the next token" translate into "check allergen flags" or "call check_item_availability"?**

***

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — satisfying 6 constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings — 2. **ACCURACY**: <5% error — 3. **LATENCY**: <3s p95 — 4. **COST**: <$0.08/conv — 5. **SAFETY**: Zero attacks — 6. **RELIABILITY**: >99% uptime

**What we know so far:**
- ✅ Ch.1: LLM fundamentals (tokenization, sampling, training) → 8% conversion with raw GPT-3.5
- ✅ Ch.2: Prompt engineering (system prompts, few-shot, structured output) → 12% conversion
- ❌ **But multi-step queries fail completely** — bot jumps to answer without reasoning steps
- 📊 **Current business metrics**: 12% conversion (phone baseline: 22%), $38.10 AOV (baseline: $38.50), $0.002/conv, ~15% error rate

**What's blocking us:**

🚨 **Complex multi-constraint queries fail — no step-by-step reasoning**

**Test scenario: Complex menu query**
```
User: "What's the cheapest gluten-free pizza under 600 calories that you can
       deliver to me by 7 pm?"

PizzaBot (Ch.2 prompt engineering only):
{
  "answer": "I'd recommend our Personal Margherita pizza. It's gluten-free
             and under 600 calories at approximately 580 calories, priced at
             $13.99 with delivery available.",
  "confidence": "high"
}
```

**Problems:**
1. ❌ **Wrong answer!** Personal Margherita is NOT gluten-free (doesn't check allergen flags)
2. ❌ **Made up calorie count** ("approximately 580" — real: 650 calories for regular crust)
3. ❌ **Didn't consider all options** (skipped Veggie Garden which IS gluten-free, 540 cal, $14.99)
4. ❌ **No multi-step reasoning shown** (should: filter gluten-free → filter <600 cal → sort by price → check availability)
5. ❌ **Didn't call any tools** (`retrieve_from_rag()`, `check_item_availability()`)

**Actual correct answer:** "Veggie Garden, gluten-free crust, medium size" — $14.99, 540 calories, available by 7pm

**Business impact:**
- Customer orders Personal Margherita expecting gluten-free → allergic reaction → **lawsuit risk**
- 15% conversion (up from 12%, but still below 22% phone baseline)
- Complex queries have 35% error rate vs. 10% for simple queries
- CEO: "Your bot just told a gluten-intolerant customer to eat gluten. This is dangerous. Fix it or we're done."

**Why prompt engineering alone isn't enough:**

Simple queries work:
```
User: "What sizes do you have?"
Bot: [Looks at system prompt] "Personal, Medium, Large, Extra-Large"
✅ Success (single-step lookup)
```

Multi-step queries fail:
```
User: "Cheapest gluten-free under 600 calories"
Bot needs to:
  1. Retrieve menu items
  2. Filter by allergen=gluten-free
  3. Filter by calories<600
  4. Sort by price ascending
  5. Check availability for top result
  6. Return answer

❌ Bot jumps straight to answer without reasoning steps
❌ Makes up facts instead of using tools
```

**What this chapter unlocks:**

🚀 **Chain-of-Thought (CoT) reasoning:**
1. **"Think step-by-step" prompting**: Explicitly ask model to show reasoning before answering
2. **Tool-augmented CoT**: Connect reasoning steps to actual tool calls (`retrieve_from_rag()`, `check_item_availability()`)
3. **ReAct pattern** (Reasoning + Acting): Interleave Thought → Action → Observation loops
4. **Verify intermediate steps**: Check reasoning chain for logical errors

⚡ **Expected improvements:**
- **Error rate**: 15% → ~10% (multi-step reasoning reduces logic errors, but still need RAG for grounding)
- **Conversion**: 12% → ~15% (complex queries now work, improving trust)
- **Safety**: Catches allergen mismatches in reasoning trace ("Wait, Margherita isn't gluten-free")
- **Cost**: $0.002 → $0.004/conv (longer outputs for reasoning steps, but still well below $0.08 target)
  - Input: ~150 tokens (system prompt) + 20 tokens (query) = 170 tokens × $0.015/1M = $0.000003
  - Output: ~200 tokens (with reasoning) × $0.06/1M = $0.000012
  - Total: $0.000015/query (2× queries per conversation) = **$0.00003/conv** (was $0.000015)
- **Complex query success**: 65% → 85% (up from 65%)

⚡ **Constraint #2 ACCURACY [PARTIAL PROGRESS]**: Error rate 15% → ~10% (target: <5%). Still need RAG (Ch.4) for menu grounding.

**Constraint status after Ch.3**:
- #1 (Business Value): 15% conversion — still below 22% phone baseline
- #2 (Accuracy): 10% error — major improvement, but need RAG (Ch.4) to hit <5%
- #4 (Cost): $0.00003/conv — still excellent headroom (target: <$0.08)

Still need Ch.4 (RAG) for real menu grounding to eliminate remaining hallucinations and hit <5% error target.

***

## 1 · What Is Chain-of-Thought (CoT) Reasoning?

**Chain-of-thought reasoning** refers to an LLM producing a **sequence of intermediate reasoning steps** that bridge the prompt to the final answer. Chain-of-thought prompting is the practice of prompting a model to perform a task step-by-step and to present each step and its result in order in the output[1](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/chain-of-thought-prompting). This simplifies prompt engineering by offloading some execution planning to the model and makes it easier to connect any problem to a specific step[1](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/chain-of-thought-prompting).

Two variants are commonly conflated:

| Variant                     | What Happens                                                                   | Visibility                      |
| --------------------------- | ------------------------------------------------------------------------------ | ------------------------------- |
| **CoT Prompting (visible)** | The model is instructed to "think step-by-step" and prints intermediate steps  | User sees the reasoning         |
| **Hidden Reasoning Tokens** | The model allocates internal "thinking tokens" before producing the final text | User sees only the final answer |

**Reasoning tokens** (also called thinking tokens) are tokens the model generates internally to work through a problem before producing the visible response.

You can instruct the model to include its chain of thought — that is, the steps it took to follow an instruction, along with the results of each step — either via explicit instructions or by providing examples that demonstrate how to break down tasks[1](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/chain-of-thought-prompting).

### 1.1 Zero-shot vs Few-shot CoT

Before CoT, PizzaBot had one structural problem: it leapt from "cheapest gluten-free under 600 cal" straight to a confident wrong answer, skipping every intermediate check. The fix comes in two flavours depending on how much guidance you give the model.

**Technical definition.** *Few-shot CoT* prepends $n$ worked examples to the prompt — each example is a (question, reasoning trace, answer) triple. Formally the prompt is $P = [e_1, \ldots, e_n, q]$ where $e_i = (\text{question}_i, \text{steps}_i, \text{answer}_i)$. *Zero-shot CoT* (Kojima et al., 2022) adds no examples at all — it appends a single instruction such as "Let's think step by step" to the user query. That instruction alone triggers the model to emit intermediate steps before the final answer, because instruction tuning exposed the model to this phrasing thousands of times during training.

**Intuition.** Few-shot CoT shows the model the *format* it should use — concrete examples of how multi-step reasoning should look. Zero-shot CoT trusts the model to infer the format from the instruction alone. Zero-shot works on large models and is noisier than few-shot, but it costs zero extra prompt tokens.

**PizzaBot grounding.** Without CoT the system prompt ends with `"Answer the user's question."` — the bot hallucinates, jumping straight to an answer. Add zero-shot CoT: `"Think through each step before answering."` The bot starts listing steps. Add two worked examples of multi-constraint queries as few-shot CoT in the system prompt, and the bot reliably formats its reasoning in the right structure on every complex query.

> 💡 **Zero-shot vs few-shot verdict:** Zero-shot CoT adds ~0 tokens; two-shot CoT adds ~400 tokens at $0.015/1\text{M} = \$0.000006$ per query — negligible cost for a measurable format improvement. For PizzaBot, start with zero-shot; upgrade to two-shot if multi-constraint accuracy stays below 90%.

### 1.2 Self-Consistency

Zero-shot and few-shot CoT still produce a single reasoning chain — and a single chain can take a wrong fork early and propagate that error all the way to the final answer.

**Technical definition.** Self-consistency (Wang et al., 2022) samples $K$ independent CoT chains at temperature $T > 0$ and returns the majority-vote answer:

$$\hat{a} = \operatorname{argmax}_{a} \sum_{k=1}^{K} \mathbf{1}[\operatorname{answer}(c_k) = a]$$

where $c_k$ is the $k$-th sampled chain. Total inference cost is $C_K = K \cdot \bar{t}$ tokens, where $\bar{t}$ is the mean token count per chain. Because chains are sampled independently with different random seeds, they tend to make *different* errors — so the majority answer is more reliable than any single chain.

**Intuition.** Ask five colleagues the same question without letting them confer. If four reason their way to "Veggie Garden" and one says "Margherita," you go with Veggie Garden. Unanimous agreement among independent reasoners is strong evidence. Disagreement tells you the query is hard and you should escalate.

**PizzaBot grounding.** For a high-stakes allergen query — "Is this pizza gluten-free?" — a wrong single-chain answer risks a customer lawsuit. With $K = 5$ chains, if four independently reason to "No — this pizza uses regular wheat dough," return that answer with high confidence. For simple queries ("What sizes do you have?"), $K = 1$ is fine; the extra chains would agree anyway and just add cost.

> ⚠️ **Self-consistency cost trap:** $C_K = K \cdot \bar{t}$ tokens. At $K = 5$, $\bar{t} \approx 200$ tokens/chain: cost per query jumps 5× compared to a single chain — still well under PizzaBot's $0.08/conv target, but **not free**. Reserve self-consistency for allergen checks and availability confirmations, not routine catalogue lookups.

***

## 2 · High-Level Architecture of an LLM-Based Agent

An AI agent is more than a one-shot text generator. Agents have three key elements: a **large language model** (the agent's "brain," using generative AI for language understanding and reasoning), **instructions** (a system prompt that defines the agent's role and behavior), and **tools** (what the agent uses to interact with the world — including knowledge tools that provide access to information, like search engines or databases, and action tools that enable the agent to perform tasks)[3](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/7-agents).

```plaintext
+------------------+                     +-------------------+
|  User Query      | ──(1)──▶  [ LLM Agent ]  ──(2)──▶  [ Tool ]
|  ("Find avg &    |           |  (Planner +       (Search, Calc,
|   max speed")    |           |   Executor)        Maps API …)
+------------------+           |                    |
                               ▼                    |
                       (3) Observation              |
                               |                    ▼
                               |        (4) Updated Context/State
                               |                    |
                               +◀──[ Memory / Scratchpad ]◀──+
                                                    |
                                                    ▼
                                          (5) Final Answer
```

**How each component works:**

1.  **User Query:** The natural language request (e.g., *"What's the cheapest gluten-free pizza under 600 calories for delivery by 7 pm?"*).
2.  **LLM Agent (Planning & Reasoning):** The LLM interprets the query and devises a plan, identifying what information and steps are needed. A foundational pattern is **ReAct** (Yao et al., 2022), which combines reasoning and acting in an interleaved loop: the agent generates a thought (reasoning trace), takes an action (tool call), and observes the result.
3.  **Tool Use (Action Execution):** Based on its plan, the agent invokes an external tool — web search, database query, calculator, or other API.
4.  **Observation & Context Update:** The tool returns a result. The agent's working memory is updated with this new information, and the LLM reads the enriched context to decide the next step.
5.  **Final Answer:** When the agent has gathered enough information and performed necessary computations, it produces a final answer drawing on all compiled context.

**ReAct** was one of the first approaches to enhance AI agent capabilities and has become a standard pattern in frameworks like LangChain and LlamaIndex. It works in a sequential manner, with the same LLM responsible for both reasoning and executing the action within a single step.

***

## 3 · The Critical Missing Bridge: How "Next-Token Prediction" Becomes a Plan

This is the key section that most agent documentation omits. At runtime:

*   The LLM receives tokens.
*   It outputs tokens.
*   It predicts the next most probable token sequence.

**So the real engineering question is: How does "predict next token" turn into "check_item_availability(store_id=3, item='Margherita')"?**

### 3.1 The Answer: Planning = Constrained Next-Token Decision Over an Action Language

**An LLM-based agent does NOT execute tools.** Instead, the surrounding system defines an **action language** inside the prompt. The model outputs tokens in that language. The host program parses those tokens and executes tools.

In practice, the prompt includes an explicit **menu of actions and tools**. For example:

```plaintext
AVAILABLE_ACTIONS:
1. retrieve_from_rag(query)                        → returns MENU_CHUNKS
2. check_item_availability(store_id, item)          → returns {available, eta_minutes}
3. find_nearest_location(address)                  → returns {store_id, name, is_open}
4. calculate_order_total(items, address)            → returns {subtotal, tax, total}

TASK:
Given the user's question and current state,
select the NEXT BEST ACTION.
Return ONLY: { "action": "...", "args": {...} }
```

Now the LLM's job becomes: **produce the most probable next tokens — but valid completions are constrained to structured actions.** For instance:

```json
{ "action": "retrieve_from_rag", "args": { "query": "gluten-free pizza calorie counts" } }
```

This is simply the model's chosen token sequence. The **host program**:

1.  Parses the structured output.
2.  Executes the tool.
3.  Feeds the result back as tokens (e.g., `Observation: DISTANCE_KM = <value>`).

That observation becomes part of the **next context window**, shifting the probability distribution for the next output. The agent is therefore an **LLM policy** over a **tool-augmented action space** (token strings that map to tool calls).

### 3.2 Why "Average Speed" Implies "Need Distance" — Two Distinct Learned Behaviors

Two separate learning stories explain how the model connects a user's question to the correct sequence of tool calls:

**A) Semantic Association (Pretraining)**

Even with "just next-token prediction," transformers learn context-aware representations. To predict the next token well, the model must compress patterns like:

*   "average speed" often appears near "distance ÷ time"
*   routes imply "distance" as a variable
*   "Seattle to Vancouver" implies a missing factual variable that can be looked up

Models encapsulate semantic relationships between language elements — that is how they generate a meaningful sequence of text[5](https://learn.microsoft.com/en-us/training/modules/get-started-ai-fundamentals/2-generative-ai). So the weights encode *statistical regularities that function like semantic knowledge*.

**B) Tool-Use Policy (Instruction Tuning + Exemplars)**

Tool-using behavior is learned or elicited through instruction tuning, demonstration trajectories (like ReAct Thought/Action/Observation traces), and reinforcement or preference optimization. The model's training data includes sequences where the correct behavior was to output a tool call when factual information was missing.

So the internal "logic" becomes:

    Need AVG speed
    → AVG = distance / time
    → time is known (4h), distance is missing
    → RouteDistance tool exists in the schema
    → output the action tokens to call it

It is still next-token prediction — just over a vocabulary that includes action-like structured tokens, constrained by the tool schemas provided in context.

### 3.3 Context Engineering = Planning Control Surface

The agent will only plan correctly if:

*   **Tool schemas** are present in the context
*   The **decision question** is explicitly framed ("Given the current state, select the next action")
*   **State** is summarized each turn

Planning quality therefore depends directly on **what tokens you allow the model to see**. This is why "context engineering" matters: the model's context window is literally its entire world state.

### 3.4 How Action Execution Happens If the Model Only Outputs Text

The primary mechanism is **"stop-and-parse" / structured output**:

1.  The model outputs a JSON blob, e.g., `{"action":"WebSearch","args":{"query":"..."}}`
2.  The host program halts generation.
3.  The host program parses the JSON and executes the tool.
4.  The host program appends the result as an Observation token.
5.  The model continues generating with the enriched context.

Some stacks provide explicit **function-calling APIs** with stricter schema enforcement, but it is still tokens under the hood.

The Augmenter framework for agent execution provides a concrete example: it uses OpenAI's native tool calling interface with schema-driven tool invocation, enabling the model to execute tools iteratively (supporting sequential and conditional flows like "get X, then if > Y call Z") and to incorporate tool outputs directly into reasoning for more natural and coherent responses.

### 3.5 One-Sentence Summary of the Connecting Logic

**An agent is an LLM whose next-token prediction is constrained to output a structured "next action," and whose environment executes that action and turns the result back into tokens, creating a feedback loop.**

***

## 4 · Step-by-Step Worked Example: PizzaBot Multi-Constraint Query

**User prompt:** *"What's the cheapest gluten-free pizza under 600 calories that you can deliver to me by 7 pm?"*

This decomposes into four sequential checks: allergen compliance → calorie filter → delivery availability → price sort. Neither the allergen data nor the calorie counts are in the model's weights — both come from the RAG corpus.

**What happens at each step in ReAct terms:**

| Loop | Thought (Planning)                                      | Action (Execution)                                              | Observation                              |
| ---- | ------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------- |
| 1    | "I need the gluten-free options from the menu"          | `retrieve_from_rag("gluten-free pizzas")`                        | `[Margherita, Veggie Feast, Napoli GF]`  |
| 2    | "Check calorie counts for each gluten-free option"     | `retrieve_from_rag("calorie counts gluten-free pizzas")`         | `Margherita 520 kcal, Veggie 490 kcal`   |
| 3    | "Two options under 600 kcal — check delivery ETA"      | `check_item_availability(store_id=nearest, item="Veggie Feast")` | `{available: true, eta_minutes: 28}`     |
| 4    | "Available in time — retrieve prices and sort"         | `retrieve_from_rag("Veggie Feast price, Margherita price")`      | `Veggie Feast £11.99, Margherita £13.99` |
| 5    | "Veggie Feast is cheapest and meets all constraints"   | `FINAL_ANSWER`                                                   | *(generates response)*                   |

Throughout this process, the LLM's internal chain-of-thought assesses at each step:*"Am I done, or do I need more info? If not done, what is the next missing piece?"* This tight feedback loop enables **dynamic replanning** based on real-world outcomes.

***

## 5 · How the Agent's Context/State Evolves at Each Step

A distinguishing feature of this architecture is the **structured memory** (scratchpad or context buffer) that accumulates conversation history, internal reasoning, and tool results:

At each step, the agent's context **grows monotonically**. After Step 2, the agent's state includes the distance. When planning Step 3, it will not repeat the distance lookup — it knows that sub-task is resolved and moves to the next gap (train type and max speed). This is exactly the **ReAct pattern** in action: the "Thought" is an internal note about what to do next; the "Action" is the tool used; the "Observation" is the tool's output.

**Memory implementation options vary:** In practice, "memory" is often the conversation history plus chain-of-thought and results appended as text for the LLM to read. More advanced agents may use key-value memory stores or vector databases. The fundamental principle remains: the agent is **building an internal representation of its progress** that informs each subsequent decision.

### Context Management in Agentic Frameworks (ReAct)

In standard ReAct implementations, agents default to a **linear, raw append** model. However, production-grade systems utilize specific architectural patterns to mitigate context window saturation and "attention drift."

#### 1. Recursive Summarization (Summarization Triggers)
* **Mechanism:** Once the conversation history crosses a predefined token threshold, a "compressor" LLM call is triggered.
* **Process:** The previous $N$ steps are transformed into a concise semantic state.
* **Technical Impact:** Preserves high-level intent and discovered facts while discarding the specific linguistic overhead of earlier iterations.

#### 2. Vector-Based Context Injection (RAG)
* **Mechanism:** Decouples tool outputs from the active prompt.
* **Process:** Raw "Observations" (e.g., full API responses or web scrapes) are stored in a local vector database. The agent is then provided only with the top-$k$ most relevant snippets via semantic search.
* **Technical Impact:** Efficiently handles massive data ingestion without exceeding the context window.

#### 3. Deterministic Truncation (Middleware Filtering)
* **Mechanism:** Applies a hard limit on "Observation" tokens.
* **Process:** Middleware intercepts tool responses and truncates them (e.g., keeping only the first 800–1000 tokens).
* **Technical Impact:** Prevents "Prompt Bloat" and ensures the model's self-attention remains focused on the reasoning trace rather than exhaustive raw data.

#### 4. Explicit State Management (State Machines)
* **Mechanism:** Transitions from a "Chat History" model to a "State Schema" model (e.g., LangGraph).
* **Process:** Only specific, typed variables (e.g., `extracted_date`, `current_query`) are passed between nodes in the graph.
* **Technical Impact:** Discards conversational "fluff" entirely, ensuring the model only processes functionally necessary data points for the next state transition.

***

## 6 · Planning vs. Execution — Two Modes of Agent Operation

The agent alternates between two distinct operational modes:

**Planning-time** is when the agent formulates its approach. The LLM reasons internally — possibly via a hidden scratchpad — to figure out the solution path. In ReAct, this is explicitly written as a "Thought" before the "Action".

**Execution-time** is when the agent carries out the planned action — interacting with external systems and receiving concrete observations. After each action, the agent returns to planning: new information is considered, and the LLM decides if another action is needed.

**Why separate the two?**

*   **Adaptability:** The plan adjusts if a tool result was unexpected or if the question needs clarification.
*   **Safety:** Planning often involves exploring uncertain possibilities that should not be shown to the user.
*   **Efficiency:** Independent sub-tasks identified during planning can be executed in parallel.

***

## 7 · Advanced Reasoning Structures Beyond Linear Chains

Chain-of-Thought prompting elicits step-by-step reasoning in a **linear chain**. When a linear chain is insufficient, agents can use more expressive structures:

```plaintext
    CoT (Linear)          ToT (Tree)              GoT (Graph)

    Step 1                  Step 1                  Step 1
      ↓                   /   |   \               /   |   \
    Step 2             2a    2b    2c           2a    2b    2c
      ↓                |     |     |             \   / \   /
    Step 3            3a    3b    3c              Merge
      ↓              (✗)   (✓)   (✗)               ↓
    Answer                  ↓                    Refine
                         Answer                    ↓
                                                Answer
```

*   **Tree of Thoughts (ToT)** explores multiple reasoning paths simultaneously using tree search (BFS/DFS). Each intermediate thought is evaluated for promise, allowing the agent to **backtrack** from unproductive branches.
*   **Graph of Thoughts (GoT)** generalizes planning to arbitrary directed graphs, enabling aggregation of partial solutions, refinement loops, and non-linear information flow.

**Practical implication for PizzaBot:** For most menu queries — tasks with clearly independent sub-tasks (filter allergens → filter calories → check availability) — a simple linear ReAct loop is sufficient. ToT and GoT become valuable for tasks requiring **exploration** (e.g., "suggest a dinner for 4 people with mixed dietary restrictions under $60" where multiple meal combinations need to be explored and evaluated, or creative menu descriptions where backtracking from unproductive phrasing is beneficial).

**Business impact:** Linear CoT handles 85% of PizzaBot queries. Advanced structures add 2-5× cost (multiple reasoning branches) for marginal accuracy gains. Deploy only for high-value queries where exploration is essential.

***

## 8 · Reasoning Tokens: How Hidden Planning Works in Practice

**Reasoning tokens** are tokens the model generates internally to work through a problem before producing the visible response. This is a concrete engineering mechanism that enables internal planning without exposing intermediate thoughts to the user.

**Why this matters for the bridging logic:** When the agent processes a PizzaBot query like "cheapest gluten-free pizza under 600 calories", the model may spend a significant portion of its token budget on internal reasoning (working out the plan, evaluating which filters to apply, deciding which tools to call) and a smaller portion on the visible answer. The user never sees the internal deliberation — only the final, polished response.

**Important:** Whether reasoning is visible or hidden does **not** change the tool-execution bridge described in Section 3. Tools still require a structured action token sequence, and the host program still executes the tool call and appends observations.

### 8.1 How Reasoning Models Are Trained — RLVR

Section 8 covers what reasoning tokens look like at **inference time**. The training recipe that produces models capable of extended hidden reasoning is distinct from standard RLHF.

**RLVR (Reinforcement Learning from Verifiable Rewards)** is the approach behind DeepSeek-R1, OpenAI o1, and o3. Instead of relying on human preference pairs (which RLHF/DPO requires), RLVR uses **automatically verifiable outcomes** as the reward signal:

| Reward type | Example verifier | Used by |
|---|---|---|
| Math correctness | Check final number against known answer | DeepSeek-R1, o1 |
| Code pass/fail | Run unit tests against test suite | o1-preview, HumanEval |
| Formal proof | Lean/Isabelle verifier | Research models |
| Game outcome | Win/loss against deterministic rules engine | AlphaCode-style |

**The training loop:**

```
1. Sample a question with a known correct answer
2. Model generates reasoning trace (chain-of-thought tokens) + final answer
3. Verifier checks the final answer only: correct → +1 reward, wrong → 0
4. RL (GRPO, PPO, or similar) updates the model to reinforce reasoning
   traces that led to correct answers
```

**Why verifiable rewards beat human preference labels:**
- **Scale:** 100M+ math problems available; human annotators cannot verify complex proofs
- **Consistency:** The verifier is deterministic — no annotator disagreement or label noise
- **Long-horizon credit assignment:** RL rewards can credit an early reasoning step that enabled a correct final answer
- **No reward hacking:** Automated verifiers can't be fooled by fluency (unlike learned reward models)

> 💡 **Key consequence:** Models trained with RLVR learn to allocate more reasoning tokens on harder problems. The training signal naturally produces **compute-difficulty scaling** — observed in o3's reported behaviour where harder math problems receive proportionally longer hidden traces.

> ⚠️ **PizzaBot note:** RLVR applies to domains with verifiable outcomes (math, code, formal logic). For PizzaBot's brand voice problem — open-ended language quality — **DPO with human preference pairs** (see [ch10 §5.5](../ch10-fine-tuning/fine-tuning.md)) is more appropriate than RLVR.

### 8.2 Process vs Outcome Reward Models (PRM / ORM)

The RLVR loop above gives one reward signal per problem: correct or wrong. That's an **Outcome Reward Model**. Its blind spot matters when intermediate reasoning steps contain errors that happen to produce a correct final answer.

**Technical definition.** An **Outcome Reward Model (ORM)** assigns a scalar reward to the final answer only: $r = \text{Verify}(a_{\text{final}}) \in \{0, 1\}$. A **Process Reward Model (PRM)** assigns a scalar reward $r_t$ to each intermediate reasoning step $s_t$, with the training signal $R = \sum_{t} \gamma^t r_t$ (a discounted sum over all steps). Human annotators — or a trained step-level verifier — label each step as correct, incorrect, or neutral. The policy is updated to maximise whichever reward signal is used.

**Intuition.** ORM is blind to how you reached the answer. A model can output "Veggie Garden, $14.99" by memorising the answer from training data and still receive full ORM reward — it has learned nothing about *why* Veggie Garden is correct. PRM requires every step to be correct, forcing the model to learn the *reasoning procedure* (filter allergens → check calories → sort by price). That procedure generalises when the menu changes.

**PizzaBot grounding.** Fine-tune PizzaBot on 1,000 query-answer pairs with ORM: it achieves 92% accuracy on the test set. Now Mamma Rosa adds three new gluten-free options — accuracy drops back to 70% because the model memorised final answers rather than filtering logic. A PRM-trained model learns step-level rules: first check allergen flags, then apply the calorie cap, then sort by price. When new items appear, each step still executes correctly.

> 💡 **PRM vs ORM verdict:** PRM requires annotating every intermediate step — roughly 10× more annotation effort than ORM. For PizzaBot's current scale and menu stability, ORM is sufficient. Consider PRM when intermediate steps must be independently auditable: medical diagnosis, legal reasoning, or multi-step financial calculations where a correct-looking answer via flawed reasoning creates liability.

***

## 9 · CoT in Production: What It Looks Like in Practice

**For PizzaBot**, showing CoT reasoning in the UI has concrete business implications:

**Scenario: Multi-constraint query**
```
User: "Cheapest gluten-free pizza under 600 calories"

PizzaBot UI shows:
  ✓ Step 1: Finding gluten-free pizzas... (0.8s)
  ✓ Step 2: Checking calorie counts... (0.5s)
  ✓ Step 3: Verifying availability... (0.6s)
  ✓ Step 4: Comparing prices... (0.3s)

  "Veggie Garden with gluten-free crust, $14.99, 540 calories, available in 35 min"
```

**Business impact of visible CoT:**
- **Trust building**: Customers see the bot actually checked allergen data (reduces "did it really check?" anxiety)
- **Perceived accuracy**: 4-step process signals thoroughness (A/B test: +8% conversion vs. instant answer)
- **Debugging**: Support team can verify which step failed when orders go wrong
- **Cost**: +200 tokens per query (was 170, now 370) = $0.000056/query (still well below $0.08 target)

**UX research findings** (from production agent deployments like Copilot Studio WebChat):
- Most users **don't want to read reasoning continuously** — they check the final answer first
- CoT is used as a **verification backstop** when something looks suspicious
- Users prefer **clear step structure (4–6 steps) + brief rationale**, not raw model thought transcripts

**Tradeoff:** Showing CoT increases user trust for complex multi-step tasks, but unscoped or unfiltered reasoning can create overload, slow perceived performance, and reduce clarity. This is one motivation behind hidden reasoning tokens — keeping the scratchpad internal while returning only a short explanation.

**PizzaBot decision:** Show 4-6 high-level steps for complex queries (multi-constraint, availability checks), hide internal reasoning for simple queries ("What sizes do you have?").

***

## 10 · The Planning-Execution Loop (Pseudocode)

Here is a simplified pseudo-code of the **planner-executor loop**, illustrating how the agent handles a query by deciding on actions step by step and updating its context:

```python
state = extract_state(user_input)      # e.g., query="cheapest gluten-free under 600 cal"
history = []

while True:
    # 1. Build context: system rules + tool schemas + current state + history
    prompt = build_context(system_rules, tool_schemas, state_summary(state), history)

    # 2. LLM decides next action (planning via next-token prediction)
    decision = LLM.generate(prompt)    # outputs JSON: {"action": ..., "args": ...}
    action = parse_action(decision)

    # 3. Check if done
    if action.name == "FINAL_ANSWER":
        print(action.args["answer"])
        break

    # 4. Execute the tool (system world, not the LLM)
    observation = execute_tool(action.name, action.args)

    # 5. Update state and history with the new observation
    history.append({"action": action, "observation": observation})
    state = update_state(state, observation)
```

**How this maps to the PizzaBot multi-constraint query:**

| Loop | `thought`                                     | `action`                                       | `observation`               |
| ---- | --------------------------------------------- | ---------------------------------------------- | --------------------------- |
| 1    | "I need gluten-free pizzas from the menu"     | `retrieve_from_rag("gluten-free pizzas")`                  | `[Veggie Garden (GF), Margherita (regular only)]`               |
| 2    | "Now I need calorie counts for GF options" | `retrieve_from_rag("Veggie Garden gluten-free calories")` | `Veggie Garden GF: 540 cal, $14.99` |
| 3    | "540 < 600, qualifies. Check availability"           | `check_item_availability(store_id=nearest, item="Veggie Garden GF")`                | `{available: true, eta: 35min}`                 |
| 4    | "Available! This is the cheapest option"        | `FINAL_ANSWER`                                   | *(generates response)*      |

**Key properties of this loop:**

*   **Dynamic planning:** The plan is not fixed upfront. If `retrieve_from_rag()` returned no results, the agent could re-plan (e.g., try a broader search query or ask for clarification).
*   **Stateful context:** Each iteration has access to the full history of prior thoughts, actions, and observations. The LLM leverages this accumulated state to make increasingly informed decisions.
*   **Termination condition:** The agent decides when to stop based on its assessment of whether the gathered information is sufficient. There is no hardcoded step count — the loop runs until the LLM's planning function concludes that the goal is met (all constraints satisfied: gluten-free ✅, <600 cal ✅, available ✅, cheapest ✅).

***

## 11 · Practical Implications: From Traditional Dev Thinking to Agentic Thinking

Understanding the bridging logic has a direct impact on how you architect agentic systems:

| Traditional Dev Thinking                  | Agentic Thinking                                     | PizzaBot Example |
| ----------------------------------------- | ---------------------------------------------------- | ---------------- |
| `if (needDistance) CallDistanceAPI();`    | Provide state + tools → let model choose next action | Provide menu tools → let LLM decide which to call |
| Imperative orchestration (hardcoded flow) | Token-space policy (model decides dynamically)       | No hardcoded "step 1: check allergens, step 2: check calories" — LLM plans |
| Finite state machine / decision tree      | Context-conditioned action selection                 | Model adapts to query complexity (1 step for "What sizes?" vs. 4 steps for multi-constraint) |

**Traditional approach (hardcoded):**
```python
if "gluten-free" in query and "calories" in query:
    results = retrieve_from_rag("gluten-free")
    filtered = [r for r in results if r.calories < extract_calorie_limit(query)]
    sorted_results = sorted(filtered, key=lambda x: x.price)
    return sorted_results[0]
```

**Agentic approach (token-space policy):**
```python
STATE + TOOLS + TASK → LLM chooses next action
# LLM outputs: {"action": "retrieve_from_rag", "args": {"query": "gluten-free pizzas"}}
# Host executes tool, appends observation
# LLM outputs: {"action": "retrieve_from_rag", "args": {"query": "calorie counts"}}
# ... continues until FINAL_ANSWER
```

**The intelligence in the agent emerges from the interplay of the LLM's learned knowledge and the sandboxed execution of tools dictated by the LLM's outputs**, all made possible by constraining the format of the model's responses to bridge the gap between text and action.

**Business advantage for PizzaBot:** Traditional approach breaks when query changes ("cheapest *vegan* under 600 cal"). Agentic approach adapts without code changes — just add vegan filter to tool schema.

***

## 12 · Key Nuance: CoT Is Not Guaranteed to Be Faithful

Even when a model prints reasoning steps, they can be:

*   Partially fabricated
*   Optimized for *looking* reasonable rather than reflecting true internal computation
*   Contain errors that still lead to a confident final answer

This is one motivation behind **process supervision** (training correctness of individual steps rather than just final answers) and behind approaches that keep reasoning internal while returning only the final answer (hidden reasoning tokens).

***

## Summary: The Complete Mental Model

**An LLM agent is:**

> A next-token predictor operating over a prompt that includes an action language (tools + schemas), where the surrounding program treats certain token patterns as executable actions, executes them, and feeds results back as tokens — forming a feedback loop.

Planning is **emergent behavior** from:

1.  **Semantic representations** learned via next-token prediction (the model "knows" that average speed requires distance and time).
2.  **A constrained decision framing** (action language + state + tools) that turns "what next?" into the next-token completion.
3.  **An iterative loop** where each tool result enriches context, shifting the model's predictions toward the next correct step.

---

## 13 · Progress Check — What We Can Solve Now

🎉 **MAJOR PROGRESS**: Complex multi-step queries now work!

**Unlocked capabilities:**
- ✅ **Chain-of-thought prompting**: Model shows reasoning steps before answering
- ✅ **ReAct pattern**: Thought → Action → Observation loop working
- ✅ **Tool-augmented reasoning**: Can connect reasoning to tool calls
- ✅ **Multi-constraint queries**: Can handle "cheapest X with Y under Z" logic
- ✅ **Reasoning verification**: Can spot logical errors in intermediate steps

**Progress toward constraints:**

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ❌ | 15% conversion (target: >25%, phone baseline: 22%), $38.10 AOV (baseline: $38.50), no labor savings yet — Complex queries now work but still below phone baseline |
| #2 ACCURACY | ⚡ | ~10% error rate (down from 15%, target: <5%) measured on 1,000 test queries — Multi-step reasoning reduces logic errors, but still hallucinates menu items. Need RAG (Ch.4). |
| #3 LATENCY | ⚡ | 2.5s p95 (target: <3s) measured on 500 conversations — CoT adds ~0.5s but acceptable. Within target! |
| #4 COST | ✅ | $0.00003/conv (target: <$0.08) — Input: 170 tokens × $0.015/1M + Output: 200 tokens × $0.06/1M = $0.000015/query × 2 queries = $0.00003/conv. Huge headroom for RAG overhead. |
| #5 SAFETY | ⚡ | Reasoning traces catch some allergen errors before answering (10/12 caught in testing) — Improved but need guardrails (Ch.9) |
| #6 RELIABILITY | ❌ | No error handling yet — tool failures cause crashes. Need orchestration layer (Ch.6) |

**What we can solve:**

✅ **Multi-constraint queries with reasoning**:
```
User: "Cheapest gluten-free pizza under 600 calories"

PizzaBot (with CoT):
Thought: "I need to find pizzas that are both gluten-free AND under 600 calories,
         then sort by price."

Action: retrieve_from_rag("gluten-free pizzas")
Observation: [Veggie Garden (gluten-free crust option), Margherita (regular only)]

Thought: "Now I need calorie information for gluten-free options."

Action: retrieve_from_rag("Veggie Garden gluten-free calories")
Observation: "Veggie Garden with gluten-free crust, medium: 540 calories, $14.99"

Thought: "540 < 600, so this qualifies. Let me check if there are other options."

Action: retrieve_from_rag("other gluten-free pizzas under 600 calories")
Observation: [No other options under 600 calories]

Thought: "Veggie Garden is the only option. Let me verify availability."

Action: check_item_availability(store_id=nearest, item="Veggie Garden gluten-free")
Observation: {available: true, eta_minutes: 35}

Answer: "The cheapest gluten-free pizza under 600 calories is the Veggie Garden
        with gluten-free crust (medium, $14.99, 540 calories). Available for
        delivery by 6:45 PM."

Result: ✅ Correct! Complex query solved with multi-step reasoning!
```

❌ **What we can't solve yet:**
- **Menu hallucinations still occur**: Model can plan "retrieve_from_rag()" but doesn't actually execute it — reasoning is still from parametric memory, not real menu data. Need Ch.4 (RAG) to ground answers in actual menu corpus.
- **No proactive upselling**: Can answer questions but can't suggest "Would you like garlic bread with that?" Need Ch.6 (ReAct & Semantic Kernel) for tool orchestration.
- **Slow responses at scale**: Single-threaded reasoning loop. No vector search optimization. Need Ch.5 (Vector DBs) for fast retrieval.
- **No error handling**: If `check_item_availability()` fails, bot crashes. Need Ch.6 for graceful degradation.

**Business metrics update:**
- **Order conversion**: 15% (up from 12%, baseline: 22%) — **Improving! Complex queries now work but still below phone baseline**
- **Average order value**: $38.10 (baseline: $38.50) — Slightly below (no upselling yet, need ReAct tools in Ch.6)
- **Cost per conversation**: $0.00003 (target: <$0.08) — **Huge headroom!** 2,666× under target, room for RAG overhead
- **Error rate**: ~10% (target: <5%) measured on 1,000 held-out test queries — **Major improvement, but need RAG (Ch.4) for final push to <5%**
- **Complex query success**: 85% (up from 65%) — **Multi-step reasoning works!**
- **Safety incidents**: 2 in 500 test conversations (reasoning caught 10/12 allergen errors) — Need systematic guardrails (Ch.9)

**Next chapter**: [RAG & Embeddings](../ch04_rag_and_embeddings) connects the reasoning chain to real menu retrieval. CoT currently says "Action: retrieve_from_rag()" but doesn't actually execute it — still reasoning from parametric memory. Ch.4 makes that tool call real, grounding all answers in actual menu data → **<5% error rate (Constraint #2 ✅), 18% conversion**.

---

## Bridge to Next Chapter

CoT gives the model a structured way to reason through multi-step problems — but it's still reasoning from parametric memory. When a customer asks "Is the Margherita gluten-free today?", CoT can plan the retrieval but cannot actually look it up. **RAG & Embeddings (Ch.4)** adds the retrieval layer: a semantic search over Mamma Rosa's menu corpus that fetches the actual answer before the model reasons about it. CoT + RAG together reduce error rate from 10% to <5% and push conversion from 15% to 18%.

---

## Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Chain-of-thought prompting adds explicit intermediate reasoning steps to the model's output; this works because it forces a left-to-right decomposition that the model can condition on, rather than jumping from prompt to answer in one step | "Why does CoT improve performance on arithmetic or multi-step tasks?" (Answer: each generated step is visible context for the next prediction, making the problem compositionally easier) | Saying CoT works because the model "thinks harder" — it works because each step is usable context, not because there is a separate reasoning system |
| Zero-shot CoT: appending "Let's think step by step" (Kojima et al. 2022) elicits reasoning without any worked examples; Wei et al. 2022 showed few-shot CoT examples outperform simple answer examples, but zero-shot CoT is a cheap approximation | "What is the difference between zero-shot and few-shot CoT?" | Confusing zero-shot CoT (prompt trick) with zero-shot learning (no training examples at all) |
| Self-consistency sampling: generate K independent CoT chains with temperature > 0, then take the majority-vote final answer; improves accuracy on reasoning benchmarks, especially when individual chains are noisy | "How does self-consistency work and when does it help?" (Answer: diversity of chains + voting reduces individual-chain errors; works best when the answer is categorical or short) | Thinking self-consistency is free — it multiplies inference cost by K and brings diminishing returns on tasks where all chains agree by step 2 |
| Reasoning tokens / scratchpad decoding in o1-class models: the model generates hidden chain-of-thought tokens (not shown to the user) before outputting the final answer; this is still next-token prediction, tool execution still requires a structured action token in the visible stream | "How do o1-style hidden reasoning tokens differ from normal CoT?" | Assuming hidden reasoning tokens bypass the tool-execution bridge — tools still need a visible action-format token that the host program can parse |
| Tree-of-Thoughts (Yao et al. 2023): the model explores multiple partial reasoning paths in a tree, evaluating nodes to decide which to expand — deliberate search over a reasoning space; Graph-of-Thoughts generalises this to DAG structures where thoughts can be merged | "What's the difference between Tree-of-Thoughts and Graph-of-Thoughts?" (one-liner: ToT uses a tree with backtracking; GoT uses a DAG so intermediate thoughts can be combined) | Conflating ToT with self-consistency — self-consistency samples independent linear chains, ToT builds and searches a tree |
| Process Reward Model (PRM): gives a scalar reward for each individual reasoning step, enabling step-level supervision during RLHF/training; Outcome Reward Model (ORM): gives a reward only for the final answer, ignoring intermediate steps | "Why use a PRM instead of an ORM?" (Answer: ORM can reward a correct answer reached via flawed reasoning; PRM enforces correctness of each step, which is important for multi-step math and code) | Saying PRM is always better — ORM is cheaper to label and sufficient when intermediate steps are not critical |
| Longer CoT increases cost (more tokens generated and therefore more inference compute + latency) and can introduce compounding errors on short factual tasks where the direct answer is already reliable | "When should you not use chain-of-thought?" | **Trap:** claiming "longer CoT always improves accuracy" — on simple factual retrieval tasks CoT can hurt accuracy by injecting unnecessary intermediate steps that introduce errors |

## Illustrations

![Chain-of-thought reasoning — direct vs CoT, thought-act-observe loop, reasoning structures, budget curve](img/CoT%20Reasoning.png)
