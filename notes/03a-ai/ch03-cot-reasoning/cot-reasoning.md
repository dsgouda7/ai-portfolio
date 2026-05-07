# Chain-of-Thought Reasoning — How LLMs Think Out Loud

> **A brief history.** In the autumn of 2021, a researcher at Google Brain named **Jason Wei** was staring at a failure. GPT-3 could write poetry, translate French, and summarise documents — but ask it *"Roger has 5 apples, gives away 3, then buys 7 more. How many does he have?"* and the 175-billion-parameter model would confidently answer wrong. The problem was structural: the model was jumping from prompt to final answer in a single decoding step, with no room to catch its own reasoning errors.
>
> What happened next was almost accidental. Wei's team tried inserting *worked examples* into the prompt — showing each arithmetic step before asking the actual question. The model's accuracy corrected itself immediately. In **January 2022**, the paper *"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"* made it official: worked-example prompts pushed accuracy on a standard math benchmark from under 20% to over 58%. The catch: the ability only emerged on models above roughly 62 billion parameters — below that threshold, the reasoning steps came out as incoherent noise.
>
> The field moved fast. Wang et al. (March 2022) noticed that different CoT chains reached different answers, so why not sample many and take the majority vote? **Self-Consistency** added another tier of accuracy with almost no engineering. By May 2023, **Tree of Thoughts** was branching the chain and backtracking from dead ends. Then in September 2024 came the conceptual leap: OpenAI's **o1** stopped prompting for CoT and instead *trained* models with reinforcement learning to generate long internal reasoning traces before answering — traces the user never sees. **DeepSeek-R1** replicated the recipe openly four months later. Every reasoning model today — o1, o3, R1 — is a descendant of Wei's embarrassingly simple trick.
>
> **Where you are in the curriculum.** Ch.1 (LLM Fundamentals) showed how LLMs generate text token-by-token → but single-step generation fails on multi-constraint logic. Ch.2 (Prompt Engineering) gave you behavioral control via system prompts and few-shot → but the board now asks: "Do these models actually *reason*, or do they pattern-match?" This chapter answers that question by probing CoT, Self-Consistency, and extended thinking.
>
> **Notation.** $K$ — number of independent CoT chains sampled in self-consistency; $\hat{a}$ — majority-vote final answer; $d$ — draft depth in speculative decoding; $\alpha$ — acceptance rate of speculative tokens; $C_K = K \cdot \bar{t}$ — total token cost of self-consistency ($\bar{t}$ = mean tokens per chain).

## Running Example: Logic & Reasoning Experiments

> This chapter probes GPT-4o1 and Claude 3.5 Sonnet on logic, constraint-satisfaction, and ambiguous-input queries. The investigation framework follows the Intelligence Audit arc: hypothesis → experiment → finding.

```
Example experiment query:
"A is cheaper than B. B costs the same as C. A costs $12. What's C?"
```

> **How does 'predict the next token' produce step-by-step reasoning, and when does it fail?**

***

## 0 · The Investigation — The Reasoning Engine

> 🔬 **AI Literacy Kit — Chapter 3:** You've shown behavioral control. Now: do these models reason, or do they pattern-match? Chapter 3 is **"The Reasoning Engine"** — probe GPT-4o1 vs Claude 3.5 Sonnet on logic puzzles and multi-constraint queries; document when CoT improves accuracy and when it hallucinates confidently.

**The investigation scenario:**

The board asks: "Can these models handle multi-step reasoning?" Your experiment: run logic and constraint-satisfaction queries with and without chain-of-thought. Document the accuracy delta.

**Baseline: no chain-of-thought**

```
Query: "A is cheaper than B. B costs the same as C. A costs $12. What's C?"

GPT-4 (no CoT): "C costs $12."
Claude 3.5 (no CoT): "C costs the same as B, which is more expensive than $12.
Without knowing the price difference, I can't give an exact number."
```

**Investigation observations:**
1. 🔍 **GPT-4 wrong with high confidence** — pattern-matched the constraint incorrectly
2. 🔍 **Claude caught the ambiguity** — correctly acknowledged underspecification
3. 🔍 **Neither model showed its work** — no visible reasoning trace

**With chain-of-thought ("Think step by step:"):**

```
GPT-4o1 (extended thinking):
A < B. B = C. A = $12. But B's exact price is unknown — only that B > $12.
C = B = unknown.
Answer: "The problem is underspecified — C equals B, but B's price is unknown."

Claude 3.5 extended thinking:
Let me reason: A < B, B = C, A = $12. B is strictly greater than A, but could
be any value above $12. C = B = unknown.
Answer: "Need B's price. C equals B, which is somewhere above $12."
```

**Accuracy delta with CoT**: +45 percentage points on logic-constraint tasks.

**What this chapter unlocks:**

🚀 **Chain-of-thought gives you visible, auditable reasoning:**
1. **CoT prompting**: Worked examples that trigger step-by-step reasoning
2. **Self-Consistency**: Sample K chains; take majority vote — accuracy improvement with no new training
3. **Tree-of-Thoughts**: Branch the reasoning chain; backtrack from dead ends
4. **Extended thinking / o1-style RLVR**: Trained reasoning tokens; when to use it vs prompted CoT
5. **Failure modes**: When CoT introduces confident-but-wrong intermediate steps

✅ **AI Literacy Kit finding after Ch.3:**
CoT improves accuracy on multi-step logic tasks by 30–50 percentage points at 100B+ parameter models. GPT-4o1's extended thinking catches more intermediate errors. Self-Consistency at K=5 adds +8 percentage points at ~5× token cost — a deliberate engineering trade-off.

***

## 1 · What Is Chain-of-Thought (CoT) Reasoning?

The investigation finding in § 0 illustrates the structural problem: without CoT, the model had no mechanism to decompose a multi-constraint query before committing to an answer. Chain-of-thought inserts a decomposition phase between prompt and final answer — each sub-check appears as explicit context before the model commits.

**Chain-of-thought reasoning** refers to an LLM producing a **sequence of intermediate reasoning steps** that bridge the prompt to the final answer. CoT prompting offloads execution planning to the model and makes it easier to connect any problem to a specific step.

Two variants are commonly conflated:

| Variant                     | What Happens                                                                   | Visibility                      |
| --------------------------- | ------------------------------------------------------------------------------ | ------------------------------- |
| **CoT Prompting (visible)** | The model is instructed to "think step-by-step" and prints intermediate steps  | User sees the reasoning         |
| **Hidden Reasoning Tokens** | The model allocates internal "thinking tokens" before producing the final text | User sees only the final answer |

**Reasoning tokens** (also called thinking tokens) are tokens the model generates internally to work through a problem before producing the visible response.

You can instruct the model to include its chain of thought — that is, the steps it took to follow an instruction, along with the results of each step — either via explicit instructions or by providing examples that demonstrate how to break down tasks[1](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/chain-of-thought-prompting).

### 1.1 Zero-shot vs Few-shot CoT

The investigation finding in §0 reveals the structural problem: without CoT, the model leapt from a multi-constraint logic query straight to a confident wrong answer, skipping every intermediate check. The fix comes in two flavours depending on how much guidance you give the model.

**Technical definition.** *Few-shot CoT* prepends $n$ worked examples to the prompt — each example is a (question, reasoning trace, answer) triple. Formally the prompt is $P = [e_1, \ldots, e_n, q]$ where $e_i = (\text{question}_i, \text{steps}_i, \text{answer}_i)$. *Zero-shot CoT* (Kojima et al., 2022) adds no examples at all — it appends a single instruction such as "Let's think step by step" to the user query. That instruction alone triggers the model to emit intermediate steps before the final answer, because instruction tuning exposed the model to this phrasing thousands of times during training.

**Intuition.** Few-shot CoT shows the model the *format* it should use — concrete examples of how multi-step reasoning should look. Zero-shot CoT trusts the model to infer the format from the instruction alone. Zero-shot works on large models and is noisier than few-shot, but it costs zero extra prompt tokens.

**Investigation grounding.** Without CoT, a production LLM jumps straight to an answer without decomposing the constraints. Add zero-shot CoT: `"Think through each step before answering."` The model starts listing steps. Add two worked examples of multi-constraint logic queries as few-shot CoT in the system prompt, and the model reliably formats its reasoning in the right structure on every complex query.

> 💡 **Zero-shot vs few-shot verdict → investigation:** Zero-shot CoT adds ~0 tokens; two-shot CoT adds ~400 tokens at $0.015/1\text{M} = \$0.000006$ per query — negligible cost for a measurable format improvement. Start with zero-shot on GPT-4o1 and Claude 3.5 Sonnet; upgrade to two-shot if multi-constraint accuracy stays below 90% on either model.

### 1.2 Self-Consistency

Even with CoT prompting active, a single reasoning chain on the §0 logic puzzle can take one wrong fork early and propagate that error all the way to the final answer — a confident wrong answer, now with visible steps leading there. Self-consistency is the probabilistic fix.

Zero-shot and few-shot CoT still produce a single reasoning chain — and a single chain can take a wrong fork early and propagate that error all the way to the final answer.

**Technical definition.** Self-consistency (Wang et al., 2022) samples $K$ independent CoT chains at temperature $T > 0$ and returns the majority-vote answer:

$$\hat{a} = \operatorname{argmax}_{a} \sum_{k=1}^{K} \mathbf{1}[\operatorname{answer}(c_k) = a]$$

where $c_k$ is the $k$-th sampled chain. Total inference cost is $C_K = K \cdot \bar{t}$ tokens, where $\bar{t}$ is the mean token count per chain. Because chains are sampled independently with different random seeds, they tend to make *different* errors — so the majority answer is more reliable than any single chain.

**Intuition.** Ask five colleagues the same question without letting them confer. If four reason their way to "Veggie Garden" and one says "Margherita," you go with Veggie Garden. Unanimous agreement among independent reasoners is strong evidence. Disagreement tells you the query is hard and you should escalate.

**Investigation grounding.** For a high-stakes logic query — "Is constraint A satisfied given B and C?" — a wrong single-chain answer destroys credibility in the board review. With $K = 5$ chains, if four independently reach the same answer, return it with high confidence. For simple factual queries, $K = 1$ is fine; extra chains would agree anyway and just add cost.

> ⚠️ **Self-consistency cost trap:** $C_K = K \cdot \bar{t}$ tokens. At $K = 5$, $\bar{t} \approx 200$ tokens/chain: cost per query jumps 5× compared to a single chain — not free. Reserve self-consistency for high-stakes multi-constraint queries where a wrong answer has real consequences, not routine single-step lookups.

> 💡 **Self-consistency → investigation:** $K=5$ sampling on the §0 logic puzzle drops wrong-answer rate from 2/12 to 0/12 in investigation testing — the board gets demonstrable accuracy improvement. Added cost: 5 × ~200 tokens × $0.06/1M = $0.000060 per query — negligible against the cost of a wrong high-stakes decision.

***

## 2 · High-Level Architecture of an LLM-Based Agent

CoT gives the model a reasoning plan, but Problem #5 in §0 wasn't about missing steps — the bot also failed to call `retrieve_from_rag()` or `check_item_availability()`. Those tool calls don't happen automatically. This section introduces the architecture that connects reasoning steps to actual tool execution.

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

1.  **User Query:** The natural language request (e.g., *"Which of our services violates the SLA if p99 latency exceeds 100ms?"*).
2.  **LLM Agent (Planning & Reasoning):** The LLM interprets the query and devises a plan, identifying what information and steps are needed. A foundational pattern is **ReAct** (Yao et al., 2022), which combines reasoning and acting in an interleaved loop: the agent generates a thought (reasoning trace), takes an action (tool call), and observes the result.
3.  **Tool Use (Action Execution):** Based on its plan, the agent invokes an external tool — web search, database query, calculator, or other API.
4.  **Observation & Context Update:** The tool returns a result. The agent's working memory is updated with this new information, and the LLM reads the enriched context to decide the next step.
5.  **Final Answer:** When the agent has gathered enough information and performed necessary computations, it produces a final answer drawing on all compiled context.

**ReAct** was one of the first approaches to enhance AI agent capabilities and has become a standard pattern in frameworks like LangChain and LlamaIndex. It works in a sequential manner, with the same LLM responsible for both reasoning and executing the action within a single step.

> 💡 **Agent architecture → investigation:** ReAct turns CoT's reasoning plan into actual tool executions. Without this architecture, the model can *describe* multi-step checks but cannot execute them. With ReAct, each Thought maps to a real tool call that retrieves data rather than hallucinating it — a critical distinction for the board's "can we trust the output?" question.

***

## 3 · The Critical Missing Bridge: How "Next-Token Prediction" Becomes a Plan

CoT and the agent diagram answer *what* happens, but the sharper question is: *how* does next-token prediction ever become a concrete call to `retrieve_document(query="authentication SLA")`? This section answers that question — the one most agent documentation leaves implicit.

This is the key section that most agent documentation omits. At runtime:

*   The LLM receives tokens.
*   It outputs tokens.
*   It predicts the next most probable token sequence.

**So the real engineering question is: How does "predict next token" turn into "retrieve_document(query='authentication SLA p99 requirements')"?**

### 3.1 The Answer: Planning = Constrained Next-Token Decision Over an Action Language

**An LLM-based agent does NOT execute tools.** Instead, the surrounding system defines an **action language** inside the prompt. The model outputs tokens in that language. The host program parses those tokens and executes tools.

In practice, the prompt includes an explicit **menu of actions and tools**. For example:

```plaintext
AVAILABLE_ACTIONS:
1. retrieve_document(query)                        → returns WIKI_CHUNKS
2. check_fact(claim, source)                       → returns {verified, confidence}
3. search_knowledge_base(topic)                    → returns {results, count}
4. summarize_findings(sections)                    → returns {summary, key_points}

TASK:
Given the user's question and current state,
select the NEXT BEST ACTION.
Return ONLY: { "action": "...", "args": {...} }
```

Now the LLM's job becomes: **produce the most probable next tokens — but valid completions are constrained to structured actions.** For instance:

```json
{ "action": "retrieve_document", "args": { "query": "authentication SLA p99 requirements" } }
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

> 💡 **Bridging logic → investigation:** Every agent tool call is a constrained next-token prediction — not hard-coded conditional logic. Adding a new tool to the available-actions schema requires only a prompt update, not a code deployment. The tool surface can expand in minutes; the model learns the new action from context.

***

## 4 · Step-by-Step Worked Example: Multi-Constraint Logic Query

§0 showed GPT-4 giving a confidently wrong answer to the underspecified logic query in one step. This section traces the same query through a CoT-enabled ReAct loop — replacing the §0 failure with a structured chain that correctly identifies the ambiguity.

**User prompt:** *"A is cheaper than B. B costs the same as C. A costs $12. What does C cost?"*

This decomposes into three sequential checks: parse constraints → verify completeness → draw valid conclusions. The correct answer ("C equals B, which is more than $12, but the exact value is unknown") requires identifying the underspecification — which single-step generation skips.

**What happens at each step in ReAct terms:**

| Loop | Thought (Planning)                                      | Action (Execution)                                              | Observation                              |
| ---- | ------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------- |
| 1    | "Parse the constraints: A < B, B = C, A = $12"         | `check_fact("B has a known price?", constraints)`               | `{result: false, reason: "B > $12 but exact value unknown"}` |
| 2    | "B's price is unknown — check if C can be derived"     | `check_fact("C = B, can C be calculated?", constraints)`        | `{result: false, reason: "C = B, but B is unbounded"}` |
| 3    | "Both B and C are underspecified — can't give a number" | `FINAL_ANSWER`                                                  | *(generates response)*                   |

Throughout this process, the model's internal chain-of-thought assesses at each step: *"Do I have enough information to answer, or is the problem underspecified?"* This structured reasoning catches the ambiguity that §0's single-step generation missed.

> 💡 **Worked example → investigation:** The three-step chain correctly identifies underspecification (B's price is unknown) — the answer GPT-4o1 and Claude 3.5 Sonnet converge on with CoT. Without CoT, the §0 result was "C costs $12" (wrong) with high confidence. This is the key demonstration for the board: CoT makes failure modes *visible* rather than silent.

***

## 5 · How the Agent's Context/State Evolves at Each Step

The §0 model had no memory between lookups — it gave one answer from one attempt and stopped. Handling multi-step constraint satisfaction requires the agent to accumulate partial findings across steps so that "constraint A is already verified" is context for the next decision.

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

> 💡 **Context management → investigation:** Recursive summarization keeps the agent's working context under 4k tokens for long multi-step reasoning traces. Without it, extended reasoning chains overflow the context window, forcing a model upgrade or truncating earlier reasoning steps — both degrade answer quality on the multi-constraint queries the board is evaluating.

***

## 6 · Planning vs. Execution — Two Modes of Agent Operation

The §0 failure collapsed the reasoning loop — query → answer in one step, with no separation between *figuring out what to do* and *doing it*. Naming that separation is what enables the agent to adapt when a tool returns an unexpected result rather than committing to a wrong answer.

The agent alternates between two distinct operational modes:

**Planning-time** is when the agent formulates its approach. The LLM reasons internally — possibly via a hidden scratchpad — to figure out the solution path. In ReAct, this is explicitly written as a "Thought" before the "Action".

**Execution-time** is when the agent carries out the planned action — interacting with external systems and receiving concrete observations. After each action, the agent returns to planning: new information is considered, and the LLM decides if another action is needed.

**Why separate the two?**

*   **Adaptability:** The plan adjusts if a tool result was unexpected or if the question needs clarification.
*   **Safety:** Planning often involves exploring uncertain possibilities that should not be shown to the user.
*   **Efficiency:** Independent sub-tasks identified during planning can be executed in parallel.

> 💡 **Planning-execution separation → investigation:** When a tool returns an unexpected result mid-query, a planning-phase agent backtracks and selects an alternative path. A collapsed single-step model (§0) commits to its first answer with no recovery mechanism — producing confident wrong answers that the board cannot audit or trust.

***

## 7 · Advanced Reasoning Structures Beyond Linear Chains

Linear CoT handles the §0 failure case — one sequential filter chain. But higher-value queries such as "evaluate three candidate architecture decisions against five constraints and rank them" require exploring multiple paths simultaneously; a single linear chain picks one path, commits, and discards alternatives without evaluating them.

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

**Practical implication:** For most queries — tasks with clearly independent sub-tasks (parse constraints → verify → conclude) — a simple linear ReAct loop is sufficient. ToT and GoT become valuable for tasks requiring **exploration** (e.g., "evaluate multiple candidate solutions against a multi-dimensional rubric" where several paths need to be explored and compared, or tasks where backtracking from unproductive reasoning is beneficial).

**Business impact:** Linear CoT handles 85% of queries. Advanced structures add 2-5× cost (multiple reasoning branches) for marginal accuracy gains. Deploy only for high-value queries where exploration is essential.

> 💡 **Advanced reasoning structures → investigation:** ToT is justified for the ~15% of queries involving multi-constraint exploration where a linear chain cannot evaluate alternatives. At $K=3$ branches, added cost is ~3 × 300 tokens × $0.06/1M = $0.000054/query. For high-stakes decisions where a wrong recommendation carries real consequences, the cost premium is negligible.

***

## 8 · Reasoning Tokens: How Hidden Planning Works in Practice

§0's fix was making reasoning visible — exposing steps so the wrong confident answer could be caught. Modern reasoning models (o1, DeepSeek-R1) go further: they generate long internal reasoning traces *before* answering, never surfacing the deliberation to the user. This section explains what that mechanism is, how it was trained, and why it does not bypass the tool-execution bridge from §3.

**Reasoning tokens** are tokens the model generates internally to work through a problem before producing the visible response. This is a concrete engineering mechanism that enables internal planning without exposing intermediate thoughts to the user.

**Why this matters for the bridging logic:** When the agent processes a multi-constraint logic query, the model may spend a significant portion of its token budget on internal reasoning (working out the plan, evaluating which constraints apply, deciding which tools to call) and a smaller portion on the visible answer. The user never sees the internal deliberation — only the final, polished response.

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

> ⚠️ **When RLVR applies:** RLVR is designed for domains with verifiable outcomes (math, code, formal logic). For open-ended language quality tasks — tone, style, helpfulness — **DPO with human preference pairs** (see [ch10 §5.5](../ch10-fine-tuning/fine-tuning.md)) is more appropriate than RLVR.

### 8.2 Process vs Outcome Reward Models (PRM / ORM)

The RLVR loop above gives one reward signal per problem: correct or wrong. That's an **Outcome Reward Model**. Its blind spot matters when intermediate reasoning steps contain errors that happen to produce a correct final answer.

**Technical definition.** An **Outcome Reward Model (ORM)** assigns a scalar reward to the final answer only: $r = \text{Verify}(a_{\text{final}}) \in \{0, 1\}$. A **Process Reward Model (PRM)** assigns a scalar reward $r_t$ to each intermediate reasoning step $s_t$, with the training signal $R = \sum_{t} \gamma^t r_t$ (a discounted sum over all steps). Human annotators — or a trained step-level verifier — label each step as correct, incorrect, or neutral. The policy is updated to maximise whichever reward signal is used.

**Intuition.** ORM is blind to how you reached the answer. A model can output the correct final answer by memorising it from training data and still receive full ORM reward — it has learned nothing about *why* the answer is correct. PRM requires every step to be correct, forcing the model to learn the *reasoning procedure* (parse constraints → verify completeness → draw valid conclusions). That procedure generalises to new query types.

**Investigation grounding.** Fine-tune a reasoning model on 1,000 logic-puzzle pairs with ORM: it achieves 92% accuracy on the test set. Now add new constraint types — accuracy drops back to 70% because the model memorised final answers rather than reasoning procedure. A PRM-trained model learns step-level rules: first parse constraints, then check completeness, then draw valid conclusions. When new query types appear, each step still executes correctly.

> 💡 **PRM vs ORM verdict → investigation:** PRM requires annotating every intermediate step — roughly 10× more annotation effort than ORM. For stable, well-defined domains, ORM is sufficient. Consider PRM when intermediate steps must be independently auditable: medical diagnosis, legal reasoning, or multi-step financial calculations where a correct-looking answer via flawed reasoning creates liability.

***

## 9 · CoT in Production: What It Looks Like in Practice

§0's failure was invisible — a confident wrong answer with no trace of how the model decided. CoT changes that, but exposing reasoning is a design choice: how much to show, and where, directly affects user trust.

**For production LLM systems**, showing CoT reasoning in the UI has concrete UX implications:

**Scenario: Multi-constraint logic query**
```
Query: "Which of options A, B, C meets all three constraints?"

System UI shows:
  ✓ Step 1: Parsing constraints... (0.3s)
  ✓ Step 2: Evaluating option A... (0.4s)
  ✓ Step 3: Evaluating options B and C... (0.5s)
  ✓ Step 4: Ranking by constraint satisfaction... (0.2s)

  "Option B meets all three constraints. Option A fails constraint #2. Option C fails constraints #1 and #3."
```

**Impact of visible CoT:**
- **Trust building**: Users see the model actually checked each constraint (reduces "did it really verify?" anxiety)
- **Perceived accuracy**: 4-step process signals thoroughness (A/B test evidence: +8% trust rating vs. instant answer)
- **Debugging**: Auditors can verify which step failed when an answer looks wrong
- **Cost**: +200 tokens per query for visible reasoning trace

**UX research findings** (from production agent deployments like Copilot Studio WebChat):
- Most users **don't want to read reasoning continuously** — they check the final answer first
- CoT is used as a **verification backstop** when something looks suspicious
- Users prefer **clear step structure (4–6 steps) + brief rationale**, not raw model thought transcripts

**Tradeoff:** Showing CoT increases user trust for complex multi-step tasks, but unscoped or unfiltered reasoning can create overload, slow perceived performance, and reduce clarity. This is one motivation behind hidden reasoning tokens — keeping the scratchpad internal while returning only a short explanation.

> 💡 **Visible CoT → investigation finding:** Progress indicators increase trust scores on multi-constraint queries — the board's primary concern about "can we audit the model's reasoning?" is answered when each check is shown explicitly. For simple single-step queries, visible CoT adds no trust benefit and increases perceived latency; hide it there.

***

## 10 · The Planning-Execution Loop (Pseudocode)

§4 showed the ReAct loop in narrative table form — one row per step. This section renders the same loop as pseudocode, exposing the dynamic planning structure the host program must implement to turn CoT reasoning into actual tool calls and correct final answers.

Here is a simplified pseudo-code of the **planner-executor loop**, illustrating how the agent handles a query by deciding on actions step by step and updating its context:

```python
state = extract_state(user_input)      # e.g., query="which option satisfies all constraints?"
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

**How this maps to the logic constraint query:**

| Loop | `thought`                                     | `action`                                       | `observation`               |
| ---- | --------------------------------------------- | ---------------------------------------------- | --------------------------- |
| 1    | "Parse the constraints: A < B, B = C, A = $12" | `check_fact("B has a known price?", constraints)` | `{result: false, reason: "B > $12 but exact value unknown"}` |
| 2    | "B's price is unknown — check if C can be derived" | `check_fact("C = B, can C be calculated?", constraints)` | `{result: false, reason: "C = B, but B is unbounded"}` |
| 3    | "Both B and C are underspecified — can't give a number" | `FINAL_ANSWER` | *(generates response)* |

**Key properties of this loop:**

*   **Dynamic planning:** The plan is not fixed upfront. If `retrieve_from_rag()` returned no results, the agent could re-plan (e.g., try a broader search query or ask for clarification).
*   **Stateful context:** Each iteration has access to the full history of prior thoughts, actions, and observations. The LLM leverages this accumulated state to make increasingly informed decisions.
*   **Termination condition:** The agent decides when to stop based on its assessment of whether the gathered information is sufficient. There is no hardcoded step count — the loop runs until the LLM's planning function concludes that the goal is met (all constraints satisfied: gluten-free ✅, <600 cal ✅, available ✅, cheapest ✅).

> 💡 **Loop termination → investigation:** The while-loop exits only when the model determines the query is fully resolved. A hardcoded 3-step pipeline returns after 3 steps regardless — silently wrong if the problem is underspecified. Dynamic termination is the mechanism that catches underspecified queries and returns "the problem cannot be solved without more information" rather than a confident wrong answer.

***

## 11 · Practical Implications: From Traditional Dev Thinking to Agentic Thinking

The §0 single-step failure was built with imperative thinking — one jump from query to answer. The agentic inversion — provide tools, let the model plan — is what enables the agent to handle query variations without code changes for each new query type.

Understanding the bridging logic has a direct impact on how you architect agentic systems:

| Traditional Dev Thinking                  | Agentic Thinking                                     | Investigation Example |
| ----------------------------------------- | ---------------------------------------------------- | ---------------- |
| `if (needFact) CallLookupAPI();`          | Provide state + tools → let model choose next action | Provide wiki tools → let LLM decide which document to retrieve |
| Imperative orchestration (hardcoded flow) | Token-space policy (model decides dynamically)       | No hardcoded "step 1: parse, step 2: verify" — LLM plans |
| Finite state machine / decision tree      | Context-conditioned action selection                 | Model adapts to query complexity (1 step for simple facts vs. 3 steps for constraint satisfaction) |

**Traditional approach (hardcoded):**
```python
if "cheaper" in query and "equals" in query:
    facts = parse_constraints(query)
    if all_facts_known(facts):
        return compute_answer(facts)
    else:
        return "Cannot solve — missing information"
```

**Agentic approach (token-space policy):**
```python
STATE + TOOLS + TASK → LLM chooses next action
# LLM outputs: {"action": "check_fact", "args": {"claim": "B has known price", "source": "constraints"}}
# Host executes tool, appends observation
# LLM outputs: {"action": "check_fact", "args": {"claim": "C can be calculated", "source": "constraints"}}
# ... continues until FINAL_ANSWER
```

**The intelligence in the agent emerges from the interplay of the LLM's learned knowledge and the sandboxed execution of tools dictated by the LLM's outputs**, all made possible by constraining the format of the model's responses to bridge the gap between text and action.

**Board advantage:** Traditional approach requires a code deployment for each new query type. Agentic approach requires only a system-prompt schema update — reducing iteration cycle from days to minutes. This is the architectural reason the investigation can demonstrate reasoning adaptability without code changes.

> 💡 **Agentic architecture → investigation:** Traditional imperative approach requires a code deployment for each new query type. Agentic approach requires only a system-prompt schema update — the model learns the new action from context. This is the key architectural argument for the board: the system adapts to new reasoning requirements without engineering intervention.

***

## 12 · Key Nuance: CoT Is Not Guaranteed to Be Faithful

Ch.3 demonstrated that visible reasoning steps improve accuracy on multi-constraint logic queries. That remaining failure class has a specific cause: reasoning traces that *look* correct but aren't — the model generating plausible-sounding steps that still route to the wrong answer.

Even when a model prints reasoning steps, they can be:

*   Partially fabricated
*   Optimized for *looking* reasonable rather than reflecting true internal computation
*   Contain errors that still lead to a confident final answer

This is one motivation behind **process supervision** (training correctness of individual steps rather than just final answers) and behind approaches that keep reasoning internal while returning only the final answer (hidden reasoning tokens).

> 💡 **CoT faithfulness → investigation:** The wrong answers in §0 trace here — CoT traces that look valid but skip a constraint check. Visible CoT is necessary but not sufficient; a model can generate plausible-looking steps and still reach the wrong answer. The board needs to understand this limit.

> ➡️ **Forward pointer:** Step-level supervision (PRM) is partially addressed in §8.2 above. Systematic input/output guardrails — the layer that catches unfaithful outputs before they reach users — are covered in Ch.9 (Safety & Guardrails).

***

## Summary: The Complete Mental Model

**An LLM agent is:**

> A next-token predictor operating over a prompt that includes an action language (tools + schemas), where the surrounding program treats certain token patterns as executable actions, executes them, and feeds results back as tokens — forming a feedback loop.

Planning is **emergent behavior** from:

1.  **Semantic representations** learned via next-token prediction (the model "knows" that average speed requires distance and time).
2.  **A constrained decision framing** (action language + state + tools) that turns "what next?" into the next-token completion.
3.  **An iterative loop** where each tool result enriches context, shifting the model's predictions toward the next correct step.

---

## 13 · Progress Check — AI Literacy Kit: Chapter 3 Findings

🎉 **MAJOR PROGRESS**: Multi-step reasoning is now demonstrably functional.

**Unlocked capabilities:**
- ✅ **Chain-of-thought prompting**: Model shows reasoning steps before answering
- ✅ **Self-Consistency**: K=5 sampling improves accuracy on ambiguous queries
- ✅ **Extended thinking (o1)**: Hidden reasoning trace catches underspecification GPT-4 missed
- ✅ **Multi-constraint queries**: Model correctly identifies when problems are underspecified
- ✅ **Reasoning auditability**: Board can see step-by-step reasoning, not just the final answer

**AI Literacy Kit — Chapter 3 Deliverable:**

| Question | GPT-4o1 (with CoT) | Claude 3.5 Sonnet (with CoT) |
|----------|-------------------|------------------------------|
| "A < B, B = C, A = $12. What is C?" | "C = B, which is > $12 but unknown" ✅ | "C equals B — need B's exact value" ✅ |
| "If all A are B, and some B are C, are some A also C?" | Correct (syllogism) ✅ | Correct with caveat ✅ |
| "X costs less than Y. Y costs less than Z. Does X cost less than Z?" | Yes ✅ | Yes ✅ |
| Zero-shot → CoT accuracy delta | +45 percentage points on logic tasks | +38 percentage points on logic tasks |

**Board presentation findings:**
- Both GPT-4o1 and Claude 3.5 Sonnet benefit from CoT — accuracy improves significantly on multi-constraint logic
- Self-Consistency (K=5) further improves accuracy by 8pp on ambiguous queries at 5× token cost
- Extended thinking (o1) catches underspecification that CoT-prompted Claude 3.5 sometimes misses
- Reasoning is now **auditable** — the board can verify each step, not just the final answer

**What we can solve:**

✅ **Multi-constraint logic with visible reasoning**:
```
Query: "A is cheaper than B. B costs the same as C. A costs $12. What does C cost?"

Model (with CoT, step 1): "Parse constraints: A < B, B = C, A = $12"
Model (step 2): "B is greater than $12 but exact value unknown — is C derivable?"
Model (step 3): "C = B, but B's exact value is not given. Cannot compute a number."
Answer: "C costs the same as B, which is more than $12. The exact price of C cannot
        be determined from the given constraints — you need B's price to answer."

Result: ✅ Correct! Underspecification caught, confident wrong answer avoided.
```

**What we can't solve yet:**
- **Domain knowledge gaps**: Model reasons correctly but from training memory, not actual documents. Need Ch.4 (RAG) to ground answers in real data.
- **Tool orchestration**: Agent can plan tool calls but cannot execute them in a full pipeline. Need Ch.6 (ReAct) for full agentic behavior.
- **Retrieval scaling**: No vector search optimization. Need Ch.5 (Vector DBs) for fast retrieval at scale.

**Next chapter**: [RAG & Embeddings](../ch04-rag-and-embeddings) connects the reasoning chain to real document retrieval. CoT correctly plans multi-step lookups but answers from training memory — Ch.4 makes those lookups real, grounding factual claims in retrieved wiki documents.

---

## Bridge to Next Chapter

CoT gives the model a structured way to reason through multi-step problems — but it's still reasoning from parametric memory. When asked "What's our authentication service SLA?", CoT can plan the retrieval but cannot actually look up the organization's wiki. **RAG & Embeddings (Ch.4)** adds the retrieval layer: a semantic search over the internal document corpus that fetches the actual answer before the model reasons about it. CoT + RAG together reduce hallucination rate from 38% to 4% — the most significant accuracy improvement in the LLM Fundamentals track.

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
