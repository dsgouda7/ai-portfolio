# Ch.1 — Message Formats & Shared Context

> **The story.** When **OpenAI's ChatCompletions API** launched in **March 2023**, it shipped a deceptively boring data structure — a JSON list of `{role, content}` messages with three roles (system, user, assistant). Within months, that envelope became the *de facto* lingua franca for the entire industry: Anthropic's Messages API, Google's Gemini API, every open-source serving framework (vLLM, llama.cpp, Ollama) all settled on the same shape. **Function calling** (OpenAI, June 2023) added a fourth role and turned messages into a structured action language. **JSON mode** and **structured outputs** (OpenAI, August 2024) made the envelope rigorously typed. Every multi-agent protocol in this track — [MCP](../ch02_mcp), [A2A](../ch03_a2a), [Event-driven agents](../ch04_event_driven_agents) — either reuses this envelope verbatim or wraps it in transport metadata. Get this chapter right and every later chapter is just a different choreography over the same data structure.
>
> **Where you are in the curriculum.** This is the first chapter of the multi-agent track and it intentionally starts at the wire format, not at the orchestration layer. **Central question:** how do agents actually exchange information — what is physically in the message envelope, and how is shared context managed when the accumulated conversation history exceeds a single context window? The running scenario is **OrderFlow**, a B2B purchase-order automation platform.
**Notation.** `role` = message sender identity (`system` | `user` | `assistant` | `tool`). `turn` = one exchange (human message + model response). `ctx` = context window occupancy in tokens. `tool_call` / `tool_result` = structured function invocation and return value inside the chat envelope. `history` = the accumulated list of prior turns injected into each new request.
<!-- notation: key variables defined here -->

---

## § 0 · The Challenge — Where We Are

> **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**What we know so far**:
This is Chapter 1 — the starting point. We're beginning from manual baseline.
**Manual baseline**: 3 procurement specialists processing 50 POs/day @ $420k/year labor cost. 36-hour median latency, 5% error rate.
But we can't automate this yet — no agent architecture exists.

**What's blocking us**:

### The Blocking Question This Chapter Solves

**"How do multiple agents exchange information without exceeding context limits?"**

A single-agent system hits the 8k token context window after 3 supplier negotiations. The agent starts hallucinating supplier names from earlier orders. Need to decompose work across specialized agents without losing information.

### What We Unlock in This Chapter

- Understand the OpenAI message envelope (`role`, `content`, `tool_calls`) as the lingua franca of multi-agent systems
- Three handoff strategies: full history passthrough (audit-focused), structured payloads (production), blackboard (async)
- Context budget management: Split 24k token budget across 8 specialized agents (3k each) without overflow

### Progress on the 8 Constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 THROUGHPUT | **BLOCKED** | Single-agent monolith handles 10 POs/day before context overflow |
| #2 LATENCY | **BLOCKED** | 36 hours median (manual baseline) |
| #3 ACCURACY | **BLOCKED** | 5% error rate (agent hallucinates suppliers after context fills) |
| #4 SCALABILITY | **FOUNDATION LAID** | Can decompose single 16k-token agent → 8 agents (3k each) without context overflow |
| #5 RELIABILITY | **BLOCKED** | No retry logic, no graceful degradation |
| #6 AUDITABILITY | **PARTIAL** | Full history passthrough enables audit, but no tracing infrastructure |
| #7 OBSERVABILITY | **FOUNDATION LAID** | Message structure enables future tracing (but no tooling yet) |
| #8 DEPLOYABILITY | **BLOCKED** | Monolithic agent, no versioning or rollback |

**What's still blocking**: Decomposed agents into 8 specialists, but they can't access ERP, pricing APIs, or email. Need 8 × 20 = 160 custom integrations. *(Ch.2 — MCP solves this.)*

---

## 1 · Core Idea
### The OpenAI Message Envelope

Every major agentic framework — LangChain, Semantic Kernel, AutoGen, LangGraph — serialises inter-agent communication as a list of message objects that conform to (or translate to) the OpenAI Chat Completions format. Understanding this schema makes every framework legible.

```python
# A message object — the atomic unit of agent communication
{
 "role": "user" | "assistant" | "tool" | "system",
 "content": "...", # string or list of content parts
 "tool_calls": [...], # present when role is "assistant" and agent invoked a tool
 "tool_call_id": "..." # present when role is "tool" (the response to a tool_call)
}
```

**Role semantics in a multi-agent context:**

| Role | Who produces it | What it carries |
|------|----------------|-----------------|
| `system` | Orchestrator | The receiving agent's persona, constraints, and task definition |
| `user` | Calling agent / orchestrator | The task input — what this agent is asked to do |
| `assistant` | The agent itself | The agent's reasoning and/or tool invocation decisions |
| `tool` | Tool execution environment | The result of a tool call, keyed by `tool_call_id` |

When an orchestrator calls a sub-agent, the sub-agent's *entire prior conversation* (its own `system`, `user`, `assistant`, `tool` messages) is what it needs in its own context. The question is: what does the *calling* agent receive back?

---

### The Three Handoff Strategies

When Agent A calls Agent B and B finishes its work, what does A receive, and in what form?

#### Strategy 1 — Full History Passthrough

Agent B returns its entire message list to Agent A. Agent A appends it to its own context.

```python
# Agent B returns everything
handoff = {
 "status": "completed",
 "messages": [
 {"role": "system", "content": "You are the pricing specialist..."},
 {"role": "user", "content": "Negotiate supplier price for order #4812..."},
 {"role": "assistant", "tool_calls": [{"id": "tc_01", "function": {"name": "get_quote"}}]},
 {"role": "tool", "tool_call_id": "tc_01", "content": "{\"unit_price\": 14.20}"},
 {"role": "assistant", "content": "Agreed price: $14.20 per unit, 500 units."}
 ]
}
```

**When to use:** Auditing-critical systems (financial, medical) where every reasoning step must be traceable.
**Cost:** Token count accumulates multiplicatively. 3 agent hops × 2,000 tokens each = 6,000 tokens of overhead before the next agent has processed anything.

#### Strategy 2 — Structured Handoff Payload

Agent B returns only a structured summary of its result.

```python
# Agent B returns just what the next agent needs
handoff = {
 "status": "completed",
 "result": {
 "agreed_price_usd": 14.20,
 "quantity": 500,
 "delivery_days": 7,
 "supplier_id": "SUP-88412"
 }
}
```

**When to use:** Production pipelines where context budget matters more than full auditability. The result is deterministic and machine-readable.
**Cost:** Minimal. The orchestrator pays only for the output, not the reasoning trace.

#### Strategy 3 — Shared Key-Value Store (Blackboard)

Neither Agent A nor Agent B passes data directly. Both read from and write to a shared store keyed by the task ID.

```python
# Agent B writes its result to the shared store
store.set(f"order:{task_id}:pricing", {
 "agreed_price_usd": 14.20,
 "quantity": 500,
 "delivery_days": 7,
 "supplier_id": "SUP-88412"
})

# Agent A reads directly from the store without waiting for B to "return" anything
pricing = store.get(f"order:{task_id}:pricing")
```

**When to use:** Pipelines with more than 3 agents where conversation threading becomes unmanageable, or async pipelines where agents are not sequentially ordered.
**Tradeoff:** Decouples agents (neither needs to know the other's interface) at the cost of introducing a central store that becomes a consistency and latency concern.

*(See Ch.5 — Shared Memory for the full treatment.)*

---

### Context Budget Management

A context window is a finite resource. In a multi-agent chain, it is depleted by:
- The receiving agent's system prompt (typically 500–2,000 tokens)
- The task payload passed from the orchestrator
- All in-flight reasoning (assistant messages)
- All tool call/response pairs
- Any history passed in from prior agents

**The rule of thumb:** Reserve at least 20% of the context window for the model's output generation. If your accumulation is forecast to exceed 80% before the agent finishes, truncate aggressively from the *oldest* messages — not the most recent — keeping at minimum the system prompt and the current task.

```python
def trim_to_budget(messages, max_tokens, reserve_for_output=0.2):
 budget = int(max_tokens * (1 - reserve_for_output))
 # Always keep system message (messages[0]) and trim from oldest user/assistant pairs
 system = [m for m in messages if m["role"] == "system"]
 rest = [m for m in messages if m["role"] != "system"]
 while count_tokens(system + rest) > budget and len(rest) > 1:
 rest.pop(0) # drop oldest non-system message
 return system + rest
```

---

## 2 · Running Example

OrderFlow's first version used a single agent to handle a PO end-to-end. By order #12, the context filled up before the supplier negotiation was complete, and the model started hallucinating supplier names from earlier orders.

The fix: decompose into a pipeline of 8 specialized agents, each with a bounded context. The orchestrator calls them in sequence, passing a structured handoff payload (Strategy 2) between them. The full negotiation trace is stored in a log DB (queryable for audit) but not passed as a context message.

```
Orchestrator
 │
 ├─▶ Intake Agent → validates request, extracts requirements
 ├─▶ Pricing Agent → researches supplier options, gets quotes
 ├─▶ Negotiation Agent → negotiates terms with selected supplier
 ├─▶ Legal Agent → validates contract terms against policies
 ├─▶ Finance Agent → confirms budget availability, approver
 ├─▶ Drafting Agent → generates final PO document
 ├─▶ Sending Agent → delivers PO to supplier via email
 └─▶ Reconciliation Agent → tracks delivery, closes order
```

Each agent sees only what it needs (structured payload from previous agent). No agent exceeds 40% of its context window (3k tokens used of 8k available) on the largest real orders.

---


## 3 · How It Works — Step by Step

**PO #2024-1847 message flow with structured handoff:**

```
Step 1: Orchestrator → IntakeAgent
Message: {"role": "user", "content": "Parse PO request: 10 standing desks, budget $8,000..."}
IntakeAgent processes → returns structured payload:
{
 "po_id": "2024-1847",
 "items": [{"name": "standing desk", "quantity": 10}],
 "budget_usd": 8000,
 "preferred_suppliers": ["TechFurnish", "OfficeDepot"]
}

Step 2: Orchestrator → PricingAgent
Message: {"role": "user", "content": <IntakeAgent result>}
PricingAgent queries suppliers → returns:
{
 "quotes": [
 {"supplier": "TechFurnish", "unit_price_usd": 789, "total": 7890},
 {"supplier": "OfficeDepot", "unit_price_usd": 842, "total": 8420}
 ],
 "recommendation": "TechFurnish"
}

Step 3: Orchestrator → NegotiationAgent
Message: {"role": "user", "content": <PricingAgent result>}
NegotiationAgent negotiates → returns:
{
 "agreed_price_usd": 749,
 "discount_pct": 5,
 "delivery_days": 21,
 "supplier_id": "TechFurnish"
}

Step 4–8: Continue pipeline (Legal, Finance, Drafting, Sending, Reconciliation)
```

**Key mechanism**: Each agent receives only the structured output from the previous agent, not the full conversation history. Total token cost: ~15k tokens across 8 agents (vs. 64k+ with full history passthrough).


## 4 · The Key Diagrams

### Message Envelope Anatomy

```
OpenAI Chat Completions Message Format:
┌─────────────────────────────────────────────────────┐
│ role: "system" | "user" | "assistant" | "tool" │
│ content: string or null │
│ tool_calls: [{id, type, function: {name, args}}] │ (assistant only)
│ tool_call_id: string │ (tool only)
│ name: string (optional agent identifier) │
└─────────────────────────────────────────────────────┘
```

### Three Handoff Strategies — Token Cost Comparison

```
Agent Chain: A → B → C (each agent accumulates 2k tokens internally)

Strategy 1 — Full History Passthrough:
 Agent A: 2k tokens
 Agent B: 2k (own) + 2k (A's history) = 4k tokens
 Agent C: 2k (own) + 2k (A) + 2k (B) = 6k tokens
 Total: 12k tokens (grows O(n²))

Strategy 2 — Structured Handoff:
 Agent A: 2k tokens → returns 200-token payload
 Agent B: 2k tokens → returns 200-token payload
 Agent C: 2k tokens → returns 200-token payload
 Total: 6k tokens (grows O(n))

Strategy 3 — Shared Store:
 Each agent: 2k tokens (reads/writes to external store)
 Total: 6k tokens + store I/O latency
```

### Context Window Management

```
8k Token Context Window Allocation:
┌────────────────────────────────────────┐ ← 8,192 tokens (hard limit)
│ System prompt: 800 tokens │
│ Task payload: 500 tokens │
│ Agent reasoning: 2,200 tokens │
│ Tool calls + responses: 1,500 tokens │
│ ──────────────────────────────────── │
│ Used: 5,000 tokens (61%) │ Safe (< 80%)
│ ──────────────────────────────────── │
│ Reserved for output: 1,600 tokens │ (20% buffer)
│ ──────────────────────────────────── │
│ Headroom: 1,592 tokens │
└────────────────────────────────────────┘

If usage exceeds 80% → trim oldest messages, keep system + recent context
```


## 5 · Production Considerations

**Context budget per agent** — 3k–4k tokens (50% of 8k limit) allows headroom for complex reasoning without overflow. Lower = faster but less capable; higher = more capable but risks overflow on long tasks.

**Handoff strategy selection**:
- **Full history** (Strategy 1): Use only when audit compliance requires reconstructing full reasoning chain (financial, medical, legal domains). Cost: O(n²) token growth.
- **Structured payload** (Strategy 2): Default for production. Minimal token cost, deterministic output schema. Requires upfront schema design.
- **Shared store** (Strategy 3): Use when >5 agents need access to same state, or when async event-driven (Ch.4). Adds latency and consistency concerns.

**Trimming policy** — Preserve system message + most recent 3–5 turns. Never trim current task context. Log trimmed messages to external store for audit.

**Message serialization** — Use JSON for structured payloads. Validate schemas at agent boundaries to catch type errors early. Consider Protocol Buffers for high-throughput systems (reduces token count by ~30%).

**Error handling** — If agent exceeds context budget mid-task, checkpoint current state, trim history, and resume. Don't fail the entire PO — graceful degradation is better than total failure.


## 6 · What Can Go Wrong

**1. Context overflow mid-task** — Agent exceeds 8k token limit before completing negotiation. Symptoms: incomplete responses, hallucinated data, abrupt termination.
 - **Fix**: Implement proactive trimming at 80% usage threshold. Log trimmed content to external store for audit reconstruction.

**2. Lost context after trimming** — Agent forgets critical approval threshold after oldest messages are dropped. Next agent makes wrong decision.
 - **Fix**: Never trim the system prompt or current task payload. Preserve "pinned" messages (approval rules, PO ID, budget) using message metadata flags.

**3. Schema mismatch between agents** — PricingAgent returns `unit_cost` but NegotiationAgent expects `unit_price_usd`. Pipeline breaks with KeyError.
 - **Fix**: Define shared schema types (Pydantic models) at orchestrator level. Validate payloads at every handoff boundary. Fail fast with clear error messages.

**4. Token cost explosion with full history** — 5-agent pipeline passes full history → 40k tokens → $0.08/PO → unsustainable at scale.
 - **Fix**: Switch to structured handoff (Strategy 2) for production. Reserve full history for audit-only traces stored externally, not passed in context.

**5. Audit trail gaps with structured handoff** — Compliance audit asks "Why did NegotiationAgent accept this price?" Structured payload has result but not reasoning.
 - **Fix**: Log every agent's full message history to database with PO correlation ID. Query logs for audit reconstruction. Context passes only results; logs preserve reasoning.

## 8 · Progress Check — What We Can Solve Now

```mermaid
graph LR
 Ch1["Ch.1\nMessage Formats"]:::current
 Ch2["Ch.2\nMCP"]:::upcoming
 Ch3["Ch.3\nA2A"]:::upcoming
 Ch4["Ch.4\nEvent-Driven"]:::upcoming
 Ch5["Ch.5\nShared Memory"]:::upcoming
 Ch6["Ch.6\nTrust & Sandboxing"]:::upcoming
 Ch7["Ch.7\nAgent Frameworks"]:::upcoming
 Ch1 --> Ch2 --> Ch3 --> Ch4 --> Ch5 --> Ch6 --> Ch7
 classDef done fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
 classDef current fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
 classDef upcoming fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```
**Unlocked capabilities**:
- **Multi-agent decomposition**: Split single monolithic agent into 8 specialized agents (Intake, Pricing, Negotiation, Legal, Finance, Drafting, Sending, Reconciliation)
- **Context budget management**: Each agent operates within 3k–4k token budget (50% of 8k limit), no overflow on largest POs
- **Structured message passing**: Defined OpenAI-compatible message envelope (`role`, `content`, `tool_calls`) as lingua franca for agent communication
- **Three handoff strategies**: Full history (audit), structured payload (production), shared store (async) — choose based on requirements
**Still can't solve**:
- **Tool integration**: 8 agents × 20 data sources = 160 custom integrations required (ERP, pricing APIs, supplier email, legal templates) → **Need Ch.2 (MCP) for protocol-based tool access**
- **Agent coordination**: Orchestrator still calls agents sequentially → 35-minute critical path + queue delays = 36-hour SLA → **Need Ch.4 (Event-Driven) for async parallelism**
- **Throughput bottleneck**: Sequential processing limits to 10 POs/day (vs. 1,000 target) → **Need Ch.4 for concurrency**

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 THROUGHPUT | **BLOCKED** | 10 POs/day (context overflow eliminated, but still sequential) — Target: 1,000 POs/day |
| #2 LATENCY | **BLOCKED** | 36 hours median (manual baseline unchanged) — Target: <4 hours |
| #3 ACCURACY | **IMPROVED** | 5% → **3.8% error rate** (hallucination eliminated via context management) — Target: <2% |
| #4 SCALABILITY | **FOUNDATION COMPLETE** | 8 agents, 3k tokens each, no context overflow — Target: 10 agents/PO |
| #5 RELIABILITY | **BLOCKED** | No retry logic, no graceful degradation — Target: >99.9% uptime |
| #6 AUDITABILITY | **FOUNDATION LAID** | Structured payloads logged (but no correlation IDs or distributed tracing) — Target: Full traceability |
| #7 OBSERVABILITY | **FOUNDATION LAID** | Message schema enables future tracing infrastructure — Target: Real-time monitoring |
| #8 DEPLOYABILITY | **BLOCKED** | Monolithic orchestrator, no containerization — Target: Zero-downtime updates |

**What we can solve now**:
**Complex PO processing without context overflow**:
```
Before Ch.1:
Single agent hits 8k token limit after 3 supplier negotiations
→ PO #2024-1847 stuck, agent hallucinates supplier names

After Ch.1:
8 specialized agents, each <4k tokens
→ PO #2024-1847 completes full lifecycle without overflow

Result: Context overflow eliminated (Constraint #4 foundation complete)
```

**Real-world status**: We can now decompose agent workflows to avoid context limits, but we can't yet connect agents to real data sources (ERP, pricing APIs) or process POs concurrently.

**Next up:** [Ch.2 — Model Context Protocol (MCP)](../ch02_mcp) gives us **standardized tool integration** — collapses 8 agents × 20 systems = 160 integrations to 8 + 20 = 28 components.

---

## 7 · The Math

### Context Budget Allocation

Let $C$ be the total context window (tokens), $n$ the number of agents in a pipeline, and $h_i$ the per-agent history budget. Safe operation requires:

$$\sum_{i=1}^{n} h_i + |\text{task payload}| \leq C$$

For the **structured-handoff** strategy, each agent $i$ receives only a bounded payload $p_i$. Total token cost across the chain:

$$T_\text{structured} = \sum_{i=1}^{n} \bigl(|s_i| + |p_i| + |r_i|\bigr)$$

For the **full-history** strategy, each agent receives all prior history:

$$T_\text{full-history} = \sum_{i=1}^{n} \sum_{j=1}^{i} \bigl(|s_j| + |p_j| + |r_j|\bigr)$$

The full-history cost grows $O(n^2)$ with chain depth; structured-handoff grows $O(n)$.

### Token Trimming Policy

When history accumulates beyond budget $B$, preserve system message $m_0$ and the most recent $k$ turns:

$$\text{keep} = \{m_0\} \cup \{m_{|H|-k}, \ldots, m_{|H|-1}\}$$

where $k$ is chosen such that $\sum_{i \in \text{keep}} |m_i| \leq B$.

| Symbol | Meaning |
|--------|---------|
| $C$ | Total context window (tokens) |
| $n$ | Number of agents in the pipeline |
| $p_i$ | Input payload size for agent $i$ |
| $r_i$ | Response size from agent $i$ |
| $B$ | Trimming budget threshold |
| $k$ | Number of recent turns preserved after trimming |



---

## 9 · Bridge to Chapter 2

Ch.1 established **structured message passing** between agents — the OpenAI message envelope (`role`, `content`, `tool_calls`) as the lingua franca, three handoff strategies (full history, structured payload, shared store), and context budget management to avoid overflow. We decomposed single monolithic agent into 8 specialists, each operating within 3k-4k token budget.

But the 8 agents can't access real data yet. PricingAgent needs supplier APIs. InventoryAgent needs ERP. NegotiationAgent needs email. Building custom integrations = **8 agents × 20 systems = 160 bespoke adapters** — unmaintainable.

Ch.2 ([Model Context Protocol](../ch02_mcp)) solves the **N×M integration explosion** → **standardized tool protocol** collapses 160 integrations to **8 + 20 = 28 components**. Any agent can call any tool (ERP, pricing APIs, email) through one shared protocol. MCP wraps this chapter's message envelope in a transport layer (stdio/HTTP) and defines a tool discovery handshake. **Expected outcome**: Agents can finally query real data sources without custom integration code.

---

## Where This Reappears

| Chapter | How message format concepts appear |
|---------|------------------------------------|
| **Ch.2 — MCP** | Every MCP tool call is a structured message over stdio/HTTP transport — same `role/content/tool_calls` envelope |
| **Ch.3 — A2A** | A2A wraps structured handoff payloads in an `AgentCard` envelope; the inner payload uses this same format |
| **Ch.4 — Event-Driven Agents** | Async messages carry the same structured payload schema; correlation IDs enable multi-turn conversation across async boundaries |
| **Ch.5 — Shared Memory** | The blackboard stores structured payloads by correlation ID; agents read shared context instead of receiving it in messages |
| **Ch.7 — Agent Frameworks** | LangGraph state wraps the message list; LangSmith traces individual messages per step |
| **AI track — ReAct** | The single-agent ReAct loop is the same message envelope (Thought/Action = `assistant`, Observation = `tool`); multi-agent is multiple such loops with structured handoffs |

---

## Interview Questions
`tool_calls` is present on `role: assistant` messages and contains `id`, `type`, `function.name`, and `function.arguments`. The `id` (e.g. `"tc_01"`) is echoed in the subsequent `role: tool` message as `tool_call_id`. In a multi-agent trace, this pairing is what lets you reconstruct which tool response corresponds to which invocation — essential for debugging non-deterministic agent behaviour.

**Q: When would you prefer a structured handoff payload over passing full conversation history?**
When the downstream agent does not need to understand *how* the upstream agent arrived at the answer — only *what* the answer is. Full history is for auditability; structured payload is for efficiency. The cost of full history grows linearly with chain length and can exceed the context limit of the receiving agent.

**Q: A user says their agentic pipeline "gets confused" on long-running tasks. What is the first thing you check?**
Context accumulation. Run the pipeline with token counting on each agent invocation and graph the context length over time. "Gets confused" in an otherwise correct agent almost always means the oldest context has been truncated (by a framework's default limit) and the model is missing a critical earlier instruction.

**Q: What is the risk of passing the full message history from Agent B back to Agent A?**
Token cost compounds multiplicatively: a 4-agent chain where each agent accumulates 2,000 tokens of internal reasoning delivers 8,000 tokens of overhead to its caller — before the caller has started its own reasoning. In a deep chain, the accumulated handoff history can fill the entire context window before the final agent has read its own task. Additionally, if any agent injects sensitive data (API keys, PII from tool results) the full history propagates that data to every downstream agent whether it needs it or not.

---

## Notebook

`notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) implements:
1. A minimal two-agent pipeline using raw OpenAI message lists (no framework)
2. Token counting and trimming for each strategy
3. A side-by-side comparison of the three handoff strategies on an OrderFlow scenario: total tokens sent, reconstruction fidelity, time to implement

---

## Prerequisites

- [AI / ReActAndSemanticKernel](../.03-ai/ch06_react_and_semantic_kernel/react-and-semantic-kernel.md) — single-agent ReAct loop before decomposing it
- [AI / PromptEngineering](../.03-ai/ch02_prompt_engineering/prompt-engineering.md) — system prompt construction for agent roles

## Next

→ [Ch.2 — Model Context Protocol (MCP)](../ch02_mcp) — how to standardise tool access so any agent can call any tool without bespoke integration

## Illustrations

![Message formats - anatomy, payload styles, context growth, serialization](img/Message%20Formats.png)
