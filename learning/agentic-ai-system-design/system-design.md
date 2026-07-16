# Designing an Agentic AI Platform — System Design

> **Interview framing:** *"Design a platform that lets users create and deploy AI agents, register
> tools/MCP servers/skills, configure LLM providers, and run those agents safely in production —
> with health tracking, rollback, audit, guardrails, budgets, loop detection, and security
> controls."*

This is the uber/master document for the `agentic-ai-system-design/` track. It draws the
whole platform end-to-end and points into a dedicated doc for every component that earns its
own deep-dive. Read this file first; use the linked docs as the "45-minute, one-section-at-a-time"
reference layer.

**Core thesis** (memorize this — it answers 60% of follow-up questions on its own):

> The LLM is a *reasoning and planning component*, not the system of record, not the scheduler,
> not the policy engine, and not the authority that approves side effects. It proposes intent.
> A deterministic, observable, policy-governed distributed system around it decides what is
> actually allowed to happen, persists what happened, and can explain and undo it later.

This document — and this whole track — is scoped to **what is different about agent platforms
versus "just" distributed systems**. Queues, leader election, sharding, and load balancers are
assumed background knowledge; they get a sentence each. Loop detection, semantic evaluation,
trajectory replay, tool/MCP governance, and LLM-specific recovery patterns get full documents.

---

## 1 · What the platform actually needs to do

Restating the ask as capabilities, because interviewers grade on whether you covered the
full surface, not just the parts you find interesting:

| Capability the user asked for | Where it's designed |
|---|---|
| Create agents (define role, prompt, model, tools, policies) | [02 — Agent Lifecycle & Runtime](02-agent-lifecycle-and-runtime.md) |
| Deploy agents (versioning, rollout, rollback of the *agent definition*) | [02 — Agent Lifecycle & Runtime](02-agent-lifecycle-and-runtime.md) |
| Register tools, MCP servers, skills | [03 — Tool, MCP & Skill Registry](03-tool-mcp-and-skill-registry.md) |
| Configure LLMs and LLM providers | [04 — Model Gateway & LLM Providers](04-model-gateway-and-llm-providers.md) |
| Track agent health | [08 — Observability, Tracing & Health](08-observability-tracing-and-health.md) |
| Rollback agentic *actions* (not just deployments) | [10 — Recoverability, Rollbacks & Saga](10-recoverability-rollbacks-and-saga.md) |
| Audit agentic actions | [11 — Governance, Guardrails & Security](11-governance-guardrails-and-security.md) |
| Guardrails and policy enforcement | [11 — Governance, Guardrails & Security](11-governance-guardrails-and-security.md) |
| Token budgets | [06 — Non-Determinism, Loops & Termination](06-non-determinism-loops-and-termination.md) |
| Loop detection | [06 — Non-Determinism, Loops & Termination](06-non-determinism-loops-and-termination.md) |
| Security blocks (prompt injection, tool over-permission) | [11 — Governance, Guardrails & Security](11-governance-guardrails-and-security.md) |

Supporting depth, each with its own document because the content warrants it:

| Doc | Covers |
|---|---|
| [01 — Foundations of Agentic Systems](01-foundations-of-agentic-systems.md) | What makes a system "agentic," determinism spectrum, control/data plane split |
| [05 — State Management & Memory](05-state-management-and-memory.md) | Event sourcing, checkpoints, memory hierarchy, replay |
| [07 — Agent Evaluation Frameworks](07-agent-evaluation-frameworks.md) | Golden datasets, trajectory evaluation, LLM-as-judge, regression gates |
| [09 — Multi-Agent Communication Patterns](09-multi-agent-communication-patterns.md) | Supervisor, hierarchical, blackboard, contract-net, swarm |
| [12 — Production Scale & Capacity](12-production-scale-and-capacity.md) | Admission control, tenant quotas, hyperscale capacity planning |
| [13 — Semantic Kernel vs LangGraph](13-semantic-kernel-vs-langgraph.md) | Framework-level comparison with code, requested separately |

---

## 2 · Master architecture

```mermaid
flowchart TD
    Client[Users / Apps / Webhooks / Schedulers] --> Ingress[API Gateway + Event Router]
    Ingress --> Admission[Admission Control + Tenant Quotas + Risk Pre-check]
    Admission --> Control[Agent Control Plane]

    subgraph ControlPlane [Control Plane]
        Control --> Registry[Agent / Tool / MCP / Skill Registry]
        Control --> Scheduler[Scheduler + Lease Manager]
        Control --> PolicyEngine[Policy Engine + Budget Manager]
    end

    Scheduler --> Runtime[Runtime Plane: Execution Frames]
    Runtime --> ModelGW[Model Gateway]
    ModelGW --> Providers[(LLM Providers:\nOpenAI / Azure OpenAI / Anthropic / OSS)]

    Runtime --> ToolGW[Tool Gateway]
    ToolGW --> PolicyEngine
    ToolGW --> MCP[MCP Servers / Tools / Skills]
    ToolGW --> Enterprise[(Enterprise Systems:\nCRM, ticketing, repos, email)]

    Runtime --> StatePlane[State Plane: Event Log + Checkpoints + DAG]
    StatePlane --> MemoryPlane[Memory Plane: Session / Long-term / Vector]

    Runtime --> OTel[OpenTelemetry Collector]
    OTel --> Observability[Traces / Logs / Metrics / Health Dashboard]

    StatePlane --> EvalPlane[Evaluation Plane]
    EvalPlane --> ReleaseGate[Prompt / Model / Tool Release Gates]

    PolicyEngine --> AuditStore[(Immutable Audit Store)]
    Runtime --> SagaEngine[Saga / Compensation Engine]
    SagaEngine --> AuditStore

    classDef plane fill:#eef,stroke:#556,stroke-width:1px;
    class ControlPlane plane
```

**The one boundary that matters most:** everything to the *left* of the Tool Gateway is the
model proposing intent. Everything crossing the Tool Gateway into `Enterprise` / `MCP` is a
real side effect and **must** pass through `PolicyEngine` first. If you remember one line to
draw on a whiteboard, draw that arrow and label it "authority boundary."

---

## 3 · Plane-by-plane responsibility map

| Plane | Owns | Interview angle |
|---|---|---|
| Ingress | APIs, webhooks, scheduled triggers, identity context | How requests enter safely and consistently |
| Admission | Tenant quota, priority, risk pre-check, budget pre-check | How you prevent overload before execution starts |
| Control | Agent/tool/MCP/skill registry, scheduling, leases, policy hooks | How nondeterministic work is bounded by deterministic infrastructure |
| Runtime | Model calls, tool calls, step execution, isolation | How work runs without corrupting shared state |
| Model gateway | Provider routing, rate limits, fallback, cost metering | How you configure/swap LLMs without touching agent logic |
| Tool gateway | Policy-checked side effects, MCP/tool invocation | How model intent becomes an authorized action |
| State | Event log, checkpoints, DAG, tool records, approvals | How you replay, resume, audit, and debug execution |
| Memory | Thread memory, long-term stores, vector recall | How recall is useful without leaking tenants or stale facts |
| Observability | Traces, logs, metrics, health signals, cost attribution | How you explain what happened, and track agent health |
| Evaluation | Golden sets, trajectory checks, judges, regression gates | How you stop prompt/model/tool regressions before production |
| Recovery | Sagas, compensations, rollbacks, idempotency | How you undo or contain a partially-completed agentic action |
| Governance | Policy, HITL approval, audit, least privilege | How guardrails, security blocks, and audit actually work |

**Mnemonic:** `I‑A‑C‑R‑M‑T‑S‑Mem‑O‑E‑Rec‑G` is unwieldy — compress it to
**C‑R‑S‑M‑O‑E‑G**: **C**ontrol → **R**untime → **S**tate → **M**emory → **O**bservability →
**E**valuation → **G**overnance. If you can narrate that chain end to end, you can derive
almost every other answer in this track.

---

## 4 · End-to-end request lifecycle

```mermaid
sequenceDiagram
    participant U as Caller
    participant Adm as Admission
    participant Ctl as Control Plane
    participant Rt as Runtime
    participant MG as Model Gateway
    participant TG as Tool Gateway
    participant St as State Plane
    participant Ev as Evaluation
    participant Au as Audit Store

    U->>Adm: Create/invoke agent run
    Adm->>Adm: check tenant quota, risk, budget
    Adm->>Ctl: admitted
    Ctl->>Ctl: create execution id + lease
    Ctl->>Rt: schedule execution
    loop Agent step
        Rt->>MG: LLM call (reasoning/plan)
        MG-->>Rt: proposed action / next step
        Rt->>St: checkpoint (plan)
        Rt->>TG: proposed tool/MCP call
        TG->>TG: policy check + budget check + loop check
        alt allowed
            TG->>Au: log decision (allow)
            TG-->>Rt: tool result
        else needs approval
            TG->>U: HITL request
            U-->>TG: approve/deny/edit
        else denied
            TG->>Au: log decision (deny)
        end
        Rt->>St: checkpoint (observation)
    end
    Rt->>Ev: evaluate trajectory + output
    Rt->>St: final checkpoint
    St->>Au: immutable record complete
```

Every arrow that crosses into `Tool Gateway` or writes to `Audit Store` is a place a security
review will ask "what stops this from being abused?" Have an answer ready for each.

---

## 5 · Ten things to memorize first

1. The LLM proposes intent; deterministic infrastructure grants authority.
2. The control plane owns scheduling, leases, budgets, cancellation, and policy boundaries.
3. The runtime plane executes steps but is **never** the source of truth.
4. Every meaningful transition creates durable state and observable telemetry.
5. Every mutating tool/MCP call needs policy, idempotency, audit evidence, and recovery semantics.
6. Loops are bounded by step, token, time, cost, semantic, and structural controls — never by
   "the model should know when to stop."
7. Memory is layered: context window → checkpoint → session DAG → long-term store → vector
   memory → immutable audit log.
8. Evaluation covers output quality, trajectory validity, safety, latency, cost, and recovery
   behavior — not just "did it get the right answer."
9. Tracing links model calls, tool calls, checkpoints, policy decisions, and evals into one
   correlated execution story (OpenTelemetry spans, one root span per execution).
10. Enterprise scale needs tenant isolation, quotas, admission control, noisy-neighbor
    protection, and per-tenant cost attribution.

---

## 6 · Failure mode cheat sheet

| Failure | Likely root cause | Answer |
|---|---|---|
| Infinite loop | Weak termination criteria, ambiguous observations | Max steps + token/time/cost budgets + structural fingerprinting + semantic similarity + supervisor escalation ([06](06-non-determinism-loops-and-termination.md)) |
| Stale worker write | Lease expiry + duplicate runtime ownership | Fencing tokens; reject commits from expired leases ([02](02-agent-lifecycle-and-runtime.md)) |
| Lost progress after crash | Runtime memory treated as source of truth | Checkpoint after every meaningful transition; resume from durable state ([05](05-state-management-and-memory.md)) |
| Unsafe side effect | Model output treated as authority | Tool gateway + policy engine + HITL + least privilege + audit ([11](11-governance-guardrails-and-security.md)) |
| Bad production regression | Prompt/model/tool change never evaluated | Golden datasets + trajectory checks + LLM judge + regression gates ([07](07-agent-evaluation-frameworks.md)) |
| Unexplainable incident | Missing trace correlation across async boundaries | Propagate trace context across model/tool/state/policy/eval spans ([08](08-observability-tracing-and-health.md)) |
| Cross-tenant leakage | Shared memory/logs/vector index without isolation | Partition state/memory by tenant; ACL-aware retrieval; scrub telemetry ([05](05-state-management-and-memory.md), [12](12-production-scale-and-capacity.md)) |
| Partial completion across systems | Side effects without a compensation plan | Saga orchestration + idempotency keys + compensation contracts + escalation ([10](10-recoverability-rollbacks-and-saga.md)) |
| Rogue/incompatible tool version | Tool/MCP registered without contract or sandbox | Versioned registry, capability negotiation, sandboxing ([03](03-tool-mcp-and-skill-registry.md)) |
| Wrong/expensive model routed | No fallback or budget-aware routing | Model gateway with routing policy, fallback chain, per-tenant budget ([04](04-model-gateway-and-llm-providers.md)) |

---

## 7 · Interview answer template

When asked *"design an agentic AI platform,"* narrate in this order:

1. **State the invariant.** What must always be true even when the model is wrong? (→ Section 1's
   core thesis.)
2. **Separate the planes.** Control, runtime, model gateway, tool gateway, state, memory,
   observability, evaluation, recovery, governance.
3. **Name the risks.** Loop, duplicate execution, stale write, unsafe tool call, missing audit,
   noisy neighbor, cost blowout, prompt injection.
4. **Choose mechanisms.** Queues, leases + fencing tokens, checkpoints, event logs, policy
   gateways, budgets, HITL, evaluation gates, sagas.
5. **Discuss scale.** Tenant quotas, admission control, runtime pool sizing, model gateway
   throttling, state-store write rate, trace sampling.
6. **Close with tradeoffs.** Latency vs. reliability, autonomy vs. governance, memory quality vs.
   isolation, eval coverage vs. cost.

---

## 8 · Framework landscape (summary — full comparison in doc 13)

| Approach | Best for | Production gap you must fill yourself |
|---|---|---|
| Semantic Kernel-style orchestration | Developer-friendly app orchestration, enterprise .NET/Python integration | Needs external control/state/governance planes |
| LangGraph-style stateful graphs | Durable, explicit, resumable stateful workflows | Needs enterprise governance/capacity layered on top |
| AutoGen-style multi-agent | Event-driven, distributed, async multi-agent systems | Needs policy/eval/recovery hardening |
| Plain workflow engines | Deterministic business processes | Weak for open-ended reasoning |
| Actor systems | Distributed stateful concurrency | Need LLM-specific safety/eval bolted on |

No framework in this table *is* the platform. Each is (at most) a runtime/orchestration
library that lives inside the **Runtime** plane in Section 2's diagram. See
[13 — Semantic Kernel vs LangGraph](13-semantic-kernel-vs-langgraph.md) for a hands-on,
code-level comparison of the two most commonly asked about in interviews today.

---

## 9 · Scale variants

**Startup / 10-agent deployment:** one API service, one queue, one worker pool, one relational
database, a simple vector store, structured logs (not full OTel yet), hard step/token limits,
manual approval for every write tool, nightly golden-set evals.

**Enterprise / hyperscale:** everything in Section 2, plus tenant-aware admission control,
hot/warm/cold runtime pools, per-tenant model gateway throttling, sharded state stores, sampled
but safety-aware tracing, continuous evaluation with drift monitors, and a dedicated audit store
with retention policy.

Don't reach for the enterprise version on day one — name the startup version explicitly in an
interview; it signals judgment, not just knowledge.

---

## Quick Revision Notes

- The LLM proposes intent; infrastructure grants authority — repeat this until it's reflexive.
- Memorize the plane chain: **C‑R‑S‑M‑O‑E‑G**.
- Every mutating tool call = policy check + idempotency key + audit log entry + compensation plan.
- Loops die by budget (step/token/time/cost) + fingerprint (structural/semantic), never by hope.
- State ≠ chat history. State = input + plan + tool calls + observations + checkpoints +
  versions (prompt/model/tool/policy) + approvals + final output.
- One root span per execution; every child span (model/tool/state/policy/eval) shares its trace ID.
- Rollback in agent systems usually means **compensate**, not **undo** — most tool calls aren't
  transactional across system boundaries.
- Registries (agents, tools, MCP servers, skills) are versioned and capability-scoped — never
  "just a list of functions."
- No framework (LangGraph, Semantic Kernel, AutoGen) replaces the control/governance/eval planes;
  they live *inside* the runtime plane.

## Further Reading

- OpenTelemetry docs — <https://opentelemetry.io/docs/>
- OpenTelemetry traces/spans — <https://opentelemetry.io/docs/concepts/signals/traces/>
- LangGraph persistence (checkpointers vs. stores) — <https://docs.langchain.com/oss/python/langgraph/persistence>
- Semantic Kernel agent orchestration patterns — <https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/>
- AutoGen Core user guide — <https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html>
- Azure Architecture Center — Saga pattern — <https://learn.microsoft.com/en-us/azure/architecture/patterns/saga>
- Azure Architecture Center — Compensating Transaction pattern — <https://learn.microsoft.com/en-us/azure/architecture/patterns/compensating-transaction>
- ReAct: Synergizing Reasoning and Acting in Language Models — <https://arxiv.org/abs/2210.03629>
- LLM-as-a-Judge survey — <https://arxiv.org/abs/2411.15594>
- Model Context Protocol specification — <https://modelcontextprotocol.io/>
- LangChain human-in-the-loop docs — <https://docs.langchain.com/oss/python/langchain/human-in-the-loop>
- Martin Fowler — Event Sourcing — <https://martinfowler.com/eaaDev/EventSourcing.html>
