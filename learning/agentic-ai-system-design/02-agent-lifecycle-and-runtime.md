# Agent Lifecycle & Runtime

> Part of the [Agentic AI Platform — System Design](system-design.md) track. Builds on
> [01 — Foundations of Agentic Systems](01-foundations-of-agentic-systems.md) — read that first
> for the workflow-vs-agent distinction and the control/data plane split this doc assumes.
> This doc answers the platform's headline capability: **"create and deploy agents."**

## 1 · Problem statement

Runtime design is where every other plane's guarantees either hold up or fall apart under load.
The scheduler can hand out perfectly correct leases, the policy engine can make perfectly
correct decisions, and the state plane can define a perfectly correct checkpoint schema — but if
the runtime that actually executes an agent's steps can't survive a process crash mid-reasoning,
can't recover from a slow or failed tool call, and can't be interrupted safely by a policy
decision, none of that matters. Runtime design determines four things directly: **latency**
(how fast can a step execute), **cost** (how much idle/compute capacity you're paying for),
**concurrency** (how many executions run safely at once), and **recovery** (what happens when
something crashes mid-run). Agent runtimes are harder than typical stateless-service runtimes
because they must support **long-running, stateful reasoning** — a single execution can span
multiple model calls, multiple tool calls, and real wall-clock time waiting on external systems
— while surviving process crashes, model latency spikes, tool failures, and mid-execution policy
interruptions (an approval request, a budget cutoff, a cancellation).

## 2 · The agent definition & creation lifecycle

"Creating an agent" is not writing a system prompt in a text box and calling it done — on a
production platform, an **agent definition is a versioned artifact**, treated with the same
rigor as a deployable service. Concretely, an agent definition bundles:

| Component | What it is | Owned in |
|---|---|---|
| System prompt / instructions template | The agent's role, constraints, and reasoning guidance, parameterized (not a single static string) | This doc |
| Model selection | Model + provider + generation params (temperature, max tokens, etc.) | [04 — Model Gateway & LLM Providers](04-model-gateway-and-llm-providers.md) |
| Tool allow-list | The specific registered tools/MCP servers/skills this agent version may call — never "all tools" | [03 — Tool, MCP & Skill Registry](03-tool-mcp-and-skill-registry.md) |
| Policy bindings | Which policies (approval rules, rate limits, data-access rules) apply to this agent's actions | [11 — Governance, Guardrails & Security](11-governance-guardrails-and-security.md) |
| Budget defaults | Default token / time / cost / step limits for a run of this agent | [06 — Non-Determinism, Loops & Termination](06-non-determinism-loops-and-termination.md) |
| Evaluation gates | The golden-set / trajectory checks this version must pass before promotion | [07 — Agent Evaluation Frameworks](07-agent-evaluation-frameworks.md) |

Treating this bundle as a single versioned artifact (call it `agent_version_id`) is the key
design decision in this section: it means every run of an agent can point at an exact,
immutable, auditable configuration, and "what changed between the version that worked and the
version that broke" is a diff, not an investigation.

```mermaid
flowchart LR
    Author[Author agent definition\nprompt + model + tools + policy + budget]
    Validate["Validate: schema check,\ntool allow-list resolves,\npolicy bindings resolve"]
    Stage["Stage: run evaluation gates\n(golden set + trajectory checks)"]
    Canary["Canary: shadow or low-%\nproduction traffic"]
    Production["Production: full traffic,\nversion pointer updated"]
    Rollback["Rollback: pointer swap to\nprior agent_version_id"]

    Author --> Validate --> Stage --> Canary --> Production
    Canary -- fails gates --> Rollback
    Production -- regression detected --> Rollback
    Rollback --> Stage
```

Treat this exactly like a deployment pipeline for a versioned config artifact, because that's
what it is. The single most important consequence of this framing: **rolling back a bad agent
version is a pointer swap, not a data migration.** The registry holds an `agent_name ->
active_agent_version_id` mapping; rollback overwrites that pointer to the last-known-good
version. The two things you must not forget to do alongside the pointer swap:

- **Invalidate caches** that may have memoized the old version's config (prompt templates,
  resolved tool schemas, policy bindings).
- **Invalidate or re-key in-flight sessions** that were created under the old version, so a
  long-running conversation doesn't silently continue executing against a version that was just
  pulled for a regression.

## 3 · Deployment and versioning

Because an agent definition is just a versioned config artifact, standard progressive-delivery
techniques apply almost unmodified — the only agent-specific addition is that "correctness" for
an agent version isn't just "does it start" but "does it behave acceptably across a
representative sample of inputs," which is why **shadow-mode evaluation** matters more here than
for a typical service rollout.

- **Blue-green or canary rollout:** run the new `agent_version_id` alongside the current
  production version, routing a small percentage of real traffic (or a duplicated shadow copy of
  it) to the new version before full cutover.
- **Shadow-mode evaluation:** replay real production inputs against the candidate version
  *without* letting its tool calls actually execute side effects (or executing them against a
  sandboxed/staging tool environment), and compare its trajectories/outputs against golden-set
  expectations and against the current production version's behavior. This is how you catch
  "the new prompt made the agent call the refund tool more often" before it happens for real.
- **Instant rollback of the agent version pointer** — as covered above — is deliberately kept
  separate from rolling back an in-flight *action*. Rolling back a deployment means "stop running
  the new config." Rolling back an action means "undo or compensate for a specific side effect a
  specific execution already committed" (a sent email, a charged payment, a modified record).
  Those are different problems solved by different mechanisms — see
  [10 — Recoverability, Rollbacks & Saga](10-recoverability-rollbacks-and-saga.md) for
  action-level rollback and compensation. Conflating the two is a common design mistake: an
  instant version-pointer rollback does **not** retroactively undo anything a prior execution
  already did in the real world.

#### Internals: How Canary, Blue-Green, and Shadow Actually Route Traffic

All three answer the same question — "how do you find out a new agent version is bad before it
hurts every user?" — but they differ in exactly what traffic sees the candidate version and what
happens to its output:

- **Canary:** a small, deliberately chosen percentage of *real* traffic is routed to the
  candidate `agent_version_id` while the rest continues to the current production version. The
  candidate's responses are real — users on that slice actually get them — so you compare
  health-score/eval metrics (from [07 — Agent Evaluation Frameworks](07-agent-evaluation-frameworks.md)
  and [08 — Observability, Tracing & Health](08-observability-tracing-and-health.md)) between the
  two live populations before deciding to ramp the percentage up or roll back.
- **Blue-green:** both the current version ("blue") and the new version ("green") are fully
  deployed and live simultaneously, each provisioned for 100% of production capacity. Cutover is
  a routing-table flip — all traffic moves from blue to green at once — which makes rollback
  equally instant (flip the router back). The cost is running two fully-provisioned copies of the
  runtime fleet during the transition window.
- **Shadow:** the candidate version receives a *mirrored copy* of real production traffic, but its
  output is never returned to the user — it's discarded or logged and compared offline against the
  production version's actual response and against golden-set expectations. This is the only one
  of the three with zero user-facing risk, but it's also the only one with no real feedback loop
  on side effects: a shadowed write-tool call must be executed against a sandboxed/staging
  environment (see Section 3, above) rather than the real system, or you're not actually testing
  what the candidate would have done in production.

> **Trade-offs — canary vs. blue-green vs. shadow**
>
> | | Canary | Blue-Green | Shadow |
> |---|---|---|---|
> | User-facing risk | Small — limited to the canary % | All-or-nothing at cutover, but reversible instantly | None — outputs never reach users |
> | Infra cost during transition | Low — candidate runs at a fraction of capacity | High — both versions at full capacity (~2x) | Medium — candidate processes full mirrored volume but serves nothing |
> | Feedback speed | Fast — real user outcomes on live traffic | Fast — but only after full cutover | Slow — only offline comparison, no live user signal |
> | Rollback speed | Fast — reduce % back to zero | Instant — flip the router back | N/A — candidate was never serving traffic |

```mermaid
flowchart TD
    subgraph Canary["Canary"]
        C1[Incoming traffic] -->|majority %| C2[Production version]
        C1 -->|small %| C3[Candidate version]
        C3 --> C4["Compare health-score / eval metrics\nbefore ramping %"]
    end

    subgraph BlueGreen["Blue-Green"]
        B1[Incoming traffic] --> B2{Router}
        B2 -->|pre-cutover| B3["Blue: current version\n(live)"]
        B2 -.->|instant flip| B4["Green: new version\n(live, 2x capacity during transition)"]
    end

    subgraph Shadow["Shadow"]
        S1[Incoming traffic] --> S2["Production version\n(response returned to user)"]
        S1 -.->|mirrored copy| S3["Candidate version\n(output discarded/logged)"]
        S3 --> S4["Compared offline —\nno live feedback loop"]
    end
```

## 4 · Push vs. pull execution

How work reaches a runtime instance shapes latency and overload behavior differently:

```mermaid
flowchart TD
    subgraph Push["Push model"]
        Webhook[Webhook / API call] --> Gateway[Gateway]
        Gateway --> RuntimeP[Runtime instance]
    end

    subgraph Pull["Pull model"]
        EventP[Event] --> Queue[Queue]
        Queue -.->|worker claims job| WorkerP[Worker pool]
    end
```

- **Push (Webhook → Gateway → Runtime):** minimizes routing delay — the gateway hands work
  straight to a runtime instance. The cost is that a traffic burst can overload runtimes directly;
  there's no natural buffer absorbing spikes, so you need aggressive autoscaling or admission
  control in front of it (see the uber doc's Admission plane).
- **Pull (Event → Queue → Worker claims job):** the queue provides natural backpressure and
  fairness — workers pull at their own sustainable rate, and a burst just makes the queue longer
  rather than overloading a runtime. The cost is added queue latency, since work waits for a
  worker to become free rather than being dispatched immediately.

Most production agent platforms end up hybrid: pull-based queueing for the bulk of executions
(reliability, fairness, backpressure), with a push-based or pre-warmed **hot pool** carved out for
latency-sensitive paths (covered in the Runtime Models table below).

#### Internals: How Push and Pull Actually Move Work to a Runtime

**Push**, mechanically: the control plane (gateway/scheduler) maintains a live view of which
runtime instances exist and how much capacity each has left, and on each incoming request it picks
a target and opens a connection (or sends a message) directly to that instance. This means the
control plane owns the hard problem itself — it must track capacity in near-real-time, decide what
to do when every known runtime is full, and implement backpressure explicitly, because nothing in
a push architecture provides it for free.

**Pull**, mechanically: the control plane does nothing more than append work to a queue. Each
worker independently polls for the next item, claims it via a visibility-timeout / lease mechanism
(the same idea as the runtime leasing in Section 5, just applied to a queue message instead of an
execution), and only that worker processes it. Backpressure falls out as a side effect of queue
depth rather than something the control plane has to compute — autoscaling can key directly off
queue depth, a metric that already exists, instead of a bespoke capacity-tracking system. The cost
is added latency (a worker has to notice and claim the item) and the need for redelivery
semantics: if a worker claims an item and crashes before finishing, the item's visibility timeout
must expire and return it to the queue, or the work is silently lost.

> **Trade-offs — push vs. pull**
>
> | | Push | Pull |
> |---|---|---|
> | Latency | Lower — work is dispatched directly | Higher — bounded by poll interval / claim latency |
> | Backpressure | Control plane must implement it explicitly | Free — queue depth *is* the backpressure signal |
> | Autoscaling signal | Requires custom capacity tracking | Trivial — scale workers on queue depth |
> | Crash safety | Control plane must detect a dead target itself | Visibility-timeout / lease redelivery handles it structurally |
> | Best fit | Latency-sensitive interactive paths | Bulk/async/background execution (the reliable default) |

## 5 · Runtime leasing and fencing tokens

An execution must be owned by exactly one runtime instance at a time — otherwise two instances
can both believe they're responsible for the same execution and both write conflicting state.
Leasing with fencing tokens is the standard mechanism (this is not agent-specific distributed
systems theory, but the consequences of getting it wrong *are* agent-specific: a duplicate
runtime can mean a tool call — e.g., "send the refund" — executes twice).

```mermaid
sequenceDiagram
    participant Sch as Scheduler
    participant LM as Lease Manager
    participant Rt as Runtime
    participant SS as State Store

    Sch->>LM: request lease(execution_id)
    LM->>LM: create lease with fencing token N
    LM-->>Sch: lease granted, token=N
    Sch->>Rt: assign execution (token=N)
    loop while executing
        Rt->>LM: heartbeat(execution_id, token=N)
        Rt->>SS: checkpoint(execution_id, token=N)
        SS->>SS: accept only if token >= last-seen token
    end
    alt heartbeat missed
        LM->>LM: expire lease, mint new token N+1
        LM->>Sch: lease expired, reassign
        Sch->>Rt: assign to new runtime (token=N+1)
        Note over Rt,SS: original runtime's writes with\ntoken N are now rejected as stale
    end
```

The properties this buys you, stated explicitly because interviewers want to hear the reasoning,
not just the diagram:

- **A runtime may commit state only with a valid fencing token.** The state store compares the
  incoming token against the highest token it has already accepted for that execution and
  rejects anything lower/stale.
- **Lease expiry must never allow a stale worker to overwrite newer state.** If a runtime is
  paused (GC pause, network partition, slow tool call) long enough for its lease to expire, a new
  runtime takes over with a higher token. When the original runtime wakes up and tries to write,
  its write is rejected — even though *it* doesn't know it lost ownership.
- **Heartbeats are independent of model latency.** A runtime must be able to heartbeat "I'm still
  alive and making progress" on a separate channel from "the model call I'm waiting on has
  returned," or a single slow model response looks indistinguishable from a dead runtime.
- **Cancellation must be durable.** A cancellation request (from a user, a policy engine, or a
  budget manager) must be recorded in durable state, not just sent as an in-memory signal to a
  runtime process — otherwise a lease reassignment after a crash "loses" the cancellation and
  the new runtime happily resumes work that was supposed to have stopped.

#### Internals: The Fencing-Token Mechanism, Concretely

This is the same technique Chubby and ZooKeeper use to stop a "zombie" lock holder from
corrupting shared state after it's lost its lease — Martin Kleppmann's write-up on distributed
locking (Further Reading) is the canonical reference if asked where this idea comes from. Reduced
to its essentials:

1. The lease authority (Lease Manager) hands out leases with a **monotonically increasing
   integer** attached — the fencing token. Values only ever increase; they're never reused or
   reset.
2. Every write a runtime sends to a shared resource (the state store) must carry its current
   fencing token alongside the write.
3. The resource being mutated — not the lease manager, and not the runtime itself — enforces the
   rule: **reject any write whose token is lower than the highest token already accepted for that
   execution.**

Concrete sequence:

- Runtime A is granted the lease for `execution_42` with fencing token `7`.
- Runtime A pauses (GC pause / network partition) longer than the lease TTL.
- Lease Manager expires A's lease and grants a new lease to Runtime B, with token `8`.
- Runtime B checkpoints with token `8` — the state store accepts it and records `8` as the
  highest token seen for `execution_42`.
- Runtime A wakes up, still believing it holds the lease, and checkpoints with token `7`.
- The state store compares `7 < 8` and **rejects the write** — even though Runtime A has no way
  of knowing, from its own local view, that it ever lost ownership.

The comparison happening *at the resource*, not at the lease manager, is what makes this work: a
network partition can prevent Runtime A from ever hearing "your lease expired," but it cannot
prevent the state store from refusing a stale write.

## 6 · Runtime models

Different parts of a platform legitimately want different runtime shapes — this table is the one
to reproduce when asked "how would you actually host these agents?"

| Runtime model | Best for | Primary tradeoff |
|---|---|---|
| Ephemeral / serverless | Spiky, short-lived tasks | Cold-start latency; limited ability to hold large in-memory runtime state across steps |
| Long-lived worker | Sustained throughput workloads | Idle cost when underutilized; noisy-neighbor risk between executions sharing a process |
| Hot pool (pre-warmed) | Latency-sensitive interactive agents | Pool-sizing complexity — under-provisioning reintroduces cold starts, over-provisioning wastes cost |
| Actor runtime | Stateful concurrency, one actor per execution/session | Operational complexity (actor placement, rebalancing, supervision trees) |
| Queue worker | Reliable background/async execution | Added queue latency between enqueue and pickup |

No single row is "the" answer — production platforms typically mix at least three: a queue-worker
backbone for reliability, a hot pool for latency-sensitive paths, and ephemeral/serverless for
bursty low-frequency workloads.

## 7 · Failure modes

| Failure mode | Cause | Mitigation |
|---|---|---|
| Duplicate execution after lease expiry | Original runtime resumes writing after losing the lease | Fencing tokens rejected at the state store (Section 5) |
| Stale worker writes | Same root cause as above, framed from the state-store side | State store enforces monotonic token acceptance per execution |
| Worker starvation | Queue fairness not enforced across tenants/agents | Per-tenant queue partitioning or weighted fair queueing (see [12 — Production Scale & Capacity](12-production-scale-and-capacity.md)) |
| Cold-start latency | Ephemeral/serverless runtime spun up on demand | Hot pool for latency-sensitive agents; pre-warming heuristics |
| Runtime memory leaks | Long-lived worker accumulates state across many executions | Bounded worker lifetime/recycling; per-execution isolation |
| External tool timeout | Tool/MCP server slow or unresponsive | Timeout + retry policy at the tool gateway, independent of runtime heartbeat |
| Checkpoint missing after model call | Runtime crashes between receiving a model response and persisting it | Checkpoint immediately after every model/tool call, before acting on the result (see [05 — State Management & Memory](05-state-management-and-memory.md)) |

## 8 · Enterprise vs. startup recommendation

**Enterprise recommendation:** hybrid runtime strategy — queue-backed pull execution as the
reliable backbone, hot pools carved out for latency-sensitive interactive paths, ephemeral
runtimes absorbing burst/overflow, and strict lease fencing enforced everywhere so that no
runtime model is allowed to write state without a valid, current token.

**Startup recommendation:** queue + container workers, full stop. Don't build a hot pool until
you have p95 latency data proving you need one — most early-stage agent products are not
latency-critical enough to justify the operational complexity of pool sizing on day one. Add hot
pools as a targeted optimization once you can point at a specific latency-sensitive path that
the queue-worker model can't satisfy.

## 9 · Interview questions

1. **What does "creating an agent" actually mean on a production platform?** — It means
   authoring a versioned artifact bundling prompt template, model/provider/params, a tool
   allow-list, policy bindings, budget defaults, and evaluation gates — not just writing a system
   prompt. That artifact goes through validation, staging evaluation, canary, and production
   promotion like any other deployable config.
2. **How do you roll back a bad agent deployment, and how is that different from rolling back an
   agent's action?** — Deployment rollback is a pointer swap (`agent_name -> agent_version_id`)
   plus cache/session invalidation — instant and cheap. Action rollback means compensating for a
   specific side effect a specific execution already committed (e.g., refunding a charge) — see
   [10 — Recoverability, Rollbacks & Saga](10-recoverability-rollbacks-and-saga.md). The two are
   unrelated: rolling back the deployment does not undo anything already done in the real world.
3. **Push vs. pull — how do you decide?** — Push minimizes routing delay but can overload
   runtimes during bursts since there's no buffer; pull (via a queue) provides backpressure and
   fairness at the cost of queue latency. Most platforms use pull as the reliable default and add
   push/hot-pool paths only where latency data justifies it.
4. **What is a fencing token and why does a lease alone not solve the duplicate-writer
   problem?** — A lease alone can expire while a paused/slow runtime is still unaware it lost
   ownership; when it later tries to write, without a fencing token the state store has no way to
   know that write is stale. A monotonically increasing fencing token, checked at the state store
   on every write, lets the store reject writes from a runtime that no longer holds the current
   lease.
5. **Why must heartbeats be independent of model call latency?** — Because a single slow (but
   healthy) model response would otherwise be indistinguishable from a dead runtime, causing
   unnecessary lease expiry and reassignment — and potentially a duplicate execution — for a
   runtime that was never actually unhealthy.

## Quick Revision Notes

- An agent definition is a **versioned artifact**: prompt template + model/provider/params + tool
  allow-list + policy bindings + budget defaults + evaluation gates.
- Promotion pipeline: Author → Validate → Stage (eval gates) → Canary (shadow/low-%) →
  Production, with Rollback as a pointer swap back to a prior `agent_version_id`.
- Rollback of a deployment (pointer swap + cache/session invalidation) is **not** the same as
  rollback of an action (compensation for a committed side effect) — see
  [10 — Recoverability, Rollbacks & Saga](10-recoverability-rollbacks-and-saga.md).
- Push (webhook → gateway → runtime) minimizes routing delay but can overload under burst; pull
  (event → queue → worker) gives backpressure/fairness at the cost of queue latency.
- Fencing tokens: state store accepts writes only from the current lease holder's token;
  heartbeats must be independent of model latency; cancellation must be durable, not in-memory.
- Runtime models: ephemeral (cold start), long-lived worker (idle cost/noisy neighbor), hot pool
  (sizing complexity), actor runtime (operational complexity), queue worker (queue latency).
- Startup default: queue + container workers. Add hot pools only once p95 latency data demands it.
- Fencing tokens are enforced at the resource being written, not the lease manager — a
  monotonically increasing token rejects stale writes from a runtime that lost its lease, even if
  that runtime doesn't yet know it (same mechanism as Chubby/ZooKeeper lease fencing).
- Push makes the control plane own capacity-tracking and backpressure explicitly; pull gets
  backpressure for free from queue depth but adds poll latency and needs visibility-timeout/lease
  redelivery to handle a worker crashing mid-claim.
- Canary = partial live traffic, compare metrics before ramping; blue-green = both versions fully
  live, instant router-flip cutover at ~2x capacity cost; shadow = mirrored traffic with discarded
  output — zero user-facing risk, zero live feedback loop on real side effects.
- Exhaustively testing an agent-version rollout is fundamentally harder than testing a workflow
  rollout — which is why canary/shadow evaluation on real trajectories matters more for agent
  deployments than for typical stateless-service deploys.

## Further Reading

- LangGraph persistence (checkpointers vs. stores) — <https://docs.langchain.com/oss/python/langgraph/persistence>
- AutoGen Core user guide — <https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/index.html>
- Kleppmann, "How to do distributed locking" (fencing tokens, the GC-pause example) — <https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html>
- Burrows, "The Chubby lock service for loosely-coupled distributed systems" (OSDI 2006) — <https://www.usenix.org/legacy/events/osdi06/tech/burrows.html>
- Hunt et al., "ZooKeeper: Wait-free coordination for Internet-scale systems" (USENIX ATC 2010) — <https://www.usenix.org/conference/usenixatc10/zookeeper-wait-free-coordination-internet-scale-systems>
- Back to [01 — Foundations of Agentic Systems](01-foundations-of-agentic-systems.md)
- Back to the [Agentic AI Platform — System Design](system-design.md) uber doc
- Next: [03 — Tool, MCP & Skill Registry](03-tool-mcp-and-skill-registry.md)
